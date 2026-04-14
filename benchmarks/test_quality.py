#!/usr/bin/env python3
"""Quality test: does AlembicKV actually produce good output at long context?

Tests perplexity and generation quality at increasing context lengths,
especially beyond the codebook budget where absorption is active.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B')
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--window', type=int, default=512)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from alembic_kv.hf_cache import AlembicHFCache

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map='auto',
        trust_remote_code=True)
    model.eval()

    # Build long text
    seed = ('The quick brown fox jumps over the lazy dog. '
            'Scientists discovered new properties of quantum entanglement. '
            'The stock market showed unexpected resilience despite global uncertainty. '
            'Machine learning models continue to improve at an accelerating pace. ')
    seed_ids = tokenizer.encode(seed, add_special_tokens=False)
    long_ids = (seed_ids * 500)[:10000]
    input_ids = torch.tensor([long_ids], device='cuda')

    print(f"\n{'='*60}")
    print(f"  PERPLEXITY AT DIFFERENT CONTEXT LENGTHS")
    print(f"  Budget={args.budget}, Window={args.window}")
    print(f"{'='*60}")
    print(f"  {'Context':>10s} {'Standard':>12s} {'AlembicKV':>12s} {'Delta':>10s} {'Mode':>15s}")
    print(f"  {'-'*60}")

    for ctx_len in [256, 512, 1024, 2048, 3000, 4096, 6000, 8192]:
        if ctx_len > len(long_ids):
            break

        chunk = input_ids[:, :ctx_len]
        targets = chunk[:, 1:]

        # Standard
        with torch.no_grad():
            out_std = model(chunk)
        logits_std = out_std.logits[:, :-1, :]
        loss_std = F.cross_entropy(
            logits_std.reshape(-1, logits_std.shape[-1]),
            targets.reshape(-1))
        ppl_std = torch.exp(loss_std).item()

        # AlembicKV — feed in chunks to trigger codebook absorption
        cache = AlembicHFCache(budget=args.budget, window_size=args.window)
        chunk_size = min(256, ctx_len)  # small chunks to force absorption
        all_logits = []
        with torch.no_grad():
            for start in range(0, ctx_len, chunk_size):
                end = min(start + chunk_size, ctx_len)
                c = chunk[:, start:end]
                pos_ids = torch.arange(start, end, device=c.device).unsqueeze(0)
                out_alb = model(c, past_key_values=cache, use_cache=True,
                                position_ids=pos_ids)
                cache = out_alb.past_key_values
                all_logits.append(out_alb.logits)
        logits_alb = torch.cat(all_logits, dim=1)[:, :-1, :]
        loss_alb = F.cross_entropy(
            logits_alb.reshape(-1, logits_alb.shape[-1]),
            targets.reshape(-1))
        ppl_alb = torch.exp(loss_alb).item()

        delta = (ppl_alb / ppl_std - 1) * 100
        mode = cache.stats()['mode']
        print(f"  {ctx_len:>10d} {ppl_std:>12.2f} {ppl_alb:>12.2f} {delta:>+9.1f}% {mode:>15s}")

    # Generation quality test
    print(f"\n{'='*60}")
    print(f"  GENERATION QUALITY BEYOND BUDGET")
    print(f"{'='*60}")

    for prefill_len in [1000, 3000, 5000, 8000]:
        if prefill_len > len(long_ids):
            break

        prefill = input_ids[:, :prefill_len]

        # Standard
        with torch.no_grad():
            out = model.generate(prefill, max_new_tokens=30,
                                 do_sample=False, use_cache=True)
        std_text = tokenizer.decode(out[0, prefill_len:], skip_special_tokens=True)

        # AlembicKV
        cache = AlembicHFCache(budget=args.budget, window_size=args.window)
        with torch.no_grad():
            out = model.generate(prefill, max_new_tokens=30,
                                 do_sample=False, past_key_values=cache,
                                 use_cache=True)
        alb_text = tokenizer.decode(out[0, prefill_len:], skip_special_tokens=True)

        mode = cache.stats()['mode']
        match = "MATCH" if std_text.strip() == alb_text.strip() else "DIFFER"
        print(f"\n  Prefill: {prefill_len} tokens ({mode})")
        print(f"  Standard:  {std_text[:150]}")
        print(f"  AlembicKV: {alb_text[:150]}")
        print(f"  [{match}]")


if __name__ == '__main__':
    main()
