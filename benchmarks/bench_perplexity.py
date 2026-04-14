#!/usr/bin/env python3
"""Perplexity benchmark: AlembicKV vs standard KV cache at various budgets.

Measures actual perplexity on WikiText-2 to validate quality claims.

Usage:
    python benchmarks/bench_perplexity.py --model Qwen/Qwen2.5-7B
    python benchmarks/bench_perplexity.py --model Qwen/Qwen2.5-7B --budgets 32,64,128,256,512,1024,2048
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_perplexity(model, tokenizer, texts, max_len=2048, cache_class=None,
                       cache_kwargs=None, device='cuda'):
    """Compute perplexity over a list of texts.

    If cache_class is None, uses standard KV cache.
    If cache_class is provided, creates a fresh cache per text.
    """
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True,
                           max_length=max_len).to(device)
        input_ids = inputs['input_ids']
        T = input_ids.shape[1]
        if T < 2:
            continue

        if cache_class is not None:
            cache = cache_class(**(cache_kwargs or {}))
        else:
            cache = None

        with torch.no_grad():
            if cache is not None:
                outputs = model(input_ids, past_key_values=cache, use_cache=True)
            else:
                outputs = model(input_ids)

        logits = outputs.logits[:, :-1, :]  # (B, T-1, V)
        targets = input_ids[:, 1:]  # (B, T-1)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               targets.reshape(-1), reduction='sum')
        total_loss += loss.item()
        total_tokens += targets.numel()

    return torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B')
    parser.add_argument('--budgets', default='32,64,128,256,512,1024,2048',
                        help='Comma-separated budget sizes to test')
    parser.add_argument('--n_samples', type=int, default=50,
                        help='Number of text samples for perplexity')
    parser.add_argument('--max_len', type=int, default=2048)
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    budgets = [int(b) for b in args.budgets.split(',')]
    dtype = getattr(torch, args.dtype)

    print(f"{'='*60}")
    print(f"  AlembicKV Perplexity Benchmark")
    print(f"  Model: {args.model}")
    print(f"  Budgets: {budgets}")
    print(f"  Samples: {args.n_samples}, Max len: {args.max_len}")
    print(f"{'='*60}")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map='auto',
        trust_remote_code=True)
    model.eval()

    # Load test data
    print("Loading WikiText-2...")
    from datasets import load_dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    texts = [t for t in dataset['text'] if len(t) > 100][:args.n_samples]
    print(f"  {len(texts)} text samples loaded")

    # Baseline: standard KV cache
    print(f"\n{'─'*60}")
    print(f"  Standard KV Cache (baseline)")
    print(f"{'─'*60}")
    t0 = time.perf_counter()
    ppl_standard = compute_perplexity(model, tokenizer, texts, args.max_len)
    elapsed = time.perf_counter() - t0
    print(f"  Perplexity: {ppl_standard:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # AlembicKV at various budgets
    from alembic_kv.hf_cache import AlembicHFCache

    results = [('Standard', float('inf'), ppl_standard)]

    for budget in budgets:
        print(f"\n{'─'*60}")
        print(f"  AlembicKV (budget={budget})")
        print(f"{'─'*60}")
        t0 = time.perf_counter()
        ppl = compute_perplexity(
            model, tokenizer, texts, args.max_len,
            cache_class=AlembicHFCache,
            cache_kwargs={'budget': budget, 'n_sink': 4, 'recent_window': min(64, budget // 4)},
        )
        elapsed = time.perf_counter() - t0
        ppl_increase = (ppl / ppl_standard - 1) * 100
        results.append((f'Budget={budget}', budget, ppl))
        print(f"  Perplexity: {ppl:.4f} ({ppl_increase:+.1f}% vs standard)")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Method':<20s} {'Budget':>8s} {'Perplexity':>12s} {'vs Standard':>12s}")
    print(f"  {'─'*52}")
    for name, budget, ppl in results:
        if budget == float('inf'):
            print(f"  {name:<20s} {'∞':>8s} {ppl:>12.4f} {'baseline':>12s}")
        else:
            delta = (ppl / ppl_standard - 1) * 100
            print(f"  {name:<20s} {budget:>8d} {ppl:>12.4f} {delta:>+11.1f}%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
