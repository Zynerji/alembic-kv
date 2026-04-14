#!/usr/bin/env python3
"""Integration test: AlembicKV as true drop-in KV cache replacement.

Loads a HuggingFace model, generates text with standard KV cache,
then generates with AlembicHFCache passed as past_key_values.

Usage:
    python benchmarks/test_real_model.py --model Qwen/Qwen2.5-7B
    python benchmarks/test_real_model.py --model Qwen/Qwen2.5-72B-Instruct
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--max_new', type=int, default=100)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    print(f"{'='*60}")
    print(f"  AlembicKV Drop-In Integration Test")
    print(f"  Model: {args.model}")
    print(f"  Budget: {args.budget} concept slots")
    print(f"{'='*60}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map='auto',
        trust_remote_code=True)
    model.eval()

    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config
    print(f"  Layers: {config.num_hidden_layers}, "
          f"KV heads: {getattr(config, 'num_key_value_heads', config.num_attention_heads)}, "
          f"head_dim: {config.hidden_size // config.num_attention_heads}")

    prompts = [
        "The capital of France is",
        "In a surprising turn of events, the scientist discovered that",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    ]

    # ── Standard KV cache ────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Standard KV Cache")
    print(f"{'─'*60}")

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new,
                                 do_sample=True, temperature=0.7, top_p=0.9)
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:300]}")
        print(f"  Time: {elapsed:.1f}s ({args.max_new/elapsed:.0f} tok/s)")

    # ── AlembicKV cache ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  AlembicKV Cache (budget={args.budget})")
    print(f"{'─'*60}")

    from alembic_kv.hf_cache import AlembicHFCache

    for prompt in prompts:
        cache = AlembicHFCache(budget=args.budget)

        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new,
                                 do_sample=True, temperature=0.7, top_p=0.9,
                                 past_key_values=cache)
        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {text[:300]}")
        print(f"  Time: {elapsed:.1f}s ({args.max_new/elapsed:.0f} tok/s)")
        s = cache.stats()
        print(f"  Cache: {s}")

    print(f"\n{'='*60}")
    print(f"  Test complete")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
