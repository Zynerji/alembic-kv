#!/usr/bin/env python3
"""A/B VRAM benchmark: Standard DynamicCache vs AlembicKV.

Measures actual GPU memory at increasing sequence lengths.
Standard cache grows linearly. AlembicKV stays flat after budget.

Usage:
    python benchmarks/bench_vram.py --model Qwen/Qwen2.5-7B --max_tokens 100000
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_gpu_memory_mb():
    """Return current GPU memory usage in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def get_gpu_memory_reserved_mb():
    """Return reserved (allocated + cached) GPU memory in MB."""
    torch.cuda.synchronize()
    return torch.cuda.memory_reserved() / (1024 ** 2)


def measure_cache_vram(model, tokenizer, seq_lengths, cache_class=None,
                       cache_kwargs=None, label="Standard"):
    """Measure VRAM at each sequence length by running actual model inference.

    Feeds tokens in chunks and measures memory after each chunk.
    """
    results = []
    device = next(model.parameters()).device

    # Generate a long token sequence (repeating text)
    seed_text = "The quick brown fox jumps over the lazy dog. " * 100
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    # Repeat to get enough tokens
    max_needed = max(seq_lengths) + 1000
    token_ids = (seed_ids * (max_needed // len(seed_ids) + 1))[:max_needed]
    token_ids = torch.tensor([token_ids], device=device)

    for target_len in seq_lengths:
        # Clear GPU cache
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        baseline_mem = get_gpu_memory_mb()

        # Create cache
        if cache_class is not None:
            cache = cache_class(**(cache_kwargs or {}))
        else:
            from transformers.cache_utils import DynamicCache
            cache = DynamicCache()

        try:
            # Feed tokens in chunks to avoid OOM on single forward
            chunk_size = min(2048, target_len)
            total_fed = 0

            with torch.no_grad():
                while total_fed < target_len:
                    end = min(total_fed + chunk_size, target_len)
                    chunk = token_ids[:, total_fed:end]

                    # For chunks after the first, we need position_ids
                    position_ids = torch.arange(
                        total_fed, end, device=device
                    ).unsqueeze(0)

                    outputs = model(
                        chunk,
                        past_key_values=cache,
                        use_cache=True,
                        position_ids=position_ids,
                    )
                    cache = outputs.past_key_values
                    total_fed = end

                    # Clear output to free activation memory
                    del outputs

            cache_mem = get_gpu_memory_mb() - baseline_mem
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) - baseline_mem

            # Get cache-specific stats
            cache_stats = {}
            if hasattr(cache, 'stats'):
                cache_stats = cache.stats()

            results.append({
                'seq_len': target_len,
                'cache_mem_mb': cache_mem,
                'peak_mem_mb': peak_mem,
                'status': 'ok',
                **cache_stats,
            })

            print(f"  {label} @ {target_len:>8,d} tokens: "
                  f"cache={cache_mem:>8.1f} MB, peak={peak_mem:>8.1f} MB"
                  f"  {cache_stats.get('mode', '')}")

        except torch.cuda.OutOfMemoryError:
            results.append({
                'seq_len': target_len,
                'cache_mem_mb': float('inf'),
                'peak_mem_mb': float('inf'),
                'status': 'OOM',
            })
            print(f"  {label} @ {target_len:>8,d} tokens: OOM!")
            gc.collect()
            torch.cuda.empty_cache()
            break  # No point testing larger sizes

        finally:
            del cache
            gc.collect()
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen2.5-7B')
    parser.add_argument('--max_tokens', type=int, default=100000)
    parser.add_argument('--budget', type=int, default=2048)
    parser.add_argument('--window', type=int, default=512)
    parser.add_argument('--dtype', default='bfloat16')
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)

    # Sequence lengths to test
    seq_lengths = [1000, 2000, 5000, 10000, 20000, 50000]
    seq_lengths = [s for s in seq_lengths if s <= args.max_tokens]
    if args.max_tokens not in seq_lengths:
        seq_lengths.append(args.max_tokens)

    print(f"{'='*70}")
    print(f"  AlembicKV vs Standard KV Cache — VRAM Benchmark")
    print(f"  Model: {args.model}")
    print(f"  AlembicKV: budget={args.budget}, window={args.window}")
    print(f"  Sequence lengths: {seq_lengths}")
    print(f"{'='*70}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(
        torch_dtype=dtype, device_map='auto', trust_remote_code=True,
    )
    # Auto-detect if model needs quantization to fit
    if '72B' in args.model or '70B' in args.model or '65B' in args.model:
        from transformers import BitsAndBytesConfig
        load_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type='nf4',
        )
        print("  Using 4-bit quantization for large model")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    model_mem = get_gpu_memory_mb()
    print(f"  Model VRAM: {model_mem:.0f} MB")
    print(f"  Free VRAM: {(torch.cuda.get_device_properties(0).total_memory / 1024**2 - model_mem):.0f} MB")

    # ── Standard KV Cache ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  Standard DynamicCache (unbounded)")
    print(f"{'─'*70}")
    std_results = measure_cache_vram(
        model, tokenizer, seq_lengths, label="Standard")

    # ── AlembicKV Cache ───────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  AlembicKV (budget={args.budget}, window={args.window})")
    print(f"{'─'*70}")

    from alembic_kv.hf_cache import AlembicHFCache
    alb_results = measure_cache_vram(
        model, tokenizer, seq_lengths,
        cache_class=AlembicHFCache,
        cache_kwargs={'budget': args.budget, 'window_size': args.window},
        label="AlembicKV")

    # ── Summary ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VRAM COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Tokens':>10s} {'Standard MB':>12s} {'AlembicKV MB':>13s} {'Savings':>10s}")
    print(f"  {'─'*50}")

    for std, alb in zip(std_results, alb_results):
        s_len = std['seq_len']
        s_mem = std['cache_mem_mb']
        a_mem = alb['cache_mem_mb']

        if std['status'] == 'OOM':
            print(f"  {s_len:>10,d} {'OOM':>12s} {a_mem:>12.1f}  AlembicKV only")
        elif alb['status'] == 'OOM':
            print(f"  {s_len:>10,d} {s_mem:>12.1f} {'OOM':>13s}")
        else:
            if s_mem > 0 and a_mem > 0:
                ratio = s_mem / a_mem
                saving = (1 - a_mem / s_mem) * 100
                print(f"  {s_len:>10,d} {s_mem:>12.1f} {a_mem:>12.1f}  {saving:>+8.1f}% ({ratio:.1f}x)")
            else:
                print(f"  {s_len:>10,d} {s_mem:>12.1f} {a_mem:>12.1f}")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
