#!/usr/bin/env python3
"""AlembicKV retrieval quality benchmark — runs on CPU with multiprocessing.

Tests whether the concept codebook retains retrievable information after
absorbing many tokens. Measures:

1. Needle-in-haystack: absorb N random tokens, then absorb 1 distinctive
   "needle" token, then query with the needle — does the retrieved K/V
   correlate with the needle more than with random?

2. Absorption capacity: absorb K tokens, then absorb them again —
   does the codebook produce lower MSE on seen data vs unseen?

3. QARC anti-collapse: after 100K absorptions, are codebook slots
   still diverse (high pairwise cosine distance)?

4. Throughput: tokens/sec absorption rate on CPU, multiprocessed.

Usage:
    python benchmarks/bench_retrieval.py --workers 32 --tokens 100000

    # Quick test
    python benchmarks/bench_retrieval.py --workers 4 --tokens 1000
"""

import argparse
import math
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from alembic_kv import ConceptChamber, AlembicKVConfig


def needle_in_haystack(args):
    """Test: can the codebook retrieve a needle after absorbing haystack?"""
    codebook_size, kv_dim, haystack_size, trial = args

    chamber = ConceptChamber(
        kv_dim=kv_dim, codebook_size=codebook_size,
        num_kv_heads=8, head_dim=kv_dim // 8,
        write_alpha=0.01,
    )

    # Absorb haystack (random noise)
    batch = min(haystack_size, 256)
    for i in range(0, haystack_size, batch):
        n = min(batch, haystack_size - i)
        k = torch.randn(1, n, kv_dim) * 0.5
        v = torch.randn(1, n, kv_dim) * 0.5
        chamber.absorb(k, v)

    # Absorb needle (distinctive signal)
    needle_k = torch.ones(1, 1, kv_dim) * 3.0  # high magnitude, uniform direction
    needle_v = torch.ones(1, 1, kv_dim) * 3.0
    chamber.absorb(needle_k, needle_v)

    # Resonate to stabilize
    chamber.resonate()

    # Query with needle
    retrieved_k, retrieved_v = chamber.retrieve(needle_k)  # (1, codebook_size, H, D)

    # Check: does the codebook have higher correlation with needle than random?
    codebook_k_flat = chamber.codebook_k  # (codebook_size, kv_dim)
    needle_flat = needle_k.squeeze()       # (kv_dim,)

    # Cosine similarity of each codebook slot with needle
    cos_sims = F.cosine_similarity(codebook_k_flat, needle_flat.unsqueeze(0), dim=-1)
    max_cos = cos_sims.max().item()
    mean_cos = cos_sims.mean().item()

    # Compare with random query baseline
    random_q = torch.randn(1, kv_dim)
    random_cos = F.cosine_similarity(codebook_k_flat, random_q, dim=-1)
    random_max = random_cos.max().item()

    return {
        'trial': trial,
        'haystack_size': haystack_size,
        'needle_max_cos': max_cos,
        'needle_mean_cos': mean_cos,
        'random_max_cos': random_max,
        'needle_advantage': max_cos - random_max,
    }


def absorption_capacity(args):
    """Test: does the codebook prefer seen data over unseen?"""
    codebook_size, kv_dim, num_tokens, trial = args

    chamber = ConceptChamber(
        kv_dim=kv_dim, codebook_size=codebook_size,
        num_kv_heads=8, head_dim=kv_dim // 8,
        write_alpha=0.01,
    )

    # Generate "training" data — fixed patterns
    torch.manual_seed(trial * 1000)
    train_k = torch.randn(1, num_tokens, kv_dim)
    train_v = torch.randn(1, num_tokens, kv_dim)

    # Absorb training data multiple times
    for epoch in range(5):
        batch = min(num_tokens, 256)
        for i in range(0, num_tokens, batch):
            n = min(batch, num_tokens - i)
            chamber.absorb(train_k[:, i:i+n], train_v[:, i:i+n])
        chamber.resonate()

    # Unseen data
    unseen_k = torch.randn(1, num_tokens, kv_dim)

    # MSE: codebook vs seen data
    seen_recon = chamber.codebook_k.unsqueeze(0)  # (1, cs, kv)
    # For each training token, find most similar codebook entry
    sims = torch.matmul(train_k.squeeze(0), chamber.codebook_k.T)  # (N, cs)
    best_idx = sims.argmax(dim=-1)  # (N,)
    seen_closest = chamber.codebook_k[best_idx]  # (N, kv)
    seen_mse = F.mse_loss(seen_closest, train_k.squeeze(0)).item()

    # MSE: codebook vs unseen
    sims_u = torch.matmul(unseen_k.squeeze(0), chamber.codebook_k.T)
    best_idx_u = sims_u.argmax(dim=-1)
    unseen_closest = chamber.codebook_k[best_idx_u]
    unseen_mse = F.mse_loss(unseen_closest, unseen_k.squeeze(0)).item()

    return {
        'trial': trial,
        'num_tokens': num_tokens,
        'seen_mse': seen_mse,
        'unseen_mse': unseen_mse,
        'preference_ratio': unseen_mse / max(seen_mse, 1e-9),
    }


def qarc_diversity(args):
    """Test: do codebook slots stay diverse after many absorptions?"""
    codebook_size, kv_dim, num_tokens, trial = args

    chamber = ConceptChamber(
        kv_dim=kv_dim, codebook_size=codebook_size,
        num_kv_heads=8, head_dim=kv_dim // 8,
        qarc_gamma=0.03, qarc_iterations=3,
        write_alpha=0.01,
    )

    # Also test without QARC for comparison
    chamber_no_qarc = ConceptChamber(
        kv_dim=kv_dim, codebook_size=codebook_size,
        num_kv_heads=8, head_dim=kv_dim // 8,
        qarc_gamma=0.0, qarc_iterations=0,  # disabled
        write_alpha=0.01,
    )

    batch = min(num_tokens, 256)
    for i in range(0, num_tokens, batch):
        n = min(batch, num_tokens - i)
        k = torch.randn(1, n, kv_dim)
        v = torch.randn(1, n, kv_dim)
        chamber.absorb(k, v)
        chamber_no_qarc.absorb(k, v)
        if i % 1024 == 0:
            chamber.resonate()

    # Measure diversity: average pairwise cosine distance
    def codebook_diversity(cb):
        cb_norm = F.normalize(cb, dim=-1)
        sim_matrix = torch.matmul(cb_norm, cb_norm.T)
        # Exclude diagonal
        mask = ~torch.eye(codebook_size, dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()
        return 1.0 - avg_sim  # diversity = 1 - similarity

    div_qarc = codebook_diversity(chamber.codebook_k)
    div_no_qarc = codebook_diversity(chamber_no_qarc.codebook_k)

    return {
        'trial': trial,
        'num_tokens': num_tokens,
        'diversity_with_qarc': div_qarc,
        'diversity_without_qarc': div_no_qarc,
        'qarc_advantage': div_qarc - div_no_qarc,
    }


def throughput_test(args):
    """Measure absorption throughput (tokens/sec on CPU)."""
    codebook_size, kv_dim, num_tokens, trial = args

    chamber = ConceptChamber(
        kv_dim=kv_dim, codebook_size=codebook_size,
        num_kv_heads=8, head_dim=kv_dim // 8,
        write_alpha=0.01,
    )

    batch = 128
    data_k = torch.randn(1, batch, kv_dim)
    data_v = torch.randn(1, batch, kv_dim)

    t0 = time.perf_counter()
    absorbed = 0
    while absorbed < num_tokens:
        chamber.absorb(data_k, data_v)
        absorbed += batch
    elapsed = time.perf_counter() - t0

    return {
        'trial': trial,
        'tokens': absorbed,
        'elapsed_s': elapsed,
        'tokens_per_sec': absorbed / elapsed,
    }


def run_benchmark(test_fn, args_list, name, workers):
    """Run a benchmark across workers and aggregate results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {len(args_list)} trials across {workers} workers")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    if workers > 1:
        with Pool(workers) as pool:
            results = pool.map(test_fn, args_list)
    else:
        results = [test_fn(a) for a in args_list]
    elapsed = time.perf_counter() - t0

    # Print results
    for r in results:
        parts = []
        for k, v in r.items():
            if k == 'trial':
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"  [{r['trial']:>3d}] {', '.join(parts)}")

    # Aggregate
    numeric_keys = [k for k in results[0] if k != 'trial' and isinstance(results[0][k], (int, float))]
    print(f"\n  Averages:")
    for k in numeric_keys:
        vals = [r[k] for r in results]
        print(f"    {k}: {sum(vals)/len(vals):.4f}")
    print(f"  Wall time: {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description='AlembicKV retrieval benchmark')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of worker processes (default: cpu_count // 2)')
    parser.add_argument('--tokens', type=int, default=10000,
                        help='Tokens per trial (default: 10000)')
    parser.add_argument('--trials', type=int, default=16,
                        help='Number of trials per benchmark (default: 16)')
    parser.add_argument('--codebook', type=int, default=2048,
                        help='Codebook size (default: 2048)')
    parser.add_argument('--kv_dim', type=int, default=1024,
                        help='K/V dimension (default: 1024 = 8 heads * 128 dim)')
    args = parser.parse_args()

    workers = args.workers or max(1, cpu_count() // 2)
    cs = args.codebook
    kv = args.kv_dim
    N = args.tokens
    T = args.trials

    print(f"AlembicKV Benchmark")
    print(f"  Codebook: {cs} slots × {kv}d = {cs * kv * 4 / 1024:.1f} KB")
    print(f"  Workers: {workers}, Trials: {T}, Tokens/trial: {N}")
    print(f"  Standard KV for {N} tokens: {N * kv * 2 * 2 / (1024**2):.1f} MB")

    # 1. Needle in haystack
    run_benchmark(
        needle_in_haystack,
        [(cs, kv, N, t) for t in range(T)],
        "Needle-in-Haystack", workers)

    # 2. Absorption capacity
    run_benchmark(
        absorption_capacity,
        [(cs, kv, min(N, 1000), t) for t in range(T)],
        "Absorption Capacity (seen vs unseen)", workers)

    # 3. QARC diversity
    run_benchmark(
        qarc_diversity,
        [(cs, kv, N, t) for t in range(T)],
        "QARC Anti-Collapse (diversity)", workers)

    # 4. Throughput
    run_benchmark(
        throughput_test,
        [(cs, kv, N, t) for t in range(T)],
        "Throughput (tokens/sec)", workers)


if __name__ == '__main__':
    main()
