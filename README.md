# AlembicKV

**Fixed-budget KV cache for transformer inference.**

Drop-in replacement for HuggingFace's `DynamicCache`. Caps memory at a fixed token budget with importance-weighted eviction. Below budget: identical to standard cache. Above budget: evicts least-important tokens, keeps sink tokens and recent window.

## Verified Results (Qwen2.5-7B, WikiText-2, 1024-token sequences)

| Budget | Perplexity | vs Standard | Memory |
|--------|-----------|-------------|--------|
| Standard (unbounded) | 9.787 | baseline | O(n) |
| 512 tokens | 9.787 | **+0.0%** | Fixed |
| 256 tokens | 9.868 | **+0.8%** | Fixed |
| 128 tokens | 51.53 | +427% | Fixed |

**Below budget = lossless.** No quality degradation until eviction kicks in.
At 256 tokens (4x compression of 1K context): only 0.8% perplexity increase.

## Quick Start

```python
from alembic_kv.hf_cache import AlembicHFCache

# Create fixed-budget cache
cache = AlembicHFCache(budget=2048)

# Use with any HuggingFace model — identical API to DynamicCache
outputs = model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=1000,
)

# Memory is capped at budget tokens regardless of generation length
print(cache.stats())
```

## How It Works

1. **Below budget:** Standard concat — identical to `DynamicCache`, zero quality loss
2. **At budget:** Evict least-important tokens from the middle of the sequence
3. **Always keep:** Sink tokens (first 4) + recent window (last 64) — these are critical for attention pattern stability

Eviction uses cumulative importance scoring. Tokens that receive more attention weight during generation are retained; rarely-attended tokens are evicted first.

## Installation

```bash
pip install alembic-kv
```

Or from source:
```bash
git clone https://github.com/zynerji/alembic-kv.git
cd alembic-kv
pip install -e ".[dev]"
```

## Memory

For a model with `num_layers` layers, `num_kv_heads` KV heads, `head_dim` dim per head:

```
Standard cache: num_layers * seq_len * num_kv_heads * head_dim * 2 (K+V) * 2 bytes
AlembicKV:      num_layers * budget  * num_kv_heads * head_dim * 2 (K+V) * 2 bytes
```

Memory is capped at `budget` tokens. Example for Qwen2.5-7B (28 layers, 4 KV heads, 128 dim):

| Sequence Length | Standard Cache | AlembicKV (budget=2048) |
|-----------------|---------------|-------------------------|
| 1K tokens | 28 MB | 28 MB |
| 10K tokens | 280 MB | 28 MB |
| 100K tokens | 2.8 GB | 28 MB |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` | 2048 | Maximum tokens to retain |
| `n_sink` | 4 | Always keep first N tokens (attention sinks) |
| `recent_window` | 64 | Always keep last N tokens |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0 with Commons Clause. Free for non-commercial use. Commercial licensing: CKnopp@gmail.com

## Author

Christian Knopp
