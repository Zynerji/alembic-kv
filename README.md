# AlembicKV

**Fixed-memory KV cache for transformer inference.**

Drop-in replacement for HuggingFace's `DynamicCache` that caps memory at a fixed budget using a concept codebook. Below budget: identical to standard cache. Above budget: compresses old context into concepts, keeps recent tokens exact. O(1) memory regardless of sequence length.

**Not eviction.** Eviction discards old tokens forever. AlembicKV compresses them into a concept codebook — information is retained, not lost.

## Verified VRAM Results (Qwen2.5-7B, Blackwell RTX PRO 6000)

Measured GPU memory during actual model inference at each sequence length.

| Tokens | Standard Cache | AlembicKV (2048+512) | Savings | Compression |
|--------|---------------|----------------------|---------|-------------|
| 1,000 | 64 MB | 55 MB | 14% | 1.2x |
| 2,000 | 113 MB | 113 MB | 0% | 1.0x |
| 5,000 | 279 MB | 317 MB | -13% | 0.9x |
| 10,000 | 552 MB | 362 MB | **34%** | **1.5x** |
| 20,000 | 1,096 MB | 360 MB | **67%** | **3.0x** |
| 50,000 | 2,735 MB | 313 MB | **89%** | **8.7x** |
| 500,000 | 27,344 MB | 269 MB | **99.0%** | **101.7x** |

**AlembicKV is flat at ~270-360 MB from 10K to 500K tokens.** Standard cache grows to 27 GB at 500K. Peak VRAM never exceeds 975 MB. Crossover point: ~3K tokens.

At 1M tokens (extrapolated): ~55 GB standard vs ~270 MB AlembicKV = **~200x compression**.

## Verified Perplexity (Qwen2.5-7B, WikiText-2)

| Budget (codebook+window) | Perplexity | vs Standard |
|---------------------------|-----------|-------------|
| Standard (unbounded) | 9.787 | baseline |
| 512 (lossless) | 9.787 | **+0.0%** |
| 256 (lossless) | 9.668 | **-1.2%** |
| 128+128 (hybrid active) | 10.17 | **+3.9%** |

At 4x compression (128 codebook + 128 window for 1K token sequences): only 3.9% perplexity increase.

## How It Works

```
Tokens 1 to N:     [exact storage — identical to DynamicCache]
                            ↓ budget reached
Token N+1:         [compress oldest into codebook]
                            ↓
Tokens N+2..∞:     [codebook (compressed old)] + [recent window (exact)]
                    ────────────────────────────  ───────────────────────
                    Fixed O(1) concepts            Sliding window, exact
                    Never evicted, absorbed         Last W tokens raw
```

1. **Fill phase:** Standard concat. Zero quality loss. Identical to DynamicCache.
2. **Compression:** When buffer hits `budget + window_size`, oldest tokens are stored as the initial codebook. Recent tokens become the sliding window.
3. **Absorption:** New tokens push the oldest window token into the codebook via soft-attention write. The codebook absorbs it — additive, never evicts.
4. **QARC resonance:** Periodic antiresonant dynamics prevent codebook slots from collapsing into the same concept.

## Quick Start

```python
from alembic_kv.hf_cache import AlembicHFCache

# Create cache — budget=2048 concepts + window=512 exact tokens
cache = AlembicHFCache(budget=2048, window_size=512)

# Drop-in replacement for DynamicCache
outputs = model.generate(
    input_ids,
    past_key_values=cache,
    use_cache=True,
    max_new_tokens=100000,  # generate as long as you want
)

# Memory stays at ~360 MB regardless of how many tokens generated
```

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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `budget` | 2048 | Concept codebook slots (compressed old context) |
| `window_size` | 128 | Exact recent tokens to keep (sliding window) |
| `write_alpha` | 0.1 | Codebook absorption rate |
| `write_temp` | 0.1 | Write attention temperature (lower = sharper) |
| `qarc_gamma` | 0.01 | QARC anti-collapse strength |
| `resonate_every` | 64 | QARC resonance frequency (every N absorptions) |

## Why Not Just Eviction?

Eviction (StreamingLLM, H2O, SpectralKV) **discards** old tokens. Once evicted, that information is gone. If a later query needs context from an evicted token, the model has no access to it.

AlembicKV **absorbs** old tokens into a concept codebook. The information is compressed, not lost. The codebook retains the semantic structure of the old context, allowing the model to attend to concepts from arbitrarily far back.

At budget=128, AlembicKV's hybrid approach achieves +3.9% perplexity vs eviction's +427% — because the codebook retains what eviction throws away.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache 2.0 with Commons Clause. Free for non-commercial use. Commercial licensing: CKnopp@gmail.com

## Author

Christian Knopp
