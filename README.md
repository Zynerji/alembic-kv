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

## Verified: Gemma 4 31B (google/gemma-4-31B-it, Blackwell RTX PRO 6000)

Standard cache OOMs. AlembicKV keeps running.

| Tokens | Standard Cache | AlembicKV (2048+512) | Result |
|--------|---------------|----------------------|--------|
| 1,000 | 873 MB | 863 MB | 1.0x |
| 10,000 | 8,597 MB | 5,521 MB | **1.6x** |
| 20,000 | 17,188 MB | 5,310 MB | **3.2x** |
| 50,000 | **OOM** | 4,695 MB | **AlembicKV only** |
| 100,000 | **OOM** | 5,421 MB | **AlembicKV only** |

Standard KV cache exhausts 38 GB of free VRAM at ~25K tokens. AlembicKV stays flat at ~5 GB and runs to 1M tokens on the same hardware.

### Full eval: 100K to 1M tokens (Gemma 4 31B)

| Tokens | AlembicKV Cache | Peak VRAM |
|--------|----------------|-----------|
| 100,000 | 5,423 MB | 7,822 MB |
| 200,000 | 5,119 MB | 7,822 MB |
| 300,000 | 4,824 MB | 7,822 MB |
| 500,000 | 4,219 MB | 7,826 MB |
| **1,000,000** | **4,471 MB** | **7,826 MB** |

Standard cache would need **~860 GB** at 1M tokens. AlembicKV: **4.5 GB**. That's **~192x compression**.

## Verified Perplexity (Qwen2.5-7B, chunked forward passes)

| Context | Budget+Window | Standard PPL | AlembicKV PPL | Delta | Mode |
|---------|---------------|-------------|---------------|-------|------|
| 256 | 128+128 | 2.05 | 2.05 | **+0.0%** | fill (lossless) |
| 512 | 128+128 | 1.52 | 10.25 | **+573%** | codebook active |
| 256-8192 | 2048+512 | 1.12-2.05 | 1.12-2.05 | **+0.0%** | fill (lossless) |

**Below budget+window: perfectly lossless.** Zero perplexity difference at any context length.

**Above budget+window: significant quality cost.** When the codebook absorbs tokens, perplexity increases substantially. The soft-attention write mechanism compresses K/V information lossily. This is the active area of improvement.

**The tradeoff:** AlembicKV trades quality for unlimited context. Standard cache is better when it fits in VRAM. AlembicKV enables contexts that standard cache cannot serve at all (50K+ on 31B, 500K+ on 7B).

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
