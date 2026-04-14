# AlembicKV

**Fixed-memory KV cache replacement for transformer inference.**

Standard KV cache grows linearly with sequence length. At 100K tokens on a 70B model, that's 32 GB of VRAM just for the cache. At 1M tokens: 328 GB.

AlembicKV replaces the unbounded cache with a fixed-size concept codebook. **1.3 GB for a 70B model, regardless of sequence length.** 1K tokens, 1M tokens, 1B tokens — same memory.

## How it works

Each transformer layer gets a concept chamber: a fixed codebook of K/V vectors in native attention space.

**Write (absorb a token):** New K/V pairs are distributed across the codebook via soft-attention. The token's information blends into the most relevant concept slots. Nothing is evicted — information accumulates.

**Read (attention):** The codebook entries are returned directly as K/V for the attention mechanism. No projections, no training required.

**QARC (Quasicrystal Autopoietic Resonance Cascade):** Periodic resonance dynamics prevent the codebook from collapsing into a mean-field summary. Coupled pendulums at incommensurate metallic-mean frequencies (golden ratio, bronze ratio) create quasiperiodic dynamics that maintain slot diversity.

## Memory comparison

| Sequence length | Standard KV (70B) | AlembicKV (2048 slots) | Compression |
|-----------------|-------------------|------------------------|-------------|
| 1K tokens       | 32 MB             | 1.3 GB                 | 0.02x       |
| 10K tokens      | 320 MB            | 1.3 GB                 | 0.25x       |
| 100K tokens     | 32 GB             | 1.3 GB                 | **25x**     |
| 1M tokens       | 328 GB            | 1.3 GB                 | **252x**    |
| 10M tokens      | 3.2 TB            | 1.3 GB                 | **2,521x**  |

Crossover point: ~42K tokens. Below that, standard KV is cheaper. Above that, AlembicKV wins and the advantage compounds linearly.

## Quick start

```python
from alembic_kv import AlembicKVCache, AlembicKVConfig

# Configure for your model
config = AlembicKVConfig(
    num_layers=32,       # model layers
    num_heads=8,         # KV heads (use num_key_value_heads for GQA)
    head_dim=128,        # dimension per head
    budget=2048,         # concept slots (NOT token slots)
)

print(config.summary())
# AlembicKV Config:
#   Total VRAM: 1.3 GB (K+V codebooks, float32)
#   Compression vs standard KV cache:
#     At 100K tokens: 25x  (std: 32.00 GB)
#     At   1M tokens: 252x (std: 328.00 GB)

# Create cache
cache = AlembicKVCache(config)

# Use in generation loop (DynamicCache-compatible interface)
keys, values = cache.update(key_states, value_states, layer_idx=0)
# keys/values contain: [concept_codebook, raw_current_token]
# Pass to attention as usual
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
| `budget` | 2048 | Concept codebook slots. More = more capacity, more VRAM |
| `write_alpha` | 0.1 | Absorption rate. Higher = faster absorption |
| `write_temp` | 0.1 | Write attention temperature. Lower = sharper (fewer slots per token) |
| `qarc_gamma` | 0.03 | QARC resonance strength. Higher = more anti-collapse force |
| `qarc_iterations` | 3 | QARC steps per resonance cycle |
| `resonate_every` | 64 | Run QARC every N tokens |

## Architecture

```
Token K/V ──► Normalize ──► Soft-attention score against codebook
                                    │
                                    ▼
                           Additive write into most relevant slots
                                    │
                                    ▼
                           QARC resonance (periodic)
                           Prevents slot convergence via
                           quasiperiodic coupled pendulums
                                    │
                                    ▼
Attention query ──────────► Read codebook directly as K/V
                           No projections, no training
```

**No learned projections.** The codebook operates in the model's native K/V space. Any model with standard multi-head attention works out of the box. No fine-tuning, no adapter training.

**No eviction.** Information accumulates indefinitely. The codebook absorbs concepts — not individual tokens. 2048 concept slots can represent far more than 2048 tokens of context because natural language is massively redundant.

**QARC anti-collapse.** Without resonance, repeated soft-writes cause all codebook slots to converge toward the data mean. QARC pushes correlated slots apart via quasiperiodic dynamics at metallic-mean frequencies, maintaining the diversity needed for precise retrieval.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Benchmarks

```bash
# Needle-in-haystack retrieval test
python benchmarks/bench_retrieval.py --workers 8 --tokens 10000 --trials 16
```

## License

Apache 2.0 with Commons Clause. Free for non-commercial use. For commercial licensing, contact christian@tricameral.ai.

## Author

Christian Knopp
