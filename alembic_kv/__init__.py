"""AlembicKV — Fixed-memory KV cache replacement for transformers.

Drop-in replacement for the standard attention KV cache that uses a fixed-size
antiresonant concept codebook instead of unbounded per-token storage.

Standard KV cache: O(n) memory in sequence length — explodes at long context.
AlembicKV:         O(1) memory — fixed budget regardless of sequence length.

The codebook absorbs tokens via soft-write (additive, never evicts) and
reconstructs K/V via learned projections. QARC (Quasicrystal Autopoietic
Resonance Cascade) dynamics prevent mode collapse, ensuring the codebook
retains diverse concepts instead of degenerating into a mean-field summary.

Usage:
    from alembic_kv import AlembicKVCache, AlembicKVConfig

    config = AlembicKVConfig(
        num_layers=32,
        num_heads=32,
        head_dim=128,
        budget=2048,       # concept slots (NOT token slots)
    )
    cache = AlembicKVCache(config)

    # 2048 concepts × 4096 dim × 32 layers × 4 bytes = ~1 GB
    # Stores 100K+ tokens of context in that fixed budget.
    # Standard KV cache for 100K tokens: 52 GB.
"""

from .chamber import ConceptChamber
from .cache import AlembicKVCache
from .resonance import QARCResonator, AntiResonantInit
from .config import AlembicKVConfig

__version__ = "0.1.0"

__all__ = [
    "AlembicKVCache",
    "ConceptChamber",
    "QARCResonator",
    "AntiResonantInit",
    "AlembicKVConfig",
]
