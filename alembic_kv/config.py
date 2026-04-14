"""Configuration for AlembicKV cache."""

import math
from dataclasses import dataclass
from typing import Optional


# Metallic mean constants — incommensurate frequencies for antiresonance
PHI = (1.0 + math.sqrt(5)) / 2.0
BRONZE = (3.0 + math.sqrt(13)) / 2.0
SUPERGOLDEN = 1.4655712318767680


@dataclass
class AlembicKVConfig:
    """Configuration for AlembicKV cache.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads (num_kv_heads for GQA).
        head_dim: Dimension per attention head.
        budget: Number of concept slots in each chamber's codebook.
            NOT the same as token budget — each concept absorbs many tokens.
            2048 concepts can absorb 100K+ tokens.
        qarc_iterations: QARC resonance steps per cycle. More = better diversity.
        qarc_gamma: QARC resonance strength. 0.03 = conservative.
        write_alpha: Learning rate for codebook absorption. Lower = slower
            absorption, better stability. Higher = faster adaptation.
        resonate_every: Run QARC resonance every N tokens.
        dtype: Storage dtype for codebook. float32 recommended for precision.
    """
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    budget: int = 2048
    qarc_iterations: int = 3
    qarc_gamma: float = 0.03
    write_alpha: float = 0.01
    resonate_every: int = 64
    dtype: str = 'float32'

    @property
    def kv_dim(self) -> int:
        """Total K or V dimension: num_heads * head_dim."""
        return self.num_heads * self.head_dim

    @property
    def codebook_bytes(self) -> int:
        """Codebook VRAM per layer in bytes."""
        elem_bytes = 4 if self.dtype == 'float32' else 2
        return self.budget * self.kv_dim * elem_bytes

    @property
    def total_bytes(self) -> int:
        """Total VRAM for all layers (codebook K + V only, no projections)."""
        return self.codebook_bytes * 2 * self.num_layers  # K and V codebooks

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 ** 2)

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    def standard_cache_gb(self, seq_len: int) -> float:
        """VRAM a standard KV cache needs for seq_len tokens."""
        elem_bytes = 2  # bf16
        per_layer = seq_len * self.kv_dim * 2 * elem_bytes  # K + V
        return per_layer * self.num_layers / (1024 ** 3)

    def compression_at(self, seq_len: int) -> float:
        """Compression ratio vs standard cache at given sequence length."""
        standard = self.standard_cache_gb(seq_len)
        return standard / max(self.total_gb, 1e-9)

    def summary(self) -> str:
        """Print a summary of memory usage."""
        lines = [
            f"AlembicKV Config:",
            f"  Layers: {self.num_layers}, Heads: {self.num_heads}, HeadDim: {self.head_dim}",
            f"  Concepts: {self.budget} × {self.kv_dim}d per layer",
            f"  Total VRAM: {self.total_mb:.1f} MB ({self.total_gb:.2f} GB)",
            f"  Compression vs standard KV cache:",
            f"    At   1K tokens: {self.compression_at(1000):.1f}x  (std: {self.standard_cache_gb(1000)*1024:.0f} MB)",
            f"    At  10K tokens: {self.compression_at(10000):.1f}x  (std: {self.standard_cache_gb(10000):.2f} GB)",
            f"    At 100K tokens: {self.compression_at(100000):.1f}x  (std: {self.standard_cache_gb(100000):.2f} GB)",
            f"    At   1M tokens: {self.compression_at(1000000):.0f}x  (std: {self.standard_cache_gb(1000000):.1f} GB)",
        ]
        return "\n".join(lines)
