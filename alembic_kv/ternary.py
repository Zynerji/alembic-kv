"""Ternary codebook encoding for AlembicKV.

Compresses concept vectors from float32 (32 bits/dim) to ternary
(+1, 0, -1) with a per-concept scale factor (~2 bits/dim effective).

float32 codebook: 2048 concepts × 4096 dim = 33 MB per layer
ternary codebook: 2048 concepts × 4096 dim = 2 MB per layer (16x smaller)

Encoding:
    For each concept vector c:
      scale = c.abs().mean()          (1 float per concept)
      threshold = scale * sparsity    (controls how many zeros)
      ternary[i] = +1 if c[i] > threshold
                   -1 if c[i] < -threshold
                    0 otherwise
      Store: (scale, ternary_packed)

    Packed as int8: 4 ternary values per byte
      encoding: -1 → 0, 0 → 1, +1 → 2
      pack: t[0]*27 + t[1]*9 + t[2]*3 + t[3]  (base-3 packing, 4 per byte)

Decoding:
    c_reconstructed = scale * unpack(ternary_packed)

Write path:
    1. Dequantize codebook to float32 (temporary)
    2. Apply soft-attention absorption (full precision)
    3. Re-quantize to ternary (store)

This gives the memory savings of ternary storage with the precision
of float32 absorption dynamics.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class TernaryCodebook:
    """Ternary-encoded concept codebook with float32 absorption.

    Stores concepts as ternary vectors + scale factors.
    Absorbs in float32, re-quantizes after each write cycle.
    """

    def __init__(self, size: int, dim: int, sparsity: float = 0.5,
                 device: torch.device = None):
        """
        Args:
            size: number of concept slots
            dim: dimension per concept
            sparsity: fraction of values zeroed (0.5 = 50% sparse)
            device: torch device
        """
        self.size = size
        self.dim = dim
        self.sparsity = sparsity
        self.device = device or torch.device('cpu')

        # Per-concept scale factors (float32)
        self.scales = torch.ones(size, device=self.device)

        # Ternary values packed as int8 (-1, 0, +1 stored as 0, 1, 2)
        # 4 values per byte via base-3 packing
        self.packed_dim = (dim + 3) // 4  # ceil(dim / 4)
        self.packed = torch.zeros(size, self.packed_dim,
                                  dtype=torch.uint8, device=self.device)

        # Initialize with random ternary (diverse starting point)
        self._init_random()

    def _init_random(self):
        """Initialize with random ternary vectors."""
        for i in range(self.size):
            # Random direction, then quantize
            vec = torch.randn(self.dim, device=self.device)
            vec = F.normalize(vec, dim=0)
            self.scales[i] = vec.abs().mean()
            ternary = self._quantize_vector(vec, self.scales[i])
            self.packed[i] = self._pack(ternary)

    def _quantize_vector(self, vec: torch.Tensor, scale: float) -> torch.Tensor:
        """Quantize a float vector to ternary (-1, 0, +1)."""
        threshold = scale * self.sparsity
        ternary = torch.zeros_like(vec, dtype=torch.int8)
        ternary[vec > threshold] = 1
        ternary[vec < -threshold] = -1
        return ternary

    def _pack(self, ternary: torch.Tensor) -> torch.Tensor:
        """Pack ternary values (4 per byte) using base-3 encoding.

        -1 → 0, 0 → 1, +1 → 2
        pack: t[0]*27 + t[1]*9 + t[2]*3 + t[3]
        """
        # Map: -1→0, 0→1, +1→2
        mapped = (ternary + 1).to(torch.uint8)  # now 0, 1, 2

        # Pad to multiple of 4
        padded = torch.zeros(self.packed_dim * 4, dtype=torch.uint8,
                             device=self.device)
        padded[:self.dim] = mapped

        # Pack 4 values per byte
        reshaped = padded.reshape(-1, 4)
        packed = (reshaped[:, 0] * 27 + reshaped[:, 1] * 9 +
                  reshaped[:, 2] * 3 + reshaped[:, 3])
        return packed.to(torch.uint8)

    def _unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Unpack ternary values from base-3 encoding."""
        result = torch.zeros(self.packed_dim * 4, dtype=torch.int8,
                             device=self.device)
        remaining = packed.to(torch.int16)  # avoid uint8 overflow

        result[0::4] = (remaining // 27) - 1       # map 0,1,2 → -1,0,+1
        remaining = remaining % 27
        result[1::4] = (remaining // 9) - 1
        remaining = remaining % 9
        result[2::4] = (remaining // 3) - 1
        result[3::4] = (remaining % 3) - 1

        return result[:self.dim]

    def dequantize(self) -> torch.Tensor:
        """Reconstruct full float32 codebook from ternary encoding.

        Returns: (size, dim) float32 tensor
        """
        result = torch.zeros(self.size, self.dim, device=self.device)
        for i in range(self.size):
            ternary = self._unpack(self.packed[i]).float()
            result[i] = self.scales[i] * ternary
        return result

    def dequantize_batch(self) -> torch.Tensor:
        """Vectorized dequantize — faster for full codebook read."""
        # Unpack all at once
        all_ternary = torch.zeros(self.size, self.dim, device=self.device)
        for i in range(self.size):
            all_ternary[i] = self._unpack(self.packed[i]).float()
        return self.scales.unsqueeze(1) * all_ternary

    def quantize_from(self, float_codebook: torch.Tensor):
        """Quantize a float32 codebook into ternary encoding.

        Args:
            float_codebook: (size, dim) float32 tensor
        """
        for i in range(self.size):
            vec = float_codebook[i]
            self.scales[i] = vec.abs().mean().clamp(min=1e-8)
            ternary = self._quantize_vector(vec, self.scales[i])
            self.packed[i] = self._pack(ternary)

    @property
    def memory_bytes(self) -> int:
        """Actual memory usage in bytes."""
        scales_bytes = self.size * 4  # float32
        packed_bytes = self.size * self.packed_dim  # uint8
        return scales_bytes + packed_bytes

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 ** 2)

    @property
    def float32_equivalent_mb(self) -> float:
        """What this codebook would cost in float32."""
        return self.size * self.dim * 4 / (1024 ** 2)

    @property
    def compression_ratio(self) -> float:
        return self.float32_equivalent_mb / max(self.memory_mb, 1e-9)

    def to(self, device: torch.device) -> 'TernaryCodebook':
        self.device = device
        self.scales = self.scales.to(device)
        self.packed = self.packed.to(device)
        return self
