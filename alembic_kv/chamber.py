"""ConceptChamber — projection-free concept codebook for K/V compression.

No learned projections. No training. Model-agnostic.

The codebook operates in native K/V space (num_kv_heads × head_dim).
Tokens are absorbed via soft-attention additive write directly in K/V space.
Retrieval returns the codebook as-is for attention — no projections.

Write:  scores = softmax(normalize(new_kv) @ normalize(codebook)^T / temperature)
        codebook += alpha * scores^T @ new_kv

Read:   Return codebook_k, codebook_v directly as K/V for attention.

QARC resonance prevents mode collapse. No eviction. No training.
Works with any model that has standard multi-head attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .resonance import QARCResonator, AntiResonantInit


class ConceptChamber(nn.Module):
    """Per-layer concept codebook in native K/V space.

    Stores separate K and V codebooks, both (codebook_size, kv_dim).
    Absorption writes into both via normalized soft-attention.
    Retrieval returns codebooks directly as K/V tensors.

    Args:
        kv_dim: num_kv_heads × head_dim — matches model's K/V dimension.
        codebook_size: Number of concept slots. 2048-4096 recommended.
        num_kv_heads: For reshaping output to (heads, head_dim).
        head_dim: For reshaping output.
        qarc_gamma: QARC resonance strength.
        qarc_iterations: QARC resonance steps per cycle.
        write_alpha: Base absorption rate. Actual rate adapts to codebook state.
        write_temp: Temperature for write attention. Lower = sharper writes
            (each token targets fewer slots). Higher = smoother writes.
    """

    def __init__(self, kv_dim: int, codebook_size: int = 2048,
                 num_kv_heads: int = 32, head_dim: int = 128,
                 qarc_gamma: float = 0.03, qarc_iterations: int = 3,
                 write_alpha: float = 0.1, write_temp: float = 0.1):
        super().__init__()
        self.kv_dim = kv_dim
        self.codebook_size = codebook_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.write_alpha = write_alpha
        self.write_temp = write_temp

        # Separate K and V codebooks in native attention space
        # Initialize with unit-normalized random vectors for meaningful dot products
        self.register_buffer('codebook_k', self._init_codebook(codebook_size, kv_dim))
        self.register_buffer('codebook_v', self._init_codebook(codebook_size, kv_dim))

        # QARC resonators (one per codebook)
        self.resonator_k = QARCResonator(dim=kv_dim, gamma=qarc_gamma,
                                          iterations=qarc_iterations)
        self.resonator_v = QARCResonator(dim=kv_dim, gamma=qarc_gamma,
                                          iterations=qarc_iterations)

        self.tokens_absorbed = 0
        # EMA of input magnitude for adaptive scaling
        self._input_ema = 1.0

    @staticmethod
    def _init_codebook(size: int, dim: int) -> torch.Tensor:
        """Initialize codebook with antiresonant unit-normalized vectors.

        Unit normalization ensures softmax attention scores are meaningful
        from the first write (dot products in [-1, 1] range).
        Antiresonant phase spacing maximizes initial diversity.
        """
        # Start with random directions
        cb = torch.randn(size, dim)
        # Orthogonalize as many as possible
        n_ortho = min(size, dim)
        basis = torch.linalg.qr(cb[:n_ortho].T).Q.T
        cb[:n_ortho] = basis
        # Apply antiresonant phase modulation to break remaining symmetries
        for k in range(size):
            phase = k * 2 * math.pi / size
            # Rotate in the first two dims to add phase diversity
            c, s = math.cos(phase), math.sin(phase)
            cb[k, 0], cb[k, 1] = (
                c * cb[k, 0] - s * cb[k, 1],
                s * cb[k, 0] + c * cb[k, 1],
            )
        # Normalize to unit vectors
        cb = F.normalize(cb, dim=-1)
        return cb

    def absorb(self, keys: torch.Tensor, values: torch.Tensor):
        """Absorb K/V into concept codebooks via normalized soft-attention write.

        Uses cosine similarity (normalized dot product) for scoring, so the
        write attention is meaningful regardless of input/codebook magnitudes.
        Temperature controls sharpness: lower = each token writes to fewer slots.

        Args:
            keys: (batch, seq_len, num_kv_heads, head_dim) or (batch, seq_len, kv_dim)
            values: same shape as keys
        """
        if keys.dim() == 4:
            B, T, H, D = keys.shape
            keys = keys.reshape(B, T, H * D)
            values = values.reshape(B, T, H * D)
        else:
            B, T, _ = keys.shape

        k_float = keys.float()
        v_float = values.float()

        # Update input magnitude EMA for adaptive scaling
        input_mag = k_float.norm(dim=-1).mean().item()
        self._input_ema = 0.99 * self._input_ema + 0.01 * input_mag

        # Normalized scoring: cosine similarity / temperature
        # This ensures attention is meaningful regardless of magnitudes
        k_norm = F.normalize(k_float.reshape(B * T, -1), dim=-1)
        cb_norm = F.normalize(self.codebook_k, dim=-1)

        scores = torch.matmul(k_norm, cb_norm.T) / self.write_temp  # (BT, codebook_size)
        attn = F.softmax(scores, dim=-1)

        # Additive update: codebook += alpha * attn^T @ input
        k_flat = k_float.reshape(B * T, -1)
        v_flat = v_float.reshape(B * T, -1)

        k_update = torch.matmul(attn.T, k_flat) / max(B * T, 1)  # (codebook_size, kv_dim)
        v_update = torch.matmul(attn.T, v_flat) / max(B * T, 1)

        self.codebook_k.add_(self.write_alpha * k_update)
        self.codebook_v.add_(self.write_alpha * v_update)

        self.tokens_absorbed += B * T

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V from concept codebooks for attention.

        Returns the raw codebook entries as K/V tensors. The attention
        mechanism uses these directly — no projection needed.

        Args:
            query: (batch, query_len, kv_dim) — unused, kept for interface compat

        Returns:
            keys: (batch, codebook_size, num_kv_heads, head_dim)
            values: (batch, codebook_size, num_kv_heads, head_dim)
        """
        B = query.shape[0]

        keys = self.codebook_k.unsqueeze(0).expand(B, -1, -1)
        values = self.codebook_v.unsqueeze(0).expand(B, -1, -1)

        keys = keys.reshape(B, self.codebook_size, self.num_kv_heads, self.head_dim)
        values = values.reshape(B, self.codebook_size, self.num_kv_heads, self.head_dim)

        return keys, values

    def resonate(self):
        """Apply QARC to both codebooks to prevent mode collapse."""
        self.codebook_k.copy_(self.resonator_k.resonate(self.codebook_k))
        self.codebook_v.copy_(self.resonator_v.resonate(self.codebook_v))

    def reset(self):
        """Clear codebooks (new conversation)."""
        self.codebook_k.copy_(self._init_codebook(
            self.codebook_size, self.kv_dim).to(self.codebook_k.device))
        self.codebook_v.copy_(self._init_codebook(
            self.codebook_size, self.kv_dim).to(self.codebook_v.device))
        self.resonator_k.reset()
        self.resonator_v.reset()
        self.tokens_absorbed = 0
        self._input_ema = 1.0

    @property
    def memory_bytes(self) -> int:
        return (self.codebook_k.nelement() + self.codebook_v.nelement()) * self.codebook_k.element_size()

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 ** 2)

    def diversity(self) -> float:
        """Measure codebook diversity: 1 - mean pairwise cosine similarity."""
        cb_norm = F.normalize(self.codebook_k, dim=-1)
        sim = torch.matmul(cb_norm, cb_norm.T)
        mask = ~torch.eye(self.codebook_size, dtype=torch.bool, device=sim.device)
        return 1.0 - sim[mask].mean().item()
