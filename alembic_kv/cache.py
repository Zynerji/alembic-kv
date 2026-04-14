"""AlembicKVCache — projection-free concept-compressed KV cache.

No learned projections. No training. Model-agnostic drop-in replacement.

Standard DynamicCache at 1M tokens, 70B model (8 KV heads, 128 dim, 80 layers):
    = 328 GB

AlembicKVCache with 2048 concept slots:
    = 672 MB (fixed, forever)

Compression: 488x. The number never changes regardless of sequence length.
"""

import torch
from typing import Tuple, List

from .chamber import ConceptChamber
from .config import AlembicKVConfig


class AlembicKVCache:
    """Projection-free concept-compressed KV cache.

    Each layer gets a ConceptChamber with separate K/V codebooks.
    Tokens are absorbed directly in native K/V space.
    Retrieval returns the codebook as-is — pure attention, no projections.
    """

    def __init__(self, config: AlembicKVConfig):
        self.config = config
        self.chambers: List[ConceptChamber] = []
        for i in range(config.num_layers):
            self.chambers.append(ConceptChamber(
                kv_dim=config.kv_dim,
                codebook_size=config.budget,
                num_kv_heads=config.num_heads,
                head_dim=config.head_dim,
                qarc_gamma=config.qarc_gamma,
                qarc_iterations=config.qarc_iterations,
                write_alpha=config.write_alpha,
            ))
        self._seen_tokens = 0
        self.resonate_every = config.resonate_every

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, cache_kwargs: dict = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Absorb new K/V and return concept codebook + raw token for attention.

        Args:
            key_states: (batch, num_heads, new_seq_len, head_dim)
            value_states: (batch, num_heads, new_seq_len, head_dim)
            layer_idx: which transformer layer

        Returns:
            (keys, values): (batch, num_heads, codebook_size + new_seq_len, head_dim)
        """
        chamber = self.chambers[layer_idx]
        B, H, T, D = key_states.shape

        # Reshape to (B, T, H, D) for absorption
        k_in = key_states.transpose(1, 2)
        v_in = value_states.transpose(1, 2)

        # Absorb into concept codebooks
        chamber.absorb(k_in, v_in)

        if layer_idx == 0:
            self._seen_tokens += T

        # Periodic QARC
        if layer_idx == 0 and self._seen_tokens % self.resonate_every == 0:
            self._resonate_all()

        # Retrieve: codebook K/V in (B, codebook_size, H, D)
        query = k_in.reshape(B, T, H * D)
        r_keys, r_values = chamber.retrieve(query)

        # Transpose to (B, H, codebook_size, D)
        r_keys = r_keys.transpose(1, 2).to(key_states.dtype)
        r_values = r_values.transpose(1, 2).to(value_states.dtype)

        # Concat: [concept_codebook, raw_current_token]
        all_keys = torch.cat([r_keys, key_states], dim=2)
        all_values = torch.cat([r_values, value_states], dim=2)

        return all_keys, all_values

    # ── DynamicCache interface ────────────────────────────────────

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens

    def get_max_length(self) -> int:
        return None

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    def __len__(self) -> int:
        return len(self.chambers)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chamber = self.chambers[layer_idx]
        k = chamber.codebook_k.unsqueeze(0)
        v = chamber.codebook_v.unsqueeze(0)
        k = k.reshape(1, chamber.codebook_size, chamber.num_kv_heads, chamber.head_dim).transpose(1, 2)
        v = v.reshape(1, chamber.codebook_size, chamber.num_kv_heads, chamber.head_dim).transpose(1, 2)
        return k, v

    def __iter__(self):
        for i in range(len(self.chambers)):
            yield self[i]

    def reorder_cache(self, beam_idx: torch.Tensor):
        pass

    # ── Resonance ─────────────────────────────────────────────────

    def _resonate_all(self):
        for chamber in self.chambers:
            chamber.resonate()

    # ── Lifecycle ─────────────────────────────────────────────────

    def reset(self):
        for chamber in self.chambers:
            chamber.reset()
        self._seen_tokens = 0

    def to(self, device: torch.device) -> 'AlembicKVCache':
        for chamber in self.chambers:
            chamber.to(device)
        return self

    # ── Stats ─────────────────────────────────────────────────────

    @property
    def memory_mb(self) -> float:
        return sum(c.memory_mb for c in self.chambers)

    def stats(self) -> dict:
        std_gb = self.config.standard_cache_gb(self._seen_tokens)
        mem_mb = self.memory_mb
        return {
            'seen_tokens': self._seen_tokens,
            'codebook_size': self.config.budget,
            'kv_dim': self.config.kv_dim,
            'num_layers': len(self.chambers),
            'memory_mb': mem_mb,
            'memory_gb': mem_mb / 1024,
            'tokens_absorbed': sum(c.tokens_absorbed for c in self.chambers),
            'standard_cache_gb': std_gb,
            'compression_ratio': (std_gb * 1024 / max(mem_mb, 0.001)) if self._seen_tokens > 0 else 0,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (f"AlembicKVCache(layers={s['num_layers']}, "
                f"slots={s['codebook_size']}×{s['kv_dim']}d, "
                f"absorbed={s['seen_tokens']} tokens, "
                f"mem={s['memory_mb']:.1f}MB, "
                f"compression={s['compression_ratio']:.1f}x)")
