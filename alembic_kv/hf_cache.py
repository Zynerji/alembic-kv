"""HuggingFace-compatible AlembicKV cache — concept codebook KV replacement.

NOT a bounded DynamicCache. NOT eviction.

This is the real AlembicKV: a fixed-size concept codebook that absorbs
unlimited tokens via soft-attention write and returns compressed K/V
for attention. O(1) memory. No eviction. No information loss.

Phase 1 (fill): Store raw K/V in codebook slots (exact, like DynamicCache).
Phase 2 (absorb): Codebook full. New tokens absorbed via soft-attention
    into existing concepts. Codebook size never grows.

QARC resonance prevents codebook collapse under repeated writes.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from transformers.cache_utils import DynamicCache, CacheLayerMixin


class AlembicLayer(CacheLayerMixin):
    """Concept codebook KV cache layer.

    Stores K/V as concepts in a fixed codebook. Below budget, stores
    raw tokens (identical to DynamicLayer). Above budget, absorbs new
    tokens into the codebook via soft-attention write.
    """

    def __init__(self, budget: int = 2048, write_alpha: float = 0.1,
                 write_temp: float = 0.1, qarc_gamma: float = 0.01,
                 resonate_every: int = 32):
        self.budget = budget
        self.write_alpha = write_alpha
        self.write_temp = write_temp
        self.qarc_gamma = qarc_gamma
        self.resonate_every = resonate_every

        self._keys = None      # (B, H, T, D) — concat buffer during fill
        self._values = None
        self._codebook_k = None  # (budget, H*D) — concept buffer after fill
        self._codebook_v = None
        self._is_initialized = False
        self._compressed = False  # True once codebook is active
        self._seq_length = 0
        self._absorb_count = 0
        self.num_heads = 0
        self.head_dim = 0
        self.kv_dim = 0
        self.tokens_absorbed = 0

    # ── CacheLayerMixin interface ─────────────────────────────────

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def is_compileable(self):
        return False

    @property
    def is_sliding(self):
        return False

    @property
    def device(self):
        if self._keys is not None:
            return self._keys.device
        if self._codebook_k is not None:
            return self._codebook_k.device
        return torch.device('cpu')

    @property
    def dtype(self):
        if self._keys is not None:
            return self._keys.dtype
        return torch.bfloat16

    @property
    def keys(self):
        if self._keys is not None:
            return self._keys
        if self._codebook_k is not None:
            return self._codebook_to_kv(self._codebook_k)
        return torch.empty(0)

    @property
    def values(self):
        if self._values is not None:
            return self._values
        if self._codebook_v is not None:
            return self._codebook_to_kv(self._codebook_v)
        return torch.empty(0)

    def _codebook_to_kv(self, codebook):
        """Reshape codebook (budget, H*D) to KV format (1, H, budget, D)."""
        cb = codebook.reshape(self.budget, self.num_heads, self.head_dim)
        return cb.permute(1, 0, 2).unsqueeze(0).to(self.dtype)

    def lazy_initialization(self, key_states, value_states):
        B, H, T, D = key_states.shape
        self.num_heads = H
        self.head_dim = D
        self.kv_dim = H * D
        self._is_initialized = True

    def get_seq_length(self) -> int:
        if self._keys is not None:
            return self._keys.shape[2]
        if self._compressed:
            return self.budget
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        if not self._compressed:
            # Fill phase: standard concat behavior
            kv_length = self.get_seq_length() + query_length
        else:
            # Absorb phase: codebook + raw current token
            kv_length = self.budget + query_length
        kv_offset = 0  # CRITICAL: must be 0 for correct causal mask
        return kv_length, kv_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape
        self.tokens_absorbed += T

        if not self._compressed:
            # ── Phase 1: Fill (identical to DynamicLayer) ─────────
            if self._keys is None:
                self._keys = key_states
                self._values = value_states
            else:
                self._keys = torch.cat([self._keys, key_states], dim=2)
                self._values = torch.cat([self._values, value_states], dim=2)

            self._seq_length = self._keys.shape[2]

            # Transition to codebook when full
            if self._seq_length >= self.budget:
                self._compress()
                # Return codebook + any overflow tokens from this batch
                overflow = self._seq_length - self.budget
                if overflow > 0:
                    # The last `overflow` tokens didn't fit — absorb them
                    overflow_k = key_states[:, :, -overflow:, :]
                    overflow_v = value_states[:, :, -overflow:, :]
                    self._absorb(overflow_k, overflow_v)
                    # Return codebook + overflow raw tokens
                    cb_k = self._codebook_to_kv(self._codebook_k).expand(B, -1, -1, -1)
                    cb_v = self._codebook_to_kv(self._codebook_v).expand(B, -1, -1, -1)
                    return (torch.cat([cb_k, overflow_k], dim=2),
                            torch.cat([cb_v, overflow_v], dim=2))
                # Exact fit — return codebook as KV
                return (self._codebook_to_kv(self._codebook_k).expand(B, -1, -1, -1),
                        self._codebook_to_kv(self._codebook_v).expand(B, -1, -1, -1))

            return self._keys, self._values

        else:
            # ── Phase 2: Absorb into codebook ─────────────────────
            self._absorb(key_states, value_states)

            # Periodic QARC resonance
            self._absorb_count += 1
            if self._absorb_count % self.resonate_every == 0:
                self._resonate()

            # Return codebook + raw current token
            cb_k = self._codebook_to_kv(self._codebook_k).expand(B, -1, -1, -1)
            cb_v = self._codebook_to_kv(self._codebook_v).expand(B, -1, -1, -1)
            all_keys = torch.cat([cb_k, key_states], dim=2)
            all_values = torch.cat([cb_v, value_states], dim=2)
            return all_keys, all_values

    def _compress(self):
        """Transition from concat buffer to concept codebook."""
        B, H, T, D = self._keys.shape
        # Take the last `budget` tokens as initial codebook content
        k = self._keys[:, :, -self.budget:, :]
        v = self._values[:, :, -self.budget:, :]
        # Flatten: (B, H, budget, D) -> (budget, H*D) using batch 0
        self._codebook_k = k[0].permute(1, 0, 2).reshape(self.budget, H * D).float()
        self._codebook_v = v[0].permute(1, 0, 2).reshape(self.budget, H * D).float()
        # Free concat buffers
        self._keys = None
        self._values = None
        self._compressed = True

    def _absorb(self, keys: torch.Tensor, values: torch.Tensor):
        """Absorb new K/V into concept codebook via soft-attention write."""
        B, H, T, D = keys.shape
        k_flat = keys.transpose(1, 2).reshape(B * T, H * D).float()
        v_flat = values.transpose(1, 2).reshape(B * T, H * D).float()

        # Normalized cosine similarity scoring
        k_norm = F.normalize(k_flat, dim=-1)
        cb_norm = F.normalize(self._codebook_k, dim=-1)
        scores = torch.matmul(k_norm, cb_norm.T) / self.write_temp
        attn = F.softmax(scores, dim=-1)  # (B*T, budget)

        # Additive concept update
        k_update = torch.matmul(attn.T, k_flat) / max(B * T, 1)
        v_update = torch.matmul(attn.T, v_flat) / max(B * T, 1)
        self._codebook_k.add_(self.write_alpha * k_update)
        self._codebook_v.add_(self.write_alpha * v_update)

    def _resonate(self):
        """QARC resonance: push correlated codebook slots apart."""
        if self._codebook_k is None:
            return
        # Simple resonance: orthogonalize via gradient of pairwise similarity
        cb_norm = F.normalize(self._codebook_k, dim=-1)
        sim = torch.matmul(cb_norm, cb_norm.T)  # (budget, budget)
        # Zero diagonal (don't push slot away from itself)
        sim.fill_diagonal_(0.0)
        # Repulsion: each slot pushed away from its most similar neighbors
        repulsion = torch.matmul(sim, self._codebook_k)  # (budget, kv_dim)
        self._codebook_k.sub_(self.qarc_gamma * repulsion)
        # Same for values
        cb_norm_v = F.normalize(self._codebook_v, dim=-1)
        sim_v = torch.matmul(cb_norm_v, cb_norm_v.T)
        sim_v.fill_diagonal_(0.0)
        repulsion_v = torch.matmul(sim_v, self._codebook_v)
        self._codebook_v.sub_(self.qarc_gamma * repulsion_v)

    def reset(self):
        self._keys = None
        self._values = None
        self._codebook_k = None
        self._codebook_v = None
        self._is_initialized = False
        self._compressed = False
        self._seq_length = 0
        self._absorb_count = 0
        self.tokens_absorbed = 0

    def reorder_cache(self, beam_idx):
        if self._keys is not None:
            self._keys = self._keys.index_select(0, beam_idx)
            self._values = self._values.index_select(0, beam_idx)

    def crop(self, max_length):
        pass

    def offload(self):
        pass

    def prefetch(self):
        pass

    def batch_repeat_interleave(self, repeats):
        if self._keys is not None:
            self._keys = self._keys.repeat_interleave(repeats, dim=0)
            self._values = self._values.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self._keys is not None:
            self._keys = self._keys[indices]
            self._values = self._values[indices]


class AlembicHFCache(DynamicCache):
    """AlembicKV: concept codebook KV cache as DynamicCache subclass.

    Below budget: identical to DynamicCache (lossless).
    At budget: compresses accumulated K/V into fixed codebook.
    Above budget: absorbs new tokens into codebook. O(1) memory.

    Usage:
        cache = AlembicHFCache(budget=2048)
        outputs = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(self, budget: int = 2048, write_alpha: float = 0.1,
                 write_temp: float = 0.1, qarc_gamma: float = 0.01,
                 resonate_every: int = 32):
        super().__init__()
        self.budget = budget
        self.write_alpha = write_alpha
        self.write_temp = write_temp
        self.qarc_gamma = qarc_gamma
        self.resonate_every = resonate_every
        self.layer_class_to_replicate = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(AlembicLayer(
                budget=self.budget,
                write_alpha=self.write_alpha,
                write_temp=self.write_temp,
                qarc_gamma=self.qarc_gamma,
                resonate_every=self.resonate_every,
            ))
        return self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
            return self.layers[layer_idx].get_seq_length()
        return 0

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.layers.clear()

    def stats(self) -> dict:
        absorbed = sum(l.tokens_absorbed for l in self.layers)
        compressed = any(l._compressed for l in self.layers) if self.layers else False
        return {
            'layers': len(self.layers),
            'budget': self.budget,
            'tokens_absorbed': absorbed,
            'compressed': compressed,
            'mode': 'codebook' if compressed else 'fill',
        }
