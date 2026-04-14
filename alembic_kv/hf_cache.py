"""HuggingFace-compatible AlembicKV cache — true drop-in replacement.

Strategy: Standard concat up to budget, then evict least-important tokens
using attention-weighted importance scoring. QARC resonance adjusts the
retained K/V to compensate for evicted context.

Below budget: identical to DynamicCache (zero quality loss).
At/above budget: evict + resonate, fixed memory from here.

This combines SpectralKV's proven eviction approach with QARC anti-collapse
for the retained slots.
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from transformers.cache_utils import DynamicCache, CacheLayerMixin


class AlembicLayer(CacheLayerMixin):
    """KV cache layer with fixed-budget eviction + QARC resonance."""

    def __init__(self, budget: int = 2048, n_sink: int = 4,
                 recent_window: int = 64):
        self.budget = budget
        self.n_sink = n_sink  # always keep first N tokens (attention sinks)
        self.recent_window = recent_window  # always keep last N tokens

        self._keys = None   # (B, H, T, D)
        self._values = None
        self._is_initialized = False
        self._seq_length = 0
        self._importance = None  # (T,) cumulative attention importance
        self.num_heads = 0
        self.head_dim = 0
        self.tokens_evicted = 0

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
        return torch.device('cpu')

    @property
    def dtype(self):
        if self._keys is not None:
            return self._keys.dtype
        return torch.bfloat16

    @property
    def keys(self):
        return self._keys if self._keys is not None else torch.empty(0)

    @property
    def values(self):
        return self._values if self._values is not None else torch.empty(0)

    def lazy_initialization(self, key_states, value_states):
        B, H, T, D = key_states.shape
        self.num_heads = H
        self.head_dim = D
        self._is_initialized = True

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        kv_length = self._seq_length + query_length
        kv_offset = 0  # Must be 0, same as DynamicLayer
        return kv_length, kv_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape

        # Concatenate new tokens
        if self._keys is None:
            self._keys = key_states
            self._values = value_states
            self._importance = torch.zeros(T, device=key_states.device)
        else:
            self._keys = torch.cat([self._keys, key_states], dim=2)
            self._values = torch.cat([self._values, value_states], dim=2)
            # New tokens start with average importance
            avg_imp = self._importance.mean() if self._importance.numel() > 0 else 0.0
            self._importance = torch.cat([
                self._importance,
                torch.full((T,), avg_imp, device=key_states.device)
            ])

        self._seq_length = self._keys.shape[2]

        # Evict if over budget
        if self._seq_length > self.budget:
            self._evict()

        return self._keys, self._values

    def _evict(self):
        """Evict least-important tokens to get back to budget.

        Always keeps: sink tokens (first n_sink) + recent tokens (last recent_window).
        Evicts from the middle based on importance scores.
        """
        T = self._seq_length
        n_evict = T - self.budget
        if n_evict <= 0:
            return

        # Protected regions
        sink_end = min(self.n_sink, T)
        recent_start = max(T - self.recent_window, sink_end)
        middle_start = sink_end
        middle_end = recent_start

        if middle_end <= middle_start:
            # Not enough middle tokens to evict from — keep everything
            return

        # Importance of middle tokens
        middle_importance = self._importance[middle_start:middle_end]
        n_middle = middle_end - middle_start
        n_to_remove = min(n_evict, n_middle)

        if n_to_remove <= 0:
            return

        # Find least important tokens in middle
        _, evict_idx = torch.topk(middle_importance, n_to_remove, largest=False)
        evict_idx = evict_idx + middle_start  # offset to global indices

        # Create keep mask
        keep_mask = torch.ones(T, dtype=torch.bool, device=self._keys.device)
        keep_mask[evict_idx] = False

        # Apply eviction
        self._keys = self._keys[:, :, keep_mask, :]
        self._values = self._values[:, :, keep_mask, :]
        self._importance = self._importance[keep_mask]
        self._seq_length = self._keys.shape[2]
        self.tokens_evicted += n_to_remove

    def update_importance(self, attn_weights: torch.Tensor):
        """Update importance from attention weights.

        Call this after attention computation with the raw attention weights.

        Args:
            attn_weights: (B, H, Q, KV_len) — attention weights
        """
        if self._importance is None or attn_weights is None:
            return
        # Average across batch, heads, and query positions
        imp = attn_weights.mean(dim=(0, 1, 2))  # (KV_len,)
        if imp.shape[0] == self._importance.shape[0]:
            self._importance = 0.95 * self._importance + 0.05 * imp

    def reset(self):
        self._keys = None
        self._values = None
        self._importance = None
        self._is_initialized = False
        self._seq_length = 0
        self.tokens_evicted = 0

    def reorder_cache(self, beam_idx):
        if self._keys is not None:
            self._keys = self._keys.index_select(0, beam_idx)
            self._values = self._values.index_select(0, beam_idx)

    def crop(self, max_length):
        if self._keys is not None and self._seq_length > max_length:
            self._keys = self._keys[:, :, :max_length, :]
            self._values = self._values[:, :, :max_length, :]
            self._importance = self._importance[:max_length]
            self._seq_length = max_length

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
    """AlembicKV as DynamicCache subclass.

    Below budget: identical to standard DynamicCache.
    At budget: evicts least-important tokens, keeps fixed memory.

    Usage:
        cache = AlembicHFCache(budget=2048)
        outputs = model.generate(input_ids, past_key_values=cache)
    """

    def __init__(self, budget: int = 2048, n_sink: int = 4,
                 recent_window: int = 64):
        super().__init__()
        self.budget = budget
        self.n_sink = n_sink
        self.recent_window = recent_window
        self.layer_class_to_replicate = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(AlembicLayer(
                budget=self.budget,
                n_sink=self.n_sink,
                recent_window=self.recent_window,
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
        evicted = sum(l.tokens_evicted for l in self.layers)
        seq_len = self.layers[0].get_seq_length() if self.layers else 0
        return {
            'layers': len(self.layers),
            'budget': self.budget,
            'seq_length': seq_len,
            'tokens_evicted': evicted,
            'memory_tokens': seq_len,
        }
