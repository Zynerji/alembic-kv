"""HuggingFace-compatible AlembicKV cache — hybrid concept codebook + recent window.

The key insight: attention needs EXACT K/V for recent tokens but can tolerate
compressed representations for older context. AlembicKV exploits this:

  [codebook (compressed old context)] + [recent window (exact raw tokens)]

Below budget: identical to DynamicCache (standard concat).
At budget: compress oldest tokens into codebook, keep recent window exact.
Above budget: absorb new old tokens into codebook, slide recent window.

Memory: O(codebook_size + window_size) = O(1) regardless of total tokens.

vs Eviction (StreamingLLM, H2O, SpectralKV):
  - Eviction DISCARDS old tokens — information is lost forever
  - AlembicKV COMPRESSES old tokens into concepts — information is retained

vs Standard DynamicCache:
  - DynamicCache stores every token — O(n) memory, explodes at long context
  - AlembicKV caps memory at budget — O(1), fixed VRAM
"""

import math
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from transformers.cache_utils import DynamicCache, CacheLayerMixin


class AlembicLayer(CacheLayerMixin):
    """Hybrid concept codebook + recent window KV cache layer.

    Maintains two buffers:
      1. Codebook (budget slots): compressed old context via soft-attention absorption
      2. Recent window (window_size tokens): exact raw K/V, slides forward

    Attention sees: [codebook | recent_window | current_token]
    """

    def __init__(self, budget: int = 2048, window_size: int = 128,
                 write_alpha: float = 0.1, write_temp: float = 0.1,
                 qarc_gamma: float = 0.01, resonate_every: int = 64,
                 ternary: bool = False, ternary_sparsity: float = 0.5):
        self.budget = budget           # codebook concept slots
        self.window_size = window_size # exact recent tokens to keep
        self.write_alpha = write_alpha
        self.write_temp = write_temp
        self.qarc_gamma = qarc_gamma
        self.resonate_every = resonate_every
        self.use_ternary = ternary
        self.ternary_sparsity = ternary_sparsity

        # Fill phase: standard concat
        self._keys = None      # (B, H, T, D)
        self._values = None

        # Codebook phase
        self._codebook_k = None  # (budget, H*D) float32 or TernaryCodebook
        self._codebook_v = None
        self._ternary_k = None   # TernaryCodebook when ternary=True
        self._ternary_v = None
        self._recent_k = None    # (B, H, window_size, D) raw recent tokens
        self._recent_v = None

        self._is_initialized = False
        self._compressed = False
        self._needs_compress = False
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
        if self._compressed:
            float_k = self._get_codebook_float('k')
            if float_k is not None:
                return self._codebook_to_kv(float_k)
        return torch.empty(0)

    @property
    def values(self):
        if self._values is not None:
            return self._values
        if self._compressed:
            float_v = self._get_codebook_float('v')
            if float_v is not None:
                return self._codebook_to_kv(float_v)
        return torch.empty(0)

    def _codebook_to_kv(self, codebook):
        """(budget, H*D) -> (1, H, budget, D)"""
        if codebook is None:
            return torch.empty(0)
        cb = codebook.reshape(self.budget, self.num_heads, self.head_dim)
        return cb.permute(1, 0, 2).unsqueeze(0).to(self.dtype)

    def _get_codebook_float(self, which='k'):
        """Get codebook as float32 tensor, dequantizing ternary if needed."""
        if self.use_ternary:
            tcb = self._ternary_k if which == 'k' else self._ternary_v
            return tcb.dequantize_batch() if tcb is not None else None
        return self._codebook_k if which == 'k' else self._codebook_v

    def _set_codebook(self, k_float: torch.Tensor, v_float: torch.Tensor):
        """Store codebook, quantizing to ternary if enabled."""
        if self.use_ternary:
            from .ternary import TernaryCodebook
            if self._ternary_k is None:
                self._ternary_k = TernaryCodebook(
                    self.budget, self.kv_dim,
                    sparsity=self.ternary_sparsity, device=k_float.device)
                self._ternary_v = TernaryCodebook(
                    self.budget, self.kv_dim,
                    sparsity=self.ternary_sparsity, device=v_float.device)
            self._ternary_k.quantize_from(k_float)
            self._ternary_v.quantize_from(v_float)
            # Keep float refs as None to save memory
            self._codebook_k = None
            self._codebook_v = None
        else:
            self._codebook_k = k_float
            self._codebook_v = v_float

    def _build_full_kv(self):
        """Build [codebook | recent] K/V tensors."""
        B = self._recent_k.shape[0] if self._recent_k is not None else 1
        float_k = self._get_codebook_float('k')
        float_v = self._get_codebook_float('v')
        cb_k = self._codebook_to_kv(float_k).expand(B, -1, -1, -1)
        cb_v = self._codebook_to_kv(float_v).expand(B, -1, -1, -1)
        if self._recent_k is not None and self._recent_k.shape[2] > 0:
            all_k = torch.cat([cb_k, self._recent_k], dim=2)
            all_v = torch.cat([cb_v, self._recent_v], dim=2)
        else:
            all_k, all_v = cb_k, cb_v
        return all_k, all_v

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
            recent_len = self._recent_k.shape[2] if self._recent_k is not None else 0
            return self.budget + recent_len
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def get_mask_sizes(self, query_length: int) -> Tuple[int, int]:
        if self._needs_compress and not self._compressed:
            # Next update: compress → absorb new into window
            # Returns codebook + window (new tokens slide into window)
            kv_length = self.budget + self.window_size
        elif not self._compressed:
            recent_len = self._recent_k.shape[2] if self._recent_k is not None else 0
            kv_length = self.budget + min(recent_len + query_length, self.window_size)
        kv_offset = 0
        return kv_length, kv_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._is_initialized:
            self.lazy_initialization(key_states, value_states)

        B, H, T, D = key_states.shape
        self.tokens_absorbed += T

        # Deferred compression from previous large prefill
        if self._needs_compress and not self._compressed:
            self._compress()
            self._needs_compress = False
            # Now in compressed mode — fall through to absorb phase

        if not self._compressed:
            # ── Fill phase: standard concat ───────────────────────
            if self._keys is None:
                self._keys = key_states
                self._values = value_states
            else:
                self._keys = torch.cat([self._keys, key_states], dim=2)
                self._values = torch.cat([self._values, value_states], dim=2)

            total = self._keys.shape[2]

            # Compress when we have enough — but only if this isn't the
            # initial prefill (where the mask was sized for full concat).
            # Defer compression if we just received a large chunk.
            if total >= self.budget + self.window_size:
                if total == key_states.shape[2]:
                    # First call with a big prefill — return full concat now,
                    # compress on the NEXT update call
                    self._needs_compress = True
                    return self._keys, self._values
                else:
                    # Accumulated past budget — compress now
                    self._compress()
                    all_k, all_v = self._build_full_kv()
                    return all_k, all_v

            return self._keys, self._values

        else:
            # ── Hybrid phase: absorb old into codebook, slide window ──

            # Append new tokens to recent window
            if self._recent_k is not None and self._recent_k.shape[2] > 0:
                self._recent_k = torch.cat([self._recent_k, key_states], dim=2)
                self._recent_v = torch.cat([self._recent_v, value_states], dim=2)
            else:
                self._recent_k = key_states
                self._recent_v = value_states

            # If window exceeds size, absorb overflow into codebook
            if self._recent_k.shape[2] > self.window_size:
                overflow = self._recent_k.shape[2] - self.window_size
                old_k = self._recent_k[:, :, :overflow, :]
                old_v = self._recent_v[:, :, :overflow, :]
                self._absorb(old_k, old_v)
                self._recent_k = self._recent_k[:, :, overflow:, :]
                self._recent_v = self._recent_v[:, :, overflow:, :]

            # Periodic QARC resonance
            self._absorb_count += 1
            if self._absorb_count % self.resonate_every == 0:
                self._resonate()

            # Return [codebook | recent_window]
            all_k, all_v = self._build_full_kv()
            return all_k, all_v

    def _compress(self):
        """Split accumulated K/V into codebook (old) + recent window (new)."""
        B, H, T, D = self._keys.shape

        # Recent window = last window_size tokens (exact)
        self._recent_k = self._keys[:, :, -self.window_size:, :].clone()
        self._recent_v = self._values[:, :, -self.window_size:, :].clone()

        # Codebook = first budget tokens (compressed representation)
        old_k = self._keys[:, :, :self.budget, :]
        old_v = self._values[:, :, :self.budget, :]
        k_float = old_k[0].permute(1, 0, 2).reshape(self.budget, H * D).float()
        v_float = old_v[0].permute(1, 0, 2).reshape(self.budget, H * D).float()
        self._set_codebook(k_float, v_float)

        # If there are tokens between budget and window, absorb them
        middle_start = self.budget
        middle_end = T - self.window_size
        if middle_end > middle_start:
            mid_k = self._keys[:, :, middle_start:middle_end, :]
            mid_v = self._values[:, :, middle_start:middle_end, :]
            self._absorb(mid_k, mid_v)

        # Free concat buffer
        self._keys = None
        self._values = None
        self._compressed = True

    def _absorb(self, keys: torch.Tensor, values: torch.Tensor):
        """Absorb K/V into concept codebook via soft-attention write.

        For ternary mode: dequantize → absorb in float32 → re-quantize.
        """
        B, H, T, D = keys.shape
        k_flat = keys.transpose(1, 2).reshape(B * T, H * D).float()
        v_flat = values.transpose(1, 2).reshape(B * T, H * D).float()

        # Get current codebook as float32
        cb_k = self._get_codebook_float('k')
        cb_v = self._get_codebook_float('v')

        k_norm = F.normalize(k_flat, dim=-1)
        cb_norm = F.normalize(cb_k, dim=-1)
        scores = torch.matmul(k_norm, cb_norm.T) / self.write_temp
        attn = F.softmax(scores, dim=-1)

        k_update = torch.matmul(attn.T, k_flat) / max(B * T, 1)
        v_update = torch.matmul(attn.T, v_flat) / max(B * T, 1)
        cb_k = cb_k + self.write_alpha * k_update
        cb_v = cb_v + self.write_alpha * v_update

        # Store back (quantizes to ternary if enabled)
        self._set_codebook(cb_k, cb_v)

    def _resonate(self):
        """QARC: push correlated codebook slots apart."""
        if self.qarc_gamma <= 0:
            return
        cb_k = self._get_codebook_float('k')
        cb_v = self._get_codebook_float('v')
        if cb_k is None:
            return

        cb_norm = F.normalize(cb_k, dim=-1)
        sim = torch.matmul(cb_norm, cb_norm.T)
        sim.fill_diagonal_(0.0)
        cb_k = cb_k - self.qarc_gamma * torch.matmul(sim, cb_k)

        cb_norm_v = F.normalize(cb_v, dim=-1)
        sim_v = torch.matmul(cb_norm_v, cb_norm_v.T)
        sim_v.fill_diagonal_(0.0)
        cb_v = cb_v - self.qarc_gamma * torch.matmul(sim_v, cb_v)

        self._set_codebook(cb_k, cb_v)

    def reset(self):
        self._keys = None
        self._values = None
        self._codebook_k = None
        self._codebook_v = None
        self._ternary_k = None
        self._ternary_v = None
        self._recent_k = None
        self._recent_v = None
        self._needs_compress = False
        self._is_initialized = False
        self._compressed = False
        self._seq_length = 0
        self._absorb_count = 0
        self.tokens_absorbed = 0

    def reorder_cache(self, beam_idx):
        if self._keys is not None:
            self._keys = self._keys.index_select(0, beam_idx)
            self._values = self._values.index_select(0, beam_idx)
        if self._recent_k is not None:
            self._recent_k = self._recent_k.index_select(0, beam_idx)
            self._recent_v = self._recent_v.index_select(0, beam_idx)

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
        if self._recent_k is not None:
            self._recent_k = self._recent_k.repeat_interleave(repeats, dim=0)
            self._recent_v = self._recent_v.repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices):
        if self._keys is not None:
            self._keys = self._keys[indices]
            self._values = self._values[indices]
        if self._recent_k is not None:
            self._recent_k = self._recent_k[indices]
            self._recent_v = self._recent_v[indices]


class AlembicHFCache(DynamicCache):
    """AlembicKV: hybrid concept codebook + recent window.

    Below budget+window: identical to DynamicCache (lossless).
    Above: old tokens compressed into codebook, recent kept exact.
    Memory: O(budget + window_size) per layer — fixed.
    """

    def __init__(self, budget: int = 2048, window_size: int = 128,
                 write_alpha: float = 0.1, write_temp: float = 0.1,
                 qarc_gamma: float = 0.01, resonate_every: int = 64,
                 ternary: bool = False, ternary_sparsity: float = 0.5):
        super().__init__()
        self.budget = budget
        self.window_size = window_size
        self.write_alpha = write_alpha
        self.write_temp = write_temp
        self.qarc_gamma = qarc_gamma
        self.resonate_every = resonate_every
        self.use_ternary = ternary
        self.ternary_sparsity = ternary_sparsity
        self.layer_class_to_replicate = None

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               layer_idx: int, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        while len(self.layers) <= layer_idx:
            self.layers.append(AlembicLayer(
                budget=self.budget,
                window_size=self.window_size,
                write_alpha=self.write_alpha,
                write_temp=self.write_temp,
                qarc_gamma=self.qarc_gamma,
                resonate_every=self.resonate_every,
                ternary=self.use_ternary,
                ternary_sparsity=self.ternary_sparsity,
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
            'window_size': self.window_size,
            'tokens_absorbed': absorbed,
            'compressed': compressed,
            'mode': 'codebook+window' if compressed else 'fill',
            'memory_slots': self.budget + self.window_size if compressed else 'growing',
        }
