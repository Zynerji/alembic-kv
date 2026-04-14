"""Tests for AlembicKVCache — projection-free drop-in replacement."""

import torch
import pytest
from alembic_kv import AlembicKVCache, AlembicKVConfig


@pytest.fixture
def config():
    return AlembicKVConfig(num_layers=2, num_heads=4, head_dim=32, budget=16,
                           write_alpha=0.1)


@pytest.fixture
def cache(config):
    return AlembicKVCache(config)


class TestCacheInterface:

    def test_update_returns_kv(self, cache):
        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)
        out_k, out_v = cache.update(k, v, layer_idx=0)
        assert out_k.shape == (1, 4, 17, 32)  # 16 codebook + 1 raw
        assert out_v.shape == (1, 4, 17, 32)

    def test_tracks_tokens(self, cache):
        cache.update(torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32), layer_idx=0)
        assert cache.seen_tokens == 5

    def test_memory_constant(self, cache):
        mb = cache.memory_mb
        for _ in range(100):
            cache.update(torch.randn(1, 4, 1, 32), torch.randn(1, 4, 1, 32), layer_idx=0)
        assert cache.memory_mb == mb

    def test_reset(self, cache):
        cache.update(torch.randn(1, 4, 10, 32), torch.randn(1, 4, 10, 32), layer_idx=0)
        cache.reset()
        assert cache.seen_tokens == 0

    def test_len(self, cache, config):
        assert len(cache) == config.num_layers

    def test_getitem(self, cache):
        k, v = cache[0]
        assert k.shape == (1, 4, 16, 32)

    def test_stats(self, cache):
        cache.update(torch.randn(1, 4, 5, 32), torch.randn(1, 4, 5, 32), layer_idx=0)
        s = cache.stats()
        assert s['seen_tokens'] == 5
        assert s['memory_mb'] > 0

    def test_repr(self, cache):
        assert 'AlembicKVCache' in repr(cache)

    def test_multi_layer(self, cache):
        """Both layers get updates."""
        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)
        cache.update(k, v, layer_idx=0)
        cache.update(k, v, layer_idx=1)
        assert cache.chambers[0].tokens_absorbed == 1
        assert cache.chambers[1].tokens_absorbed == 1


class TestCompression:

    def test_compression_ratio(self):
        config = AlembicKVConfig(num_layers=32, num_heads=32, head_dim=128, budget=2048)
        cache = AlembicKVCache(config)
        cache._seen_tokens = 100000
        s = cache.stats()
        assert s['compression_ratio'] > 10

    def test_summary(self):
        config = AlembicKVConfig(num_layers=32, num_heads=32, head_dim=128, budget=2048)
        summary = config.summary()
        assert 'AlembicKV' in summary

    def test_70b_vram(self):
        """Verify the 70B model VRAM numbers from our discussion."""
        config = AlembicKVConfig(
            num_layers=80, num_heads=8, head_dim=128, budget=2048)
        # AlembicKV should be < 2 GB (K+V codebooks in float32)
        assert config.total_gb < 2.0
        # Standard KV at 100K should be >> AlembicKV
        std = config.standard_cache_gb(100000)
        assert std > 10  # ~32 GB
        # Compression > 20x
        assert config.compression_at(100000) > 20

    def test_4096_budget_vram(self):
        """4096 budget for 70B should be ~1.3 GB."""
        config = AlembicKVConfig(
            num_layers=80, num_heads=8, head_dim=128, budget=4096)
        assert config.total_gb < 3.0
