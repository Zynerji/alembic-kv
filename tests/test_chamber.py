"""Tests for projection-free ConceptChamber."""

import torch
import torch.nn.functional as F
import pytest
from alembic_kv import ConceptChamber


@pytest.fixture
def chamber():
    return ConceptChamber(kv_dim=128, codebook_size=32,
                          num_kv_heads=4, head_dim=32,
                          qarc_gamma=0.03, qarc_iterations=2,
                          write_alpha=0.1, write_temp=0.1)


class TestAbsorb:

    def test_absorb_single(self, chamber):
        k = torch.randn(1, 1, 128)
        v = torch.randn(1, 1, 128)
        chamber.absorb(k, v)
        assert chamber.tokens_absorbed == 1

    def test_absorb_sequence(self, chamber):
        chamber.absorb(torch.randn(1, 64, 128), torch.randn(1, 64, 128))
        assert chamber.tokens_absorbed == 64

    def test_absorb_changes_codebook(self, chamber):
        before_k = chamber.codebook_k.clone()
        chamber.absorb(torch.randn(1, 10, 128), torch.randn(1, 10, 128))
        assert not torch.allclose(before_k, chamber.codebook_k, atol=1e-7)

    def test_absorb_10k_tokens_stays_finite(self, chamber):
        for _ in range(100):
            chamber.absorb(torch.randn(1, 100, 128), torch.randn(1, 100, 128))
        assert chamber.tokens_absorbed == 10000
        assert chamber.codebook_k.shape == (32, 128)
        assert torch.isfinite(chamber.codebook_k).all()
        assert torch.isfinite(chamber.codebook_v).all()

    def test_absorb_4d(self, chamber):
        chamber.absorb(torch.randn(1, 10, 4, 32), torch.randn(1, 10, 4, 32))
        assert chamber.tokens_absorbed == 10

    def test_absorb_batch(self, chamber):
        chamber.absorb(torch.randn(4, 16, 128), torch.randn(4, 16, 128))
        assert chamber.tokens_absorbed == 64


class TestRetrieve:

    def test_retrieve_shape(self, chamber):
        query = torch.randn(1, 1, 128)
        keys, values = chamber.retrieve(query)
        assert keys.shape == (1, 32, 4, 32)
        assert values.shape == (1, 32, 4, 32)

    def test_retrieve_finite(self, chamber):
        chamber.absorb(torch.randn(1, 50, 128), torch.randn(1, 50, 128))
        keys, values = chamber.retrieve(torch.randn(1, 1, 128))
        assert torch.isfinite(keys).all()
        assert torch.isfinite(values).all()


class TestNeedleRetrieval:
    """Verify that absorbed information is actually retrievable."""

    def test_needle_in_haystack(self):
        ch = ConceptChamber(kv_dim=128, codebook_size=64,
                            num_kv_heads=4, head_dim=32,
                            write_alpha=0.1, write_temp=0.1)
        # Absorb random haystack
        for _ in range(10):
            ch.absorb(torch.randn(1, 100, 128) * 0.5,
                      torch.randn(1, 100, 128) * 0.5)
        # Absorb distinctive needle
        needle = torch.ones(1, 1, 128) * 3.0
        ch.absorb(needle, needle)

        # Check codebook has needle signal
        needle_flat = F.normalize(needle.squeeze().unsqueeze(0), dim=-1)
        cb_norm = F.normalize(ch.codebook_k, dim=-1)
        cos_sims = torch.matmul(needle_flat, cb_norm.T).squeeze()

        # Best slot should have high correlation with needle
        assert cos_sims.max().item() > 0.3

    def test_different_needles_hit_different_slots(self):
        ch = ConceptChamber(kv_dim=128, codebook_size=64,
                            num_kv_heads=4, head_dim=32,
                            write_alpha=0.1, write_temp=0.1)
        # Two orthogonal needles
        n1 = torch.zeros(1, 1, 128)
        n1[0, 0, :64] = 3.0
        n2 = torch.zeros(1, 1, 128)
        n2[0, 0, 64:] = 3.0

        ch.absorb(n1, n1)
        ch.absorb(n2, n2)

        # Each should activate a different slot
        cb_norm = F.normalize(ch.codebook_k, dim=-1)
        n1_scores = torch.matmul(F.normalize(n1.squeeze().unsqueeze(0), dim=-1), cb_norm.T)
        n2_scores = torch.matmul(F.normalize(n2.squeeze().unsqueeze(0), dim=-1), cb_norm.T)

        best_slot_1 = n1_scores.argmax().item()
        best_slot_2 = n2_scores.argmax().item()
        # Different needles should land in different slots
        assert best_slot_1 != best_slot_2


class TestDiversity:
    """Verify QARC prevents codebook collapse."""

    def test_initial_diversity_high(self, chamber):
        div = chamber.diversity()
        assert div > 0.9  # unit-normalized codebook should be very diverse

    def test_diversity_after_absorb(self, chamber):
        div_before = chamber.diversity()
        for _ in range(50):
            chamber.absorb(torch.randn(1, 100, 128), torch.randn(1, 100, 128))
            chamber.resonate()
        div_after = chamber.diversity()
        # Diversity should stay high (QARC prevents collapse)
        assert div_after > 0.8

    def test_collapse_without_qarc(self):
        """Without QARC, repeated writes of identical data cause convergence."""
        ch_no_qarc = ConceptChamber(
            kv_dim=128, codebook_size=32,
            num_kv_heads=4, head_dim=32,
            qarc_gamma=0.0, qarc_iterations=0,
            write_alpha=1.0, write_temp=0.05)  # aggressive write

        # Write the SAME single vector many times — forces convergence
        data = torch.randn(1, 1, 128) * 10.0
        for _ in range(500):
            ch_no_qarc.absorb(data, data)

        div = ch_no_qarc.diversity()
        # Without QARC + aggressive identical writes, diversity must drop
        assert div < 0.97


class TestResonance:

    def test_resonate_changes_codebook(self, chamber):
        chamber.absorb(torch.randn(1, 50, 128), torch.randn(1, 50, 128))
        before = chamber.codebook_k.clone()
        chamber.resonate()
        assert not torch.allclose(before, chamber.codebook_k, atol=1e-7)

    def test_resonate_preserves_shape(self, chamber):
        chamber.absorb(torch.randn(1, 50, 128), torch.randn(1, 50, 128))
        chamber.resonate()
        assert chamber.codebook_k.shape == (32, 128)

    def test_resonate_100x_stays_finite(self, chamber):
        chamber.absorb(torch.randn(1, 50, 128), torch.randn(1, 50, 128))
        for _ in range(100):
            chamber.resonate()
        assert torch.isfinite(chamber.codebook_k).all()


class TestReset:

    def test_reset_clears_count(self, chamber):
        chamber.absorb(torch.randn(1, 100, 128), torch.randn(1, 100, 128))
        chamber.reset()
        assert chamber.tokens_absorbed == 0

    def test_reset_restores_diversity(self, chamber):
        for _ in range(100):
            chamber.absorb(torch.randn(1, 100, 128), torch.randn(1, 100, 128))
        chamber.reset()
        assert chamber.diversity() > 0.9


class TestMemory:

    def test_memory_constant_over_tokens(self, chamber):
        mb_before = chamber.memory_mb
        for _ in range(100):
            chamber.absorb(torch.randn(1, 100, 128), torch.randn(1, 100, 128))
        assert chamber.memory_mb == mb_before
