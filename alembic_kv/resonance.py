"""QARC Resonance — antiresonant dynamics for the concept chamber.

The core problem with fixed-size KV buffers: they degenerate. After many writes,
all slots converge to similar values (mode collapse). The buffer becomes a
mean-field summary that loses the diversity needed for precise attention retrieval.

QARC prevents this through quasiperiodic dynamics:
  c[i] = c[i] + gamma * tanh(W @ c[i]) * g_damp * R(t)

where R(t) is a ratchet factor driven by coupled pendulums at incommensurate
metallic-mean frequencies (phi^-1 and bronze^-1). Because these frequencies
are irrational and algebraically independent, the dynamics never repeat —
preventing the fixed points that cause mode collapse.

The antiresonant initialization ensures wave modes start orthogonal with
phases that cancel to zero-sum, so the initial buffer state has maximum
diversity.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import PHI, BRONZE


class AntiResonantInit:
    """Initialize buffer slots with antiresonant phase structure.

    Standard initialization (random, zeros) creates correlated slots that
    quickly collapse under repeated updates. Antiresonant init spaces slots
    at phases k*2pi/N, which sum to zero and resist correlation buildup.
    """

    @staticmethod
    def init_buffer(budget: int, dim: int, device: torch.device,
                    dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Create antiresonant buffer of shape (budget, dim).

        Each slot k has phase offset k*2pi/budget applied to a random
        orthogonal basis, ensuring maximum initial diversity.
        """
        # Start with orthogonal directions (as many as we can, then cycle)
        n_ortho = min(budget, dim)
        basis = torch.randn(n_ortho, dim, device=device, dtype=torch.float32)
        basis = torch.linalg.qr(basis.T).Q.T  # orthonormal rows

        buf = torch.zeros(budget, dim, device=device, dtype=torch.float32)
        for k in range(budget):
            # Cycle through orthogonal basis
            base = basis[k % n_ortho]
            # Phase rotation: scale by antiresonant amplitude
            phase = k * 2 * math.pi / budget
            amplitude = 0.02 / math.sqrt(max(budget, 1))
            buf[k] = base * amplitude * (1.0 + 0.1 * math.cos(phase))

        return buf.to(dtype)


class QARCResonator(nn.Module):
    """Quasicrystal Autopoietic Resonance Cascade.

    Applies antiresonant dynamics to a buffer to prevent mode collapse.
    Each resonance step pushes correlated slots apart while preserving
    the information content.

    The coupled pendulum ratchet R(t) ensures the dynamics are quasiperiodic:
    - theta_C oscillates at bronze^-1 frequency (consequentialist)
    - theta_D oscillates at phi^-1 frequency (deontological)
    - Their phase difference drives chiral bias: R(t) = 1 + kappa * sin(theta_C - theta_D)

    Because bronze^-1 and phi^-1 are incommensurate, R(t) never repeats,
    preventing periodic attractors.
    """

    def __init__(self, dim: int, gamma: float = 0.03,
                 iterations: int = 3, kappa: float = 0.1):
        super().__init__()
        self.dim = dim
        self.iterations = iterations

        # Resonance feedback: tanh(W @ c) * g_damp
        self.W_feedback = nn.Linear(dim, dim, bias=False)
        nn.init.orthogonal_(self.W_feedback.weight, gain=0.1)
        self.g_damp = nn.Parameter(torch.ones(dim) * 0.9)
        self.gamma = nn.Parameter(torch.tensor(gamma))

        # Pendulum coupling
        self.kappa = nn.Parameter(torch.tensor(kappa))

        # Pendulum state (not parameters — evolve dynamically)
        self.register_buffer('theta_C', torch.zeros(1))
        self.register_buffer('theta_D', torch.zeros(1))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        # Metallic mean frequencies
        self.phi_inv = 2.0 / (1.0 + math.sqrt(5))
        self.bronze_inv = 2.0 / (3.0 + math.sqrt(13))

    def _advance_pendulums(self) -> torch.Tensor:
        """Advance coupled pendulums and return ratchet factor R(t)."""
        self.step_count += 1
        t = self.step_count.float()

        # Incommensurate angular velocities
        self.theta_C = (self.theta_C + self.bronze_inv) % (2 * math.pi)
        self.theta_D = (self.theta_D + self.phi_inv) % (2 * math.pi)

        # Ratchet: chiral bias from phase difference
        ratchet = 1.0 + self.kappa * torch.sin(self.theta_C - self.theta_D)
        return ratchet

    def resonate(self, buffer: torch.Tensor,
                 mask: torch.Tensor = None) -> torch.Tensor:
        """Apply QARC resonance to buffer slots.

        Args:
            buffer: (budget, dim) or (num_heads, budget, head_dim)
            mask: (budget,) bool — which slots are occupied (skip empty)

        Returns:
            Resonated buffer, same shape.
        """
        orig_shape = buffer.shape
        if buffer.dim() == 3:
            # (num_heads, budget, head_dim) → flatten heads into dim
            nh, b, hd = buffer.shape
            buf = buffer.reshape(b, nh * hd)
        else:
            buf = buffer

        for _ in range(self.iterations):
            ratchet = self._advance_pendulums()

            # Feedback: phi(c) = tanh(W @ c) * g_damp
            phi_c = torch.tanh(self.W_feedback(buf.float())) * self.g_damp
            delta = self.gamma * phi_c * ratchet

            if mask is not None:
                # Only resonate occupied slots
                delta = delta * mask.unsqueeze(-1).float()

            buf = buf + delta.to(buf.dtype)

        if buffer.dim() == 3:
            return buf.reshape(orig_shape)
        return buf

    def reset(self):
        """Reset pendulums (call between independent sequences)."""
        self.theta_C.zero_()
        self.theta_D.zero_()
        self.step_count.zero_()
