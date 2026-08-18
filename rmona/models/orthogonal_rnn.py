"""Orthogonal RNN models (for pMNIST / sMNIST experiments).

Two paradigms:
- :class:`OrthogonalRNN`: the recurrent weight ``W_hh in O(hidden)`` is a
  **hard Stiefel-constrained parameter**, optimized directly by a manifold
  optimizer.
- :class:`ParamRNN`: ``W_hh = exp(A-A^T)`` (expRNN) or Cayley parameterization
  (cayleyRNN); ``A`` is an unconstrained free parameter (optimized with AdamW).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..manifold import random_stiefel

__all__ = ["OrthogonalRNN", "ParamRNN"]


class OrthogonalRNN(nn.Module):
    """RNN whose recurrent weight W_hh is a Stiefel-constrained parameter (O(hidden))."""

    def __init__(self, input_size=784, hidden=256, num_classes=10,
                 init='identity'):
        super().__init__()
        self.hidden = hidden
        self.W_in = nn.Linear(input_size, hidden, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden))
        self.W_out = nn.Linear(hidden, num_classes, bias=False)
        # Recurrent weight (manifold parameter)
        if init == 'identity':
            Wh = torch.eye(hidden)
        else:
            Wh = random_stiefel(hidden, hidden)
        self.W_hh = nn.Parameter(Wh, requires_grad=True)

    def forward(self, x, W_hh):
        """x: (B, 784) -> logits (B, 10). W_hh: (hidden, hidden)."""
        Wm = self.W_in.weight.T              # (784, hidden) per-pixel encoding vectors
        xe = x.unsqueeze(2) * Wm.unsqueeze(0)   # (B, 784, hidden) pixel-wise independent encoding
        h = torch.zeros(x.size(0), self.hidden, device=x.device)
        for t in range(x.size(1)):
            h = torch.tanh(xe[:, t] + h @ W_hh + self.b_h)
        return self.W_out(h)


class ParamRNN(nn.Module):
    """Parameterized orthogonal RNN: W_hh = exp(A - A^T) (expRNN) or Cayley(A-A^T) (cayleyRNN)."""

    def __init__(self, hidden=256, num_classes=10, param='exp',
                 input_size=784):
        super().__init__()
        self.hidden = hidden
        self.param = param
        self.W_in = nn.Linear(input_size, hidden, bias=False)
        self.b_h = nn.Parameter(torch.zeros(hidden))
        self.W_out = nn.Linear(hidden, num_classes, bias=False)
        # A is initialized as a small random skew-symmetric matrix (avoids the
        # cayleyRNN "dead zone" at A=0, where dW/dA is degenerate and gradients
        # cannot propagate).
        A = torch.randn(hidden, hidden) * 0.1
        self.A = nn.Parameter(A - A.T)   # free parameter

    def _make_W(self):
        A = self.A - self.A.T                # skew-symmetric
        if self.param == 'exp':
            return torch.matrix_exp(A)
        else:  # cayley
            I = torch.eye(self.hidden, device=A.device, dtype=A.dtype)
            return torch.linalg.solve(I - 0.5 * A, I + 0.5 * A)

    def forward(self, x):
        W_hh = self._make_W()
        Wm = self.W_in.weight.T
        xe = x.unsqueeze(2) * Wm.unsqueeze(0)
        h = torch.zeros(x.size(0), self.hidden, device=x.device)
        for t in range(x.size(1)):
            h = torch.tanh(xe[:, t] + h @ W_hh + self.b_h)
        return self.W_out(h)
