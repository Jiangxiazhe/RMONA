"""RMONA: curvature-aware accelerated optimizer on the Stiefel manifold.

Riemannian-MONA (RMONA) targets orthogonal-constrained optimization (orthogonal
RNNs, orthogonal convolutions, QK-regularized Transformers, metric learning)
and combines:

- **Exact SMP direction solving** (Skewon closed-form / Bernstein dual /
  alternating projections)
- **EMA gradient-difference curvature channel** (MONA-style, transported via
  the reprojection approximation)
- **Reprojection momentum transport** (zero-extra-cost tangent transport)

Public API: ``FlowOptimizer`` (unified 11 manifold/Euclidean methods).
"""
from .manifold import (
    cayley_retraction,
    msign,
    msign_newtonschulz,
    msign_skew,
    msign_svd,
    orthogonality_error,
    parallel_transport_approx,
    proj_tangent,
    qr_retraction,
    random_stiefel,
    retract,
    solve_smp,
)
from .optim import FlowOptimizer, Rmona

__version__ = "0.1.0"

__all__ = [
    "FlowOptimizer",
    "Rmona",
    "proj_tangent",
    "parallel_transport_approx",
    "orthogonality_error",
    "random_stiefel",
    "msign",
    "msign_svd",
    "msign_newtonschulz",
    "msign_skew",
    "solve_smp",
    "retract",
    "qr_retraction",
    "cayley_retraction",
]
