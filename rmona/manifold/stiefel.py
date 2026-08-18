"""Core Stiefel-manifold operations (RMONA package).

Implements all manifold primitives required by the RMONA design document
(docs/design.md §3):

- Tangent-space projection ``Proj_{T_W St}(g) = g - 1/2 W (W^T g + g^T W)``
- Spectral sign ``msign`` (SVD / Newton-Schulz backends, incl. skew-symmetric path)
- SMP direction solving (closed-form / Bernstein dual iteration / alternating
  projection)
- QR / Cayley retraction (square O(p) and general St(n,p))
- Reprojection approximation of parallel transport (Thm C' error bound O(eta·rho))
- Constraint-violation metric ``||W^T W - I||_F``

All implementations are pure PyTorch with no geoopt dependency.
"""
from __future__ import annotations

import torch

__all__ = [
    "proj_tangent",
    "parallel_transport_approx",
    "orthogonality_error",
    "random_stiefel",
    "msign",
    "msign_svd",
    "msign_newtonschulz",
    "msign_skew",
    "orth_complement",
    "solve_smp",
    "solve_smp_closed",
    "solve_smp_dual",
    "solve_smp_ap",
    "retract",
    "qr_retraction",
    "cayley_retraction",
]


# ---------------------------------------------------------------------------
# Basic manifold operations
# ---------------------------------------------------------------------------


def proj_tangent(W: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """Stiefel tangent-space projection: ``G - 1/2 W (W^T G + G^T W)``.

    Args:
        W: (n, p) point on the Stiefel manifold.
        G: (n, p) arbitrary matrix (usually the Euclidean gradient).

    Returns:
        (n, p) tangent vector (satisfies ``W^T X + X^T W = 0``).
    """
    return G - 0.5 * W @ (W.T @ G + G.T @ W)


def parallel_transport_approx(W_cur: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Reprojection approximation of parallel transport (RMONA default, zero extra
    cost, O(eta) error; see Thm C')."""
    return proj_tangent(W_cur, v)


def orthogonality_error(W: torch.Tensor) -> torch.Tensor:
    """Constraint violation: ``||W^T W - I_p||_F``."""
    return (
        W.T @ W
        - torch.eye(W.shape[1], device=W.device, dtype=W.dtype)
    ).norm()


def random_stiefel(
    n: int, p: int, dtype=torch.float32, device: str = "cuda"
) -> torch.Tensor:
    """Random Stiefel point (semi-orthogonal matrix via QR)."""
    A = torch.randn(n, p, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :p]


# ---------------------------------------------------------------------------
# Spectral sign msign (zeropower)
# ---------------------------------------------------------------------------


def msign_svd(M: torch.Tensor) -> torch.Tensor:
    """``msign(M) = U Vh`` where ``M = U S Vh`` (exact spectral sign via SVD).

    Note that ``torch.linalg.svd`` returns ``Vh`` as its third output (not ``V``),
    so ``U @ Vh`` is the correct composition.
    """
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh


def msign_newtonschulz(M: torch.Tensor, steps: int = 8) -> torch.Tensor:
    """Fifth-order Newton-Schulz iteration approximating zeropower (as in Muon;
    fast, bfloat16-friendly)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M
    X = X / (X.norm() + 1e-7)
    if X.size(0) > X.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if M.size(0) > M.size(1):
        X = X.T
    return X


def msign(M: torch.Tensor, backend: str = "svd", steps: int = 8) -> torch.Tensor:
    """Unified entry point for the spectral sign.

    Args:
        backend: ``"svd"`` for the exact spectral sign, or ``"newtonschulz"``
            for the fifth-order polynomial approximation.
        steps: number of Newton-Schulz iterations.
    """
    if backend == "svd":
        return msign_svd(M)
    elif backend == "newtonschulz":
        return msign_newtonschulz(M, steps=steps)
    else:
        raise ValueError(backend)


def msign_skew(A: torch.Tensor) -> torch.Tensor:
    """Spectral sign ``msign(A)`` of a skew-symmetric matrix (preserves skew-symmetry).

    For skew-symmetric ``A = Q diag(i sigma) Q^T`` (orthogonal diagonalization),

    .. math::
        msign(A) = A (A^T A)^{-1/2} = Q diag(i sign(sigma)) Q^T

    which is always skew-symmetric and satisfies ``<A, msign(A)> = ||A||_*``
    (nuclear norm).

    A naive SVD-based ``msign(A) = U V^T`` is not skew-symmetric (and can even be
    nearly orthogonal to ``A``) due to the sign freedom of paired singular values,
    so the ``eigh(-A^2)`` path is required.
    """
    B = -(A @ A)  # = A^T A, positive semi-definite and symmetric
    evals, Q = torch.linalg.eigh(B)
    inv_sqrt = torch.zeros_like(evals)
    mask = evals > 1e-8
    inv_sqrt[mask] = 1.0 / evals[mask].sqrt()
    return A @ (Q * inv_sqrt) @ Q.T


# ---------------------------------------------------------------------------
# SMP direction solving:  min_B <M, B>  s.t.  ||B||_2 <= 1,  B in T_W St(n,p)
# The result O satisfies <M, O> <= 0 (descent direction); O is the negative
# spectral sign with a tangent-space correction.
# ---------------------------------------------------------------------------


def _orth_complement_qr(W0: torch.Tensor) -> torch.Tensor:
    """Orthogonal complement of ``span(W0)`` via QR of the concatenated matrix
    ``[W0 | Z]`` with a random Z (shared numerical core).

    Why not project-then-QR (``qr(Z - W0 W0^T Z)``)? When a random column of Z
    lies nearly inside span(W0), the projected residual is near-singular and a
    column-pivot-free ``torch.linalg.qr`` breaks down numerically on it,
    reintroducing cross terms ||W0^T W_perp|| up to ~1e-2 in float32 (observed
    as transient constraint spikes in the cayley method). QR of the
    well-conditioned concatenation is backward stable: Householder never mixes
    the already-orthogonal leading columns, so ``Q[:, p:]`` spans the
    complement with O(eps) accuracy, independent of the random draw.
    """
    n, p = W0.shape
    Z = torch.randn(n, n - p, dtype=W0.dtype, device=W0.device)
    Q, R = torch.linalg.qr(torch.cat([W0, Z], dim=1))
    # Sign correction so Q[:, :p] keeps the column orientation of W0
    d = torch.diag(R)
    s = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    Q = Q * s.unsqueeze(0)
    return Q[:, p:]


def orth_complement(W: torch.Tensor):
    """Orthogonal complement ``W_perp in St(n, n-p)`` of a Stiefel point
    ``W in St(n,p)`` (when n > p)."""
    n, p = W.shape
    if n == p:
        return None
    return _orth_complement_qr(W)


def solve_smp_closed(M: torch.Tensor, W: torch.Tensor, backend: str = "svd") -> torch.Tensor:
    """Skewon SMP exact closed-form solution (block spectral-sign alignment).

    Let ``[W, W_perp] in O(n)`` and write the tangent vector as
    ``M = W A + W_perp C`` with ``A = W^T M`` skew-symmetric and
    ``C = W_perp^T M``. Then

    .. math::
        B^* = -(W msign(A) + W_perp msign(C))

    automatically satisfies the tangent-space constraint
    ``W^T B^* + B^{*T} W = -msign(A) + msign(A) = 0`` and gives
    ``<M, B*> = -(||A||_* + ||C||_*) < 0`` (strong descent). Finally the
    direction is spectrally compressed to ``||B*||_2 <= 1``.

    The square case (n = p) reduces to ``B* = -msign(M)``, consistent with
    Muon's NS orthogonalization (shen2025: square Muon = Stiefel natural
    gradient).
    """
    M = proj_tangent(W, M)  # numerical safety
    A = W.T @ M  # p x p skew-symmetric
    B = W @ msign_skew(A)  # skew-symmetric msign (keeps tangent space)
    if W.shape[0] > W.shape[1]:
        Wp = orth_complement(W)
        C = Wp.T @ M  # (n-p) x p (general matrix, plain msign)
        B = B + Wp @ msign(C, backend=backend)
    B = -B  # min<M,B> -> negative direction
    # Spectral-norm compression
    _, S, _ = torch.linalg.svd(B, full_matrices=False)
    s = S[0] if S.numel() > 0 else torch.tensor(0.0, device=B.device)
    if s > 1.0:
        B = B / s
    # Final tangent-space projection: consistent with the dual/ap paths, removes
    # the numerical error of the msign_skew eigendecomposition (~1e-4~1e-2),
    # guaranteeing the returned direction strictly satisfies W^T B + B^T W = 0.
    return proj_tangent(W, B)


def solve_smp_dual(
    M: torch.Tensor, W: torch.Tensor, T: int = 8, backend: str = "svd", rho: float = 0.5
) -> torch.Tensor:
    """Bernstein dual iteration path (design doc §3.4):

    .. math::
        A(Lambda) = -msign(M + 2W(Lambda+Lambda^T))
        Lambda <- Lambda + rho * (W^T A + A^T W)   # dual ascent (Uzawa)
    """
    p = W.shape[1]
    Lambda = torch.zeros(p, p, dtype=W.dtype, device=W.device)
    A = None
    for _ in range(T):
        X = M + 2.0 * W @ (Lambda + Lambda.T)
        A = -msign(X, backend=backend)
        r = W.T @ A + A.T @ W
        Lambda = Lambda + rho * r
    return proj_tangent(W, A)


def solve_smp_ap(
    M: torch.Tensor, W: torch.Tensor, T: int = 8, backend: str = "svd"
) -> torch.Tensor:
    """Alternating-projection path: alternate between the spectral-norm ball
    (spectral-sign projection) and the tangent-space projection."""
    A = -msign(M, backend=backend)
    for _ in range(T):
        A = proj_tangent(W, A)
        A = -msign(A, backend=backend)
    return proj_tangent(W, A)


def solve_smp(
    M: torch.Tensor,
    W: torch.Tensor,
    T: int = 8,
    path: str = "closed",
    backend: str = "svd",
) -> torch.Tensor:
    """Unified entry point for SMP solving. Returns a descent direction O in the
    tangent space (``<M,O> <= 0``).

    Args:
        path: ``"closed"`` exact closed form / ``"dual"`` Bernstein dual
            iteration / ``"ap"`` alternating projection.
    """
    if path == "closed":
        return solve_smp_closed(M, W, backend=backend)
    elif path == "dual":
        return solve_smp_dual(M, W, T=T, backend=backend)
    elif path == "ap":
        return solve_smp_ap(M, W, T=T, backend=backend)
    else:
        raise ValueError(path)


# ---------------------------------------------------------------------------
# Retraction back to the manifold
# ---------------------------------------------------------------------------


def qr_retraction(W: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """QR retraction: ``qf(W + xi)`` with an R-diagonal sign correction for
    first-order compatibility."""
    Q, R = torch.linalg.qr(W + xi)
    # Sign correction: make R's diagonal positive to avoid sign flips of Q's
    # columns (first-order compatibility).
    d = torch.diag(R)
    s = torch.where(d < 0, -torch.ones_like(d), torch.ones_like(d))
    Q = Q * s.unsqueeze(0)
    return Q


def cayley_retraction(W: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    """Cayley retraction.

    Square case (``W in O(p)``): ``R = W (I - A/2)^{-1} (I + A/2)`` with
    ``A = W^T xi`` skew-symmetric.
    Non-square case (``W in St(n,p), n>p``): complete ``[W, W_perp] in O(n)``
    via QR, build an n x n skew-symmetric extension, apply Cayley, and keep the
    first p columns.

    Robustness: W is first exactly orthogonalized with QR (to prevent error
    amplification in the non-square complement construction when W drifts from
    orthogonality); the skew generator A is re-symmetrized (``A <- (A-A^T)/2``)
    so the Cayley transform is orthogonal to machine precision regardless of
    accumulated drift (without this, each step's tangent mismatch is amplified
    by ``||xi||`` and the constraint error grows geometrically); falls back to
    the QR retraction if the solve fails.
    """
    n, p = W.shape
    if torch.isnan(xi).any():
        return qr_retraction(W, torch.zeros_like(xi))
    W0 = qr_retraction(W, torch.zeros_like(W))  # qf(W): exact orthogonalization
    I_p = torch.eye(p, dtype=W.dtype, device=W.device)
    if n == p:
        A = W0.T @ xi  # p x p, skew-symmetric up to the drift of W
        A = 0.5 * (A - A.T)  # exact skew-symmetrization (numerical safety)
        try:
            inner = torch.linalg.solve(I_p - 0.5 * A, I_p + 0.5 * A)
        except Exception:
            return qr_retraction(W, xi)
        return W0 @ inner
    else:
        # Complete the orthogonal basis via QR of the concatenated matrix
        # (backward-stable; see _orth_complement_qr for why project-then-QR
        # is numerically unsafe here)
        W_perp = _orth_complement_qr(W0)
        Wbar = torch.cat([W0, W_perp], dim=1)  # (n, n) orthogonal
        A1 = W0.T @ xi  # (p, p), skew-symmetric up to the drift of W
        A1 = 0.5 * (A1 - A1.T)  # exact skew-symmetrization (numerical safety)
        A2 = W_perp.T @ xi  # (n-p, p)
        # Skew-symmetric extension
        Aext = torch.zeros(n, n, dtype=W.dtype, device=W.device)
        Aext[:p, :p] = A1
        Aext[p:, :p] = A2
        Aext[:p, p:] = -A2.T
        I_n = torch.eye(n, dtype=W.dtype, device=W.device)
        try:
            inner = torch.linalg.solve(I_n - 0.5 * Aext, I_n + 0.5 * Aext)
        except Exception:
            return qr_retraction(W, xi)
        return (Wbar @ inner)[:, :p]


def retract(W: torch.Tensor, xi: torch.Tensor, kind: str = "qr") -> torch.Tensor:
    """Unified retraction entry point."""
    if kind == "qr":
        return qr_retraction(W, xi)
    elif kind == "cayley":
        return cayley_retraction(W, xi)
    else:
        raise ValueError(kind)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def self_test() -> None:
    """Self-check of manifold primitives: tangent condition, first-order
    retraction compatibility, SMP descent direction, parallel transport."""
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (n, p) in [(10, 10), (20, 10)]:
        W = random_stiefel(n, p, device=dev)
        g = torch.randn(n, p, device=dev)
        G = proj_tangent(W, g)
        # Tangent condition
        err = (W.T @ G + G.T @ W).norm().item()
        assert err < 1e-5, f"tangent cond fail ({n},{p}): {err}"
        # QR retraction preserves orthogonality and is first-order compatible (rel. err < 5%)
        xi = 0.01 * G
        W2 = qr_retraction(W, xi)
        assert orthogonality_error(W2).item() < 1e-5, "qr retract not orth"
        rel = (W2 - (W + xi)).norm().item() / (xi.norm().item() + 1e-12)
        assert rel < 0.05, f"qr not 1st order: rel={rel}"
        # Cayley retraction
        W3 = cayley_retraction(W, xi)
        assert orthogonality_error(W3).item() < 1e-5, "cayley retract not orth"
        rel = (W3 - (W + xi)).norm().item() / (xi.norm().item() + 1e-12)
        assert rel < 0.05, f"cayley not 1st order: rel={rel}"
        # SMP: direction must be tangent and satisfy <M, O> < 0 (strict for closed;
        # tangent legality for the other paths)
        M = proj_tangent(W, torch.randn(n, p, device=dev))
        for path in ["closed", "dual", "ap"]:
            O = solve_smp(M, W, T=8, path=path)
            t_err = (W.T @ O + O.T @ W).norm().item()
            inner = (M * O).sum().item()
            assert t_err < 1e-4, f"smp tangent fail {path}: {t_err}"
            if path == "closed":
                assert inner < 0, f"smp not descent {path}: {inner}"
            print(f"  smp[{path}] ({n},{p}) inner={inner:.4f} t_err={t_err:.1e}")
        # Parallel-transport approximation
        W4 = qr_retraction(W, xi)
        Gt = parallel_transport_approx(W4, G)
        assert (W4.T @ Gt + Gt.T @ W4).norm().item() < 1e-5
    print("stiefel self_test passed")


if __name__ == "__main__":
    self_test()
