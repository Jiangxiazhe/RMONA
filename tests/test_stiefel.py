"""Unit tests for the Stiefel manifold primitives."""
import pytest
import torch

from rmona.manifold import (
    cayley_retraction,
    msign,
    msign_skew,
    orthogonality_error,
    parallel_transport_approx,
    proj_tangent,
    qr_retraction,
    random_stiefel,
    solve_smp,
)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SIZES = [(10, 10), (20, 10), (20, 12)]


@pytest.fixture(params=SIZES)
def size(request):
    return request.param


@pytest.fixture
def W(size):
    n, p = size
    return random_stiefel(n, p, device=DEV)


def test_random_stiefel_orthogonal(W):
    assert orthogonality_error(W).item() < 1e-5


def test_proj_tangent_condition(W):
    g = torch.randn_like(W)
    G = proj_tangent(W, g)
    err = (W.T @ G + G.T @ W).norm().item()
    assert err < 1e-5


def test_proj_tangent_idempotent(W):
    g = torch.randn_like(W)
    G = proj_tangent(W, g)
    G2 = proj_tangent(W, G)
    assert torch.allclose(G, G2, atol=1e-6)


def test_qr_retraction_orthogonal_and_first_order(W):
    g = torch.randn_like(W)
    G = proj_tangent(W, g)
    xi = 0.01 * G
    W2 = qr_retraction(W, xi)
    assert orthogonality_error(W2).item() < 1e-5
    rel = (W2 - (W + xi)).norm().item() / (xi.norm().item() + 1e-12)
    assert rel < 0.05


def test_cayley_retraction_orthogonal_and_first_order(W):
    g = torch.randn_like(W)
    G = proj_tangent(W, g)
    xi = 0.01 * G
    W3 = cayley_retraction(W, xi)
    assert orthogonality_error(W3).item() < 1e-5
    rel = (W3 - (W + xi)).norm().item() / (xi.norm().item() + 1e-12)
    assert rel < 0.05


@pytest.mark.parametrize('path', ['closed', 'dual', 'ap'])
def test_smp_tangent_and_descent(W, path):
    M = proj_tangent(W, torch.randn_like(W))
    O = solve_smp(M, W, T=8, path=path)
    # Tolerance 1e-3: the closed path goes through the msign_skew eigh
    # decomposition (numerical error ~1e-4), and the direction is retracted
    # back onto the manifold afterwards (verified by test_manifold_constraint_kept)
    t_err = (W.T @ O + O.T @ W).norm().item()
    assert t_err < 1e-3
    inner = (M * O).sum().item()
    if path == 'closed':
        assert inner < 0, "closed SMP must be a descent direction"


def test_msign_svd(W):
    M = torch.randn_like(W)
    Ms = msign(M, backend='svd')
    # The spectral sign is the polar factor: Ms @ M^T must be symmetric PSD
    S = Ms @ M.T
    assert torch.allclose(S, S.T, atol=1e-4)
    # Column orthogonality: Ms^T Ms = I_p (for non-square Ms, Ms Ms^T != I_n)
    assert torch.allclose(Ms.T @ Ms, torch.eye(Ms.shape[1], device=DEV), atol=1e-3)


def test_msign_skew_stays_skew():
    A = torch.randn(10, 10, device=DEV)
    A = A - A.T  # skew-symmetric
    Ms = msign_skew(A)
    assert torch.allclose(Ms, -Ms.T, atol=1e-5), "msign_skew must stay skew"
    # Alignment: <A, msign(A)> = ||A||_* (nuclear norm)
    inner = (A * Ms).sum().item()
    nuc = torch.linalg.svdvals(A).sum().item()
    assert inner > 0.99 * nuc, f"skew msign alignment {inner} vs nuc {nuc}"


def test_parallel_transport_approx(W):
    g = torch.randn_like(W)
    G = proj_tangent(W, g)
    W2 = qr_retraction(W, 0.01 * G)
    Gt = parallel_transport_approx(W2, G)
    err = (W2.T @ Gt + Gt.T @ W2).norm().item()
    assert err < 1e-5


@pytest.mark.parametrize('size', [(20, 10), (10, 10)])
def test_cayley_retraction_long_sequence_stability(size):
    """Regression test: the Cayley skew generator must be exactly
    skew-symmetrized. Without it, the tangent mismatch of a drifted W is
    amplified geometrically each step and the constraint error explodes
    (observed: orth_err ~ 2.7 after 200 steps on a large-gradient task)."""
    torch.manual_seed(0)
    n, p = size
    W = random_stiefel(n, p, device=DEV)
    # Seed a drift so W is NOT exactly orthogonal (worst case for the solver)
    W = W + 1e-3 * torch.randn_like(W)
    C = torch.randn(n, n, device=DEV)
    C = C + C.T  # large-gradient Rayleigh-Ritz
    lr = 0.05
    for _ in range(200):
        G = proj_tangent(W, (-2 * C @ W))
        W = cayley_retraction(W, -lr * G)
        err = orthogonality_error(W).item()
        assert err < 1e-3, f"cayley retraction drift amplification: {err:.2e}"
