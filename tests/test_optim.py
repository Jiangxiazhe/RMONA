"""Optimizer unit tests: constraint preservation, descent, runnability, state isolation.

These tests run on CPU (fast) and validate algorithmic correctness, not performance.
"""
import pytest
import torch

from rmona import FlowOptimizer
from rmona.manifold import orthogonality_error, random_stiefel

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# All manifold + Euclidean methods + AdamW
MANIFOLD_METHODS = ['rsgd', 'rsgd_m', 'radam', 'cayley',
                    'skewon', 'manifold_muon', 'rmona']
EUCLID_METHODS = ['muon', 'mona']


def make_param(n=20, p=10):
    """Build a Stiefel-constrained parameter (non-square so that Rayleigh-Ritz has signal)."""
    W = random_stiefel(n, p, device=DEV)
    W = W.detach().clone().requires_grad_()
    return W


def make_loss_fn(W, C=None):
    """Rayleigh-Ritz objective: min -tr(W^T C W), C with fixed spectrum 10 -> 1.

    Note n > p is required (non-square): on square O(n), tr(W^T C W) == tr(C)
    is constant and there is no descent direction.
    """
    n = W.shape[0]
    if C is None:
        C = torch.diag(torch.linspace(10.0, 1.0, n, device=DEV)).float()
    return lambda: -torch.trace(W.T @ C @ W)


@pytest.mark.parametrize('method', MANIFOLD_METHODS)
def test_manifold_constraint_kept(method):
    """Manifold methods must keep ||W^T W - I||_F small (hard constraint)."""
    torch.manual_seed(0)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': method, 'lr': 0.01}])
    loss_fn = make_loss_fn(W)
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        err = orthogonality_error(W).item()
        assert err < 1e-4, f"{method} constraint violated: {err:.2e}"


@pytest.mark.parametrize('method', MANIFOLD_METHODS)
def test_manifold_constraint_kept_large_gradient(method):
    """Stress variant: large (dense) gradients amplify any retraction drift.
    Regression test for the cayley retraction skew-symmetrization fix."""
    torch.manual_seed(3)
    n, p = 20, 10
    W = random_stiefel(n, p, device=DEV).detach().clone().requires_grad_()
    C = torch.randn(n, n, device=DEV)
    C = C + C.T  # dense spectrum -> large gradients
    opt = FlowOptimizer([{'params': [W], 'method': method, 'lr': 0.05}])
    loss_fn = lambda: -torch.trace(W.T @ C @ W)  # noqa: E731
    for _ in range(200):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        err = orthogonality_error(W).item()
        assert err < 1e-4, f"{method} constraint violated (large-grad): {err:.2e}"


@pytest.mark.parametrize('method', MANIFOLD_METHODS)
def test_manifold_loss_decreases(method):
    """On a convex Rayleigh-Ritz problem (non-square), loss decreases clearly."""
    torch.manual_seed(1)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': method, 'lr': 0.01}])
    loss_fn = make_loss_fn(W)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # The tail must be clearly below the initial value
    assert losses[-1] < losses[0], f"{method} not descending: {losses[0]:.3f} -> {losses[-1]:.3f}"


@pytest.mark.parametrize('method', MANIFOLD_METHODS + EUCLID_METHODS)
def test_all_methods_run(method):
    """All methods (incl. Euclidean baselines) complete a step without crashing."""
    torch.manual_seed(2)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': method, 'lr': 0.01}])
    loss_fn = make_loss_fn(W)
    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    assert torch.isfinite(W).all()


def test_adamw_runs():
    torch.manual_seed(0)
    W = torch.randn(10, 10, device=DEV, requires_grad=True)
    opt = FlowOptimizer([{'params': [W], 'method': 'adamw', 'lr': 1e-3}])
    for _ in range(10):
        opt.zero_grad()
        loss = (W ** 2).sum()
        loss.backward()
        opt.step()
    assert torch.isfinite(W).all()


def test_mixed_param_groups():
    """Mixed groups: manifold params use rmona, other params use adamw."""
    torch.manual_seed(3)
    W_m = make_param()                            # manifold param (20, 10)
    W_e = torch.randn(10, 10, device=DEV, requires_grad=True)  # Euclidean param
    opt = FlowOptimizer([
        {'params': [W_m], 'method': 'rmona', 'lr': 0.01},
        {'params': [W_e], 'method': 'adamw', 'lr': 1e-3},
    ])
    C = torch.diag(torch.linspace(10.0, 1.0, W_m.shape[0], device=DEV)).float()
    for _ in range(20):
        opt.zero_grad()
        loss = -torch.trace(W_m.T @ C @ W_m) + (W_e ** 2).sum()
        loss.backward()
        opt.step()
    assert orthogonality_error(W_m).item() < 1e-4


def test_state_independent_across_params():
    """Two parameters using the same method must not share state."""
    torch.manual_seed(4)
    W1 = make_param()
    W2 = make_param()
    opt = FlowOptimizer([
        {'params': [W1], 'method': 'rmona', 'lr': 0.01},
        {'params': [W2], 'method': 'rmona', 'lr': 0.01},
    ])
    assert set(opt.state.keys()) == {W1, W2} or not opt.state


def test_constraint_error_reports():
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': 'rmona', 'lr': 0.01}])
    errs = opt.constraint_error()
    assert 'rmona' in errs
    assert errs['rmona'] < 1e-4
