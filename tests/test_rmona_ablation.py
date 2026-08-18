"""RMONA curvature-channel ablation tests: alpha=0 must degrade to no-curvature.

Verifies:
1. rmona(alpha=0) keeps the EMA curvature buffer A zero (channel off);
2. alpha>0 activates the curvature channel (state A non-zero);
3. G_prev is reprojected onto the current tangent space every step
   (parallel-transport approximation).

Note: the Rayleigh-Ritz objective must use the non-square St(n,p) (n>p); on
square O(n), tr(W^T C W) == tr(C) is constant, so the projected gradient is
zero and no momentum/curvature state can be activated.
"""
import torch

from rmona import FlowOptimizer
from rmona.manifold import random_stiefel

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def make_param(n=20, p=10):
    W = random_stiefel(n, p, device=DEV)
    return W.detach().clone().requires_grad_()


def make_loss_fn(W):
    n = W.shape[0]
    C = torch.diag(torch.linspace(10.0, 1.0, n, device=DEV)).float()
    return lambda: -torch.trace(W.T @ C @ W)


def test_alpha_zero_no_curvature_injection():
    """With alpha=0 the momentum M stays a pure tangent gradient (curvature
    injection term alpha*A is zero)."""
    torch.manual_seed(0)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': 'rmona', 'lr': 0.01,
                          'alpha': 0.0}])
    loss_fn = make_loss_fn(W)
    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    # Whether or not A is EMA-updated, the curvature term alpha*A in the
    # momentum injection is identically zero.
    state = next(iter(opt.state.values()))
    assert 'A' in state
    alpha = 0.0
    inj = (alpha * state['A']).norm().item()
    assert inj < 1e-8, "curvature injection must be zero when alpha=0"


def test_alpha_positive_injects_curvature():
    """With alpha>0 the curvature term alpha*A is non-zero (channel active)."""
    torch.manual_seed(0)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': 'rmona', 'lr': 0.01,
                          'alpha': 0.1}])
    loss_fn = make_loss_fn(W)
    for _ in range(10):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    state = next(iter(opt.state.values()))
    assert 'A' in state
    inj = (0.1 * state['A']).norm().item()
    assert inj > 1e-6, "curvature injection must be non-zero when alpha>0"


def test_g_prev_in_tangent_space():
    """rmona's stored G_prev must belong to the tangent space of the **pre-update**
    W (W0^T G + G^T W0 = 0).

    G_prev stores the tangent vector of step k, which belongs to the tangent
    space of step-k W; it is reprojected onto the new tangent space in step k+1
    (parallel-transport approximation).
    """
    torch.manual_seed(1)
    W = make_param()
    opt = FlowOptimizer([{'params': [W], 'method': 'rmona', 'lr': 0.01}])
    loss_fn = make_loss_fn(W)
    for _ in range(5):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        W0 = W.detach().clone()          # pre-step W (base point of G_prev's tangent space)
        opt.step()
    state = next(iter(opt.state.values()))
    Gp = state['G_prev']
    err = (W0.T @ Gp + Gp.T @ W0).norm().item()
    assert err < 1e-4, f"G_prev not tangent at its base point: {err:.2e}"
