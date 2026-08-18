"""Unified optimizer module: RMONA and all comparison baselines.

Supported ``method`` values (per parameter group):

Manifold methods (for 2D parameters with Stiefel constraints):
  - rsgd           : Riemannian SGD (no momentum)
  - rsgd_m         : Riemannian SGD + reprojection momentum
  - radam          : Riemannian Adam (tangent-space EMA + reprojection)
  - cayley         : Cayley SGD (tangent SGD + Cayley retraction)
  - skewon         : SMP direction + reprojection momentum (alternating-projection
                     solver, no curvature channel)
  - manifold_muon  : SMP direction + reprojection momentum (Bernstein dual solver,
                     no curvature channel)
  - rmona          : the proposed algorithm (SMP + EMA gradient-difference
                     curvature channel + reprojection parallel transport)

Euclidean methods (2D parameters, unconstrained baselines):
  - muon           : standard Muon (SGD momentum + NS orthogonalization)
  - mona           : MONA (Muon + EMA gradient-difference curvature channel)

General Euclidean methods:
  - adamw          : AdamW (any dimensionality, for non-manifold parameters)

The state of all manifold methods (momentum M / acceleration buffer A / previous
tangent vector G_prev / Adam moments m, v) is reprojected onto the current tangent
space every step, keeping tangent vectors valid (docs/design.md §3.6).
"""
from __future__ import annotations

import torch

from .manifold import msign, orthogonality_error, proj_tangent, retract, solve_smp

__all__ = ["FlowOptimizer", "Rmona"]


class FlowOptimizer:
    """Unified manifold/Euclidean optimizer.

    Deliberately does not subclass ``torch.optim.Optimizer``: it implements the
    minimal ``param_groups`` / ``state`` / ``zero_grad`` interface itself
    (behavior is compatible with standard optimizers), avoiding the
    ``torch._dynamo`` import crash on ``torch.optim`` seen in some environments.

    Example parameter groups::

        opt = FlowOptimizer([
            {'params': manifold_2d_params, 'method': 'rmona',
             'lr': 0.02, 'momentum': 0.9, 'alpha': 0.1, 'beta_a': 0.9,
             'smp_iter': 8, 'smp_path': 'closed', 'msign': 'svd',
             'retraction': 'qr'},
            {'params': other_params, 'method': 'adamw', 'lr': 1e-3},
        ])
    """

    # ---- Public hyperparameters ----
    LRS = dict(
        rsgd=0.01, rsgd_m=0.01, radam=0.01, cayley=0.01,
        skewon=0.01, manifold_muon=0.01, rmona=0.01,
        muon=0.02, mona=0.02, adamw=1e-3,
    )

    def __init__(self, params, defaults=None):
        self.defaults = defaults or {}
        self.state = {}
        self.param_groups = []
        for group in params:
            pg = dict(group)
            # Materialize params: model.parameters() is a generator and is
            # exhausted after a single iteration.
            if 'params' in pg:
                pg['params'] = list(pg['params'])
            pg.setdefault('method', 'rmona')
            pg.setdefault('lr', self.LRS.get(pg['method'], 1e-3))
            pg.setdefault('momentum', 0.9)
            pg.setdefault('nesterov', True)
            pg.setdefault('alpha', 0.1)
            pg.setdefault('beta_a', 0.9)
            pg.setdefault('betas', (0.9, 0.999))
            pg.setdefault('eps', 1e-8)
            pg.setdefault('weight_decay', 0.0)
            pg.setdefault('smp_iter', 8)
            pg.setdefault('smp_path', 'closed')
            pg.setdefault('msign', 'svd')
            pg.setdefault('retraction', 'qr')
            self.param_groups.append(pg)

    def zero_grad(self, set_to_none=True):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None

    def step(self, closure=None):
        with torch.no_grad():
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
            for group in self.param_groups:
                method = group['method']
                for p in group['params']:
                    if p.grad is None:
                        continue
                    if method == 'adamw':
                        self._adamw_step(p, group)
                    elif method in ('muon', 'mona'):
                        self._euclid_matrix_step(p, group)
                    elif method in ('rsgd', 'rsgd_m', 'radam', 'cayley',
                                    'skewon', 'manifold_muon', 'rmona'):
                        self._stiefel_step(p, group)
                    else:
                        raise ValueError(f"unknown method {method}")
            return loss

    # ------------------------------------------------------------------
    # Manifold methods
    # ------------------------------------------------------------------
    def _stiefel_step(self, p, group):
        g = p.grad
        G = proj_tangent(p, g)                     # (1) tangent-space projection
        state = self.state.setdefault(p, {})
        method = group['method']
        mu = group['momentum']
        lr = group['lr']
        retr = group['retraction']

        # Every step, reproject tangent-space state onto the current tangent
        # space (reprojection momentum transport). Note: only tangent vectors
        # (M/A/G_prev/m) are reprojected; v is an elementwise second moment,
        # not a tangent vector, and reprojecting it would break its meaning
        # (standard Riemannian-Adam convention).
        for key in ('M', 'A', 'G_prev', 'm'):
            if state.get(key) is not None:
                state[key] = proj_tangent(p, state[key])

        if method == 'rsgd':
            p.data = retract(p, -lr * G, retr)

        elif method == 'rsgd_m':
            M = state.setdefault('M', torch.zeros_like(G))
            M.mul_(mu).add_(G)
            p.data = retract(p, -lr * M, retr)

        elif method == 'radam':
            t = state.setdefault('t', 0) + 1
            state['t'] = t
            b1, b2 = group['betas']
            eps = group['eps']
            m = state.setdefault('m', torch.zeros_like(G))
            v = state.setdefault('v', torch.zeros_like(G))
            m.mul_(b1).add_(G, alpha=1 - b1)
            v.mul_(b2).addcmul_(G, G, value=1 - b2)
            mhat = m / (1 - b1 ** t)
            vhat = v / (1 - b2 ** t)
            upd = mhat / (vhat.sqrt() + eps)
            p.data = retract(p, -lr * upd, retr)

        elif method == 'cayley':
            p.data = retract(p, -lr * G, 'cayley')

        elif method in ('skewon', 'manifold_muon'):
            M = state.setdefault('M', torch.zeros_like(G))
            M.mul_(mu).add_(G)
            path = 'ap' if method == 'skewon' else 'dual'
            O = solve_smp(M, p, T=group['smp_iter'], path=path,
                          backend=group['msign'])
            p.data = retract(p, lr * O, retr)

        elif method == 'rmona':
            # (2) Curvature channel (parallel transport + difference)
            A = state.setdefault('A', torch.zeros_like(G))
            M = state.setdefault('M', torch.zeros_like(G))
            Gp = state.setdefault('G_prev', torch.zeros_like(G))
            GpT = proj_tangent(p, Gp)              # reprojection approx. of parallel transport
            D = G - GpT
            A.mul_(group['beta_a']).add_(D, alpha=1 - group['beta_a'])
            # (3) Curvature-aware momentum (MONA-style series)
            M.mul_(mu).add_(G + group['alpha'] * A)
            # (4) SMP direction solving
            O = solve_smp(M, p, T=group['smp_iter'], path=group['smp_path'],
                          backend=group['msign'])
            # (5) retraction back to the manifold
            p.data = retract(p, lr * O, retr)
            # (6) Save the tangent vector for the next-step difference
            state['G_prev'] = G
        else:
            raise ValueError(method)

    # ------------------------------------------------------------------
    # Euclidean matrix methods (Muon / MONA)
    # ------------------------------------------------------------------
    def _euclid_matrix_step(self, p, group):
        g = p.grad
        if g.ndim > 2:
            g = g.view(g.size(0), -1)
        state = self.state.setdefault(p, {})
        method = group['method']
        mu = group['momentum']
        lr = group['lr']

        buf = state.setdefault('momentum_buffer', torch.zeros_like(g))
        if method == 'muon':
            buf.mul_(mu).add_(g)
            if group['nesterov']:
                gn = g + mu * buf
            else:
                gn = buf
        else:  # mona
            a = state.setdefault('a', torch.zeros_like(g))
            gp = state.setdefault('g_prev', torch.zeros_like(g))
            a.mul_(group['beta_a']).add_(g - gp, alpha=1 - group['beta_a'])
            g_eff = g + group['alpha'] * a
            buf.mul_(mu).add_(g_eff)
            if group['nesterov']:
                gn = g_eff + mu * buf
            else:
                gn = buf
            state['a'] = a
            state['g_prev'] = g
        # Align with the original Muon zeropower: normalize the input first and
        # use the Newton-Schulz polynomial iteration (stable for arbitrary /
        # ill-conditioned matrices; it does not fail like SVD can).
        # Since the Euclidean W drifts from orthogonality, gradients may become
        # ill-conditioned, so robustness matters here.
        gn = gn / (gn.norm() + 1e-6)
        O = msign(gn, backend='newtonschulz', steps=5)
        scale = max(1, g.size(0) / g.size(1)) ** 0.5
        p.data.add_(O * scale, alpha=-lr)

    # ------------------------------------------------------------------
    # AdamW
    # ------------------------------------------------------------------

    def _adamw_step(self, p, group):
        g = p.grad
        state = self.state.setdefault(p, {})
        t = state.setdefault('t', 0) + 1
        state['t'] = t
        b1, b2 = group['betas']
        eps = group['eps']
        lr = group['lr']
        m = state.setdefault('m', torch.zeros_like(p))
        v = state.setdefault('v', torch.zeros_like(p))
        m.mul_(b1).add_(g, alpha=1 - b1)
        v.mul_(b2).addcmul_(g, g, value=1 - b2)
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        p.data.mul_(1 - lr * group['weight_decay'])
        p.data.addcdiv_(mhat, vhat.sqrt() + eps, value=-lr)

    def constraint_error(self):
        """Mean constraint violation (||W^T W - I||_F) over all manifold groups."""
        errs = {}
        for group in self.param_groups:
            if group['method'] == 'adamw':
                continue
            for p in group['params']:
                if p.ndim == 2:
                    errs.setdefault(group['method'], []).append(
                        orthogonality_error(p).item())
        return {k: float(sum(v) / len(v)) for k, v in errs.items()}


# Alias for documentation convenience: FlowOptimizer's rmona group is used directly
Rmona = FlowOptimizer
