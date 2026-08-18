"""
Experiment 1: orthogonal matrix learning benchmark (docs/experiments.md §3)

Task A (Rayleigh-Ritz eigenproblem):
    min_{W in St(n,p)} -tr(W^T C W)
    Optimal solution = top-p eigenvectors of C; optimal value = -(sum of top-p eigenvalues).
Task B (orthogonal Procrustes least squares):
    min_{W in St(n,p)} ||W A - B||_F^2
    Optimal solution = polar factor U V^T of (B A^T); analytic reference solution.

Methods: 7 manifold methods + Euclidean Muon/MONA (soft-constraint baselines).
Metrics: steps to convergence (threshold), final loss, constraint violation
         ||W^T W - I||_F, wall-clock time.

Usage:
    CUDA_VISIBLE_DEVICES=2 python exp1_matrix.py --task rr --iters 2000
    CUDA_VISIBLE_DEVICES=2 python exp1_matrix.py --task proc --iters 2000
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch

from rmona import FlowOptimizer
from rmona.manifold import orthogonality_error, random_stiefel
from rmona.utils.io import ensure_dir

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_DIR = ensure_dir(os.path.join(ROOT, 'results'))

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

# All methods (7 manifold + 2 Euclidean)
MANIFOLD_METHODS = ['rsgd', 'rsgd_m', 'radam', 'cayley',
                    'skewon', 'manifold_muon', 'rmona']
EUCLID_METHODS = ['muon', 'mona']
ALL_METHODS = MANIFOLD_METHODS + EUCLID_METHODS

# Learning-rate candidates per method (small sweep; pick best final loss)
LR_CANDIDATES = [0.005, 0.01, 0.02, 0.05]


def rayleigh_ritz_loss(W, C, soft_lambda=0.0):
    loss = -torch.trace(W.T @ C @ W)
    if soft_lambda > 0:
        loss = loss + soft_lambda * orthogonality_error(W) ** 2
    return loss


def procrustes_loss(W, A, B, soft_lambda=0.0):
    loss = ((W @ A - B) ** 2).sum()
    if soft_lambda > 0:
        loss = loss + soft_lambda * orthogonality_error(W) ** 2
    return loss


def run_task(method, task, n, p, k, iters, lr, seed, log_every=20):
    """Run a single (method, lr) combination; returns the history records."""
    torch.manual_seed(seed)
    W = random_stiefel(n, p, device=DEV)
    W.requires_grad_()

    if task == 'rr':
        # Symmetric matrix with spectrum 10 -> 1
        C = torch.diag(torch.linspace(10.0, 1.0, n, device=DEV)).float()
        ref_opt = -(torch.linspace(10.0, 1.0, n, device=DEV)[:p].sum()).item()
        soft_lambda = 0.1 if method in EUCLID_METHODS else 0.0

        def obj():
            return rayleigh_ritz_loss(W, C, soft_lambda)
    else:  # procrustes
        A = torch.randn(p, k, device=DEV) * 0.5
        B = torch.randn(n, k, device=DEV) * 0.5
        soft_lambda = 0.1 if method in EUCLID_METHODS else 0.0
        # Reference optimum: W* = U V^T where B A^T = U S V^T (orthogonal Procrustes closed form)
        U, S, Vh = torch.linalg.svd(B @ A.T, full_matrices=False)
        W_opt = U @ Vh
        ref_opt = procrustes_loss(W_opt, A, B, 0.0).item()

        def obj():
            return procrustes_loss(W, A, B, soft_lambda)

    opt = FlowOptimizer([{'params': [W], 'method': method, 'lr': lr}])

    hist = []
    t0 = time.time()
    for it in range(iters):
        loss = obj()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % log_every == 0 or it == iters - 1:
            hist.append(dict(step=it, loss=loss.item(),
                             orth=orthogonality_error(W).item()))
    wall = time.time() - t0
    return hist, ref_opt, wall


def threshold_steps(hist, ref_opt, tol=0.05):
    """Smallest step at which loss < ref_opt + tol (None if never reached)."""
    for h in hist:
        if h['loss'] < ref_opt + tol:
            return h['step']
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', choices=['rr', 'proc'], default='rr')
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--p', type=int, default=12)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--lr_scan', action='store_true', default=True)
    args = ap.parse_args()

    task = args.task
    tag = 'rr' if task == 'rr' else 'proc'
    rows = []          # summary rows
    full = []          # all trajectory rows

    for seed in args.seeds:
        for method in ALL_METHODS:
            lrs = LR_CANDIDATES if args.lr_scan else [0.02]
            best = None
            for lr in lrs:
                hist, ref_opt, wall = run_task(
                    method, task, args.n, args.p, args.k, args.iters, lr, seed)
                final = hist[-1]['loss']
                if best is None or final < best['final']:
                    best = dict(method=method, lr=lr, hist=hist,
                                ref_opt=ref_opt, wall=wall, final=final,
                                final_orth=hist[-1]['orth'])
                for h in hist:
                    full.append(dict(seed=seed, method=method, lr=lr, **h))
            st = threshold_steps(best['hist'], best['ref_opt'])
            rows.append(dict(seed=seed, method=method, lr=best['lr'],
                             steps_to_opt=st, final_loss=best['final'],
                             final_orth=best['final_orth'],
                             wall_time=best['wall']))
            print(f"[{task}] seed={seed} {method:14s} lr={best['lr']:.3f} "
                  f"steps_to_opt={st} final_loss={best['final']:.6f} "
                  f"orth={best['final_orth']:.2e}")

    # Save summary and trajectories
    with open(os.path.join(RESULT_DIR, f'exp1_{tag}_summary.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(RESULT_DIR, f'exp1_{tag}_traj.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=full[0].keys())
        w.writeheader()
        w.writerows(full)
    print(f"saved -> {RESULT_DIR}/exp1_{tag}_summary.csv / _traj.csv")


if __name__ == '__main__':
    main()
