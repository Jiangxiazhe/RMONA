"""
Experiment 2: pMNIST orthogonal RNN benchmark (docs/experiments.md §4)

Task: permuted MNIST (pMNIST) sequence classification, sequence length 784.
Model: single-hidden-layer tanh RNN with recurrent weight W_hh in O(hidden)
       (orthogonal constraint). Input encoding / output classification layers
       are unconstrained parameters (AdamW).

Methods:
  Manifold methods (optimize W_hh in O(hidden) directly):
    rsgd_m, radam, cayley, skewon, manifold_muon, rmona
  Parameterized methods (W_hh = exp(A) / Cayley(A), A via AdamW):
    expRNN, cayleyRNN
  Euclidean methods (W_hh via Euclidean Muon/MONA, constraint-violation baseline):
    muon, mona

Metrics: training loss, test accuracy, constraint violation ||W^T W - I||_F,
         wall-clock time.

Usage:
    CUDA_VISIBLE_DEVICES=2 python exp2_pmnist.py --steps 3000
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.nn as nn

from rmona import FlowOptimizer
from rmona.data import load_mnist_tensors, permutation
from rmona.manifold import orthogonality_error
from rmona.models import OrthogonalRNN, ParamRNN
from rmona.utils.io import append_rows, ensure_dir, load_existing

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')
RESULT_DIR = ensure_dir(os.path.join(ROOT, 'results'))

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_optimizer(method, model, lr, alpha=0.1, beta_a=0.9):
    """Build (optimizer, orthogonality checker)."""
    if hasattr(model, 'W_hh'):
        W_hh = model.W_hh
        if method in ('expRNN', 'cayleyRNN'):
            raise ValueError('param methods use ParamRNN')
        if method in ('muon', 'mona'):
            opt = FlowOptimizer([
                {'params': [W_hh], 'method': method, 'lr': lr,
                 'alpha': alpha, 'beta_a': beta_a},
                {'params': [p for n, p in model.named_parameters()
                            if n != 'W_hh'], 'method': 'adamw', 'lr': 1e-3},
            ])
            orth = lambda: orthogonality_error(W_hh.detach()).item()
        else:
            opt = FlowOptimizer([
                {'params': [W_hh], 'method': method, 'lr': lr,
                 'alpha': alpha, 'beta_a': beta_a},
                {'params': [p for n, p in model.named_parameters()
                            if n != 'W_hh'], 'method': 'adamw', 'lr': 1e-3},
            ])
            orth = lambda: orthogonality_error(W_hh.detach()).item()
    else:  # ParamRNN (parameterized methods: A via AdamW, lr can be swept)
        opt = FlowOptimizer([
            {'params': model.parameters(), 'method': 'adamw', 'lr': lr},
        ])
        W_hh = model._make_W().detach()
        orth = lambda: orthogonality_error(W_hh).item()
    return opt, orth


def evaluate(model, x_te, y_te, batch=1000, W_hh=None):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, x_te.size(0), batch):
            xb = x_te[i:i + batch]
            yb = y_te[i:i + batch]
            if isinstance(model, ParamRNN):
                out = model(xb)
            else:
                out = model(xb, W_hh if W_hh is not None else model.W_hh)
            correct += (out.argmax(1) == yb).sum().item()
            total += xb.size(0)
    model.train()
    return correct / total


def train(method, x_tr, y_tr, x_te, y_te, perm, steps, hidden, batch, lr,
          seed, log_every=100):
    torch.manual_seed(seed)
    if method in ('expRNN', 'cayleyRNN'):
        model = ParamRNN(hidden=hidden, param='exp' if method == 'expRNN'
                         else 'cayley').to(DEV)
    else:
        model = OrthogonalRNN(input_size=784, hidden=hidden,
                              init='identity').to(DEV)
    opt, orth_fn = build_optimizer(method, model, lr)

    x_tr_p = x_tr[:, perm]
    x_te_p = x_te[:, perm]
    n = x_tr_p.size(0)
    criterion = nn.CrossEntropyLoss()
    hist = []
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=DEV)
        xb, yb = x_tr_p[idx], y_tr[idx]
        opt.zero_grad()
        if isinstance(model, ParamRNN):
            out = model(xb)
            W_hh = model._make_W().detach()
        else:
            W_hh = model.W_hh
            out = model(xb, W_hh)
        loss = criterion(out, yb)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            acc = evaluate(model, x_te_p, y_te,
                           W_hh=W_hh if not isinstance(model, ParamRNN)
                           else None)
            hist.append(dict(step=step, loss=loss.item(), acc=acc,
                             orth=orth_fn()))
            print(f"    [{method}] step={step:5d} loss={loss.item():.4f} "
                  f"acc={acc*100:.2f}% orth={orth_fn():.2e}", flush=True)
    wall = time.time() - t0
    return model, hist, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--lr_grid', type=str, default='0.01',
                    help='comma-separated lr candidates for manifold methods '
                         '(per-method sweep; best_acc chosen afterwards)')
    ap.add_argument('--param_lr_grid', type=str, default='0.001',
                    help='comma-separated adamw lr candidates for expRNN/cayleyRNN')
    ap.add_argument('--log_every', type=int, default=200)
    ap.add_argument('--out_tag', type=str, default='',
                    help='suffix for result files (to distinguish parallel writers)')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--methods', type=str, nargs='+',
                    default=['rsgd_m', 'radam', 'cayley', 'skewon',
                             'manifold_muon', 'rmona', 'expRNN', 'cayleyRNN',
                             'muon', 'mona'])
    ap.add_argument('--alpha', type=float, default=0.1)
    ap.add_argument('--resume', action='store_true',
                    help='skip (seed, method, lr) already present in the summary CSV')
    args = ap.parse_args()

    tag = f'_{args.out_tag}' if args.out_tag else ''
    sum_path = os.path.join(RESULT_DIR, f'exp2_summary{tag}.csv')
    traj_path = os.path.join(RESULT_DIR, f'exp2_traj{tag}.csv')
    done = load_existing(sum_path) if args.resume else set()

    # Per-method lr candidates (per-method sweep)
    manifold_lrs = [float(x) for x in args.lr_grid.split(',')]
    param_lrs = [float(x) for x in args.param_lr_grid.split(',')]
    lr_map = {}
    for m in ['rsgd_m', 'radam', 'cayley', 'skewon', 'manifold_muon', 'rmona']:
        lr_map[m] = manifold_lrs
    for m in ['expRNN', 'cayleyRNN']:
        lr_map[m] = param_lrs
    lr_map['muon'] = [args.lr]
    lr_map['mona'] = [args.lr]

    print('loading MNIST ...', flush=True)
    x_tr, y_tr, x_te, y_te = load_mnist_tensors(DATA_DIR, device=DEV)
    perm = permutation(DATA_DIR, seed=0).to(DEV)
    print(f'data: train {x_tr.shape} test {x_te.shape}', flush=True)

    for seed in args.seeds:
        for method in args.methods:
            for lr in lr_map.get(method, [args.lr]):
                if (seed, method, lr) in done:
                    print(f'skip {method} lr={lr} (seed {seed}): done',
                          flush=True)
                    continue
                print(f'=== {method} lr={lr} (seed {seed}) ===', flush=True)
                try:
                    model, hist, wall = train(
                        method, x_tr, y_tr, x_te, y_te, perm, args.steps,
                        args.hidden, args.batch, lr, seed, args.log_every)
                except Exception as e:
                    print(f'    {method} FAILED: {e}', flush=True)
                    continue
                best_acc = max(h['acc'] for h in hist)
                last = hist[-1]
                srow = dict(seed=seed, method=method, lr=lr,
                            best_acc=best_acc, final_acc=last['acc'],
                            final_loss=last['loss'], final_orth=last['orth'],
                            wall_time=wall)
                append_rows(sum_path, [srow])
                append_rows(traj_path,
                            [dict(seed=seed, method=method, lr=lr, **h)
                             for h in hist])
                print(f'    {method}: best_acc={best_acc*100:.2f}% '
                      f'orth={last["orth"]:.2e} time={wall:.1f}s', flush=True)

    print(f'results appended -> {sum_path} / {traj_path}')


if __name__ == '__main__':
    main()
