"""Summarize Experiment 1/2 results and plot them (output to plots/)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from rmona.utils.io import ensure_dir

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULT_DIR = os.path.join(ROOT, 'results')
PLOT_DIR = ensure_dir(os.path.join(ROOT, 'plots'))

# Method -> color/linestyle
METHOD_STYLE = {
    'rsgd':          dict(color='#888888', ls='-',  lw=1.4, label='RSGD'),
    'rsgd_m':        dict(color='#999999', ls='-',  lw=1.4, label='RSGD-mom'),
    'radam':         dict(color='#666666', ls='--', lw=1.4, label='RAdam'),
    'cayley':        dict(color='#b58900', ls='-.', lw=1.6, label='Cayley SGD'),
    'skewon':        dict(color='#2b8cbe', ls='-',  lw=1.6, label='Skewon'),
    'manifold_muon': dict(color='#35978f', ls='--', lw=1.6, label='Manifold-Muon'),
    'rmona':         dict(color='#d73027', ls='-',  lw=2.2, label='RMONA (ours)'),
    'muon':          dict(color='#7b3294', ls=':',  lw=1.6, label='Muon (Euclid)'),
    'mona':          dict(color='#f4a582', ls=':',  lw=1.6, label='MONA (Euclid)'),
    'expRNN':        dict(color='#a1d99b', ls='-.', lw=1.6, label='expRNN'),
    'cayleyRNN':     dict(color='#fdae6b', ls='-.', lw=1.6, label='cayleyRNN'),
}


def style(method):
    return METHOD_STYLE.get(method, dict(color='k', ls='-', lw=1.2,
                                         label=method))


def plot_exp1(tag, title):
    """Experiment 1: loss/orth curves + convergence-step bar chart"""
    traj = pd.read_csv(os.path.join(RESULT_DIR, f'exp1_{tag}_traj.csv'))
    summ = pd.read_csv(os.path.join(RESULT_DIR, f'exp1_{tag}_summary.csv'))

    methods = list(traj['method'].unique())
    best_lr = {}
    for m in methods:
        sub = summ[summ['method'] == m]
        best_lr[m] = sub.loc[sub['final_loss'].idxmin(), 'lr']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for m in methods:
        s = style(m)
        sub = traj[(traj['method'] == m) & (traj['lr'] == best_lr[m])]
        grp = sub.groupby('step')[['loss', 'orth']].mean()
        axes[0].plot(grp.index, grp['loss'], color=s['color'], ls=s['ls'],
                     lw=s['lw'], label=s['label'])
        axes[1].plot(grp.index, grp['orth'], color=s['color'], ls=s['ls'],
                     lw=s['lw'], label=s['label'])
    axes[0].set_xlabel('iteration'); axes[0].set_ylabel('loss')
    axes[0].set_title(f'Exp1 {title}: loss')
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel('iteration'); axes[1].set_ylabel(r'$\|W^\top W - I\|_F$')
    axes[1].set_title(f'Exp1 {title}: constraint violation')
    axes[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'exp1_{tag}_curves.png'), dpi=150)

    # Convergence-step bar chart (steps for loss < opt + tol; median over 3 seeds)
    agg = summ.groupby('method')['steps_to_opt'].median()
    agg = agg.reindex(methods)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [style(m)['color'] for m in agg.index]
    ax.bar(range(len(agg)), agg.values, color=colors)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([style(m)['label'] for m in agg.index], rotation=30,
                       ha='right', fontsize=9)
    ax.set_ylabel('steps to reach optimum + tol')
    ax.set_title(f'Exp1 {title}: convergence speed')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'exp1_{tag}_steps.png'), dpi=150)
    print(f'plot exp1_{tag} saved')


def plot_exp2(traj_path=None, summ_path=None, out_prefix='exp2'):
    """Experiment 2: pMNIST acc/loss curves (best lr by best_acc per method)"""
    traj_path = traj_path or os.path.join(RESULT_DIR, 'exp2_traj.csv')
    summ_path = summ_path or os.path.join(RESULT_DIR, 'exp2_summary.csv')
    traj = pd.read_csv(traj_path)
    summ = pd.read_csv(summ_path)
    methods = list(traj['method'].unique())
    best_lr = {}
    if 'lr' in summ.columns:
        for m in methods:
            sub = summ[summ['method'] == m]
            if sub['lr'].notna().all():
                best_lr[m] = sub.groupby('lr')['best_acc'].mean().idxmax()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for m in methods:
        s = style(m)
        sub = traj[traj['method'] == m]
        if m in best_lr:
            sub = sub[sub['lr'] == best_lr[m]]
        grp = sub.groupby('step')[['acc', 'loss']].mean()
        axes[0].plot(grp.index, grp['acc'] * 100, color=s['color'], ls=s['ls'],
                     lw=s['lw'], label=s['label'])
        axes[1].plot(grp.index, grp['loss'], color=s['color'], ls=s['ls'],
                     lw=s['lw'], label=s['label'])
    axes[0].set_xlabel('training step'); axes[0].set_ylabel('test accuracy (%)')
    axes[0].set_title('Exp2 pMNIST: test accuracy')
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel('training step'); axes[1].set_ylabel('train loss')
    axes[1].set_title('Exp2 pMNIST: train loss')
    axes[1].set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'{out_prefix}_pmnist.png'), dpi=150)

    # Final-accuracy bar chart (best lr per method, best_acc averaged over seeds)
    rows = []
    for m in methods:
        sub = summ[summ['method'] == m]
        if m in best_lr:
            sub = sub[sub['lr'] == best_lr[m]]
        rows.append(dict(method=m, best_acc=sub['best_acc'].mean()))
    agg = pd.DataFrame(rows).set_index('method').reindex(methods)['best_acc']
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [style(m)['color'] for m in agg.index]
    ax.bar(range(len(agg)), agg.values * 100, color=colors)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels([style(m)['label'] for m in agg.index], rotation=30,
                       ha='right', fontsize=9)
    ax.set_ylabel('best test accuracy (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Exp2 pMNIST: best test accuracy (best-lr, mean over seeds)')
    for i, v in enumerate(agg.values * 100):
        ax.text(i, v + 1, f'{v:.1f}', ha='center', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f'{out_prefix}_acc.png'), dpi=150)
    print('plot exp2 saved')


if __name__ == '__main__':
    plot_exp1('rr', 'Rayleigh-Ritz')
    plot_exp1('proc', 'Procrustes')
    plot_exp2()
    print('all plots saved to', PLOT_DIR)
