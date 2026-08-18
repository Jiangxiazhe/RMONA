# Reproducing the experiments

This directory contains the two benchmark scripts behind the RMONA paper,
both built on top of the `rmona` package.

## Experiment 1: orthogonal matrix learning (convex sanity check)

Two tasks (Rayleigh-Ritz eigenproblem / orthogonal Procrustes least squares),
7 manifold methods + 2 Euclidean baselines, 3 seeds × lr sweep.

```bash
# Run from the repository root (so that `import rmona` works)
cd ..   # back to the RMONA root
python examples/exp1_matrix.py --task rr     # Rayleigh-Ritz
python examples/exp1_matrix.py --task proc   # Procrustes
```

## Experiment 2: pMNIST orthogonal RNN (ill-conditioned long-range task)

Permuted MNIST sequence classification (784 steps), recurrent weight
`W_hh ∈ O(128)`.

```bash
cd ..
# First prepare MNIST data (put IDX gz files into data/)
python -c "from rmona.data import load_mnist_tensors; load_mnist_tensors('data')"

# All 10 methods, 12000 steps
CUDA_VISIBLE_DEVICES=0 python examples/exp2_pmnist.py \
    --steps 12000 --hidden 128 --lr_grid 0.005,0.01,0.02 \
    --param_lr_grid 0.001 --seeds 0 1 2

# Resume from checkpoints (skips completed runs)
CUDA_VISIBLE_DEVICES=0 python examples/exp2_pmnist.py --resume
```

## Multi-GPU parallelism

`scripts/run_exp2_4gpu.sh` dispatches method subsets across 4 GPUs; each GPU
writes `results/exp2_summary_gpu<i>.csv` / `exp2_traj_gpu<i>.csv`.

Merge the per-GPU results afterwards:

```bash
python -c "
import pandas as pd
s = pd.concat([pd.read_csv(f'results/exp2_summary_gpu{i}.csv') for i in range(4)])
s = s.drop_duplicates(subset=['seed','method','lr'])
s.to_csv('results/exp2_summary.csv', index=False)
"
```

## Plotting

```bash
python examples/plot_results.py
# Output: plots/exp1_*.png, plots/exp2_*.png
```
