# API Reference

## `rmona.FlowOptimizer`

Unified manifold/Euclidean optimizer. It does not subclass
`torch.optim.Optimizer`, but provides compatible `param_groups` / `state` /
`zero_grad` / `step` interfaces.

### Constructor

```python
FlowOptimizer(param_groups, defaults=None)
```

- `param_groups`: a list of group dicts; each must contain `params`, and may
  contain `method` and other hyperparameters.
- `defaults`: optional global default hyperparameters.

### Parameter-group hyperparameters

| key | default | description |
|---|---|---|
| `method` | `'rmona'` | `rmona`/`manifold_muon`/`skewon`/`cayley`/`rsgd`/`rsgd_m`/`radam`/`muon`/`mona`/`adamw` |
| `lr` | per-method default | see `LRS` below |
| `momentum` | `0.9` | momentum coefficient |
| `nesterov` | `True` | Nesterov momentum for Muon/MONA |
| `alpha` | `0.1` | curvature-channel injection strength (rmona/mona) |
| `beta_a` | `0.9` | curvature EMA coefficient |
| `betas` | `(0.9, 0.999)` | Adam moment coefficients (radam/adamw) |
| `eps` | `1e-8` | Adam numerical stabilizer |
| `weight_decay` | `0.0` | AdamW weight decay |
| `smp_iter` | `8` | SMP dual / alternating-projection iterations |
| `smp_path` | `'closed'` | SMP solver path: `closed`/`dual`/`ap` |
| `msign` | `'svd'` | spectral-sign backend: `svd`/`newtonschulz` |
| `retraction` | `'qr'` | `qr`/`cayley` |

Default learning rates: `rmona=0.01, manifold_muon=0.01, skewon=0.01, cayley=0.01,
rsgd=0.01, rsgd_m=0.01, radam=0.01, muon=0.02, mona=0.02, adamw=1e-3`.

### Methods

- `zero_grad(set_to_none=True)`: zero out all parameter gradients.
- `step(closure=None)`: perform one update step (optionally re-computing the loss
  via a closure).
- `constraint_error() -> dict`: mean constraint violation `‖WᵀW − I‖_F` per
  manifold method.

### Example

```python
import torch
from rmona import FlowOptimizer

opt = FlowOptimizer([
    {'params': [W_hh], 'method': 'rmona', 'lr': 0.005, 'smp_path': 'closed',
     'retraction': 'qr'},
    {'params': other_params, 'method': 'adamw', 'lr': 1e-3},
])
```

## `rmona.manifold` — Stiefel manifold primitives

| function | description |
|---|---|
| `proj_tangent(W, G)` | tangent-space projection `G - 0.5W(WᵀG+GᵀW)` |
| `parallel_transport_approx(W, v)` | reprojection approximation of parallel transport (= proj_tangent) |
| `orthogonality_error(W)` | `‖WᵀW − I‖_F` |
| `random_stiefel(n, p)` | random semi-orthogonal matrix (QR) |
| `msign(M, backend='svd')` | spectral sign (`svd`/`newtonschulz`) |
| `msign_skew(A)` | spectral sign of a skew-symmetric matrix (preserves skew-symmetry) |
| `solve_smp(M, W, T=8, path='closed', backend='svd')` | SMP descent direction |
| `retract(W, xi, kind='qr')` | unified retraction entry |
| `qr_retraction(W, xi)` | QR retraction |
| `cayley_retraction(W, xi)` | Cayley retraction |

## `rmona.models`

- `OrthogonalRNN(input_size, hidden, num_classes, init='identity')`:
  RNN whose recurrent weight `W_hh` is a Stiefel parameter (pass `W_hh` to
  `forward`).
- `ParamRNN(hidden, num_classes, param='exp')`: parameterized orthogonal RNN
  (`W_hh = exp(A−Aᵀ)` / Cayley parameterization); `A` is a free parameter.

## `rmona.data`

- `load_mnist_arrays(root, train=True)`: read IDX gz data.
- `load_mnist_tensors(root, device='cuda')`: load as device tensors.
- `permutation(root, seed=0)`: fixed pMNIST permutation (cached).

## `rmona.utils`

- `append_rows(path, rows)`: append rows to a CSV.
- `load_existing(path, key_cols)`: read completed task keys (for `--resume`).
- `ensure_dir(path)`: ensure a directory exists.
- `summarize(rows, group, stat)`: group-wise mean/median summary.
