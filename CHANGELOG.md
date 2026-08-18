# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- CI workflow (`.github/workflows/ci.yml`): ruff lint + pytest (with coverage)
  across Python 3.9–3.12 + sdist/wheel build check.
- `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`.
- `docs/README.zh.md`: Chinese translation of the README.
- `rmona/py.typed` (PEP 561) marker, shipped in the wheel.
- GitHub issue templates, pull-request template, `SECURITY.md`.

### Removed
- `docs/implement.md` (internal engineering/debugging notes) ahead of the
  open-source release; its essential numerical-safety rules already live in
  the module docstrings and `docs/design.md`.
- Cleaned internal references in `docs/` (legacy project/file names such as
  `RMONA.md`, `RMONA/optims.py`) to point to the actual repository layout.

### Fixed
- `cayley_retraction`: the Cayley skew generator `A = W0^T xi` is now exactly
  skew-symmetrized (`A <- (A - A^T)/2`) in both the square and non-square
  branches. Previously the generator was only approximately skew-symmetric
  (xi is tangent to the drifted W, not to the re-orthogonalized W0), so the
  Cayley transform was not orthogonal and each step amplified the constraint
  error geometrically — on large-gradient tasks `‖WᵀW − I‖_F` exploded to ~2.7
  within 200 steps, breaking the hard-orthogonality guarantee of the `cayley`
  method. Added regression tests with large gradients and long horizons
  (`test_cayley_retraction_long_sequence_stability`,
  `test_manifold_constraint_kept_large_gradient`).

### Changed
- `pyproject.toml`: license declared in the widely-compatible `{ file = "LICENSE" }`
  form (builds with older setuptools); added `[project.urls]`.
- README rewritten in English (primary language of the repo).

## [0.1.0] - 2026-08-17

### Added
- `FlowOptimizer`: unified manifold/Euclidean optimizer supporting 11 methods
  (`rmona`, `manifold_muon`, `skewon`, `cayley`, `rsgd`, `rsgd_m`, `radam`,
  `muon`, `mona`, `adamw`).
- Stiefel manifold primitives (`rmona.manifold`): tangent projection, spectral
  sign (`svd` / Newton-Schulz / skew-symmetric path), SMP direction solvers
  (closed-form / dual / alternating projection), QR and Cayley retractions,
  reprojection-based parallel transport.
- Curvature channel: EMA of transported gradient differences, injected into
  momentum (`alpha`, `beta_a`).
- Models: `OrthogonalRNN` (hard Stiefel constraint) and `ParamRNN`
  (exp/Cayley parameterization).
- Data: dependency-free MNIST loader with pMNIST/sMNIST permutations.
- Examples: Exp1 (Rayleigh-Ritz + Procrustes correctness), Exp2 (pMNIST
  orthogonal RNN, 12000 steps × lr sweep × 3 seeds), plotting.
- Tests: 64 unit tests covering manifold primitives, constraint preservation,
  curvature-channel ablations, and the public API surface.
- Docs: design, experiments, implementation notes, and API reference.

### Fixed
- `solve_smp_closed`: final tangent-space projection now applied (was missing,
  causing tangent error up to ~1e-2 from the `msign_skew` eigendecomposition).
- Default learning rates are now per-method (`FlowOptimizer.LRS`) and honor the
  `lr` override in each parameter group.

[Unreleased]: https://github.com/Jiangxiazhe/RMONA
[0.1.0]: https://github.com/Jiangxiazhe/RMONA/releases/tag/v0.1.0
