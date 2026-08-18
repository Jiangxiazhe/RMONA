# Contributing to RMONA

Thanks for your interest in contributing! This project is an open-source optimizer
library, and we welcome bug reports, feature requests, documentation improvements,
and pull requests.

## Getting started

```bash
git clone https://github.com/Jiangxiazhe/RMONA
cd rmona
pip install -e ."[dev]"    # installs pytest, pytest-cov, ruff
pytest tests/              # run the test suite (73 tests)
```

## Development workflow

1. **Fork** the repository and create a feature branch.
2. Make your changes. Keep each PR focused on one concern.
3. Run the linter and the test suite before submitting:

   ```bash
   ruff check rmona/ tests/ examples/
   python -m pytest tests/
   ```

   All tests must pass. If your change is a bug fix, add a regression test.
   If it's a new feature (e.g., a new manifold method or retraction), add tests
   covering correctness and edge cases.

4. Update documentation where relevant:
   - API changes → `docs/api.md` and module docstrings.
   - New hyperparameters → the parameter-group table in `docs/api.md`.
   - Behavior changes → `CHANGELOG.md` (under "Unreleased").

5. Open a pull request with a clear description of the change and, if applicable,
   the evidence (e.g., experimental results) supporting it.

## Style guide

- Follow [ruff](https://docs.astral.sh/ruff/) defaults configured in `pyproject.toml`
  (line length 88, `E`/`F`/`W`/`I`/`UP` rules selected).
- Type annotations are required on all public functions.
- Docstrings: English, in Google style (short summary, then `Args:`/`Returns:`).
- Keep the package self-contained: do not add new mandatory dependencies.
  Experimental dependencies belong in the `experiments` extra.

## Running experiments

Experiments live in `examples/`. To reproduce the paper results:

```bash
python examples/exp1_matrix.py --task rr --seeds 0 1 2
python examples/exp2_pmnist.py --steps 12000 --seeds 0 1 2
```

Large sweeps can be launched with `scripts/run_exp2_4gpu.sh` (multi-GPU, resumable).

## Reporting issues

Please include:

- A minimal reproducible snippet (preferably a standalone script).
- Your environment: `python -c "import rmona, torch; print(rmona.__version__, torch.__version__)"`
- Expected vs. actual behavior, and any error traceback.

## Code of conduct

Be respectful and constructive. Harassment and discrimination of any kind are not
tolerated.
