# Contributing

Thanks for considering a contribution. This repository is small on purpose: changes should keep the benchmark controlled, readable, and reproducible.

## Setup

~~~bash
git clone https://github.com/kopolart/activation-functions-benchmark.git
cd activation-functions-benchmark
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
~~~

## Before opening a PR

Run the checks that CI runs:

~~~bash
ruff check src tests
pytest -q
~~~

If you touched plotting or the notebook, also rebuild the affected artifacts and confirm they look right.

## What makes a good contribution

- **New activation functions** — extend `make_activation` in `src/activation_benchmark/models.py`, add the name to `ACTIVATIONS`, and extend tests/plots/docs as needed.
- **More seeds or a sweep** — write outputs to a new directory; do not silently replace `results/results.json` unless the canonical claim is intentionally changing.
- **Documentation fixes** — especially clarity, broken links, or reproducibility gaps.
- **Bug fixes** — include a regression test when possible.

## Adding an activation

1. Add the constructor to `make_activation` (`src/activation_benchmark/models.py`).
2. Add the name to `ACTIVATIONS`.
3. Decide how health metrics apply. If the activation can produce dead or saturated units under the existing definitions, update `src/activation_benchmark/metrics.py`.
4. Add a color/label in `src/activation_benchmark/plotting.py`.
5. Update docs and, if you rerun the canonical benchmark, the results artifacts and checksums.

## Style

- Keep functions small and typed.
- Prefer explicit configuration over hidden constants.
- Do not add dependencies unless they clearly pay for themselves.
- Do not commit secrets, large generated caches, or machine-specific paths.

## Commit messages

Use clear, imperative messages. Conventional prefixes such as `feat:`, `fix:`, `docs:`, and `test:` are welcome but not required.

## Reporting issues

When reporting a bug or a numerical discrepancy, include:

- OS and Python version
- package versions (`pip freeze` or equivalent)
- the exact command you ran
- the observed output and the expected output

## Code of conduct

Be respectful and constructive. Assume good intent, review the work rather than the person, and keep discussions technical.