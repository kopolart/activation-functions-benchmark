# Reproducibility

This guide explains how to regenerate the canonical artifacts and how to verify that a rerun matches the committed results.

## Environment

The committed `results/results.json` was produced with:

| Component | Version |
|---|---|
| Python | 3.14.7 |
| PyTorch | 2.14.1+cpu |
| NumPy | 2.5.2 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| Matplotlib | 3.11.1 |
| pandas | 3.0.6 |
| OS | Windows 11 |

The package supports Python 3.10+. Newer patch versions of the scientific stack should produce equivalent accuracy numbers on CPU for this configuration; if you observe differences, record the versions and open an issue.

## Determinism contract

The experiment controls randomness at three points:

1. **Dataset split and scaler** — `random_state=42`, stratified 70/30 split.
2. **Model initialization** — `torch.manual_seed(seed)` before constructing the MLP. For the same seed, all four activations receive identical initial weights.
3. **Minibatch order** — `torch.Generator().manual_seed(seed + 1000)` for the per-epoch minibatch permutation.

In our environment, repeated runs with the same seed produced identical accuracy and loss values. The code also sets `torch.set_num_threads(1)` so wall-clock timings are less noisy; this does not change the learned weights.

## Regenerate everything

~~~bash
# from the repository root, with the package installed
activation-benchmark run --epochs 150 --seeds 5 --output-dir results
~~~

This writes `results/results.json`, `results/activation_summary.csv`, and four figures in `results/figures/`. The command takes roughly 30–60 seconds on a modern CPU.

## Rebuild figures only

~~~bash
activation-benchmark figures --results results/results.json --output-dir results/figures
~~~

This reads the committed JSON and regenerates the PNGs. Use it when you change plot styling but not the experiment.

## Smoke test

~~~bash
activation-benchmark run --epochs 2 --seeds 1 --output-dir /tmp/activation-smoke
~~~

The smoke test confirms the pipeline runs end to end. It is not expected to match the canonical numbers.

## Verify the canonical JSON

The committed `results/results.json` has the following SHA-256:

~~~text
42b2d5a51ded02dfcc7b796a2e0818b6ddf30a3e907389287539663b3e373428
~~~

Verify on Windows PowerShell:

~~~powershell
Get-FileHash results\results.json -Algorithm SHA256
~~~

Verify on macOS / Linux:

~~~bash
sha256sum results/results.json
~~~

A byte-identical file means the JSON artifact was not altered. If you rerun the experiment in a different environment, the hash may differ because JSON formatting or floating-point serialization can change; compare the summary metrics instead:

~~~bash
python -c "from activation_benchmark.analysis import load_results, summarize; print(summarize(load_results('results/results.json')).to_string(index=False))"
~~~

## Tests and lint

~~~bash
ruff check src tests
pytest -q
~~~

The test suite covers configuration validation, data leakage prevention, model shape/count, metric correctness, and a two-epoch smoke run. It is designed to run quickly on CPU.

## Artifact layout

~~~text
results/
├── results.json                 # canonical per-run records with histories
├── activation_summary.csv       # aggregate metrics table
└── figures/
    ├── activation_shapes.png
    ├── loss_and_gradient.png
    ├── accuracy_and_convergence.png
    └── health_profile.png
~~~

`results.json` contains, for each of the 20 runs: activation, seed, final accuracies/losses, convergence epoch, gradient/dead/saturated diagnostics, timing, and the per-epoch histories used by the plots. It is the source of truth; CSV and figures are derived artifacts.

## What is and is not deterministic

| Item | Deterministic? | Notes |
|---|---:|---|
| Dataset split | Yes | Fixed by `data_seed=42` |
| Initial weights | Yes | Fixed by per-run seed |
| Minibatch order | Yes | Fixed by `seed + 1000` |
| Final accuracy/loss | Yes on same CPU stack | Verified across reruns in this environment |
| Epoch histories | Yes on same CPU stack | Same caveat as above |
| Wall-clock time | No | Hardware and system load dependent |
| PNG bytes | Usually not across environments | Font rendering / Matplotlib backend differences |
| JSON byte hash | Environment-sensitive | Compare metric values if hash differs |

## Reporting changes

If you change the experiment configuration, do **not** overwrite the canonical results silently. Instead:

1. Record the new config in `docs/METHODOLOGY.md` or a new experiment note.
2. Write outputs to a separate directory (e.g., `results/width-64/`).
3. Update `docs/RESULTS.md` only when the canonical claim changes.
4. Note the change in `CHANGELOG.md`.