# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-11-27

### Added

- Controlled benchmark of Sigmoid, Tanh, ReLU, and LeakyReLU in a 2 → 32 → 32 → 32 → 1 MLP.
- Twenty training runs (four activations × five seeds) on the two-moons dataset.
- Per-epoch training/test loss and first-layer gradient-norm histories.
- Convergence-epoch, dead-unit, and saturated-unit diagnostics.
- One-way ANOVA and Kruskal-Wallis tests for test accuracy.
- Reproducible `activation-benchmark` CLI with `run` and `figures` commands.
- Publication-quality figures for activation shapes, training dynamics, accuracy/convergence, and unit health.
- Executed end-to-end notebook walkthrough.
- Methodology, results, and reproducibility guides.
- GitHub Actions CI (Python 3.11 and 3.12), issue templates, and citation metadata.

### Notes

- Canonical results are in `results/results.json`; the SHA-256 is recorded in `docs/REPRODUCIBILITY.md`.
- Timings are single-thread CPU measurements and are not a hardware benchmark.