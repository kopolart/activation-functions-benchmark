"""Command-line interface for running the benchmark and rebuilding figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from .analysis import load_results, save_results, statistical_tests, summarize, write_summary_csv
from .config import ExperimentConfig
from .plotting import make_all_figures
from .training import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="activation-benchmark",
        description="A controlled PyTorch comparison of activation functions in a deep MLP.",
    )
    parser.add_argument("--version", action="version", version="activation-benchmark 1.0.0")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Train the full activation x seed sweep.")
    run_parser.add_argument("--epochs", type=int, default=ExperimentConfig().epochs)
    run_parser.add_argument("--seeds", type=int, default=ExperimentConfig().seeds)
    run_parser.add_argument("--output-dir", default="results")
    run_parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="PyTorch CPU threads (1 keeps timings comparable).",
    )
    run_parser.add_argument("--quiet", action="store_true", help="Suppress per-run progress.")
    run_parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write JSON/CSV only and skip figure generation.",
    )

    figure_parser = subparsers.add_parser("figures", help="Rebuild figures from a results file.")
    figure_parser.add_argument("--results", default="results/results.json")
    figure_parser.add_argument("--output-dir", default="results/figures")

    return parser


def _run(args: argparse.Namespace) -> int:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    config = ExperimentConfig(epochs=args.epochs, seeds=args.seeds)
    output_dir = Path(args.output_dir)
    print(
        f"Running {config.n_runs} trainings "
        f"({len(config.widths) - 1} layers, widths {config.widths}, "
        f"{config.epochs} epochs, {config.seeds} seeds)\n"
    )

    results = run_experiment(config, progress=not args.quiet)
    save_results(results, output_dir / "results.json")
    summary = summarize(results)
    write_summary_csv(summary, output_dir / "activation_summary.csv")
    tests = statistical_tests(results)

    print("\nAggregate results")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nOne-way ANOVA: F = {tests['anova_F']:.4f}, p = {tests['anova_p']:.4f}  |  "
        f"Kruskal-Wallis: H = {tests['kruskal_H']:.4f}, p = {tests['kruskal_p']:.4f}"
    )
    print(f"Wrote {output_dir / 'results.json'} and {output_dir / 'activation_summary.csv'}")

    if not args.no_figures:
        paths = make_all_figures(results, output_dir / "figures")
        for name, path in paths.items():
            print(f"Wrote {name}: {path}")
    return 0


def _figures(args: argparse.Namespace) -> int:
    records = load_results(args.results)
    paths = make_all_figures(records, args.output_dir)
    for name, path in paths.items():
        print(f"Wrote {name}: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    if raw[0] not in {"run", "figures", "--version"}:
        raw = ["run", *raw]
    args = build_parser().parse_args(raw)
    if args.command == "figures":
        return _figures(args)
    return _run(args)