#!/usr/bin/env python3
"""Compare two X-kernel benchmark executables with alternating runs."""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


RUN_PARAMETERS = {
    64: (3, 20),
    128: (2, 10),
}
METRICS = (
    "momentum_x_ns_per_cell",
    "pressure_x_ns_per_cell",
    "timestep_ns_per_cell",
)


@dataclass(frozen=True)
class Sample:
    executable: str
    workload: str
    extent: int
    values: dict[str, float]


def run_once(executable: Path, workload: str, extent: int) -> Sample:
    warmup, steps = RUN_PARAMETERS[extent]
    result = subprocess.run(
        [
            str(executable),
            "--grid",
            str(extent),
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--workload",
            workload,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = list(csv.DictReader(io.StringIO(result.stdout)))
    if len(rows) != 1:
        raise RuntimeError(f"unexpected output from {executable}: {result.stdout}")
    return Sample(
        executable=str(executable),
        workload=workload,
        extent=extent,
        values={metric: float(rows[0][metric]) for metric in METRICS},
    )


def relative_mad(values: list[float]) -> float:
    median = statistics.median(values)
    if median == 0.0:
        return 0.0
    return statistics.median(abs(value - median) for value in values) / median


def summarize(samples: list[Sample]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in METRICS:
        values = [sample.values[metric] for sample in samples]
        summary[metric] = statistics.median(values)
        summary[f"{metric}_relative_mad"] = relative_mad(values)
    return summary


def improvement(baseline: float, candidate: float) -> float:
    return 100.0 * (baseline - candidate) / baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure median X-kernel ns/cell for a frozen baseline and candidate."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("momentum", "pressure"), required=True
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("--repeats must be at least 3")

    raw_rows: list[dict[str, str | int | float]] = []
    summaries: dict[tuple[str, int, str], dict[str, float]] = {}
    for workload in ("paper", "synthetic"):
        for extent in (64, 128):
            groups = {"baseline": [], "candidate": []}
            for repeat in range(args.repeats):
                order = (
                    (("baseline", args.baseline), ("candidate", args.candidate))
                    if repeat % 2 == 0
                    else (("candidate", args.candidate), ("baseline", args.baseline))
                )
                for label, executable in order:
                    sample = run_once(executable, workload, extent)
                    groups[label].append(sample)
                    raw_rows.append(
                        {
                            "label": label,
                            "repeat": repeat,
                            "workload": workload,
                            "extent": extent,
                            **sample.values,
                        }
                    )
            for label, samples in groups.items():
                summaries[(workload, extent, label)] = summarize(samples)

    if args.csv:
        with args.csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=raw_rows[0].keys())
            writer.writeheader()
            writer.writerows(raw_rows)

    target_metric = f"{args.target}_x_ns_per_cell"
    passed = True
    print(
        "workload extent momentum_x_base momentum_x_candidate momentum_% "
        "pressure_x_base pressure_x_candidate pressure_% total_base "
        "total_candidate total_% max_rmad_%"
    )
    for workload in ("paper", "synthetic"):
        for extent in (64, 128):
            baseline = summaries[(workload, extent, "baseline")]
            candidate = summaries[(workload, extent, "candidate")]
            improvements = {
                metric: improvement(baseline[metric], candidate[metric])
                for metric in METRICS
            }
            max_rmad = 100.0 * max(
                baseline[f"{metric}_relative_mad"]
                for metric in METRICS
            )
            max_rmad = max(
                max_rmad,
                100.0 * max(
                    candidate[f"{metric}_relative_mad"]
                    for metric in METRICS
                ),
            )
            print(
                f"{workload:9s} {extent:6d} "
                f"{baseline['momentum_x_ns_per_cell']:15.6f} "
                f"{candidate['momentum_x_ns_per_cell']:20.6f} "
                f"{improvements['momentum_x_ns_per_cell']:10.3f} "
                f"{baseline['pressure_x_ns_per_cell']:15.6f} "
                f"{candidate['pressure_x_ns_per_cell']:20.6f} "
                f"{improvements['pressure_x_ns_per_cell']:10.3f} "
                f"{baseline['timestep_ns_per_cell']:10.6f} "
                f"{candidate['timestep_ns_per_cell']:15.6f} "
                f"{improvements['timestep_ns_per_cell']:8.3f} {max_rmad:10.3f}"
            )
            if max_rmad > 3.0:
                passed = False
            if extent == 128:
                if improvements[target_metric] < 15.0:
                    passed = False
                if (
                    workload == "paper"
                    and improvements["timestep_ns_per_cell"] < 5.0
                ):
                    passed = False
            if extent == 64 and (
                improvements[target_metric] < -2.0 or
                improvements["timestep_ns_per_cell"] < -2.0
            ):
                passed = False

    print(f"gate={'PASS' if passed else 'FAIL'} target={args.target}")
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise
