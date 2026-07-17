#!/usr/bin/env python3
"""Build once and run the runtime-grid convergence assertions."""

from __future__ import annotations

import argparse
import pathlib
import subprocess


def run(command: list[str], cwd: pathlib.Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Configure, build, and run the solver convergence tests."
    )
    parser.add_argument(
        "--build-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/nsb-convergence"),
    )
    parser.add_argument(
        "--mode",
        choices=("spatial", "temporal", "all"),
        default="all",
        help="study to run (default: all)",
    )
    parser.add_argument("--build-type", default="Release")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print errors and measured convergence orders",
    )
    args = parser.parse_args()

    source_dir = pathlib.Path(__file__).resolve().parent
    build_dir = args.build_dir.resolve()
    run(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_BUILD_TYPE={args.build_type}",
        ],
        source_dir,
    )
    run(
        ["cmake", "--build", str(build_dir), "--target", "test_convergence", "--parallel", "4"],
        source_dir,
    )

    modes = ("spatial", "temporal") if args.mode == "all" else (args.mode,)
    for mode in modes:
        command = [str(build_dir / "test_convergence"), f"--{mode}"]
        if args.verbose:
            command.append("--verbose")
        run(command, source_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
