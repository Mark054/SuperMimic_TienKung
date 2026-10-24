#!/usr/bin/env python3
"""Convert Tiangong 3 CSV motion files into isolated training clips."""

from __future__ import annotations

import argparse
from pathlib import Path

from tiangong_supermimic.data.motion_convert import convert_csv_to_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="CSV file or directory of CSV files")
    parser.add_argument("--output", type=Path, default=Path("data/npz/tiangong3"))
    parser.add_argument("--input-fps", type=float, required=True)
    parser.add_argument("--output-fps", type=float, default=50.0)
    args = parser.parse_args()

    files = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not files:
        raise FileNotFoundError(f"no CSV files under {args.input}")
    for source in files:
        destination = args.output / f"{source.stem}.npz" if args.input.is_dir() else args.output
        if destination.suffix != ".npz":
            destination = destination / f"{source.stem}.npz"
        summary = convert_csv_to_npz(source, destination, args.input_fps, args.output_fps)
        print(summary)


if __name__ == "__main__":
    main()
