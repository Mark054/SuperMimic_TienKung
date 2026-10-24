#!/usr/bin/env python3
"""Copy reference clips for training, dropping whole clips whose name matches ``--exclude``.

The LAFAN1 obstacle clips climb onto boxes that the flat training plane does not have, so
they are excluded by default.  Kept clips are re-aligned so the feet rest on the floor with the
current foot colliders, and get a per-frame ``reset_lift`` so resets never start with a collider
of any collision mode below the floor.

  uv run python scripts/tiangong3/filter_clips.py \\
    --input /path/to/data/npz/tiangong3 --output data/npz/tiangong3
"""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatch
from pathlib import Path

import numpy as np

from tiangong_supermimic.data.motion_convert import collider_floor_lift, lowest_foot_height
from tiangong_supermimic.robot import load_tiangong3_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/npz/tiangong3"))
    parser.add_argument("--exclude", nargs="*", default=["obstacles*"], help="clip name globs")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no npz clips under {args.input}")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("--output must differ from --input")
    args.output.mkdir(parents=True, exist_ok=True)
    models = [load_tiangong3_model("mesh"), load_tiangong3_model("simplified")]

    report = []
    total_in = total_out = 0
    for source in files:
        with np.load(source) as data:
            clip = {key: data[key] for key in data.files}
        frames = len(clip["joint_pos"])
        total_in += frames
        kept = not any(fnmatch(source.stem, pattern) for pattern in args.exclude)
        report.append((source.stem, frames / float(clip["fps"][0]), kept))
        if not kept:
            continue
        shift = -float(
            np.percentile(
                lowest_foot_height(models[0], clip["body_pos_w"], clip["body_quat_w"]), 1.0
            )
        )
        clip["body_pos_w"][:, :, 2] += shift
        clip["ground_offset"] = clip["ground_offset"] + shift
        root = [str(n) for n in clip["body_names"]].index("pelvis")
        clip["reset_lift"] = np.max(
            [
                collider_floor_lift(
                    m, clip["body_pos_w"][:, root], clip["body_quat_w"][:, root], clip["joint_pos"]
                )
                for m in models
            ],
            axis=0,
        ).astype(np.float32)
        print(
            f"{source.stem}: ground shift {100 * shift:+.2f} cm, "
            f"frame0 lift {100 * clip['reset_lift'][0]:.2f} cm",
            flush=True,
        )
        np.savez(args.output / f"{source.stem}.npz", **clip)
        total_out += frames

    with open(args.output / "clips.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("name", "seconds", "kept"))
        writer.writerows(report)
    dropped = [name for name, _, kept in report if not kept]
    print(f"dropped {len(dropped)}: {' '.join(dropped)}")
    print(
        f"{len(files)} source clips, {total_in / 50 / 3600:.2f} h -> {len(files) - len(dropped)} "
        f"clips, {total_out / 50 / 3600:.2f} h ({100 * total_out / total_in:.1f}%)"
    )


if __name__ == "__main__":
    main()
