#!/usr/bin/env python3
"""Build a URDF-derived Tiangong 3 standing clip for smoke training."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from tiangong_supermimic.data.motion_convert import convert_csv_to_npz
from tiangong_supermimic.robot import NUM_JOINTS, TIANGONG3_JOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/npz/tiangong3/standing.npz"))
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    frames = max(2, round(args.seconds * 50.0) + 1)
    row = np.zeros(7 + NUM_JOINTS, dtype=np.float64)
    row[2] = 1.05
    row[6] = 1.0  # input quaternion xyzw
    joint_index = {name: i for i, name in enumerate(TIANGONG3_JOINT_NAMES)}
    for side in ("l", "r"):
        row[7 + joint_index[f"hip_pitch_{side}_joint"]] = -0.1
        row[7 + joint_index[f"knee_pitch_{side}_joint"]] = 0.2
        row[7 + joint_index[f"ankle_pitch_{side}_joint"]] = -0.1
    motion = np.repeat(row[None, :], frames, axis=0)

    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        np.savetxt(tmp.name, motion, delimiter=",")
        summary = convert_csv_to_npz(tmp.name, args.output, input_fps=50.0)
    print(summary)


if __name__ == "__main__":
    main()
