#!/usr/bin/env python3
"""Batch-retarget original LAFAN1 BVH clips directly to Tiangong 3 CSV files.

Run this script with the GMR virtualenv.  It registers the Tiangong 3 MJCF and
constructs a LAFAN1 IK configuration from GMR's proven G1 mapping while replacing
all target frames with Tiangong link names.  Output rows are root xyz, root
quaternion xyzw, then the 31 Tiangong joints in MuJoCo order.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GMR_ROOT = Path(os.environ.get("GMR_ROOT", PROJECT_ROOT.parent / "GMR")).expanduser().resolve()
ROBOT_XML = PROJECT_ROOT / "assets/tiangong3/tiangong3.xml"
BASE_IK_CONFIG = GMR_ROOT / "general_motion_retargeting/ik_configs/bvh_lafan1_to_g1.json"

TIANGONG3_JOINT_NAMES = (
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "elbow_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
)

FOREARM_LINK_PREFIXES = ("elbow_pitch_", "wrist_roll_")
FOREARM_AXIS_FIX = Rotation.from_euler("y", -90, degrees=True)

BODY_REMAP = {
    "pelvis": "pelvis",
    "left_hip_yaw_link": "hip_yaw_l_link",
    "left_knee_link": "knee_pitch_l_link",
    "left_ankle_roll_link": "ankle_roll_l_link",
    "right_hip_yaw_link": "hip_yaw_r_link",
    "right_knee_link": "knee_pitch_r_link",
    "right_ankle_roll_link": "ankle_roll_r_link",
    "torso_link": "waist_pitch_link",
    "left_shoulder_yaw_link": "shoulder_yaw_l_link",
    "left_elbow_link": "elbow_pitch_l_link",
    "left_wrist_yaw_link": "wrist_roll_l_link",
    "right_shoulder_yaw_link": "shoulder_yaw_r_link",
    "right_elbow_link": "elbow_pitch_r_link",
    "right_wrist_yaw_link": "wrist_roll_r_link",
}


def build_ik_config(path: Path) -> None:
    config = json.loads(BASE_IK_CONFIG.read_text())
    config["robot_root_name"] = "pelvis"
    config["ground_height"] = 0.0
    for name in (
        "Hips",
        "Spine2",
        "LeftUpLeg",
        "RightUpLeg",
        "LeftLeg",
        "RightLeg",
        "LeftFootMod",
        "RightFootMod",
    ):
        config["human_scale_table"][name] = 0.95
    for name in ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"):
        config["human_scale_table"][name] = 0.8
    for table_name in ("ik_match_table1", "ik_match_table2"):
        old = config[table_name]
        config[table_name] = {BODY_REMAP[target]: entry for target, entry in old.items()}
        # G1's forearm/hand links point along local +x; Tiangong's point along local -z.
        for link, entry in config[table_name].items():
            if link.startswith(FOREARM_LINK_PREFIXES):
                offset = Rotation.from_quat(entry[4], scalar_first=True) * FOREARM_AXIS_FIX
                entry[4] = offset.as_quat(scalar_first=True).tolist()
    path.write_text(json.dumps(config, indent=2))


def register_tiangong(config_path: Path) -> None:
    sys.path.insert(0, str(GMR_ROOT))
    from general_motion_retargeting import params

    params.ROBOT_XML_DICT["tiangong3"] = ROBOT_XML
    params.IK_CONFIG_DICT["bvh_lafan1"]["tiangong3"] = config_path


def retarget_one(job: tuple[str, str, str, int | None]) -> dict:
    source_s, destination_s, config_s, max_frames = job
    source = Path(source_s)
    destination = Path(destination_s)
    config_path = Path(config_s)
    try:
        register_tiangong(config_path)
        import mujoco
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        frames, human_height = load_bvh_file(str(source), format="lafan1")
        if max_frames is not None:
            frames = frames[:max_frames]
        retargeter = GeneralMotionRetargeting(
            src_human="bvh_lafan1",
            tgt_robot="tiangong3",
            actual_human_height=human_height,
            verbose=False,
        )
        model = retargeter.model
        joint_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
            for index in range(1, model.njnt)
        )
        if joint_names != TIANGONG3_JOINT_NAMES:
            raise ValueError(f"Tiangong joint order mismatch: {joint_names}")

        output = np.empty((len(frames), 7 + len(TIANGONG3_JOINT_NAMES)), dtype=np.float64)
        for index, frame in enumerate(frames):
            qpos = retargeter.retarget(copy.deepcopy(frame))
            output[index, :3] = qpos[:3]
            output[index, 3:7] = qpos[3:7][[1, 2, 3, 0]]
            output[index, 7:] = qpos[7:]
        if not np.isfinite(output).all():
            raise ValueError("retargeted motion contains non-finite values")

        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".csv.tmp")
        np.savetxt(temp, output, delimiter=",", fmt="%.8f")
        temp.replace(destination)
        return {"name": source.stem, "frames": len(frames), "ok": True, "error": ""}
    except Exception as exc:  # noqa: BLE001 - worker must report per-file failures
        return {"name": source.stem, "frames": 0, "ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=GMR_ROOT / "source_motions/lafan1")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw/lafan1/tiangong3")
    parser.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input.glob("*.bvh"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no BVH files under {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="tiangong3_gmr_") as temp_dir:
        config_path = Path(temp_dir) / "bvh_lafan1_to_tiangong3.json"
        build_ik_config(config_path)
        jobs = []
        for source in files:
            destination = args.output / f"{source.stem}.csv"
            if destination.exists() and not args.overwrite:
                continue
            jobs.append((str(source), str(destination), str(config_path), args.max_frames))
        print(f"{len(files)} BVH clips, {len(jobs)} to retarget with {args.workers} workers")
        started = time.time()
        results = []
        if jobs:
            with mp.Pool(args.workers) as pool:
                for index, result in enumerate(pool.imap_unordered(retarget_one, jobs), 1):
                    results.append(result)
                    status = "ok" if result["ok"] else result["error"]
                    print(f"[{index}/{len(jobs)}] {result['name']}: {status}", flush=True)
        failures = [result for result in results if not result["ok"]]
        if failures:
            raise RuntimeError(f"{len(failures)} clips failed: {failures[:5]}")
        print(f"completed in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
