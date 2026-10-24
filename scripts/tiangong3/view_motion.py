#!/usr/bin/env python3
"""Kinematically replay a reference clip (no policy, no physics).

Only Tiangong 3 clips with the exact 31-joint ordering are accepted.

  uv run python scripts/tiangong3/view_motion.py walk1_subject1
  uv run python scripts/tiangong3/view_motion.py data/npz/tiangong3/run1_subject2.npz --speed 0.5
  uv run python scripts/tiangong3/view_motion.py data/npz/tiangong3/run1_subject5.npz

Native viewer keys: space pause/resume, left/right arrow step one frame while paused.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from tiangong_supermimic.robot import TIANGONG3_XML

NPZ_DIR = Path(__file__).resolve().parents[2] / "data" / "npz" / "tiangong3"


def hinge_names(model: mujoco.MjModel) -> set[str]:
    return {
        model.joint(i).name
        for i in range(model.njnt)
        if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    }


def validate_joint_names(model: mujoco.MjModel, joint_names: list[str]) -> None:
    expected = hinge_names(model)
    if expected != set(joint_names) or len(joint_names) != len(expected):
        raise ValueError("clip joint_names do not match the Tiangong 3 model")


def load_clip(path: Path) -> tuple[dict[str, np.ndarray], list[str], float]:
    with np.load(path) as clip:
        joint_names = list(clip["joint_names"].astype(str))
        root = list(clip["body_names"].astype(str)).index("pelvis")
        arrays = {
            "root_pos": clip["body_pos_w"][:, root].astype(np.float64),
            "root_quat": clip["body_quat_w"][:, root].astype(np.float64),
            "joint_pos": clip["joint_pos"].astype(np.float64),
        }
        return arrays, joint_names, float(clip["fps"][0])


def to_qpos(
    model: mujoco.MjModel, clip: dict[str, np.ndarray], joint_names: list[str]
) -> np.ndarray:
    qpos = np.zeros((len(clip["root_pos"]), model.nq))
    qpos[:, 0:3] = clip["root_pos"]
    qpos[:, 3:7] = clip["root_quat"]
    for column, name in enumerate(joint_names):
        qpos[:, model.joint(name).qposadr[0]] = clip["joint_pos"][:, column]
    return qpos


def build_scene(robot_xml: Path) -> mujoco.MjModel:
    """Robot MJCF plus the ground, sky and light the training scene adds around it."""
    spec = mujoco.MjSpec.from_file(str(robot_xml))
    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="groundplane",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.2, 0.3, 0.4],
        rgb2=[0.1, 0.2, 0.3],
        markrgb=[0.8, 0.8, 0.8],
        width=300,
        height=300,
    )
    material = spec.add_material(
        name="groundplane", texuniform=True, texrepeat=[4.0, 4.0], reflectance=0.2
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "groundplane"
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.0, 0.0, 0.05],
        material="groundplane",
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        name="sun",
        pos=[0.0, 0.0, 3.0],
        dir=[0.0, 0.0, -1.0],
        type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    return spec.compile()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", help="clip name under data/npz/tiangong3 or a path to an .npz")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    args = parser.parse_args()

    path = Path(args.clip)
    if not path.exists():
        path = NPZ_DIR / f"{args.clip.removesuffix('.npz')}.npz"
    clip, joint_names, fps = load_clip(path)
    model = build_scene(TIANGONG3_XML)
    validate_joint_names(model, joint_names)
    qpos = to_qpos(model, clip, joint_names)
    data = mujoco.MjData(model)
    print(f"{path.stem} (tiangong3): {len(qpos)} frames, {len(qpos) / fps:.1f} s @ {fps:.0f} Hz")

    state = {"frame": int(args.start * fps), "paused": False}

    def on_key(key: int) -> None:
        if key == ord(" "):
            state["paused"] = not state["paused"]
        elif key == 262:  # right arrow
            state["frame"] = min(state["frame"] + 1, len(qpos) - 1)
        elif key == 263:  # left arrow
            state["frame"] = max(state["frame"] - 1, 0)

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.trackbodyid = model.body("pelvis").id
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.distance = 3.0
        dt = 1.0 / (fps * args.speed)
        while viewer.is_running():
            t0 = time.perf_counter()
            data.qpos[:] = qpos[state["frame"]]
            mujoco.mj_kinematics(model, data)
            viewer.sync()
            if not state["paused"]:
                state["frame"] = (state["frame"] + 1) % len(qpos)
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
