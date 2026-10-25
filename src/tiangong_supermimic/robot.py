"""Tiangong 3 robot asset and actuator configuration.

This module is intentionally independent from the Unitree G1 asset/configuration.
The URDF provides effort limits but no identified servo gains or rotor inertias.
The position actuators retain effort-matched stiffness and add reflected inertia
and damping for a stable 10 Hz closed-loop response.
"""

from __future__ import annotations

from math import pi
from pathlib import Path
from typing import Literal

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIANGONG3_MESH_XML = PROJECT_ROOT / "assets" / "tiangong3" / "tiangong3.xml"
TIANGONG3_SIMPLIFIED_XML = PROJECT_ROOT / "assets" / "tiangong3" / "tiangong3_simplified.xml"
TIANGONG3_XML = TIANGONG3_MESH_XML
CollisionMode = Literal["none", "simplified", "mesh"]

TIANGONG3_JOINT_NAMES: tuple[str, ...] = (
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
NUM_JOINTS = len(TIANGONG3_JOINT_NAMES)

ROOT_BODY = "pelvis"
ANCHOR_BODY = "waist_pitch_link"
FOOT_GEOM_REGEX = r"^ankle_roll_[lr]_link_rail[12]_collision$"
# Conservative visual-envelope colliders overlap nearby bodies in the same kinematic
# assembly. Exclude only structural neighbors; contacts between separate limbs remain active.
SELF_COLLISION_EXCLUDES: tuple[tuple[str, str], ...] = (
    ("pelvis", "waist_roll_link"),
    ("pelvis", "waist_pitch_link"),
    ("waist_yaw_link", "waist_pitch_link"),
    ("hip_pitch_l_link", "hip_yaw_l_link"),
    ("hip_pitch_r_link", "hip_yaw_r_link"),
    ("pelvis", "hip_roll_l_link"),
    ("pelvis", "hip_roll_r_link"),
    ("pelvis", "hip_yaw_l_link"),
    ("pelvis", "hip_yaw_r_link"),
    ("waist_yaw_link", "hip_pitch_l_link"),
    ("waist_yaw_link", "hip_pitch_r_link"),
    ("waist_pitch_link", "hip_pitch_l_link"),
    ("waist_pitch_link", "hip_pitch_r_link"),
    ("waist_pitch_link", "shoulder_roll_l_link"),
    ("waist_pitch_link", "shoulder_roll_r_link"),
    ("shoulder_pitch_l_link", "shoulder_yaw_l_link"),
    ("shoulder_pitch_r_link", "shoulder_yaw_r_link"),
    ("waist_pitch_link", "head_pitch_link"),
    ("knee_pitch_l_link", "ankle_roll_l_link"),
    ("knee_pitch_r_link", "ankle_roll_r_link"),
)
TRACKED_BODIES: tuple[str, ...] = (
    "pelvis",
    "hip_roll_l_link",
    "knee_pitch_l_link",
    "ankle_roll_l_link",
    "hip_roll_r_link",
    "knee_pitch_r_link",
    "ankle_roll_r_link",
    "waist_pitch_link",
    "head_pitch_link",
    "shoulder_roll_l_link",
    "elbow_pitch_l_link",
    "wrist_roll_l_link",
    "shoulder_roll_r_link",
    "elbow_pitch_r_link",
    "wrist_roll_r_link",
)


def _model_path(collision_mode: CollisionMode) -> Path:
    return TIANGONG3_MESH_XML if collision_mode == "mesh" else TIANGONG3_SIMPLIFIED_XML


def get_spec(collision_mode: CollisionMode = "mesh") -> mujoco.MjSpec:
    path = _model_path(collision_mode)
    if not path.exists():
        raise FileNotFoundError(path)
    spec = mujoco.MjSpec.from_file(str(path))
    for body1, body2 in SELF_COLLISION_EXCLUDES:
        spec.add_exclude(bodyname1=body1, bodyname2=body2)
    return spec


def _none_spec() -> mujoco.MjSpec:
    return get_spec("none")


def _simplified_spec() -> mujoco.MjSpec:
    return get_spec("simplified")


def _mesh_spec() -> mujoco.MjSpec:
    return get_spec("mesh")


def load_tiangong3_model(collision_mode: CollisionMode = "mesh") -> mujoco.MjModel:
    model = get_spec(collision_mode).compile()
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, model.njnt)
    )
    if names != TIANGONG3_JOINT_NAMES:
        raise ValueError(f"Tiangong joint order mismatch: {names}")
    if model.nq != 7 + NUM_JOINTS or model.nv != 6 + NUM_JOINTS:
        raise ValueError(f"unexpected Tiangong dimensions: nq={model.nq}, nv={model.nv}")
    return model


TIANGONG3_CONTROL_FREQUENCY = 10.0
TIANGONG3_DAMPING_RATIO = 2.0


def _position_actuator(names: tuple[str, ...], effort: float) -> BuiltinPositionActuatorCfg:
    # The imported URDF has effectively zero rotor inertia.  Using its torque limit
    # directly as stiffness without reflected inertia makes the light distal joints
    # numerically unstable.  Match G1's 10 Hz, damping-ratio-2 actuator construction.
    omega = TIANGONG3_CONTROL_FREQUENCY * 2.0 * pi
    armature = effort / omega**2
    damping = 2.0 * TIANGONG3_DAMPING_RATIO * armature * omega
    return BuiltinPositionActuatorCfg(
        target_names_expr=names,
        stiffness=effort,
        damping=damping,
        effort_limit=effort,
        armature=armature,
        frictionloss=0.5 if effort >= 50.0 else 0.1,
    )


TIANGONG3_ACTUATORS = (
    _position_actuator(("knee_pitch_[lr]_joint",), 400.0),
    _position_actuator(("hip_(pitch|roll)_[lr]_joint",), 235.0),
    _position_actuator(("waist_yaw_joint",), 200.0),
    _position_actuator(("hip_yaw_[lr]_joint", "waist_(roll|pitch)_joint"), 150.0),
    _position_actuator(("shoulder_(pitch|roll)_[lr]_joint",), 90.0),
    _position_actuator(("ankle_(pitch|roll)_[lr]_joint",), 55.0),
    _position_actuator(("shoulder_yaw_[lr]_joint", "elbow_pitch_[lr]_joint"), 50.0),
    _position_actuator(("elbow_yaw_[lr]_joint", "wrist_(pitch|roll)_[lr]_joint"), 40.0),
    _position_actuator(("head_(yaw|pitch)_joint",), 6.3),
)

TIANGONG3_ARTICULATION = EntityArticulationInfoCfg(
    actuators=TIANGONG3_ACTUATORS,
    soft_joint_pos_limit_factor=0.9,
)

TIANGONG3_ACTION_SCALE: dict[str, float] = {}
for actuator in TIANGONG3_ACTUATORS:
    assert isinstance(actuator, BuiltinPositionActuatorCfg)
    assert actuator.effort_limit is not None
    for expression in actuator.target_names_expr:
        TIANGONG3_ACTION_SCALE[expression] = 0.25 * actuator.effort_limit / actuator.stiffness

def _collision_cfg(collision_mode: CollisionMode) -> CollisionCfg:
    # conaffinity=2 prevents robot-robot pairs (1 & 2 == 0) while preserving
    # robot-floor pairs through the floor's default conaffinity=1.
    conaffinity = 2 if collision_mode == "none" else 1
    return CollisionCfg(
        geom_names_expr=(".*_collision",),
        contype=1,
        conaffinity=conaffinity,
        condim={FOOT_GEOM_REGEX: 3, ".*_collision": 1},
        priority={FOOT_GEOM_REGEX: 1, ".*": 0},
        friction={FOOT_GEOM_REGEX: (0.6,)},
    )

TIANGONG3_HOME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.05),
    joint_pos={
        ".*": 0.0,
        "hip_pitch_[lr]_joint": -0.10,
        "knee_pitch_[lr]_joint": 0.20,
        "ankle_pitch_[lr]_joint": -0.10,
    },
    joint_vel={".*": 0.0},
)


def get_tiangong3_robot_cfg(collision_mode: CollisionMode = "mesh") -> EntityCfg:
    spec_fns = {
        "none": _none_spec,
        "simplified": _simplified_spec,
        "mesh": _mesh_spec,
    }
    return EntityCfg(
        init_state=TIANGONG3_HOME,
        collisions=(_collision_cfg(collision_mode),),
        spec_fn=spec_fns[collision_mode],
        articulation=TIANGONG3_ARTICULATION,
        sort_actuators=True,
    )
