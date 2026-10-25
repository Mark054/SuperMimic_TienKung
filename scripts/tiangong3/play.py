"""Watch a trained Tiangong 3 policy track retargeted reference clips.

The translucent robot is the reference pose and the solid robot is the policy.
Native viewer keys: space pause, right arrow step, +/- speed, backspace reset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from tiangong_supermimic.envs import TianGongEnvCfgArgs, make_env
from tiangong_supermimic.envs.mdp.motion_library import discover_npz
from tiangong_supermimic.eval.benchmark import load_policy_from_checkpoint
from tiangong_supermimic.robot import FOOT_GEOM_REGEX, CollisionMode


class _DictPolicy:
    def __init__(self, policy):
        self.policy = policy

    @torch.no_grad()
    def __call__(self, obs):
        return self.policy(obs["policy"], obs["goal"])


def main(
    ckpt: Path,
    npz_dir: Path = Path("data/npz/tiangong3"),
    clips: list[str] | None = None,
    num_envs: int = 1,
    device: str = "cuda:0",
    collision: CollisionMode | None = None,
    viewer: Literal["auto", "native", "viser"] = "auto",
    ghost: bool = True,
    colliders: bool = False,
    foot_colliders: bool = False,
    sampling: Literal["start", "uniform"] = "start",
) -> None:
    files = [str(path) for path in discover_npz(npz_dir, names=clips)]
    if not files:
        raise FileNotFoundError(f"no Tiangong clips selected under {npz_dir}")
    policy, cfg = load_policy_from_checkpoint(ckpt, device)
    collision_mode = collision or cfg["env"].get("collision_mode", "none")
    args = TianGongEnvCfgArgs(
        npz_files=files,
        num_envs=num_envs,
        collision_mode=collision_mode,
        future_frames=cfg["env"]["future_frames"],
        episode_length_s=1.0e5,
        action_noise_std=0.0,
        obs_noise=False,
        reset_noise=False,
        domain_randomization=False,
        sampling_mode=sampling,
        record_transitions=False,
        seed=0,
        anchor_obs=cfg["env"].get("anchor_obs", False),
        debug_vis=ghost or colliders or foot_colliders,
        extra_command_kwargs={
            "show_reference_ghost": ghost,
            "show_colliders": colliders or foot_colliders,
            "collider_geom_regex": FOOT_GEOM_REGEX if foot_colliders else r".*_collision",
            "show_collider_primitives": foot_colliders,
        },
    )
    env = RslRlVecEnvWrapper(make_env(args, device=device))
    wrapped_policy = _DictPolicy(policy)
    if viewer == "auto":
        viewer = "native" if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) else "viser"
    print(
        f"[INFO] {len(files)} Tiangong clip(s), {num_envs} env(s), viewer={viewer}, "
        f"ghost={ghost}, colliders={colliders}, foot_colliders={foot_colliders}, "
        f"collision={collision_mode}"
    )
    try:
        if viewer == "native":
            NativeMujocoViewer(env, wrapped_policy).run()
        else:
            ViserPlayViewer(env, wrapped_policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    tyro.cli(main)
