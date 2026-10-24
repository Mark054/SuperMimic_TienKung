"""Independent Tiangong 3 training configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tiangong_supermimic.imagination.policy_trainer import PolicyTrainerCfg
from tiangong_supermimic.policy.nets import CriticCfg, PolicyCfg
from tiangong_supermimic.robot import CollisionMode
from tiangong_supermimic.world_model.wm_trainer import WMTrainerCfg
from tiangong_supermimic.world_model.world_model import WorldModelCfg


@dataclass
class DataCfg:
    npz_dir: Path = Path("data/npz/tiangong3")
    clips: list[str] | None = None  # None = all npz in npz_dir
    limit: int | None = None
    manifest: Path | None = None
    library: Path | None = None


@dataclass
class EnvCfg:
    # 512 is the safe single-RTX-4090 default for Tiangong's larger contact model.
    num_envs: int = 512
    collision_mode: CollisionMode = "none"
    episode_length_s: float = 10.0
    action_noise_std: float = 0.1
    obs_noise: bool = True
    reset_noise: bool = True
    domain_randomization: bool = True
    sampling_mode: str = "return"
    replay_capacity: int = 512
    future_frames: int = 10
    compact_library: bool = False
    anchor_obs: bool = False


@dataclass
class ScheduleCfg:
    total_iterations: int = 320_000  # env steps (plan D12 reading A)
    updates_per_env_step: float = 0.4  # UTD
    policy_updates_per_wm_update: int = 1
    warmup_iterations: int = 40  # env steps before the first update (fills k-step windows)
    lr_final_ratio: float = 0.03
    lr_decay_fraction: float = 250_000 / 320_000
    log_interval: int = 50  # iterations
    checkpoint_interval: int = 4000  # iterations
    max_hours: float | None = None


@dataclass
class TrainConfig:
    run_name: str = "tiangong3"
    log_root: Path = Path("logs/tiangong3")
    seed: int = 0
    device: str = "cuda:0"
    data: DataCfg = field(default_factory=DataCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    world_model: WorldModelCfg = field(default_factory=WorldModelCfg)
    wm_trainer: WMTrainerCfg = field(default_factory=WMTrainerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    critic: CriticCfg = field(default_factory=CriticCfg)
    policy_trainer: PolicyTrainerCfg = field(default_factory=PolicyTrainerCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    resume: Path | None = None
