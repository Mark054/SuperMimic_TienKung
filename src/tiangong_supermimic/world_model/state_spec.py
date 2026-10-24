"""Tiangong 3 world-model state and explicit integration.

State: root xyz/quaternion, 31 joint positions, root linear/angular velocity,
and 31 joint velocities (75 values).  The independent Tiangong dynamics head
predicts 74 velocity increments and integration corrections.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tiangong_supermimic.kinematics.rotations import (
    heading_quat,
    quat_apply,
    quat_apply_inverse,
    quat_canonical,
    quat_from_rotvec,
    quat_mul,
    quat_normalize,
    rot6d_from_quat,
    rotvec_from_quat,
)
from tiangong_supermimic.robot import NUM_JOINTS
from tiangong_supermimic.spec import NQ, STATE_DIM


@dataclass
class StateSpecCfg:
    dt: float = 0.02
    # Nominal error scales (D3) for [root pos m, root rotvec rad, root lin vel m/s, root ang vel rad/s, joint pos rad, joint vel rad/s].
    err_scale: tuple[float, ...] = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
    # Output scales of the dynamics head for [dv_h, dw_b, dqd, dp_h, dth_b, dq].
    out_scale: tuple[float, ...] = (0.3, 0.5, 2.0, 0.01, 0.02, 0.02)
    # Input feature scales for [root_z, rot6d, lin_vel_h, ang_vel_b, joint_pos, joint_vel] (divide by these).
    in_scale: tuple[float, ...] = (1.0, 1.0, 1.0, 2.0, 1.0, 5.0)
    explicit_integration: bool = True
    # xi normalisation: (offset, scale) for friction, encoder bias, com offset.
    xi_offset: tuple[float, ...] = field(
        default_factory=lambda: (0.75,) + (0.0,) * NUM_JOINTS + (0.0,) * 3
    )
    xi_scale: tuple[float, ...] = field(
        default_factory=lambda: (0.45,) + (0.01,) * NUM_JOINTS + (0.05,) * 3
    )


class StateSpec:
    def __init__(self, cfg: StateSpecCfg, device, dtype=torch.float32):
        self.cfg = cfg
        self.dt = cfg.dt
        rep = lambda scales, sizes: torch.cat(
            [torch.full((n,), float(s)) for s, n in zip(scales, sizes)]
        )
        self.err_scale = rep(cfg.err_scale, (3, 3, 3, 3, NUM_JOINTS, NUM_JOINTS)).to(
            device=device, dtype=dtype
        )
        self.out_scale = rep(cfg.out_scale, (3, 3, NUM_JOINTS, 3, 3, NUM_JOINTS)).to(
            device=device, dtype=dtype
        )
        self.in_scale = rep(cfg.in_scale, (1, 6, 3, 3, NUM_JOINTS, NUM_JOINTS)).to(
            device=device, dtype=dtype
        )
        self.xi_offset = torch.tensor(cfg.xi_offset, device=device, dtype=dtype)
        self.xi_scale = torch.tensor(cfg.xi_scale, device=device, dtype=dtype)

    # ------------------------------------------------------------------ split / features

    @staticmethod
    def split(s: torch.Tensor):
        return (
            s[..., 0:3],
            s[..., 3:7],
            s[..., 7:NQ],
            s[..., NQ : NQ + 3],
            s[..., NQ + 3 : NQ + 6],
            s[..., NQ + 6 : STATE_DIM],
        )

    def canonical(self, s: torch.Tensor) -> torch.Tensor:
        """Heading-invariant, scaled Tiangong network features."""
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        q_local = quat_canonical(quat_mul(quat_apply_inverse_quat(hq), q))
        v_h = quat_apply_inverse(hq, v)
        feat = torch.cat([p[..., 2:3], rot6d_from_quat(q_local), v_h, w, qj, qd], dim=-1)
        return feat / self.in_scale

    def normalize_xi(self, xi: torch.Tensor) -> torch.Tensor:
        return (xi - self.xi_offset) / self.xi_scale

    # ------------------------------------------------------------------ integration

    def integrate(self, s: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Apply the Tiangong dynamics correction to one state."""
        d = delta * self.out_scale
        j = NUM_JOINTS
        dv_h = d[..., 0:3]
        dw_b = d[..., 3:6]
        dqd = d[..., 6 : 6 + j]
        dp_h = d[..., 6 + j : 9 + j]
        dth_b = d[..., 9 + j : 12 + j]
        dq = d[..., 12 + j : 12 + 2 * j]
        p, q, qj, v, w, qd = self.split(s)
        q = quat_normalize(q)
        hq = heading_quat(q)
        v_new = v + quat_apply(hq, dv_h)
        w_new = w + dw_b
        qd_new = qd + dqd
        dt = self.dt if self.cfg.explicit_integration else 0.0
        p_new = p + dt * v_new + quat_apply(hq, dp_h)
        q_new = quat_normalize(quat_mul(q, quat_from_rotvec(dt * w_new + dth_b)))
        qj_new = qj + dt * qd_new + dq
        return torch.cat([p_new, q_new, qj_new, v_new, w_new, qd_new], dim=-1)

    # ------------------------------------------------------------------ errors

    def error_vector(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Normalised Tiangong state error vector."""
        p1, q1, qj1, v1, w1, qd1 = self.split(s_pred)
        p2, q2, qj2, v2, w2, qd2 = self.split(s_true)
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
        rot_err = rotvec_from_quat(quat_mul(quat_apply_inverse_quat(q2), q1))
        err = torch.cat([p1 - p2, rot_err, v1 - v2, w1 - w2, qj1 - qj2, qd1 - qd2], dim=-1)
        return err / self.err_scale

    def normalized_sq_error(self, s_pred: torch.Tensor, s_true: torch.Tensor) -> torch.Tensor:
        """Mean over dims of the squared normalised error (the reported 'prediction error')."""
        e = self.error_vector(s_pred, s_true)
        return (e * e).mean(-1)


def quat_apply_inverse_quat(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a unit quaternion (inverse)."""
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
