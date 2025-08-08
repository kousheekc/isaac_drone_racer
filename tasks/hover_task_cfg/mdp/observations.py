# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of target in body frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos_tensor = env.command_manager.get_term(command_name).command[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    drone_pos_w = asset.data.root_pos_w
    drone_att_w = asset.data.root_quat_w

    log(env, ["px", "py", "pz"], drone_pos_w)
    log(env, ["qw", "qx", "qy", "qz"], drone_att_w)

    pos_b, _ = math_utils.subtract_frame_transforms(drone_pos_w, drone_att_w, target_pos_tensor)

    return pos_b


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_lin_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    log(env, ["vx", "vy", "vz"], lin_vel)
    return lin_vel


def root_ang_vel_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the body frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel = asset.data.root_ang_vel_b
    log(env, ["wx", "wy", "wz"], ang_vel)
    return ang_vel
