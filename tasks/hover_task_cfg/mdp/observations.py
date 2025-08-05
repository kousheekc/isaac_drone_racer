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


def root_pos_t(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the target frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_w = env.command_manager.get_term(command_name).command
    else:
        target_w = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    drone_pos_w = asset.data.root_pos_w
    drone_att_w = asset.data.root_quat_w

    log(env, ["px", "py", "pz"], drone_pos_w)

    drone_pos_t, _ = math_utils.subtract_frame_transforms(target_w[:, :3], target_w[:, 3:], drone_pos_w, drone_att_w)

    return drone_pos_t


def root_rotmat_t(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root attitude (3x3 flattened rotation matrix) in the target frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_w = env.command_manager.get_term(command_name).command
    else:
        target_w = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    drone_pos_w = asset.data.root_pos_w
    drone_att_w = asset.data.root_quat_w

    log(env, ["qw", "qx", "qy", "qz"], drone_att_w)

    _, drone_quat_t = math_utils.subtract_frame_transforms(target_w[:, :3], target_w[:, 3:], drone_pos_w, drone_att_w)
    drone_rotmat_t = math_utils.matrix_from_quat(drone_quat_t)
    drone_flat_rotmat_t = drone_rotmat_t.view(-1, 9)

    return drone_flat_rotmat_t
