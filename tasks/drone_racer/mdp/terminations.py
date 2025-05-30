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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def flip(env: ManagerBasedRLEnv, angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate when the asset roll or pitch is more that angle threshold"""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    current_angle = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    current_angle_wrapped_abs = [torch.abs(math_utils.wrap_to_pi(angle)) for angle in current_angle]
    threshold_rad = torch.tensor(angle * (torch.pi / 180.0), device=env.device)
    angle_exceeds_threshold = (current_angle_wrapped_abs[0] > threshold_rad) | (
        current_angle_wrapped_abs[1] > threshold_rad
    )
    return angle_exceeds_threshold


def flyaway(
    env: ManagerBasedRLEnv,
    distance: float,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset's is too far away from the target position."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    # Compute distance
    distance_tensor = torch.linalg.norm(asset.data.root_pos_w - target_pos_tensor, dim=1)
    return distance_tensor > distance


def missed_gate(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
) -> torch.Tensor:
    """Terminate when the robot misses a gate."""
    return env.command_manager.get_term(command_name).gate_missed


def out_of_bounds(
    env: ManagerBasedRLEnv,
    x_range: tuple | None = None,
    y_range: tuple | None = None,
    z_range: tuple | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset is out of the specified bounds."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if x_range is not None:
        x_exceeds = (asset.data.root_pos_w[:, 0] < x_range[0]) | (asset.data.root_pos_w[:, 0] > x_range[1])
    if y_range is not None:
        y_exceeds = (asset.data.root_pos_w[:, 1] < y_range[0]) | (asset.data.root_pos_w[:, 1] > y_range[1])
    if z_range is not None:
        z_exceeds = (asset.data.root_pos_w[:, 2] < z_range[0]) | (asset.data.root_pos_w[:, 2] > z_range[1])

    return x_exceeds | y_exceeds | z_exceeds
