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


def root_quat_w(
    env: ManagerBasedRLEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame."""

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    log(env, ["qw", "qx", "qy", "qz"], quat)
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_rotmat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (3x3 flattened rotation matrix) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return flat_rotmat


def root_rotmat6d_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root orientation (6D rotation representation) in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    rotmat = math_utils.matrix_from_quat(quat)
    rotmat6d = rotmat[:, :2, :].reshape(-1, 6)
    # logging purposes
    flat_rotmat = rotmat.view(-1, 9)
    log(env, ["r11", "r12", "r13", "r21", "r22", "r23", "r31", "r32", "r33"], flat_rotmat)
    return rotmat6d


def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    position = asset.data.root_pos_w
    log(env, ["px", "py", "pz"], position)
    return position


def root_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    asset: RigidObject = env.scene[asset_cfg.name]

    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    drone_pose_w = asset.data.root_state_w[:, :7]  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    drone_pos_w = drone_pose_w[:, :3]
    drone_quat_w = drone_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = drone_pos_w - gate_pos_w
    drone_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    drone_quat_g = math_utils.quat_mul(gate_quat_w_inv, drone_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([drone_pos_g, drone_quat_g], dim=-1)

    return position


def next_gate_pose_g(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Asset root position in the gate frame."""
    gate_pose_w = env.command_manager.get_term(command_name).command  # (num_envs, 7)
    next_gate_pose_w = env.command_manager.get_term(command_name).next_gate  # (num_envs, 7)

    # Extract positions and quaternions
    gate_pos_w = gate_pose_w[:, :3]
    gate_quat_w = gate_pose_w[:, 3:7]
    next_gate_pos_w = next_gate_pose_w[:, :3]
    next_gate_quat_w = next_gate_pose_w[:, 3:7]

    # Compute drone pose in gate frame
    # Inverse gate quaternion
    gate_quat_w_inv = math_utils.quat_inv(gate_quat_w)

    # Position of drone in gate frame
    rel_pos = next_gate_pos_w - gate_pos_w
    next_gate_pos_g = math_utils.quat_rotate(gate_quat_w_inv, rel_pos)

    # Orientation of drone in gate frame
    next_gate_quat_g = math_utils.quat_mul(gate_quat_w_inv, next_gate_quat_w)

    # Concatenate position and quaternion
    position = torch.cat([next_gate_pos_g, next_gate_quat_g], dim=-1)

    return position


def target_pos_b(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position of targets in body frame.

    Returns:
        torch.Tensor: If command_name is used and returns multiple gates (num_envs, n, 7),
                     this returns flattened relative positions (num_envs, 3*n).
                     If target_pos is a list or single gate, returns (num_envs, 3).
    """

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_poses = env.command_manager.get_term(command_name).command  # Shape: (num_envs, n, 7)

        # Check if we have multiple gates (3D tensor) or single gate (2D tensor)
        if len(target_poses.shape) == 3:
            # Multiple gates: (num_envs, n, 7)
            num_envs, num_gates, _ = target_poses.shape

            # Prepare robot poses for broadcasting
            robot_pos_w = asset.data.root_pos_w  # (num_envs, 3)
            robot_quat_w = asset.data.root_quat_w  # (num_envs, 4)

            # Initialize output tensor
            all_pos_b = torch.zeros(num_envs, num_gates, 3, device=asset.device)

            # Compute relative position for each gate
            for i in range(num_gates):
                gate_poses = target_poses[:, i, :]  # (num_envs, 7)
                pos_b, _ = math_utils.subtract_frame_transforms(
                    robot_pos_w, robot_quat_w, gate_poses[:, :3], gate_poses[:, 3:7]
                )
                all_pos_b[:, i, :] = pos_b

            # Flatten to (num_envs, 3*num_gates)
            pos_b = all_pos_b.view(num_envs, -1)
        else:
            # Single gate: (num_envs, 7)
            pos_b, _ = math_utils.subtract_frame_transforms(
                asset.data.root_pos_w, asset.data.root_quat_w, target_poses[:, :3], target_poses[:, 3:7]
            )

    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )
        pos_b, _ = math_utils.subtract_frame_transforms(
            asset.data.root_pos_w, asset.data.root_quat_w, target_pos_tensor
        )

    return pos_b


def pos_error_w(
    env: ManagerBasedRLEnv,
    command_name: str | None = None,
    target_pos: list | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error in world frame."""

    asset: RigidObject = env.scene[asset_cfg.name]

    if target_pos is None:
        target_pos = env.command_manager.get_term(command_name).command[:, :3]
        target_pos_tensor = target_pos[:, :3]
    else:
        target_pos_tensor = (
            torch.tensor(target_pos, dtype=torch.float32, device=asset.device).repeat(env.num_envs, 1)
            + env.scene.env_origins
        )

    pos_error_w = target_pos_tensor - asset.data.root_pos_w
    return pos_error_w
