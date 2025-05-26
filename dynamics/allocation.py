# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


def build_allocation_matrix(num_envs, arm_length, phi_deg, tilt_deg, thrust_coef, drag_coef):
    """
    Build actuator allocation matrices for multiple environments, assuming all share the same parameters.

    Args:
        num_envs     : int, number of environments
        arm_length   : float, length of arm
        phi_deg      : list[float] of size 4, motor angle positions (degrees)
        tilt_deg     : list[float] of size 4, motor tilt angles (degrees)
        thrust_coef  : float, thrust coefficient
        drag_coef    : float, drag torque coefficient

    Returns:
        A : (num_envs, 6, 4) allocation matrices
    """
    assert len(phi_deg) == 4 and len(tilt_deg) == 4, "Expected 4 motors"

    # Build identical tensors for each environment
    phi_deg_tensor = torch.tensor(phi_deg).repeat(num_envs, 1)  # (num_envs, 4)
    tilt_deg_tensor = torch.tensor(tilt_deg).repeat(num_envs, 1)  # (num_envs, 4)
    arm_length_tensor = torch.tensor([arm_length] * num_envs)  # (num_envs,)
    thrust_coef_tensor = torch.tensor([thrust_coef] * num_envs)  # (num_envs,)
    drag_coef_tensor = torch.tensor([drag_coef] * num_envs)  # (num_envs,)

    # Assume quadrotor with spin_dir [+1, -1, +1, -1] for CW/CCW
    spin_dir_tensor = torch.tensor([1, -1, 1, -1]).repeat(num_envs, 1)  # (num_envs, 4)

    return _build_allocation_matrix_from_tensors(
        num_envs,
        arm_length_tensor,
        phi_deg_tensor,
        tilt_deg_tensor,
        spin_dir_tensor,
        thrust_coef_tensor,
        drag_coef_tensor,
    )


def _build_allocation_matrix_from_tensors(num_envs, arm_length, phi_deg, tilt_deg, spin_dir, thrust_coef, drag_coef):
    phi = torch.deg2rad(phi_deg)
    tilt = torch.deg2rad(tilt_deg)

    # sin/cos
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_tilt = torch.sin(tilt)
    cos_tilt = torch.cos(tilt)

    # Position vectors
    r_x = arm_length.view(num_envs, 1) * cos_phi
    r_y = arm_length.view(num_envs, 1) * sin_phi
    r_z = torch.zeros_like(r_x)
    r = torch.stack([r_x, r_y, r_z], dim=2)

    # Thrust directions
    z_x = sin_tilt * cos_phi
    z_y = sin_tilt * sin_phi
    z_z = cos_tilt
    z = torch.stack([z_x, z_y, z_z], dim=2)

    # Thrust forces
    force = thrust_coef.view(num_envs, 1, 1) * z

    # Torques
    tau_pos = torch.cross(r, force, dim=2)

    # Torque from rotor drag
    tau_drag = (spin_dir * drag_coef.view(num_envs, 1))[:, :, None] * z

    # Total torque
    torque = tau_pos + tau_drag

    A = torch.cat([force.permute(0, 2, 1), torque.permute(0, 2, 1)], dim=1)

    return A
