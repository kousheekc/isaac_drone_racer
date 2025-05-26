# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


def build_allocation_matrix_torch(num_envs, L, phi_deg, tilt_deg, spin_dir, k_T, k_D):
    """
    Build actuator allocation matrices for multiple environments.

    Args:
        num_envs : int, number of environments
        L        : (num_envs,) arm lengths
        phi_deg  : (num_envs, 4) motor angle positions (degrees)
        tilt_deg : (num_envs, 4) motor tilt angles (degrees)
        spin_dir : (num_envs, 4) +1 or -1 for each motor (CW/CCW)
        k_T      : (num_envs,) thrust coefficients
        k_D      : (num_envs,) drag torque coefficients

    Returns:
        A : (num_envs, 6, 4) allocation matrices
    """
    # Convert angles to radians
    phi = torch.deg2rad(phi_deg)  # (num_envs, 4)
    tilt = torch.deg2rad(tilt_deg)  # (num_envs, 4)

    # sin/cos
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    sin_tilt = torch.sin(tilt)
    cos_tilt = torch.cos(tilt)

    # Position vectors r_i = [L*cos(phi), L*sin(phi), 0]
    r_x = L.view(num_envs, 1) * cos_phi
    r_y = L.view(num_envs, 1) * sin_phi
    r_z = torch.zeros_like(r_x)
    r = torch.stack([r_x, r_y, r_z], dim=2)  # (num_envs, 4, 3)

    # Thrust directions z_i = [sin(tilt)*cos(phi), sin(tilt)*sin(phi), cos(tilt)]
    z_x = sin_tilt * cos_phi
    z_y = sin_tilt * sin_phi
    z_z = cos_tilt
    z = torch.stack([z_x, z_y, z_z], dim=2)  # (num_envs, 4, 3)

    # Thrust forces: f_i = k_T * z_i
    force = k_T.view(num_envs, 1, 1) * z  # (num_envs, 4, 3)

    # Torque from cross product r_i x f_i
    tau_pos = torch.cross(r, force, dim=2)  # (num_envs, 4, 3)

    # Torque from rotor drag
    tau_drag = (spin_dir * k_D.view(num_envs, 1))[:, :, None] * z  # (num_envs, 4, 3)

    # Total torque
    torque = tau_pos + tau_drag  # (num_envs, 4, 3)

    # Build full allocation matrix A: (num_envs, 6, 4)
    A = torch.cat(
        [force.permute(0, 2, 1), torque.permute(0, 2, 1)], dim=1  # (num_envs, 3, 4)  # (num_envs, 3, 4)
    )  # (num_envs, 6, 4)

    return A
