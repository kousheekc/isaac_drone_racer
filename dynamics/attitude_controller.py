# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.utils.math as math_utils
import torch


class AttitudeController:
    def __init__(self, num_envs, J, K_attitude, K_omega, device="cpu"):
        """
        Initializes the attitude controller.

        Parameters:
        - num_envs: Number of envs.
        - J: (3, 3) Tensor of inertia matrix (same for all drones).
        - K_attitude: (3, 3) Tensor of proportional gain for attitude error.
        - K_omega: (3, 3) Tensor of proportional gain for angular velocity error.
        - device: Device to store tensors ('cpu' or 'cuda').
        """
        self.device = device
        self.num_envs = num_envs

        self.J = J.to(device).expand(self.num_envs, -1, -1)  # (N, 3, 3)
        self.K_attitude = K_attitude.to(device).expand(self.num_envs, -1, -1)  # (N, 3, 3)
        self.K_omega = K_omega.to(device).expand(self.num_envs, -1, -1)  # (N, 3, 3)

    def inverse_skew(self, R):
        """Extracts the inverse of skew-symmetric from a batch of 3x3 matrices."""
        return torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], dim=1) / 2

    def compute_moment(self, attitude_d, attitude, omega):
        """
        Compute the control moment for a batch of drones.

        Parameters:
        - attitude_d: (N, 3) Tensor of desired orientations expressed in euler angles.
        - attitude: (N, 4) Tensor of current orientations expressed in quaternions.
        - omega: (N, 3) Tensor of current angular velocities.

        Returns:
        - M: (N, 3) Tensor of control moments.
        """
        attitude_d_matrix = math_utils.matrix_from_euler(attitude_d, convention="XYZ")  # (N, 3, 3)
        attitude_matrix = math_utils.matrix_from_quat(attitude)  # (N, 3, 3)

        # Compute attitude error e_R
        attitude_error_matrix = torch.bmm(attitude_d_matrix.transpose(1, 2), attitude_matrix) - torch.bmm(
            attitude_matrix.transpose(1, 2), attitude_d_matrix
        )  # (N, 3, 3)
        e_R = self.inverse_skew(attitude_error_matrix)  # (N, 3)

        # Compute angular velocity error e_omega (simplified: e_omega = omega)
        e_omega = omega  # (N, 3)

        # Compute Coriolis term omega x (J omega)
        omega_cross_J_omega = torch.cross(omega, torch.bmm(self.J, omega.unsqueeze(-1)).squeeze(-1), dim=1)  # (N, 3)

        # Compute control moment
        M = (
            -torch.bmm(self.K_attitude, e_R.unsqueeze(-1)).squeeze(-1)
            - torch.bmm(self.K_omega, e_omega.unsqueeze(-1)).squeeze(-1)
            + omega_cross_J_omega
        )  # (N, 3)
        return M
