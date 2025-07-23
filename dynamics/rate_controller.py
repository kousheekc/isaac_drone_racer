# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class BodyRateController:
    def __init__(self, num_envs, J, K_omega, device="cpu"):
        """
        Initializes the body-rate controller.

        Parameters:
        - num_envs: Number of envs.
        - J: (3, 3) Tensor of inertia matrix (same for all drones).
        - K_omega: (3, 3) Tensor of proportional gain matrix (same for all drones).
        - device: Device to store tensors ('cpu' or 'cuda').
        """
        self.device = device
        self.num_envs = num_envs

        self.J = J.to(device).expand(self.num_envs, -1, -1)  # (N, 3, 3)
        self.K_omega = K_omega.to(device).expand(self.num_envs, -1, -1)  # (N, 3, 3)

    def compute_moment(self, omega_ref, omega):
        """
        Compute the control moment for a batch of drones.

        Parameters:
        - omega_ref: (N, 3) Tensor of desired angular velocities.
        - omega: (N, 3) Tensor of current angular velocities.

        Returns:
        - tau: (N, 3) Tensor of control moments.
        """
        # Compute angular velocity error
        e_omega = omega_ref - omega  # (N, 3)

        omega_cross_J_omega = torch.cross(omega, torch.bmm(self.J, omega.unsqueeze(-1)).squeeze(-1), dim=1)  # (N, 3)

        # Compute control moment
        tau = torch.bmm(self.K_omega, e_omega.unsqueeze(-1)).squeeze(-1) + omega_cross_J_omega  # (N, 3)

        return tau
