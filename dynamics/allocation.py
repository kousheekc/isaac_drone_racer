# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


class Allocation:
    def __init__(self, num_envs, arm_length, thrust_coeff, drag_coeff, device="cpu", dtype=torch.float32):
        """
        Initializes the allocation matrix for a quadrotor for multiple environments.

        Parameters:
        - num_envs (int): Number of environments
        - arm_length (float): Distance from the center to the rotor
        - thrust_coeff (float): Rotor thrust constant
        - drag_coeff (float): Rotor torque constant
        - device (str): 'cpu' or 'cuda'
        - dtype (torch.dtype): Desired tensor dtype
        """
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        A = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [arm_length * sqrt2_inv, -arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [-arm_length * sqrt2_inv, -arm_length * sqrt2_inv, arm_length * sqrt2_inv, arm_length * sqrt2_inv],
                [drag_coeff, -drag_coeff, drag_coeff, -drag_coeff],
            ],
            dtype=dtype,
            device=device,
        )
        self._allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_coeff = thrust_coeff

    def compute(self, omega):
        """
        Computes the total thrust and body torques given the rotor angular velocities.

        Parameters:
        - omega (torch.Tensor): Tensor of shape (num_envs, 4) representing rotor angular velocities

        Returns:
        - thrust_torque (torch.Tensor): Tensor of shape (num_envs, 4)
        """
        thrusts_ref = self._thrust_coeff * omega**2
        thrust_torque = torch.bmm(self._allocation_matrix, thrusts_ref.unsqueeze(-1)).squeeze(-1)
        return thrust_torque
