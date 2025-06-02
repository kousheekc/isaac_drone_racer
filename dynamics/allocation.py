# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch


def generate_allocation_matrix(num_envs, arm_length, drag_coeff, device="cpu", dtype=torch.float32):
    """
    Generates an allocation matrix for a quadrotor for multiple environments using PyTorch.

    Parameters:
    - num_envs (int): Number of environments
    - arm_length (float): Distance from the center to the rotor
    - drag_coeff (float): Rotor torque constant (kappa)
    - device (str): 'cpu' or 'cuda'
    - dtype (torch.dtype): Desired tensor dtype

    Returns:
    - allocation_matrix (torch.Tensor): Tensor of shape (num_envs, 4, 4)
    """
    sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))

    # Define base allocation matrix
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

    allocation_matrix = A.unsqueeze(0).repeat(num_envs, 1, 1)

    return allocation_matrix
