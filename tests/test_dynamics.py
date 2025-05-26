# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import pytest
import torch

from dynamics import build_allocation_matrix_torch


@pytest.fixture
def setup_allocation_matrix():
    num_envs = 2
    L = torch.full((num_envs,), 0.1)
    phi_deg = torch.tensor([[0, 90, 180, 270]]).repeat(num_envs, 1)
    tilt_deg = torch.zeros((num_envs, 4))
    spin_dir = torch.tensor([[1, -1, 1, -1]]).repeat(num_envs, 1)
    k_T = torch.full((num_envs,), 1e-6)
    k_D = torch.full((num_envs,), 1e-7)

    A = build_allocation_matrix_torch(num_envs, L, phi_deg, tilt_deg, spin_dir, k_T, k_D)
    return A, num_envs


def test_allocation_matrix_shape(setup_allocation_matrix):
    A, num_envs = setup_allocation_matrix
    assert A.shape == (num_envs, 6, 4)


def test_wrench_output_shape(setup_allocation_matrix):
    A, num_envs = setup_allocation_matrix
    thrusts = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).repeat(num_envs, 1)
    wrench = torch.bmm(A, thrusts.unsqueeze(2)).squeeze(2)
    assert wrench.shape == (num_envs, 6)


def test_force_is_vertical_for_symmetric_quad(setup_allocation_matrix):
    A, num_envs = setup_allocation_matrix
    thrusts = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).repeat(num_envs, 1)
    wrench = torch.bmm(A, thrusts.unsqueeze(2)).squeeze(2)
    fx, fy, fz = wrench[:, 0], wrench[:, 1], wrench[:, 2]
    assert torch.allclose(fx, torch.zeros_like(fx), atol=1e-9)
    assert torch.allclose(fy, torch.zeros_like(fy), atol=1e-9)
    assert torch.all(fz > 0)


def test_torque_zero_for_balanced_spin(setup_allocation_matrix):
    A, num_envs = setup_allocation_matrix
    thrusts = torch.tensor([[1.0, 1.0, 1.0, 1.0]]).repeat(num_envs, 1)
    wrench = torch.bmm(A, thrusts.unsqueeze(2)).squeeze(2)
    mx, my, mz = wrench[:, 3], wrench[:, 4], wrench[:, 5]
    assert torch.allclose(mx, torch.zeros_like(mx), atol=1e-9)
    assert torch.allclose(my, torch.zeros_like(my), atol=1e-9)
    assert torch.allclose(mz, torch.zeros_like(mz), atol=1e-9)


def test_asymmetrical_thrust_produces_torque(setup_allocation_matrix):
    A, num_envs = setup_allocation_matrix
    thrusts = torch.tensor([[1.0, 0.5, 1.0, 0.5]]).repeat(num_envs, 1)
    wrench = torch.bmm(A, thrusts.unsqueeze(2)).squeeze(2)
    torque = wrench[:, 3:]
    assert torch.any(torch.abs(torque) > 1e-9)
