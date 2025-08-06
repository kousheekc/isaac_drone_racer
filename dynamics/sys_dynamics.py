# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import torch
import torch.nn as nn


class SystemDynamics(nn.Module):
    def __init__(self, dt=0.01, tau_omega=0.03, tau_thrust=0.03, dx=0.34, dy=0.43):
        super().__init__()
        self.dt = dt
        self.tau_omega = tau_omega
        self.tau_thrust = tau_thrust
        self.dx = dx
        self.dy = dy

    def forward(
        self,
        current_v_body: torch.Tensor,  # (num_envs, 3) linear velocity in body frame
        current_omega: torch.Tensor,  # (num_envs, 3) current actuator angular velocity
        current_thrust: torch.Tensor,  # (num_envs, 1) current actuator thrust
        action: torch.Tensor,  # (num_envs, 4): [omega_cmd(3), thrust_cmd(1)]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            v_body: (num_envs, 3) linear velocity in body frame
            current_omega: (num_envs, 3) current actuator angular velocity
            current_thrust: (num_envs, 1) current actuator thrust
            action: (num_envs, 4) = [thrust_cmd(1), omega_cmd(3)]
            dt: time step (float)

        Returns:
            force_body: (num_envs, 3)
            moment_body: (num_envs, 3)
        """
        thrust_cmd = action[:, 0:1]
        omega_cmd = action[:, 1:4]

        # First-order actuator lag
        thrust_dot = (thrust_cmd - current_thrust) / self.tau_thrust
        thrust = current_thrust + self.dt * thrust_dot

        force_body = torch.stack(
            [-self.dx * current_v_body[:, 0], -self.dy * current_v_body[:, 1], thrust.squeeze(-1)], dim=-1
        )

        moment_body = (omega_cmd - current_omega) / self.tau_omega

        return force_body, moment_body
