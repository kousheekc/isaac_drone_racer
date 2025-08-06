# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

import csv
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics.sys_dynamics import SystemDynamics
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    r"""Control action term.

    This action term applies the system dynamics to the drone based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the control action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg
        self.env = env

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._mass = self._robot.data.default_mass.sum(dim=1, keepdim=True).to(self.device)
        self._inertia = self._robot.data.default_inertia[:, 0].view(-1, 3, 3).to(self.device)

        self._model = SystemDynamics(
            dt=self.env.physics_dt,
            tau_omega=self.cfg.tau_omega,
            tau_thrust=self.cfg.tau_thrust,
            dx=self.cfg.dx,
            dy=self.cfg.dy,
            inertia=self._inertia,
        ).to(self.device)

        # Initialize CSV logging if num_envs = 1
        if self.num_envs == 1:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._csv_file_path = f"drone_log_{timestamp}.csv"
            self._csv_initialized = False

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def has_debug_vis_implementation(self) -> bool:
        return False

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        log(self._env, ["a1", "a2", "a3", "a4"], actions)

        clamped = self._raw_actions.clone().clamp_(-1.0, 1.0)
        log(self._env, ["a1_clamped", "a2_clamped", "a3_clamped", "a4_clamped"], clamped)

        mapped = clamped.clone()
        mapped[:, :1] = (mapped[:, :1] + 1) / 2
        mapped[:, :1] *= -1.0 * self.env.sim.cfg.gravity[2] * self._mass * self.cfg.thrust_weight_ratio
        mapped[:, 1:] *= torch.tensor(self.cfg.max_ang_vel, device=self.device, dtype=self._raw_actions.dtype)
        log(self._env, ["t_d", "w1_d", "w2_d", "w3_d"], clamped)

        self._processed_actions[:] = mapped

    def apply_actions(self):
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b

        force, moment = self._model(
            lin_vel_b, ang_vel_b, self._thrust.squeeze(1)[:, -1].unsqueeze(1), self._processed_actions
        )

        self._thrust[:] = force.unsqueeze(1)
        self._moment[:] = moment.unsqueeze(1)

        # Log to CSV if num_envs = 1
        if self.num_envs == 1:
            self._log_to_csv(ang_vel_b, lin_vel_b, force, moment, self._processed_actions)

        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def _log_to_csv(
        self,
        ang_vel_b: torch.Tensor,
        lin_vel_b: torch.Tensor,
        force: torch.Tensor,
        moment: torch.Tensor,
        processed_actions: torch.Tensor,
    ):
        """Log drone state and control data to CSV file when num_envs = 1"""
        # Initialize CSV file with header if not done yet
        if not self._csv_initialized:
            with open(self._csv_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "timestamp",
                    "ang_vel_x",
                    "ang_vel_y",
                    "ang_vel_z",
                    "lin_vel_x",
                    "lin_vel_y",
                    "lin_vel_z",
                    "force_x",
                    "force_y",
                    "force_z",
                    "moment_x",
                    "moment_y",
                    "moment_z",
                    "action_thrust",
                    "action_roll",
                    "action_pitch",
                    "action_yaw",
                ])
            self._csv_initialized = True

        # Convert tensors to CPU and extract values for the single environment
        elapsed_time_value = self._elapsed_time[0, 0].cpu().item()
        ang_vel_values = ang_vel_b[0].cpu().numpy()  # Shape: (3,)
        lin_vel_values = lin_vel_b[0].cpu().numpy()  # Shape: (3,)
        force_values = force[0].cpu().numpy()  # Shape: (3,)
        moment_values = moment[0].cpu().numpy()  # Shape: (3,)
        action_values = processed_actions[0].cpu().numpy()  # Shape: (4,)

        # Write data to CSV
        with open(self._csv_file_path, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                elapsed_time_value,
                ang_vel_values[0],
                ang_vel_values[1],
                ang_vel_values[2],
                lin_vel_values[0],
                lin_vel_values[1],
                lin_vel_values[2],
                force_values[0],
                force_values[1],
                force_values[2],
                moment_values[0],
                moment_values[1],
                moment_values[2],
                action_values[0],
                action_values[1],
                action_values[2],
                action_values[3],
            ])

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
            # Reset CSV initialization when all environments are reset
            if self.num_envs == 1:
                self._csv_initialized = False

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._thrust[env_ids, :, 2] = -1.0 * self.env.sim.cfg.gravity[2] * self._mass
        self._moment[env_ids] = 0.0

        self._robot.reset(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_state_to_sim(default_root_state, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    thrust_weight_ratio: float = 2.0
    """Thrust weight ratio of the drone."""
    max_ang_vel: list[float] = [3.5, 3.5, 3.5]
    """Maximum angular velocity in rad/s"""
    tau_omega: float = 0.03
    """Time constant for angular velocity control."""
    tau_thrust: float = 0.03
    """Time constant for thrust control."""
    dx: float = 0.35
    """Body drag coefficient along x-axis."""
    dy: float = 0.35
    """Body drag coefficient along y-axis."""
