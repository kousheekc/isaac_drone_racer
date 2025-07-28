# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import BodyRateController
from utils.logger import log

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ControlAction(ActionTerm):
    r"""Body torque control action term.

    This action term applies a wrench to the drone body frame based on action commands

    """

    cfg: ControlActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ControlActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.cfg = cfg

        self._robot: Articulation = env.scene[self.cfg.asset_name]
        self._body_id = self._robot.find_bodies("body")[0]

        self._elapsed_time = torch.zeros(self.num_envs, 1, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.max_thrust = (
            self._robot.data.default_mass.sum(dim=1, keepdim=True)
            * -self._env.sim.cfg.gravity[-1]
            * self.cfg.thrust_weight_ratio
        ).to(self.device)

        self._rate_controller = BodyRateController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_rates,
            self.device,
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # TODO: make more explicit (thrust = 6, rates = 6, attitude = 6) all happen to be 6, but they represent different things
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

        self._raw_actions[:] = actions.clone()
        clamped = self._raw_actions.clamp_(-1.0, 1.0)
        log(self._env, ["a1", "a2", "a3", "a4"], actions)
        log(self._env, ["a1_clamped", "a2_clamped", "a3_clamped", "a4_clamped"], clamped)

        # Clamp rates setpoint and total thrust
        # Calculate wrench based on rates setpoint
        # Calculate thrust setpoint based on wrench and allocation inverse
        # Clamp thrust setpoint

        # print(self._robot.data.default_mass.sum(dim=1, keepdim=True))
        # print(self._robot.data.default_inertia)

        mapped = clamped.clone()
        mapped[:, :1] = (mapped[:, :1] + 1) / 2
        mapped[:, :1] *= self.max_thrust
        mapped[:, 1:] *= torch.tensor(self.cfg.max_ang_vel, device=self.device, dtype=self._raw_actions.dtype)
        mapped[:, 1:] = self._rate_controller.compute_moment(mapped[:, 1:], self._robot.data.root_ang_vel_b)
        log(self._env, ["T", "rate1", "rate2", "rate3"], mapped)
        self._processed_actions = mapped

    def apply_actions(self):
        self._thrust[:, 0, 2] = self._processed_actions[:, 0]
        self._moment[:, 0, :] = self._processed_actions[:, 1:]
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

        self._elapsed_time += self._env.physics_dt
        log(self._env, ["time"], self._elapsed_time)

    def reset(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._elapsed_time[env_ids] = 0.0

        self._robot.reset(env_ids)
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._env.scene.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@configclass
class ControlActionCfg(ActionTermCfg):
    """
    See :class:`ControlAction` for more details.
    """

    class_type: type[ActionTerm] = ControlAction
    """ Class of the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""
    thrust_weight_ratio: float = 2.5
    """Thrust weight ratio of the drone."""
    max_ang_vel: list[float] = [3.5, 3.5, 3.5]
    """Maximum angular velocity in rad/s"""
    k_rates: float = 0.01
    """Proportional gain for angular velocity error."""
