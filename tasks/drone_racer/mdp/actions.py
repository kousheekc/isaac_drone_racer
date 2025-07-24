# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from dynamics import Allocation, AttitudeController, BodyRateController, Motor
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

        self._allocation = Allocation(
            num_envs=self.num_envs,
            arm_length=self.cfg.arm_length,
            thrust_coeff=self.cfg.thrust_coef,
            drag_coeff=self.cfg.drag_coef,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._motor = Motor(
            num_envs=self.num_envs,
            taus=self.cfg.taus,
            init=self.cfg.init,
            max_rate=self.cfg.max_rate,
            min_rate=self.cfg.min_rate,
            dt=env.physics_dt,
            use=self.cfg.use_motor_model,
            device=self.device,
            dtype=self._raw_actions.dtype,
        )
        self._rate_controller = BodyRateController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_rates,
            self.device,
        )
        self._attitude_controller = AttitudeController(
            self.num_envs,
            self._robot.data.default_inertia[:, 0].view(-1, 3, 3),
            torch.eye(3) * self.cfg.k_attitude,
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

        if self.cfg.control_level == "thrust":
            mapped = (clamped.clone() + 1.0) / 2.0
            omega_ref = self.cfg.omega_max * mapped
            omega_real = self._motor.compute(omega_ref)
            log(self._env, ["w1", "w2", "w3", "w4"], omega_real)
            self._processed_actions = self._allocation.compute_with_omega(omega_real)
        elif self.cfg.control_level == "rates":
            # Clamp rates setpoint and total thrust
            # Calculate wrench based on rates setpoint
            # Calculate thrust setpoint based on wrench and allocation inverse
            # Clamp thrust setpoint
            mapped = clamped.clone()
            mapped[:, 0] *= torch.tensor(self.cfg.max_thrust, device=self.device, dtype=self._raw_actions.dtype)
            mapped[:, 1:] *= torch.tensor(self.cfg.max_ang_vel, device=self.device, dtype=self._raw_actions.dtype)
            mapped[:, 1:] = self._rate_controller.compute_moment(mapped[:, 1:], self._robot.data.root_ang_vel_b)
            log(self._env, ["T", "rate1", "rate2", "rate3"], mapped)
            self._processed_actions = mapped
        elif self.cfg.control_level == "attitude":
            # Clamp orientation setpoint and total thrust
            # Calculate wrench based on orientation setpoint
            # Calculate thrust setpoint based on wrench and allocation inverse
            # Clamp thrust setpoint
            mapped = clamped.clone()
            mapped[:, 0] *= torch.tensor(self.cfg.max_thrust, device=self.device, dtype=self._raw_actions.dtype)
            mapped[:, 1:] *= torch.tensor(self.cfg.max_attitude, device=self.device, dtype=self._raw_actions.dtype)
            mapped[:, 1:] = self._attitude_controller.compute_moment(
                mapped[:, 1:], self._robot.data.root_quat_w, self._robot.data.root_ang_vel_b
            )
            log(self._env, ["T", "att1", "att2", "att3"], mapped)
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

        self._motor.reset(env_ids)
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
    arm_length: float = 0.035
    """Length of the arms of the drone in meters."""
    drag_coef: float = 1.5e-9
    """Drag torque coefficient."""
    thrust_coef: float = 2.25e-7
    """Thrust coefficient.
    Calculated with 5145 rad/s max angular velociy, thrust to weight: 4, mass: 0.6076 kg and gravity: 9.81 m/s^2.
    thrust_coef = (4 * 0.6076 * 9.81) / (4 * 5145**2) = 2.25e-7."""
    omega_max: float = 5145.0
    """Maximum angular velocity of the drone motors in rad/s.
    Calculated with 1950KV motor, with 6S LiPo battery with 4.2V per cell.
    1950 * 6 * 4.2 = 49,140 RPM ~= 5145 rad/s."""
    max_thrust: float = 24.0
    """Maximum thrust of the drone in N.
    Calculated with 4 * thrust_coef * omega_max^2 = 4 * 2.25e-7 * 5145^2 = 24.0 N."""
    taus: list[float] = (0.0001, 0.0001, 0.0001, 0.0001)
    """Time constants for each motor."""
    init: list[float] = (2572.5, 2572.5, 2572.5, 2572.5)
    """Initial angular velocities for each motor in rad/s."""
    max_rate: list[float] = (50000.0, 50000.0, 50000.0, 50000.0)
    """Maximum rate of change of angular velocities for each motor in rad/s^2."""
    min_rate: list[float] = (-50000.0, -50000.0, -50000.0, -50000.0)
    """Minimum rate of change of angular velocities for each motor in rad/s^2."""
    use_motor_model: bool = False
    """Flag to determine if motor delay is bypassed."""
    max_ang_vel: list[float] = [3.5, 3.5, 3.5]
    """Maximum angular velocity."""
    max_attitude: list[float] = [torch.pi, torch.pi, torch.pi]
    """Maximum angular velocity."""
    k_attitude: float = 1.0
    """Proportional gain for attitude error."""
    k_rates: float = 0.2
    """Proportional gain for angular velocity error."""
    ControlLevel = Union[Literal["thrust"], Literal["rates"], Literal["attitude"]]
    control_level: ControlLevel = "rates"
    """Control level."""
