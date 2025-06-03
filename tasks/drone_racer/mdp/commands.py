# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GateTargetingCommand(CommandTerm):
    """Command generator that generates a pose command from a uniform distribution."""

    cfg: GateTargetingCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: GateTargetingCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and track for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.track: RigidObjectCollection = env.scene[cfg.track_name]
        self.gate_size = cfg.gate_size
        self.num_gates = self.track.num_objects

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in simulation world frame
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.prev_robot_pos_w = self.robot.data.root_pos_w
        self._gate_missed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.next_gate_idx = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.next_gate_w = torch.zeros(self.num_envs, 7, device=self.device)

    def __str__(self) -> str:
        msg = "GateTargetingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.next_gate_w

    @property
    def gate_missed(self) -> torch.Tensor:
        return self._gate_missed

    @property
    def gate_passed(self) -> torch.Tensor:
        return self._gate_passed

    @property
    def previous_pos(self) -> torch.Tensor:
        return self.prev_robot_pos_w

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        self.next_gate_idx[env_ids] = 0

    def _update_command(self):
        next_gate_positions = self.track.data.object_com_pos_w[self.env_ids, self.next_gate_idx]
        next_gate_orientations = self.track.data.object_quat_w[self.env_ids, self.next_gate_idx]
        self.next_gate_w = torch.cat([next_gate_positions, next_gate_orientations], dim=1)

        # Gate passing logic
        (roll, pitch, yaw) = math_utils.euler_xyz_from_quat(self.next_gate_w[:, 3:7])
        normal = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        pos_old_projected = (self.prev_robot_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.prev_robot_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        pos_new_projected = (self.robot.data.root_pos_w[:, 0] - self.next_gate_w[:, 0]) * normal[:, 0] + (
            self.robot.data.root_pos_w[:, 1] - self.next_gate_w[:, 1]
        ) * normal[:, 1]
        passed_gate_plane = (pos_old_projected < 0) & (pos_new_projected > 0)

        self._gate_passed = passed_gate_plane & (
            torch.all(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) < (self.gate_size / 2), dim=1)
        )

        self._gate_missed = passed_gate_plane & (
            torch.any(torch.abs(self.robot.data.root_pos_w - self.next_gate_w[:, :3]) > (self.gate_size / 2), dim=1)
        )

        # Update next gate target for the envs that passed the gate
        self.next_gate_idx[self._gate_passed] += 1
        self.next_gate_idx = self.next_gate_idx % self.num_gates

        self.prev_robot_pos_w = self.robot.data.root_pos_w

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                # -- goal pose
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                # -- current body pose
                self.drone_visualizer = VisualizationMarkers(self.cfg.drone_visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
            self.drone_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)
                self.drone_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        self.target_visualizer.visualize(self.next_gate_w[:, :3], self.next_gate_w[:, 3:])
        self.drone_visualizer.visualize(self.robot.data.root_pos_w, self.robot.data.root_quat_w)


@configclass
class GateTargetingCommandCfg(CommandTermCfg):
    """Configuration for gate targeting command generator."""

    class_type: type = GateTargetingCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    track_name: str = MISSING
    """Name of the track in the environment for which the commands are generated."""

    gate_size: float = 1.5
    """Size of the gate in meters. This is used to determine if the drone has passed through the gate."""

    target_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    drone_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    target_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    drone_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
