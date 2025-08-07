# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

from assets.five_in_drone import FIVE_IN_DRONE  # isort:skip


@configclass
class HoverSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # robot
    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    collision_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True)

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        pos_t = ObsTerm(func=mdp.root_pos_t, params={"command_name": "target"})
        att_t = ObsTerm(func=mdp.root_rotmat_t, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.history_length = 10
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "target"})
        force = ObsTerm(func=mdp.force_from_action)
        moment = ObsTerm(func=mdp.moment_from_action)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.history_length = 10
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target = mdp.PosCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        ranges=mdp.PosCommandCfg.Ranges(pos_x=(-2.0, 2.0), pos_y=(-2.0, 2.0), pos_z=(0.5, 1.5)),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (0.5, 1.5),
                "roll": (-torch.pi / 4, torch.pi / 4),
                "pitch": (-torch.pi / 4, torch.pi / 4),
                "yaw": (-torch.pi, torch.pi),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    pos_error_l2 = RewTerm(func=mdp.pos_error_l2, weight=15.0, params={"command_name": "target"})
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-1.0)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.1)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 10.0})
    collision = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01}
    )


@configclass
class HoverEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: HoverSceneCfg = HoverSceneCfg(num_envs=4096, env_spacing=4.0)
    # MDP settings
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # general settings
        self.decimation = 2
        self.episode_length_s = 20.0
        # viewer settings
        self.viewer.eye = (-3.0, -3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 480
        self.sim.render_interval = self.decimation
