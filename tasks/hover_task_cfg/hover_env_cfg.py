# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
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

        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"}, history_length=10)
        rotmat_w = ObsTerm(func=mdp.root_rotmat_w, history_length=10)
        last_action = ObsTerm(func=mdp.last_action, history_length=10)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"}, history_length=10)
        rotmat_w = ObsTerm(func=mdp.root_rotmat_w, history_length=10)
        last_action = ObsTerm(func=mdp.last_action, history_length=10)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        force = ObsTerm(func=mdp.force_from_action)
        moment = ObsTerm(func=mdp.moment_from_action)
        gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self) -> None:
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
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (1.0, 1.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_inertia = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="body"),
            "inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_control_terms = EventTerm(
        func=mdp.randomize_control_terms,
        mode="reset",
        params={
            "action": "control_action",
            "randomization_params": {
                "twr": (0.8, 1.2),
                "tau_omega": (0.8, 1.2),
                "tau_thrust": (0.8, 1.2),
                "dx": (0.8, 1.2),
                "dy": (0.8, 1.2),
            },
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    pos_error_tanh = RewTerm(func=mdp.pos_error_tanh, weight=15.0, params={"command_name": "target", "std": 0.8})
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 5.0})
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
        self.decimation = 4
        self.episode_length_s = 10.0
        # viewer settings
        self.viewer.eye = (-3.0, -3.0, 3.0)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        # simulation settings
        self.sim.dt = 1 / 480
        self.sim.render_interval = self.decimation
