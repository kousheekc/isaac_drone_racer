# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, TiledCameraCfg
from isaaclab.utils import configclass

from . import mdp
from .track_generator import generate_track

from assets.five_in_drone import FIVE_IN_DRONE  # isort:skip


@configclass
class DroneRacerSceneCfg(InteractiveSceneCfg):

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # track
    track: RigidObjectCollectionCfg = generate_track(
        track_config={
            "1": {"pos": (0.0, 0.0, 1.0), "yaw": 0.0},
            "2": {"pos": (10.0, 5.0, 0.0), "yaw": 0.0},
            "3": {"pos": (10.0, -5.0, 0.0), "yaw": (5 / 4) * torch.pi},
            "4": {"pos": (-5.0, -5.0, 2.5), "yaw": torch.pi},
            "5": {"pos": (-5.0, -5.0, 0.0), "yaw": 0.0},
            "6": {"pos": (5.0, 0.0, 0.0), "yaw": (1 / 2) * torch.pi},
            "7": {"pos": (0.0, 5.0, 0.0), "yaw": torch.pi},
        }
    )

    # robot
    robot: ArticulationCfg = FIVE_IN_DRONE.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    collision_sensor: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", debug_vis=True)
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/body", debug_vis=False)
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.14, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(),
        width=1000,
        height=1000,
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    control_action: mdp.ControlActionCfg = mdp.ControlActionCfg(use_motor_model=False)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        position = ObsTerm(func=mdp.root_pos_w)
        attitude = ObsTerm(func=mdp.root_quat_w)
        lin_vel = ObsTerm(func=mdp.root_lin_vel_b)
        ang_vel = ObsTerm(func=mdp.root_ang_vel_b)
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        image = ObsTerm(func=mdp.image)
        imu_ang_vel = ObsTerm(func=mdp.imu_ang_vel)
        imu_lin_acc = ObsTerm(func=mdp.imu_lin_acc)
        imu_att = ObsTerm(func=mdp.imu_orientation)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    # TODO: Resetting base happens in the command reset also for the moment
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-3.5, -1.5),
                "y": (-0.5, 0.5),
                "z": (1.5, 0.5),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # intervals
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 0.2),
        params={
            "force_range": (-0.1, 0.1),
            "torque_range": (-0.05, 0.05),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    target = mdp.GateTargetingCommandCfg(
        asset_name="robot",
        track_name="track",
        randomise_start=None,
        record_fpv=False,
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    terminating = RewTerm(func=mdp.is_terminated, weight=-500.0)
    ang_vel_l2 = RewTerm(func=mdp.ang_vel_l2, weight=-0.0001)
    progress = RewTerm(func=mdp.progress, weight=20.0, params={"command_name": "target"})
    gate_passed = RewTerm(func=mdp.gate_passed, weight=400.0, params={"command_name": "target"})
    lookat_next = RewTerm(func=mdp.lookat_next_gate, weight=0.1, params={"command_name": "target", "std": 0.5})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flyaway = DoneTerm(func=mdp.flyaway, params={"command_name": "target", "distance": 20.0})
    collision = DoneTerm(
        func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("collision_sensor"), "threshold": 0.01}
    )


@configclass
class DroneRacerEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None
        self.events.reset_base = None
        self.commands.target.randomise_start = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation


@configclass
class DroneRacerEnvCfg_PLAY(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: DroneRacerSceneCfg = DroneRacerSceneCfg(num_envs=4096, env_spacing=0.0)
    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""

        # Disable IMU and Tiled Camera
        self.scene.imu = None
        self.scene.tiled_camera = None

        # MDP settings
        self.observations.critic = None

        # Disable push robot events
        self.events.push_robot = None

        # Enable recording fpv footage
        # self.commands.target.record_fpv = True

        # general settings
        self.decimation = 4
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (-10.0, -10.0, 10.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # simulation settings
        self.sim.dt = 1 / 400
        self.sim.render_interval = self.decimation
