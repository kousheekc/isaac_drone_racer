# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test script to run RL environment directly.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
from isaaclab.envs import ManagerBasedRLEnv

from tasks.hover_task_cfg.hover_env_cfg import HoverEnvCfg


def main():
    env_cfg = HoverEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :] = 0.0

            obs, rew, terminated, truncated, info = env.step(actions)
            # print(rew)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
