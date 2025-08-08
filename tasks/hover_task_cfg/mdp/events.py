# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.
    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.
    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract the specific diagonal element for the specified envs and bodies
        current_inertias = inertias[env_ids[:, None], body_ids, idx]

        # Randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            current_inertias,
            inertia_distribution_params,
            torch.arange(len(env_ids), device="cpu"),  # Use sequential indices for the subset
            torch.arange(len(body_ids), device="cpu"),  # Use sequential indices for the subset
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_control_terms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action: str,
    randomization_params: dict[str, tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize control terms by adding, scaling, or setting random values.

    Args:
        env: The environment instance
        env_ids: Environment IDs to randomize (None for all environments)
        action: Name of the action term
        randomization_params: Dictionary mapping parameter names to distribution params.
                            Keys should match both config attribute names and action term attribute names.
                            For example: {"twr": (0.8, 1.2), "tau_omega": (0.5, 2.0), ...}
        operation: Operation to perform ("add", "scale", "abs")
        distribution: Distribution type for sampling
    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    config = env.action_manager.get_term(action).config
    action_term = env.action_manager.get_term(action)

    # Loop through each parameter to randomize
    for param_name, distribution_params in randomization_params.items():
        # Get default value from config
        default_value = getattr(config, param_name)
        term_default = torch.full((env.num_envs, 1), default_value, device=env.device)

        # Get current value from action term
        term_current = getattr(action_term, param_name)
        term_current[env_ids] = term_default[env_ids]

        # Randomize the parameter
        term_new = _randomize_prop_by_op(
            term_current,
            distribution_params,
            env_ids,
            slice(None),
            operation,
            distribution,
        )

        # Set the new randomized value
        setattr(action_term, param_name, term_new)
