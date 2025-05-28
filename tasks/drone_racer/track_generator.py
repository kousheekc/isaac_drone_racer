# Copyright (c) 2025, Kousheek Chakraborty
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# This project uses the IsaacLab framework (https://github.com/isaac-sim/IsaacLab),
# which is licensed under the BSD-3-Clause License.

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg


def generate_track(num_gates: int = 1) -> RigidObjectCollectionCfg:

    return RigidObjectCollectionCfg(
        rigid_objects={
            f"gate_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Gate_{i}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path="assets/gate.usd",
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                    ),
                    scale=(1.0, 1.0, 1.0),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0 * i, 0.0, i)),
            )
            for i in range(num_gates)
        }
    )
