# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def self_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
    threshold: float = 1.0
    
) -> torch.Tensor:
    """Terminate when the robot collides with itself.
    """
    # # extract the used quantities (to enable type-hinting)
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_forces = contact_sensor.data.net_forces_w_history[:, :, ]
    
    
    
    # asset: RigidObject = env.scene[asset_cfg.name]
    
    pass
    
    
    
    
