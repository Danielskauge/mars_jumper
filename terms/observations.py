# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Mars Jumper environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from numpy import indices
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from terms.phase import Phase
import torch
import isaaclab.utils.math as math_utils
from terms.utils import get_center_of_mass_lin_vel
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def center_of_mass_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """Observation that returns the center of mass linear velocity.
    """
    return get_center_of_mass_lin_vel(env)

def has_taken_off(
    env: ManagerBasedEnv, 
) -> torch.Tensor:
    """Observation that indices whether the robot has taken off.
    
    The observation returns 1.0 if the robot has transitioned to flight or landing,
    and 0.0 otherwise. 
    
    Args:
        env: The environment instance.
        
    Returns:
        A tensor of shape (num_envs, 1) with 1.0 for environments where the robot has taken off
        and 0.0 for environments where it hasn't.
    """
    if not hasattr(env, "jump_phase"):
        env.jump_phase = torch.full((env.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=env.device)

    return ((env.jump_phase == Phase.LANDING) | (env.jump_phase == Phase.FLIGHT)).float().unsqueeze(-1)

def base_rotation_vector(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Observation that returns the base rotation vector.
    """
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    quat = robot.data.root_quat_w
    quat = math_utils.quat_unique(quat)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]
    
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    sin_angle_2 = torch.sin(angle/2)
    
    # Avoid division by zero
    mask = sin_angle_2 != 0
    rot_vec = torch.zeros_like(quat[:, 1:])
    rot_vec[mask] = angle[mask].unsqueeze(-1) * torch.stack([x[mask], y[mask], z[mask]], dim=-1) / (sin_angle_2[mask].unsqueeze(-1))
    
    return rot_vec

def takeoff_vel_vec_cmd(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Observation that returns the takeoff velocity vector command.
    
    Args:
        env: The environment instance.
        
    Returns:
        A tensor of shape (num_envs, 2) with the takeoff velocity vector command (pitch, magnitude)
    """
    # During observation manager initialization, command_manager might not be fully set up
    if not hasattr(env, "_command_buffer") or env._command_buffer is None:
        # Return a dummy tensor with the expected shape (num_envs, 2)
        return torch.zeros((env.num_envs, 2), device=env.device)
    
    return env._command_buffer


