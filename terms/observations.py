# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Mars Jumper environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def has_taken_off(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_threshold: float = 0.1
) -> torch.Tensor:
    """Observation that indicates whether the robot has jumped (taken off).
    
    The observation returns 1.0 if the robot's base is above the height_threshold,
    and 0.0 otherwise. Once an environment has toggled to 1.0, it stays at 1.0
    until the environment is reset.
    
    Args:
        env: The environment instance.
        asset_cfg: The asset configuration for the robot.
        height_threshold: The height threshold above which the robot is considered to have jumped.
        
    Returns:
        A tensor of shape (num_envs, 1) with 1.0 for environments where the robot has jumped
        and 0.0 for environments where it hasn't.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    current_height = robot.data.root_pos_w[:, 2]
    
    if not hasattr(env, "_has_taken_off_buffer"):
        env._has_taken_off_buffer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        
    
    above_threshold = current_height > height_threshold
    
    # Update the jump toggle buffer - once toggled to 1, it stays at 1
    env._has_taken_off_buffer = torch.logical_or(env._has_taken_off_buffer, above_threshold).to(torch.float32)
    
    # Return the jump toggle as a tensor of shape (num_envs, 1)
    return env._has_taken_off_buffer.unsqueeze(-1) 


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
    if not hasattr(env, "command_manager") or env.command_manager is None:
        # Return a dummy tensor with the expected shape (num_envs, 2)
        return torch.zeros((env.num_envs, 2), device=env.device)
    
    return env.command_manager.get_command("takeoff_vel_vec")

