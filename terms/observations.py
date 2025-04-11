# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for the Mars Jumper environment."""

from __future__ import annotations

from typing import TYPE_CHECKING
from terms.phase import Phase
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def has_taken_off(
    env: ManagerBasedEnv, 
) -> torch.Tensor:
    """Observation that indicates whether the robot has taken off.
    
    The observation returns 1.0 if the robot has transitioned to flight or landing,
    and 0.0 otherwise. 
    
    Args:
        env: The environment instance.
        
    Returns:
        A tensor of shape (num_envs, 1) with 1.0 for environments where the robot has taken off
        and 0.0 for environments where it hasn't.
    """
    if not hasattr(env, "_phase_buffer"):
        return torch.zeros(env.num_envs, 1, device=env.device)
    
    return (env._phase_buffer == (Phase.FLIGHT | Phase.LANDING)).float().unsqueeze(-1)
    

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


