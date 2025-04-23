# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple

import torch

import logging
logger = logging.getLogger(__name__)
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def progress_command_ranges(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    num_curriculum_levels: int = 10,
    success_rate_threshold: float = 0.9,
    min_steps_above_threshold: int = 100,
) -> None:
    """Set the command takeoff vector magnitude ranges for the robot. 
    The initial range is set to the initial magnitude range of the command term, and will be incremented linearly to the final range over the number of curriculum steps.
    
    Args:
        env: The environment instance
        env_ids: Not used since all environments are affected
        num_curriculum_levels: Number of curriculum levels to split the curriculum into from 0 to 1 percent completion from initial to final magnitude range
    """  
    
    #Initialize attributes when first called
    if not hasattr(env, "cmd_curriculum_progress_ratio"): env.cmd_curriculum_progress_ratio = 0
    if not hasattr(env, "env_steps_since_last_curriculum_update"): env.env_steps_since_last_curriculum_update = 0
    if not hasattr(env, "steps_above_success_threshold"): env.steps_above_success_threshold = 0
    
    progress_ratio = env.cmd_curriculum_progress_ratio
    initial_magnitude_range = env.cfg.command_ranges.initial_magnitude_range
    final_magnitude_range = env.cfg.command_ranges.final_magnitude_range

    current_magnitude_min = initial_magnitude_range[0] + progress_ratio * (final_magnitude_range[0] - initial_magnitude_range[0])
    current_magnitude_max = initial_magnitude_range[1] + progress_ratio * (final_magnitude_range[1] - initial_magnitude_range[1])
    
    env.cmd_magnitude_range = (current_magnitude_min, current_magnitude_max)
    env.cmd_pitch_range = (env.cfg.command_ranges.initial_pitch_range[0], env.cfg.command_ranges.final_pitch_range[0])
    
    if env.success_rate > success_rate_threshold:
        env.steps_above_success_threshold += 1
    else:
        env.steps_above_success_threshold = 0
    
    if (env.steps_above_success_threshold >= min_steps_above_threshold and 
        env.env_steps_since_last_curriculum_update > env.mean_episode_env_steps and 
        env.cmd_curriculum_progress_ratio < 1):
        
        env.steps_above_success_threshold = 0
        env.cmd_curriculum_progress_ratio += 1/num_curriculum_levels
        env.env_steps_since_last_curriculum_update = 0
        print("Advancing takeoff magnitude curriculum: current_step_counter %s, progress ratio %s, mag=[%s, %s]", env.common_step_counter, env.cmd_curriculum_progress_ratio, current_magnitude_min, current_magnitude_max)
        
    # Need to return state to be logged
    return {
        "progress_ratio": env.cmd_curriculum_progress_ratio,
        "cmd_magnitude_min": current_magnitude_min,
        "cmd_magnitude_max": current_magnitude_max,
    }
    
    # Might change success based on reward later
    #mag_reward = env.reward_manager.episode_sums.get("takeoff_vel_vec_magnitude", torch.zeros(len(env_ids), device=env.device))
    
