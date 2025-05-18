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
    success_rate_threshold: float = 0.4,
    min_steps_between_updates: int = 50,
    enable_regression: bool = True,
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
    if not hasattr(env, "steps_since_curriculum_update"): env.steps_since_curriculum_update = 0
    if not hasattr(env, "steps_above_success_threshold"): env.steps_above_success_threshold = 0
    if not hasattr(env, "prev_running_takeoff_success_rate"):
        env.prev_running_takeoff_success_rate = getattr(env, "running_takeoff_success_rate", 0.0)
    
    progress_ratio = env.cmd_curriculum_progress_ratio
    
    initial_magnitude_range = env.cfg.command_ranges.magnitude_range
    final_magnitude_range = env.cfg.command_ranges.curriculum_final_magnitude_range
    current_magnitude_min = initial_magnitude_range[0] + progress_ratio * (final_magnitude_range[0] - initial_magnitude_range[0])
    current_magnitude_max = initial_magnitude_range[1] + progress_ratio * (final_magnitude_range[1] - initial_magnitude_range[1])
    env.cmd_magnitude_range = (current_magnitude_min, current_magnitude_max)
    
    initial_pitch_range = env.cfg.command_ranges.pitch_range
    final_pitch_range = env.cfg.command_ranges.curriculum_final_pitch_range
    current_pitch_min = initial_pitch_range[0] + progress_ratio * (final_pitch_range[0] - initial_pitch_range[0])
    current_pitch_max = initial_pitch_range[1] + progress_ratio * (final_pitch_range[1] - initial_pitch_range[1])
    env.cmd_pitch_range = (current_pitch_min, current_pitch_max)
    
    current_success_rate = getattr(env, "running_takeoff_success_rate", 0.0)
    previous_recorded_success_rate = getattr(env, "prev_running_takeoff_success_rate", 0.0)
    success_rate_not_decreasing = current_success_rate >= previous_recorded_success_rate

    # if current_success_rate > success_rate_threshold:
    #     env.steps_above_success_threshold += 1
    # else:
    #     env.steps_above_success_threshold = 0 # Reset if success rate is not above threshold
        
    if env.steps_since_curriculum_update > max(2*env.mean_episode_env_steps, min_steps_between_updates):
        progressed_this_cycle = False
        # Try to progress curriculum
        if env.cmd_curriculum_progress_ratio < 1 and success_rate_not_decreasing and env.running_takeoff_success_rate > success_rate_threshold:
            env.cmd_curriculum_progress_ratio += 1/num_curriculum_levels
            env.steps_since_curriculum_update = 0
            progressed_this_cycle = True
            print("Advancing takeoff magnitude curriculum: current_step_counter %s, progress ratio %s, mag=[%s, %s]", env.common_step_counter, env.cmd_curriculum_progress_ratio, current_magnitude_min, current_magnitude_max)
    
    if enable_regression:
        # Try to regress curriculum (if not progressed)
        if not progressed_this_cycle and current_success_rate < success_rate_threshold and env.cmd_curriculum_progress_ratio > 0:
            env.cmd_curriculum_progress_ratio -= 1/num_curriculum_levels
            env.cmd_curriculum_progress_ratio = max(0, env.cmd_curriculum_progress_ratio) # Ensure not < 0
            env.steps_since_curriculum_update = 0
            # progressed_this_cycle = True # This was for the if not progressed_this_cycle condition, not needed to set true here
            print("Decreasing takeoff magnitude curriculum: current_step_counter %s, progress ratio %s, mag=[%s, %s]", env.common_step_counter, env.cmd_curriculum_progress_ratio, current_magnitude_min, current_magnitude_max)

    env.prev_running_takeoff_success_rate = current_success_rate
        
    # Need to return state to be logged
    return {
        "progress_ratio": env.cmd_curriculum_progress_ratio,
        "cmd_magnitude_min": current_magnitude_min,
        "cmd_magnitude_max": current_magnitude_max,
    }
    
    # Might change success based on reward later
    #mag_reward = env.reward_manager.episode_sums.get("takeoff_vel_vec_magnitude", torch.zeros(len(env_ids), device=env.device))
    
