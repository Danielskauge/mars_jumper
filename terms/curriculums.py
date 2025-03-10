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


def change_command_ranges(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    num_curriculum_steps: int = 10,
    final_magnitude_range: Tuple[float, float] = (4.0, 5.0),
    steps_per_increment: int = 10,
) -> None:
    """Set the command takeoff vector magnitude ranges for the robot. 
    The initial range is set to the initial magnitude range of the command term, and will be incremented linearly to the final range over the number of curriculum steps.
    
    Args:
        env: The environment instance
        env_ids: Not used since all environments are affected
        num_steps: Number of steps to split the curriculum into from 0 to 1 percent completion from initial to final magnitude range
        final_magnitude_range: Final range for takeoff velocity magnitude
        steps_per_increment: Number of steps between curriculum increments
        
    
    """  
    
    initial_magnitude_range = env.command_manager.get_term("takeoff_vel_vec").cfg.ranges.magnitude

    progress_ratio: torch.float32 = env._takeoff_vel_magnitude_curriculum_progress_ratio

    current_mag_min: torch.float32 = initial_magnitude_range[0] + progress_ratio * (final_magnitude_range[0] - initial_magnitude_range[0])
    current_mag_max: torch.float32 = initial_magnitude_range[1] + progress_ratio * (final_magnitude_range[1] - initial_magnitude_range[1])
    
    takeoff_vel_cmd = env.command_manager.get_term("takeoff_vel_vec")
    takeoff_vel_cmd.cfg.magnitude_range = (current_mag_min, current_mag_max) #TODO make sure namign is correct
    
    increment_curriculum: bool = (env.common_step_counter % steps_per_increment == 0 and env.common_step_counter != 0)
    
    if increment_curriculum:
        env._takeoff_vel_magnitude_curriculum_progress_ratio += 1/num_curriculum_steps
        print("Advancing takeoff magnitude curriculum: current_step_counter %s, progress ratio %s, mag=[%s, %s]", env.common_step_counter, env._takeoff_vel_magnitude_curriculum_progress_ratio, current_mag_min, current_mag_max)

        
    # Need to return state to be logged
    return {
        "takeoff_vel_magnitude_curriculum_progress_ratio": env._takeoff_vel_magnitude_curriculum_progress_ratio,
        "takeoff_vel_magnitude_min": current_mag_min,
        "takeoff_vel_magnitude_max": current_mag_max,
    }
    
    # Might change success based on reward later
    #mag_reward = env.reward_manager.episode_sums.get("takeoff_vel_vec_magnitude", torch.zeros(len(env_ids), device=env.device))
    
