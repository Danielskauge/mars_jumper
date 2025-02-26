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
from typing import TYPE_CHECKING

from isaaclab.envs import mdp

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

def modify_z_pos_cmd(env: ManagerBasedRLEnv, env_ids: Sequence[int], z_pos: float, num_steps: int):
    """Curriculum that modifies the magnitude for the base velocity command"""
    print(f"Curriculum: Modifying z_pos_cmd")
    if env.common_step_counter > num_steps:
        # Get the command term directly
        term = env.command_manager.get_term("pose")
        # Get current ranges and only update pos_z
        current_ranges = term.cfg.ranges
        new_ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=current_ranges.pos_x,
            pos_y=current_ranges.pos_y,
            pos_z=(0.8*z_pos, 1.2*z_pos),
            roll=current_ranges.roll,
            pitch=current_ranges.pitch,
            yaw=current_ranges.yaw
        )
        term.cfg.ranges = new_ranges
    