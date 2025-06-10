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
from typing import TYPE_CHECKING, Tuple, Dict, Any

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


def progress_reward_weights_by_metric(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    metric_name: str = "full_jump_success_rate",
    reward_weight_configs: Dict[str, Dict[str, Any]] = None,
    min_steps_between_updates: int = 100,
    smoothing_factor: float = 0.1,
) -> None:
    """Gradually increase reward weights based on a success metric.
    
    Args:
        env: The environment instance
        env_ids: Not used since all environments are affected
        metric_name: Name of the metric attribute to track (e.g., "full_jump_success_rate")
        reward_weight_configs: Dict mapping reward term names to their config:
            {
                "reward_term_name": {
                    "initial_weight": float,    # Starting weight
                    "target_weight": float,     # Final weight when metric reaches target
                    "metric_threshold": float,  # Metric value at which target_weight is reached
                    "metric_start": float,      # Metric value at which progression starts (default: 0.0)
                }
            }
        min_steps_between_updates: Minimum steps between weight updates
        smoothing_factor: How much to adjust weights per update (0-1)
    """
    
    # Initialize tracking attributes
    if not hasattr(env, "reward_weight_steps_since_update"):
        env.reward_weight_steps_since_update = 0
    if not hasattr(env, "reward_weight_initial_weights"):
        env.reward_weight_initial_weights = {}
        
    # Default configuration if none provided
    if reward_weight_configs is None:
        reward_weight_configs = {
            "attitude_landing": {
                "initial_weight": 0.05,
                "target_weight": 0.3,
                "metric_threshold": 0.8,
                "metric_start": 0.3,
            },
            "attitude_landing_trans": {
                "initial_weight": 10.0,
                "target_weight": 50.0,
                "metric_threshold": 0.8,
                "metric_start": 0.3,
            },
        }
    
    env.reward_weight_steps_since_update += 1
    
    # Only update weights periodically
    if env.reward_weight_steps_since_update < min_steps_between_updates:
        return
        
    env.reward_weight_steps_since_update = 0
    
    # Get current metric value
    current_metric = getattr(env, metric_name, 0.0)
    
    # Update weights for each configured reward term
    for reward_name, config in reward_weight_configs.items():
        try:
            # Get configuration values
            initial_weight = config["initial_weight"]
            target_weight = config["target_weight"]
            metric_threshold = config["metric_threshold"]
            metric_start = config.get("metric_start", 0.0)
            
            # Store initial weight if not already stored
            if reward_name not in env.reward_weight_initial_weights:
                try:
                    current_term_cfg = env.reward_manager.get_term_cfg(reward_name)
                    env.reward_weight_initial_weights[reward_name] = current_term_cfg.weight
                except:
                    env.reward_weight_initial_weights[reward_name] = initial_weight
                    logger.warning(f"Could not get initial weight for {reward_name}, using configured initial_weight")
            
            # Calculate progress ratio
            if current_metric <= metric_start:
                progress_ratio = 0.0
            elif current_metric >= metric_threshold:
                progress_ratio = 1.0
            else:
                progress_ratio = (current_metric - metric_start) / (metric_threshold - metric_start)
            
            # Calculate target weight based on progress
            new_weight = initial_weight + progress_ratio * (target_weight - initial_weight)
            
            # Get current weight and apply smoothing
            try:
                current_term_cfg = env.reward_manager.get_term_cfg(reward_name)
                current_weight = current_term_cfg.weight
                
                # Smooth the weight update
                smoothed_weight = current_weight + smoothing_factor * (new_weight - current_weight)
                
                # Update the weight
                current_term_cfg.weight = smoothed_weight
                env.reward_manager.set_term_cfg(reward_name, current_term_cfg)
                
                logger.debug(f"Updated {reward_name} weight: {current_weight:.4f} -> {smoothed_weight:.4f} "
                           f"(target: {new_weight:.4f}, metric: {current_metric:.3f})")
                           
            except Exception as e:
                logger.warning(f"Failed to update weight for reward term '{reward_name}': {e}")
                
        except KeyError as e:
            logger.warning(f"Missing configuration key for reward term '{reward_name}': {e}")
        except Exception as e:
            logger.warning(f"Error processing reward term '{reward_name}': {e}")
    
    return {
        "metric_value": current_metric,
        "reward_weights": {name: env.reward_manager.get_term_cfg(name).weight 
                          for name in reward_weight_configs.keys() 
                          if name in env.reward_manager.active_terms}
    }


def progress_reward_weights_by_error_metric(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    metric_name: str = "takeoff_relative_error",
    reward_weight_configs: Dict[str, Dict[str, Any]] = None,
    min_steps_between_updates: int = 100,
    smoothing_factor: float = 0.1,
) -> None:
    """Gradually increase reward weights based on an error metric (where lower values indicate better performance).
    
    Args:
        env: The environment instance
        env_ids: Not used since all environments are affected
        metric_name: Name of the error metric attribute to track (e.g., "takeoff_relative_error")
        reward_weight_configs: Dict mapping reward term names to their config:
            {
                "reward_term_name": {
                    "initial_weight": float,    # Starting weight (when error is high)
                    "target_weight": float,     # Final weight (when error is low)
                    "error_threshold": float,   # Error value at which target_weight is reached (low error)
                    "error_start": float,       # Error value at which progression starts (high error)
                }
            }
        min_steps_between_updates: Minimum steps between weight updates
        smoothing_factor: How much to adjust weights per update (0-1)
    """
    
    # Initialize tracking attributes
    if not hasattr(env, "error_weight_steps_since_update"):
        env.error_weight_steps_since_update = 0
    if not hasattr(env, "error_weight_initial_weights"):
        env.error_weight_initial_weights = {}
        
    # Default configuration if none provided
    if reward_weight_configs is None:
        reward_weight_configs = {
            "contact_forces": {
                "initial_weight": 0.01,
                "target_weight": 0.05,
                "error_threshold": 0.1,
                "error_start": 0.5,
            },
        }
    
    env.error_weight_steps_since_update += 1
    
    # Only update weights periodically
    if env.error_weight_steps_since_update < min_steps_between_updates:
        return
        
    env.error_weight_steps_since_update = 0
    
    # Get current metric value from extras log if available, otherwise from env attribute
    current_metric = env.extras.get("log", {}).get(metric_name, float('nan'))
    if torch.isnan(torch.tensor(current_metric)) or current_metric == float('nan'):
        current_metric = getattr(env, metric_name, float('nan'))
    
    # Skip update if metric is not available
    if torch.isnan(torch.tensor(current_metric)) or current_metric == float('nan'):
        logger.debug(f"Skipping curriculum update - metric '{metric_name}' not available")
        return
    
    # Update weights for each configured reward term
    for reward_name, config in reward_weight_configs.items():
        try:
            # Get configuration values
            initial_weight = config["initial_weight"]
            target_weight = config["target_weight"]
            error_threshold = config["error_threshold"]
            error_start = config.get("error_start", 1.0)
            
            # Store initial weight if not already stored
            if reward_name not in env.error_weight_initial_weights:
                try:
                    current_term_cfg = env.reward_manager.get_term_cfg(reward_name)
                    env.error_weight_initial_weights[reward_name] = current_term_cfg.weight
                except:
                    env.error_weight_initial_weights[reward_name] = initial_weight
                    logger.warning(f"Could not get initial weight for {reward_name}, using configured initial_weight")
            
            # Calculate progress ratio (inverted for error metrics)
            # When error is high (>= error_start), progress_ratio = 0.0 (use initial_weight)
            # When error is low (<= error_threshold), progress_ratio = 1.0 (use target_weight)
            if current_metric >= error_start:
                progress_ratio = 0.0
            elif current_metric <= error_threshold:
                progress_ratio = 1.0
            else:
                # Linear interpolation between error_start and error_threshold
                progress_ratio = (error_start - current_metric) / (error_start - error_threshold)
            
            # Calculate target weight based on progress
            new_weight = initial_weight + progress_ratio * (target_weight - initial_weight)
            
            # Get current weight and apply smoothing
            try:
                current_term_cfg = env.reward_manager.get_term_cfg(reward_name)
                current_weight = current_term_cfg.weight
                
                # Smooth the weight update
                smoothed_weight = current_weight + smoothing_factor * (new_weight - current_weight)
                
                # Update the weight
                current_term_cfg.weight = smoothed_weight
                env.reward_manager.set_term_cfg(reward_name, current_term_cfg)
                
                logger.debug(f"Updated {reward_name} weight: {current_weight:.4f} -> {smoothed_weight:.4f} "
                           f"(target: {new_weight:.4f}, error: {current_metric:.3f})")
                           
            except Exception as e:
                logger.warning(f"Failed to update weight for reward term '{reward_name}': {e}")
                
        except KeyError as e:
            logger.warning(f"Missing configuration key for reward term '{reward_name}': {e}")
        except Exception as e:
            logger.warning(f"Error processing reward term '{reward_name}': {e}")
    
    return {
        "metric_value": current_metric,
        "reward_weights": {name: env.reward_manager.get_term_cfg(name).weight 
                          for name in reward_weight_configs.keys() 
                          if name in env.reward_manager.active_terms}
    }


def progress_command_ranges(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    num_curriculum_levels: int = 10,
    success_rate_threshold: float = 0.4,
    min_steps_between_updates: int = 50,
    enable_regression: bool = True,
) -> None:
    """Set the command height and length ranges for the robot. 
    The initial range is set to the initial height/length range of the command term, and will be incremented linearly to the final range over the number of curriculum steps.
    
    Args:
        env: The environment instance
        env_ids: Not used since all environments are affected
        num_curriculum_levels: Number of curriculum levels to split the curriculum into from 0 to 1 percent completion from initial to final height/length range
    """  
    
    #Initialize attributes when first called
    if not hasattr(env, "cmd_curriculum_progress_ratio"): env.cmd_curriculum_progress_ratio = 0
    if not hasattr(env, "steps_since_curriculum_update"): env.steps_since_curriculum_update = 0
    if not hasattr(env, "steps_above_success_threshold"): env.steps_above_success_threshold = 0
    if not hasattr(env, "prev_running_takeoff_success_rate"):
        env.prev_running_takeoff_success_rate = getattr(env, "running_takeoff_success_rate", 0.0)
    
    progress_ratio = env.cmd_curriculum_progress_ratio
    
    # Update height ranges
    initial_height_range = env.cfg.command_ranges.height_range
    final_height_range = env.cfg.command_ranges.curriculum_final_height_range
    current_height_min = initial_height_range[0] + progress_ratio * (final_height_range[0] - initial_height_range[0])
    current_height_max = initial_height_range[1] + progress_ratio * (final_height_range[1] - initial_height_range[1])
    env.cmd_height_range = (current_height_min, current_height_max)
    
    # Update length ranges
    initial_length_range = env.cfg.command_ranges.length_range
    final_length_range = env.cfg.command_ranges.curriculum_final_length_range
    current_length_min = initial_length_range[0] + progress_ratio * (final_length_range[0] - initial_length_range[0])
    current_length_max = initial_length_range[1] + progress_ratio * (final_length_range[1] - initial_length_range[1])
    env.cmd_length_range = (current_length_min, current_length_max)
    
    # Update derived pitch/magnitude ranges for backward compatibility
    env.cmd_pitch_range = env.cfg.command_ranges.pitch_range
    env.cmd_magnitude_range = env.cfg.command_ranges.magnitude_range
    
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
            print("Advancing takeoff command curriculum: current_step_counter %s, progress ratio %s, height=[%s, %s], length=[%s, %s]", env.common_step_counter, env.cmd_curriculum_progress_ratio, current_height_min, current_height_max, current_length_min, current_length_max)
    
    if enable_regression:
        # Try to regress curriculum (if not progressed)
        if not progressed_this_cycle and current_success_rate < success_rate_threshold and env.cmd_curriculum_progress_ratio > 0:
            env.cmd_curriculum_progress_ratio -= 1/num_curriculum_levels
            env.cmd_curriculum_progress_ratio = max(0, env.cmd_curriculum_progress_ratio) # Ensure not < 0
            env.steps_since_curriculum_update = 0
            # progressed_this_cycle = True # This was for the if not progressed_this_cycle condition, not needed to set true here
            print("Decreasing takeoff command curriculum: current_step_counter %s, progress ratio %s, height=[%s, %s], length=[%s, %s]", env.common_step_counter, env.cmd_curriculum_progress_ratio, current_height_min, current_height_max, current_length_min, current_length_max)

    env.prev_running_takeoff_success_rate = current_success_rate
        
    # Need to return state to be logged
    return {
        "progress_ratio": env.cmd_curriculum_progress_ratio,
        "cmd_height_min": current_height_min,
        "cmd_height_max": current_height_max,
        "cmd_length_min": current_length_min,
        "cmd_length_max": current_length_max,
    }
    
    # Might change success based on reward later
    #mag_reward = env.reward_manager.episode_sums.get("takeoff_vel_vec_magnitude", torch.zeros(len(env_ids), device=env.device))
    

def enable_reward_on_error_threshold(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    metric_name: str,
    reward_name: str,
    initial_weight: float,
    target_weight: float,
    error_threshold: float,
) -> Dict[str, float]:
    """Step reward weight from initial to target when error metric dips below a threshold."""
    # Initialize enabled set
    if not hasattr(env, "enabled_rewards"):
        env.enabled_rewards = set()
    # Get current metric
    raw_metric = getattr(env, metric_name, None)
    if raw_metric is None:
        return {}
    if isinstance(raw_metric, torch.Tensor):
        if raw_metric.numel() == 0:
            return {}
        current_metric = raw_metric.mean().item()
    else:
        current_metric = float(raw_metric)
    # Get term config
    try:
        term_cfg = env.reward_manager.get_term_cfg(reward_name)
    except ValueError:
        logger.warning(f"Reward term '{reward_name}' not found")
        return {}
    # Set weight based on threshold, only once
    if reward_name not in env.enabled_rewards:
        if current_metric <= error_threshold:
            new_weight = target_weight
            env.enabled_rewards.add(reward_name)
        else:
            new_weight = initial_weight
        term_cfg.weight = new_weight
        env.reward_manager.set_term_cfg(reward_name, term_cfg)
    # Log current weight
    current_weight = env.reward_manager.get_term_cfg(reward_name).weight
    return {f"curriculum/{reward_name}_weight": current_weight}
    
