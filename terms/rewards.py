# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
from enum import IntEnum
import torch
from typing import TYPE_CHECKING, Literal
import numpy as np
import logging
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from terms.phase import Phase
from terms.utils import get_center_of_mass_lin_vel, convert_command_to_euclidean_vector, convert_height_length_to_pitch_magnitude, get_dynamic_takeoff_vector
from collections.abc import Sequence

logger = logging.getLogger(__name__)

class Kernel(IntEnum):
    INVERSE_LINEAR = 0
    INVERSE_SQUARE = 1
    EXPONENTIAL = 2
    LINEAR = 3
    SQUARE = 4
    SHIFTED_SQUARE = 5

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from envs.takeoff_env import MarsJumperEnv
    
def shifted_huber_kernel(e, delta, e_max):
    """
    Shifted Huber kernel for reward shaping.
    
    Args:
        e: Error values (tensor)
        delta: Error threshold at which huber loss goes from quadratic to linear (in actual error units)
        e_max: Maximum error allowed (in actual error units) - determines where reward becomes 0
        
    Returns:
        Reward values where:
        - e=0 gives maximum reward
        - e=e_max gives reward=0
        - Transition from quadratic to linear penalty at e=delta
        
    Note: Use the 'weight' parameter in RewardTermCfg to scale the magnitude
    """
    use_quadratic_term = torch.abs(e) <= delta
    linear_term = torch.abs(e) - delta/2
    quadratic_term = 0.5 * e**2 / delta
    huber = torch.where(use_quadratic_term, quadratic_term, linear_term)
    
    # Calculate the shift constant to make reward=0 at e=e_max
    c_shift = 0.5 * e_max**2 / delta if e_max <= delta else abs(e_max) - delta/2
    return c_shift - huber

def relative_cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float, bias: float = 0.0) -> torch.Tensor:
    """Computes reward based on relative error between robot's velocity and commanded takeoff vector.
    
    Similar to cmd_error() but normalizes error by command magnitude to be scale-invariant.
    Only active during takeoff phase. Returns values in [0,1] where 1 means perfect tracking.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        bias: Bias term for shifted square kernel.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * relative_error) 
            - "inverse_linear": 1/(1 + scale * relative_error)
            - "shifted_square": bias - relative_error**2
    Returns:
        Reward tensor of shape (num_envs,)
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    if not torch.any(takeoff_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    takeoff_env_ids = takeoff_envs.nonzero(as_tuple=False).squeeze(-1)
    takeoff_vector = _get_dynamic_takeoff_vector(env, takeoff_env_ids)
    robot_vel_vec = get_center_of_mass_lin_vel(env)[takeoff_envs]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(robot_vel_vec - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    takeoff_rewards = torch.zeros_like(relative_error)
    if kernel == Kernel.EXPONENTIAL:
        takeoff_rewards = torch.exp(-scale*relative_error)
    elif kernel == Kernel.INVERSE_LINEAR:
        takeoff_rewards = 1/(1 + scale*relative_error)
    elif kernel == Kernel.INVERSE_SQUARE:
        takeoff_rewards = 1/(1 + scale*relative_error**2)
    elif kernel == Kernel.SHIFTED_SQUARE:
        takeoff_rewards = bias - relative_error**2

    # Check vertical velocity (assuming index 2 is vertical)
    vertical_velocity = robot_vel_vec[:, 2]
    downward_mask = vertical_velocity < 0

    takeoff_rewards[downward_mask] = 0

    reward_tensor[takeoff_envs] = takeoff_rewards
    return reward_tensor

def landing_base_vertical_vel_l1(env: ManagerBasedRLEnv) -> torch.Tensor:
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_mask = env.landing_mask # Or any other relevant mask
    if torch.any(active_mask): # Guard against empty mask
        # Ensure the RHS is also indexed by the mask if it's a full tensor
        velocities = env.robot.data.root_com_lin_vel_w[:, 2]
        reward_tensor[active_mask] = torch.abs(velocities[active_mask])
    return reward_tensor

def landing_abduction_zero_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the robot for having the abduction joint at zero position, linear kernel. Use negative weight."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_mask = env.landing_mask
    if torch.any(active_mask):
        abduction_pos = env.robot.data.joint_pos[active_mask, env.abduction_joint_idx]
        reward_tensor[active_mask] = torch.sum(torch.abs(abduction_pos), dim=-1)
    return reward_tensor
    
def landing_foot_ground_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """+1 reward for each foot in contact with the ground"""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)

    # If no environments are in the landing phase, return zeros
    if not torch.any(env.landing_mask):
        return reward_tensor

    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_sensor").name]
    feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
    feet_forces = contact_sensor.data.net_forces_w[:, feet_idx]
    is_foot_in_contact = torch.norm(feet_forces, dim=-1) > contact_sensor.cfg.force_threshold  # Shape: (num_envs, num_feet)
    
    # Calculate sum of contacts for ALL envs first
    sum_contact_values_all_envs = torch.sum(is_foot_in_contact, dim=-1).float()  # Shape: (num_envs,)
    
    # Apply this sum ONLY to the environments that are actually in the landing_mask
    reward_tensor[env.landing_mask] = sum_contact_values_all_envs[env.landing_mask]
    return reward_tensor

def contact_forces(env: ManagerBasedRLEnv, phases: list[Phase], kernel: Literal[Kernel.LINEAR, Kernel.SQUARE]) -> torch.Tensor:
    """Penalize the contact forces of the robot except for the feet, using L2 squared kernel."""
    
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
    if not torch.any(active_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    net_contact_forces = env.contact_sensor.data.net_forces_w[:, env.bodies_except_feet_idx] 
    forces_magnitude = torch.norm(net_contact_forces, dim=-1) #shape [num_envs, num_bodies]
    sum_forces_magnitude = torch.sum(forces_magnitude, dim=-1) #shape [num_envs]
    if kernel == Kernel.LINEAR:
        reward_tensor[active_envs] = sum_forces_magnitude[active_envs]
    elif kernel == Kernel.SQUARE:
        reward_tensor[active_envs] = torch.square(sum_forces_magnitude[active_envs])
    
    return reward_tensor
    
    
def contact_forces_potential_based(env: ManagerBasedRLEnv, 
                             sensor_cfg: SceneEntityCfg, 
                             phases: list[Phase], 
                             kernel: Literal[Kernel.LINEAR, Kernel.SQUARE],
                             potential_buffer_postfix: str) -> torch.Tensor:
    """Calculates potential-based reward shaping for contact forces.

    The potential P(s) is defined as the current penalty V(s) (sum of forces).
    The reward component from this function is P(s') - P(s) = V(s') - V(s).
    If V is a cost (higher is worse), a negative weight in RewardTermCfg is needed.
    This function dynamically manages a buffer on the `env` object to store V(s).
    """
    buffer_name = "_prev_potential_contact_forces_" + potential_buffer_postfix
    # Initialize buffer on env if it doesn't exist
    if not hasattr(env, buffer_name):
        setattr(env, buffer_name, torch.zeros(env.num_envs, device=env.device))
    
    previous_penalty_values = getattr(env, buffer_name)

    # Determine active environments based on phases
    active_for_current_phase = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    active_for_prev_phase = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        active_for_current_phase |= (env.jump_phase == phase)
        active_for_prev_phase |= (env.prev_jump_phase == phase)
        
    if not torch.any(active_for_current_phase):
        return torch.zeros(env.num_envs, device=env.device)

    # Calculate current penalty V(s') for ALL environments
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    forces_magnitude = torch.norm(net_contact_forces, dim=-1)  # shape [num_envs, num_bodies_in_sensor_cfg]
    current_penalty_values = torch.sum(forces_magnitude, dim=-1)  # shape [num_envs]

    if kernel == Kernel.SQUARE:
        current_penalty_values = torch.square(current_penalty_values)
    elif kernel != Kernel.LINEAR:
        raise ValueError(f"Unsupported kernel type: {kernel} for contact_forces reward term.")

    # Mask for environments where the reward was active in the previous phase AND is active in the current phase
    continuously_active_mask = active_for_current_phase & active_for_prev_phase

    # Initialize reward shaping component to zeros.
    reward_shaping_component = torch.zeros_like(current_penalty_values)

    # Calculate reward V(s_curr) - V(s_prev) only for continuously active environments.
    # For newly active environments (active_for_current_phase & ~active_for_prev_phase),
    # reward_shaping_component remains 0 because they are not in continuously_active_mask.
    # For environments not active in current_s_begin, it also remains 0.
    if torch.any(continuously_active_mask):
        reward_shaping_component[continuously_active_mask] = \
            current_penalty_values[continuously_active_mask] - \
            previous_penalty_values[continuously_active_mask]

    # Buffer update: store V(s_current_end) for the next step.
    # For environments that are resetting in this step, their "previous value" for the *next* episode
    # should reflect an initial state (e.g., 0 for contact forces).
    # env.reset_buf is set by TerminationManager before RewardManager.compute() is called.
    next_step_prev_potential_values = current_penalty_values.clone().detach()
    if hasattr(env, 'reset_buf'): # ManagerBasedRLEnv has reset_buf
        reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            next_step_prev_potential_values[reset_env_ids] = 0.0  # Assuming 0 is the initial potential for reset states
    
    setattr(env, buffer_name, next_step_prev_potential_values)

    return reward_shaping_component
    
def action_rate_l2(env: ManagerBasedRLEnv, phases: list[Phase]) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
        
    reward_tensor[active_envs] = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward_tensor
    
def landing_base_height(env: ManagerBasedRLEnv,
                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                           sensor_cfg: SceneEntityCfg | None = None,
                           target_height: float = 0.0,
                           kernel: Kernel = Kernel.INVERSE_LINEAR,
                           scale: float = 1.0) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    landing_envs = env.jump_phase == Phase.LANDING
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    if kernel == Kernel.INVERSE_LINEAR:
        reward_tensor[landing_envs] = 1 / (1 + scale * (asset.data.root_pos_w[landing_envs, 2] - adjusted_target_height))
    elif kernel == Kernel.INVERSE_SQUARE:
        reward_tensor[landing_envs] = 1 / (1 + scale * (asset.data.root_pos_w[landing_envs, 2] - adjusted_target_height)**2)
    elif kernel == Kernel.EXPONENTIAL:
        reward_tensor[landing_envs] = torch.exp(-scale * (asset.data.root_pos_w[landing_envs, 2] - adjusted_target_height))
    elif kernel == Kernel.LINEAR:
        reward_tensor[landing_envs] = (asset.data.root_pos_w[landing_envs, 2] - adjusted_target_height)
    elif kernel == Kernel.SQUARE:
        reward_tensor[landing_envs] = (asset.data.root_pos_w[landing_envs, 2] - adjusted_target_height)**2
    
    return reward_tensor
    
def joint_vel_l1(env: ManagerBasedRLEnv, phases: list[Phase]) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
        

    asset_cfg = SceneEntityCfg("robot")
    asset: Articulation = env.scene[asset_cfg.name]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[active_envs] = torch.sum(torch.abs(asset.data.joint_vel[active_envs, asset_cfg.joint_ids]), dim=1)
    return reward_tensor

def action_rate(env: ManagerBasedRLEnv, phases: list[Phase], kernel: Kernel, scale: float = 1.0) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
        
    if kernel == Kernel.SQUARE:
        reward_tensor[active_envs] = torch.sum(torch.square(env.action_manager.action[active_envs] - env.action_manager.prev_action[active_envs]), dim=1)
    elif kernel == Kernel.LINEAR:
        reward_tensor[active_envs] = torch.sum(torch.abs(env.action_manager.action[active_envs] - env.action_manager.prev_action[active_envs]), dim=1)
    elif kernel == Kernel.EXPONENTIAL:
        reward_tensor[active_envs] = torch.sum(torch.exp(-scale * (env.action_manager.action[active_envs] - env.action_manager.prev_action[active_envs])), dim=1)
    elif kernel == Kernel.INVERSE_SQUARE:
        reward_tensor[active_envs] = torch.sum(1/(1 + scale * (env.action_manager.action[active_envs] - env.action_manager.prev_action[active_envs])**2), dim=1)
    return reward_tensor


    
def cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float) -> torch.Tensor:
    """Computes reward based on absolute error between robot's velocity and commanded takeoff vector.
    
    Only active during takeoff phase. Returns values in [0,1] where 1 means perfect tracking.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * error) 
            - "inverse_linear": 1/(1 + scale * error)
            - "inverse_linear": 1/(1 + scale * error)
            
    Returns:
        Reward tensor of shape (num_envs,)
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    cmd = _get_command_as_pitch_magnitude(env, takeoff_envs.nonzero(as_tuple=False).squeeze(-1))
    cmd_vec = convert_command_to_euclidean_vector(cmd)
    robot_vel_vec = get_center_of_mass_lin_vel(env)[takeoff_envs]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    absolute_error = torch.norm(robot_vel_vec - cmd_vec, dim=-1)
    takeoff_rewards = torch.zeros_like(absolute_error)
    if kernel == Kernel.EXPONENTIAL:
        takeoff_rewards = torch.exp(-scale*absolute_error)
    elif kernel == Kernel.INVERSE_LINEAR:
        takeoff_rewards = 1/(1 + scale*absolute_error)
    elif kernel == Kernel.INVERSE_SQUARE:
        takeoff_rewards = 1/(1 + scale*absolute_error**2)

    # Check vertical velocity (assuming index 2 is vertical)
    vertical_velocity = robot_vel_vec[:, 2]
    downward_mask = vertical_velocity < 0

    # Zero out rewards for takeoff envs moving downwards
    takeoff_rewards[downward_mask] = 0

    # Assign the computed rewards back to the correct indices
    reward_tensor[takeoff_envs] = takeoff_rewards
    
    return reward_tensor

def liftoff_relative_cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float, bias: float = 0.0) -> torch.Tensor:
    """Computes reward based on relative error between robot's velocity and commanded takeoff vector, ONLY at the moment of liftoff.
    
    Similar to relative_cmd_error() but only active at the transition from TAKEOFF to FLIGHT.
    Returns values in [0,1] where 1 means perfect tracking, for appropriate kernels.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        bias: Bias term for shifted square kernel.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * relative_error) 
            - "inverse_linear": 1/(1 + scale * relative_error)
            - "shifted_square": bias - relative_error**2
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Identify environments that transitioned from TAKEOFF to FLIGHT in this step
    transitioned_to_flight = (env.prev_jump_phase == Phase.TAKEOFF) & (env.jump_phase == Phase.FLIGHT)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)

    if not torch.any(transitioned_to_flight):
        return reward_tensor # Return zeros if no envs transitioned
    
    transitioned_env_ids = transitioned_to_flight.nonzero(as_tuple=False).squeeze(-1)
    takeoff_vector = _get_dynamic_takeoff_vector(env, transitioned_env_ids)
    robot_vel_vec = get_center_of_mass_lin_vel(env)[transitioned_to_flight]
    
    takeoff_vector_norm = torch.norm(takeoff_vector, dim=-1)
    # Avoid division by zero if command magnitude can be zero.
    # For takeoff commands, magnitude should generally be positive.
    safe_takeoff_vector_norm = torch.where(takeoff_vector_norm < 1e-6, torch.ones_like(takeoff_vector_norm), takeoff_vector_norm)
    
    relative_error = torch.norm(robot_vel_vec - takeoff_vector, dim=-1) / safe_takeoff_vector_norm
    
    liftoff_rewards = torch.zeros_like(relative_error)
    if kernel == Kernel.EXPONENTIAL:
        liftoff_rewards = torch.exp(-scale*relative_error)
    elif kernel == Kernel.INVERSE_LINEAR:
        liftoff_rewards = 1/(1 + scale*relative_error)
    elif kernel == Kernel.INVERSE_SQUARE:
        liftoff_rewards = 1/(1 + scale*relative_error**2)
    elif kernel == Kernel.SHIFTED_SQUARE:
        liftoff_rewards = bias - relative_error**2

    reward_tensor[transitioned_to_flight] = liftoff_rewards
    return reward_tensor

def flat_orientation(env: ManagerBasedRLEnv, phases: list[Phase]) -> torch.Tensor:
    """ Penalize the xy-components of the projected gravity vector. Sum of squared values.
    Returns postive value, use negative weight.
    """
    asset: RigidObject = env.scene[SceneEntityCfg("robot").name]
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
        
    if not torch.any(active_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[active_envs] = torch.sum(torch.square(asset.data.projected_gravity_b[active_envs, :2]), dim=1)
    
    return reward_tensor

def left_right_joint_symmetry(
    env: ManagerBasedRLEnv,
    phases: list[Phase],
    scale: float = 4.0
) -> torch.Tensor:
    """ Reward the robot for maintaining left-right joint angle symmetry using angle differences.

    Calculates the difference between corresponding joints on diagonal legs
    (LF-RH and RF-LH pairs) for knee, hip flexion, and abductor joints.
    The reward encourages these differences to be close to zero.

    The reward is calculated as: exp(-scale * penalty), where penalty is either
    the sum of absolute differences ('linear') or the sum of squared differences ('square')
    across the 6 joint pairs.

    Active in specified phases (e.g., landing, takeoff, crouch), but potentially not
    in flight phase to allow for attitude control during aerial maneuvers.

    Args:
        env: The environment instance.
        phases: List of phases during which this reward is active.
        reward_type: How to measure the difference ('linear' for absolute, 'square' for squared).
        scale: Scaling factor for the exponential reward shaping. Higher values penalize deviations more.

    Returns:
        reward_tensor: Shape (num_envs,), normalized between (0, 1] where 1 means perfect symmetry.
    """
    asset: Articulation = env.scene[SceneEntityCfg("robot").name]
    active_envs_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        active_envs_mask |= (env.jump_phase == phase)

    reward_tensor = torch.zeros(env.num_envs, device=env.device)

    if not torch.any(active_envs_mask):
        return reward_tensor

    # Find joint indices
    LF_HFE_idx, _ = asset.find_joints(".*LF_HFE.*")
    RH_HFE_idx, _ = asset.find_joints(".*RH_HFE.*")
    RF_HFE_idx, _ = asset.find_joints(".*RF_HFE.*")
    LH_HFE_idx, _ = asset.find_joints(".*LH_HFE.*")

    LF_KFE_idx, _ = asset.find_joints(".*LF_KFE.*")
    RH_KFE_idx, _ = asset.find_joints(".*RH_KFE.*")
    RF_KFE_idx, _ = asset.find_joints(".*RF_KFE.*")
    LH_KFE_idx, _ = asset.find_joints(".*LH_KFE.*")

    LF_HAA_idx, _ = asset.find_joints(".*LF_HAA.*")
    RH_HAA_idx, _ = asset.find_joints(".*RH_HAA.*")
    RF_HAA_idx, _ = asset.find_joints(".*RF_HAA.*")
    LH_HAA_idx, _ = asset.find_joints(".*LH_HAA.*")

    # Get joint positions for active environments
    q_pos = asset.data.joint_pos[active_envs_mask]

    # Calculate differences for diagonal pairs (squeeze to remove the last dim)
    diff_front_knee = (q_pos[:, LF_KFE_idx] - q_pos[:, RH_KFE_idx]).squeeze(-1)
    diff_back_knee = (q_pos[:, RF_KFE_idx] - q_pos[:, LH_KFE_idx]).squeeze(-1)
    diff_front_hfe = (q_pos[:, LF_HFE_idx] - q_pos[:, RH_HFE_idx]).squeeze(-1)
    diff_back_hfe = (q_pos[:, RF_HFE_idx] - q_pos[:, LH_HFE_idx]).squeeze(-1)
    diff_front_haa = (q_pos[:, LF_HAA_idx] - q_pos[:, RH_HAA_idx]).squeeze(-1)
    diff_back_haa = (q_pos[:, RF_HAA_idx] - q_pos[:, LH_HAA_idx]).squeeze(-1)

    penalty = (
        torch.square(diff_front_knee) + torch.square(diff_back_knee) +
        torch.square(diff_front_hfe) + torch.square(diff_back_hfe) +
        torch.square(diff_front_haa) + torch.square(diff_back_haa)
    )
    # Calculate exponential reward
    symmetry_reward = torch.exp(-scale * penalty)

    reward_tensor[active_envs_mask] = symmetry_reward

    return reward_tensor

def base_height_l1(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L1 kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L1 penalty
    return torch.abs(env.robot.data.root_pos_w[:, 2] - adjusted_target_height)


def feet_ground_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Make the robot maintain ground contact. Active in crouch and landing phases
    Return:
        reward_tensor: Shape (num_envs,), Value between 0 and 1, +0.25 for each foot in contact with the ground
    """
    #landing_envs = env.jump_phase == Phase.LANDING
    #crouch_envs = env.jump_phase == Phase.CROUCH
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_sensor").name]
    feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    #active_envs = landing_envs | crouch_envs
    
    #if not torch.any(active_envs):
    #    return torch.zeros(env.num_envs, device=env.device)
    
    # Check if feet are in contact with the ground
    feet_forces = contact_sensor.data.net_forces_w[:, feet_idx] #tensor [num_envs, num_feet, 3]
    is_foot_in_contact = torch.norm(feet_forces, dim=-1) > contact_sensor.cfg.force_threshold #bool tensor [num_envs, num_feet]
    sum_feet_in_contact = torch.mean(is_foot_in_contact.float(), dim=-1) #scalar tensor [num_envs]
    
    reward_tensor = sum_feet_in_contact
    
    return reward_tensor
    
def landing_com_accel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize excessive center of mass acceleration during landing phase.
    
    This reward encourages smooth landings by penalizing high accelerations
    of the robot's center of mass during the landing phase.
    
    Args:
        env: Environment instance
        
    Returns:
        Tensor of shape (num_envs,) containing acceleration penalties
        
    Note:
        The acceleration is computed using finite differences from velocity data.
        Some acceleration is expected during landing due to impact forces.
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    landing_envs = env.jump_phase == Phase.LANDING
    
    if not torch.any(landing_envs):
        return reward_tensor
    
    # Get current and previous center of mass velocities
    current_com_vel = get_center_of_mass_lin_vel(env)[landing_envs]
    
    # Estimate acceleration using finite differences (requires velocity history)
    if hasattr(env, 'prev_com_vel') and env.prev_com_vel is not None:
        prev_com_vel = env.prev_com_vel[landing_envs]
        dt = env.physics_dt if hasattr(env, 'physics_dt') else 1/60  # Default to 60Hz
        
        # Calculate acceleration magnitude
        com_accel = (current_com_vel - prev_com_vel) / dt
        accel_magnitude = torch.norm(com_accel, dim=-1)
        
        # Penalize high accelerations (landing will naturally have some acceleration)
        # Use a reasonable threshold for "excessive" acceleration
        accel_threshold = 20.0  # m/s^2 (roughly 2g)
        excess_accel = torch.clamp(accel_magnitude - accel_threshold, min=0.0)
        
        reward_tensor[landing_envs] = -excess_accel  # Negative reward for penalty
    
    # Store current velocity for next step
    env.prev_com_vel = get_center_of_mass_lin_vel(env).clone()
    
    return reward_tensor
    
def feet_ground_impact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Reward the robot for landing with low impact force on the feet.
    
    The reward is calculated as:
    r = exp(-||f||²)
    
    where ||f|| is the L2 norm of the impact force on the feet over the landing phase.
    - Zero force → reward of 1.0
    - High force → reward approaches 0.0
    """
    landing_envs = env.jump_phase == Phase.LANDING
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
    feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
    forces = contact_sensor.data.net_forces_w[landing_envs, feet_idx] 
    force_norm = torch.norm(forces, dim=-1)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[landing_envs] = torch.exp(-torch.square(force_norm)) #TODO: consider changing the reward shaping. Want to avoid the exponentail function "hidding" peaks
    
    return reward_tensor
    
def crouch_knee_angle(env: ManagerBasedRLEnv, target_angle_rad: float, reward_type: str = "cosine") -> torch.Tensor:
    """ Reward knee angles similar to the target angle to load springs. Use positive weight, for all reward types.
    Args:
        target_angle_rad: The target knee angle in radians, [0, pi], (should be near 0 pi for full flexing)
        reward_type: "cosine" or "absolute" or "square"
    Returns:
        reward_tensor: Shape (num_envs,), bouded [0, 1]
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    crouch_envs = env.jump_phase == Phase.CROUCH
    knee_joints_idx, _ = env.robot.find_joints(".*KFE")
    
    angle_rad = env.robot.data.joint_pos[crouch_envs, :][:, torch.tensor(knee_joints_idx)]
    angle_diff_abs = torch.abs(angle_rad - target_angle_rad)
    angle_diff_abs_normalized = angle_diff_abs / torch.pi  # normalize to 0-1, max error is pi
    
    if reward_type == "cosine":
        cosine_similarity = (torch.cos(angle_diff_abs) + 1) / 2
        reward_tensor[crouch_envs] = torch.sum(cosine_similarity, dim=-1) / 4
    
    elif reward_type == "absolute":
        reward_tensor[crouch_envs] = -torch.sum(angle_diff_abs_normalized, dim=-1) / 4
        print(reward_tensor[crouch_envs])
        
    elif reward_type == "square":
        reward_tensor[crouch_envs] = -torch.sum(torch.square(angle_diff_abs_normalized), dim=-1) / 4
    
    return reward_tensor

def crouch_hip_angle(env: ManagerBasedRLEnv, target_angle_rad: float, reward_type: str = "cosine") -> torch.Tensor:
    """ Reward hip angles similar to the target angle to load springs. Use positive weight, for all reward types.  
    Args:
        target_angle_rad: The target angle in radians, [0, pi], (should be near 0 pi for full flexing)
        reward_type: "cosine" or "absolute" or "square"
    Returns:
        reward_tensor: Shape (num_envs,), bouded [0, 1]
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    crouch_envs = env.jump_phase == Phase.CROUCH
    hip_joints_idx, _ = env.robot.find_joints(".*HFE.*")
    
    angle_rad = env.robot.data.joint_pos[crouch_envs, :][:, torch.tensor(hip_joints_idx)]
    angle_diff_abs= torch.abs(angle_rad - target_angle_rad)
    angle_diff_abs_normalized = angle_diff_abs / torch.pi #normalize to 0-1, max error is pi
    
    if reward_type == "cosine":
        cosine_similarity = (torch.cos(angle_diff_abs) + 1) / 2
        reward_tensor[crouch_envs] = torch.mean(cosine_similarity, dim=-1)

    elif reward_type == "absolute":
        reward_tensor[crouch_envs] = -torch.mean(angle_diff_abs_normalized, dim=-1)
        
    elif reward_type == "square":
        reward_tensor[crouch_envs] = -torch.mean(torch.square(angle_diff_abs_normalized), dim=-1) 
    
    return reward_tensor

def crouch_abductor_angle(env: ManagerBasedRLEnv, target_angle_rad: float, reward_type: str = "cosine") -> torch.Tensor:
    """ Reward abductor angles similar to the target angle to load springs. Use positive weight, for all reward types.
    Args:
        target_angle_rad: The target abductor angle in radians, [0, pi], (should be near 0 pi for full flexing)
        reward_type: "cosine" or "absolute" or "square"
    Returns:
        reward_tensor: Shape (num_envs,), bouded [0, 1]
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    crouch_envs = env.jump_phase == Phase.CROUCH
    abductor_joints_idx, _ = env.robot.find_joints(".*HAA.*")
    
    angle_rad = env.robot.data.joint_pos[crouch_envs, :][:, torch.tensor(abductor_joints_idx)]
    angle_diff_abs = torch.abs(angle_rad - target_angle_rad)
    angle_diff_abs_normalized = angle_diff_abs / torch.pi  # normalize to 0-1, max error is pi
    
    if reward_type == "cosine":
        cosine_similarity = (torch.cos(angle_diff_abs) + 1) / 2
        reward_tensor[crouch_envs] = torch.sum(cosine_similarity, dim=-1) / 4
    
    elif reward_type == "absolute":
        reward_tensor[crouch_envs] = -torch.sum(angle_diff_abs_normalized, dim=-1) / 4
        
    elif reward_type == "square":
        reward_tensor[crouch_envs] = -torch.sum(torch.square(angle_diff_abs_normalized), dim=-1) / 4
    
    return reward_tensor
        
def takeoff_angle_error(env: ManagerBasedRLEnv, scale: float | int = 1.0) -> torch.Tensor:
    """Rewards for small velocity vector angular misalignment during takeoff phase.

    Normalized by: exp(-scale * angle_error)
    Filtered by: magnitude must be at least 50% of commanded magnitude.
    Only active during takeoff phase.

    Returns:
        reward: 1.0 for perfect alignment, approaches 0 for large misalignment.
    """
    num_envs = env.num_envs
    device = env.device
    reward = torch.zeros(num_envs, device=env.device)
    takeoff_mask = env.jump_phase == Phase.TAKEOFF
    if not torch.any(takeoff_mask):
        return reward

    # Extract commanded and actual velocities for takeoff envs
    takeoff_env_ids = takeoff_mask.nonzero(as_tuple=False).squeeze(-1)
    takeoff_vector = _get_dynamic_takeoff_vector(env, takeoff_env_ids)         # [M,3]
    act_vec = get_center_of_mass_lin_vel(env)[takeoff_mask]    # [M,3]

    # Compute norms
    takeoff_vector_norm = takeoff_vector.norm(dim=1)     # [M]
    act_norm = act_vec.norm(dim=1)     # [M]

    # Mask for sufficient velocity magnitude
    valid = act_norm >= 0.5 * takeoff_vector_norm  # [M]

    # Normalize vectors (avoid division by zero)
    takeoff_vector_unit = takeoff_vector / takeoff_vector_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]
    act_unit = act_vec / act_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]

    # Compute angle error
    dot = (takeoff_vector_unit * act_unit).sum(dim=1).clamp(-1.0, 1.0)      # [M]
    angle_err = torch.acos(dot)                                # [M]

    # Exponential scaling and apply validity mask
    scaled_err = torch.exp(-scale * angle_err) * valid.to(torch.float32)  # [M]

    # Assign back to the global reward tensor
    reward[takeoff_mask] = scaled_err
    return reward

def upward_velocity(env: ManagerBasedRLEnv, shape: str = "linear") -> torch.Tensor:
    """Reward upward velocity of the base, but only during the takeoff phase."""
    takeoff_envs = env.jump_phase == Phase.TAKEOFF # Use env.jump_phase
    
    reward = torch.zeros(env.num_envs, device=env.device)
    
    # Only calculate and apply reward for environments in the takeoff phase
    if torch.any(takeoff_envs):
        # Extract upward velocity component for active environments
        upward_vel = env.robot.data.root_com_lin_vel_w[takeoff_envs, 2]
        # Apply reward where velocity is positive
        reward[takeoff_envs] = torch.clamp(upward_vel, min=0.0)
        
        if shape == "square":
            reward[takeoff_envs] = torch.square(reward[takeoff_envs])
       
    return reward

def landing_orientation(
    env: ManagerBasedRLEnv, target_orientation_quat: torch.Tensor, actual_orientation_quat: torch.Tensor
) -> torch.Tensor:
    """ Difference between the commanded and actual landing orientation measured by quaternion logarithm
    
    Args:
        target_orientation_quat: Shape (num_envs, 4)
        actual_orientation_quat: Shape (num_envs, 4)
    """
    
    assert torch.all(torch.isclose(torch.norm(target_orientation_quat, dim=-1), torch.ones_like(torch.norm(target_orientation_quat, dim=-1)))), "Target orientation quaternion must be normalized"
    assert torch.all(torch.isclose(torch.norm(actual_orientation_quat, dim=-1), torch.ones_like(torch.norm(actual_orientation_quat, dim=-1)))), "Actual orientation quaternion must be normalized"

    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)

    target_conj = torch.cat([target_orientation_quat[:, :1], -target_orientation_quat[:, 1:]], dim=-1)
    q_rel = quat_mul(actual_orientation_quat, target_conj)
    
    assert torch.all(torch.isclose(torch.norm(q_rel, dim=-1), torch.ones_like(q_rel[:, 0]))), "Relative quaternion must be normalized"
    
    scalar_part = q_rel[:, 0]
    vector_part = q_rel[:, 1:]
    
    angle = torch.acos(torch.clamp(scalar_part, min=-1.0, max=1.0)).unsqueeze(-1)
    vector_norm = torch.norm(vector_part, dim=-1).unsqueeze(-1)
    log_q = torch.where(vector_norm > 1e-6, angle * (vector_part / vector_norm), torch.zeros_like(vector_part))
    
    return log_q #TODO check if this is correct


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
 
    return reward

def quaternion_logarithm(q_rel: torch.Tensor) -> torch.Tensor:
    """Calculate quaternion logarithm using numerically stable approach.
    
    The quaternion logarithm maps unit quaternions to rotation vectors.
    For a unit quaternion q = [w, x, y, z], the logarithm is:
    log(q) = arccos(w) * [x, y, z] / ||[x, y, z]||
    
    Args:
        q_rel: Relative quaternion tensor of shape (..., 4) in [w, x, y, z] format
        
    Returns:
        Rotation vector tensor of shape (..., 3)
        
    Note:
        This implementation is numerically stable for small angles and handles
        the singularity when the quaternion represents no rotation.
    """
    assert q_rel.shape[-1] == 4, "Quaternion must have 4 components [w, x, y, z]"
    
    # Ensure quaternions are normalized (unit quaternions)
    q_rel = q_rel / torch.norm(q_rel, dim=-1, keepdim=True)
    
    scalar_part = q_rel[..., 0:1]  # w component
    vector_part = q_rel[..., 1:]   # [x, y, z] components
    
    # Clamp scalar part to avoid numerical issues with arccos
    scalar_clamped = torch.clamp(scalar_part, min=-1.0, max=1.0)
    
    # Calculate angle (half-angle of rotation)
    half_angle = torch.acos(torch.abs(scalar_clamped))
    
    # Calculate vector norm
    vector_norm = torch.norm(vector_part, dim=-1, keepdim=True)
    
    # For small rotations (vector_norm ≈ 0), use Taylor expansion: log(q) ≈ vector_part
    # For larger rotations, use the full formula: log(q) = angle * vector_part / vector_norm
    small_angle_threshold = 1e-6
    
    # Use the sign of scalar part to determine direction
    sign = torch.sign(scalar_part)
    
    log_q = torch.where(
        vector_norm > small_angle_threshold,
        sign * half_angle * (vector_part / vector_norm),
        sign * vector_part  # Taylor approximation for small angles
    )
    
    return log_q

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding during ground contact.

    This function penalizes sliding of the feet when they are in contact with the ground.
    The penalty is computed as the horizontal velocity magnitude of the feet weighted by
    the contact forces, encouraging the robot to minimize foot slippage.
    
    Args:
        env: Environment instance
        sensor_cfg: Configuration for the contact sensor
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Tensor of shape (num_envs,) containing sliding penalties for each environment
    """
    # Get contact sensor and forces
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    
    # Determine which feet are in contact (using force threshold)
    force_threshold = 1.0  # N
    in_contact = torch.norm(contact_forces, dim=-1) > force_threshold
    
    # Get robot asset and foot velocities
    asset = env.scene[asset_cfg.name]
    
    # Get horizontal (x-y) velocities of the feet
    foot_velocities = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]  # Only x-y components
    foot_speed = torch.norm(foot_velocities, dim=-1)
    
    # Calculate sliding penalty: horizontal speed weighted by contact
    sliding_penalty = torch.sum(foot_speed * in_contact.float(), dim=-1)
    
    return sliding_penalty

def is_alive(env: ManagerBasedRLEnv, phases: list[Phase] = [Phase.LANDING]) -> torch.Tensor:
    """Reward for being alive during the specified phases"""
    # Initialize as boolean tensor
    active_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        active_envs |= env.jump_phase == phase

    # Check if any environment is in the target phases
    if torch.any(active_envs):
        # Reward is 1 for active envs that are not terminated, 0 otherwise
        return (active_envs & (~env.termination_manager.terminated)).float()
    # Return zeros if no environments are in the target phases
    return torch.zeros(env.num_envs, device=env.device)

def liftoff_vertical_velocity(env: MarsJumperEnv, shape: str = "linear") -> torch.Tensor:
    """Reward vertical velocity at the exact moment of transition from TAKEOFF to FLIGHT phase."""
    # Identify environments that transitioned from TAKEOFF to FLIGHT in this step
    transitioned_to_flight = (env.prev_jump_phase == Phase.TAKEOFF) & (env.jump_phase == Phase.FLIGHT)

    reward = torch.zeros(env.num_envs, device=env.device)

    # Only calculate and apply reward for environments that just transitioned
    if torch.any(transitioned_to_flight):
        # Extract upward velocity component for transitioning environments
        upward_vel = env.robot.data.root_lin_vel_w[transitioned_to_flight, 2]
        # Apply reward based on velocity (clamp negative values)
        liftoff_reward = torch.clamp(upward_vel, min=0.0)

        if shape == "square":
            liftoff_reward = torch.square(liftoff_reward)

        reward[transitioned_to_flight] = liftoff_reward

    return reward

def attitude_error_on_way_down(env: ManagerBasedRLEnv, scale: float | int = 1.0) -> torch.Tensor:
    """Rewards for no attitude rotation for robot in flight phase when going downwards. Uses inverse quadratic kernel."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    flight_envs = env.jump_phase == Phase.FLIGHT
    reward_envs = flight_envs & (get_center_of_mass_lin_vel(env)[:, 2] < 0.01)
    quat = env.robot.data.root_quat_w[reward_envs]
    w = quat[:, 0]
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    reward_tensor[reward_envs] = 1/(1 + scale * torch.abs(angle)**2)
    return reward_tensor

def attitude_at_landing(env: ManagerBasedRLEnv, scale: float | int = 1.0) -> torch.Tensor:
    """Rewards for no attitude rotation for robot at transition to landing. Uses inverse quadratic kernel."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_envs = (env.jump_phase == Phase.LANDING) & (env.prev_jump_phase == Phase.FLIGHT)
    quat = env.robot.data.root_quat_w[reward_envs]
    w = quat[:, 0]
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    reward_tensor[reward_envs] = 1/(1 + scale * torch.abs(angle)**2)
    return reward_tensor

def attitude_rotation_magnitude(env: ManagerBasedRLEnv, kernel: str = "inverse_linear", scale: float | int = 1.0, phases: list[Phase] = [Phase.FLIGHT]) -> torch.Tensor:
    """Penalize any rotation from upright orientation using rotation vector magnitude.
    
    The rotation vector magnitude represents the total rotation angle in radians.
    For an upright robot, this should be 0. Returns values in [0,1] where:
    - 0 rotation -> reward of 1.0  
    - Large rotation -> reward approaches 0.0
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_envs = torch.zeros(env.num_envs, device=env.device)
    for phase in phases:
        active_envs = torch.logical_or(active_envs, env.jump_phase == phase)
    quat = env.robot.data.root_quat_w[active_envs]
    
    # Convert quaternion to rotation vector (axis-angle)
    # For quaternion q = [w,x,y,z], rotation vector = 2*arccos(w)*[x,y,z]/sqrt(1-w^2)
    w = quat[:, 0]
    xyz = quat[:, 1:]
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    angle = torch.abs(angle)
    
    # Return exponential of negative rotation magnitude to bound between [0,1]
    if kernel == "inverse_linear":
        reward_tensor[active_envs] = 1 / (1 + scale * angle)
    elif kernel == "inverse_quadratic":
        reward_tensor[active_envs] = 1 / (1 + scale * angle**2)
    elif kernel == "exponential":
        reward_tensor[active_envs] = torch.exp(-scale * angle)
    elif kernel == "square":
        reward_tensor[active_envs] = torch.square(angle)
    elif kernel == "linear":
        reward_tensor[active_envs] = torch.abs(angle)
    else:
        raise ValueError(f"Invalid kernel: {kernel}")
    
    return reward_tensor

def _get_dynamic_takeoff_vector(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Helper function to get the dynamic takeoff vector based on current COM position.
    
    Args:
        env: Environment instance with target_height and target_length attributes
        env_ids: Environment indices to calculate for
        
    Returns:
        Tensor of shape (len(env_ids), 3) containing [x, y, z] velocity components
    """
    from terms.utils import get_dynamic_takeoff_vector
    return get_dynamic_takeoff_vector(env, env_ids)

def _get_command_as_pitch_magnitude(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Helper function to convert target_height and target_length to pitch/magnitude format.
    
    NOTE: This function is kept for backward compatibility but should be replaced with 
    _get_dynamic_takeoff_vector for position-aware calculations.
    
    Args:
        env: Environment instance with target_height and target_length attributes
        env_ids: Environment indices to convert
        
    Returns:
        Tensor of shape (len(env_ids), 2) containing [pitch, magnitude]
    """
    height = env.target_height[env_ids]
    length = env.target_length[env_ids]
    pitch, magnitude = convert_height_length_to_pitch_magnitude(height, length, gravity=9.81)
    return torch.stack([pitch, magnitude], dim=-1)

def relative_cmd_error_huber(env: ManagerBasedRLEnv, delta: float, e_max: float) -> torch.Tensor:
    """Shifted huber kernel for reward. Applies a quadratic penalty for small errors and a linear penalty for large errors
    
    Args:
        delta: Error threshold at which huber loss goes from quadratic to linear (in actual relative error units)
        e_max: Maximum error after which reward becomes 0 (in actual relative error units)
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    if not torch.any(takeoff_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    takeoff_env_ids = takeoff_envs.nonzero(as_tuple=False).squeeze(-1)
    takeoff_vector = _get_dynamic_takeoff_vector(env, takeoff_env_ids)
    robot_vel_vec = get_center_of_mass_lin_vel(env)
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(robot_vel_vec - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    reward_tensor[takeoff_envs] = shifted_huber_kernel(e=relative_error, delta=delta, e_max=e_max)[takeoff_envs]

    # Check vertical velocity (assuming index 2 is vertical)
    vertical_velocity = robot_vel_vec[:, 2]
    downward_mask = vertical_velocity < 0

    reward_tensor[downward_mask] = 0    
    return reward_tensor

def attitude_penalty_takeoff_threshold(
    env: ManagerBasedRLEnv, 
    threshold_deg: float | int = 10.0
) -> torch.Tensor:
    """Penalize attitude rotation during takeoff only when it exceeds a threshold.
    
    This function helps prevent gaming where the robot twists excessively during takeoff
    to maintain ground contact while still being rewarded. Only applies penalty when
    rotation exceeds the specified threshold.
    
    Args:
        env: Environment instance
        threshold_deg: Threshold in degrees above which penalty is applied
        
    Returns:
        Tensor of shape (num_envs,) containing excess rotation in radians (use negative weight in config)
    """
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    # Only active during takeoff phase
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    if not torch.any(takeoff_envs):
        return reward_tensor
    
    # Get quaternion and calculate rotation magnitude
    quat = env.robot.data.root_quat_w[takeoff_envs]
    w = quat[:, 0]
    
    # Calculate rotation angle magnitude
    angle = 2 * torch.acos(torch.clamp(torch.abs(w), min=0.0, max=1.0))
    
    # Only apply penalty for rotations above threshold
    excess_rotation = torch.clamp(angle - threshold_deg, min=0.0)

    reward_tensor[takeoff_envs] = excess_rotation
    return reward_tensor

def feet_near_ground_reward(env: ManagerBasedRLEnv, 
                           height_threshold: float = 0.02,
                           ground_height: float = 0.0,
                           phases: list[Phase] = [Phase.LANDING]) -> torch.Tensor:
    """Reward for each foot that is within a height threshold of the ground.
    
    This encourages the robot to stand properly after landing by rewarding
    feet that are close to the ground surface. Returns average reward across all feet.
    
    Args:
        env: Environment instance
        height_threshold: Maximum height above ground to receive reward (in meters)
        ground_height: Height of the ground plane (in meters)
        phases: List of phases during which this reward is active
        
    Returns:
        Tensor of shape (num_envs,) with reward values (0 to 1, representing fraction of feet near ground)
    """
    # Check if any environment is in the target phases
    active_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        active_envs |= (env.jump_phase == phase)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    if not torch.any(active_envs):
        return reward_tensor
    
    # Get foot body indices and positions
    feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
    feet_positions = env.robot.data.body_pos_w[:, feet_idx, :]  # Shape: (num_envs, num_feet, 3)
    
    # Get feet heights above ground
    feet_heights = feet_positions[:, :, 2] - ground_height  # Shape: (num_envs, num_feet)
    
    # Check which feet are within threshold (and above ground)
    feet_near_ground = (feet_heights >= 0) & (feet_heights <= height_threshold)  # Shape: (num_envs, num_feet)
    
    # Calculate reward: average over all feet (gives value between 0 and 1)
    feet_reward = torch.mean(feet_near_ground.float(), dim=-1)  # Shape: (num_envs,)
    
    # Apply only to active environments
    reward_tensor[active_envs] = feet_reward[active_envs]
    
    return reward_tensor

def feet_height_penalty(env: ManagerBasedRLEnv,
                       ground_height: float = 0.0,
                       phases: list[Phase] = [Phase.LANDING],
                       kernel: Kernel = Kernel.LINEAR) -> torch.Tensor:
    """Penalize feet that are far from the ground based on their height.
    
    This encourages the robot to keep its feet close to the ground surface
    by penalizing the vertical distance from ground. Use negative weight in config.
    
    Args:
        env: Environment instance
        ground_height: Height of the ground plane (in meters)
        phases: List of phases during which this penalty is active
        kernel: Type of penalty kernel (LINEAR or SQUARE)
        
    Returns:
        Tensor of shape (num_envs,) with penalty values (use negative weight in config)
    """
    # Check if any environment is in the target phases
    active_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase in phases:
        active_envs |= (env.jump_phase == phase)
    
    penalty_tensor = torch.zeros(env.num_envs, device=env.device)
    
    if not torch.any(active_envs):
        return penalty_tensor
    
    # Get foot body indices and positions
    feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
    feet_positions = env.robot.data.body_pos_w[:, feet_idx, :]  # Shape: (num_envs, num_feet, 3)
    
    # Get feet heights above ground
    feet_heights = feet_positions[:, :, 2] - ground_height  # Shape: (num_envs, num_feet)
    
    # Only penalize positive heights (feet above ground)
    feet_heights_above_ground = torch.clamp(feet_heights, min=0.0)  # Shape: (num_envs, num_feet)
    
    # Apply kernel
    if kernel == Kernel.LINEAR:
        feet_penalties = feet_heights_above_ground  # Linear penalty
    elif kernel == Kernel.SQUARE:
        feet_penalties = torch.square(feet_heights_above_ground)  # Quadratic penalty
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")
    
    # Sum penalties across all feet
    total_penalty = torch.sum(feet_penalties, dim=-1)  # Shape: (num_envs,)
    
    # Apply only to active environments
    penalty_tensor[active_envs] = total_penalty[active_envs]
    
    return penalty_tensor

