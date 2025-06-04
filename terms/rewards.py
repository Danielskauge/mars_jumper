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
import logging
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from terms.utils import Phase
from terms.utils import convert_height_length_to_pitch_magnitude
from terms.utils import convert_pitch_magnitude_to_vector

logger = logging.getLogger(__name__)

class Kernel(IntEnum):
    INVERSE_LINEAR = 0
    INVERSE_SQUARE = 1
    EXPONENTIAL = 2
    NONE = 3
    SQUARE = 4
    SHIFTED_SQUARE = 5
    HUBER = 6

from isaaclab.envs import ManagerBasedRLEnv
    
# =================================================================================
# Helper Functions
# =================================================================================

def update_env_data(env: ManagerBasedRLEnv) -> torch.Tensor:
    env.com_vel = env.get_com_vel()
    env.com_pos = env.get_com_pos()
    env.com_acc = env.get_com_acc()
    env.update_dynamic_takeoff_vector()
    return torch.zeros(env.num_envs, device=env.device)
    
def update_jump_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    env.update_jump_phase()
    return torch.zeros(env.num_envs, device=env.device)

def _get_active_envs_mask(env: ManagerBasedRLEnv, phases: list[Phase]) -> torch.Tensor:
    """Get boolean mask for environments active in specified phases."""
    active_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for phase_val in phases: # Renamed to avoid conflict with terms.phase.Phase
        active_mask |= (env.jump_phase == phase_val)
    return active_mask

def _init_reward_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)

def _apply_kernel(values: torch.Tensor, kernel: Kernel, scale: float = 1.0, delta: float = 1.0, e_max: float = 1.0) -> torch.Tensor:
    """Apply specified kernel to values.
    
    """
    if kernel == Kernel.EXPONENTIAL:
        return torch.exp(-scale * values)
    elif kernel == Kernel.INVERSE_LINEAR:
        return 1 / (1 + scale * values)
    elif kernel == Kernel.INVERSE_SQUARE:
        return 1 / (1 + scale * values**2)
    elif kernel == Kernel.NONE:
        return  values
    elif kernel == Kernel.SQUARE:
        return scale * values**2
    elif kernel == Kernel.HUBER:
        return shifted_huber_kernel(values, delta, e_max)
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

# =================================================================================
# Reward Functions
# =================================================================================
    
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
        - e=e_max gives reward=0, negative beyond that. This controls steepness of the reward curve.
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

def landing_com_vel(env: ManagerBasedRLEnv, 
                    kernel: Kernel = Kernel.HUBER,
                    scale: float = 1.0,
                    delta: float = 1.0,
                    e_max: float = 1.0) -> torch.Tensor:
    """Penalize the velocity of the center of mass during landing phase."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    if not torch.any(env.landing_mask):
        return reward_tensor
    
    landing_env_ids = env.landing_mask.nonzero(as_tuple=False).squeeze(-1)
    landing_com_vel = env.com_vel[landing_env_ids]
    norm = torch.norm(landing_com_vel, dim=-1)
    values = _apply_kernel(norm, kernel, scale, delta, e_max)
    reward_tensor[landing_env_ids] = values
    return reward_tensor

def landing_base_height(env: ManagerBasedRLEnv,
                           target_height: float = 0.0,
                           kernel: Kernel = Kernel.INVERSE_LINEAR,
                           scale: float = 1.0,
                           delta: float = 1.0,
                           e_max: float = 1.0) -> torch.Tensor:
    """Penalize asset height from its min height.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    reward_tensor = _init_reward_tensor(env)
    
    active_mask = env.landing_mask
    
    active_mask = active_mask & ~env.termination_manager.terminated
    
    if torch.any(active_mask):
        height_error = abs(env.robot.data.root_pos_w[active_mask, 2] - target_height)
        reward_tensor[active_mask] = _apply_kernel(values=height_error, kernel=kernel, scale=scale, delta=delta, e_max=e_max)
    
    return reward_tensor

def contact_forces(env: ManagerBasedRLEnv, 
                   phases: list[Phase] = None,
                   kernel: Kernel = Kernel.NONE,
                   scale: float = 1.0,
                   delta: float = 1.0,
                   e_max: float = 1.0) -> torch.Tensor:
    """Penalize the contact forces of the robot except for the feet."""
    if phases is None:
        phases = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING, Phase.CROUCH]
    
    active_mask = _get_active_envs_mask(env, phases)
    active_mask = active_mask & ~env.termination_manager.terminated

    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        # Corrected indexing: apply active_mask first, then select bodies
        forces_active_envs = env.contact_sensor.data.net_forces_w[active_mask]
        idx = torch.cat((env.base_body_idx, env.hips_body_idx, env.thighs_body_idx, env.shanks_body_idx))
        net_contact_forces = forces_active_envs[:, idx]
        forces_magnitude = torch.norm(net_contact_forces, dim=-1)
        values = torch.sum(forces_magnitude, dim=-1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor
    
def action_rate(env: ManagerBasedRLEnv, 
                phases: list[Phase] = None,
                kernel: Kernel = Kernel.SQUARE,
                scale: float = 1.0,
                delta: float = 1.0,
                e_max: float = 1.0) -> torch.Tensor:
    """Penalize the rate of change of the actions."""
    if phases is None:
        phases = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING, Phase.CROUCH]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        action_diff = env.action_manager.action[active_mask] - env.action_manager.prev_action[active_mask]
        values = torch.sum(torch.square(action_diff), dim=1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor
    
def joint_vel_l1(env: ManagerBasedRLEnv, 
                 asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                 phases: list[Phase] = None,
                 kernel: Kernel = Kernel.NONE,
                 scale: float = 1.0,
                 delta: float = 1.0,
                 e_max: float = 1.0) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    if phases is None:
        phases = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING, Phase.CROUCH]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        asset: Articulation = env.scene[asset_cfg.name]
        # Ensure joint_ids from asset_cfg are applied correctly if asset_cfg has them.
        # Defaulting to all joints of the asset if asset_cfg.joint_ids is None or not present.
        joint_ids_to_penalize = asset_cfg.joint_ids if hasattr(asset_cfg, 'joint_ids') and asset_cfg.joint_ids is not None else slice(None)
        values = torch.sum(torch.abs(asset.data.joint_vel[active_mask, joint_ids_to_penalize]), dim=1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def joint_vel_l1_delayed_landing(env: ManagerBasedRLEnv, 
                                asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                                delay_seconds: float = 0.1,
                                kernel: Kernel = Kernel.NONE,
                                scale: float = 1.0,
                                delta: float = 1.0,
                                e_max: float = 1.0) -> torch.Tensor:
    """Penalize joint velocities using L1 kernel, but only after a delay when entering landing phase.
    
    This reward only activates after the robot has been in landing phase for a specified duration.
    
    Args:
        env: The RL environment.
        asset_cfg: Scene entity configuration for the robot.
        delay_seconds: Time in seconds to wait after entering landing before activating the penalty.
        kernel: Kernel function to apply to the joint velocity values.
        scale: Scaling factor for the kernel.
        delta: Delta parameter for Huber kernel.
        e_max: Maximum error for Huber kernel.
    
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # Initialize landing time tracker if not present
    if not hasattr(env, '_landing_time_tracker'):
        env._landing_time_tracker = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    
    # Get time step duration
    dt = env.step_dt  # This should be the real-time control dt (e.g., 1/120 seconds)
    
    # Reset time tracker when transitioning to landing phase
    if hasattr(env, 'flight_to_landing_mask'):
        env._landing_time_tracker[env.flight_to_landing_mask] = 0.0
    
    # Increment time tracker for environments in landing phase
    landing_mask = env.jump_phase == Phase.LANDING
    env._landing_time_tracker[landing_mask] += dt
    
    # Only apply penalty to environments that have been in landing for more than delay_seconds
    active_mask = landing_mask & (env._landing_time_tracker >= delay_seconds)
    
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids_to_penalize = asset_cfg.joint_ids if hasattr(asset_cfg, 'joint_ids') and asset_cfg.joint_ids is not None else slice(None)
        values = torch.sum(torch.abs(asset.data.joint_vel[active_mask, joint_ids_to_penalize]), dim=1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float = 7.0, delta: float = 1.0, e_max: float = 1.0, error_type: str = "relative") -> torch.Tensor:
    """Unified command tracking error function.
    
    Computes reward based on error between robot's velocity and commanded takeoff vector.
    Only active during takeoff phase. Returns values in [0,1] where 1 means perfect tracking.
    
    Args:
        env: Environment instance
        kernel: Error mapping function (EXPONENTIAL, INVERSE_LINEAR, INVERSE_SQUARE, SHIFTED_SQUARE, NONE)
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        error_type: "relative" (normalized by command magnitude) or "absolute" (raw error)
        
    Returns:
        Reward tensor of shape (num_envs,)
    """

    reward_tensor = _init_reward_tensor(env)
    
    if not torch.any(env.takeoff_mask):
        return reward_tensor

    takeoff_vector = env.dynamic_takeoff_vector[env.takeoff_mask] 
    robot_vel_vec = env.com_vel[env.takeoff_mask] 

    error_vec = robot_vel_vec - takeoff_vector
    if error_type == "relative":
        error = torch.norm(error_vec, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    elif error_type == "absolute":
        error = torch.norm(error_vec, dim=-1)
    else:
        raise ValueError(f"error_type must be 'relative' or 'absolute', got {error_type}")
    
    rewards = _apply_kernel(values=error, kernel=kernel, scale=scale, delta=delta, e_max=e_max)
    
    downward_mask = robot_vel_vec[:, 2] < 0
    rewards[downward_mask] = 0
    
    reward_tensor[env.takeoff_mask] = rewards
    return reward_tensor

def liftoff_relative_cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float = 0, delta: float = 0, e_max: float = 0) -> torch.Tensor:
    """Computes reward based on relative error between robot's velocity and commanded takeoff vector, ONLY at the moment of liftoff.
    
    Similar to relative_cmd_error() but only active at the transition from TAKEOFF to FLIGHT.
    Returns values in [0,1] where 1 means perfect tracking, for appropriate kernels.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * relative_error) 
            - "inverse_linear": 1/(1 + scale * relative_error)
            - "huber": shifted_huber_kernel(relative_error, delta, e_max)
    Returns:
        Reward tensor of shape (num_envs,)
    """
    # This function has specific transition logic (TAKEOFF -> FLIGHT) and doesn't fit the generic decorator.
    transitioned_to_flight = (env.prev_jump_phase == Phase.TAKEOFF) & (env.jump_phase == Phase.FLIGHT)
    reward_tensor = _init_reward_tensor(env)

    if not torch.any(transitioned_to_flight):
        return reward_tensor
    
    transitioned_env_ids = transitioned_to_flight.nonzero(as_tuple=False).squeeze(-1)
    if transitioned_env_ids.numel() == 0:
        return reward_tensor
        
    takeoff_vector = env.dynamic_takeoff_vector[transitioned_env_ids]
    robot_vel_vec = env.com_vel[transitioned_to_flight]
    
    relative_error = torch.norm(robot_vel_vec - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    
    rewards = _apply_kernel(relative_error, kernel, scale, delta, e_max)

    reward_tensor[transitioned_to_flight] = rewards
    return reward_tensor

def feet_ground_contact(env: ManagerBasedRLEnv, 
                       phases: list[Phase] = None,
                       kernel: Kernel = Kernel.NONE,
                       scale: float = 1.0,
                       delta: float = 1.0,
                       e_max: float = 1.0) -> torch.Tensor:
    """Make the robot maintain ground contact."""
    if phases is None:
        phases = [Phase.CROUCH, Phase.LANDING]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_sensor").name]
        feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
        
        all_feet_forces = contact_sensor.data.net_forces_w[:, feet_idx] 
        active_feet_forces = all_feet_forces[active_mask]
        
        is_foot_in_contact = torch.norm(active_feet_forces, dim=-1) > contact_sensor.cfg.force_threshold
        values = torch.mean(is_foot_in_contact.float(), dim=-1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def landing_max_feet_force_above_threshold(env: ManagerBasedRLEnv, 
                                           kernel: Kernel = Kernel.NONE,
                                           scale: float = 1.0,
                                           delta: float = 1.0,
                                           e_max: float = 1.0,
                                           threshold: float = 100.0) -> torch.Tensor:
    """Reward the robot for landing with low impact force on the feet."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)

    if not torch.any(env.landing_mask):
        return reward_tensor
    
    landing_feet_forces = env.contact_sensor.data.net_forces_w[env.landing_mask][:, env.feet_body_idx]
    max_force = torch.max(torch.norm(landing_feet_forces, dim=-1), dim=-1).values
    max_force_above_threshold = torch.clamp(max_force - threshold, min=0.0)
    e_max = e_max - threshold
    values = _apply_kernel(max_force_above_threshold, kernel, scale, delta, e_max)
    reward_tensor[env.landing_mask] = values
    return reward_tensor
    
def takeoff_angle_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float | int = 1.0) -> torch.Tensor:
    """Rewards for small velocity vector angular misalignment during takeoff phase.

    Normalized by: exp(-scale * angle_error)
    Filtered by: magnitude must be at least 50% of commanded magnitude.
    Only active during takeoff phase.

    Returns:
        reward: 1.0 for perfect alignment, approaches 0 for large misalignment.
    """
    reward = torch.zeros(env.num_envs, device=env.device)
    if not torch.any(env.takeoff_mask):
        return reward

    takeoff_env_ids = env.takeoff_mask.nonzero(as_tuple=False).squeeze(-1)
    takeoff_vector = env.dynamic_takeoff_vector[takeoff_env_ids]         # [M,3]
    com_lin_vel = env.com_vel[env.takeoff_mask]    # [M,3]

    takeoff_vector_norm = takeoff_vector.norm(dim=1)     # [M]
    com_lin_vel_norm = com_lin_vel.norm(dim=1)     # [M]

    valid = com_lin_vel_norm >= 0.4 * takeoff_vector_norm  # [M]

    # Normalize vectors (avoid division by zero)
    takeoff_vector_unit = takeoff_vector / takeoff_vector_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]
    com_lin_vel_unit = com_lin_vel / com_lin_vel_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]

    dot = (takeoff_vector_unit * com_lin_vel_unit).sum(dim=1).clamp(-1.0, 1.0)      # [M]
    angle_err = torch.acos(dot)                                # [M]

    scaled_err = _apply_kernel(angle_err, kernel, scale)

    # Fix: Using indexing to ensure matching tensor sizes
    reward_indices = takeoff_env_ids[valid]
    reward[reward_indices] = scaled_err[valid]
    return reward

def liftoff_vertical_velocity(env: ManagerBasedRLEnv, shape: str = "linear") -> torch.Tensor:
    """Reward vertical velocity at the exact moment of transition from TAKEOFF to FLIGHT phase."""
    transitioned_to_flight = (env.prev_jump_phase == Phase.TAKEOFF) & (env.jump_phase == Phase.FLIGHT)

    reward = torch.zeros(env.num_envs, device=env.device)

    if torch.any(transitioned_to_flight):
        upward_vel = env.robot.data.root_lin_vel_w[transitioned_to_flight, 2]
        liftoff_reward = torch.clamp(upward_vel, min=0.0)

        if shape == "square":
            liftoff_reward = torch.square(liftoff_reward)

        reward[transitioned_to_flight] = liftoff_reward

    return reward

def attitude_descent(env: ManagerBasedRLEnv, 
                               kernel: Kernel = Kernel.INVERSE_LINEAR,
                               scale: float = 1.0,
                               delta: float = 1.0,
                               e_max: float = 1.0) -> torch.Tensor:
    """Rewards for no attitude rotation for robot in flight phase when going downwards."""

    reward_tensor = _init_reward_tensor(env)
    
    if not torch.any(env.flight_mask):
        return reward_tensor
    
    rewards_for_flight_phase = torch.zeros(torch.sum(env.flight_mask), device=env.device)

    com_vel_flight_envs = env.com_vel[env.flight_mask]
    actually_reward_mask_within_flight = com_vel_flight_envs[:, 2] < 0.01

    if torch.any(actually_reward_mask_within_flight):
        quat = env.robot.data.root_quat_w[env.flight_mask][actually_reward_mask_within_flight]
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        angle = torch.abs(angle)

        # Apply kernel
        kernel_values = _apply_kernel(angle, kernel, scale, delta, e_max)
        rewards_for_flight_phase[actually_reward_mask_within_flight] = kernel_values
    
    reward_tensor[env.flight_mask] = rewards_for_flight_phase
    return reward_tensor



def attitude(env: ManagerBasedRLEnv, 
                   phases: list[Phase] = None,
                   kernel: Kernel = Kernel.INVERSE_LINEAR,
                   scale: float = 1.0,
                   delta: float = 1.0,
                   e_max: float = 1.0) -> torch.Tensor:
    """Penalize any rotation from upright orientation using rotation vector magnitude."""
    if phases is None:
        phases = [Phase.FLIGHT]
    
    active_mask = _get_active_envs_mask(env, phases)
    active_mask = active_mask & ~env.termination_manager.terminated

    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        quat = env.robot.data.root_quat_w[active_mask]
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        values = torch.abs(angle)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def attitude_landing_trans(env: ManagerBasedRLEnv, 
                   kernel: Kernel = Kernel.INVERSE_LINEAR,
                   scale: float = 1.0,
                   delta: float = 1.0,
                   e_max: float = 1.0) -> torch.Tensor:
    """Penalize any rotation from upright orientation at the transition to landing phase."""
  
    active_mask = (env.jump_phase == Phase.LANDING) & (env.prev_jump_phase == Phase.FLIGHT)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        quat = env.robot.data.root_quat_w[active_mask]
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        values = torch.abs(angle)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

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
    takeoff_vector = env.dynamic_takeoff_vector[takeoff_env_ids]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(env.com_vel - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    reward_tensor[takeoff_envs] = shifted_huber_kernel(e=relative_error, delta=delta, e_max=e_max)[takeoff_envs]

    vertical_velocity = env.com_vel[:, 2]
    downward_mask = vertical_velocity < 0

    reward_tensor[downward_mask] = 0    
    return reward_tensor

def attitude_penalty_takeoff_threshold(
    env: ManagerBasedRLEnv, 
    phases: list[Phase] = None,
    kernel: Kernel = Kernel.NONE,
    scale: float = 1.0,
    delta: float = 1.0,
    e_max: float = 1.0,
    threshold_deg: float | int = 10.0
) -> torch.Tensor:
    """Penalize attitude rotation during takeoff only when it exceeds a threshold."""
    if phases is None:
        phases = [Phase.TAKEOFF]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        quat = env.robot.data.root_quat_w[active_mask]
        w = quat[:, 0]
        
        angle_rad = 2 * torch.acos(torch.clamp(torch.abs(w), min=0.0, max=1.0))
        threshold_rad = torch.deg2rad(torch.tensor(threshold_deg, device=env.device, dtype=angle_rad.dtype))
        
        values = torch.clamp(angle_rad - threshold_rad, min=0.0)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def feet_near_ground(env: ManagerBasedRLEnv, 
                           phases: list[Phase] = None,
                           kernel: Kernel = Kernel.NONE,
                           scale: float = 1.0,
                           delta: float = 1.0,
                           e_max: float = 1.0,
                           height_threshold: float = 0.02,
                           ) -> torch.Tensor:
    """Reward for each foot that is within a height threshold of the ground."""
    if phases is None:
        phases = [Phase.LANDING]
    
    active_mask = _get_active_envs_mask(env, phases)
    active_mask = active_mask & ~env.termination_manager.terminated
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
        feet_positions = env.robot.data.body_pos_w[active_mask, :, :][:, feet_idx, :] 
        
        feet_heights = feet_positions[:, :, 2]
        feet_near_ground = (feet_heights >= 0) & (feet_heights <= height_threshold)
        values = torch.mean(feet_near_ground.float(), dim=-1)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def feet_height(env: ManagerBasedRLEnv,
                       phases: list[Phase] = [Phase.LANDING],
                       kernel: Kernel = Kernel.NONE,
                       scale: float = 1.0,
                       delta: float = 1.0,
                       e_max: float = 1.0,
                       ) -> torch.Tensor:
    """Penalize feet that are far from the ground based on their height."""

    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    active_mask = active_mask & ~env.termination_manager.terminated

    if torch.any(active_mask):
        selected_envs_data = env.robot.data.body_pos_w[active_mask]
        feet_heights = selected_envs_data[:, env.feet_body_idx, 2]
        values = torch.mean(feet_heights)
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def yaw_penalty(env: ManagerBasedRLEnv, 
                phases: list[Phase] = None,
                kernel: Kernel = Kernel.HUBER,
                scale: float = 1.0,
                delta: float = 1.0,
                e_max: float = 1.0) -> torch.Tensor:
    """Penalize yaw rotation during takeoff and ascent part of flight phase."""
    if phases is None:
        phases = [Phase.TAKEOFF, Phase.FLIGHT]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if not torch.any(active_mask):
        return reward_tensor
    
    # For flight phase, only penalize during ascent (positive vertical velocity)
    if Phase.FLIGHT in phases:
        flight_mask = (env.flight_mask) & active_mask
        if torch.any(flight_mask):
            ascending_mask = env.com_vel[:, 2] > 0
            # Only keep flight environments that are ascending
            active_mask = active_mask & (~flight_mask | ascending_mask)
    
    if torch.any(active_mask):
        # Extract yaw angle from quaternion (w, x, y, z)
        quat = env.robot.data.root_quat_w[active_mask]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Calculate yaw angle: atan2(2(wz + xy), 1 - 2(y² + z²))
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        values = torch.abs(yaw)
        
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def roll_penalty(env: ManagerBasedRLEnv, 
                 phases: list[Phase] = None,
                 kernel: Kernel = Kernel.HUBER,
                 scale: float = 1.0,
                 delta: float = 1.0,
                 e_max: float = 1.0) -> torch.Tensor:
    """Penalize roll rotation during takeoff and ascent part of flight phase."""
    if phases is None:
        phases = [Phase.TAKEOFF, Phase.FLIGHT]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if not torch.any(active_mask):
        return reward_tensor
    
    # For flight phase, only penalize during ascent (positive vertical velocity)
    if Phase.FLIGHT in phases:
        flight_mask = (env.jump_phase == Phase.FLIGHT) & active_mask
        if torch.any(flight_mask):
            ascending_mask = env.com_vel[:, 2] > 0
            # Only keep flight environments that are ascending
            active_mask = active_mask & (~flight_mask | ascending_mask)
    
    if torch.any(active_mask):
        # Extract roll angle from quaternion (w, x, y, z)
        quat = env.robot.data.root_quat_w[active_mask]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Calculate roll angle: atan2(2(wy + xz), 1 - 2(y² + z²))
        roll = torch.atan2(2 * (w * y + x * z), 1 - 2 * (y**2 + z**2))
        values = torch.abs(roll)
        
        computed_values = _apply_kernel(values, kernel, scale, delta, e_max)
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor

def landing_walking(env: ManagerBasedRLEnv,
                             kernel: Kernel = Kernel.HUBER,
                             scale: float = 1.0,
                             delta: float = 0.05,
                             e_max: float = 0.20) -> torch.Tensor:
    """Reward staying near the target landing position during landing phase.
    
    Encourages the robot to land and stay near the target position (target_length in x, 0 in y).
    Uses distance from target position as the error metric.
    
    Args:
        env: The RL environment
        phases: List of phases when this reward is active (default: [Phase.LANDING])
        kernel: Kernel function to apply to the distance error
        scale: Scale parameter for kernel (not used for HUBER)
        delta: Huber loss threshold - distance where reward transitions from quadratic to linear
        e_max: Maximum distance where reward becomes zero
    
    Returns:
        Reward tensor of shape (num_envs,) where higher values indicate better position tracking
    """

    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(env.landing_mask):
        # Get current center of mass position (already in environment local frame)
        current_com_pos = env.com_pos[env.landing_mask]  # Shape: (active_envs, 3)
        
        # Target position: x should be target_length, y should be 0
        target_x = env.target_length[env.landing_mask]  # Shape: (active_envs,)
        target_y = torch.zeros_like(target_x)  # y target is 0
        
        # Calculate 2D distance from target position (ignore z for landing position tracking)
        x_error = current_com_pos[:, 0] - target_x
        y_error = current_com_pos[:, 1] - target_y
        distance_error = torch.sqrt(x_error**2 + y_error**2)  # Shape: (active_envs,)
        
        computed_values = _apply_kernel(distance_error, kernel, scale, delta, e_max)
        reward_tensor[env.landing_mask] = computed_values
    
    return reward_tensor

def landing_foot_impact_force_penalty(env: ManagerBasedRLEnv,
                             force_threshold: float = 50.0):
    """Penalize foot impact forces that exceed a safe threshold.
    
    Applies a hinged quadratic penalty when the maximum foot force exceeds the threshold:
    - cost = 0 if max_foot_force <= threshold  
    - cost = (max_foot_force - threshold)^2 if max_foot_force > threshold
    
    Args:
        env: The RL environment
        force_threshold: Safe force threshold (F_safe)
        phases: List of phases when this penalty is active (default: all phases)
    
    Returns:
        Cost tensor (use negative weight in config to make it a penalty)
    """
    

    if torch.any(env.landing_mask):
        reward_tensor = torch.zeros(env.num_envs, device=env.device)
        # Get foot contact forces
        feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
        contact_sensor: ContactSensor = env.scene["contact_sensor"]
        
        # Get forces for active environments and foot bodies only
        foot_forces = contact_sensor.data.net_forces_w[env.landing_mask][:, feet_idx]  # Shape: (active_envs, num_feet, 3)
        
        # Calculate force magnitudes for each foot
        foot_force_magnitudes = torch.norm(foot_forces, dim=-1)  # Shape: (active_envs, num_feet)
        
        # Get maximum force across all feet for each environment
        max_foot_forces = torch.max(foot_force_magnitudes, dim=-1)[0]  # Shape: (active_envs,)
        
        # Apply hinged penalty: 0 if below threshold, (force - threshold)^2 if above
        excess_force = torch.clamp(max_foot_forces - force_threshold, min=0.0)
        cost = excess_force ** 2
        
        reward_tensor[env.landing_mask] = cost
    
    return reward_tensor

