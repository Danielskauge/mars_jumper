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
        net_contact_forces = forces_active_envs[:, env.bodies_except_feet_idx]
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
    robot_vel_vec = env.center_of_mass_lin_vel[env.takeoff_mask] 

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

def liftoff_relative_cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float) -> torch.Tensor:
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
    robot_vel_vec = env.center_of_mass_lin_vel[transitioned_to_flight]
    
    relative_error = torch.norm(robot_vel_vec - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    
    rewards = _apply_kernel(relative_error, kernel, scale)

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
    
def feet_ground_impact_force(env: ManagerBasedRLEnv, 
                            phases: list[Phase] = None,
                            kernel: Kernel = Kernel.EXPONENTIAL,
                            scale: float = 1.0,
                            delta: float = 1.0,
                            e_max: float = 1.0) -> torch.Tensor:
    """Reward the robot for landing with low impact force on the feet."""
    if phases is None:
        phases = [Phase.LANDING]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if torch.any(active_mask):
        contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_sensor").name] 
        feet_idx, _ = env.robot.find_bodies(".*FOOT.*")
        
        all_forces = contact_sensor.data.net_forces_w[:, feet_idx]
        active_forces = all_forces[active_mask]
        
        foot_force_norms = torch.norm(active_forces, dim=-1)
        values = torch.sum(torch.square(foot_force_norms), dim=-1)
        computed_values = torch.exp(-values)  # Using the original exponential formula
        reward_tensor[active_mask] = computed_values
    
    return reward_tensor
    
def takeoff_angle_error(env: ManagerBasedRLEnv, kernel: Kernel = Kernel.EXPONENTIAL, scale: float | int = 1.0) -> torch.Tensor:
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
    com_lin_vel = env.center_of_mass_lin_vel[env.takeoff_mask]    # [M,3]

    takeoff_vector_norm = takeoff_vector.norm(dim=1)     # [M]
    com_lin_vel_norm = com_lin_vel.norm(dim=1)     # [M]

    valid = com_lin_vel_norm >= 0.4 * takeoff_vector_norm  # [M]

    # Normalize vectors (avoid division by zero)
    takeoff_vector_unit = takeoff_vector / takeoff_vector_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]
    com_lin_vel_unit = com_lin_vel / com_lin_vel_norm.clamp(min=1e-6).unsqueeze(1)  # [M,3]

    dot = (takeoff_vector_unit * com_lin_vel_unit).sum(dim=1).clamp(-1.0, 1.0)      # [M]
    angle_err = torch.acos(dot)                                # [M]

    scaled_err = _apply_kernel(angle_err, kernel, scale)

    reward[env.takeoff_mask & valid] = scaled_err[valid]
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

def attitude_error_on_way_down(env: ManagerBasedRLEnv, 
                               phases: list[Phase] = None,
                               kernel: Kernel = Kernel.INVERSE_LINEAR,
                               scale: float = 1.0,
                               delta: float = 1.0,
                               e_max: float = 1.0) -> torch.Tensor:
    """Rewards for no attitude rotation for robot in flight phase when going downwards."""
    if phases is None:
        phases = [Phase.FLIGHT]
    
    active_mask = _get_active_envs_mask(env, phases)
    reward_tensor = _init_reward_tensor(env)
    
    if not torch.any(active_mask):
        return reward_tensor
    
    rewards_for_flight_phase = torch.zeros(torch.sum(active_mask), device=env.device)

    com_vel_flight_envs = env.center_of_mass_lin_vel[active_mask]
    actually_reward_mask_within_flight = com_vel_flight_envs[:, 2] < 0.01

    if torch.any(actually_reward_mask_within_flight):
        quat = env.robot.data.root_quat_w[active_mask][actually_reward_mask_within_flight]
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        angle = torch.abs(angle)

        # Apply kernel
        kernel_values = _apply_kernel(angle, kernel, scale, delta, e_max)
        rewards_for_flight_phase[actually_reward_mask_within_flight] = kernel_values
    
    reward_tensor[active_mask] = rewards_for_flight_phase
    return reward_tensor

def attitude_at_transition_to_landing(env: ManagerBasedRLEnv, scale: float | int = 1.0) -> torch.Tensor:
    """Rewards for no attitude rotation for robot at transition to landing. Uses inverse quadratic kernel."""
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_envs = (env.jump_phase == Phase.LANDING) & (env.prev_jump_phase == Phase.FLIGHT)
    reward_envs = reward_envs & ~env.termination_manager.get_term(name="bad_takeoff_at_landing") & ~env.termination_manager.get_term(name="landing")

    quat = env.robot.data.root_quat_w[reward_envs]
    w = quat[:, 0]
    angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
    reward_tensor[reward_envs] = 1/(1 + scale * torch.abs(angle)**2)
    return reward_tensor

def attitude_error(env: ManagerBasedRLEnv, 
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
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(env.center_of_mass_lin_vel - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    reward_tensor[takeoff_envs] = shifted_huber_kernel(e=relative_error, delta=delta, e_max=e_max)[takeoff_envs]

    vertical_velocity = env.center_of_mass_lin_vel[:, 2]
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
            ascending_mask = env.center_of_mass_lin_vel[:, 2] > 0
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
            ascending_mask = env.center_of_mass_lin_vel[:, 2] > 0
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

