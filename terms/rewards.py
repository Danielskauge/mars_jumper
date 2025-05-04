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

from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from terms.phase import Phase
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.assets import Articulation
from terms.utils import convert_command_to_euclidean_vector, get_center_of_mass_lin_vel


class Kernel(IntEnum):
    INVERSE_LINEAR = 0
    INVERSE_SQUARE = 1
    EXPONENTIAL = 2
    LINEAR = 3
    SQUARE = 4

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from envs.takeoff_env import MarsJumperEnv
    
def contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, kernel: Literal[Kernel.LINEAR, Kernel.SQUARE]) -> torch.Tensor:
    """Penalize the contact forces of the robot except for the feet, using L2 squared kernel."""
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] 
    forces_magnitude = torch.norm(net_contact_forces, dim=-1) #shape [num_envs, num_bodies]
    sum_forces_magnitude = torch.sum(forces_magnitude, dim=-1) #shape [num_envs]
    if kernel == Kernel.LINEAR:
        return sum_forces_magnitude
    elif kernel == Kernel.SQUARE:
        return torch.square(sum_forces_magnitude)
    
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
    # extract the used quantities (to enable type-ﬁﬁﬁhinting†)
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
    """Computes reward based on distance between robot's velocity vector and commanded takeoff vector.
    
    Only active during takeoff phase. Returns values in [0,1] where 1 means perfect tracking.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * error)
            - "inverse_linear": 1/(1 + scale * error)
            
    Returns:
        Reward tensor of shape (num_envs,)
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    cmd = env.command[takeoff_envs]
    cmd_vec = convert_command_to_euclidean_vector(cmd)
    robot_vel_vec = get_center_of_mass_lin_vel(env)[takeoff_envs]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(robot_vel_vec - cmd_vec, dim=-1) / torch.norm(cmd_vec, dim=-1)
    takeoff_rewards = torch.zeros_like(relative_error)
    if kernel == Kernel.EXPONENTIAL:
        takeoff_rewards = torch.exp(-scale*relative_error)
    elif kernel == Kernel.INVERSE_LINEAR:
        takeoff_rewards = 1/(1 + scale*relative_error)
    elif kernel == Kernel.INVERSE_SQUARE:
        takeoff_rewards = 1/(1 + scale*relative_error**2)

    # Check vertical velocity (assuming index 2 is vertical)
    vertical_velocity = robot_vel_vec[:, 2]
    downward_mask = vertical_velocity < 0

    # Zero out rewards for takeoff envs moving downwards
    takeoff_rewards[downward_mask] = 0

    # Assign the computed rewards back to the correct indices
    reward_tensor[takeoff_envs] = takeoff_rewards
    
    return reward_tensor

def relative_cmd_error(env: ManagerBasedRLEnv, kernel: Kernel, scale: float) -> torch.Tensor:
    """Computes reward based on relative error between robot's velocity and commanded takeoff vector.
    
    Similar to cmd_error() but normalizes error by command magnitude to be scale-invariant.
    Only active during takeoff phase. Returns values in [0,1] where 1 means perfect tracking.
    
    Args:
        env: Environment instance
        scale: Scaling factor for error. Higher values create sharper reward gradients.
        kernel: Error mapping function. Options:
            - "exponential": exp(-scale * relative_error) 
            - "inverse_linear": 1/(1 + scale * relative_error)
            
    Returns:
        Reward tensor of shape (num_envs,)
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    if not torch.any(takeoff_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    cmd = env.command[takeoff_envs]
    cmd_vec = convert_command_to_euclidean_vector(cmd)
    robot_vel_vec = get_center_of_mass_lin_vel(env)[takeoff_envs]
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    relative_error = torch.norm(robot_vel_vec - cmd_vec, dim=-1) / torch.norm(cmd_vec, dim=-1)
    takeoff_rewards = torch.zeros_like(relative_error)
    if kernel == Kernel.EXPONENTIAL:
        takeoff_rewards = torch.exp(-scale*relative_error)
    elif kernel == Kernel.INVERSE_LINEAR:
        takeoff_rewards = 1/(1 + scale*relative_error)
    elif kernel == Kernel.INVERSE_SQUARE:
        takeoff_rewards = 1/(1 + scale*relative_error**2)

    # Check vertical velocity (assuming index 2 is vertical)
    vertical_velocity = robot_vel_vec[:, 2]
    downward_mask = vertical_velocity < 0

    # Zero out rewards for takeoff envs moving downwards
    takeoff_rewards[downward_mask] = 0

    # Assign the computed rewards back to the correct indices
    reward_tensor[takeoff_envs] = takeoff_rewards
    
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
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
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

#TODO: needs to be fixed
# def equal_force_distribution(env: ManagerBasedRLEnv) -> torch.Tensor:
#     """Reward equal force distribution across the feet during takeoff and landing phases.
    
#     This function calculates how evenly the contact forces are distributed across all feet
#     that are in contact with the ground. The reward is higher when forces are more evenly
#     distributed, which encourages balanced takeoffs and landings.
    
#     The reward is calculated as:
#     r = 1 - (std_dev / mean)
    
#     where std_dev is the standard deviation of forces across feet in contact,
#     and mean is the average force. This coefficient of variation approach rewards
#     when forces are more uniform (lower std_dev relative to mean).
#     """
#     takeoff_envs = env.jump_phase == Phase.TAKEOFF
#     landing_envs = env.jump_phase == Phase.LANDING
#     active_envs = takeoff_envs | landing_envs
    
#     if not torch.any(active_envs):
#         return torch.zeros(env.num_envs, device=env.device)
    
#     contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
#     robot: Articulation = env.scene[SceneEntityCfg("robot").name]
#     feet_idx, _ = robot.find_bodies(".*FOOT.*")
    
#     # Get forces for all feet
#     feet_forces = contact_sensor.data.net_forces_w[:, feet_idx]
#     force_magnitudes = torch.norm(feet_forces, dim=-1)
        
#     reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
#     std_dev = torch.std(force_magnitudes, dim=-1)
#     mean_force = torch.mean(force_magnitudes, dim=-1)
    
#     if mean_force > 0.001:
#         reward_tensor[active_envs] = -std_dev / mean_force 
#     else:
#         reward_tensor[active_envs] = 0
    
#     return reward_tensor
    
    
def landing_com_accel(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Reward the robot for landing with low base acceleration.
    
    The reward is calculated as:
    r = exp(-||a||²)
    
    where ||a|| is the L2 norm of the center of mass linear acceleration vector.
    This exponential function maps:
    - Zero acceleration → reward of 1.0
    - High acceleration → reward approaches 0.0
    
    The squared norm in the exponent makes the reward decay more rapidly as acceleration increases.
    
    #TODO: there will be accelration regardless, so maybe that will make this a bad reward signal?
    """
    landing_envs = env.jump_phase == Phase.LANDING
    if not torch.any(landing_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    com_accel_norm = env.robot.data.body_lin_acc_w[landing_envs].norm(dim=-1)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[landing_envs] = torch.exp(-torch.square(com_accel_norm)) 
    
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
    feet_idx, _ = env.robot.find_joints(".*FOOT.*")
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
        
def get_cartesian_cmd_vel_vec(cmd_vel_vec: torch.Tensor) -> torch.Tensor:
    """ _Convert the takeoff velocity vector command to cartesian coordinates 
    pitch is positive clockwise form positive z axis
    """
    pitch, magnitude = cmd_vel_vec.unbind(-1)
    x_dot = magnitude * torch.sin(pitch)
    z_dot = magnitude * torch.cos(pitch)
    y_dot = torch.zeros_like(x_dot)
    return torch.stack([x_dot, y_dot, z_dot], dim=-1)
    
def takeoff_vel_vec_angle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Difference between the commanded and actual takeoff velocity vector angle measured by cosine similarity
    
    Args:
        env: The environment instance.
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    cmd_vel_vec = env.command[takeoff_envs]
    cmd_vel_vec_cartesian = get_cartesian_cmd_vel_vec(cmd_vel_vec)
    
    actual_vel_vec_cartesian = env.robot.data.body_com_state_w[takeoff_envs, 6:9]
    
    similarity = torch.nn.functional.cosine_similarity(cmd_vel_vec_cartesian, actual_vel_vec_cartesian)
    return (1 + similarity) / 2 #bound similarity to 0-1

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

def takeoff_vel_vec_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Difference between the magnitude of the commanded and actual takeoff velocity vector magnitude, 
    Uses exponential reward shaping to bound error to 0-1
    """
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    body_com_lin_vel_vec = env.robot.data.body_com_state_w[takeoff_envs, 6:9]
    body_com_lin_vel_magnitude = torch.norm(body_com_lin_vel_vec, dim=-1)
    cmd_magnitude = env.command[takeoff_envs, 1]
    error = torch.abs(cmd_magnitude - body_com_lin_vel_magnitude)
    exp_diff = torch.exp(error) #reward shaping that bounds error to 0-1
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[takeoff_envs] = exp_diff
    
    return reward_tensor

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

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    #TODO: HAS TO BE FIXED
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

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
def ang_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xyz-axis base angular velocity using L1 kernel."""
    # extract the used quantities (to enable type-hinting)
    return torch.sum(torch.abs(env.robot.data.root_ang_vel_b), dim=1)

def change_joint_direction_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize change in joint movement direction. Use negative weight. Returns values between 0 and 1"""
    
    current_joint_vel = env.robot.data.joint_vel
    changed_direction = torch.sign(current_joint_vel) != torch.sign(env.prev_joint_vel)
    penalty = torch.sum(changed_direction, dim=1) / env.robot.num_joints
    
    return penalty

