from __future__ import annotations

from collections.abc import Sequence
import math
import os
from typing import TYPE_CHECKING, List
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv



    
def sum_contact_forces(env: ManagerBasedEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Sum the contact forces of the robot, expect feet"""
    sensor_cfg = SceneEntityCfg(name="contact_sensor", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"])
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w[env_ids, sensor_cfg.body_ids] 
    sum_forces_magnitude = torch.sum(net_contact_forces, dim=-1) #shape [num_envs]
    return sum_forces_magnitude
    
def all_feet_ok_air_time(env: ManagerBasedEnv, air_time_threshold: float) -> torch.Tensor:
    air_time_above_threshold = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    for sensor_name in env.cfg.feet_ground_contact_name_map.values():
        sensor: ContactSensor = env.scene[sensor_name]
        air_time_above_threshold = torch.logical_and(air_time_above_threshold, (sensor.data.air_time > air_time_threshold))
    return air_time_above_threshold

def all_feet_off_the_ground(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if all feet are off the ground using the general contact sensor."""
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    contact_sensor: ContactSensor = env.scene["contact_sensor"]

    # Find the indices corresponding to the foot bodies within the contact sensor data
    foot_body_indices, _ = contact_sensor.find_bodies([".*FOOT"])
    if not foot_body_indices:
        # Handle case where no foot bodies are found in the sensor (shouldn't happen with prim_path=".*")
        print("[Warning] No foot bodies found in the general contact sensor.")
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

    # Get net forces for all bodies from the general sensor
    # Use history[0] to get the most recent force data
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    # Select forces only for the foot bodies
    foot_forces = net_forces[:, foot_body_indices, :] # Shape: [num_envs, num_feet, 3]

    # Calculate the magnitude of forces for each foot
    foot_force_magnitudes = torch.norm(foot_forces, dim=-1) # Shape: [num_envs, num_feet]

    # Check if *all* foot force magnitudes are below the threshold
    all_feet_off = torch.all(foot_force_magnitudes <= contact_sensor.cfg.force_threshold, dim=1) # Shape: [num_envs]

    return all_feet_off

def any_body_high_contact_force(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if any body is in contact with the ground using the general contact sensor."""
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    contact_sensor: ContactSensor = env.scene["contact_sensor"]
    
    # Get net forces for all bodies from the general sensor
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    
    # Calculate the magnitude of forces for each body
    body_force_magnitudes = torch.norm(net_forces, dim=-1) # Shape: [num_envs, num_bodies]
    
    # Check if *any* body force magnitude is above the threshold
    any_body_high_contact_force = torch.any(body_force_magnitudes > contact_sensor.cfg.force_threshold, dim=1) # Shape: [num_envs]
    
    return any_body_high_contact_force

def any_feet_on_the_ground(env: ManagerBasedEnv) -> torch.Tensor:
    """Check if any feet are on the ground using the general contact sensor."""
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    contact_sensor: ContactSensor = env.scene["contact_sensor"]

    # Find the indices corresponding to the foot bodies within the contact sensor data
    foot_body_indices, _ = contact_sensor.find_bodies([".*FOOT"])
    if not foot_body_indices:
        # Handle case where no foot bodies are found in the sensor
        print("[Warning] No foot bodies found in the general contact sensor.")
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Get net forces for all bodies from the general sensor
    # Use history[0] to get the most recent force data
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, :, :]
    # Select forces only for the foot bodies
    foot_forces = net_forces[:, foot_body_indices, :] # Shape: [num_envs, num_feet, 3]

    # Calculate the magnitude of forces for each foot
    foot_force_magnitudes = torch.norm(foot_forces, dim=-1) # Shape: [num_envs, num_feet]

    # Check if *any* foot force magnitude is above the threshold
    any_feet_on = torch.any(foot_force_magnitudes > contact_sensor.cfg.force_threshold, dim=1) # Shape: [num_envs]

    return any_feet_on

def get_center_of_mass_lin_vel(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Returns the center of mass linear velocity of the robot.
    Returns a tensor of shape (num_envs, 3)
    """
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    total_mass = torch.sum(robot.data.default_mass, dim=1).unsqueeze(-1).to(env.device) # Shape: (num_envs, 1)
    masses = robot.data.default_mass.unsqueeze(-1).to(env.device) # Shape: (num_envs, num_bodies, 1)
    weighed_lin_vels = robot.data.body_com_lin_vel_w * masses # Shape: (num_envs, num_bodies, 3)
    com_lin_vel = torch.sum(weighed_lin_vels, dim=1) / total_mass # Shape: (num_envs, 3)
    return com_lin_vel

def get_center_of_mass_pos(env: ManagerBasedEnv) -> torch.Tensor:
    """
    Returns the center of mass position of the robot.
    Returns a tensor of shape (num_envs, 3)
    """
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    total_mass = torch.sum(robot.data.default_mass, dim=1).unsqueeze(-1).to(env.device) # Shape: (num_envs, 1)
    masses = robot.data.default_mass.unsqueeze(-1).to(env.device) # Shape: (num_envs, num_bodies, 1)
    weighed_positions = robot.data.body_com_pos_w * masses # Shape: (num_envs, num_bodies, 3)
    com_pos = torch.sum(weighed_positions, dim=1) / total_mass # Shape: (num_envs, 3)
    return com_pos

def change_reward_weight(env: ManagerBasedEnv, reward_name: str, new_weight: float) -> None:
    """Change the weight of a reward term in the environment.
    
    Args:
        env: The environment instance
        reward_name: Name of the reward term to modify
        new_weight: New weight value to set
    """
    # Get the reward term config
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    # Update the weight
    term_cfg.weight = new_weight
    
def calculate_takeoff_errors(
    command_vec: torch.Tensor,
    takeoff_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Check if the takeoff was successful for the envs to be reset
    
    Args:
        command_vec: Tensor of shape (num_envs, 3) takeoff velocity command as x,y,z components
        takeoff_vec: Tensor of shape (num_envs, 3) takeoff velocity as x,y,z components
    
    Returns:
        angle_error: Tensor of shape (num_envs) angle error between command and takeoff velocity
        magnitude_ratio_error: Tensor of shape (num_envs) ratio error between command and takeoff velocity
    """

    angle_error = torch.acos(torch.sum(command_vec * takeoff_vec, dim=-1) / (torch.norm(command_vec, dim=-1) * torch.norm(takeoff_vec, dim=-1)))
    magnitude_ratio = torch.norm(takeoff_vec, dim=-1) / torch.norm(command_vec, dim=-1)
    return angle_error, magnitude_ratio

def convert_command_to_euclidean_vector(command: torch.Tensor) -> torch.Tensor:
    """Convert the command to a euclidean vector
    
    Args:
        command: Tensor of shape (num_envs, 2) containing [pitch, magnitude]
                 where pitch follows the convention:
                 - pitch = 0 means vertical jump (straight up)
                 - pitch increases clockwise toward horizontal
        
    Returns:
        Tensor of shape (num_envs, 3) containing [x, y, z] velocity components
    """

    result = torch.zeros((command.shape[0], 3), device=command.device)
    
    pitch_cmd = command[:, 0]
    magnitude_cmd = command[:, 1]
    
    # With pitch=0 being vertical:
    # - Horizontal component (x) = magnitude * sin(pitch)  
    # - Vertical component (z) = magnitude * cos(pitch)
    result[:, 0] = magnitude_cmd * torch.sin(pitch_cmd)  # x_dot (horizontal)
    result[:, 2] = magnitude_cmd * torch.cos(pitch_cmd)  # z_dot (vertical)
    # y_dot remains zero (no lateral movement)
    
    return result

def convert_height_length_to_pitch_magnitude(height: torch.Tensor, length: torch.Tensor, gravity: float = 9.81) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert height and length to pitch and magnitude.
    
    Using projectile motion physics with the convention:
    - pitch = 0 means vertical jump (straight up)  
    - pitch increases clockwise toward horizontal
    - pitch = 90° means horizontal trajectory
    
    Physics equations:
    - tan(pitch) = length/(4*height)
    - magnitude = sqrt(2*gravity*height / cos²(pitch))  [height-based formula]
    - Alternative: magnitude = sqrt(gravity*length / sin(2*pitch))  [range-based formula]
    
    Args:
        height: Target jump height (m)
        length: Target jump length (m) 
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        pitch: Launch angle in radians (0 = vertical, increases clockwise)
        magnitude: Launch velocity magnitude (m/s)
    """
    # Calculate pitch from tan(pitch) = length/(4*height)
    pitch = torch.atan(length / (4 * height))
    
    # Calculate magnitude using height-based formula to make height's role explicit
    # magnitude = sqrt(2*gravity*height / cos²(pitch))
    magnitude = torch.sqrt(2 * gravity * height / torch.cos(pitch)**2)
    
    return pitch, magnitude

def convert_pitch_magnitude_to_height_length(pitch: torch.Tensor, magnitude: torch.Tensor, gravity: float = 9.81) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pitch and magnitude to height and length.
    
    Using projectile motion physics with the convention:
    - pitch = 0 means vertical jump (straight up)
    - pitch increases clockwise toward horizontal  
    - pitch = 90° means horizontal trajectory
    
    Physics equations:
    - height = magnitude^2 * cos^2(pitch) / (2*gravity)
    - length = magnitude^2 * sin(2*pitch) / gravity
    
    Args:
        pitch: Launch angle in radians (0 = vertical, increases clockwise)
        magnitude: Launch velocity magnitude (m/s)
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        height: Jump height (m)
        length: Jump length (m)
    """
    # Calculate height from h = v^2 * cos^2(pitch) / (2*g)
    # Note: cos(pitch) because pitch=0 is vertical, so vertical component is cos(pitch)
    height = magnitude**2 * torch.cos(pitch)**2 / (2 * gravity)
    
    # Calculate length from l = v^2 * sin(2*pitch) / g  
    length = magnitude**2 * torch.sin(2 * pitch) / gravity
    
    return height, length

def convert_height_length_to_pitch_magnitude_from_position(
    height: torch.Tensor, 
    length: torch.Tensor, 
    current_pos: torch.Tensor,
    gravity: float = 9.81
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert height and length to pitch and magnitude, considering current position.
    
    This function calculates the required takeoff velocity vector to reach a target trajectory
    defined by height and length, starting from the current center of mass position.
    
    Args:
        height: Target jump height above current position (m)
        length: Target jump length from current position (m) 
        current_pos: Current center of mass position (num_envs, 3)
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        pitch: Launch angle in radians (0 = vertical, increases clockwise)
        magnitude: Launch velocity magnitude (m/s)
    """
    # The physics equations remain the same, but now height and length are relative to current position
    # Calculate pitch from tan(pitch) = length/(4*height)
    pitch = torch.atan(length / (4 * height))
    
    # Calculate magnitude using height-based formula
    # magnitude = sqrt(2*gravity*height / cos²(pitch))
    magnitude = torch.sqrt(2 * gravity * height / torch.cos(pitch)**2)
    
    return pitch, magnitude

def sample_command(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Use in env reset_idx to sample a new command for the specified environments.
    
    Now directly samples and returns height and length.
    """
    # Sample height and length from their respective ranges
    height_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_height_range)
    length_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_length_range)
    
    return height_cmd, length_cmd

def get_dynamic_takeoff_vector(env: ManagerBasedRLEnv, env_ids: Sequence[int] | None = None) -> torch.Tensor:
    """Calculate the required takeoff velocity vector based on current COM position and target trajectory.
    
    This function continuously updates the required velocity vector based on:
    - The robot's current center of mass position
    - The target height and length trajectory 
    - Physics calculations for projectile motion
    
    Args:
        env: Environment instance with target_height and target_length attributes
        env_ids: Environment indices to calculate for. If None, calculates for all environments.
        
    Returns:
        Tensor of shape (num_envs or len(env_ids), 3) containing [x, y, z] velocity components
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device)
    
    # Get current center of mass position
    current_com_pos = get_center_of_mass_pos(env)[env_ids]  # Shape: (len(env_ids), 3)
    
    # Get target trajectory parameters
    height = env.target_height[env_ids]  # Target height above current position
    length = env.target_length[env_ids]  # Target length from current position
    
    # Convert to pitch/magnitude based on current position
    pitch, magnitude = convert_height_length_to_pitch_magnitude_from_position(
        height, length, current_com_pos, gravity=9.81
    )
    
    # Convert to euclidean vector
    command_tensor = torch.stack([pitch, magnitude], dim=-1)
    return convert_command_to_euclidean_vector(command_tensor)


