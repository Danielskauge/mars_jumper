from __future__ import annotations

from collections.abc import Sequence
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
        
    Returns:
        Tensor of shape (num_envs, 3) containing [x, y, z] velocity components
    """

    result = torch.zeros((command.shape[0], 3), device=command.device)
    
    pitch_cmd = command[:, 0]
    magnitude_cmd = command[:, 1]
    
    result[:, 0] = magnitude_cmd * torch.sin(pitch_cmd)  # x_dot
    result[:, 2] = magnitude_cmd * torch.cos(pitch_cmd)  # z_dot
    # y_dot remains zero
    
    return result

def sample_command(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Use in env reset_idx to sample a new command for the specified environments."""
    pitch_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_pitch_range)
    magnitude_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_magnitude_range)
    return torch.stack([pitch_cmd, magnitude_cmd], dim=-1)
