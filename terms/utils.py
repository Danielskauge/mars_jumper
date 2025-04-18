from __future__ import annotations

from collections.abc import Sequence
import os
from typing import TYPE_CHECKING

import wandb
import yaml
from terms.phase import Phase
import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

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
    magnitude_ratio = takeoff_vec[:, 2] / command_vec[:, 2]
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

def update_jump_phase(
    env: ManagerBasedEnv, 
) -> None:
    """Update the robot phase for the specified environments"""
    base_height = env.robot.data.root_pos_w[:, 2]
    
    crouch_envs = env.jump_phase == Phase.CROUCH
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    flight_envs = env.jump_phase == Phase.FLIGHT
    
    env.prev_jump_phase = env.jump_phase.clone()
    
    env.jump_phase[crouch_envs & (base_height > env.cfg.crouch_to_takeoff_height_trigger)] = Phase.TAKEOFF
    env.jump_phase[takeoff_envs & (base_height > env.cfg.takeoff_to_flight_height_trigger)] = Phase.FLIGHT
    env.jump_phase[flight_envs & (base_height < env.cfg.flight_to_landing_height_trigger)] = Phase.LANDING


def log_phase_info(env: ManagerBasedEnv, extras: dict):
    """Logs the distribution of phases and average height per phase to the extras dict.
    
    Args:
        env: Environment instance
        extras: Extras dictionary to update with logs
        log_frequency: How often to compute full statistics (default: every 20 steps)
                      Set to 1 to log every step
    """
    # Quick exit if it's not time to log yet - use a step counter on the env
    
    # Keep computation on GPU as much as possible to avoid transfers
    
    # Calculate phase distribution on GPU
    num_envs = env.num_envs
    phases = env.jump_phase  # Keep on GPU
    
    # Calculate statistics for each phase directly on GPU
    phase_log = {}
    for phase_enum in Phase:
        phase_val = phase_enum.value
        phase_name = phase_enum.name
        
        # Calculate count and percentage
        phase_mask = (phases == phase_val)
        count = torch.sum(phase_mask).item()  # Single value transfer
        phase_log[f"phase_dist/{phase_name}"] = count / num_envs
        
        # Calculate average height only if environments exist in this phase
        if count > 0:
            # Compute mean height directly on GPU, only transfer the result
            heights = env.robot.data.root_pos_w[:, 2]
            avg_height = torch.mean(heights[phase_mask]).item()  # Single value transfer
            phase_log[f"avg_height/{phase_name}"] = avg_height
        else:
            phase_log[f"avg_height/{phase_name}"] = 0.0
             
    extras["log"].update(phase_log)

# def check_crouch_to_takeoff(env: ManagerBasedRLEnv, robot: Articulation, env_ids: Sequence[int] = None) -> torch.Tensor:
#     """Check if the robot should transition from crouch to takeoff"""
#     angle_error_threshold_rad = 0.1*torch.pi
#     target_knee_angle = torch.pi*0.8
#     target_hip_angle = -torch.pi*0.4
#     target_abductor_angle = 0.0
#     max_joint_vel = 0.1

#     #if all the current angles of a given robot are within the threshold, and joint velocities are close to 0, then transition to takeoff
#     joint_pos = robot.data.joint_pos[env_ids]
#     joint_vel = robot.data.joint_vel[env_ids]
#     knee_joints_idx, _ = robot.find_joints(".*KFE")
#     hip_joints_idx, _ = robot.find_joints(".*HFE")
#     abductor_joints_idx, _ = robot.find_joints(".*HAA")
#     knee_angle_error = torch.abs(joint_pos[:, knee_joints_idx] - target_knee_angle)
#     hip_angle_error = torch.abs(joint_pos[:, hip_joints_idx] - target_hip_angle)
#     abductor_angle_error = torch.abs(joint_pos[:, abductor_joints_idx] - target_abductor_angle)
#     joint_vel_error = torch.abs(joint_vel)
    
#     # Check if joint angles are within threshold and velocities are low
#     angle_conditions = torch.cat([
#         knee_angle_error < angle_error_threshold_rad,
#         hip_angle_error < angle_error_threshold_rad,
#         abductor_angle_error < angle_error_threshold_rad,
#         joint_vel_error < max_joint_vel
#     ], dim=1) #shape (num_envs, 4)
    
#     transition_mask = torch.all(angle_conditions, dim=1)
    
#     return env_ids[transition_mask]
    

