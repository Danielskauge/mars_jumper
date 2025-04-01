
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from terms.phase import Phase
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    
def calculate_takeoff_errors(
    command_vec: torch.Tensor,
    takeoff_vec: torch.Tensor) -> torch.Tensor:
    """Check if the takeoff was successful for the envs to be reset
    
    Args:
        command_vec: Tensor of shape (num_envs, 3) takeoff velocity command as x,y,z components
        takeoff_vec: Tensor of shape (num_envs, 3) takeoff velocity as x,y,z components
    
    Returns:
        angle_error: Tensor of shape (num_envs) angle error between command and takeoff velocity
        magnitude_ratio_error: Tensor of shape (num_envs) ratio error between command and takeoff velocity
    """
    
    angle_error = torch.acos(torch.sum(command_vec * takeoff_vec, dim=-1) / (torch.norm(command_vec, dim=-1) * torch.norm(takeoff_vec, dim=-1)))
    magnitude_ratio_error = torch.abs(torch.norm(takeoff_vec, dim=-1) / torch.norm(command_vec, dim=-1) - 1)
    return angle_error, magnitude_ratio_error

def convert_command_to_euclidean_vector(command: torch.Tensor) -> torch.Tensor:
    """Convert the command to a euclidean vector"""
    pitch_cmd = command[:, 0]
    magnitude_cmd = command[:, 1]
    x_dot = magnitude_cmd * torch.sin(pitch_cmd)
    z_dot = magnitude_cmd * torch.cos(pitch_cmd)
    y_dot = torch.zeros_like(x_dot)
    return torch.stack([x_dot, y_dot, z_dot], dim=-1)

def sample_command(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> torch.Tensor:
    """Use in env reset_idx to sample a new command for the specified environments."""
    pitch_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_pitch_range)
    magnitude_cmd = torch.empty(len(env_ids), device=env.device).uniform_(*env.cmd_magnitude_range)
    return torch.stack([pitch_cmd, magnitude_cmd], dim=-1)

def update_robot_phase_buffer(
    env: ManagerBasedEnv, 
) -> None:
    """Update the robot phase for the specified environments.
    """
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    
    crouch_envs = env._phase_buffer == Phase.CROUCH
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    flight_envs = env._phase_buffer == Phase.FLIGHT
    landing_envs = env._phase_buffer == Phase.LANDING
    
    crouch_to_takeoff_transition_envs = check_crouch_to_takeoff(env, robot, crouch_envs)
    takeoff_to_flight_transition_envs = check_takeoff_to_flight(env, robot, takeoff_envs)
    flight_to_landing_transition_envs = check_flight_to_landing(env, robot, flight_envs)
    
    env._phase_buffer[crouch_to_takeoff_transition_envs] = Phase.TAKEOFF
    env._phase_buffer[takeoff_to_flight_transition_envs] = Phase.FLIGHT
    env._phase_buffer[flight_to_landing_transition_envs] = Phase.LANDING
    
    return torch.zeros(env.num_envs, 1, device=env.device) #Needed as observation term, but value not used by policy.
    
def check_crouch_to_takeoff(env: ManagerBasedRLEnv, robot: Articulation, env_ids: Sequence[int] = None) -> torch.Tensor:
    """Check if the robot should transition from crouch to takeoff"""
    angle_error_threshold_rad = 0.1*torch.pi
    target_knee_angle = torch.pi*0.8
    target_hip_angle = -torch.pi*0.4
    target_abductor_angle = 0.0
    max_joint_vel = 0.1

    #if all the current angles of a given robot are within the threshold, and joint velocities are close to 0, then transition to takeoff
    joint_pos = robot.data.joint_pos[env_ids]
    joint_vel = robot.data.joint_vel[env_ids]
    knee_joints_idx, _ = robot.find_joints(".*KFE")
    hip_joints_idx, _ = robot.find_joints(".*HFE")
    abductor_joints_idx, _ = robot.find_joints(".*HAA")
    knee_angle_error = torch.abs(joint_pos[:, knee_joints_idx] - target_knee_angle)
    hip_angle_error = torch.abs(joint_pos[:, hip_joints_idx] - target_hip_angle)
    abductor_angle_error = torch.abs(joint_pos[:, abductor_joints_idx] - target_abductor_angle)
    joint_vel_error = torch.abs(joint_vel)
    
    # Check if joint angles are within threshold and velocities are low
    angle_conditions = torch.cat([
        knee_angle_error < angle_error_threshold_rad,
        hip_angle_error < angle_error_threshold_rad,
        abductor_angle_error < angle_error_threshold_rad,
        joint_vel_error < max_joint_vel
    ], dim=1) #shape (num_envs, 4)
    
    transition_mask = torch.all(angle_conditions, dim=1)
    
    return env_ids[transition_mask]
    
def check_takeoff_to_flight(env: ManagerBasedRLEnv, robot: Articulation, env_ids: Sequence[int] = None) -> torch.Tensor:
    """Check if the robot should transition from takeoff to flight, based on feet-to-ground contact."""
    height_threshold = 0.20
    base_height = robot.data.root_pos_w[env_ids, 2]

    takeoff_envs = env_ids[env._phase_buffer[env_ids] == Phase.TAKEOFF]
    return takeoff_envs[base_height[takeoff_envs] > height_threshold]

def check_flight_to_landing(env: ManagerBasedRLEnv, robot: Articulation, env_ids: Sequence[int] = None) -> torch.Tensor:
    """Check if the robot should transition from flight to landing."""
    height_threshold = 0.20
    base_height = robot.data.root_pos_w[env_ids, 2]

    flight_envs = env_ids[env._phase_buffer[env_ids] == Phase.FLIGHT]
    return flight_envs[base_height[flight_envs] < height_threshold]


def check_transition_from_crouch_to_takeoff(env: ManagerBasedRLEnv, env_ids: Sequence[int] = None) -> torch.Tensor:
    """ Unused, but kept for reference.
    """
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    height_threshold = 0.15
    base_height = robot.data.root_pos_w[env_ids, 2]

    crouch_envs = env_ids[env._phase_buffer[env_ids] == Phase.CROUCH]
    transition_ids = crouch_envs[base_height[crouch_envs] > height_threshold]

    env._phase_buffer[transition_ids] = Phase.TAKEOFF

    # Return a tensor (required for observation terms).  The value doesn't matter, as not used by policy.
    return torch.zeros(env.num_envs, 1, device=env.device)
