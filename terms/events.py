# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the Mars Jumper environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
import logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def reset_command(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """Reset the command for the specified environments.
    
    This is handeled here instead of in the _resample_command method of the command term,
    as the command is needed in both the events terms for state initialization,
    which are run before the command manager is reset.
    
    Args:
        env: The environment instance
        env_ids: The environment IDs to reset the command for
    """
    command_cfg = env.command_manager.get_term("takeoff_vel_vec").cfg
    
    env._command_buffer[env_ids, 0] = torch.empty(len(env_ids), device=env.device).uniform_(*command_cfg.ranges.pitch_rad)
    env._command_buffer[env_ids, 1] = torch.empty(len(env_ids), device=env.device).uniform_(*command_cfg.ranges.magnitude)
    
    print("Reset Event: Command resampled: %s", env._command_buffer[env_ids])
    
    
    
def reset_robot_initial_state(env: ManagerBasedRLEnv, 
                              env_ids: Sequence[int], 
                              crouch_flight_ratio: float = 0.5,
                              crouch_flexor_angle_range_rad: Tuple[float, float] = (-torch.pi/4, torch.pi/4),
                              flight_base_euler_angles_range_rad: Tuple[float, float] = (-torch.pi/16, torch.pi/16),
                              flight_flexor_angles_range_rad: Tuple[float, float] = (-torch.pi/4, torch.pi/4),
                              flight_abductor_angles_range_rad: Tuple[float, float] = (-torch.pi/8, torch.pi/8)
                              ) -> None:
    
    num_crouch_envs = int(crouch_flight_ratio * len(env_ids))
    crouch_ids = env_ids[:num_crouch_envs]
    flight_ids = env_ids[num_crouch_envs:]
    
    reset_robot_crouch_state(env, crouch_ids, crouch_flexor_angle_range_rad)
    reset_robot_flight_state(env, 
                             flight_ids, 
                             flight_base_euler_angles_range_rad, 
                             flight_flexor_angles_range_rad, 
                             flight_abductor_angles_range_rad)
    
    
    
def reset_robot_crouch_state(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             flexor_angle_range_rad: Tuple[float, float],
                             ) -> None:
    
    
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    
    hip_joint_idx, _ = robot.find_joints(".*HFE")
    knee_joint_idx, _ = robot.find_joints(".*KFE")
    
    hip_flexor_angle = torch.empty(len(env_ids), device=env.device).uniform_(*flexor_angle_range_rad) #Shape: (num_envs)
    knee_flexor_angle = -hip_flexor_angle
    
    joint_pos[:, hip_joint_idx] = hip_flexor_angle.unsqueeze(-1) #Shape: (num_envs, 1)
    joint_pos[:, knee_joint_idx] = knee_flexor_angle.unsqueeze(-1) 
    
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    hip_link_len = robot.cfg.HIP_LINK_LENGTH
    knee_link_len = robot.cfg.KNEE_LINK_LENGTH
    
    root_z: torch.Tensor = hip_link_len * torch.cos(hip_flexor_angle) + \
            knee_link_len * torch.cos(knee_flexor_angle + hip_flexor_angle) + \
            robot.cfg.PAW_DIAMETER/2 + 0.01 
             
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 2] = root_z
    
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    print("Reset Event: Reset to crouch state of envs %s", env_ids)

    
def reset_robot_flight_state(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             base_euler_angles_range_rad: Tuple[float, float] = (-torch.pi/16, torch.pi/16),
                             flexor_angles_range_rad: Tuple[float, float] = (-torch.pi/4, torch.pi/4),
                             abductor_angles_range_rad: Tuple[float, float] = (-torch.pi/8, torch.pi/8)
                             ) -> None:

    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    
    takeoff_vel_vec = env._command_buffer[env_ids]
    
    print("In reset_robot_flight_state: takeoff_vel_vec: %s", takeoff_vel_vec)
    
    x_dot_0 = takeoff_vel_vec[:, 1] * torch.sin(takeoff_vel_vec[:, 0]).squeeze(-1) #Shape: (num_envs)
    z_dot_0 = takeoff_vel_vec[:, 1] * torch.cos(takeoff_vel_vec[:, 0]).squeeze(-1) #Shape: (num_envs)
    
    assert torch.all(x_dot_0 >= 0), f"x_dot_0 is negative: {x_dot_0}"
    assert torch.all(z_dot_0 > 0), f"z_dot_0 is not positive: {z_dot_0}"
    
    z_dot_dot = env.cfg.mars_gravity
    
    t_top = -z_dot_0/z_dot_dot
    
    z_top = z_dot_0 * t_top + 0.5 * z_dot_dot * t_top**2
    z_min = 0.15 #lowest height for sampling
    
    assert torch.all(z_top > z_min), "Maximum height is less than the minimum sampling height, takeoff velocity is too low for the robot to sufficiently take off"
    
    
    z_min_time_1 = (-z_dot_0 - torch.sqrt(z_dot_0**2 + 2*z_dot_dot*z_min))/z_dot_dot #TODO: think through
    z_min_time_2 = (-z_dot_0 + torch.sqrt(z_dot_0**2 + 2*z_dot_dot*z_min))/z_dot_dot
    
    assert torch.all(z_min_time_1 > 0), "z_min_time_1 is not positive"
    assert torch.all(z_min_time_2 > 0), "z_min_time_2 is not positive"
    
    sample_time = math_utils.sample_uniform(z_min_time_1, z_min_time_2, (len(env_ids)), device=env.device) #Shape: (num_envs)
    
    x_sample = x_dot_0 * sample_time
    z_sample = z_dot_0 * sample_time + 0.5 * z_dot_dot * sample_time**2
    
    x_dot_sample = x_dot_0
    z_dot_sample = z_dot_0 + z_dot_dot * sample_time
    
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x_sample
    root_state[:, 2] = z_sample
    root_state[:, 7] = x_dot_sample
    root_state[:, 9] = z_dot_sample
    
    #Base rotation quat
    euler_angles_sample = torch.empty(len(env_ids), 3, device=env.device).uniform_(*base_euler_angles_range_rad)
    
    quat_sample = math_utils.quat_from_euler_xyz(euler_angles_sample[:, 0], euler_angles_sample[:, 1], euler_angles_sample[:, 2])
    root_state[:, 3:7] = quat_sample
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    #Joint angles
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    
    hip_joints_idx, _ = robot.find_joints(".*HFE")
    knee_joints_idx, _ = robot.find_joints(".*KFE")
    abductor_joints_idx, _ = robot.find_joints(".*HAA")
    
    joint_pos[:, hip_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*flexor_angles_range_rad) #Broadcast shape (num_envs, 1) to (num_envs, 4)
    joint_pos[:, knee_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*flexor_angles_range_rad)
    joint_pos[:, abductor_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*abductor_angles_range_rad)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    print("Reset Event: Reset to flight state of envs %s", env_ids)
    
def reset_episodic_buffers(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """Reset buffers that need to be reset at the end of each episode.
    
    _command_buffer should not be reset here, as it is sampled in the reset_command event, 
    and resetting would override the command if this term is run before the state initialization term.
    
    Args:
        env: The environment instance
        env_ids: The environment IDs to reset the buffers for
    """
    env._has_taken_off_buffer[env_ids] = 0.0
    
    print("Reset Event: Reset episodic buffers of envs %s", env_ids)
    

def initialize_buffers(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """
    Initialize extra buffers that are not included in the default environment buffers. #TODO: find out if buffers also need to be reset at episode end
    """
    env._has_taken_off_buffer = torch.zeros(env.num_envs, device=env.device)
    env._takeoff_vel_magnitude_curriculum_progress_ratio = 0.0
    env._command_buffer = torch.zeros(env.num_envs, 2, device=env.device)
    
    print("Event:Initialized buffers")
 
