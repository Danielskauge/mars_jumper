# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event functions for the Mars Jumper environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple
from envs import env
from terms.phase import Phase
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils
import logging
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
from torch import Tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def reset_robot_attitude_state(env: ManagerBasedRLEnv,
                               env_ids: Sequence[int],
                               height: float,
                               base_euler_angle_max_deg: float = 30.0, # max angle to sample from for roll, pitch, yaw
                               base_ang_vel_max_deg: float = 0.5, # max angular velocity (deg/s) to sample from for roll, pitch, yaw
                               joint_pos_max_limit_ratio: float = 0.1, # Ratio of joint range to sample from, centered on the range center
                               joint_vel_max_deg: float = 0.5, # Joint velocity range (rad/s)
                               ) -> None:
    """Resets the robot's base attitude state to a random orientation and angular velocity, and joints to random positions and velocities."""
    num_envs_to_reset = len(env_ids)

    # -- Base state --
    # Default root state (typically origin or initial height, zero velocity)
    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 2] = height  # Set z height

    # Random orientation
    max_angle_rad = torch.deg2rad(torch.tensor(base_euler_angle_max_deg, device=env.device))
    # Sample roll and pitch within limits
    roll = sample_uniform(-max_angle_rad, max_angle_rad, (num_envs_to_reset,), device=env.device)
    pitch = sample_uniform(-max_angle_rad, max_angle_rad, (num_envs_to_reset,), device=env.device)
    yaw = sample_uniform(-torch.pi, torch.pi, (num_envs_to_reset,), device=env.device)
    random_quat_w = quat_from_euler_xyz(roll, pitch, yaw)
    root_state[:, 3:7] = random_quat_w

    # Random angular velocity
    max_ang_vel_rad = torch.deg2rad(torch.tensor(base_ang_vel_max_deg, device=env.device))
    random_ang_vel = sample_uniform(-max_ang_vel_rad, max_ang_vel_rad, (num_envs_to_reset, 3), device=env.device)
    root_state[:, 10:13] = random_ang_vel

    # Set root state
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    # -- Joint state --
    # Default joint positions
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()

    # Random joint positions within ratio of limits
    lower_limits = env.robot.data.soft_joint_pos_limits[env_ids, :, 0]
    upper_limits = env.robot.data.soft_joint_pos_limits[env_ids, :, 1]
    joint_range = (upper_limits - lower_limits) * joint_pos_max_limit_ratio
    mid_pos = (upper_limits + lower_limits) / 2
    joint_pos = sample_uniform(mid_pos - joint_range/2, mid_pos + joint_range/2, joint_pos.shape, device=env.device)

    # Random joint velocities
    max_joint_vel_rad = torch.deg2rad(torch.tensor(joint_vel_max_deg, device=env.device))
    joint_vel = sample_uniform(-max_joint_vel_rad, max_joint_vel_rad, joint_vel.shape, device=env.device)

    # Set joint state
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_robot_initial_state(env: ManagerBasedRLEnv, 
                              env_ids: Sequence[int], 
                              crouch_flight_ratio: float,
                              crouch_hip_angle_range_rad: Tuple[float, float],
                              flight_base_euler_angles_range_rad: Tuple[float, float],
                              flight_flexor_angles_range_rad: Tuple[float, float],
                              flight_abductor_angles_range_rad: Tuple[float, float]
                              ) -> None:
    
    num_crouch_envs = int(crouch_flight_ratio * len(env_ids))
    crouch_ids = env_ids[:num_crouch_envs]
    flight_ids = env_ids[num_crouch_envs:]
    
    reset_robot_crouch_state(env, 
                             crouch_ids, 
                             crouch_hip_angle_range_rad)
    reset_robot_flight_state(env, 
                             flight_ids, 
                             flight_base_euler_angles_range_rad, 
                             flight_flexor_angles_range_rad, 
                             flight_abductor_angles_range_rad)
    

def set_phase_to_takeoff(env: ManagerBasedRLEnv, 
                              env_ids: Sequence[int],
                              ) -> None:
    
    """ Set the robot initial state to a crouch position with the specified hip angle, 
    meaning it is ready to take off, thus starts in takeoff phase"""
    
    # robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    
    # joint_pos = torch.zeros_like(robot.data.default_joint_pos[env_ids]) #Shape: (num_envs, num_joints)
    
    # hip_joint_idx, _ = robot.find_joints(".*HFE")
    # knee_joint_idx, _ = robot.find_joints(".*KFE")
    
    # hip_flexor_angle = torch.full_like(joint_pos[:, hip_joint_idx], hip_angle_rad) #Shape: (num_envs, 4)
    # knee_flexor_angle = -hip_flexor_angle * 2 #Shape: (num_envs, 4)
    
    # joint_pos[:, hip_joint_idx] = hip_flexor_angle #Shape: (num_envs, 4)
    # joint_pos[:, knee_joint_idx] = knee_flexor_angle 
    
    
    # joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids]) #Shape: (num_envs, num_joints)
    
    env._phase_buffer[env_ids] = Phase.TAKEOFF
    # robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
def reset_robot_crouch_state(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             hip_angle_range_rad: Tuple[float, float],
                             ) -> None:
    
    
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()
    
    hip_joint_idx, _ = env.robot.find_joints(".*HFE")
    knee_joint_idx, _ = env.robot.find_joints(".*KFE")
    
    hip_flexor_angle = torch.empty(len(env_ids), device=env.device).uniform_(*hip_angle_range_rad) #Shape: (num_envs)
    knee_flexor_angle = -hip_flexor_angle * 2
    
    joint_pos[:, hip_joint_idx] = hip_flexor_angle.unsqueeze(-1) #Shape: (num_envs, 1)
    joint_pos[:, knee_joint_idx] = knee_flexor_angle.unsqueeze(-1) 
    
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    hip_link_len = env.robot.cfg.HIP_LINK_LENGTH
    knee_link_len = env.robot.cfg.KNEE_LINK_LENGTH
    
    root_z: torch.Tensor = hip_link_len * torch.cos(hip_flexor_angle) + \
            knee_link_len * torch.cos(knee_flexor_angle + hip_flexor_angle) + 0.01
             
    root_state = env.robot.data.default_root_state[env_ids].clone()
    
    root_state[:, :3] = env.scene.env_origins[env_ids]
    
    root_state[:, 2] = root_z
    
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env._phase_buffer[env_ids] = Phase.CROUCH
    #print("Reset Event: Reset to crouch state of envs %s", env_ids)

    
def reset_robot_flight_state(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             base_euler_angles_range_rad: Tuple[float, float],
                             flexor_angles_range_rad: Tuple[float, float],
                             abductor_angles_range_rad: Tuple[float, float]
                             ) -> None:
    takeoff_vel_vec = env._command_buffer[env_ids]
    
    #print("In reset_robot_flight_state: takeoff_vel_vec: %s", takeoff_vel_vec)
    
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
    
    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x_sample
    root_state[:, 2] = z_sample
    root_state[:, 7] = x_dot_sample
    root_state[:, 9] = z_dot_sample
    
    #Base rotation quat
    euler_angles_sample = torch.empty(len(env_ids), 3, device=env.device).uniform_(*base_euler_angles_range_rad)
    
    quat_sample = math_utils.quat_from_euler_xyz(euler_angles_sample[:, 0], euler_angles_sample[:, 1], euler_angles_sample[:, 2])
    root_state[:, 3:7] = quat_sample
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    #Joint angles
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()
    
    hip_joints_idx, _ = env.robot.find_joints(".*HFE")
    knee_joints_idx, _ = env.robot.find_joints(".*KFE")
    abductor_joints_idx, _ = env.robot.find_joints(".*HAA")
    
    joint_pos[:, hip_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*flexor_angles_range_rad) #Broadcast shape (num_envs, 1) to (num_envs, 4)
    joint_pos[:, knee_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*flexor_angles_range_rad)
    joint_pos[:, abductor_joints_idx] = torch.empty((len(env_ids), 1), device=env.device).uniform_(*abductor_angles_range_rad)
    
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._phase_buffer[env_ids] = Phase.FLIGHT
    
    #print("Reset Event: Reset to flight state of envs %s", env_ids)
    

def set_joint_limits_from_config(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """Sets the joint position limits based on values defined in the robot's config."""
    
    # Ensure the robot config has the expected limit attributes
    if not hasattr(env.robot.cfg, "HIP_ABDUCTION_ANGLE_LIMITS_RAD") or \
       not hasattr(env.robot.cfg, "KNEE_ANGLE_LIMITS_RAD") or \
       not hasattr(env.robot.cfg, "HIP_FLEXION_ANGLE_LIMITS_RAD"):
        logger.error("Robot config is missing expected joint limit attributes. Skipping limit setting.")
        return

    # Get current joint limits (shape: (num_envs, num_joints, 2))
    current_limits = env.robot.data.joint_limits.clone()

    # Find joint indices
    haa_joint_idx, _ = env.robot.find_joints(env.robot.cfg.HIP_ABDUCTION_JOINTS_REGEX)
    hfe_joint_idx, _ = env.robot.find_joints(env.robot.cfg.HIP_FLEXION_JOINTS_REGEX) # Corrected regex source
    kfe_joint_idx, _ = env.robot.find_joints(env.robot.cfg.KNEE_JOINTS_REGEX)       # Corrected regex source

    # Apply limits from config
    # Note: Limits are often defined as (min, max), but PhysX expects (lower, upper)
    # Ensure the order matches PhysX expectations. The write_joint_limits_to_sim handles this internally usually.
    # We assume the config defines limits as (lower, upper) matching PhysX.
    if haa_joint_idx:
        current_limits[:, haa_joint_idx, 0] = env.robot.cfg.HIP_ABDUCTION_ANGLE_LIMITS_RAD[0]
        current_limits[:, haa_joint_idx, 1] = env.robot.cfg.HIP_ABDUCTION_ANGLE_LIMITS_RAD[1]
    if hfe_joint_idx:
        current_limits[:, hfe_joint_idx, 0] = env.robot.cfg.HIP_FLEXION_ANGLE_LIMITS_RAD[0]
        current_limits[:, hfe_joint_idx, 1] = env.robot.cfg.HIP_FLEXION_ANGLE_LIMITS_RAD[1]
    if kfe_joint_idx:
        # Be careful with KFE limits if they are defined (max, min) in config
        lower = min(env.robot.cfg.KNEE_ANGLE_LIMITS_RAD)
        upper = max(env.robot.cfg.KNEE_ANGLE_LIMITS_RAD)
        current_limits[:, kfe_joint_idx, 0] = lower
        current_limits[:, kfe_joint_idx, 1] = upper

    # Write the updated limits to the simulation
    # Using warn_limit_violation=False as this happens at startup before resets typically
    env.robot.write_joint_limits_to_sim(current_limits, warn_limit_violation=False)
    logger.info("Successfully set joint limits from robot configuration.")
