# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from terms.phase import Phase
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.assets import Articulation
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def dist_to_takeoff_vec(env: ManagerBasedRLEnv, dist_type: str = "linear") -> torch.Tensor:
    """ Distance from robots velocity vector to the takeoff velocity vector. 
    Use negative weight, unless dist_type is "exponential"
    Args:
        dist_type: "linear", "square", "exponential"
    """
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    takeoff_vec = env._command_buffer[takeoff_envs]
    robot_vel_vec = robot.data.body_com_state_w[takeoff_envs, 6:9]
    
    if dist_type == "linear":
        return torch.norm(robot_vel_vec - takeoff_vec, dim=-1)
    elif dist_type == "square":
        return torch.square(torch.norm(robot_vel_vec - takeoff_vec, dim=-1))
    elif dist_type == "exponential":
        return torch.exp(-torch.norm(robot_vel_vec - takeoff_vec, dim=-1))
    else:
        raise ValueError(f"Invalid distance type: {dist_type}")

def flat_orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Penalize the xy-components of the projected gravity vector. Sum of squared values.
    Active in landing and crouch phases.
    Returns postive value, use negative weight.
    """
    asset: RigidObject = env.scene[SceneEntityCfg("robot").name]
    landing_envs = env._phase_buffer == Phase.LANDING
    crouch_envs = env._phase_buffer == Phase.CROUCH
    active_envs = torch.logical_or(landing_envs, crouch_envs)
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[active_envs] = torch.sum(torch.square(asset.data.projected_gravity_b[active_envs, :2]), dim=1)
    
    return reward_tensor

def left_right_joint_symmetry(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Reward the robot for maintaining left-right joint angle symmetry.
    
    Calculates cosine similarity between corresponding joints on diagonal legs
    (LF-RH and RF-LH pairs) for knee, hip flexion, and abductor joints.
    
    Active in landing, takeoff, and crouch phases, but not in flight phase
    to allow for attitude control during aerial maneuvers.
    
    Returns:
        reward_tensor: Shape (num_envs,), normalized between 0 and 1
    """
    asset: Articulation = env.scene[SceneEntityCfg("robot").name]
    landing_envs = env._phase_buffer == Phase.LANDING
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    crouch_envs = env._phase_buffer == Phase.CROUCH
    active_envs = landing_envs | takeoff_envs | crouch_envs
    
    if not torch.any(active_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
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
    
    front_knee_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, LF_KFE_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, RH_KFE_idx].unsqueeze(1),
        dim=1
    )
    back_knee_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, RF_KFE_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, LH_KFE_idx].unsqueeze(1),
        dim=1
    )
    front_hip_flexion_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, LF_HFE_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, RH_HFE_idx].unsqueeze(1),
        dim=1
    )
    back_hip_flexion_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, RF_HFE_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, LH_HFE_idx].unsqueeze(1),
        dim=1
    )
    front_abductor_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, LF_HAA_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, RH_HAA_idx].unsqueeze(1),
        dim=1
    )
    back_abductor_similarity = torch.nn.functional.cosine_similarity(
        asset.data.joint_pos[active_envs, RF_HAA_idx].unsqueeze(1),
        asset.data.joint_pos[active_envs, LH_HAA_idx].unsqueeze(1),
        dim=1
    )
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    reward_tensor[active_envs] = (front_knee_similarity + 1)/2 + (back_knee_similarity + 1)/2 \
                               + (front_hip_flexion_similarity + 1)/2 + (back_hip_flexion_similarity + 1)/2 \
                               + (front_abductor_similarity + 1)/2 + (back_abductor_similarity + 1)/2
    
    reward_tensor[active_envs] /= 6  # Normalize to be between 0 and 1 (num joints = 6)
    
    return reward_tensor

def feet_ground_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Make the robot maintain ground contact. Active in crouch and landing phases
    Return:
        reward_tensor: Shape (num_envs,), Value between 0 and 1, +0.25 for each foot in contact with the ground
    """
    asset: Articulation = env.scene[SceneEntityCfg("robot").name]
    landing_envs = env._phase_buffer == Phase.LANDING
    crouch_envs = env._phase_buffer == Phase.CROUCH
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
    feet_idx, _ = asset.find_bodies(".*FOOT.*")
    
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    active_envs = landing_envs | crouch_envs
    
    if not torch.any(active_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    # Check if feet are in contact with the ground
    feet_forces = contact_sensor.data.net_forces_w[:, feet_idx] #tensor [num_envs, num_feet, 3]
    is_foot_in_contact = torch.norm(feet_forces, dim=-1) > contact_sensor.cfg.force_threshold #bool tensor [num_envs, num_feet]
    sum_feet_in_contact = torch.mean(is_foot_in_contact.float(), dim=-1) #scalar tensor [num_envs]
    
    reward_tensor[active_envs] = sum_feet_in_contact[active_envs]
    
    return reward_tensor

def equal_force_distribution(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward equal force distribution across the feet during takeoff and landing phases.
    
    This function calculates how evenly the contact forces are distributed across all feet
    that are in contact with the ground. The reward is higher when forces are more evenly
    distributed, which encourages balanced takeoffs and landings.
    
    The reward is calculated as:
    r = 1 - (std_dev / mean)
    
    where std_dev is the standard deviation of forces across feet in contact,
    and mean is the average force. This coefficient of variation approach rewards
    when forces are more uniform (lower std_dev relative to mean).
    """
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    landing_envs = env._phase_buffer == Phase.LANDING
    active_envs = takeoff_envs | landing_envs
    
    if not torch.any(active_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    feet_idx, _ = robot.find_bodies(".*FOOT.*")
    
    # Get forces for all feet
    feet_forces = contact_sensor.data.net_forces_w[:, feet_idx]
    force_magnitudes = torch.norm(feet_forces, dim=-1)
        
    reward_tensor = torch.zeros(env.num_envs, device=env.device)
    
    std_dev = torch.std(force_magnitudes, dim=-1)
    mean_force = torch.mean(force_magnitudes, dim=-1)
    
    reward_tensor[active_envs] = torch.exp(-torch.square(std_dev / mean_force)) #TODO: consider changing the reward shaping. Want to avoid the exponentail function "hidding" peaks
    
    return reward_tensor
    
    
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
    landing_envs = env._phase_buffer == Phase.LANDING
    if not torch.any(landing_envs):
        return torch.zeros(env.num_envs, device=env.device)
    
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    com_accel_norm = robot.data.body_lin_acc_w[landing_envs].norm(dim=-1)
    
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
    landing_envs = env._phase_buffer == Phase.LANDING
    contact_sensor: ContactSensor = env.scene[SceneEntityCfg("contact_forces").name]
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    feet_idx, _ = robot.find_joints(".*FOOT.*")
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
    
    crouch_envs = env._phase_buffer == Phase.CROUCH
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    knee_joints_idx, _ = robot.find_joints(".*KFE")
    
    angle_rad = robot.data.joint_pos[crouch_envs, :][:, torch.tensor(knee_joints_idx)]
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
    
    crouch_envs = env._phase_buffer == Phase.CROUCH
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    hip_joints_idx, _ = robot.find_joints(".*HFE.*")
    
    angle_rad = robot.data.joint_pos[crouch_envs, :][:, torch.tensor(hip_joints_idx)]
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
    
    crouch_envs = env._phase_buffer == Phase.CROUCH
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    abductor_joints_idx, _ = robot.find_joints(".*HAA.*")
    
    angle_rad = robot.data.joint_pos[crouch_envs, :][:, torch.tensor(abductor_joints_idx)]
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
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    cmd_vel_vec = env._command_buffer[takeoff_envs]
    cmd_vel_vec_cartesian = get_cartesian_cmd_vel_vec(cmd_vel_vec)
    
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    actual_vel_vec_cartesian = robot.data.body_com_state_w[takeoff_envs, 6:9]
    
    similarity = torch.nn.functional.cosine_similarity(cmd_vel_vec_cartesian, actual_vel_vec_cartesian)
    return (1 + similarity) / 2 #bound similarity to 0-1

def upward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Reward the robot for moving upward
    
    Args:
        env: The environment instance.
    """
    
    return torch.clamp(mdp.base_lin_vel(env)[:, 2], min=0.0)

def takeoff_vel_vec_magnitude(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Difference between the magnitude of the commanded and actual takeoff velocity vector magnitude, 
    Uses exponential reward shaping to bound error to 0-1
    """
    takeoff_envs = env._phase_buffer == Phase.TAKEOFF
    robot: Articulation = env.scene[SceneEntityCfg("robot").name]
    body_com_lin_vel_vec = robot.data.body_com_state_w[takeoff_envs, 6:9]
    body_com_lin_vel_magnitude = torch.norm(body_com_lin_vel_vec, dim=-1)
    cmd_magnitude = env._command_buffer[takeoff_envs, 1]
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

