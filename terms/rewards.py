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

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    
def get_cartesian_cmd_vel_vec(cmd_vel_vec: torch.Tensor) -> torch.Tensor:
    """ Convert the takeoff velocity vector command to cartesian coordinates 
    pitch is positive clockwise form positive z axis
    """
    pitch, magnitude = cmd_vel_vec.unbind(-1)
    x_dot = magnitude * torch.sin(pitch)
    z_dot = magnitude * torch.cos(pitch)
    y_dot = torch.zeros_like(x_dot)
    return torch.stack([x_dot, y_dot, z_dot], dim=-1)
    
def takeoff_vel_vec_angle_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Difference between the commanded and actual takeoff velocity vector angle measured by cosine similarity
    
    Args:
        env: The environment instance.
    """
    cmd_vel_vec = env.command_manager.get_command("takeoff_vel_vec")
    cmd_vel_vec_cartesian = get_cartesian_cmd_vel_vec(cmd_vel_vec)
    
    actual_vel_vec_cartesian = mdp.base_lin_vel(env)
    
    similarity = torch.nn.functional.cosine_similarity(cmd_vel_vec_cartesian, actual_vel_vec_cartesian)
    diff = 1 - similarity  # Convert to a distance measure TODO: use diff or similarity?
    
    diff[env._has_taken_off_buffer.bool()] = 0.0 # only use reward for robots that have not taken off
    
    return diff

def takeoff_vel_vec_magnitude_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """ Difference between the magnitude of the commanded and actual takeoff velocity vector magnitude
    
    Args:
        env: The environment instance.
    """
    cmd_magnitude = env.command_manager.get_command("takeoff_vel_vec")[:, 1]
    actual_magnitude = torch.norm(mdp.base_lin_vel(env), dim=-1)
    
    diff = cmd_magnitude - actual_magnitude
    
    diff[env._has_taken_off_buffer.bool()] = 0.0 # only use reward for robots that have not taken off
    
    return diff

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
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward

