
from __future__ import annotations
import numpy as np
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import RigidObject
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from terms.phase import Phase

DEG2RAD = np.pi/180



def bad_knee_angle(
    env: ManagerBasedRLEnv) -> torch.Tensor:
    #Terminate when knee exceeds angle limits in flexion direction aka shank passes thight of the leg
    knee_joint_idx, _ = env.robot.find_joints(".*KFE.*")
    knee_angle = env.robot.data.joint_pos[:, knee_joint_idx] #shape (num_envs, 4)
    knee_angle_limit = env.robot.cfg.knee_joint_limits
    return torch.any(knee_angle > 180*DEG2RAD, dim=-1)

def bad_takeoff_at_flight(
    env: ManagerBasedRLEnv,
    relative_error_threshold: float = 0.1,
) -> torch.Tensor:
    return (env.takeoff_relative_error > relative_error_threshold) & (env.jump_phase == Phase.FLIGHT)

def bad_takeoff_success_rate(
    env: ManagerBasedRLEnv,
    success_rate_threshold: float = 0.9,
) -> torch.Tensor:
    """Terminate at landing when the takeoff success rate is too low."""
    return (env.running_takeoff_success_rate < success_rate_threshold) & (env.jump_phase == Phase.LANDING)

def bad_flight_success_rate(
    env: ManagerBasedRLEnv,
    success_rate_threshold: float = 0.9,
) -> torch.Tensor:
    """Terminate at landing when the flight success rate is too low."""
    return (env.running_flight_success_rate < success_rate_threshold) & (env.jump_phase == Phase.LANDING)

def bad_takeoff_at_landing(
    env: ManagerBasedRLEnv,
    relative_error_threshold: float = 0.1,
) -> torch.Tensor:
    return (env.takeoff_relative_error > relative_error_threshold) & (env.jump_phase == Phase.LANDING)

def bad_flight_at_landing(
    env: ManagerBasedRLEnv,
    angle_error_threshold: float = 10*DEG2RAD,
) -> torch.Tensor:
    return (env.angle_error_at_landing > angle_error_threshold) & (env.jump_phase == Phase.LANDING)

def bad_orientation(
    env: ManagerBasedRLEnv,
    limit_angle: float = np.pi/2,
    phases: list[Phase] = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING]
) -> torch.Tensor:
    
    """Terminate when the robot's orientation is too far from the desired orientation limits."""
    # Calculate the orientation condition (a boolean tensor)
    orientation_bad = torch.acos(-env.robot.data.projected_gravity_b[:, 2]).abs() > limit_angle
    
    # Check if the current phase for each environment is in the allowed phases
    # Initialize a tensor of False with the same shape as env.jump_phase
    phase_match = torch.zeros_like(env.jump_phase, dtype=torch.bool)
    # Iterate through allowed phases and set corresponding entries to True
    for phase in phases:
        phase_match = torch.logical_or(phase_match, env.jump_phase == phase)
        
    # Terminate only if both orientation is bad AND phase is allowed for that env
    return torch.logical_and(orientation_bad, phase_match)

def self_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*"),
    threshold: float = 1.0
    
) -> torch.Tensor:
    """Terminate when the robot collides with itself.
    """
    # # extract the used quantities (to enable type-hinting)
    # contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # contact_forces = contact_sensor.data.net_forces_w_history[:, :, ]
    
    
    
    # asset: RigidObject = env.scene[asset_cfg.name]
    
    pass


def reached_takeoff_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.22
) -> torch.Tensor:
    """Terminate when the robot reaches a certain height."""
    
    height = env.robot.data.root_pos_w[:, 2]
    return height > height_threshold

def landed(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Terminate when the robot lands on the ground."""
    #TODO: this uses the phase defintion of when to start to land, not the same as landing in terms of touch down
    return env.jump_phase == Phase.LANDING
    
def entered_flight(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Terminate when the robot enters the flight phase."""
    return env.jump_phase == Phase.FLIGHT


    
