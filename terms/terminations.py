
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

def bad_takeoff(
    env: ManagerBasedRLEnv,
    relative_error_threshold: float = 0.1,
) -> torch.Tensor:
    """Terminate when the robot's takeoff is too far from the desired takeoff vector."""
    return (env.takeoff_relative_error > relative_error_threshold) & (env.jump_phase == Phase.FLIGHT)

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


    
