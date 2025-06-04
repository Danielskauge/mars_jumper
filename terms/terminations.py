from __future__ import annotations
from git import List
import numpy as np
from pyparsing import Literal
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.contact_sensor.contact_sensor import ContactSensor
from terms.utils import Phase

DEG2RAD = np.pi/180

def illegal_contact(env: ManagerBasedRLEnv, threshold: float, body: List[Literal["base", "thigh", "hip", "feet", "shank"]]) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    net_contact_forces = env.contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    
    if body == "base":
        body_ids = env.base_body_idx
    elif body == "thigh":
        body_ids = env.thighs_body_idx
    elif body == "hip":
        body_ids = env.hips_body_idx
    elif body == "feet":
        body_ids = env.feet_body_idx
    elif body == "shank":
        body_ids = env.shanks_body_idx
    elif body == "all":
        body_ids = env.bodies_except_feet_idx
    else:
        raise ValueError(f"Invalid body type: {body}")
    
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )


def landing_walking(
    env: ManagerBasedRLEnv,
    x_tolerance: float = 0.15,
    y_tolerance: float = 0.10,
) -> torch.Tensor:
    """Terminate when the robot is walking/moving too far from target position during landing phase.
    
    In landing phase, the robot should stay near the target landing position.
    This termination triggers if the center of mass moves outside the allowed range
    around the target position.
    
    Args:
        env: The RL environment
        x_tolerance: Allowed distance from target_length in x direction (meters)  
        y_tolerance: Allowed distance from 0 in y direction (meters)
    
    Returns:
        Boolean tensor indicating which environments should terminate due to walking
    """
    # Only apply during landing phase
    condition_phase = env.jump_phase == Phase.LANDING
    
    # Target position: x should be target_length, y should be 0, z doesn't matter for this check
    target_x = env.target_length  # Shape: (num_envs,)
    target_y = torch.zeros_like(target_x)  # y target is 0
    
    # Calculate displacement from target position
    x_displacement = torch.abs(env.com_pos[:, 0] - target_x)
    y_displacement = torch.abs(env.com_pos[:, 1] - target_y)
    
    # Check if outside tolerance ranges
    condition_pos_x = x_displacement > x_tolerance
    condition_pos_y = y_displacement > y_tolerance
    
    return condition_phase & (condition_pos_x | condition_pos_y)

def takeoff_timeout(
    env: ManagerBasedRLEnv,
    timeout: float = 0.5,
) -> torch.Tensor:
    return (env.episode_length_buf * env.step_dt > timeout) & (env.jump_phase == Phase.TAKEOFF)

def landing_timeout(
    env: ManagerBasedRLEnv,
    timeout: float = 0.5,
) -> torch.Tensor:
    """Terminate when the robot has been in landing phase for too long.
    
    Note: This checks total episode time while in landing phase. For precise phase 
    duration tracking, the environment would need to track when landing phase started.
    
    Args:
        env: The RL environment.
        timeout: Maximum time allowed in landing phase (seconds).
    
    Returns:
        Boolean tensor indicating which environments should terminate due to landing timeout.
    """
    return (env.episode_length_buf * env.step_dt > timeout) & (env.jump_phase == Phase.LANDING)

def bad_knee_angle(
    env: ManagerBasedRLEnv) -> torch.Tensor:
    #Terminate when knee exceeds angle limits in flexion direction aka shank passes thight of the leg
    knee_joint_idx, _ = env.robot.find_joints(".*KFE.*")
    knee_angle = env.robot.data.joint_pos[:, knee_joint_idx] #shape (num_envs, 4)
    knee_angle_limit = env.robot.cfg.knee_joint_limits
    return torch.any(knee_angle > 180*DEG2RAD, dim=-1)

def bad_takeoff_at_descent(
    env: ManagerBasedRLEnv,
    relative_error_threshold: float = 0.1,
) -> torch.Tensor:
    return (env.takeoff_relative_error > relative_error_threshold) & (env.com_vel[:, 2] < 0.4) & env.flight_mask

def bad_takeoff_at_flight(
    env: ManagerBasedRLEnv,
    relative_error_threshold: float = 0.1,
) -> torch.Tensor:
    return (env.takeoff_relative_error > relative_error_threshold) & (env.jump_phase == Phase.FLIGHT)

def bad_takeoff_success_rate(
    env: ManagerBasedRLEnv,
    success_rate_threshold: float = 0.9,
    phase: Phase = Phase.LANDING,
) -> torch.Tensor:
    """Terminate at landing when the takeoff success rate is too low."""
    return (env.running_takeoff_success_rate < success_rate_threshold) & (env.jump_phase == phase)

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

def bad_yaw(
    env: ManagerBasedRLEnv,
    limit_angle: float = np.pi/2,
    phases: list[Phase] = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING]
) -> torch.Tensor:
    phase_match = torch.zeros_like(env.jump_phase, dtype=torch.bool)
    for phase in phases:
        phase_match = torch.logical_or(phase_match, env.jump_phase == phase)
    yaw = env.robot.data.heading_w
    return torch.logical_and(torch.abs(yaw) > limit_angle, phase_match)

def bad_roll(
    env: ManagerBasedRLEnv,
    limit_angle: float = np.pi/4,
    phases: list[Phase] = [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING]
) -> torch.Tensor:
    """Terminate when the robot's roll angle exceeds the limit in specified phases.
    
    Args:
        env: The RL environment.
        limit_angle: Maximum allowed roll angle in radians.
        phases: List of phases where this termination applies.
    
    Returns:
        Boolean tensor indicating which environments should terminate due to excessive roll.
    """
    # Check if the current phase for each environment is in the allowed phases
    phase_match = torch.zeros_like(env.jump_phase, dtype=torch.bool)
    for phase in phases:
        phase_match = torch.logical_or(phase_match, env.jump_phase == phase)
    
    # Calculate roll angle from projected gravity
    # Roll is rotation around forward (x) axis
    # In base frame: gravity_y and gravity_z components give us roll
    gravity_proj = env.robot.data.projected_gravity_b
    roll_angle = torch.atan2(gravity_proj[:, 1], -gravity_proj[:, 2])
    
    # Check if roll exceeds limit
    roll_bad = torch.abs(roll_angle) > limit_angle
    
    return torch.logical_and(roll_bad, phase_match)

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

def failed_to_reach_target_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.10,  # 10cm threshold
) -> torch.Tensor:
    """Terminate when robot is in flight phase and descending without reaching within threshold of target height.
    
    Args:
        env: The RL environment.
        height_threshold: Threshold in meters below target height (default 0.10m = 10cm).
    
    Returns:
        Boolean tensor indicating which environments should terminate due to failed height reach.
    """
    in_flight_phase = env.jump_phase == Phase.FLIGHT
    is_descending = env.com_vel[:, 2] < 0.3
    height_error = env.target_height - env.max_height_achieved
    return in_flight_phase & is_descending & (height_error > height_threshold)
    
def entered_flight(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Terminate when the robot enters the flight phase."""
    return env.jump_phase == Phase.FLIGHT


    
