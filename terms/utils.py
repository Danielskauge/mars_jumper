from __future__ import annotations
from enum import IntEnum

import torch

from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

class Phase(IntEnum):
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3 
    
def convert_vector_to_pitch_magnitude(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

    return torch.atan(vector[:, 0] / vector[:, 2]), torch.norm(vector, dim=-1)
    
    
def convert_pitch_magnitude_to_vector(pitch: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Convert the command to a euclidean vector
    
    Args:
        command: Tensor of shape (num_envs, 2) containing [pitch, magnitude]
                 where pitch follows the convention:
                 - pitch = 0 means vertical jump (straight up)
                 - pitch increases clockwise toward horizontal
        
    Returns:
        Tensor of shape (num_envs, 3) containing [x, y, z] velocity components
    """

    result = torch.zeros((pitch.shape[0], 3), device=pitch.device)
    
    
    # With pitch=0 being vertical:
    # - Horizontal component (x) = magnitude * sin(pitch)  
    # - Vertical component (z) = magnitude * cos(pitch)
    result[:, 0] = magnitude * torch.sin(pitch)  # x_dot (horizontal)
    result[:, 2] = magnitude * torch.cos(pitch)  # z_dot (vertical)
    # y_dot remains zero (no lateral movement)
    
    return result

def convert_height_length_to_pitch_magnitude(height: torch.Tensor, length: torch.Tensor, gravity: float = 9.81) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert height and length to pitch and magnitude.
    
    Using projectile motion physics with the convention:
    - pitch = 0 means vertical jump (straight up)  
    - pitch increases clockwise toward horizontal
    - pitch = 90° means horizontal trajectory
    
    Physics equations:
    - tan(pitch) = length/(4*height)
    - magnitude = sqrt(2*gravity*height / cos²(pitch))  [height-based formula]
    - Alternative: magnitude = sqrt(gravity*length / sin(2*pitch))  [range-based formula]
    
    Args:
        height: Target jump height (m)
        length: Target jump length (m) 
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        pitch: Launch angle in radians (0 = vertical, increases clockwise)
        magnitude: Launch velocity magnitude (m/s)
    """
    # Calculate pitch from tan(pitch) = length/(4*height)
    pitch = torch.atan(length / (4 * height))
    
    # Calculate magnitude using height-based formula to make height's role explicit
    # magnitude = sqrt(2*gravity*height / cos²(pitch))
    magnitude = torch.sqrt(2 * gravity * height / torch.cos(pitch)**2)
    
    return pitch, magnitude

def convert_pitch_magnitude_to_height_length(pitch: torch.Tensor, magnitude: torch.Tensor, gravity: float = 9.81) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert pitch and magnitude to height and length.
    
    Using projectile motion physics with the convention:
    - pitch = 0 means vertical jump (straight up)
    - pitch increases clockwise toward horizontal  
    - pitch = 90° means horizontal trajectory
    
    Physics equations:
    - height = magnitude^2 * cos^2(pitch) / (2*gravity)
    - length = magnitude^2 * sin(2*pitch) / gravity
    
    Args:
        pitch: Launch angle in radians (0 = vertical, increases clockwise)
        magnitude: Launch velocity magnitude (m/s)
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        height: Jump height (m)
        length: Jump length (m)
    """
    # Calculate height from h = v^2 * cos^2(pitch) / (2*g)
    # Note: cos(pitch) because pitch=0 is vertical, so vertical component is cos(pitch)
    height = magnitude**2 * torch.cos(pitch)**2 / (2 * gravity)
    
    # Calculate length from l = v^2 * sin(2*pitch) / g  
    length = magnitude**2 * torch.sin(2 * pitch) / gravity
    
    return height, length

def convert_height_length_to_pitch_magnitude_from_position(
    target_height: torch.Tensor, 
    target_length: torch.Tensor, 
    current_pos: torch.Tensor,
    gravity: float = 9.81
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert height and length to pitch and magnitude, considering current position.
    
    This function calculates the required takeoff velocity vector to reach a target trajectory
    defined by height and length, starting from the current center of mass position.
    
    Args:
        height: Target jump height in the world frame (m)
        length: Target jump length in the world frame (m) 
        current_pos: Current center of mass position in the world frame (num_envs, 3)
        gravity: Gravitational acceleration (m/s^2)
        
    Returns:
        pitch: Launch angle in radians (0 = vertical, increases clockwise, aka with forward jumps)
        magnitude: Launch velocity magnitude (m/s)pstep
    """
    # The physics equations remain the same, but now height and length are relative to current position
    # Calculate pitch from tan(pitch) = length/(4*height)
    delta_height = target_height - current_pos[:, 2]
    delta_length = target_length - current_pos[:, 0]
    
    # print("delta_height: {:.2f}".format(delta_height[0].item()))
    # print("delta_length: {:.2f}".format(delta_length[0].item()))

    pitch = torch.atan(delta_length / (4 * delta_height))
    
    # Calculate magnitude using height-based formula
    # magnitude = sqrt(2*gravity*height / cos²(pitch))
    magnitude = torch.sqrt(2 * gravity * delta_height / torch.cos(pitch)**2)
    
    return pitch, magnitude


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

