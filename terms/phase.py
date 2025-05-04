from __future__ import annotations
from enum import IntEnum
import torch
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terms.utils import get_center_of_mass_lin_vel, all_feet_off_the_ground, any_feet_on_the_ground

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class Phase(IntEnum):
    CROUCH = 0
    TAKEOFF = 1
    FLIGHT = 2
    LANDING = 3 
    

def update_jump_phase(
    env: ManagerBasedEnv, 
) -> None:
    """Update the robot phase for the specified environments"""
    base_height = env.robot.data.root_pos_w[:, 2]
    base_com_vertical_vel = get_center_of_mass_lin_vel(env)[:, 2]
    
    takeoff_envs = env.jump_phase == Phase.TAKEOFF
    flight_envs = env.jump_phase == Phase.FLIGHT
    
    env.prev_jump_phase = env.jump_phase.clone()
        
    env.jump_phase[takeoff_envs & ((all_feet_off_ground & (base_height > env.cfg.takeoff_to_flight_height_trigger)) | (base_height > 0.30))] = Phase.FLIGHT
    
    neg_vertical_vel_envs = base_com_vertical_vel < 0.1
    any_feet_on_ground = any_feet_on_the_ground(env)
    env.jump_phase[flight_envs & neg_vertical_vel_envs & (any_feet_on_ground | (base_height < 0.1))] = Phase.LANDING

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

