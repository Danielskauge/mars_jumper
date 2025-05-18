from __future__ import annotations
from enum import IntEnum
import torch
from typing import TYPE_CHECKING
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terms.utils import get_center_of_mass_lin_vel, all_feet_off_the_ground, any_body_high_contact_force

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
    env.prev_jump_phase = env.jump_phase.clone()
    
    vel_not_increasing = base_com_vertical_vel < env.max_takeoff_vel_magnitude - 0.1 #add margin for numerical errors and small variations
    
    env.jump_phase[env.takeoff_mask & vel_not_increasing & (base_com_vertical_vel > 0.5)] = Phase.FLIGHT
    
    neg_vertical_vel_mask = base_com_vertical_vel < -0.5
    #contact_force_mask = any_body_high_contact_force(env)
    
    # Calculate new condition for any body being too low
    all_body_heights_w = env.robot.data.body_pos_w[:, :, 2]
    min_body_height_each_env = torch.min(all_body_heights_w, dim=1).values
    # Assuming the robot config is accessible via env.robot.cfg
    min_clearance_config = 0.1
    any_body_too_low = min_body_height_each_env < min_clearance_config

    env.jump_phase[env.flight_mask & neg_vertical_vel_mask & any_body_too_low] = Phase.LANDING
    
    env.takeoff_mask = env.jump_phase == Phase.TAKEOFF
    env.flight_mask = env.jump_phase == Phase.FLIGHT
    env.landing_mask = env.jump_phase == Phase.LANDING
    env.takeoff_to_flight_mask = (env.jump_phase == Phase.FLIGHT) & (env.prev_jump_phase == Phase.TAKEOFF)
    env.flight_to_landing_mask = (env.jump_phase == Phase.LANDING) & (env.prev_jump_phase == Phase.FLIGHT)

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

