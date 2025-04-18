from envs.env_cfg import MarsJumperEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from terms.phase import Phase
from terms.utils import update_jump_phase, sample_command, convert_command_to_euclidean_vector, calculate_takeoff_errors, log_phase_info
import logging
import numpy as np

logger = logging.getLogger(__name__)

class MarsJumperEnv(ManagerBasedRLEnv):
    cfg: "MarsJumperEnvCfg"

    def __init__(self, cfg, **kwargs):
        # Initialize internal variables for tracking success
        #Adjust the threshold as needed for your definition of success
        
        super().__init__(cfg=cfg, **kwargs)
        # Need to potentially re-initialize buffers after super().__init__ determines num_envs
        self.jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lin_vel_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range #Is updated by curriculum if used
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range
        self.cmd_curriculum_progress_ratio = 0.0
        self.env_steps_since_last_curriculum_update = 0
        
        self.reset_envs_success_rate = 0.0
        self.running_success_rate = 0.0
        
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Store previous phase before update
        self.prev_jump_phase = self.jump_phase.clone()
        update_jump_phase(self)
        
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)

        log_phase_info(self, extras)
                
        self.env_steps_since_last_curriculum_update += 1

        self.lin_vel_vec = self.robot.data.root_com_lin_vel_w

        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # Log success rate for the environments being reset
        # Call the parent's reset method to handle standard resets and logging

        if len(env_ids) > 0:
                        
            # Has to be called before sample_command, so the command in the to-be-reset envs is used.
            command_vec = convert_command_to_euclidean_vector(self.command[env_ids]) # Convert from (pitch, magnitude) to (x,y,z) components

            takeoff_vel_vec = self.lin_vel_vec[env_ids] # The last velocity vector in the takeoff phase

            cos_angle = torch.sum(command_vec * takeoff_vel_vec, dim=-1) / (torch.norm(command_vec, dim=-1) * torch.norm(takeoff_vel_vec, dim=-1))
            angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))

            takeoff_vs_cmd_magnitude_ratio = torch.norm(takeoff_vel_vec, dim=-1) / torch.norm(command_vec, dim=-1)
            
            takeoff_vec_error = torch.norm(takeoff_vel_vec - command_vec, dim=-1)
            takeoff_vs_cmd_magnitude_ratio_error = torch.abs(takeoff_vs_cmd_magnitude_ratio - 1)
            self.jump_success[env_ids] = (takeoff_vs_cmd_magnitude_ratio_error < self.cfg.takeoff_magnitude_ratio_error_threshold) & (angle_error < self.cfg.takeoff_angle_error_threshold) # Use AND, convert threshold to radians
            self.reset_envs_success_rate = torch.mean(self.jump_success[env_ids].float()).item() #the ratio of the nums envs to be reset that were successful

            # Update running success rate using exponential moving average
            alpha = 0.1# Smoothing factor (adjust as needed, smaller values mean slower updates)
            self.running_success_rate = alpha * self.reset_envs_success_rate + (1 - alpha) * self.running_success_rate

            # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
            self.command[env_ids] = sample_command(self, env_ids)
            
            relative_takeoff_vel_error = torch.norm(takeoff_vel_vec - command_vec, dim=-1) / torch.norm(command_vec, dim=-1)

            # Store log data *before* calling super()._reset_idx which resets extras["log"]
            log_data = {
                "reset_envs_success_rate": self.reset_envs_success_rate,
                "running_success_rate": self.running_success_rate, # Add running average to log
                "relative_takeoff_vel_error": relative_takeoff_vel_error.mean().item(),
                "takeoff_vec_error": takeoff_vec_error.mean().item(),
                "takeoff_vel_vec_magnitude": torch.norm(takeoff_vel_vec, dim=-1).mean().item(),
                "cmd_vec_magnitude": torch.norm(command_vec, dim=-1).mean().item(),
                "last_vel_vec_angle_error": angle_error.mean().item(),
                "takeoff_vs_cmd_magnitude_ratio": takeoff_vs_cmd_magnitude_ratio.mean().item(),
            }
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(log_data)
            
            # Reset tracking buffers for these environments
            self.jump_success[env_ids] = False
            self.lin_vel_vec[env_ids] = torch.zeros_like(self.lin_vel_vec[env_ids])
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
