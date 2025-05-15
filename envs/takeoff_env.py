from envs.takeoff_env_cfg import MarsJumperEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from terms.phase import Phase, update_jump_phase, log_phase_info
from terms.utils import sample_command, convert_command_to_euclidean_vector, get_center_of_mass_lin_vel
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
        self.com_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range #Is updated by curriculum if used
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range
        self.cmd_curriculum_progress_ratio = 0.0
        self.env_steps_since_last_curriculum_update = 0
        self.mean_episode_env_steps = 0
        
        self.success_rate = 0.0
        
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Store previous phase before update
        self.prev_jump_phase = self.jump_phase.clone()
        update_jump_phase(self)
        
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)

        log_phase_info(self, extras)
                
        self.env_steps_since_last_curriculum_update += 1

        self.com_lin_vel = get_center_of_mass_lin_vel(self)

        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # Log success rate for the environments being reset
        # Call the parent's reset method to handle standard resets and logging

        if len(env_ids) > 0:
                        
            # Has to be called before sample_command, so the command in the to-be-reset envs is used.
            cmd_vec = convert_command_to_euclidean_vector(self.command[env_ids]) # Convert from (pitch, magnitude) to (x,y,z) components
            liftoff_com_lin_vel = self.com_lin_vel[env_ids] # The last velocity vector before termination

            cos_angle = torch.sum(cmd_vec * liftoff_com_lin_vel, dim=-1) / (torch.norm(cmd_vec, dim=-1) * torch.norm(liftoff_com_lin_vel, dim=-1))
            angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
            cmd_error = torch.norm(liftoff_com_lin_vel - cmd_vec, dim=-1)
            relative_cmd_error = torch.norm(liftoff_com_lin_vel - cmd_vec, dim=-1) / torch.norm(cmd_vec, dim=-1)
            magnitude_ratio_error = torch.norm(liftoff_com_lin_vel, dim=-1) / torch.norm(cmd_vec, dim=-1) - 1
            
            magnitude_success = torch.abs(magnitude_ratio_error) < self.cfg.success_criteria.takeoff_magnitude_ratio_error_threshold
            angle_success = angle_error < self.cfg.success_criteria.takeoff_angle_error_threshold

            terminated_by_liftoff = self.termination_manager.get_term("entered_flight")[env_ids]

            self.jump_success[env_ids] = magnitude_success & angle_success & terminated_by_liftoff # Use AND, convert threshold to radians
            
            #Exponentially update success rate
            alpha = len(env_ids) / self.num_envs
            self.success_rate = self.success_rate * (1 - alpha) + alpha * torch.mean(self.jump_success[env_ids].float()).item()
            
            self.mean_episode_env_steps = self.mean_episode_env_steps * (1 - alpha) + alpha * torch.mean(self.episode_length_buf[env_ids].float()).item()

            # Store log data *before* calling super()._reset_idx which resets extras["log"]
            log_data = {
                "success_rate": self.success_rate,
                "relative_cmd_error": relative_cmd_error.mean().item(),
                "magnitude_ratio_error": magnitude_ratio_error.mean().item(),
                "cmd_error": cmd_error.mean().item(),
                "cmd_vec_magnitude": torch.norm(cmd_vec, dim=-1).mean().item(),
                "cmd_angle_error": angle_error.mean().item(),
                
            }
            # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
            self.command[env_ids] = sample_command(self, env_ids)
            
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(log_data)
            
            # Reset tracking buffers for these environments
            self.jump_success[env_ids] = False
            self.com_lin_vel[env_ids] = torch.zeros_like(self.com_lin_vel[env_ids])
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
