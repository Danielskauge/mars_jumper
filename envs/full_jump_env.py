from envs.full_jump_env_cfg import FullJumpEnvCfg
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

class FullJumpEnv(ManagerBasedRLEnv):
    cfg: "FullJumpEnvCfg"

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
        
        #Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.lin_vel_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self.reset_envs_takeoff_success_rate = 0.0
        
        #Flight Success
        self.flight_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_envs_flight_success_rate = 0.0
        
        #Jump Phase
        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        
        #Command Curriculum
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range #Is updated by curriculum if used
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range
        self.cmd_curriculum_progress_ratio = 0.0
        self.env_steps_since_last_curriculum_update = 0
        
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        update_jump_phase(self)
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        log_phase_info(self, extras)
                
        self.env_steps_since_last_curriculum_update += 1
        self.lin_vel_vec = self.robot.data.root_com_lin_vel_w

        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            # Has to be called before sample_command, so the command in the to-be-reset envs is used.
            command_vec = convert_command_to_euclidean_vector(self.command[env_ids]) # Convert from (pitch, magnitude) to (x,y,z) components

            takeoff_vel_vec = self.lin_vel_vec[env_ids] # The last velocity vector in the takeoff phase

            cos_angle = torch.sum(command_vec * takeoff_vel_vec, dim=-1) / (torch.norm(command_vec, dim=-1) * torch.norm(takeoff_vel_vec, dim=-1))
            angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))

            takeoff_vs_cmd_magnitude_ratio = torch.norm(takeoff_vel_vec, dim=-1) / torch.norm(command_vec, dim=-1)
            takeoff_vec_error = torch.norm(takeoff_vel_vec - command_vec, dim=-1)
            takeoff_vs_cmd_magnitude_ratio_error = torch.abs(takeoff_vs_cmd_magnitude_ratio - 1)
            self.takeoff_success[env_ids] = (takeoff_vs_cmd_magnitude_ratio_error < self.cfg.takeoff_magnitude_ratio_error_threshold) & (angle_error < self.cfg.takeoff_angle_error_threshold) # Use AND, convert threshold to radians
            self.takeoff_success_rate = torch.mean(self.takeoff_success[env_ids].float()).item() #the ratio of the nums envs to be reset that were successful

            self.command[env_ids] = sample_command(self, env_ids) # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset

            # Store log data *before* calling super()._reset_idx which resets extras["log"]
            flight_log_data = {
                "flight_success_rate": self.flight_success_rate,
                "flight_angle_error": angle_error.mean().item(), #TODO: Add ang_vel_error
                "flight_ang_vel_error": ang_vel_error.mean().item(), #TODO: Add ang_vel_error
            }
            takeoff_log_data = {
                "takeoff_success_rate": self.takeoff_success_rate,
                "takeoff_cmd_error": takeoff_vec_error.mean().item(),
            }
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            
            # Reset tracking buffers for these environments
            self.takeoff_success[env_ids] = False
            self.flight_success[env_ids] = False
            
            self.lin_vel_vec[env_ids] = torch.zeros_like(self.lin_vel_vec[env_ids])
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
