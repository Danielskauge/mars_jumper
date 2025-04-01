from envs.env_cfg import MarsJumperEnvCfg
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
from collections.abc import Sequence
import torch
from terms.utils import update_robot_phase_buffer, sample_command, convert_command_to_euclidean_vector, calculate_takeoff_errors

class MarsJumperEnv(ManagerBasedRLEnv):
    cfg: "MarsJumperEnvCfg"

    def __init__(self, cfg, **kwargs):
        # Initialize internal variables for tracking success
        #Adjust the threshold as needed for your definition of success
        self._peak_height_buf = None
        self._success_buf = None
        
        super().__init__(cfg=cfg, **kwargs)
        # Need to potentially re-initialize buffers after super().__init__ determines num_envs
        self._vel_vec = torch.zeros(self.num_envs, 3, device=self.device)
        self._success_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._phase_buffer = torch.zeros(self.num_envs, device=self.device)
        self._command_buffer = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range #Is updated by curriculum if used
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range
        self.cmd_curriculum_progress_ratio = 0.0
        
    
        
        #self.scene.clone_environments(copy_from_source=False)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # Standard step execution from the parent class
        
        assert not torch.any(torch.isnan(self._command_buffer)), "NaN in command_buffer"
        assert not torch.any(torch.isinf(self._command_buffer)), "Inf in command_buffer"
        
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        
        for k, v in obs_buf.items():
            assert not torch.any(torch.isnan(v)), f"NaN in obs_buf, key: {k}"
            assert not torch.any(torch.isinf(v)), f"Inf in obs_buf, key: {k}"
        #update_robot_phase_buffer(self)
       
        self._vel_vec = self.scene["robot"].data.root_com_lin_vel_w

        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # Log success rate for the environments being reset
        # Call the parent's reset method to handle standard resets and logging
        
        if len(env_ids) > 0:            
            #Has to be called before sample_command, so the command in the to-be-reset envs is used.
            command_magnitude_mean = self._command_buffer[env_ids][:, 1].mean().item()
            command_vec = convert_command_to_euclidean_vector(self._command_buffer[env_ids])
            
            takeoff_vel_vec = self._vel_vec[env_ids].clone()
            
            angle_error, magnitude_ratio_error = calculate_takeoff_errors(command_vec, takeoff_vel_vec)
            self._success_buf[env_ids] = (magnitude_ratio_error < self.cfg.takeoff_magnitude_ratio_error_threshold) | (angle_error < self.cfg.takeoff_angle_error_threshold)
            success_rate = torch.mean(self._success_buf[env_ids].float())
            
            #Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
            self._command_buffer[env_ids] = sample_command(self, env_ids)
            
            super()._reset_idx(env_ids)
            
            #Has to be run after super()._reset_idx, as _reset_idx resets extras["log"]
            self.extras["log"]["Episode_SuccessRate"] = success_rate.item()
            self.extras["log"]["takeoff_vec_magnitude_mean"] = takeoff_vel_vec[:, 1].mean().item()
            self.extras["log"]["cmd_vec_magnitude_mean"] = command_magnitude_mean
            self.extras["log"]["takeoff_vec_angle_error_mean"] = angle_error.mean().item()
            self.extras["log"]["takeoff_vec_magnitude_ratio_error_mean"] = magnitude_ratio_error.mean().item()
        
            # Reset tracking buffers for these environments
            self._success_buf[env_ids] = False
            self._vel_vec[env_ids] = torch.zeros_like(self._vel_vec[env_ids])
