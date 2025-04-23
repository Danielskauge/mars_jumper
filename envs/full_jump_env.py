from envs.full_jump_env_cfg import FullJumpEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from terms.phase import Phase
from terms.utils import update_jump_phase, sample_command, convert_command_to_euclidean_vector, log_phase_info, get_center_of_mass_lin_vel
import logging

logger = logging.getLogger(__name__)

class FullJumpEnv(ManagerBasedRLEnv):
    cfg: "FullJumpEnvCfg"
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
        
        #Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.liftoff_com_lin_vel = torch.zeros(self.num_envs, 3, device=self.device) 
        self.takeoff_success_rate = 0.0
        
        #Flight Success
        self.body_angle_quat_at_landing = torch.zeros(self.num_envs, 4, device=self.device)
        self.body_ang_vel_at_landing = torch.zeros(self.num_envs, 3, device=self.device)
        self.flight_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_success_rate = 0.0
        
        #Full Jump Success
        self.full_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.full_jump_success_rate = 0.0
        self.success_rate = 0.0

        
        #Jump Phase
        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        
        #Command Curriculum
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range

    def _takeoff_success(self, env_ids: Sequence[int]) -> dict | None:
        """
        Calculates the success rate of the takeoff phase for the environments being reset.

        Success is defined based on the similarity between the commanded takeoff velocity
        vector and the actual velocity vector at liftoff (transition from TAKEOFF to FLIGHT).
        The success rate reflects the proportion of successfully taken-off environments
        within the provided `env_ids` batch.

        Args:
            env_ids: Indices of the environments being reset.

        Returns:
            A dictionary containing the takeoff success rate and average takeoff error
            for the environments in `env_ids` that reached the flight phase, or None if
            no environments in the batch reached flight.
        """
        has_taken_off = (self.jump_phase[env_ids] == Phase.FLIGHT) | (self.jump_phase[env_ids] == Phase.LANDING)
        num_has_taken_off = torch.sum(has_taken_off).item()    # Denominator for success rate

        if num_has_taken_off > 0:
            cmd_vec = convert_command_to_euclidean_vector(self.command[env_ids])
            liftoff_vec = self.liftoff_com_lin_vel[env_ids]
        
            cos_angle = torch.sum(cmd_vec * liftoff_vec, dim=-1) / (torch.norm(cmd_vec, dim=-1) * torch.norm(liftoff_vec, dim=-1))
            angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
            relative_error = torch.norm(liftoff_vec - cmd_vec, dim=-1) / torch.norm(cmd_vec, dim=-1)
            magnitude_ratio_error = torch.norm(liftoff_vec, dim=-1) / torch.norm(cmd_vec, dim=-1) - 1
            
            magnitude_ok = torch.abs(magnitude_ratio_error) < self.cfg.takeoff_magnitude_ratio_error_threshold
            angle_ok = angle_error < self.cfg.takeoff_angle_error_threshold
            
            self.takeoff_success[env_ids] = magnitude_ok & angle_ok & has_taken_off
            
            num_successful_takeoffs = torch.sum(self.takeoff_success[env_ids]).item()
            num_reset_envs = len(env_ids)
            self.takeoff_success_rate = num_successful_takeoffs / num_reset_envs
            
            return {
                "takeoff_angle_error": angle_error.mean().item(),
                "takeoff_magnitude_ratio_error": magnitude_ratio_error.mean().item(),
                "takeoff_success_rate": self.takeoff_success_rate,
                "takeoff_relative_error": relative_error.mean().item(),
            }
        else:
            return {
                "takeoff_angle_error": 0.0,
                "takeoff_magnitude_ratio_error": 0.0,
                "takeoff_success_rate": 0.0,
                "takeoff_relative_error": 0.0,
            }
        
    def _calculate_flight_success(self, env_ids: Sequence[int]) -> dict | None:
        """
        Calculates the success rate of the flight phase for the environments being reset.

        Success is defined based on the robot's body orientation angle and angular velocity
        at the moment of landing (transition from FLIGHT to LANDING). The success rate
        reflects the proportion of successful landings among the environments in `env_ids`
        that have taken off, meaning transitioned from TAKEOFF to FLIGHT. (includes landing)
        
        Does not check angular velocity, for now.

        Args:
            env_ids: Indices of the environments being reset.

        Returns:
            A dictionary containing the flight success rate, average angle error, and
            average angular velocity error for the environments in `env_ids` that
            reached the landing phase, or None if no environments in the batch reached landing.
        """
        # 1. Identify which envs in the reset batch were in the flight phase
        num_taken_off = torch.sum((self.jump_phase[env_ids] == Phase.FLIGHT) | (self.jump_phase[env_ids] == Phase.LANDING)).item()

        if num_taken_off > 0:
            angle_error = self._abs_angle_error(self.body_angle_quat_at_landing[env_ids]) # Shape: (len(env_ids),)

            angle_ok = angle_error < self.cfg.flight_angle_error_threshold # Shape: (len(env_ids),), bool
            
            num_successful_landings = torch.sum(angle_ok).item() # Numerator

            ids_successful_landings = env_ids[angle_ok]
            self.flight_success[ids_successful_landings] = True
            self.flight_success_rate = num_successful_landings / num_taken_off
            
            return {
                "flight_success_rate": self.flight_success_rate,
                "flight_angle_error": torch.mean(angle_error).item(), # Avg error for those in flight
            }
        else:
            return {
                "flight_success_rate": 0.0,
                "flight_angle_error": 0.0,
            }

    def _abs_angle_error(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Returns the absolute angle error of the robot for the given envs
        """
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        return torch.abs(angle)
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        update_jump_phase(self)
        
        transitioned_to_flight = (self.jump_phase == Phase.FLIGHT) & (self.prev_jump_phase == Phase.TAKEOFF)
        if torch.any(transitioned_to_flight):
            self.liftoff_com_lin_vel[transitioned_to_flight] = get_center_of_mass_lin_vel(self)[transitioned_to_flight]
        
        transitioned_to_landing = (self.jump_phase == Phase.LANDING) & (self.prev_jump_phase == Phase.FLIGHT)
        if torch.any(transitioned_to_landing):
            self.body_angle_quat_at_landing[transitioned_to_landing] = self.robot.data.root_quat_w[transitioned_to_landing]
            self.body_ang_vel_at_landing[transitioned_to_landing] = self.robot.data.root_ang_vel_w[transitioned_to_landing]
        
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        log_phase_info(self, extras)
                
        self.env_steps_since_last_curriculum_update += 1

        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras
    

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            
            takeoff_log_data = self._takeoff_success(env_ids)
            flight_log_data = self._calculate_flight_success(env_ids)
            
            self.full_jump_success[env_ids] = self.takeoff_success[env_ids] & self.flight_success[env_ids]
            self.full_jump_success_rate = torch.mean(self.full_jump_success[env_ids].float()).item()
            
            #Exponential smoothing of success rate
            alpha = len(env_ids) / self.num_envs
            self.success_rate = alpha * self.full_jump_success_rate + (1 - alpha) * self.success_rate
            
            self.command[env_ids] = sample_command(self, env_ids) # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
    
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
                
            self.extras["log"].update({
                "full_jump_success_rate": self.full_jump_success_rate, # Log the rate for this batch
                "running_success_rate": self.success_rate, # Log the running success rate
            })

            self.liftoff_com_lin_vel[env_ids] = torch.zeros_like(self.liftoff_com_lin_vel[env_ids])
            self.body_angle_quat_at_landing[env_ids] = torch.zeros_like(self.body_angle_quat_at_landing[env_ids])
            self.body_ang_vel_at_landing[env_ids] = torch.zeros_like(self.body_ang_vel_at_landing[env_ids])
            
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
            
            self.flight_success[env_ids] = False
            self.takeoff_success[env_ids] = False
