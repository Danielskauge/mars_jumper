import numpy as np
from envs.full_jump_env_cfg import FullJumpEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terms.phase import Phase, update_jump_phase, log_phase_info
from terms.utils import sample_command, convert_command_to_euclidean_vector, get_center_of_mass_lin_vel, sum_contact_forces
import logging
import gymnasium as gym

logger = logging.getLogger(__name__)

class FullJumpEnv(ManagerBasedRLEnv):
    cfg: "FullJumpEnvCfg"
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
        self.contact_sensor: ContactSensor = self.scene["contact_sensor"]
        
        self.feet_body_idx = torch.tensor(self.robot.find_bodies(".*FOOT.*")[0])
        self.hips_body_idx = torch.tensor(self.robot.find_bodies(".*HIP.*")[0])
        self.thighs_body_idx = torch.tensor(self.robot.find_bodies(".*THIGH.*")[0])
        self.shanks_body_idx = torch.tensor(self.robot.find_bodies(".*SHANK.*")[0])
        self.bodies_except_feet_idx = torch.cat((self.hips_body_idx, self.thighs_body_idx, self.shanks_body_idx))
        
        self.hip_joint_idx = torch.tensor(self.robot.find_joints(".*HAA.*")[0])
        self.abduction_joint_idx = torch.tensor(self.robot.find_joints(".*HFE.*")[0])
        self.knee_joint_idx = torch.tensor(self.robot.find_joints(".*KFE.*")[0])
        
        self.num_feet = 4
                
        # --- REINSTATED LIMIT WRITING SECTION ---
        # Limits must be written after robot initialization to update internal buffers,
        # even if they match USD defaults, to ensure correct data state.
        # We then need to ensure the action_space uses these updated limits.
        
        # Initialize limits_tensor by cloning the default limits from the articulation data.
        # This ensures that any joints not explicitly overridden will retain their defaults
        # instead of being set to [0,0].
        # default_joint_pos_limits is typically (num_joints, 2) or (1, num_joints, 2) from physx,
        # so we ensure it's correctly shaped for (num_envs, num_joints, 2).
        default_limits_from_sim = self.robot.root_physx_view.get_dof_limits() # Get fresh from sim
        if default_limits_from_sim.ndim == 2: # (num_joints, 2)
            limits_tensor = default_limits_from_sim.unsqueeze(0).expand(self.num_envs, -1, -1).clone().to(self.device)
        elif default_limits_from_sim.ndim == 3: # (1, num_joints, 2)
             limits_tensor = default_limits_from_sim.expand(self.num_envs, -1, -1).clone().to(self.device)
        else: # Should not happen, but as a fallback
            logger.warning("Unexpected ndim for default_joint_pos_limits, re-initializing to zeros.")
            limits_tensor = torch.zeros((self.num_envs, self.robot.num_joints, 2), device=self.device)

        # Assuming _hip_flex_joint_idx, etc., are lists of indices for ALL envs if used directly
        # If they are per-env, this needs adjustment. For now, assume they are global joint indices.
        limits_tensor[:, self.hip_joint_idx, 0] = self.robot.cfg.hip_joint_limits[0]
        limits_tensor[:, self.hip_joint_idx, 1] = self.robot.cfg.hip_joint_limits[1]
        
        limits_tensor[:, self.abduction_joint_idx, 0] = self.robot.cfg.abduction_joint_limits[0]
        limits_tensor[:, self.abduction_joint_idx, 1] = self.robot.cfg.abduction_joint_limits[1]
        
        limits_tensor[:, self.knee_joint_idx, 0] = self.robot.cfg.knee_joint_limits[0]
        limits_tensor[:, self.knee_joint_idx, 1] = self.robot.cfg.knee_joint_limits[1]

        # self.robot.write_joint_limits_to_sim(limits=limits_tensor) # Deprecated
        self.robot.write_joint_position_limit_to_sim(limits=limits_tensor)

        # -- DIAGNOSTIC PRINT --
        # Get limits for environment 0 to check if they were set correctly in PhysX
        # Note: get_dof_limits() returns a tensor on CPU by default from physx view
        actual_sim_limits_env0 = self.robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
        logger.info("--------------------------------------------------------------------")
        logger.info(f"FullJumpEnv: Attempted to set joint limits. Verifying for env 0:")
        logger.info(f"FullJumpEnv: Joint Names: {self.robot.joint_names}")
        logger.info(f"FullJumpEnv: Limits Tensor (first env) written to sim: \\n{limits_tensor[0].cpu().numpy()}")
        logger.info(f"FullJumpEnv: Actual Limits in PhysX (env 0): \\n{actual_sim_limits_env0}")
        logger.info("--------------------------------------------------------------------")
        # --- END REINSTATED SECTION ---
        
        # --- REMOVE ACTION SPACE OVERRIDE ---
        # Reverting this change, as the scaling should be handled by 
        # the JointPositionToLimitsAction term, not by overriding the env's action space.
        # action_dim = self.action_manager.total_action_dim 
        # if action_dim != self.robot.num_joints:
        #     logger.warning(f"Action manager total_action_dim ({action_dim}) does not match robot.num_joints ({self.robot.num_joints}). Assuming action space corresponds to all robot joints for limit setting. This may be incorrect if using a subset of joints or other action terms.")
        # low_bounds = self.robot.data.joint_pos_limits[0, :action_dim, 0].cpu().numpy()
        # high_bounds = self.robot.data.joint_pos_limits[0, :action_dim, 1].cpu().numpy()
        # self.single_action_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        # self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        # --- END REMOVE ACTION SPACE OVERRIDE ---
        
        if not hasattr(self, 'prev_feet_contact_state'):
            self.prev_feet_contact_state = torch.zeros(self.num_envs, self.num_feet, dtype=torch.bool, device=self.device)

        #Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_takeoff_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_takeoff_vel_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.takeoff_relative_error = torch.zeros(self.num_envs, device=self.device)
        self.takeoff_success_rate = 0.0
        self.running_takeoff_success_rate = 0.0
        
        #Flight Success
        self.angle_error_at_landing = torch.zeros(self.num_envs, device=self.device)
        self.body_ang_vel_at_landing = torch.zeros(self.num_envs, 3, device=self.device)
        self.flight_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_success_rate = 0.0
        self.running_flight_success_rate = 0.0
        
        #Landing Success
        self.landing_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing_success_rate = 0.0
        self.running_landing_success_rate = 0.0
        
        #Full Jump Success
        self.full_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.full_jump_success_rate = 0.0
        self.success_rate = 0.0
        
        #Jump Phase
        if not hasattr(self, "jump_phase"):
            self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        self.takeoff_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.takeoff_to_flight_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_to_landing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        #Command Curriculum
        self.command = torch.zeros(self.num_envs, 2, device=self.device)
        self.cmd_pitch_range = cfg.command_ranges.initial_pitch_range
        self.cmd_magnitude_range = cfg.command_ranges.initial_magnitude_range

        # Store references to foot sensors (optional but can be convenient)
        self.body_contact_sensor: ContactSensor = self.scene["contact_sensor"]
        
    def relative_takeoff_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        cmd_vec = convert_command_to_euclidean_vector(self.command[env_ids])
        return torch.norm(self.max_takeoff_vel[env_ids] - cmd_vec, dim=-1) / torch.norm(cmd_vec, dim=-1)
    
    def takeoff_angle_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        cmd_vec = convert_command_to_euclidean_vector(self.command[env_ids])
        max_takeoff_vel_vec = self.max_takeoff_vel[env_ids]
        cos_angle = torch.sum(cmd_vec * max_takeoff_vel_vec, dim=-1) / (torch.norm(cmd_vec, dim=-1) * torch.norm(max_takeoff_vel_vec, dim=-1))
        angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
        return angle_error

    def _takeoff_success(self, env_ids: Sequence[int]) -> dict | None:
        """
        Success rate is calculated by successfull takeoffs divided by the number of reset environments.
        
        Other metrics are only calcualted for those that have taken off.

        Success is defined based on the similarity between the commanded takeoff velocity
        vector and the actual maximum velocity vector achieved during takeoff phase.
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
            has_taken_off_ids = env_ids[has_taken_off]
            
            relative_error = self.relative_takeoff_error(has_taken_off_ids)
            angle_error = self.takeoff_angle_error(has_taken_off_ids)
            
            cmd_vec = convert_command_to_euclidean_vector(self.command[has_taken_off_ids])
            max_takeoff_vel_vec = self.max_takeoff_vel[has_taken_off_ids]
            
            magnitude_ratio_error = torch.norm(max_takeoff_vel_vec, dim=-1) / torch.norm(cmd_vec, dim=-1) - 1
            magnitude_ok = torch.abs(magnitude_ratio_error) < self.cfg.takeoff_magnitude_ratio_error_threshold
            angle_ok = angle_error < self.cfg.takeoff_angle_error_threshold_rad
            
            self.takeoff_success[has_taken_off_ids] = magnitude_ok & angle_ok
            
            num_successful_takeoffs = torch.sum(self.takeoff_success[has_taken_off_ids]).item()
            num_reset_envs = len(env_ids)
            self.takeoff_success_rate = num_successful_takeoffs / num_reset_envs
            
            valid_angle_errors = angle_error[~torch.isnan(angle_error)]
            mean_angle_error = torch.mean(valid_angle_errors) if valid_angle_errors.numel() > 0 else 0.0
            
            return {
                #"taken_off_ratio": num_has_taken_off / num_reset_envs,
                "takeoff_angle_error": mean_angle_error,
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
            in_landing_phase = self.jump_phase[env_ids] == Phase.LANDING
            landed_env_ids = env_ids[in_landing_phase]
            
            angle_ok = self.angle_error_at_landing[landed_env_ids] < self.cfg.flight_angle_error_threshold # Shape: (len(landed_env_ids),), bool
            
            num_successful_landings = torch.sum(angle_ok).item() # Numerator

            ids_successful_landings = landed_env_ids[angle_ok]
            self.flight_success[ids_successful_landings] = True
            self.flight_success_rate = num_successful_landings / num_taken_off
            
            return {
                "flight_success_rate": self.flight_success_rate,
                "flight_angle_error_at_landing": torch.mean(self.angle_error_at_landing[landed_env_ids]).item(), # Avg error for those in flight
            }
        else:
            return {
                "flight_success_rate": 0.0,
                "flight_angle_error_at_landing": 0.0,
            }
            
    def _calculate_landing_success(self, env_ids: Sequence[int]) -> dict | None:
        num_timed_out_envs = torch.sum(self.termination_manager.get_term("time_out")[env_ids]).item()
        num_landing_envs = torch.sum(self.jump_phase[env_ids] == Phase.LANDING).item()
        self.landing_success[env_ids] = self.termination_manager.get_term("time_out")[env_ids]
        
        if num_landing_envs > 0:
            current_batch_landing_success_rate = num_timed_out_envs / num_landing_envs
            self.landing_success_rate = current_batch_landing_success_rate # Update attribute
            return {
                "landing_success_rate": self.landing_success_rate,
            }
        else:
            self.landing_success_rate = 0.0 # Update attribute
            return {
                "landing_success_rate": 0.0,
            }

    def _abs_angle_error(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Returns the absolute angle error of the robot for the given envs
        """
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        return torch.abs(angle)
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        update_jump_phase(self)
        log_phase_info(self, extras)
        
        if torch.any(self.takeoff_mask):
            current_vel_vec = get_center_of_mass_lin_vel(self) # Shape: (num_envs, 3)
            current_vel_magnitude = torch.norm(current_vel_vec, dim=-1) # Shape: (num_envs,)
            
            update_mask = self.takeoff_mask & (current_vel_magnitude > self.max_takeoff_vel_magnitude) # Shape: (num_envs,)
            
            self.max_takeoff_vel_magnitude[update_mask] = current_vel_magnitude[update_mask]
            self.max_takeoff_vel[update_mask] = current_vel_vec[update_mask]

        if torch.any(self.takeoff_to_flight_mask):
            self.takeoff_relative_error[self.takeoff_to_flight_mask] = self.relative_takeoff_error(self.takeoff_to_flight_mask)
        
        if torch.any(self.flight_to_landing_mask):
            self.angle_error_at_landing[self.flight_to_landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.flight_to_landing_mask])
            self.body_ang_vel_at_landing[self.flight_to_landing_mask] = self.robot.data.root_ang_vel_w[self.flight_to_landing_mask]
    
        self.extras["log"]["takeoff_contact_forces"] = sum_contact_forces(self, self.takeoff_mask).mean().item()
        self.extras["log"]["flight_contact_forces"] = sum_contact_forces(self, self.flight_mask).mean().item()
        self.extras["log"]["landing_contact_forces"] = sum_contact_forces(self, self.landing_mask).mean().item()
        
        if torch.any(self.flight_mask):
            flight_quat = self.robot.data.root_quat_w[self.flight_mask]
            flight_angle_error = self._abs_angle_error(flight_quat)
            mean_flight_angle_error = torch.mean(flight_angle_error).item()
            self.extras["log"]["flight_angle_error"] = mean_flight_angle_error
        else:
            self.extras["log"]["flight_angle_error"] = 0.0

        if self.cfg.curriculum is not None:
            self.env_steps_since_last_curriculum_update += 1

        # Update previous foot contact state for the next step
        if hasattr(self, 'prev_feet_contact_state'): # Check if initialized
            self.prev_feet_contact_state = self._get_current_feet_contact_state().clone()
        
        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras
    
    def _get_current_feet_contact_state(self) -> torch.Tensor:
        contact_sensor: ContactSensor = self.scene["contact_sensor"]
        feet_body_idx, _ = self.robot.find_bodies(".*FOOT.*") 
        forces = contact_sensor.data.net_forces_w[:, feet_body_idx, :]        
        contact_state = torch.norm(forces, dim=-1) > contact_sensor.cfg.force_threshold
        return contact_state

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            
            takeoff_log_data = self._takeoff_success(env_ids)
            flight_log_data = self._calculate_flight_success(env_ids)
            landing_log_data = self._calculate_landing_success(env_ids)
            
            alpha = len(env_ids) / self.num_envs

            # Update running success rates for individual phases
            self.running_takeoff_success_rate = alpha * self.takeoff_success_rate + \
                                               (1 - alpha) * self.running_takeoff_success_rate
            
            self.running_flight_success_rate = alpha * self.flight_success_rate + \
                                              (1 - alpha) * self.running_flight_success_rate
            
            self.running_landing_success_rate = alpha * self.landing_success_rate + \
                                               (1 - alpha) * self.running_landing_success_rate

            # Calculate full jump success for the current batch and its running average
            self.full_jump_success[env_ids] = self.takeoff_success[env_ids] & self.flight_success[env_ids] & self.landing_success[env_ids]
            self.full_jump_success_rate = torch.mean(self.full_jump_success[env_ids].float()).item()
            
            #Exponential smoothing of success rate
            self.success_rate = alpha * self.full_jump_success_rate + (1 - alpha) * self.success_rate
            
            self.command[env_ids] = sample_command(self, env_ids) # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
    
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            self.extras["log"].update(landing_log_data)
            
            self.extras["log"].update({
                "full_jump_success_rate": self.full_jump_success_rate, # Log the rate for this batch
                "running_success_rate": self.success_rate, # Log the running success rate
            })

            current_contacts_for_reset_envs = self._get_current_feet_contact_state()
            self.prev_feet_contact_state[env_ids] = current_contacts_for_reset_envs[env_ids]

            self.max_takeoff_vel[env_ids] = torch.zeros_like(self.max_takeoff_vel[env_ids])
            self.max_takeoff_vel_magnitude[env_ids] = torch.zeros_like(self.max_takeoff_vel_magnitude[env_ids])
            self.body_ang_vel_at_landing[env_ids] = torch.zeros_like(self.body_ang_vel_at_landing[env_ids])
            self.angle_error_at_landing[env_ids] = torch.zeros_like(self.angle_error_at_landing[env_ids])
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
            
            self.flight_success[env_ids] = False
            self.takeoff_success[env_ids] = False
            self.landing_success[env_ids] = False
            self.takeoff_mask[env_ids] = True
            self.flight_mask[env_ids] = False
            self.landing_mask[env_ids] = False
            self.takeoff_relative_error[env_ids] = 0.0