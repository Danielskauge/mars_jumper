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
        self.cmd_pitch_range = cfg.command_ranges.pitch_range
        self.cmd_magnitude_range = cfg.command_ranges.magnitude_range
        self.mean_episode_env_steps = 0

        # Store references to foot sensors (optional but can be convenient)
        self.body_contact_sensor: ContactSensor = self.scene["contact_sensor"]
        
        # Metrics Bucketing Initialization
        self.pitch_bucket_info = None
        self.magnitude_bucket_info = None
        if self.cfg.metrics_bucketing is not None:
            mb_cfg = self.cfg.metrics_bucketing

            pitch_min, pitch_max = self.cfg.command_ranges.pitch_range
            mag_min, mag_max = self.cfg.command_ranges.magnitude_range
            
            num_pitch_buckets = mb_cfg.num_pitch_buckets
            num_magnitude_buckets = mb_cfg.num_magnitude_buckets

            # If min and max are the same for a dimension, force num_buckets to 1 for that dimension
            if pitch_min == pitch_max:
                num_pitch_buckets = 1
            if mag_min == mag_max:
                num_magnitude_buckets = 1

            if num_pitch_buckets <= 0: num_pitch_buckets = 1 # Ensure at least one bucket
            if num_magnitude_buckets <= 0: num_magnitude_buckets = 1

            pitch_width = (pitch_max - pitch_min) / num_pitch_buckets if num_pitch_buckets > 0 else 0
            if num_pitch_buckets > 0 and pitch_width == 0 and pitch_min != pitch_max: # Avoid issues if range is non-zero but too small for float precision with num_buckets
                 pitch_width = 1e-5 # nominal small width
            elif num_pitch_buckets > 0 and pitch_width == 0 and pitch_min == pitch_max:
                 pitch_width = 1.0 # If min=max, any value is in bucket 0, width is for calculation logic

            magnitude_width = (mag_max - mag_min) / num_magnitude_buckets if num_magnitude_buckets > 0 else 0
            if num_magnitude_buckets > 0 and magnitude_width == 0 and mag_min != mag_max:
                magnitude_width = 1e-5
            elif num_magnitude_buckets > 0 and magnitude_width == 0 and mag_min == mag_max:
                magnitude_width = 1.0

            self.pitch_bucket_info = {
                "min": pitch_min, "max": pitch_max, "num": num_pitch_buckets,
                "width": pitch_width
            }
            self.magnitude_bucket_info = {
                "min": mag_min, "max": mag_max, "num": num_magnitude_buckets,
                "width": magnitude_width
            }

            self.bucketed_takeoff_error_sum = torch.zeros(
                (num_pitch_buckets, num_magnitude_buckets), dtype=torch.float32, device=self.device
            )
            self.bucketed_takeoff_error_count = torch.zeros(
                (num_pitch_buckets, num_magnitude_buckets), dtype=torch.int64, device=self.device
            )
            self.bucketed_flight_angle_error_sum = torch.zeros(
                (num_pitch_buckets, num_magnitude_buckets), dtype=torch.float32, device=self.device
            )
            self.bucketed_flight_angle_error_count = torch.zeros(
                (num_pitch_buckets, num_magnitude_buckets), dtype=torch.int64, device=self.device
            )

        # Counter for periodic bucketing metric print
        self.env_steps_since_last_bucket_print = 0

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
        # Get environments that are in landing phase
        landing_mask = self.jump_phase[env_ids] == Phase.LANDING
        landing_env_ids = env_ids[landing_mask]
        
        # Get environments that timed out while in landing phase
        timed_out_mask = self.termination_manager.get_term("time_out")[landing_env_ids]
        num_successful_landings = torch.sum(timed_out_mask).item()
        num_landing_envs = len(landing_env_ids)
        
        # Update success tracking
        self.landing_success[landing_env_ids] = timed_out_mask
        
        if num_landing_envs > 0:
            current_batch_landing_success_rate = num_successful_landings / num_landing_envs
            self.landing_success_rate = current_batch_landing_success_rate
            return {
                "landing_success_rate": self.landing_success_rate,
            }
        else:
            self.landing_success_rate = 0.0
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
        self._log_bucketed_takeoff_errors()

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
            self.steps_since_curriculum_update += 1

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

    def _get_bucket_indices(self, pitch_cmds: torch.Tensor, magnitude_cmds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pitch_bucket_info is None or self.magnitude_bucket_info is None:
            # Should not be called if bucketing is not configured.
            # Default to a single bucket (index 0) if called unexpectedly.
            return torch.zeros_like(pitch_cmds, dtype=torch.long), torch.zeros_like(magnitude_cmds, dtype=torch.long)

        p_info = self.pitch_bucket_info
        m_info = self.magnitude_bucket_info

        if p_info["num"] <= 0: # Should have been corrected to >= 1 in init
            pitch_indices = torch.zeros_like(pitch_cmds, dtype=torch.long)
        elif p_info["num"] == 1 or p_info["width"] == 0: # Single bucket or zero width (e.g. min=max)
            pitch_indices = torch.zeros_like(pitch_cmds, dtype=torch.long)
        else:
            pitch_indices = ((pitch_cmds - p_info["min"]) / p_info["width"]).long()
            pitch_indices = torch.clamp(pitch_indices, 0, p_info["num"] - 1)

        if m_info["num"] <= 0:
            magnitude_indices = torch.zeros_like(magnitude_cmds, dtype=torch.long)
        elif m_info["num"] == 1 or m_info["width"] == 0:
            magnitude_indices = torch.zeros_like(magnitude_cmds, dtype=torch.long)
        else:
            magnitude_indices = ((magnitude_cmds - m_info["min"]) / m_info["width"]).long()
            magnitude_indices = torch.clamp(magnitude_indices, 0, m_info["num"] - 1)
            
        return pitch_indices, magnitude_indices

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            # -- Start Metrics Bucketing Collection --
            if self.cfg.metrics_bucketing is not None and self.pitch_bucket_info is not None and self.magnitude_bucket_info is not None:
                cmds_ended_episode = self.command[env_ids].clone()
                takeoff_errors_ended_episode = self.takeoff_relative_error[env_ids].clone() # Metric for takeoff success
                flight_angle_errors_ended_episode = self.angle_error_at_landing[env_ids].clone() # Metric for flight success

                phase_at_termination = self.jump_phase[env_ids].clone() # Phase when episode ended for these env_ids

                # Takeoff error is valid if the robot took off (i.e., reached FLIGHT or LANDING phase before reset)
                valid_for_takeoff_metric_mask = (phase_at_termination == Phase.FLIGHT) | (phase_at_termination == Phase.LANDING)
                # Flight angle error at landing is valid if the robot actually landed (i.e., reached LANDING phase before reset)
                valid_for_flight_metric_mask = (phase_at_termination == Phase.LANDING)
                
                # Only proceed if there's at least one valid metric to record to avoid unnecessary computation
                if torch.any(valid_for_takeoff_metric_mask) or torch.any(valid_for_flight_metric_mask):
                    pitch_cmds_for_bucketing = cmds_ended_episode[:, 0]
                    magnitude_cmds_for_bucketing = cmds_ended_episode[:, 1]
                    
                    pitch_bucket_indices, magnitude_bucket_indices = self._get_bucket_indices(
                        pitch_cmds_for_bucketing, magnitude_cmds_for_bucketing
                    )
                    
                    alpha = len(env_ids) / self.num_envs
                    mean_episode_env_steps = torch.mean(self.episode_length_buf[env_ids])

                    self.mean_episode_env_steps = self.mean_episode_env_steps * (1 - alpha) + alpha * mean_episode_env_steps

                    num_mag_buckets = self.magnitude_bucket_info["num"]
                    # Flatten the 2D bucket indices to 1D for scatter_add_
                    # flat_indices = pitch_idx * num_magnitude_buckets + magnitude_idx
                    flat_indices = pitch_bucket_indices * num_mag_buckets + magnitude_bucket_indices

                    # Update takeoff error metrics
                    if torch.any(valid_for_takeoff_metric_mask):
                        # Select only the errors from envs that are valid for this metric
                        src_takeoff_errors = takeoff_errors_ended_episode[valid_for_takeoff_metric_mask]
                        # Select the corresponding flat bucket indices
                        indices_for_takeoff_update = flat_indices[valid_for_takeoff_metric_mask]
                        
                        # Reshape sum/count tensors to 1D for scatter_add_
                        self.bucketed_takeoff_error_sum.view(-1).scatter_add_(
                            0, indices_for_takeoff_update, src_takeoff_errors
                        )
                        self.bucketed_takeoff_error_count.view(-1).scatter_add_(
                            0, indices_for_takeoff_update, torch.ones_like(src_takeoff_errors, dtype=torch.int64)
                        )

                    # Update flight angle error metrics
                    if torch.any(valid_for_flight_metric_mask):
                        # Select only the errors from envs that are valid for this metric
                        src_flight_angle_errors = flight_angle_errors_ended_episode[valid_for_flight_metric_mask]
                        # Select the corresponding flat bucket indices
                        indices_for_flight_update = flat_indices[valid_for_flight_metric_mask]

                        self.bucketed_flight_angle_error_sum.view(-1).scatter_add_(
                            0, indices_for_flight_update, src_flight_angle_errors
                        )
                        self.bucketed_flight_angle_error_count.view(-1).scatter_add_(
                            0, indices_for_flight_update, torch.ones_like(src_flight_angle_errors, dtype=torch.int64)
                        )
            # -- End Metrics Bucketing Collection --

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

    def _log_bucketed_takeoff_errors(self):
        if self.cfg.metrics_bucketing is None or self.pitch_bucket_info is None or self.magnitude_bucket_info is None:
            return

        # Prevent division by zero, replace with NaN if count is 0
        counts = self.bucketed_takeoff_error_count.float()
        # Add a small epsilon to counts to avoid division by zero, then handle NaNs if sum was also 0
        avg_errors = self.bucketed_takeoff_error_sum / (counts + 1e-8) 
        avg_errors[counts == 0] = torch.nan # Explicitly set to NaN where counts are zero

        p_info = self.pitch_bucket_info
        m_info = self.magnitude_bucket_info

        # Clear previous bucketed logs to avoid stale data if some buckets are not hit in an interval
        # This is a simple way, alternatively, one could collect all keys and remove them.
        # For now, let's assume we overwrite or add new ones.
        # A more robust solution might involve clearing keys with a specific prefix.
        # For wandb, it's often fine to just log the new values; it will create new data points.

        for i in range(p_info["num"]):
            pitch_low = p_info["min"] + i * p_info["width"]
            pitch_high = p_info["min"] + (i + 1) * p_info["width"]
            pitch_range_str = f"pitch_{pitch_low:.2f}-{pitch_high:.2f}"
            if p_info["num"] == 1: # Special case for single pitch bucket
                pitch_range_str = f"pitch_{p_info['min']:.2f}-{p_info['max']:.2f}"

            for j in range(m_info["num"]):
                mag_low = m_info["min"] + j * m_info["width"]
                mag_high = m_info["min"] + (j + 1) * m_info["width"]
                mag_range_str = f"mag_{mag_low:.2f}-{mag_high:.2f}"
                if m_info["num"] == 1: # Special case for single magnitude bucket
                    mag_range_str = f"mag_{m_info['min']:.2f}-{m_info['max']:.2f}"

                error_val = avg_errors[i, j].item()
                count_val = self.bucketed_takeoff_error_count[i,j].item()

                base_key = f"bucketed_takeoff/{pitch_range_str}_{mag_range_str}"
                
                if torch.isnan(torch.tensor(error_val)):
                    # wandb might not handle NaN well depending on configuration,
                    # logging as 0 or skipping might be alternatives.
                    # For now, let's log it as a very distinct number or skip.
                    # Using a very high number to indicate N/A if direct NaN logging is problematic.
                    self.extras["log"][f"{base_key}_avg_error"] = -1.0 # Or float('nan') if wandb handles it
                else:
                    self.extras["log"][f"{base_key}_avg_error"] = error_val