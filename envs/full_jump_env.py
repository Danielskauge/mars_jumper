import numpy as np
from envs.full_jump_env_cfg import FullJumpEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terms.phase import Phase, update_jump_phase, log_phase_info
from terms.utils import sample_command, convert_command_to_euclidean_vector, get_center_of_mass_lin_vel, sum_contact_forces, convert_pitch_magnitude_to_height_length, convert_height_length_to_pitch_magnitude, get_center_of_mass_pos
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
        
        #Error Metrics
        self.start_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_height_achieved = torch.zeros(self.num_envs, device=self.device)
        self.length_error_at_landing = torch.zeros(self.num_envs, device=self.device)
        self.height_error_peak = torch.zeros(self.num_envs, device=self.device)
        self.length_error_at_termination = torch.zeros(self.num_envs, device=self.device)
        
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
        self.target_height = torch.zeros(self.num_envs, device=self.device)  # Target jump height in meters
        self.target_length = torch.zeros(self.num_envs, device=self.device)  # Target jump length in meters
        self.cmd_height_range = cfg.command_ranges.height_range
        self.cmd_length_range = cfg.command_ranges.length_range
        self.cmd_pitch_range = cfg.command_ranges.pitch_range  # Derived property
        self.cmd_magnitude_range = cfg.command_ranges.magnitude_range  # Derived property
        self.mean_episode_env_steps = 0

        # Store references to foot sensors (optional but can be convenient)
        self.body_contact_sensor: ContactSensor = self.scene["contact_sensor"]
        
        # Metrics Bucketing Initialization
        self.height_bucket_info = None
        self.length_bucket_info = None
        if self.cfg.metrics_bucketing is not None:
            mb_cfg = self.cfg.metrics_bucketing

            height_min, height_max = self.cfg.command_ranges.height_range
            length_min, length_max = self.cfg.command_ranges.length_range
            
            num_height_buckets = mb_cfg.num_height_buckets
            num_length_buckets = mb_cfg.num_length_buckets

            # If min and max are the same for a dimension, force num_buckets to 1 for that dimension
            if height_min == height_max:
                num_height_buckets = 1
            if length_min == length_max:
                num_length_buckets = 1

            if num_height_buckets <= 0: num_height_buckets = 1 # Ensure at least one bucket
            if num_length_buckets <= 0: num_length_buckets = 1

            height_width = (height_max - height_min) / num_height_buckets if num_height_buckets > 0 else 0
            if num_height_buckets > 0 and height_width == 0 and height_min != height_max: # Avoid issues if range is non-zero but too small for float precision with num_buckets
                 height_width = 1e-5 # nominal small width
            elif num_height_buckets > 0 and height_width == 0 and height_min == height_max:
                 height_width = 1.0 # If min=max, any value is in bucket 0, width is for calculation logic

            length_width = (length_max - length_min) / num_length_buckets if num_length_buckets > 0 else 0
            if num_length_buckets > 0 and length_width == 0 and length_min != length_max:
                length_width = 1e-5
            elif num_length_buckets > 0 and length_width == 0 and length_min == length_max:
                length_width = 1.0

            self.height_bucket_info = {
                "min": height_min, "max": height_max, "num": num_height_buckets,
                "width": height_width
            }
            self.length_bucket_info = {
                "min": length_min, "max": length_max, "num": num_length_buckets,
                "width": length_width
            }

            self.bucketed_takeoff_error_sum = torch.zeros(
                (num_height_buckets, num_length_buckets), dtype=torch.float32, device=self.device
            )
            self.bucketed_takeoff_error_count = torch.zeros(
                (num_height_buckets, num_length_buckets), dtype=torch.int64, device=self.device
            )
            self.bucketed_flight_angle_error_sum = torch.zeros(
                (num_height_buckets, num_length_buckets), dtype=torch.float32, device=self.device
            )
            self.bucketed_flight_angle_error_count = torch.zeros(
                (num_height_buckets, num_length_buckets), dtype=torch.int64, device=self.device
            )

        # Counter for periodic bucketing metric print
        self.env_steps_since_last_bucket_print = 0

    def _height_length_to_euclidean_vector(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Convert target height and length to euclidean velocity vector for given environments.
        
        NOTE: This method is kept for backward compatibility. Use _get_dynamic_takeoff_vector
        for position-aware calculations.
        
        Args:
            env_ids: Environment indices
            
        Returns:
            Tensor of shape (len(env_ids), 3) containing [x, y, z] velocity components
        """
        height = self.target_height[env_ids]
        length = self.target_length[env_ids]
        
        # Convert to pitch/magnitude first, then to euclidean vector
        pitch, magnitude = convert_height_length_to_pitch_magnitude(height, length, gravity=9.81)
        command_tensor = torch.stack([pitch, magnitude], dim=-1)
        return convert_command_to_euclidean_vector(command_tensor)

    def _get_dynamic_takeoff_vector(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Get dynamic takeoff vector based on current COM position and target trajectory.
        
        Args:
            env_ids: Environment indices
            
        Returns:
            Tensor of shape (len(env_ids), 3) containing [x, y, z] velocity components
        """
        from terms.utils import get_dynamic_takeoff_vector
        return get_dynamic_takeoff_vector(self, env_ids)

    def relative_takeoff_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        takeoff_vector = self._get_dynamic_takeoff_vector(env_ids)
        return torch.norm(self.max_takeoff_vel[env_ids] - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    
    def takeoff_angle_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        takeoff_vector = self._get_dynamic_takeoff_vector(env_ids)
        max_takeoff_vel_vec = self.max_takeoff_vel[env_ids]
        cos_angle = torch.sum(takeoff_vector * max_takeoff_vel_vec, dim=-1) / (torch.norm(takeoff_vector, dim=-1) * torch.norm(max_takeoff_vel_vec, dim=-1))
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
            
            takeoff_vector = self._get_dynamic_takeoff_vector(has_taken_off_ids)
            max_takeoff_vel_vec = self.max_takeoff_vel[has_taken_off_ids]
            
            magnitude_ratio_error = torch.norm(max_takeoff_vel_vec, dim=-1) / torch.norm(takeoff_vector, dim=-1) - 1
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
        
        # Track maximum height achieved during flight
        if torch.any(self.flight_mask):
            current_com_pos = get_center_of_mass_pos(self)
            current_heights = current_com_pos[:, 2] - self.start_com_pos[:, 2]  # Height above starting position
            flight_envs_mask = self.flight_mask
            update_height_mask = flight_envs_mask & (current_heights > self.max_height_achieved)
            self.max_height_achieved[update_height_mask] = current_heights[update_height_mask]
        
        if torch.any(self.flight_to_landing_mask):
            self.angle_error_at_landing[self.flight_to_landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.flight_to_landing_mask])
            self.body_ang_vel_at_landing[self.flight_to_landing_mask] = self.robot.data.root_ang_vel_w[self.flight_to_landing_mask]
            
            # Calculate error metrics at landing transition
            landing_env_ids = self.flight_to_landing_mask.nonzero(as_tuple=False).squeeze(-1)
            if landing_env_ids.numel() > 0:
                # Length error at landing
                current_com_pos = get_center_of_mass_pos(self)[landing_env_ids]
                horizontal_distance = torch.norm(current_com_pos[:, :2] - self.start_com_pos[landing_env_ids, :2], dim=-1)
                self.length_error_at_landing[landing_env_ids] = horizontal_distance - self.target_length[landing_env_ids]
                
                # Height error peak (difference between max height achieved and target)
                self.height_error_peak[landing_env_ids] = self.max_height_achieved[landing_env_ids] - self.target_height[landing_env_ids]
        
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

    def _get_bucket_indices(self, height_cmds: torch.Tensor, length_cmds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.height_bucket_info is None or self.length_bucket_info is None:
            # Should not be called if bucketing is not configured.
            # Default to a single bucket (index 0) if called unexpectedly.
            return torch.zeros_like(height_cmds, dtype=torch.long), torch.zeros_like(length_cmds, dtype=torch.long)

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        if h_info["num"] <= 0: # Should have been corrected to >= 1 in init
            height_indices = torch.zeros_like(height_cmds, dtype=torch.long)
        elif h_info["num"] == 1 or h_info["width"] == 0: # Single bucket or zero width (e.g. min=max)
            height_indices = torch.zeros_like(height_cmds, dtype=torch.long)
        else:
            height_indices = ((height_cmds - h_info["min"]) / h_info["width"]).long()
            height_indices = torch.clamp(height_indices, 0, h_info["num"] - 1)

        if l_info["num"] <= 0:
            length_indices = torch.zeros_like(length_cmds, dtype=torch.long)
        elif l_info["num"] == 1 or l_info["width"] == 0:
            length_indices = torch.zeros_like(length_cmds, dtype=torch.long)
        else:
            length_indices = ((length_cmds - l_info["min"]) / l_info["width"]).long()
            length_indices = torch.clamp(length_indices, 0, l_info["num"] - 1)
            
        return height_indices, length_indices

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            # -- Start Metrics Bucketing Collection --
            if self.cfg.metrics_bucketing is not None and self.height_bucket_info is not None and self.length_bucket_info is not None:
                # Get height and length commands directly
                height_cmds_ended_episode = self.target_height[env_ids].clone()
                length_cmds_ended_episode = self.target_length[env_ids].clone()
                
                takeoff_errors_ended_episode = self.takeoff_relative_error[env_ids].clone() # Metric for takeoff success
                flight_angle_errors_ended_episode = self.angle_error_at_landing[env_ids].clone() # Metric for flight success

                phase_at_termination = self.jump_phase[env_ids].clone() # Phase when episode ended for these env_ids

                # Takeoff error is valid if the robot took off (i.e., reached FLIGHT or LANDING phase before reset)
                valid_for_takeoff_metric_mask = (phase_at_termination == Phase.FLIGHT) | (phase_at_termination == Phase.LANDING)
                # Flight angle error at landing is valid if the robot actually landed (i.e., reached LANDING phase before reset)
                valid_for_flight_metric_mask = (phase_at_termination == Phase.LANDING)
                
                # Only proceed if there's at least one valid metric to record to avoid unnecessary computation
                if torch.any(valid_for_takeoff_metric_mask) or torch.any(valid_for_flight_metric_mask):
                    height_bucket_indices, length_bucket_indices = self._get_bucket_indices(
                        height_cmds_ended_episode, length_cmds_ended_episode
                    )
                    
                    alpha = len(env_ids) / self.num_envs
                    mean_episode_env_steps = torch.mean(self.episode_length_buf[env_ids])

                    self.mean_episode_env_steps = self.mean_episode_env_steps * (1 - alpha) + alpha * mean_episode_env_steps

                    num_length_buckets = self.length_bucket_info["num"]
                    # Flatten the 2D bucket indices to 1D for scatter_add_
                    # flat_indices = pitch_idx * num_magnitude_buckets + magnitude_idx
                    flat_indices = height_bucket_indices * num_length_buckets + length_bucket_indices

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
            
            # Calculate length error at termination for environments that terminate in landing phase
            terminating_landing_mask = torch.tensor([i in env_ids for i in range(self.num_envs)], device=self.device) & (self.jump_phase == Phase.LANDING)
            if torch.any(terminating_landing_mask):
                terminating_landing_ids = terminating_landing_mask.nonzero(as_tuple=False).squeeze(-1)
                current_com_pos = get_center_of_mass_pos(self)[terminating_landing_ids]
                horizontal_distance = torch.norm(current_com_pos[:, :2] - self.start_com_pos[terminating_landing_ids, :2], dim=-1)
                self.length_error_at_termination[terminating_landing_ids] = horizontal_distance - self.target_length[terminating_landing_ids]
            
            # Calculate error metric averages for logging
            # Initialize error_log_data with default values to ensure all keys are always present
            error_log_data = {
                "length_error_at_landing": 0.0,
                "height_error_peak": 0.0,
                "length_error_at_termination": 0.0,
            }
            
            if len(env_ids) > 0:
                # Only log for environments that actually reached landing phase
                landing_phase_mask = torch.tensor([i in env_ids for i in range(self.num_envs)], device=self.device) & ((self.jump_phase == Phase.LANDING) | (self.prev_jump_phase == Phase.LANDING))
                if torch.any(landing_phase_mask):
                    landing_ids = landing_phase_mask.nonzero(as_tuple=False).squeeze(-1)
                    
                    # Log average errors (filtering out zero/unset values for cleaner averages)
                    length_landing_errors = self.length_error_at_landing[landing_ids]
                    height_peak_errors = self.height_error_peak[landing_ids]
                    length_termination_errors = self.length_error_at_termination[landing_ids]
                    
                    # Only include non-zero values for averages (zero means metric wasn't calculated)
                    # Update the default values if we have valid data
                    if torch.any(length_landing_errors > 0):
                        error_log_data["length_error_at_landing"] = length_landing_errors[length_landing_errors > 0].mean().item()
                    if torch.any(height_peak_errors > 0):
                        error_log_data["height_error_peak"] = height_peak_errors[height_peak_errors > 0].mean().item()
                    if torch.any(length_termination_errors > 0):
                        error_log_data["length_error_at_termination"] = length_termination_errors[length_termination_errors > 0].mean().item()
            
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
            
            self.target_height[env_ids], self.target_length[env_ids] = sample_command(self, env_ids) # Has to be called before super()._reset_idx, as the command is needed in the events terms for state initialization, which are run before the command manager is reset
    
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            # Log the stored data
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            self.extras["log"].update(landing_log_data)
            
            self.extras["log"].update(error_log_data)
            
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
            
            # Reset error metrics and store starting position for new episodes
            current_com_pos = get_center_of_mass_pos(self)
            self.start_com_pos[env_ids] = current_com_pos[env_ids]
            self.max_height_achieved[env_ids] = 0.0
            self.length_error_at_landing[env_ids] = 0.0
            self.height_error_peak[env_ids] = 0.0
            self.length_error_at_termination[env_ids] = 0.0

    def _log_bucketed_takeoff_errors(self):
        if self.cfg.metrics_bucketing is None or self.height_bucket_info is None or self.length_bucket_info is None:
            return

        # Prevent division by zero, replace with NaN if count is 0
        counts = self.bucketed_takeoff_error_count.float()
        # Add a small epsilon to counts to avoid division by zero, then handle NaNs if sum was also 0
        avg_errors = self.bucketed_takeoff_error_sum / (counts + 1e-8) 
        avg_errors[counts == 0] = torch.nan # Explicitly set to NaN where counts are zero

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        # Clear previous bucketed logs to avoid stale data if some buckets are not hit in an interval
        # This is a simple way, alternatively, one could collect all keys and remove them.
        # For now, let's assume we overwrite or add new ones.
        # A more robust solution might involve clearing keys with a specific prefix.
        # For wandb, it's often fine to just log the new values; it will create new data points.

        for i in range(h_info["num"]):
            height_low = h_info["min"] + i * h_info["width"]
            height_high = h_info["min"] + (i + 1) * h_info["width"]
            height_range_str = f"height_{height_low:.2f}-{height_high:.2f}"
            if h_info["num"] == 1: # Special case for single height bucket
                height_range_str = f"height_{h_info['min']:.2f}-{h_info['max']:.2f}"

            for j in range(l_info["num"]):
                length_low = l_info["min"] + j * l_info["width"]
                length_high = l_info["min"] + (j + 1) * l_info["width"]
                length_range_str = f"length_{length_low:.2f}-{length_high:.2f}"
                if l_info["num"] == 1: # Special case for single length bucket
                    length_range_str = f"length_{l_info['min']:.2f}-{l_info['max']:.2f}"

                error_val = avg_errors[i, j].item()
                count_val = self.bucketed_takeoff_error_count[i,j].item()

                base_key = f"bucketed_takeoff/{height_range_str}_{length_range_str}"
                
                if torch.isnan(torch.tensor(error_val)):
                    # wandb might not handle NaN well depending on configuration,
                    # logging as 0 or skipping might be alternatives.
                    # For now, let's log it as a very distinct number or skip.
                    # Using a very high number to indicate N/A if direct NaN logging is problematic.
                    self.extras["log"][f"{base_key}_avg_error"] = -1.0 # Or float('nan') if wandb handles it
                else:
                    self.extras["log"][f"{base_key}_avg_error"] = error_val