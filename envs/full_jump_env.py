from enum import IntEnum
import numpy as np
from envs.full_jump_env_cfg import FullJumpEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from terms.observations import has_taken_off
from terms.utils import *
import logging
from typing import Dict, Tuple


logger = logging.getLogger(__name__)


class FullJumpEnv(ManagerBasedRLEnv):
    cfg: "FullJumpEnvCfg"
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
        self.contact_sensor: ContactSensor = self.scene["contact_sensor"]
        
        self.base_body_idx = torch.tensor(self.robot.find_bodies(".*base.*")[0])
        self.feet_body_idx = torch.tensor(self.robot.find_bodies(".*FOOT.*")[0])
        self.hips_body_idx = torch.tensor(self.robot.find_bodies(".*HIP.*")[0])
        self.thighs_body_idx = torch.tensor(self.robot.find_bodies(".*THIGH.*")[0])
        self.shanks_body_idx = torch.tensor(self.robot.find_bodies(".*SHANK.*")[0])
        self.bodies_except_feet_idx = torch.cat((self.hips_body_idx, self.thighs_body_idx, self.shanks_body_idx))
        
        self.hip_joint_idx = torch.tensor(self.robot.find_joints(".*HAA.*")[0])
        self.abduction_joint_idx = torch.tensor(self.robot.find_joints(".*HFE.*")[0])
        self.knee_joint_idx = torch.tensor(self.robot.find_joints(".*KFE.*")[0])
        
        # Set knee joint limits after robot initialization
        self._set_knee_joint_limits()
        #self._set_abduction_joint_limits()
        
        self.com_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.com_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.com_acc = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Initialize metrics system
        self._init_success_metrics()
        self._init_error_metrics()
        self._init_bucketing_system()
        
        #Jump Phase
        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        self.prev_jump_phase = torch.full_like(self.jump_phase, Phase.TAKEOFF)
        self.takeoff_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.takeoff_to_flight_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_to_landing_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Store velocity vectors at the moment of liftoff for accurate error calculation
        self.actual_vel_at_liftoff = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_vel_at_liftoff = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.target_height = torch.zeros(self.num_envs, device=self.device)  # Target jump height in meters
        self.target_length = torch.zeros(self.num_envs, device=self.device)  # Target jump length in meters
        
        self.cmd_height_range = self.cfg.command_ranges.height_range
        self.cmd_length_range = self.cfg.command_ranges.length_range
        self.dynamic_takeoff_vector = torch.zeros(self.num_envs, 3, device=self.device) #Will be updated while env is in takeoff phase, then hold its last value until reset
        #for curriculum
        self.mean_episode_env_steps = 0
        
        # Phase-specific reward accumulators
        self.takeoff_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.flight_reward_sum = torch.zeros(self.num_envs, device=self.device)
        self.landing_reward_sum = torch.zeros(self.num_envs, device=self.device)
        
        # Landing time tracker
        self.landing_time_tracker = torch.zeros(self.num_envs, device=self.device)
    
    def _set_knee_joint_limits(self):
        """Set knee joint position limits based on robot configuration."""
        
        # Get knee limits from robot config
        knee_lower_limit = self.robot.cfg.KNEE_LIMITS[0]  # 0 degrees
        knee_upper_limit = self.robot.cfg.KNEE_LIMITS[1]  # 175 degrees in radians
        
        # Create limits tensor: shape (num_envs, num_knee_joints, 2)
        limits_tensor = torch.zeros((self.num_envs, len(self.knee_joint_idx), 2), device=self.device)
        limits_tensor[:, :, 0] = knee_lower_limit   # Lower limit
        limits_tensor[:, :, 1] = knee_upper_limit   # Upper limit
        
        # Apply limits to simulation
        self.robot.write_joint_position_limit_to_sim(
            limits=limits_tensor,
            joint_ids=self.knee_joint_idx,
            warn_limit_violation=True
        )
        
    def _set_abduction_joint_limits(self):
        """Set abduction joint position limits based on robot configuration."""
        limits_tensor = torch.zeros((self.num_envs, len(self.abduction_joint_idx), 2), device=self.device)
        limits_tensor[:, :, 0] = self.robot.cfg.ABDUCTION_LIMITS[0]   # Lower limit
        limits_tensor[:, :, 1] = self.robot.cfg.ABDUCTION_LIMITS[1]   # Upper limit
        
        self.robot.write_joint_position_limit_to_sim(
            limits=limits_tensor,
            joint_ids=self.abduction_joint_idx,
            warn_limit_violation=True
        )
        
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        #Current state k-1
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action) #Must be after updating jump phase since rewards are based on active phase
        #Now state k
        
        print(f"com_vel_z_mean: {self.com_vel[:, 2].mean().item()}")
        print(f"com_acc_z_mean: {self.com_acc[:, 2].mean().item()}")
        print(f"com_pos_z_mean: {self.com_pos[:, 2].mean().item()}")
        print(f"dynamic_takeoff_vector_z_mean: {self.dynamic_takeoff_vector[:, 2].mean().item()}")
        
        # Check if any dynamic takeoff vectors are NaNs
        nan_vector_mask = torch.any(torch.isnan(self.dynamic_takeoff_vector), dim=1)
        if torch.any(nan_vector_mask):
            num_nan_vectors = torch.sum(nan_vector_mask).item()
            print(f"CRITICAL WARNING: {num_nan_vectors}/{self.num_envs} dynamic takeoff vectors are NaN!")
            
            nan_vector_env_ids = nan_vector_mask.nonzero(as_tuple=True)[0]
            if len(nan_vector_env_ids) > 0:
                first_nan_env_id = nan_vector_env_ids[0]
                print(f"Debug info for first NaN env ({first_nan_env_id}):")
                print(f"  Target Height: {self.target_height[first_nan_env_id].item()}")
                print(f"  Target Length: {self.target_length[first_nan_env_id].item()}")
                print(f"  COM Pos: {self.com_pos[first_nan_env_id]}")
                # For more detailed debugging, you might need to inspect intermediate values
                # within get_dynamic_takeoff_vector or its sub-functions for this specific environment.
                # Example:
                # _, _, com_pos_debug = self.target_height[first_nan_env_id], self.target_length[first_nan_env_id], self.com_pos[first_nan_env_id]
                # pitch_debug, magnitude_debug = convert_height_length_to_pitch_magnitude_from_position(
                #    self.target_height[first_nan_env_id].unsqueeze(0), 
                #    self.target_length[first_nan_env_id].unsqueeze(0), 
                #    self.com_pos[first_nan_env_id].unsqueeze(0), 
                #    gravity=9.81
                # )
                # print(f"  Calculated Pitch: {pitch_debug.item()}, Magnitude: {magnitude_debug.item()}")

        # Check if any dynamic takeoff vectors are zeros
        zero_vectors_mask = torch.all(self.dynamic_takeoff_vector == 0, dim=1)
        if torch.any(zero_vectors_mask):
            num_zero_vectors = torch.sum(zero_vectors_mask).item()
            print(f"WARNING: {num_zero_vectors}/{self.num_envs} dynamic takeoff vectors are zeros")
            
            # Optional: Print additional debug info about environments with zero vectors
            zero_vector_env_ids = zero_vectors_mask.nonzero(as_tuple=True)[0]
            if len(zero_vector_env_ids) > 0:
                print(f"Example env with zero vector - target height: {self.target_height[zero_vector_env_ids[0]].item()}, "
                      f"target length: {self.target_length[zero_vector_env_ids[0]].item()}, "
                      f"com pos: {self.com_pos[zero_vector_env_ids[0]]}")
        
        # Accumulate phase-specific rewards
        # These masks represent the phase *before* update_jump_phase() is called for the current step k
        # So, reward_buf[self.takeoff_mask] are rewards generated while in takeoff phase
        self.takeoff_reward_sum[self.takeoff_mask] += reward_buf[self.takeoff_mask]
        self.flight_reward_sum[self.flight_mask] += reward_buf[self.flight_mask]
        self.landing_reward_sum[self.landing_mask] += reward_buf[self.landing_mask]

        current_height = self.com_pos[:, 2]
        new_max_height_mask = (current_height > self.max_height_achieved)
        self.max_height_achieved[new_max_height_mask] = current_height[new_max_height_mask]
            
        self.update_jump_phase()
        self.log_phase_info()
        
        # Update bucketed takeoff errors via metrics system
        bucketed_logs = self.log_bucketed_errors()
        self.extras["log"].update(bucketed_logs)


        # Capture velocity vectors at the moment of takeoff-to-flight transition
        if torch.any(self.takeoff_to_flight_mask):
            takeoff_to_flight_env_ids = self.takeoff_to_flight_mask.nonzero(as_tuple=False).squeeze(-1)
            self.actual_vel_at_liftoff[takeoff_to_flight_env_ids] = self.com_vel[takeoff_to_flight_env_ids]
            self.target_vel_at_liftoff[takeoff_to_flight_env_ids] = self.dynamic_takeoff_vector[takeoff_to_flight_env_ids]
            self.takeoff_relative_error[takeoff_to_flight_env_ids] = self.calculate_takeoff_relative_error(takeoff_to_flight_env_ids)
        
        # Log COM state at takeoff-to-flight transition
        if torch.any(self.takeoff_to_flight_mask):
            self.extras["log"]["takeoff_to_flight_com_height"] = self.com_pos[self.takeoff_to_flight_mask, 2].mean().item()
            self.extras["log"]["takeoff_to_flight_com_z_vel"] = self.com_vel[self.takeoff_to_flight_mask, 2].mean().item()
            self.extras["log"]["takeoff_to_flight_com_z_accel"] = self.com_acc[self.takeoff_to_flight_mask, 2].mean().item()
        else:
            self.extras["log"]["takeoff_to_flight_com_height"] = float('nan')
            self.extras["log"]["takeoff_to_flight_com_z_vel"] = float('nan')
            self.extras["log"]["takeoff_to_flight_com_z_accel"] = float('nan')
        
        #Track landing errors (attitude at transition to landing is used for flight success criteria)
        if torch.any(self.flight_to_landing_mask):
            self.angle_error_at_landing[self.flight_to_landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.flight_to_landing_mask])
        
        # Log COM state at flight-to-landing transition
        if torch.any(self.flight_to_landing_mask):
            self.extras["log"]["flight_to_landing_com_height"] = self.com_pos[self.flight_to_landing_mask, 2].mean().item()
            self.extras["log"]["flight_to_landing_com_z_vel"] = self.com_vel[self.flight_to_landing_mask, 2].mean().item()
            self.extras["log"]["flight_to_landing_com_z_accel"] = self.com_acc[self.flight_to_landing_mask, 2].mean().item()
        else:
            self.extras["log"]["flight_to_landing_com_height"] = float('nan')
            self.extras["log"]["flight_to_landing_com_z_vel"] = float('nan')
            self.extras["log"]["flight_to_landing_com_z_accel"] = float('nan')
        
        # Track feet height and attitude during landing phase
        if torch.any(self.landing_mask):
            feet_height = self.robot.data.body_pos_w[self.landing_mask][:, self.feet_body_idx, 2]
            self.feet_height_at_landing[self.landing_mask] = torch.mean(feet_height, dim=-1)
            self.attitude_error_at_landing[self.landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.landing_mask])
            landing_com_vel_norm = torch.norm(self.com_vel[self.landing_mask], dim=-1)
            self.extras["log"]["landing_com_vel_norm"] = landing_com_vel_norm.mean().item()
            self.landing_time_tracker[self.landing_mask] += self.step_dt

        else:
            self.extras["log"]["landing_com_vel_norm"] = float('nan')

        current_sum_contact_forces = self.sum_contact_forces() # Avoid rebinding self.sum_contact_forces
        self.extras["log"]["takeoff_contact_forces"] = current_sum_contact_forces[self.takeoff_mask].mean().item() if torch.any(self.takeoff_mask) else float('nan')
        self.extras["log"]["flight_contact_forces"] = current_sum_contact_forces[self.flight_mask].mean().item() if torch.any(self.flight_mask) else float('nan')
        self.extras["log"]["landing_contact_forces"] = current_sum_contact_forces[self.landing_mask].mean().item() if torch.any(self.landing_mask) else float('nan')
        
        if torch.any(self.flight_mask):
            flight_quat = self.robot.data.root_quat_w[self.flight_mask]
            flight_angle_error = self._abs_angle_error(flight_quat)
            mean_flight_angle_error = torch.nanmean(flight_angle_error).item() if flight_angle_error.numel() > 0 else float('nan')
            self.extras["log"]["flight_angle_error"] = mean_flight_angle_error
        else:
            self.extras["log"]["flight_angle_error"] = float('nan')

        if self.cfg.curriculum is not None:
            self.steps_since_curriculum_update += 1
            
        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras
    
    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            # Convert env_ids to a tensor if it's a list/sequence for consistent indexing
            landing_timeout_mask = self.termination_manager.get_term("landing_timeout")[env_ids]
            if torch.any(landing_timeout_mask):
                landing_timeout_env_ids = env_ids[landing_timeout_mask]
                current_com_x = self.com_pos[landing_timeout_env_ids, 0]
                self.length_error_at_landing[landing_timeout_env_ids] = current_com_x - self.target_length[landing_timeout_env_ids]


            # Update bucketed metrics before reset
            self.update_bucketed_metrics(env_ids, self.jump_phase) # Removed redundant self

            # Calculate success metrics
            takeoff_log_data = self.calculate_takeoff_success(env_ids) # Removed redundant self
            flight_log_data = self.calculate_flight_success(env_ids)
            landing_log_data = self.calculate_landing_success(env_ids)
            
            error_log_data = self.calculate_error_metrics(env_ids) # Removed redundant self
            
            # Update running success rates (uses batch rates calculated above)
            # self.update_running_success_rates(env_ids)
            
            # --- Log and prepare phase-specific reward sums for reset envs ---
            # Clone sums for envs about to be reset
            takeoff_rewards_to_log = self.takeoff_reward_sum[env_ids].clone()
            flight_rewards_to_log = self.flight_reward_sum[env_ids].clone()
            landing_rewards_to_log = self.landing_reward_sum[env_ids].clone()
            
            # Clone landing times for envs about to be reset
            landing_times_to_log = self.landing_time_tracker[env_ids].clone()

            # Sample new commands before calling super()._reset_idx
            self.sample_command(env_ids)   
            
            #self.update_dynamic_takeoff_vector()

            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            self.extras["log"].update(landing_log_data) # Log landing metrics
            self.extras["log"].update(error_log_data)
            
            self.extras["log"].update({
                # "full_jump_success_rate": self.full_jump_success_rate,
                "reward_sum/takeoff_phase_sum": torch.mean(takeoff_rewards_to_log).item() if len(takeoff_rewards_to_log) > 0 else float('nan'),
                "reward_sum/flight_phase_sum": torch.mean(flight_rewards_to_log).item() if len(flight_rewards_to_log) > 0 else float('nan'),
                "reward_sum/landing_phase_sum": torch.mean(landing_rewards_to_log).item() if len(landing_rewards_to_log) > 0 else float('nan'),
                "landing_time_seconds": torch.mean(landing_times_to_log).item() if len(landing_times_to_log) > 0 else float('nan'),
            })

            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
            
            self.takeoff_mask[env_ids] = True
            self.flight_mask[env_ids] = False
            self.landing_mask[env_ids] = False
            
            self.actual_vel_at_liftoff[env_ids] = float('nan')
            self.target_vel_at_liftoff[env_ids] = float('nan')
            self.angle_error_at_landing[env_ids] = float('nan')
            self.takeoff_relative_error[env_ids] = float('nan')
            self.max_height_achieved[env_ids] = 0 # Reset to 0.0 not COM height
            self.length_error_at_landing[env_ids] = float('nan')
            self.height_error_peak[env_ids] = float('nan')
            self.feet_height_at_landing[env_ids] = float('nan')
            self.attitude_error_at_landing[env_ids] = float('nan')
            self.root_height_at_timeout[env_ids] = float('nan')

            self.flight_success[env_ids] = False
            self.takeoff_success[env_ids] = False
            self.landing_success[env_ids] = False
            
            # Reset phase-specific reward sums for these envs
            self.takeoff_reward_sum[env_ids] = 0.0
            self.flight_reward_sum[env_ids] = 0.0
            self.landing_reward_sum[env_ids] = 0.0
            
            # Reset landing time tracker
            self.landing_time_tracker[env_ids] = 0.0
             
    def _init_success_metrics(self):
        """Initialize success tracking tensors."""
        # Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.takeoff_relative_error = torch.full((self.num_envs,), float('nan'), device=self.device)
        # self.takeoff_success_rate = 0.0
        # self.running_takeoff_success_rate = 0.0
        
        # Flight Success
        self.angle_error_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.flight_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # self.flight_success_rate = 0.0
        # self.running_flight_success_rate = 0.0
        
        # Landing Success
        self.landing_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # self.landing_success_rate = 0.0
        # self.running_landing_success_rate = 0.0
        
        # Full Jump Success
        self.full_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # self.full_jump_success_rate = 0.0
        # self.success_rate = 0.0

    def _init_error_metrics(self):
        """Initialize error tracking tensors."""
        self.max_height_achieved = torch.zeros(self.num_envs, device=self.device)
        
        # Use NaN as sentinel value to indicate "not calculated yet" instead of 0.0
        self.length_error_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.feet_height_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.attitude_error_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.root_height_at_timeout = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.height_error_peak = torch.full((self.num_envs,), float('nan'), device=self.device)

    def _init_bucketing_system(self):
        """Initialize metrics bucketing system if configured."""
        self.height_bucket_info = None
        self.length_bucket_info = None
        
        if self.cfg.metrics_bucketing is not None:
            mb_cfg = self.cfg.metrics_bucketing

            height_min, height_max = self.cmd_height_range
            length_min, length_max = self.cmd_length_range
            
            num_height_buckets = mb_cfg.num_height_buckets
            num_length_buckets = mb_cfg.num_length_buckets

            # If min and max are the same for a dimension, force num_buckets to 1 for that dimension
            if height_min == height_max:
                num_height_buckets = 1
            if length_min == length_max:
                num_length_buckets = 1

            if num_height_buckets <= 0: 
                num_height_buckets = 1
            if num_length_buckets <= 0: 
                num_length_buckets = 1

            height_width = (height_max - height_min) / num_height_buckets if num_height_buckets > 0 else 0
            if num_height_buckets > 0 and height_width == 0 and height_min != height_max:
                height_width = 1e-5  # nominal small width
            elif num_height_buckets > 0 and height_width == 0 and height_min == height_max:
                height_width = 1.0  # If min=max, any value is in bucket 0

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

    def calculate_takeoff_success(self, env_ids: Sequence[int]) -> Dict[str, float]:
        has_taken_off_mask = self.flight_mask[env_ids] | self.landing_mask[env_ids]
        num_has_taken_off = torch.sum(has_taken_off_mask).item()

        if num_has_taken_off == 0: 
            return {
                "takeoff_angle_error": float('nan'),
                "takeoff_magnitude_ratio_error": float('nan'),
                #"takeoff_success_rate": self.running_takeoff_success_rate,
                "takeoff_relative_error": float('nan'),
                "takeoff_abs_pitch_error": float('nan'),
                "takeoff_signed_pitch_error": float('nan'),
            }
        
        has_taken_off_ids = env_ids[has_taken_off_mask]
        
        relative_error = self.calculate_takeoff_relative_error(has_taken_off_ids) # env -> self
        angle_error = self.calculate_takeoff_angle_error(has_taken_off_ids) # env -> self
        abs_pitch_error, signed_pitch_error = self.calculate_takeoff_pitch_error(has_taken_off_ids)
        
        target_vector = self.target_vel_at_liftoff[has_taken_off_ids]
        actual_vector = self.actual_vel_at_liftoff[has_taken_off_ids]
        
        target_magnitude = torch.norm(target_vector, dim=-1)
        actual_magnitude = torch.norm(actual_vector, dim=-1)
        magnitude_ratio_error = actual_magnitude / (target_magnitude + 1e-8) - 1
        magnitude_ok = torch.abs(magnitude_ratio_error) < self.cfg.takeoff_magnitude_ratio_error_threshold
        angle_ok = angle_error < self.cfg.takeoff_angle_error_threshold_rad
        
        self.takeoff_success[has_taken_off_ids] = magnitude_ok & angle_ok
        
        # num_successful_takeoffs = torch.sum(self.takeoff_success[has_taken_off_ids]).item()
        # num_reset_envs = len(env_ids)
        # self.takeoff_success_rate = num_successful_takeoffs / num_reset_envs
        
        return {
            "takeoff_angle_error": torch.nanmean(angle_error).item(),
            "takeoff_magnitude_ratio_error": torch.nanmean(magnitude_ratio_error).item(),
            # "takeoff_success_rate": self.running_takeoff_success_rate,
            "takeoff_relative_error": torch.nanmean(relative_error).item(),
            "takeoff_abs_pitch_error": torch.nanmean(abs_pitch_error).item(),
            "takeoff_signed_pitch_error": torch.nanmean(signed_pitch_error).item(),
        }

    def calculate_flight_success(self, env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate flight success metrics for given environments."""
        taken_off_mask = self.flight_mask[env_ids] | self.landing_mask[env_ids]
        num_taken_off = torch.sum(taken_off_mask).item()
        
        if num_taken_off == 0:
            return {
                # "flight_success_rate": self.running_flight_success_rate,
                "flight_angle_error_at_landing": float('nan'),
            }

        in_landing_phase = self.jump_phase[env_ids] == Phase.LANDING
        landed_env_ids = env_ids[in_landing_phase]
        
        angle_ok = self.angle_error_at_landing[landed_env_ids] < self.cfg.flight_angle_error_threshold
        
        num_successful_landings = torch.sum(angle_ok).item()

        ids_successful_landings = landed_env_ids[angle_ok]
        self.flight_success[ids_successful_landings] = True
        # self.flight_success_rate = num_successful_landings / num_taken_off
        
        angle_errors_at_landing = self.angle_error_at_landing[landed_env_ids]
        mean_angle_error = torch.nanmean(angle_errors_at_landing).item() if angle_errors_at_landing.numel() > 0 else float('nan')
        
        return {
            # "flight_success_rate": self.running_flight_success_rate,
            "flight_angle_error_at_landing": mean_angle_error,
        }

    def calculate_landing_success(self, env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate landing success metrics for given environments.
        Landing success is defined as triggering the 'landing_timeout' termination condition.
        """
        landing_mask_for_reset_envs = self.landing_mask[env_ids] # Env IDs among those being reset, that are in landing phase
        landing_env_ids_in_reset_batch = env_ids[landing_mask_for_reset_envs]
        
        if len(landing_env_ids_in_reset_batch) == 0:
            # self.landing_success_rate = 0.0 # Batch rate is 0 if no envs were in landing phase at reset
            # self.running_landing_success_rate remains unchanged if alpha is 0, or smoothed otherwise
            pass
        else:   
            landing_timeout_mask = self.termination_manager.get_term("landing_timeout")[landing_env_ids_in_reset_batch]
            
            successful_landing_global_ids = landing_env_ids_in_reset_batch[landing_timeout_mask]
            self.landing_success[successful_landing_global_ids] = True # Mark True for these specific IDs

            failed_landing_global_ids = landing_env_ids_in_reset_batch[~landing_timeout_mask]
            self.landing_success[failed_landing_global_ids] = False
            
            # self.landing_success_rate = torch.mean(landing_timeout_mask.float()).item()
            
        return {} # {"landing_success_rate": self.running_landing_success_rate} # Log the running average

    def calculate_error_metrics(self, reset_env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate error metrics, this is called when reset_idx is called by the environment."""
        error_log_data = {
            "length_error_at_landing": float('nan'),
            "height_error_peak": float('nan'), 
            "abs_length_error_at_landing": float('nan'),
            "abs_height_error_peak": float('nan'),
            "feet_height_at_landing": float('nan'),
            "attitude_error_at_landing": float('nan'),
            "root_height_at_timeout": float('nan'),
        }
        
        if len(reset_env_ids) > 0:
            height_error_at_peak = self.max_height_achieved[reset_env_ids] - self.target_height[reset_env_ids] # env -> self
            error_log_data["height_error_peak"] = height_error_at_peak.mean().item()
            error_log_data["abs_height_error_peak"] = torch.abs(height_error_at_peak).mean().item()

            landing_mask = self.jump_phase[reset_env_ids] == Phase.LANDING
            if torch.any(landing_mask):
                landing_env_ids = reset_env_ids[landing_mask]  # Get actual environment indices
                
                length_landing_errors = self.length_error_at_landing[landing_env_ids] # Error at TRANSITION to landing
                error_log_data["length_error_at_landing"] = torch.nanmean(length_landing_errors).item()
                error_log_data["abs_length_error_at_landing"] = torch.nanmean(torch.abs(length_landing_errors)).item()

                feet_heights_landed = self.feet_height_at_landing[landing_env_ids] # During landing phase
                error_log_data["feet_height_at_landing"] = torch.nanmean(feet_heights_landed).item()
                
                attitude_errors_land = self.attitude_error_at_landing[landing_env_ids] # During landing phase
                error_log_data["attitude_error_at_landing"] = torch.nanmean(attitude_errors_land).item()

                root_heights_land_term = self.root_height_at_timeout[landing_env_ids] # New, at TERMINATION
                error_log_data["root_height_at_timeout"] = torch.nanmean(root_heights_land_term).item()
            
        return error_log_data

    def update_running_success_rates(self, env_ids: Sequence[int]):
        """Update exponentially smoothed running success rates."""
 
        # alpha = len(env_ids) / self.num_envs

        # self.running_takeoff_success_rate = alpha * self.takeoff_success_rate + \
        #                                    (1 - alpha) * self.running_takeoff_success_rate
        
        # self.running_flight_success_rate = alpha * self.flight_success_rate + \
        #                                   (1 - alpha) * self.running_flight_success_rate
        
        # self.running_landing_success_rate = alpha * self.landing_success_rate + \
        #                                    (1 - alpha) * self.running_landing_success_rate

        # Calculate full jump success for the current batch and its running average
        if len(env_ids) > 0: # ensure env_ids is not empty before indexing
            self.full_jump_success[env_ids] = self.takeoff_success[env_ids] & \
                                            self.flight_success[env_ids] & \
                                            self.landing_success[env_ids]
            # self.full_jump_success_rate = torch.mean(self.full_jump_success[env_ids].float()).item()
            
        # else:
        #     self.full_jump_success_rate = 0.0 # Or handle as appropriate if env_ids can be empty
        
        # Exponential smoothing of success rate
        # self.success_rate = alpha * self.full_jump_success_rate + (1 - alpha) * self.success_rate

    def sample_command(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        self.target_height[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cmd_height_range)
        self.target_length[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cmd_length_range)
        
    def update_dynamic_takeoff_vector(self) -> torch.Tensor:
        """Calculate the required takeoff velocity vector based on current COM position and target trajectory.
        
        This function continuously updates the required velocity vector based on:
        - The robot's current center of mass position
        - The target height and length trajectory 
        - Physics calculations for projectile motion

        Returns:
            Tensor of shape (num_envs or len(env_ids), 3) containing [x, y, z] velocity components
        """
   

        gravity = 9.81
        delta_height = self.target_height[self.takeoff_mask] - self.com_pos[self.takeoff_mask, 2]
        delta_length = self.target_length[self.takeoff_mask] - self.com_pos[self.takeoff_mask, 0]
        
        pitch = torch.atan(delta_length / (4 * delta_height))
        
        # Calculate magnitude using height-based formula
        # magnitude = sqrt(2*gravity*height / cos²(pitch))
        magnitude = torch.sqrt(2 * gravity * delta_height / torch.cos(pitch)**2)
        
        vector = torch.zeros((pitch.shape[0], 3), device=pitch.device)
        
        vector[:, 0] = magnitude * torch.sin(pitch)  # x_dot (horizontal)
        vector[:, 2] = magnitude * torch.cos(pitch)  # z_dot (vertical)
        # y_dot remains zero (no lateral movement)
                
        self.dynamic_takeoff_vector[self.takeoff_mask] = vector

    def _abs_angle_error(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Returns the absolute angle error of the robot for the given envs
        """
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        return torch.abs(angle)
    
    def get_com_vel(self) -> torch.Tensor:
        """
        Returns the center of mass linear velocity of the robot.
        Returns a tensor of shape (num_envs, 3)
        """
        total_mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(-1).to(self.device) # Shape: (num_envs, 1)
        masses = self.robot.data.default_mass.unsqueeze(-1).to(self.device) # Shape: (num_envs, num_bodies, 1)
        weighed_lin_vels = self.robot.data.body_com_lin_vel_w * masses # Shape: (num_envs, num_bodies, 3)
        com_lin_vel = torch.sum(weighed_lin_vels, dim=1) / total_mass # Shape: (num_envs, 3)
        return com_lin_vel

    def get_com_acc(self) -> torch.Tensor:
        """
        Returns the center of mass linear acceleration of the robot.
        Returns a tensor of shape (num_envs, 3)
        """
        total_mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(-1).to(self.device) # Shape: (num_envs, 1)
        masses = self.robot.data.default_mass.unsqueeze(-1).to(self.device) # Shape: (num_envs, num_bodies, 1)
        weighed_lin_accels = self.robot.data.body_lin_acc_w * masses # Shape: (num_envs, num_bodies, 3)
        com_lin_accel = torch.sum(weighed_lin_accels, dim=1) / total_mass # Shape: (num_envs, 3)
        return com_lin_accel

    def get_com_pos(self) -> torch.Tensor:
        """
        Returns the center of mass position of the robot in its local environment frame.
        Returns a tensor of shape (num_envs, 3)
        """
        total_mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(-1).to(self.device) # Shape: (num_envs, 1)
        masses = self.robot.data.default_mass.unsqueeze(-1).to(self.device) # Shape: (num_envs, num_bodies, 1)
        weighed_positions = self.robot.data.body_com_pos_w * masses # Shape: (num_envs, num_bodies, 3)
        com_pos_w = torch.sum(weighed_positions, dim=1) / total_mass # Shape: (num_envs, 3)
        env_origins_w = self.scene.env_origins
        com_pos_e = com_pos_w - env_origins_w
        return com_pos_e
    
    def sum_contact_forces(self) -> torch.Tensor:
        """Sum the L2 norm of net contact forces on all non-feet bodies (base, hips, thighs, shanks)."""

        all_net_forces_w = self.contact_sensor.data.net_forces_w # Shape: (num_envs, num_total_robot_bodies, 3)
        
        forces_on_non_feet_bodies = all_net_forces_w[:, self.bodies_except_feet_idx, :] # Shape: (num_envs, num_non_feet_bodies, 3)
        
        magnitudes_on_non_feet_bodies = torch.norm(forces_on_non_feet_bodies, p=2, dim=-1) # Shape: (num_envs, num_non_feet_bodies)
        
        total_force_magnitude_sum_per_env = torch.sum(magnitudes_on_non_feet_bodies, dim=1) # Shape: (num_envs)
        
        return total_force_magnitude_sum_per_env

    def get_bucket_indices(self, height_cmds: torch.Tensor, length_cmds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bucket indices for given height and length commands."""
        if self.height_bucket_info is None or self.length_bucket_info is None:
            return torch.zeros_like(height_cmds, dtype=torch.long), torch.zeros_like(length_cmds, dtype=torch.long)

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        if h_info["num"] <= 0:
            height_indices = torch.zeros_like(height_cmds, dtype=torch.long)
        elif h_info["num"] == 1 or h_info["width"] == 0: # check for width == 0 added
            height_indices = torch.zeros_like(height_cmds, dtype=torch.long)
        else:
            height_indices = ((height_cmds - h_info["min"]) / h_info["width"]).long()
            height_indices = torch.clamp(height_indices, 0, h_info["num"] - 1)

        if l_info["num"] <= 0:
            length_indices = torch.zeros_like(length_cmds, dtype=torch.long)
        elif l_info["num"] == 1 or l_info["width"] == 0: # check for width == 0 added
            length_indices = torch.zeros_like(length_cmds, dtype=torch.long)
        else:
            length_indices = ((length_cmds - l_info["min"]) / l_info["width"]).long()
            length_indices = torch.clamp(length_indices, 0, l_info["num"] - 1)
            
        return height_indices, length_indices

    def update_bucketed_metrics(self, env_ids: Sequence[int], jump_phase: torch.Tensor):
        """Update bucketed metrics for environments that ended episodes."""
        if self.cfg.metrics_bucketing is None or self.height_bucket_info is None or self.length_bucket_info is None:
            return
        
        if len(env_ids) == 0: return # return early if env_ids is empty

        # Get height and length commands directly
        height_cmds_ended_episode = self.target_height[env_ids].clone() # env -> self
        length_cmds_ended_episode = self.target_length[env_ids].clone() # env -> self
        
        takeoff_errors_ended_episode = self.takeoff_relative_error[env_ids].clone()
        flight_angle_errors_ended_episode = self.angle_error_at_landing[env_ids].clone()

        phase_at_termination = jump_phase[env_ids].clone()

        # Determine valid metrics
        valid_for_takeoff_metric_mask = (phase_at_termination == Phase.FLIGHT) | (phase_at_termination == Phase.LANDING)
        valid_for_flight_metric_mask = (phase_at_termination == Phase.LANDING)
        
        if torch.any(valid_for_takeoff_metric_mask) or torch.any(valid_for_flight_metric_mask):
            height_bucket_indices, length_bucket_indices = self.get_bucket_indices(
                height_cmds_ended_episode, length_cmds_ended_episode
            )
            
            num_length_buckets = self.length_bucket_info["num"]
            # Ensure num_length_buckets is not zero to prevent division by zero or incorrect flat_indices
            if num_length_buckets == 0: 
                # This case should ideally be handled by get_bucket_indices returning zero tensors
                # or by _init_bucketing_system ensuring num_length_buckets > 0 if bucketing is enabled.
                # For now, if it somehow reaches here with 0, we can't proceed with bucketing.
                return

            flat_indices = height_bucket_indices * num_length_buckets + length_bucket_indices


            # Update takeoff error metrics
            if torch.any(valid_for_takeoff_metric_mask):
                src_takeoff_errors = takeoff_errors_ended_episode[valid_for_takeoff_metric_mask]
                # Filter out NaN values before scatter_add_
                nan_mask_takeoff = ~torch.isnan(src_takeoff_errors)
                src_takeoff_errors_filtered = src_takeoff_errors[nan_mask_takeoff]
                
                if src_takeoff_errors_filtered.numel() > 0: # Proceed only if there are non-NaN errors
                    indices_for_takeoff_update = flat_indices[valid_for_takeoff_metric_mask][nan_mask_takeoff]
                    
                    self.bucketed_takeoff_error_sum.view(-1).scatter_add_(
                        0, indices_for_takeoff_update, src_takeoff_errors_filtered
                    )
                    self.bucketed_takeoff_error_count.view(-1).scatter_add_(
                        0, indices_for_takeoff_update, torch.ones_like(src_takeoff_errors_filtered, dtype=torch.int64)
                    )

            # Update flight angle error metrics
            if torch.any(valid_for_flight_metric_mask):
                src_flight_angle_errors = flight_angle_errors_ended_episode[valid_for_flight_metric_mask]
                # Filter out NaN values before scatter_add_
                nan_mask_flight = ~torch.isnan(src_flight_angle_errors)
                src_flight_angle_errors_filtered = src_flight_angle_errors[nan_mask_flight]

                if src_flight_angle_errors_filtered.numel() > 0: # Proceed only if there are non-NaN errors
                    indices_for_flight_update = flat_indices[valid_for_flight_metric_mask][nan_mask_flight]

                    self.bucketed_flight_angle_error_sum.view(-1).scatter_add_(
                        0, indices_for_flight_update, src_flight_angle_errors_filtered
                    )
                    self.bucketed_flight_angle_error_count.view(-1).scatter_add_(
                        0, indices_for_flight_update, torch.ones_like(src_flight_angle_errors_filtered, dtype=torch.int64)
                    )

    def log_bucketed_errors(self) -> Dict[str, float]:
        """Generate bucketed error logs."""
        if self.cfg.metrics_bucketing is None or self.height_bucket_info is None or self.length_bucket_info is None:
            return {}

        logs = {}
        
        # Prevent division by zero, replace with NaN if count is 0
        counts_takeoff = self.bucketed_takeoff_error_count.float()
        avg_takeoff_errors = self.bucketed_takeoff_error_sum / (counts_takeoff + 1e-8) 
        avg_takeoff_errors[counts_takeoff == 0] = torch.nan # Use torch.nan

        counts_flight_angle = self.bucketed_flight_angle_error_count.float()
        avg_flight_angle_errors = self.bucketed_flight_angle_error_sum / (counts_flight_angle + 1e-8)
        avg_flight_angle_errors[counts_flight_angle == 0] = torch.nan # Use torch.nan

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        for i in range(h_info["num"]):
            height_low = h_info["min"] + i * h_info["width"]
            height_high = h_info["min"] + (i + 1) * h_info["width"]
            height_range_str = f"height_{height_low:.2f}-{height_high:.2f}"
            if h_info["num"] == 1: # handle single bucket case
                 height_range_str = f"height_{h_info['min']:.2f}-{h_info['max']:.2f}"

            for j in range(l_info["num"]):
                length_low = l_info["min"] + j * l_info["width"]
                length_high = l_info["min"] + (j + 1) * l_info["width"]
                length_range_str = f"length_{length_low:.2f}-{length_high:.2f}"
                if l_info["num"] == 1: # handle single bucket case
                    length_range_str = f"length_{l_info['min']:.2f}-{l_info['max']:.2f}"


                takeoff_error_val = avg_takeoff_errors[i, j].item()
                flight_angle_error_val = avg_flight_angle_errors[i,j].item()

                base_key = f"bucketed_metrics/{height_range_str}_{length_range_str}"
                
                logs[f"{base_key}/takeoff_avg_error"] = takeoff_error_val
                logs[f"{base_key}/takeoff_error_count"] = self.bucketed_takeoff_error_count[i,j].item()
                logs[f"{base_key}/flight_angle_avg_error"] = flight_angle_error_val
                logs[f"{base_key}/flight_angle_error_count"] = self.bucketed_flight_angle_error_count[i,j].item()

        return logs

    def calculate_takeoff_relative_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Calculate relative takeoff error for given environments."""
        target_vector = self.target_vel_at_liftoff[env_ids]
        actual_vector = self.actual_vel_at_liftoff[env_ids]
        norm_target_vector = torch.norm(target_vector, dim=-1)
        # Add a small epsilon to prevent division by zero if norm_target_vector is zero
        return torch.norm(actual_vector - target_vector, dim=-1) / (norm_target_vector)
    
    def calculate_takeoff_angle_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Calculate takeoff angle error for given environments."""
        target_vector = self.target_vel_at_liftoff[env_ids]
        actual_vector = self.actual_vel_at_liftoff[env_ids]
        
        norm_target_vector = torch.norm(target_vector, dim=-1)
        norm_actual_vector = torch.norm(actual_vector, dim=-1)
        
        # Add epsilon to prevent division by zero
        denominator = (norm_target_vector * norm_actual_vector) + 1e-8
        
        cos_angle = torch.sum(target_vector * actual_vector, dim=-1) / denominator
        # Clamp cos_angle to prevent acos from producing NaN due to floating point inaccuracies
        angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
        return angle_error
    
    def calculate_takeoff_pitch_error(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate pitch error between takeoff vectors by projecting to XZ plane.
        
        Returns:
            abs_pitch_error: Absolute pitch error in radians
            signed_pitch_error: Signed pitch error in radians (actual - target)
        """
        target_vector = self.target_vel_at_liftoff[env_ids]
        actual_vector = self.actual_vel_at_liftoff[env_ids]
        
        # Project vectors to XZ plane by extracting X and Z components
        target_xz = target_vector[:, [0, 2]]  # Shape: (N, 2)
        actual_xz = actual_vector[:, [0, 2]]  # Shape: (N, 2)
        
        # Calculate pitch angles using atan2(x, z) since 0 vertical and clock-wise is positive
        target_pitch = torch.atan2(target_xz[:, 0], target_xz[:, 1])  # z, x
        actual_pitch = torch.atan2(actual_xz[:, 0], actual_xz[:, 1])  # z, x
        
        # Calculate signed and absolute pitch errors
        signed_pitch_error = actual_pitch - target_pitch
        abs_pitch_error = torch.abs(signed_pitch_error)
        
        return abs_pitch_error, signed_pitch_error
    
    def update_jump_phase(self) -> None:
        """Update the robot phase for the specified environments"""
        
        self.prev_jump_phase = self.jump_phase.clone()
                    
        com_acc_z = self.com_acc[:, 2]
        gravity = -9.81
        gravity_condition = (com_acc_z < (gravity + 0.1))
        
        self.jump_phase[self.takeoff_mask & gravity_condition] = Phase.FLIGHT
        
        falling = self.com_vel[:, 2] < -0.1  
        
        all_body_except_feet_heights_w = self.robot.data.body_pos_w[:, self.bodies_except_feet_idx, 2]
        any_body_except_feet_too_low = torch.any(all_body_except_feet_heights_w < 0.1, dim=-1)
        feet_heights_w = self.robot.data.body_pos_w[:, self.feet_body_idx, 2]
        any_feet_too_low = torch.any(feet_heights_w < 0.04, dim=-1)

        self.jump_phase[self.flight_mask & falling & (any_body_except_feet_too_low | any_feet_too_low)] = Phase.LANDING
        
        self.takeoff_mask = self.jump_phase == Phase.TAKEOFF
        self.flight_mask = self.jump_phase == Phase.FLIGHT
        self.landing_mask = self.jump_phase == Phase.LANDING
        self.takeoff_to_flight_mask = (self.jump_phase == Phase.FLIGHT) & (self.prev_jump_phase == Phase.TAKEOFF)
        self.flight_to_landing_mask = (self.jump_phase == Phase.LANDING) & (self.prev_jump_phase == Phase.FLIGHT)

        
        
        # com_acc_z = self.com_acc[:, 2]
        # com_vel_z = self.com_vel[:, 2]
        # #com_height = self.com_pos[:, 2]
        # gravity = -9.81
        
        
        # # Only print debug info every 10 steps
        # if self.common_step_counter % 20 == 0:
        #     print(f"acc z")
        #     print(f"mean: {torch.mean(com_acc_z)}")
        #     print(f"acc z min: {torch.min(com_acc_z)}")
        #     print(f"acc z max: {torch.max(com_acc_z)}")
            
        #     print(f"com height")
        #     print(f"mean: {torch.mean(self.com_pos[:, 2])}")
        #     print(f"min: {torch.min(self.com_pos[:, 2])}")
        #     print(f"max: {torch.max(self.com_pos[:, 2])}")
            
        #     print(f"com vel z")
        #     print(f"mean: {torch.mean(self.com_vel[:, 2])}")
        #     print(f"min: {torch.min(self.com_vel[:, 2])}")
        #     print(f"max: {torch.max(self.com_vel[:, 2])}")
            
        # self.jump_phase[self.takeoff_mask & (com_acc_z < (gravity + 0.1))] = Phase.FLIGHT
                  
        # self.jump_phase[self.flight_mask & (com_acc_z > (gravity + 0.1)) & (com_vel_z < -0.2)] = Phase.LANDING
                
        # self.takeoff_mask = self.jump_phase == Phase.TAKEOFF
        # self.flight_mask = self.jump_phase == Phase.FLIGHT
        # self.landing_mask = self.jump_phase == Phase.LANDING
        # self.takeoff_to_flight_mask = (self.jump_phase == Phase.FLIGHT) & (self.prev_jump_phase == Phase.TAKEOFF)
        # self.flight_to_landing_mask = (self.jump_phase == Phase.LANDING) & (self.prev_jump_phase == Phase.FLIGHT)

    def log_phase_info(self):
        """Logs the distribution of phases to the extras dict."""

        phase_log = {}
        for phase_enum in Phase:
            phase_val = phase_enum.value
            phase_name = phase_enum.name
            
            # Calculate count and percentage
            phase_mask = (self.jump_phase == phase_val)
            count = torch.sum(phase_mask).item()  # Single value transfer
            phase_log[f"phase_dist/{phase_name}"] = count / self.num_envs
                
        self.extras["log"].update(phase_log)



