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
                
        self.feet_body_idx = torch.tensor(self.robot.find_bodies(".*FOOT.*")[0])
        self.hips_body_idx = torch.tensor(self.robot.find_bodies(".*HIP.*")[0])
        self.thighs_body_idx = torch.tensor(self.robot.find_bodies(".*THIGH.*")[0])
        self.shanks_body_idx = torch.tensor(self.robot.find_bodies(".*SHANK.*")[0])
        self.bodies_except_feet_idx = torch.cat((self.hips_body_idx, self.thighs_body_idx, self.shanks_body_idx))
        
        self.hip_joint_idx = torch.tensor(self.robot.find_joints(".*HAA.*")[0])
        self.abduction_joint_idx = torch.tensor(self.robot.find_joints(".*HFE.*")[0])
        self.knee_joint_idx = torch.tensor(self.robot.find_joints(".*KFE.*")[0])
        
        self.center_of_mass_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.center_of_mass_pos = torch.zeros(self.num_envs, 3, device=self.device)
        
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
        
        self.target_height = torch.zeros(self.num_envs, device=self.device)  # Target jump height in meters
        self.target_length = torch.zeros(self.num_envs, device=self.device)  # Target jump length in meters
        
        self.dynamic_takeoff_pitch = torch.zeros(self.num_envs, device=self.device)
        self.dynamic_takeoff_magnitude = torch.zeros(self.num_envs, device=self.device)
        
        self.cmd_height_range = self.cfg.command_ranges.height_range
        self.cmd_length_range = self.cfg.command_ranges.length_range
        self.dynamic_takeoff_vector = torch.zeros(self.num_envs, 3, device=self.device) #Will be updated while env is in takeoff phase, then hold its last value until reset
        #for curriculum
        self.mean_episode_env_steps = 0
    

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        
        self.center_of_mass_lin_vel = self.get_center_of_mass_lin_vel()
        self.center_of_mass_pos = self.get_center_of_mass_pos()
        current_height = self.center_of_mass_pos[:, 2]
        new_max_height_mask = (current_height > self.max_height_achieved)
        self.max_height_achieved[new_max_height_mask] = current_height[new_max_height_mask]
            
        self.update_jump_phase()
        self.log_phase_info()
            
        # Update bucketed takeoff errors via metrics system
        bucketed_logs = self.log_bucketed_errors()
        self.extras["log"].update(bucketed_logs)

        #Track takeoff velocity
        if torch.any(self.takeoff_mask):
            takeoff_env_ids = self.takeoff_mask.nonzero(as_tuple=False).squeeze(-1)
            self.update_dynamic_takeoff_vector(takeoff_env_ids)
            current_vel_magnitude = torch.norm(self.center_of_mass_lin_vel, dim=-1) # Shape: (num_envs,)
            max_vel_magnitude = torch.norm(self.max_takeoff_vel, dim=-1)
            
            update_mask = self.takeoff_mask & (current_vel_magnitude > max_vel_magnitude) # Shape: (num_envs,)
            self.max_takeoff_vel[update_mask] = self.center_of_mass_lin_vel[update_mask]

        #Track takeoff relative error
        if torch.any(self.takeoff_to_flight_mask):
            takeoff_to_flight_env_ids = self.takeoff_to_flight_mask.nonzero(as_tuple=False).squeeze(-1)
            self.takeoff_relative_error[takeoff_to_flight_env_ids] = self.calculate_takeoff_relative_error(takeoff_to_flight_env_ids)
        
        #Track landing errors
        if torch.any(self.flight_to_landing_mask):
            self.angle_error_at_landing[self.flight_to_landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.flight_to_landing_mask])
            current_com_x = self.center_of_mass_pos[:,0]
            self.length_error_at_landing[self.flight_to_landing_mask] = current_com_x[self.flight_to_landing_mask] - self.target_length[self.flight_to_landing_mask]
        
        # current_sum_contact_forces = self.sum_contact_forces() # Avoid rebinding self.sum_contact_forces
        # self.extras["log"]["takeoff_contact_forces"] = current_sum_contact_forces[self.takeoff_mask].mean().item() if torch.any(self.takeoff_mask) else float('nan')
        # self.extras["log"]["flight_contact_forces"] = current_sum_contact_forces[self.flight_mask].mean().item() if torch.any(self.flight_mask) else float('nan')
        # self.extras["log"]["landing_contact_forces"] = current_sum_contact_forces[self.landing_mask].mean().item() if torch.any(self.landing_mask) else float('nan')
        
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
            # Update bucketed metrics before reset
            self.update_bucketed_metrics(env_ids, self.jump_phase) # Removed redundant self

            # Calculate success metrics
            takeoff_log_data = self.calculate_takeoff_success(env_ids) # Removed redundant self
            flight_log_data = self.calculate_flight_success(env_ids)
            landing_log_data = self.calculate_landing_success(env_ids)
            
            error_log_data = self.calculate_error_metrics(env_ids) # Removed redundant self
            
            # Update running success rates (uses batch rates calculated above)
            self.update_running_success_rates(env_ids)
            
            # Sample new commands before calling super()._reset_idx
            self.sample_command(env_ids)   
            
            self.update_dynamic_takeoff_vector(env_ids)
            self.dynamic_takeoff_pitch[env_ids], self.dynamic_takeoff_magnitude[env_ids] = convert_vector_to_pitch_magnitude(self.dynamic_takeoff_vector[env_ids])

            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            self.extras["log"].update(landing_log_data) # Log landing metrics
            self.extras["log"].update(error_log_data)
            
            self.extras["log"].update({
                "full_jump_success_rate": self.full_jump_success_rate,
                "running_success_rate": self.success_rate,
            })

            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
            
            self.takeoff_mask[env_ids] = True
            self.flight_mask[env_ids] = False
            self.landing_mask[env_ids] = False
            
            self.max_takeoff_vel[env_ids] = torch.zeros_like(self.max_takeoff_vel[env_ids])
            self.angle_error_at_landing[env_ids] = float('nan')
            self.takeoff_relative_error[env_ids] = float('nan')
            self.max_height_achieved[env_ids] = 0 # Reset to 0.0 not COM height
            self.length_error_at_landing[env_ids] = float('nan')
            self.height_error_peak[env_ids] = float('nan')
            self.length_error_at_termination[env_ids] = float('nan')

            self.flight_success[env_ids] = False
            self.takeoff_success[env_ids] = False
            self.landing_success[env_ids] = False
             
    def _init_success_metrics(self):
        """Initialize success tracking tensors."""
        # Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_takeoff_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.takeoff_relative_error = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.takeoff_success_rate = 0.0
        self.running_takeoff_success_rate = 0.0
        
        # Flight Success
        self.angle_error_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.flight_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.flight_success_rate = 0.0
        self.running_flight_success_rate = 0.0
        
        # Landing Success
        self.landing_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.landing_success_rate = 0.0
        self.running_landing_success_rate = 0.0
        
        # Full Jump Success
        self.full_jump_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.full_jump_success_rate = 0.0
        self.success_rate = 0.0

    def _init_error_metrics(self):
        """Initialize error tracking tensors."""
        self.max_height_achieved = torch.zeros(self.num_envs, device=self.device)
        
        # Use NaN as sentinel value to indicate "not calculated yet" instead of 0.0
        self.length_error_at_landing = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.height_error_peak = torch.full((self.num_envs,), float('nan'), device=self.device)
        self.length_error_at_termination = torch.full((self.num_envs,), float('nan'), device=self.device)

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
                "takeoff_success_rate": self.running_takeoff_success_rate,
                "takeoff_relative_error": float('nan'),
            }
        
        has_taken_off_ids = env_ids[has_taken_off_mask]
        
        relative_error = self.calculate_takeoff_relative_error(has_taken_off_ids) # env -> self
        angle_error = self.calculate_takeoff_angle_error(has_taken_off_ids) # env -> self
        
        takeoff_vector = self.dynamic_takeoff_vector[has_taken_off_ids] # env -> self
        max_takeoff_vel_vec = self.max_takeoff_vel[has_taken_off_ids]
        
        magnitude_ratio_error = torch.norm(max_takeoff_vel_vec, dim=-1) / torch.norm(takeoff_vector, dim=-1) - 1
        magnitude_ok = torch.abs(magnitude_ratio_error) < self.cfg.takeoff_magnitude_ratio_error_threshold
        angle_ok = angle_error < self.cfg.takeoff_angle_error_threshold_rad
        
        self.takeoff_success[has_taken_off_ids] = magnitude_ok & angle_ok
        
        num_successful_takeoffs = torch.sum(self.takeoff_success[has_taken_off_ids]).item()
        num_reset_envs = len(env_ids)
        self.takeoff_success_rate = num_successful_takeoffs / num_reset_envs
        
        mean_angle_error = torch.nanmean(angle_error).item() if angle_error.numel() > 0 else float('nan')
        mean_relative_error = torch.nanmean(relative_error).item() if relative_error.numel() > 0 else float('nan')
        mean_magnitude_ratio_error = torch.nanmean(magnitude_ratio_error).item() if magnitude_ratio_error.numel() > 0 else float('nan')
        
        return {
            "takeoff_angle_error": mean_angle_error,
            "takeoff_magnitude_ratio_error": mean_magnitude_ratio_error,
            "takeoff_success_rate": self.running_takeoff_success_rate,
            "takeoff_relative_error": mean_relative_error,
        }

    def calculate_flight_success(self, env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate flight success metrics for given environments."""
        taken_off_mask = self.flight_mask[env_ids] | self.landing_mask[env_ids]
        num_taken_off = torch.sum(taken_off_mask).item()
        
        if num_taken_off == 0:
            return {
                "flight_success_rate": self.running_flight_success_rate,
                "flight_angle_error_at_landing": float('nan'),
            }

        in_landing_phase = self.jump_phase[env_ids] == Phase.LANDING
        landed_env_ids = env_ids[in_landing_phase]
        
        angle_ok = self.angle_error_at_landing[landed_env_ids] < self.cfg.flight_angle_error_threshold
        
        num_successful_landings = torch.sum(angle_ok).item()

        ids_successful_landings = landed_env_ids[angle_ok]
        self.flight_success[ids_successful_landings] = True
        self.flight_success_rate = num_successful_landings / num_taken_off
        
        angle_errors_at_landing = self.angle_error_at_landing[landed_env_ids]
        mean_angle_error = torch.nanmean(angle_errors_at_landing).item() if angle_errors_at_landing.numel() > 0 else float('nan')
        
        return {
            "flight_success_rate": self.running_flight_success_rate,
            "flight_angle_error_at_landing": mean_angle_error,
        }

    def calculate_landing_success(self, env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate landing success metrics for given environments."""
        landing_mask = self.landing_mask[env_ids]
        landing_env_ids = env_ids[landing_mask]
        
        if len(landing_env_ids) == 0:
            self.landing_success_rate = 0.0
        else:   
            timed_out_mask = self.termination_manager.get_term("time_out")[landing_env_ids]
            num_successful_landings = torch.sum(timed_out_mask).item()
            
            self.landing_success[landing_env_ids] = timed_out_mask
            self.landing_success_rate = num_successful_landings / len(landing_env_ids)
            
        return {"landing_success_rate": self.running_landing_success_rate}

    def calculate_error_metrics(self, reset_env_ids: Sequence[int]) -> Dict[str, float]:
        """Calculate error metrics, this is called when reset_idx is called by the environment."""
        error_log_data = {
            "length_error_at_landing": float('nan'),
            "height_error_peak": float('nan'), 
            "length_error_at_termination": float('nan'),
            "abs_length_error_at_landing": float('nan'),
            "abs_height_error_peak": float('nan'),
            "abs_length_error_at_termination": float('nan'),
        }
        
        if len(reset_env_ids) > 0:
            height_error_at_peak = self.max_height_achieved[reset_env_ids] - self.target_height[reset_env_ids] # env -> self
            error_log_data["height_error_peak"] = height_error_at_peak.mean().item()
            error_log_data["abs_height_error_peak"] = torch.abs(height_error_at_peak).mean().item()

            landing_mask = self.jump_phase[reset_env_ids] == Phase.LANDING
            if torch.any(landing_mask):
                landing_env_ids = reset_env_ids[landing_mask]  # Get actual environment indices
                
                length_landing_errors = self.length_error_at_landing[landing_env_ids]
            
                error_log_data["length_error_at_landing"] = torch.nanmean(length_landing_errors).item()
                error_log_data["abs_length_error_at_landing"] = torch.nanmean(torch.abs(length_landing_errors)).item()
            
        return error_log_data

    def update_running_success_rates(self, env_ids: Sequence[int]):
        """Update exponentially smoothed running success rates."""
 
        alpha = len(env_ids) / self.num_envs

        self.running_takeoff_success_rate = alpha * self.takeoff_success_rate + \
                                           (1 - alpha) * self.running_takeoff_success_rate
        
        self.running_flight_success_rate = alpha * self.flight_success_rate + \
                                          (1 - alpha) * self.running_flight_success_rate
        
        self.running_landing_success_rate = alpha * self.landing_success_rate + \
                                           (1 - alpha) * self.running_landing_success_rate

        # Calculate full jump success for the current batch and its running average
        if len(env_ids) > 0: # ensure env_ids is not empty before indexing
            self.full_jump_success[env_ids] = self.takeoff_success[env_ids] & \
                                            self.flight_success[env_ids] & \
                                            self.landing_success[env_ids]
            self.full_jump_success_rate = torch.mean(self.full_jump_success[env_ids].float()).item()
            
        else:
            self.full_jump_success_rate = 0.0 # Or handle as appropriate if env_ids can be empty
        
        # Exponential smoothing of success rate
        self.success_rate = alpha * self.full_jump_success_rate + (1 - alpha) * self.success_rate

    def sample_command(self, env_ids: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        self.target_height[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cmd_height_range)
        self.target_length[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cmd_length_range)
        
    def update_dynamic_takeoff_vector(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Calculate the required takeoff velocity vector based on current COM position and target trajectory.
        
        This function continuously updates the required velocity vector based on:
        - The robot's current center of mass position
        - The target height and length trajectory 
        - Physics calculations for projectile motion

        Returns:
            Tensor of shape (num_envs or len(env_ids), 3) containing [x, y, z] velocity components
        """
        target_height = self.target_height[env_ids]
        target_length = self.target_length[env_ids]
        center_of_mass_pos = self.center_of_mass_pos[env_ids]
        
        pitch, magnitude = convert_height_length_to_pitch_magnitude_from_position(
            target_height, target_length, center_of_mass_pos, gravity=9.81
        )
        
        self.dynamic_takeoff_pitch[env_ids] = pitch
        self.dynamic_takeoff_magnitude[env_ids] = magnitude
        self.dynamic_takeoff_vector[env_ids] = convert_pitch_magnitude_to_vector(pitch, magnitude)

    def _abs_angle_error(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Returns the absolute angle error of the robot for the given envs
        """
        w = quat[:, 0]
        angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
        return torch.abs(angle)
    
    def get_center_of_mass_lin_vel(self) -> torch.Tensor:
        """
        Returns the center of mass linear velocity of the robot.
        Returns a tensor of shape (num_envs, 3)
        """
        total_mass = torch.sum(self.robot.data.default_mass, dim=1).unsqueeze(-1).to(self.device) # Shape: (num_envs, 1)
        masses = self.robot.data.default_mass.unsqueeze(-1).to(self.device) # Shape: (num_envs, num_bodies, 1)
        weighed_lin_vels = self.robot.data.body_com_lin_vel_w * masses # Shape: (num_envs, num_bodies, 3)
        com_lin_vel = torch.sum(weighed_lin_vels, dim=1) / total_mass # Shape: (num_envs, 3)
        return com_lin_vel

    def get_center_of_mass_pos(self) -> torch.Tensor:
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
        """Sum the contact forces of the robot, expect feet"""
        sensor_cfg = SceneEntityCfg(name="contact_sensor", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"])
        contact_sensor: ContactSensor = self.scene[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids] 
        sum_forces_magnitude = torch.sum(net_contact_forces, dim=-1) #shape [num_envs]
        return sum_forces_magnitude

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
        takeoff_vector = self.dynamic_takeoff_vector[env_ids]
        norm_takeoff_vector = torch.norm(takeoff_vector, dim=-1)
        # Add a small epsilon to prevent division by zero if norm_takeoff_vector is zero
        return torch.norm(self.max_takeoff_vel[env_ids] - takeoff_vector, dim=-1) / (norm_takeoff_vector + 1e-8)
    
    def calculate_takeoff_angle_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Calculate takeoff angle error for given environments."""
        takeoff_vector = self.dynamic_takeoff_vector[env_ids]
        max_takeoff_vel_vec = self.max_takeoff_vel[env_ids]
        
        norm_takeoff_vector = torch.norm(takeoff_vector, dim=-1)
        norm_max_takeoff_vel = torch.norm(max_takeoff_vel_vec, dim=-1)
        
        # Add epsilon to prevent division by zero
        denominator = (norm_takeoff_vector * norm_max_takeoff_vel) + 1e-8
        
        cos_angle = torch.sum(takeoff_vector * max_takeoff_vel_vec, dim=-1) / denominator
        # Clamp cos_angle to prevent acos from producing NaN due to floating point inaccuracies
        angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
        return angle_error
    
    def update_jump_phase(self) -> None:
        """Update the robot phase for the specified environments"""
        com_vel_magnitude = torch.norm(self.center_of_mass_lin_vel, dim=-1)
        max_takeoff_vel_magnitude = torch.norm(self.max_takeoff_vel, dim=-1)
        self.prev_jump_phase = self.jump_phase.clone()
        
        vel_not_increasing = com_vel_magnitude < max_takeoff_vel_magnitude - 0.1 #add margin for numerical errors and small variations
        com_height = self.center_of_mass_pos[:, 2]
        
        vel_condition = vel_not_increasing & (com_vel_magnitude > 0.5)
        height_condition = com_height > 0.15
        self.jump_phase[self.takeoff_mask & (vel_condition | height_condition)] = Phase.FLIGHT
        
        falling = self.center_of_mass_lin_vel[:, 2] < -0.1  
        
        all_body_heights_w = self.robot.data.body_pos_w[:, :, 2]
        any_body_too_low = torch.any(all_body_heights_w < 0.1, dim=-1)

        self.jump_phase[self.flight_mask & falling & any_body_too_low] = Phase.LANDING
        
        self.takeoff_mask = self.jump_phase == Phase.TAKEOFF
        self.flight_mask = self.jump_phase == Phase.FLIGHT
        self.landing_mask = self.jump_phase == Phase.LANDING
        self.takeoff_to_flight_mask = (self.jump_phase == Phase.FLIGHT) & (self.prev_jump_phase == Phase.TAKEOFF)
        self.flight_to_landing_mask = (self.jump_phase == Phase.LANDING) & (self.prev_jump_phase == Phase.FLIGHT)

    def log_phase_info(self):
        """Logs the distribution of phases and average height per phase to the extras dict."""

        phase_log = {}
        for phase_enum in Phase:
            phase_val = phase_enum.value
            phase_name = phase_enum.name
            
            # Calculate count and percentage
            phase_mask = (self.jump_phase == phase_val)
            count = torch.sum(phase_mask).item()  # Single value transfer
            phase_log[f"phase_dist/{phase_name}"] = count / self.num_envs
            
            # Calculate average height only if environments exist in this phase
            if count > 0:
                # Compute mean height directly on GPU, only transfer the result
                heights = self.robot.data.root_pos_w[:, 2]
                avg_height = torch.mean(heights[phase_mask]).item()  # Single value transfer
                phase_log[f"avg_height/{phase_name}"] = avg_height
            else:
                phase_log[f"avg_height/{phase_name}"] = 0.0
                
        self.extras["log"].update(phase_log)

