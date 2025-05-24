"""
Jump Metrics System for Full Jump Environment.

Handles all success rate calculations, error metrics, and bucketing functionality.
"""

import torch
from collections.abc import Sequence
from typing import Dict, Any, Optional, Tuple
from terms.phase import Phase
from terms.utils import get_center_of_mass_pos


class JumpMetrics:
    """Handles all jump performance metrics and success calculations."""
    
    def __init__(self, num_envs: int, device: torch.device, cfg):
        self.num_envs = num_envs
        self.device = device
        self.cfg = cfg
        
        # Initialize all metric tensors
        self._init_success_metrics()
        self._init_error_metrics()
        self._init_bucketing_system()
    
    def _init_success_metrics(self):
        """Initialize success tracking tensors."""
        # Takeoff Success
        self.takeoff_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_takeoff_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.max_takeoff_vel_magnitude = torch.zeros(self.num_envs, device=self.device)
        self.takeoff_relative_error = torch.zeros(self.num_envs, device=self.device)
        self.takeoff_success_rate = 0.0
        self.running_takeoff_success_rate = 0.0
        
        # Flight Success
        self.angle_error_at_landing = torch.zeros(self.num_envs, device=self.device)
        self.body_ang_vel_at_landing = torch.zeros(self.num_envs, 3, device=self.device)
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
        self.start_com_pos = torch.zeros(self.num_envs, 3, device=self.device)
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

            height_min, height_max = self.cfg.command_ranges.height_range
            length_min, length_max = self.cfg.command_ranges.length_range
            
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

    def calculate_takeoff_success(self, env_ids: Sequence[int], env, jump_phase: torch.Tensor) -> Dict[str, float]:
        """Calculate takeoff success metrics for given environments."""
        has_taken_off = (jump_phase[env_ids] == Phase.FLIGHT) | (jump_phase[env_ids] == Phase.LANDING)
        num_has_taken_off = torch.sum(has_taken_off).item()

        if num_has_taken_off > 0:
            has_taken_off_ids = env_ids[has_taken_off]
            
            relative_error = self._relative_takeoff_error(has_taken_off_ids, env)
            angle_error = self._takeoff_angle_error(has_taken_off_ids, env)
            
            takeoff_vector = env._get_dynamic_takeoff_vector(has_taken_off_ids)
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

    def calculate_flight_success(self, env_ids: Sequence[int], jump_phase: torch.Tensor) -> Dict[str, float]:
        """Calculate flight success metrics for given environments."""
        num_taken_off = torch.sum((jump_phase[env_ids] == Phase.FLIGHT) | (jump_phase[env_ids] == Phase.LANDING)).item()
        
        if num_taken_off > 0:
            in_landing_phase = jump_phase[env_ids] == Phase.LANDING
            landed_env_ids = env_ids[in_landing_phase]
            
            angle_ok = self.angle_error_at_landing[landed_env_ids] < self.cfg.flight_angle_error_threshold
            
            num_successful_landings = torch.sum(angle_ok).item()

            ids_successful_landings = landed_env_ids[angle_ok]
            self.flight_success[ids_successful_landings] = True
            self.flight_success_rate = num_successful_landings / num_taken_off
            
            return {
                "flight_success_rate": self.flight_success_rate,
                "flight_angle_error_at_landing": torch.mean(self.angle_error_at_landing[landed_env_ids]).item(),
            }
        else:
            return {
                "flight_success_rate": 0.0,
                "flight_angle_error_at_landing": 0.0,
            }

    def calculate_landing_success(self, env_ids: Sequence[int], jump_phase: torch.Tensor, termination_manager) -> Dict[str, float]:
        """Calculate landing success metrics for given environments."""
        landing_mask = jump_phase[env_ids] == Phase.LANDING
        landing_env_ids = env_ids[landing_mask]
        
        timed_out_mask = termination_manager.get_term("time_out")[landing_env_ids]
        num_successful_landings = torch.sum(timed_out_mask).item()
        num_landing_envs = len(landing_env_ids)
        
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

    def calculate_error_metrics(self, reset_env_ids: Sequence[int], jump_phase: torch.Tensor, 
                              prev_jump_phase: torch.Tensor, env) -> Dict[str, float]:
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
            
            # Calculate height error peak for environments that actually jumped
            height_error_at_peak = self.max_height_achieved[reset_env_ids] - env.target_height[reset_env_ids]
            error_log_data["height_error_peak"] = height_error_at_peak.mean().item()
            error_log_data["abs_height_error_peak"] = torch.abs(height_error_at_peak).mean().item()

            # Calculate length errors only for environments that reached landing
            landing_mask = jump_phase[reset_env_ids] == Phase.LANDING
            if torch.any(landing_mask):
                landing_env_ids = reset_env_ids[landing_mask]  # Get actual environment indices
                
                # Get length error values and filter out NaN sentinel values (not calculated)
                length_landing_errors = self.length_error_at_landing[landing_env_ids]
                length_termination_errors = self.length_error_at_termination[landing_env_ids]
                
                # Filter out NaN values (not calculated) but include both positive and negative errors
                valid_length_landing = length_landing_errors[~torch.isnan(length_landing_errors)]
                valid_length_termination = length_termination_errors[~torch.isnan(length_termination_errors)]
                
                # Calculate signed (directional) error averages for length
                if valid_length_landing.numel() > 0:
                    error_log_data["length_error_at_landing"] = valid_length_landing.mean().item()
                    error_log_data["abs_length_error_at_landing"] = torch.abs(valid_length_landing).mean().item()
                
                if valid_length_termination.numel() > 0:
                    error_log_data["length_error_at_termination"] = valid_length_termination.mean().item()
                    error_log_data["abs_length_error_at_termination"] = torch.abs(valid_length_termination).mean().item()
        
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
        self.full_jump_success[env_ids] = self.takeoff_success[env_ids] & \
                                        self.flight_success[env_ids] & \
                                        self.landing_success[env_ids]
        self.full_jump_success_rate = torch.mean(self.full_jump_success[env_ids].float()).item()
        
        # Exponential smoothing of success rate
        self.success_rate = alpha * self.full_jump_success_rate + (1 - alpha) * self.success_rate

    def reset_env_metrics(self, env_ids: Sequence[int], env):
        """Reset metrics for specific environments."""
        self.max_takeoff_vel[env_ids] = torch.zeros_like(self.max_takeoff_vel[env_ids])
        self.max_takeoff_vel_magnitude[env_ids] = torch.zeros_like(self.max_takeoff_vel_magnitude[env_ids])
        self.body_ang_vel_at_landing[env_ids] = torch.zeros_like(self.body_ang_vel_at_landing[env_ids])
        self.angle_error_at_landing[env_ids] = torch.zeros_like(self.angle_error_at_landing[env_ids])
        
        self.flight_success[env_ids] = False
        self.takeoff_success[env_ids] = False
        self.landing_success[env_ids] = False
        self.takeoff_relative_error[env_ids] = 0.0
        
        # Reset error metrics and store starting position for new episodes
        current_com_pos = get_center_of_mass_pos(env)
        self.start_com_pos[env_ids] = current_com_pos[env_ids]
        self.max_height_achieved[env_ids] = 0.0
        self.length_error_at_landing[env_ids] = float('nan')
        self.height_error_peak[env_ids] = float('nan')
        self.length_error_at_termination[env_ids] = float('nan')

    def get_bucket_indices(self, height_cmds: torch.Tensor, length_cmds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get bucket indices for given height and length commands."""
        if self.height_bucket_info is None or self.length_bucket_info is None:
            return torch.zeros_like(height_cmds, dtype=torch.long), torch.zeros_like(length_cmds, dtype=torch.long)

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        if h_info["num"] <= 0:
            height_indices = torch.zeros_like(height_cmds, dtype=torch.long)
        elif h_info["num"] == 1 or h_info["width"] == 0:
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

    def update_bucketed_metrics(self, env_ids: Sequence[int], env, jump_phase: torch.Tensor):
        """Update bucketed metrics for environments that ended episodes."""
        if self.cfg.metrics_bucketing is None or self.height_bucket_info is None or self.length_bucket_info is None:
            return

        # Get height and length commands directly
        height_cmds_ended_episode = env.target_height[env_ids].clone()
        length_cmds_ended_episode = env.target_length[env_ids].clone()
        
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
            flat_indices = height_bucket_indices * num_length_buckets + length_bucket_indices

            # Update takeoff error metrics
            if torch.any(valid_for_takeoff_metric_mask):
                src_takeoff_errors = takeoff_errors_ended_episode[valid_for_takeoff_metric_mask]
                indices_for_takeoff_update = flat_indices[valid_for_takeoff_metric_mask]
                
                self.bucketed_takeoff_error_sum.view(-1).scatter_add_(
                    0, indices_for_takeoff_update, src_takeoff_errors
                )
                self.bucketed_takeoff_error_count.view(-1).scatter_add_(
                    0, indices_for_takeoff_update, torch.ones_like(src_takeoff_errors, dtype=torch.int64)
                )

            # Update flight angle error metrics
            if torch.any(valid_for_flight_metric_mask):
                src_flight_angle_errors = flight_angle_errors_ended_episode[valid_for_flight_metric_mask]
                indices_for_flight_update = flat_indices[valid_for_flight_metric_mask]

                self.bucketed_flight_angle_error_sum.view(-1).scatter_add_(
                    0, indices_for_flight_update, src_flight_angle_errors
                )
                self.bucketed_flight_angle_error_count.view(-1).scatter_add_(
                    0, indices_for_flight_update, torch.ones_like(src_flight_angle_errors, dtype=torch.int64)
                )

    def log_bucketed_errors(self) -> Dict[str, float]:
        """Generate bucketed error logs."""
        if self.cfg.metrics_bucketing is None or self.height_bucket_info is None or self.length_bucket_info is None:
            return {}

        logs = {}
        
        # Prevent division by zero, replace with NaN if count is 0
        counts = self.bucketed_takeoff_error_count.float()
        avg_errors = self.bucketed_takeoff_error_sum / (counts + 1e-8) 
        avg_errors[counts == 0] = torch.nan

        h_info = self.height_bucket_info
        l_info = self.length_bucket_info

        for i in range(h_info["num"]):
            height_low = h_info["min"] + i * h_info["width"]
            height_high = h_info["min"] + (i + 1) * h_info["width"]
            height_range_str = f"height_{height_low:.2f}-{height_high:.2f}"
            if h_info["num"] == 1:
                height_range_str = f"height_{h_info['min']:.2f}-{h_info['max']:.2f}"

            for j in range(l_info["num"]):
                length_low = l_info["min"] + j * l_info["width"]
                length_high = l_info["min"] + (j + 1) * l_info["width"]
                length_range_str = f"length_{length_low:.2f}-{length_high:.2f}"
                if l_info["num"] == 1:
                    length_range_str = f"length_{l_info['min']:.2f}-{l_info['max']:.2f}"

                error_val = avg_errors[i, j].item()
                base_key = f"bucketed_takeoff/{height_range_str}_{length_range_str}"
                
                if torch.isnan(torch.tensor(error_val)):
                    logs[f"{base_key}_avg_error"] = -1.0  # Or float('nan') if wandb handles it
                else:
                    logs[f"{base_key}_avg_error"] = error_val
        
        return logs

    def _relative_takeoff_error(self, env_ids: Sequence[int], env) -> torch.Tensor:
        """Calculate relative takeoff error for given environments."""
        takeoff_vector = env._get_dynamic_takeoff_vector(env_ids)
        return torch.norm(self.max_takeoff_vel[env_ids] - takeoff_vector, dim=-1) / torch.norm(takeoff_vector, dim=-1)
    
    def _takeoff_angle_error(self, env_ids: Sequence[int], env) -> torch.Tensor:
        """Calculate takeoff angle error for given environments."""
        takeoff_vector = env._get_dynamic_takeoff_vector(env_ids)
        max_takeoff_vel_vec = self.max_takeoff_vel[env_ids]
        cos_angle = torch.sum(takeoff_vector * max_takeoff_vel_vec, dim=-1) / \
                   (torch.norm(takeoff_vector, dim=-1) * torch.norm(max_takeoff_vel_vec, dim=-1))
        angle_error = torch.acos(torch.clamp(cos_angle, min=-1.0, max=1.0))
        return angle_error 