from __future__ import annotations

import torch
import os
import json
import numpy as np
from typing import Type, Optional, Any, Union

from isaaclab.utils.types import ArticulationActions
from isaaclab.actuators.actuator_pd import DCMotor
from isaaclab.actuators.actuator_cfg import DCMotorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import read_file

class GruActuatorDebug(DCMotor):
    """Debug version of GRU actuator that logs all inputs and outputs for analysis."""

    cfg: 'GruActuatorDebugCfg'
    network: torch.jit.ScriptModule
    sea_hidden_state: torch.Tensor

    # Normalization tensors
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    def __init__(self, cfg: 'GruActuatorDebugCfg', *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # Load Hydra config
        try:
            import yaml
            with open(cfg.hydra_config_path, 'r') as f:
                hydra_config_data = yaml.safe_load(f)
                hydra_config = hydra_config_data.get('model', hydra_config_data)
        except Exception as e:
            raise ValueError(f"Error loading Hydra config: {e}")

        self.use_residual_model = hydra_config.get("use_residual", False)
        self.training_frequency_hz = hydra_config_data.get("data", {}).get("resampling_frequency_hz", 240.0)
        
        # Get training preprocessing parameters
        self.filter_cutoff_freq_hz = hydra_config_data.get("data", {}).get("filter_cutoff_freq_hz")
        self.sensor_biases = hydra_config_data.get("data", {}).get("sensor_biases", {})
        
        print(f"Training used:")
        print(f"  - Filter cutoff: {self.filter_cutoff_freq_hz} Hz")
        print(f"  - Sensor biases: {self.sensor_biases}")
        print(f"  - Residual model: {self.use_residual_model}")

        # Load GRU architecture
        self.trained_gru_num_layers = hydra_config.get("gru_num_layers", 2)
        self.trained_gru_hidden_dim = hydra_config.get("gru_hidden_dim", 128)

        # Load network
        file_bytes = read_file(cfg.network_file)
        self.network = torch.jit.load(file_bytes, map_location=self._device).eval()

        # Load normalization stats
        with open(cfg.normalization_stats_path, 'r') as f:
            norm_stats_dict = json.load(f)
            
        self.input_mean = torch.tensor(norm_stats_dict["input_mean"], device=self._device, dtype=torch.float32).view(1, 1, -1)
        self.input_std = torch.tensor(norm_stats_dict["input_std"], device=self._device, dtype=torch.float32).view(1, 1, -1)
        self.target_mean = torch.tensor(norm_stats_dict["target_mean"], device=self._device, dtype=torch.float32).squeeze()
        self.target_std = torch.tensor(norm_stats_dict["target_std"], device=self._device, dtype=torch.float32).squeeze()

        # Prevent division by zero
        self.input_std[self.input_std < 1e-7] = 1.0
        if self.target_std.ndim == 0 and self.target_std < 1e-7:
            self.target_std = torch.tensor(1.0, device=self._device, dtype=torch.float32)

        print(f"Normalization stats:")
        print(f"  Input mean: {norm_stats_dict['input_mean']}")
        print(f"  Input std: {norm_stats_dict['input_std']}")
        print(f"  Target mean: {norm_stats_dict['target_mean']}")
        print(f"  Target std: {norm_stats_dict['target_std']}")

        # Initialize GRU hidden state
        num_layers = self.trained_gru_num_layers
        hidden_dim = self.trained_gru_hidden_dim
        
        self.sea_hidden_state = torch.zeros(
            num_layers, 
            self._num_envs * self.num_joints, 
            hidden_dim, 
            device=self._device
        )
        self.prev_torque = torch.zeros(self._num_envs, self.num_joints, device=self._device)

        # Debug logging
        self.debug_step = 0
        self.debug_log = []
        self.max_debug_steps = 1200  # 5 seconds at 240 Hz

    def reset(self, env_ids: list[int] | slice | None = None):
        if env_ids is None:
            self.sea_hidden_state.zero_()
            self.prev_torque.zero_()
            # Reset debug logging
            self.debug_step = 0
            self.debug_log = []
            return
        
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self._num_envs)
            env_list = list(range(start, stop, step))
            env_indices = torch.tensor(env_list, device=self._device, dtype=torch.long)
        else:
            env_indices = torch.tensor(env_ids, device=self._device, dtype=torch.long).unique()
        
        if env_indices.numel() == 0:
            return
            
        joint_indices = torch.arange(self.num_joints, device=self._device, dtype=torch.long)
        batch_indices = env_indices.unsqueeze(1) * self.num_joints + joint_indices.unsqueeze(0)
        resolved = batch_indices.flatten()
        
        with torch.no_grad():
            if resolved.numel() > 0:
                self.sea_hidden_state[:, resolved, :] = 0.0
                self.prev_torque[env_indices, :] = 0.0

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        
        target_pos = control_action.joint_positions
        
        # Prepare raw features (as Isaac Lab provides them)
        current_angle_flat = joint_pos.flatten()
        target_angle_flat = target_pos.flatten()
        current_ang_vel_flat = joint_vel.flatten()
        prev_torque_flat = self.prev_torque.flatten()
        
        # Log debug info for first environment and joint
        if self.debug_step < self.max_debug_steps:
            debug_info = {
                'step': self.debug_step,
                'raw_angle': float(current_angle_flat[0]),
                'raw_target': float(target_angle_flat[0]),
                'raw_vel': float(current_ang_vel_flat[0]),
                'raw_prev_torque': float(prev_torque_flat[0]),
            }
            
            # Stack features for GRU input
            current_step_features_flat = torch.stack(
                [current_angle_flat, target_angle_flat, current_ang_vel_flat, prev_torque_flat], dim=1
            )
            current_step_gru_input = current_step_features_flat.unsqueeze(1)
            
            # Log raw input before normalization
            debug_info.update({
                'input_before_norm': current_step_gru_input[0, 0, :].cpu().numpy().tolist(),
            })
            
            # Normalize
            normalized_gru_input = (current_step_gru_input - self.input_mean) / (self.input_std + 1e-7)
            
            # Log normalized input
            debug_info.update({
                'input_after_norm': normalized_gru_input[0, 0, :].cpu().numpy().tolist(),
            })
            
            self._joint_vel[:] = joint_vel
            
            # Forward pass
            with torch.inference_mode():
                nn_output_normalized_flat, next_hidden_state = self.network(
                    normalized_gru_input, self.sea_hidden_state
                )
                self.sea_hidden_state[:] = next_hidden_state
            
            # Denormalize output
            nn_output_denormalized_flat = nn_output_normalized_flat * (self.target_std + 1e-7) + self.target_mean
            nn_output_denormalized = nn_output_denormalized_flat.reshape(self._num_envs, self.num_joints)
            
            # Log outputs
            debug_info.update({
                'nn_output_normalized': float(nn_output_normalized_flat[0]),
                'nn_output_denormalized': float(nn_output_denormalized[0, 0]),
            })
            
            # Since use_residual is False, the combined torque is just the NN output
            if self.use_residual_model:
                # This shouldn't happen based on config, but just in case
                analytical_torque = self.stiffness * (target_pos - joint_pos) + self.damping * (-joint_vel)
                combined_torque = analytical_torque + nn_output_denormalized
                debug_info['analytical_torque'] = float(analytical_torque[0, 0])
            else:
                combined_torque = nn_output_denormalized
                debug_info['analytical_torque'] = 0.0
            
            self.computed_effort = combined_torque
            self.applied_effort = self._clip_effort(self.computed_effort)
            
            debug_info.update({
                'computed_torque': float(self.computed_effort[0, 0]),
                'applied_torque': float(self.applied_effort[0, 0]),
            })
            
            self.debug_log.append(debug_info)
            self.debug_step += 1
            
            # Save debug log when we reach max steps
            if self.debug_step == self.max_debug_steps:
                self._save_debug_log()
        else:
            # Normal operation without logging
            current_step_features_flat = torch.stack(
                [current_angle_flat, target_angle_flat, current_ang_vel_flat, prev_torque_flat], dim=1
            )
            current_step_gru_input = current_step_features_flat.unsqueeze(1)
            normalized_gru_input = (current_step_gru_input - self.input_mean) / (self.input_std + 1e-7)
            
            self._joint_vel[:] = joint_vel
            
            with torch.inference_mode():
                nn_output_normalized_flat, next_hidden_state = self.network(
                    normalized_gru_input, self.sea_hidden_state
                )
                self.sea_hidden_state[:] = next_hidden_state
            
            nn_output_denormalized_flat = nn_output_normalized_flat * (self.target_std + 1e-7) + self.target_mean
            nn_output_denormalized = nn_output_denormalized_flat.reshape(self._num_envs, self.num_joints)
            
            if self.use_residual_model:
                analytical_torque = self.stiffness * (target_pos - joint_pos) + self.damping * (-joint_vel)
                combined_torque = analytical_torque + nn_output_denormalized
            else:
                combined_torque = nn_output_denormalized
            
            self.computed_effort = combined_torque
            self.applied_effort = self._clip_effort(self.computed_effort)
        
        # Update previous torque for next step
        self.prev_torque[:] = self.applied_effort
        
        output_actions = ArticulationActions(joint_efforts=self.applied_effort)
        return output_actions
    
    def _save_debug_log(self):
        """Save debug log to file for analysis."""
        log_path = "debug_gru_inputs_outputs.json"
        with open(log_path, 'w') as f:
            json.dump(self.debug_log, f, indent=2)
        print(f"Saved debug log to {log_path}")
        
        # Also create a summary analysis
        import numpy as np
        log_array = np.array([[entry[key] for key in ['raw_angle', 'raw_target', 'raw_vel', 'raw_prev_torque']] 
                              for entry in self.debug_log])
        
        print("\nInput Analysis (first 5 seconds):")
        print(f"Raw angle    - mean: {np.mean(log_array[:, 0]):.4f}, std: {np.std(log_array[:, 0]):.4f}, range: [{np.min(log_array[:, 0]):.4f}, {np.max(log_array[:, 0]):.4f}]")
        print(f"Raw target   - mean: {np.mean(log_array[:, 1]):.4f}, std: {np.std(log_array[:, 1]):.4f}, range: [{np.min(log_array[:, 1]):.4f}, {np.max(log_array[:, 1]):.4f}]")
        print(f"Raw velocity - mean: {np.mean(log_array[:, 2]):.4f}, std: {np.std(log_array[:, 2]):.4f}, range: [{np.min(log_array[:, 2]):.4f}, {np.max(log_array[:, 2]):.4f}]")
        print(f"Raw prev_torque - mean: {np.mean(log_array[:, 3]):.4f}, std: {np.std(log_array[:, 3]):.4f}, range: [{np.min(log_array[:, 3]):.4f}, {np.max(log_array[:, 3]):.4f}]")
        
        print("\nExpected from training:")
        print(f"Angle        - mean: 1.6253, std: 0.6615")
        print(f"Target       - mean: 1.7256, std: 0.6775") 
        print(f"Velocity     - mean: -0.0001, std: 0.0682")
        print(f"Prev torque  - mean: -0.0592, std: 1.1416")


@configclass
class GruActuatorDebugCfg(DCMotorCfg):
    """Configuration for debug GRU actuator."""
    
    class_type: Type[GruActuatorDebug] = GruActuatorDebug

    network_file: str = ""
    hydra_config_path: str = ""
    normalization_stats_path: str = ""
    
    # Default values
    stiffness: float = 0.0
    damping: float = 0.0 