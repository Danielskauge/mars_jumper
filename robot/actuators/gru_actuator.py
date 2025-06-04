from __future__ import annotations

import torch
import os # Added for path joining
import json # For loading model summary

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, DCMotorCfg, ActuatorNetLSTMCfg
from isaaclab.actuators.actuator_pd import DCMotor
from isaaclab.actuators.actuator_net import ActuatorNetLSTM

from dataclasses import MISSING, field
from isaaclab.utils import configclass

from typing import Type, cast, Any, Union, Optional # Added Union, Optional
from isaaclab.utils.assets import read_file


class GruActuator(DCMotor):
    """
    Actuator using a GRU neural network to predict torque.
    Either in full mode, where only the GRU outputs the full torque, or in residual mode, for when the GRU is used to predict the residual torque from an analytical controller, in which case the analytical torque (same as used in training) is added to the GRU torque.
    The analytical controller is pd controller, which optionally can be saturated with a torque-speed curve, and can also have a spring component.
    Inherits from DCMotor for its clipping behavior and base motor parameters.
    GRU network loading, state management, input normalization, and output denormalization are handled herein.
    """

    cfg: GruActuatorCfg
    network: torch.jit.ScriptModule
    sea_hidden_state: torch.Tensor

    # Normalization tensors
    input_mean: torch.Tensor
    input_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    # Internal storage for spring params if loaded from summary for residual model
    _k_spring_actual: float = 0.0
    _theta0_actual: float = 0.0
    _use_internal_spring_for_residual_basis: bool = False

    def __init__(self, cfg: GruActuatorCfg, *args, **kwargs):
        # Initialize DCMotor part first. Stiffness/damping from cfg will be used by DCMotor if apply_pd_control is true.
        # If use_residual_model is true, we might overwrite these later.
        super().__init__(cfg, *args, **kwargs)

        hydra_config_path = self.cfg.hydra_config_path
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to parse YAML model summary file.")
        try:
            with open(hydra_config_path, 'r') as f:
                hydra_config_data = yaml.safe_load(f)
                hydra_config = hydra_config_data.get('model', hydra_config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Hydra config file not found: {hydra_config_path}")
        except Exception as e:
            raise ValueError(f"Error parsing Hydra config file: {hydra_config_path}: {e}")

        self.use_residual_model = hydra_config.get("use_residual", False)

        # --- Read GRU architecture from hydra config if present ---
        summary_gru_num_layers = hydra_config.get("gru_num_layers")
        summary_gru_hidden_dim = hydra_config.get("gru_hidden_dim")
        self.trained_gru_num_layers = summary_gru_num_layers
        self.trained_gru_hidden_dim = summary_gru_hidden_dim
        print(f"  Actuator configured with trained GRU num_layers: {self.trained_gru_num_layers}, hidden_dim: {self.trained_gru_hidden_dim}")

        # Load the GRU network (network_file is still direct from cfg)
        file_bytes = read_file(self.cfg.network_file)
        self.network = torch.jit.load(file_bytes, map_location=self._device).eval()

        # Flatten RNN parameters for potential performance improvement and to avoid warnings
        # Access the underlying GRU module, which is nested as self.network.model.gru
        if hasattr(self.network, 'model') and hasattr(self.network.model, 'gru') and hasattr(self.network.model.gru, 'flatten_parameters'):
            print("Attempting to flatten parameters of GRU submodule...")
            self.network.model.gru.flatten_parameters()
            print("GRU submodule parameters flattened.")
        elif hasattr(self.network, 'flatten_parameters'): # Fallback for simpler JIT models
            print("Attempting to flatten parameters of top-level JIT model...")
            self.network.flatten_parameters()
            print("Top-level JIT model parameters flattened.")
        else:
            print("Warning: Could not find flatten_parameters method on JIT model or its GRU submodule.")

        # Use GRU parameters (potentially updated from summary for sequence_length) for buffers
        #num_layers = self.cfg.gru_num_layers # Assuming these are correctly set in cfg to match model arch
        #hidden_dim = self.cfg.gru_hidden_dim # Assuming these are correctly set in cfg
        num_layers = self.trained_gru_num_layers
        hidden_dim = self.trained_gru_hidden_dim

        # Load normalization statistics from JSON file
        stats_file_path = self.cfg.normalization_stats_path
        try:
            with open(stats_file_path, 'r') as f:
                norm_stats_dict = json.load(f)
            # Convert lists to tensors and reshape for broadcasting
            # Input stats: expected shape (1, 1, input_dim) for (batch, seq, feature) normalization
            # Target stats: expected shape (1,) or (1,1) for (batch, feature) or scalar target
            self.input_mean = torch.tensor(norm_stats_dict["input_mean"], device=self._device, dtype=torch.float32).view(1, 1, -1)
            self.input_std = torch.tensor(norm_stats_dict["input_std"], device=self._device, dtype=torch.float32).view(1, 1, -1)
            self.target_mean = torch.tensor(norm_stats_dict["target_mean"], device=self._device, dtype=torch.float32).squeeze() # Squeeze to make it scalar-like if it's [1]
            self.target_std = torch.tensor(norm_stats_dict["target_std"], device=self._device, dtype=torch.float32).squeeze()   # Squeeze
            # Add a small epsilon to std to prevent division by zero if not already handled
            self.input_std[self.input_std < 1e-7] = 1.0 # Or add epsilon: self.input_std + 1e-7
            if self.target_std.ndim == 0 and self.target_std < 1e-7: # Scalar check
                 self.target_std = torch.tensor(1.0, device=self._device, dtype=torch.float32)
            elif self.target_std.ndim > 0: # Tensor check
                self.target_std[self.target_std < 1e-7] = 1.0

            print(f"Loaded and processed normalization stats from {stats_file_path}")
            print(f"  Input Mean shape: {self.input_mean.shape}, Std shape: {self.input_std.shape}")
            print(f"  Target Mean shape: {self.target_mean.shape}, Std shape: {self.target_std.shape}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Normalization statistics file not found: {stats_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from normalization statistics file: {stats_file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading or processing normalization statistics from {stats_file_path}: {e}")

        # If using a residual model, override PD parameters with those from training summary
        # These will define the PD base for the residual calculation.
        if self.use_residual_model:
            print("Using residual model. Loading PD parameters from Hydra config.")
            kp_phys = hydra_config.get("kp_phys")
            kd_phys = hydra_config.get("kd_phys")
            k_spring = hydra_config.get("k_spring", 0.0)
            theta0 = hydra_config.get("theta0", 0.0)
            
            if kp_phys is None or kd_phys is None:
                raise ValueError("kp_phys and kd_phys must be in Hydra config when use_residual_model is True.")

            num_actuated_joints = self.num_joints # from ActuatorBase
            self.stiffness = torch.full((1, num_actuated_joints), kp_phys, device=self._device, dtype=torch.float32)
            self.damping = torch.full((1, num_actuated_joints), kd_phys, device=self._device, dtype=torch.float32)

            self._k_spring_actual = k_spring
            self._theta0_actual = theta0
            self._use_internal_spring_for_residual_basis = True

            # Check for torque-speed curve parameters for the residual's PD base from Hydra config
            pd_stall_phys = hydra_config.get("pd_stall_torque_phys_training")
            pd_nls_phys = hydra_config.get("pd_no_load_speed_phys_training")
            
            # Store TSC parameters as instance variables if they exist
            self._pd_has_tsc = False
            if pd_stall_phys is not None and pd_nls_phys is not None and pd_nls_phys > 0:
                print(f"Applying training TSC to residual PD base: stall={pd_stall_phys}, nls={pd_nls_phys}")
                self._pd_has_tsc = True
                self._pd_stall_torque = pd_stall_phys
                self._pd_no_load_speed = pd_nls_phys
            else:
                print("No TSC from Hydra config for residual PD base, or params invalid. TSC for PD part is OFF.")
        else:
            print("Not using residual model. No PD controller will be active.")
            # In non-residual mode, ensure no PD components are active
            self._pd_has_tsc = False

        # Create buffers for GRU hidden states. The input is now single-step, so no sea_input sequence buffer.
        # num_layers and hidden_dim for hidden state buffer must match the trained model.
        # Effective batch size for hidden state is num_envs * num_joints
        self.sea_hidden_state = torch.zeros(
            num_layers, 
            self._num_envs * self.num_joints, 
            hidden_dim, 
            device=self._device
        )
        # self.sea_hidden_state_per_env view is removed as direct indexing to sea_hidden_state will be used for reset.

        # History buffers are no longer needed for constructing GRU input sequence, as GRU is now stateful over single steps.
        # self._history_current_angle = torch.zeros(self._num_envs, self.trained_gru_sequence_length, self.num_joints, device=self._device)
        # self._history_target_angle = torch.zeros(self._num_envs, self.trained_gru_sequence_length, self.num_joints, device=self._device)
        # self._history_current_ang_vel = torch.zeros(self._num_envs, self.trained_gru_sequence_length, self.num_joints, device=self._device)
        
        # Final validation - only needed for residual mode since non-residual mode has no PD
        if self.use_residual_model:
            if not isinstance(self.stiffness, torch.Tensor) or self.stiffness.shape[-1] != self.num_joints:
                raise ValueError(f"Stiffness tensor is not correctly initialized for {self.num_joints} joints in residual mode.")
            if not isinstance(self.damping, torch.Tensor) or self.damping.shape[-1] != self.num_joints:
                raise ValueError(f"Damping tensor is not correctly initialized for {self.num_joints} joints in residual mode.")

            if self._pd_has_tsc:
                if self._pd_stall_torque is None or self._pd_no_load_speed is None or self._pd_no_load_speed <= 0:
                    raise ValueError("Invalid TSC parameters for residual PD base.")

    def reset(self, env_ids: list[int] | slice | None = None):
        """Resets the GRU hidden states for the specified environments."""
        # Full reset if no selector provided
        if env_ids is None:
            self.sea_hidden_state.zero_()
            return
        # Build tensor of environment indices to reset
        if isinstance(env_ids, slice):
            start, stop, step = env_ids.indices(self._num_envs)
            env_list = list(range(start, stop, step))
            env_indices = torch.tensor(env_list, device=self._device, dtype=torch.long)
        elif isinstance(env_ids, (list, tuple, torch.Tensor)):
            env_indices = torch.tensor(env_ids, device=self._device, dtype=torch.long).unique()
        else:
            raise TypeError(f"Unsupported env_ids type {type(env_ids)}. Must be None, slice, list/tuple of ints, or Tensor.")
        if env_indices.numel() == 0:
            return
        # Map to hidden-state batch dimension indices
        joint_indices = torch.arange(self.num_joints, device=self._device, dtype=torch.long)
        batch_indices = env_indices.unsqueeze(1) * self.num_joints + joint_indices.unsqueeze(0)
        resolved = batch_indices.flatten()
        with torch.no_grad():
            if resolved.numel() > 0:
                self.sea_hidden_state[:, resolved, :] = 0.0

    def _apply_pd_torque_speed_curve(self, pd_joint_torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """
        Clips the PD torque based on an asymmetrical torque-speed curve.
        Used only in residual mode when TSC parameters are available from training config.
        """
        if not self._pd_has_tsc:
            print("Warning: _apply_pd_torque_speed_curve called but no TSC parameters available.")
            return pd_joint_torque
            
        stall_torque = self._pd_stall_torque
        no_load_speed = self._pd_no_load_speed
        
        vel_ratio = joint_vel.abs() / no_load_speed 
        torque_multiplier = torch.clip(1.0 - vel_ratio, min=0.0, max=1.0)
        max_torque_in_vel_direction = stall_torque * torque_multiplier

        saturated_torque = pd_joint_torque.clone()

        positive_vel_mask = joint_vel > 0
        if torch.any(positive_vel_mask):
            saturated_torque[positive_vel_mask] = torch.clip(
                pd_joint_torque[positive_vel_mask],
                min=-stall_torque,
                max=max_torque_in_vel_direction[positive_vel_mask]
            )

        negative_vel_mask = joint_vel < 0
        if torch.any(negative_vel_mask):
            saturated_torque[negative_vel_mask] = torch.clip(
                pd_joint_torque[negative_vel_mask],
                min=-max_torque_in_vel_direction[negative_vel_mask], 
                max=stall_torque 
            )
        
        zero_vel_mask = joint_vel == 0
        if torch.any(zero_vel_mask):
            saturated_torque[zero_vel_mask] = torch.clip(
                pd_joint_torque[zero_vel_mask],
                min=-stall_torque,
                max=stall_torque
            )
            
        return saturated_torque
    
    def _compute_analytical_torque(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the analytical torque from the PD controller and spring.
        """
        error_pos = target_pos - joint_pos
        error_vel = - joint_vel
        
        pd_torque = self.stiffness * error_pos + self.damping * error_vel
        
        if self._pd_has_tsc:
            pd_torque = self._apply_pd_torque_speed_curve(pd_torque, joint_vel)
        
        spring_torque_component = torch.zeros_like(joint_pos)
        if self._use_internal_spring_for_residual_basis:
            spring_torque_component = self._k_spring_actual * (joint_pos - self._theta0_actual)
        
        analytical_torque = pd_torque + spring_torque_component
        return analytical_torque

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        
        target_pos = control_action.joint_positions 
        
        # Prepare GRU input for current time step
        current_angle_flat = joint_pos.flatten()
        target_angle_flat = target_pos.flatten()
        current_ang_vel_flat = joint_vel.flatten()
        
        current_step_features_flat = torch.stack(
            [current_angle_flat, target_angle_flat, current_ang_vel_flat], dim=1
        )
        
        current_step_gru_input = current_step_features_flat.unsqueeze(1)
        
        # Normalize the input
        normalized_gru_input = (current_step_gru_input - self.input_mean) / (self.input_std + 1e-7)
        
        self._joint_vel[:] = joint_vel # For DCMotor base class

        # Forward pass through GRU
        with torch.inference_mode():
            nn_output_normalized_flat, next_hidden_state = self.network(
                normalized_gru_input, self.sea_hidden_state
            )
            self.sea_hidden_state[:] = next_hidden_state

        # Denormalize NN output
        nn_output_denormalized_flat = nn_output_normalized_flat * (self.target_std + 1e-7) + self.target_mean
        nn_output_denormalized = nn_output_denormalized_flat.reshape(self._num_envs, self.num_joints)

        if self.use_residual_model:
            analytical_torque = self._compute_analytical_torque(joint_pos, joint_vel, target_pos)
            combined_torque = analytical_torque + nn_output_denormalized
        else:
            combined_torque = nn_output_denormalized

        self.computed_effort = combined_torque 
        self.applied_effort = self._clip_effort(self.computed_effort)

        output_actions = ArticulationActions(joint_efforts=self.applied_effort)
        return output_actions
    
    
@configclass
class GruActuatorCfg(DCMotorCfg):
    """Configuration for an actuator using a GRU neural network.
    
    Two modes:
    1. Residual mode: GRU predicts residual from a PD controller used during training.
       PD parameters are loaded from training config and cannot be overridden.
    2. Non-residual mode: GRU predicts full torque directly. No PD controller.
    """
    
    class_type: Type[GruActuator] = GruActuator

    # --- Files produced by training pipeline ---
    network_file: str = MISSING # type: ignore # Path to the TorchScript (.pt) model file
    hydra_config_path: str = MISSING # type: ignore # Path to Hydra config file (from training)
    normalization_stats_path: str = MISSING # type: ignore # Path to normalization_stats.json (from training)

    # Default PD parameters for non-residual mode or if not specified in hydra_config for residual mode (though residual mode overrides).
    # These are added to satisfy the configclass validation inherited from IdealPDActuatorCfg via DCMotorCfg.
    # For non-residual mode, these are not used by the GruActuator's compute logic.
    # For residual mode, GruActuator.__init__ will overwrite self.stiffness and self.damping with values from hydra_config.
    stiffness: float = 0.0
    damping: float = 0.0
    
    def __post_init__(self):
        """Post-initialization checks."""
        if self.network_file is MISSING:
            raise ValueError("'network_file' must be provided.")
        if self.hydra_config_path is MISSING:
            raise ValueError("'hydra_config_path' must be provided.")
        if self.normalization_stats_path is MISSING:
            raise ValueError("'normalization_stats_path' must be provided.")
