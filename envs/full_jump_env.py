import numpy as np
from envs.full_jump_env_cfg import FullJumpEnvCfg
from envs.jump_metrics import JumpMetrics
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

        # Initialize metrics system
        self.metrics = JumpMetrics(self.num_envs, self.device, cfg)
        
        #Jump Phase
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
        return self.metrics._relative_takeoff_error(env_ids, self)
    
    def takeoff_angle_error(self, env_ids: Sequence[int]) -> torch.Tensor:
        return self.metrics._takeoff_angle_error(env_ids, self)

    def _takeoff_success(self, env_ids: Sequence[int]) -> dict | None:
        return self.metrics.calculate_takeoff_success(env_ids, self, self.jump_phase)
        
    def _calculate_flight_success(self, env_ids: Sequence[int]) -> dict | None:
        return self.metrics.calculate_flight_success(env_ids, self.jump_phase)
            
    def _calculate_landing_success(self, env_ids: Sequence[int]) -> dict | None:
        return self.metrics.calculate_landing_success(env_ids, self.jump_phase, self.termination_manager)

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
        
        # Update bucketed takeoff errors via metrics system
        bucketed_logs = self.metrics.log_bucketed_errors()
        self.extras["log"].update(bucketed_logs)

        if torch.any(self.takeoff_mask):
            current_vel_vec = get_center_of_mass_lin_vel(self) # Shape: (num_envs, 3)
            current_vel_magnitude = torch.norm(current_vel_vec, dim=-1) # Shape: (num_envs,)
            
            update_mask = self.takeoff_mask & (current_vel_magnitude > self.metrics.max_takeoff_vel_magnitude) # Shape: (num_envs,)
            
            self.metrics.max_takeoff_vel_magnitude[update_mask] = current_vel_magnitude[update_mask]
            self.metrics.max_takeoff_vel[update_mask] = current_vel_vec[update_mask]

        if torch.any(self.takeoff_to_flight_mask):
            self.metrics.takeoff_relative_error[self.takeoff_to_flight_mask] = self.relative_takeoff_error(self.takeoff_to_flight_mask)
        
        # Track maximum height achieved during flight
        if torch.any(self.flight_mask):
            current_com_pos = get_center_of_mass_pos(self)
            current_heights = current_com_pos[:, 2] - self.metrics.start_com_pos[:, 2]  # Height above starting position
            flight_envs_mask = self.flight_mask
            update_height_mask = flight_envs_mask & (current_heights > self.metrics.max_height_achieved)
            self.metrics.max_height_achieved[update_height_mask] = current_heights[update_height_mask]
        
        if torch.any(self.flight_to_landing_mask):
            self.metrics.angle_error_at_landing[self.flight_to_landing_mask] = self._abs_angle_error(self.robot.data.root_quat_w[self.flight_to_landing_mask])
            self.metrics.body_ang_vel_at_landing[self.flight_to_landing_mask] = self.robot.data.root_ang_vel_w[self.flight_to_landing_mask]
            
            # Calculate error metrics at landing transition
            landing_env_ids = self.flight_to_landing_mask.nonzero(as_tuple=False).squeeze(-1)
            if landing_env_ids.numel() > 0:
                # Length error at landing
                current_com_pos = get_center_of_mass_pos(self)[landing_env_ids]
                horizontal_distance = torch.norm(current_com_pos[:, :2] - self.metrics.start_com_pos[landing_env_ids, :2], dim=-1)
                self.metrics.length_error_at_landing[landing_env_ids] = horizontal_distance - self.target_length[landing_env_ids]
                
                # Height error peak (difference between max height achieved and target)
                self.metrics.height_error_peak[landing_env_ids] = self.metrics.max_height_achieved[landing_env_ids] - self.target_height[landing_env_ids]
        
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

    def _reset_idx(self, env_ids: Sequence[int]):
        if len(env_ids) > 0:
            # Update bucketed metrics before reset
            self.metrics.update_bucketed_metrics(env_ids, self, self.jump_phase)

            # Calculate success metrics
            takeoff_log_data = self._takeoff_success(env_ids)
            flight_log_data = self._calculate_flight_success(env_ids)
            landing_log_data = self._calculate_landing_success(env_ids)
            
            # Calculate length error at termination for environments that terminate in landing phase
            terminating_landing_mask = torch.tensor([i in env_ids for i in range(self.num_envs)], device=self.device) & (self.jump_phase == Phase.LANDING)
            if torch.any(terminating_landing_mask):
                terminating_landing_ids = terminating_landing_mask.nonzero(as_tuple=False).squeeze(-1)
                current_com_pos = get_center_of_mass_pos(self)[terminating_landing_ids]
                horizontal_distance = torch.norm(current_com_pos[:, :2] - self.metrics.start_com_pos[terminating_landing_ids, :2], dim=-1)
                self.metrics.length_error_at_termination[terminating_landing_ids] = horizontal_distance - self.target_length[terminating_landing_ids]
            
            # Calculate error metrics using the metrics system (includes height_error_peak)
            error_log_data = self.metrics.calculate_error_metrics(env_ids, self.jump_phase, self.prev_jump_phase, self)
            
            # Update running success rates
            self.metrics.update_running_success_rates(env_ids)
            
            # Sample new commands before calling super()._reset_idx
            self.target_height[env_ids], self.target_length[env_ids] = sample_command(self, env_ids)
    
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            
            # Log all the calculated data
            self.extras["log"].update(flight_log_data)
            self.extras["log"].update(takeoff_log_data)
            self.extras["log"].update(landing_log_data)
            self.extras["log"].update(error_log_data)
            
            self.extras["log"].update({
                "full_jump_success_rate": self.metrics.full_jump_success_rate,
                "running_success_rate": self.metrics.success_rate,
            })

            # Update contact state
            current_contacts_for_reset_envs = self._get_current_feet_contact_state()
            self.prev_feet_contact_state[env_ids] = current_contacts_for_reset_envs[env_ids]

            # Reset metrics and phase state
            self.metrics.reset_env_metrics(env_ids, self)
            self.jump_phase[env_ids] = Phase.TAKEOFF
            self.prev_jump_phase[env_ids] = Phase.TAKEOFF
            
            # Reset phase masks
            self.takeoff_mask[env_ids] = True
            self.flight_mask[env_ids] = False
            self.landing_mask[env_ids] = False