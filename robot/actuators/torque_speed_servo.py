from __future__ import annotations

import torch
import os # Added for path joining
import json # For loading model summary

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, DCMotorCfg, ActuatorNetLSTMCfg
from isaaclab.actuators.actuator_pd import DCMotor
from isaaclab.actuators.actuator_net import ActuatorNetLSTM

from isaaclab.utils import configclass



class CustomServo(IdealPDActuator):
    """The motor model class."""

    cfg: CustomServoCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: CustomServoCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg = cfg
        self.log_counter = 0
        self.log_interval = 1000
        
        # Assuming that if velocity_limit and effort_limit are set, they are floats for CustomServo.
        # The base IdealPDActuatorCfg allows for dicts, which CustomServo isn't explicitly handling here.
        # This might need more robust handling if dicts are expected at this level.
        if cfg.velocity_limit is not None and isinstance(cfg.velocity_limit, float):
            assert (cfg.velocity_limit > 0.0), "The velocity limit must be positive."
        elif cfg.velocity_limit is not None: # It's a dict
            # For now, let's assume if it's a dict, this specific actuator expects it to be resolved
            # or this is a configuration error for this simplified CustomServo.
            # We will rely on IdealPDActuator to have processed it if it's a dict.
            # If a direct float is needed here, the config for CustomServo should ensure it's passed as float.
            pass # Or raise an error if CustomServo specifically needs a float and got a dict
        
        if cfg.effort_limit is not None and isinstance(cfg.effort_limit, float):
            assert (cfg.effort_limit >= 0.0), "The effort limit must be non-negative."
        elif cfg.effort_limit is not None: # It's a dict
            pass # Similar reasoning as for velocity_limit
        
        print(f"\n")
        print(f"motor armature: {self.armature}")
        print(f"motor friction: {self.friction}")
        print(f"\n")
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor, #verify that this is set correctly when called
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        
        self.log_counter += 1
        if self.log_counter % self.log_interval == 0 or self.log_counter == 1:
            print(f"control_action before ideal compute: {control_action}")
            print(f"joint_pos: {joint_pos}")
            print(f"joint_vel: {joint_vel}")
            
        ideal_pd_control_action = super().compute(control_action, joint_pos, joint_vel) 
        #TODO: In super.compute, how does it make sense for the damping term to be added rather than subtracted?
        ideal_pd_torque = ideal_pd_control_action.joint_efforts #(num_envs, num_joints)
        
        if self.log_counter % self.log_interval == 0 or self.log_counter == 1:
            print(f"control_action after ideal compute: {ideal_pd_control_action}")
        
        min_torque = torch.min(ideal_pd_torque).item()
        max_torque = torch.max(ideal_pd_torque).item()
        mean_torque = torch.mean(ideal_pd_torque).item()
        
        if self.log_counter % self.log_interval == 0 or self.log_counter == 1:
            print(f"ideal_pd_torque: {ideal_pd_torque}")
            print(f"Torque - Min: {min_torque:.3f}, Max: {max_torque:.3f}, Mean: {mean_torque:.3f}")
        control_action.joint_efforts = self._apply_torque_speed_curve(ideal_pd_torque, joint_vel) #apply the torque-speed curve to the torque
        return control_action #Only the joint_efforts are not None
    
    def _apply_torque_speed_curve(self, joint_torque: torch.Tensor, joint_vel: torch.Tensor) -> torch.Tensor:
        """
        Clip the torque based on the torque-speed curve.
        Maximum torque decreases linearly with speed from stall torque at 0 speed to 0 at velocity limit.
        Stall torque can be given in the oppsite direction of the velocity.
        Effort limit is equivalent to stall torque
        
        Args:
            joint_torque: The torque to clip (in Nm). (num_envs, num_joints)
            joint_vel: The velocity of the joint (in rad/s). (num_envs, num_joints)
            
        Returns:
            The clipped torque (in Nm). (num_envs, num_joints)
        """
        assert not torch.isnan(joint_torque).any(), "joint_torque contains NaNs"
        assert not torch.isnan(joint_vel).any(), "joint_vel contains NaNs"
        
        stall_torque = self.cfg.effort_limit
        vel_ratio = joint_vel.abs() / self.cfg.velocity_limit
        torque_multiplier = torch.clip(1.0 - vel_ratio, min=0.0, max=1.0)
        max_abs_torque_at_vel = stall_torque * torque_multiplier
        no_vel_joints = joint_vel == 0.0 #(num_envs, num_joints)
        negative_vel_joints = joint_vel < 0.0 #(num_envs, num_joints)
        positive_vel_joints = joint_vel > 0.0 #(num_envs, num_joints)
        actual_torque_tensor = torch.zeros_like(joint_torque)
        stall_torque_tensor = torch.ones_like(joint_torque) * stall_torque #(num_envs, num_joints)
        
        actual_torque_tensor[no_vel_joints] = torch.clip(
            joint_torque[no_vel_joints], 
            min=-stall_torque_tensor[no_vel_joints], 
            max=stall_torque_tensor[no_vel_joints]
        )
        actual_torque_tensor[negative_vel_joints] = torch.clip(
            joint_torque[negative_vel_joints], 
            min=-max_abs_torque_at_vel[negative_vel_joints], 
            max=stall_torque_tensor[negative_vel_joints]
        )
        actual_torque_tensor[positive_vel_joints] = torch.clip(
            joint_torque[positive_vel_joints], 
            min=-stall_torque_tensor[positive_vel_joints], 
            max=max_abs_torque_at_vel[positive_vel_joints])
        
        min_torque = torch.min(actual_torque_tensor).item()
        max_torque = torch.max(actual_torque_tensor).item()
        mean_torque = torch.mean(actual_torque_tensor).item()
        
        if self.log_counter % self.log_interval == 0 or self.log_counter == 1:
            print("\n")
            print(f"stall_torque: {stall_torque}")
            print(f"joint_torque: {joint_torque}")
            print(f"joint_vel: {joint_vel}")
            print(f"vel_ratio: {vel_ratio}")
            print(f"torque_multiplier: {torque_multiplier}")
            print(f"max_abs_torque_at_vel: {max_abs_torque_at_vel}")
            print(f"no_vel_joints: {no_vel_joints}")
            print(f"negative_vel_joints: {negative_vel_joints}")
            print(f"positive_vel_joints: {positive_vel_joints}")
            print(f"actual_torque_tensor[no_vel_joints]: {actual_torque_tensor[no_vel_joints]}")
            print(f"actual_torque_tensor[negative_vel_joints]: {actual_torque_tensor[negative_vel_joints]}")
            print(f"actual_torque_tensor[positive_vel_joints]: {actual_torque_tensor[positive_vel_joints]}")
            print(f"stall_torque_tensor: {stall_torque_tensor}")
            print(f"actual_torque_tensor {actual_torque_tensor}")
            print(f"Torque after clipping - Min: {min_torque:.3f}, Max: {max_torque:.3f}, Mean: {mean_torque:.3f}")
            print("\n")
            self.log_counter = 0
        
        return actual_torque_tensor #(num_envs, num_joints)
        
        
@configclass
class CustomServoCfg(IdealPDActuatorCfg):
    """Configuration for custom servo actuator model.
    velocity_limit is used both as the zero-torque-speed in the torque-speed curve and the velocity limit of the underlying ideal PD actuator.
    """

    class_type: type = CustomServo
