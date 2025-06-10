from __future__ import annotations

import torch

from isaaclab.utils.types import ArticulationActions

from isaaclab.actuators import IdealPDActuator
from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg

from isaaclab.utils import configclass
from dataclasses import MISSING


class TorqueSpeedServo(IdealPDActuator):
    """The motor model class."""

    cfg: TorqueSpeedServoCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: TorqueSpeedServoCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg = cfg

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor, #verify that this is set correctly when called
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        ideal_pd_control_action = super().compute(control_action, joint_pos, joint_vel) 
        ideal_pd_torque = ideal_pd_control_action.joint_efforts #(num_envs, num_joints)
        
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

        return actual_torque_tensor #(num_envs, num_joints)
        
        
@configclass
class TorqueSpeedServoCfg(IdealPDActuatorCfg):
    class_type: type = TorqueSpeedServo

class ParallelElasticActuator(TorqueSpeedServo):    
    cfg: ParallelElasticActuatorCfg

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts -= self.cfg.spring_stiffness * joint_pos
        
        assert not torch.isnan(control_action.joint_efforts).any(), "control_action.joint_efforts contains NaNs"
        
        return control_action
    
@configclass
class ParallelElasticActuatorCfg(TorqueSpeedServoCfg):
    class_type: type = ParallelElasticActuator
    spring_stiffness: float = MISSING  # The stiffness of the spring (in N/m). Must be positive.
