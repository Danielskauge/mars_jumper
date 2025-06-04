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
from torque_speed_servo import CustomServo, CustomServoCfg


class ParallelElasticActuator(CustomServo):
    """The parallell elastic actuator class."""
    
    cfg: ParallelElasticActuatorCfg
    """The configuration for the actuator model."""
    
    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        control_action = super().compute(control_action, joint_pos, joint_vel)
        control_action.joint_efforts = control_action.joint_efforts + self.cfg.spring_stiffness * (self.cfg.spring_equilibrium_angle - joint_pos)
        return control_action
    
class ParallelElasticActuatorCfg(CustomServoCfg):
    class_type: type = ParallelElasticActuator
    spring_stiffness: float # The stiffness of the spring (in N/m). Must be positive.
    spring_equilibrium_angle: float # The angle of the equilibrium point (in rad).

    def __post_init__(self):
        # CustomServoCfg does not have a __post_init__, so no super() call needed here.
        if self.spring_stiffness is MISSING or self.spring_stiffness <= 0:
            raise ValueError("'spring_stiffness' must be provided and be positive.")
        if self.spring_equilibrium_angle is MISSING:
            raise ValueError("'spring_equilibrium_angle' must be provided.")

