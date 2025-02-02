from __future__ import annotations

import torch

from omni.isaac.core.utils.types import ArticulationActions

from omni.isaac.lab.actuators import IdealPDActuator
from omni.isaac.lab.actuators import IdealPDActuatorCfg

from dataclasses import MISSING
from omni.isaac.lab.utils import configclass


class MotorModel(IdealPDActuator):
    """The motor model class."""

    cfg: MotorModelCfg
    """The configuration for the actuator model."""

    def __init__(self, cfg: MotorModelCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg = cfg
        # parse configuration
        if self.cfg.cutoff_speed is not None:
            self._cutoff_speed = self.cfg.cutoff_speed
        else:
            self._cutoff_speed = torch.inf
        # prepare joint vel buffer for max effort computation
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # create buffer for zeros effort
        self._zeros_effort = torch.zeros_like(self.computed_effort)
        # check that quantities are provided
        if self.cfg.velocity_limit is None:
            raise ValueError(
                "The velocity limit must be provided for the  motor actuator model."
            )

        assert (
            self.cfg.velocity_limit > self._cutoff_speed
        ), "The cutoff speed must be less than the velocity limit."

        assert (
            self.cfg.velocity_limit >= 0.0
        ), "The velocity limit  must be non-negative."
        assert self.cfg.effort_limit >= 0.0, "The effort limit  must be non-negative."
        assert self.cfg.cutoff_speed >= 0.0, "The cutoff speed must be non-negative."

    """
    Operations.
    """

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        # save current joint vel
        self._joint_vel[:] = joint_vel
        # calculate the desired joint torques
        return super().compute(control_action, joint_pos, joint_vel)

    """
    Helper functions.
    """

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # compute torque limits
        # -- max limit
        abs_effort_limit = (
            self.effort_limit
            * (
                1.0
                - (self._joint_vel.abs() - self._cutoff_speed)
                / (self.velocity_limit - self._cutoff_speed)
            )
        ).clip(min=self._zeros_effort, max=self.effort_limit)

        # clip the torques based on the motor limits
        return torch.clip(effort, min=-abs_effort_limit, max=abs_effort_limit)


@configclass
class MotorModelCfg(IdealPDActuatorCfg):
    """Configuration for direct control (DC) motor actuator model."""

    class_type: type = MotorModel

    cutoff_speed: float = MISSING
    """The speed at which the motor torque starts to saturate (in rad/s)."""
