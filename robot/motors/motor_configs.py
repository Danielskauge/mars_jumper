from torch import pi
from .motor_model import MotorModelCfg


RPM2RADPS = 2.0 * pi / 60.0


def get_AK7010_cfg(
    joint_names_expr: list[str], kp: float, kd: float
) -> MotorModelCfg:
    return MotorModelCfg(
        joint_names_expr=joint_names_expr,
        effort_limit=24.8,
        velocity_limit=382 * RPM2RADPS,
        cutoff_speed=225 * RPM2RADPS,
        stiffness={".*": kp},
        damping={".*": kd},
        friction={".*": 0.00},
    )


def get_AK809_cfg(
    joint_names_expr: list[str], kp: float, kd: float
) -> MotorModelCfg:
    return MotorModelCfg(
        joint_names_expr=joint_names_expr, #TODO: arent both joint names and joint ids needed?
        effort_limit=20.0,
        velocity_limit=470 * RPM2RADPS,
        cutoff_speed=335 * RPM2RADPS,
        stiffness={".*": kp},
        damping={".*": kd},
        friction={".*": 0.00},
    )
