import os
import torch
from typing import Tuple, List

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg, Articulation
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from .motors import get_AK7010_cfg, get_AK809_cfg

DEG2RAD = torch.pi / 180.0
RPM2RADPS = 2.0 * torch.pi / 60.0


class MarsJumperRobotConfig(ArticulationCfg):
    """Configuration of mars jumper robot using cube mars motor model."""
    HIP_FLEXION_JOINTS_REGEX: str = ".*_HAA.*"
    HIP_ABDUCTION_JOINTS_REGEX: str = ".*_HFE.*"
    KNEE_JOINTS_REGEX: str = ".*_KFE.*"
    FEET_REGEX: str = ".*FOOT.*"
    AVOID_CONTACT_BODIES_REGEX: List[str] = [".*base.*", ".*HIP.*", ".*THIGH.*", ".*SHANK.*"]

    HIP_ABDUCTION_ANGLE_LIMITS_RAD: Tuple[float, float] = (90 * DEG2RAD, -90 * DEG2RAD) #TODO: Check if these are correct
    KNEE_ANGLE_LIMITS_RAD: Tuple[float, float] = (90 * DEG2RAD, -90 * DEG2RAD) #TODO: Check if these are correct
    HIP_FLEXION_ANGLE_LIMITS_RAD: Tuple[float, float] = (-90 * DEG2RAD, 90 * DEG2RAD) #TODO: Check if these are correct

    def __init__(self):
        
        abductor_angle = 0 * DEG2RAD
        flexion_angle = 0 * DEG2RAD
        knee_angle = 0 * DEG2RAD

        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3), #TODO: Check if this is correct
            joint_pos={ #TODO: Check if these are correct
            "LF_HAA": abductor_angle, 
            "LH_HAA": -abductor_angle, #postive angle is negative
            "RF_HAA": -abductor_angle, #postive angle is negative
            "RH_HAA": abductor_angle,
            "LF_HFE": flexion_angle,
            "LH_HFE": flexion_angle,
            "RF_HFE": -flexion_angle, #postive angle is negative
            "RH_HFE": flexion_angle,
            "LF_KFE": knee_angle,
            "LH_KFE": knee_angle,
            "RF_KFE": knee_angle,
            "RH_KFE": knee_angle,
        },
        )
        
        actuators = {
            # "hip_motors": get_AK809_cfg(
            #     joint_names_expr=self.HIP_FLEXION_JOINTS_REGEX, kp=0.1, kd=0.1
            # ),
            # "hip_abduction_motors": get_AK809_cfg(
            #     joint_names_expr=self.HIP_ABDUCTION_JOINTS_REGEX, kp=0.1, kd=0.1
            # ),
            # "knee_motors": get_AK809_cfg(
            #     joint_names_expr=self.KNEE_JOINTS_REGEX, kp=0.1, kd=0.1
            # ),
            # "knee_torsional_springs": IdealPDActuatorCfgg(
            #     joint_names_expr=self.KNEE_JOINTS_REGEX,
            #     effort_limit=1.0,
            #     stiffness=1.0, #TODO: Add correct values
            #     damping=10.0,
            #     friction=0.1,
            # ),
            # "hip_torsional_springs": ImplicitActuatorCfg(
            #     joint_names_expr=self.HIP_FLEXION_JOINTS_REGEX,
            #     effort_limit=1.0,
            #     stiffness=1.0, #TODO: Add correct values
            #     damping=10.0,
            #     friction=0.1,
            # ),
            "placeholder_knee_motors": ImplicitActuatorCfg(
                joint_names_expr=self.KNEE_JOINTS_REGEX,
                effort_limit=100,
                velocity_limit=100, #TODO: Add correct values
                stiffness=5.0, #TODO: Add correct values
                damping=0.1,
                friction=0.1,
            ),
            "placeholder_hip_motors": ImplicitActuatorCfg(
                joint_names_expr=self.HIP_FLEXION_JOINTS_REGEX,
                effort_limit=100,
                velocity_limit=100, #TODO: Add correct values
                stiffness=5.0, #TODO: Add correct values
                damping=0.1,
                friction=0.1,
            ),
            "placeholder_hip_abduction_motors": ImplicitActuatorCfg(
                joint_names_expr=self.HIP_ABDUCTION_JOINTS_REGEX,
                effort_limit=100,
                velocity_limit=100, #TODO: Add correct values
                stiffness=5.0, #TODO: Add correct values
                damping=0.1,
                friction=0.1,
            ),
        }
    
        
        spawn=sim_utils.UsdFileCfg(
                usd_path=f"{os.getcwd()}/submodules/cad/simplified_robot/simplified_robot.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0, #TODO: Check if these are correct
                    angular_damping=0.0,
                    max_linear_velocity=100.0,
                    max_angular_velocity=100.0,
                    max_depenetration_velocity=1.0, 
                ),
                # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                #     enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
                # ),
        )
        
        super().__init__(
            prim_path="/World/envs/env_.*/robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
            soft_joint_pos_limit_factor=1.0, #add softening to joint limits to reduce impact of hard limits
        )
