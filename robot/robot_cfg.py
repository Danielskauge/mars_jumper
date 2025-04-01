import os
import torch
from typing import Tuple, List

from isaaclab.actuators.actuator_cfg import IdealPDActuatorCfg, ImplicitActuatorCfg, ActuatorNetLSTMCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from .actuators import CustomServo, CustomServoCfg, ParallelElasticActuator, ParallelElasticActuatorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

DEG2RAD = torch.pi / 180.0
RPM2RADPS = 2.0 * torch.pi / 60.0

@configclass
class MarsJumperRobotCfg(ArticulationCfg):
    """Configuration of mars jumper robot using cube mars motor model."""
    def __init__(self):
        
        self.HIP_FLEXION_JOINTS_REGEX: str = ".*_HAA.*"
        self.HIP_ABDUCTION_JOINTS_REGEX: str = ".*_HFE.*"
        self.KNEE_JOINTS_REGEX: str = ".*_KFE.*"
        self.FEET_REGEX: str = ".*FOOT.*"
        self.AVOID_CONTACT_BODIES_REGEX: List[str] = [".*base.*", ".*HIP.*", ".*THIGH.*", ".*SHANK.*"]

        self.HIP_ABDUCTION_ANGLE_LIMITS_RAD: Tuple[float, float] = (0, 90 * DEG2RAD) 
        self.KNEE_ANGLE_LIMITS_RAD: Tuple[float, float] = (90 * DEG2RAD, -90 * DEG2RAD) 
        self.HIP_FLEXION_ANGLE_LIMITS_RAD: Tuple[float, float] = (-90 * DEG2RAD, 90 * DEG2RAD)
            
        self.HIP_LINK_LENGTH: float = 0.11
        self.KNEE_LINK_LENGTH: float = 0.11
        
        abductor_angle = 0 * DEG2RAD
        hip_angle = -70 * DEG2RAD
        knee_angle = 140 * DEG2RAD
        init_height = 0.08

        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, init_height),
            joint_pos={ #TODO: Check if these are correct
            "LF_HAA": abductor_angle, 
            "LH_HAA": abductor_angle, #postive angle is negative
            "RF_HAA": abductor_angle, #postive angle is negative
            "RH_HAA": abductor_angle,
            "LF_HFE": hip_angle,
            "LH_HFE": hip_angle,
            "RF_HFE": hip_angle, #postive angle is negative
            "RH_HFE": hip_angle,
            "LF_KFE": knee_angle,
            "LH_KFE": knee_angle,
            "RF_KFE": knee_angle,
            "RH_KFE": knee_angle,
        },
        )
        
        #stable parameters for explicit PD control   
        # stall_torque = 1 #Nm
        # velocity_limit = 13 #rad/s 
        # stiffness = 0.5 #Nm/rad
        # damping = 0.02 #Nm/rad/s
        
        #Implicit control (verified to work in test env)
        stall_torque = 2 #Nm
        velocity_limit = 30 #rad/s 
        stiffness = 10 #Nm/rad
        damping = 0.02 #Nm/rad/s
        
        # #Weaker implicit control (to make it easier to learn to control)
        # stall_torque = 1 #Nm
        # velocity_limit = 15 #rad/s 
        # stiffness = 5 #Nm/rad
        # damping = 0.05 #Nm/rad/s

        actuators = {
            "motors": ImplicitActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                stiffness=stiffness,
                damping=damping,
                velocity_limit=velocity_limit,
                effort_limit=stall_torque,
            ),
        }
    
        spawn=sim_utils.UsdFileCfg(
                usd_path=f"{os.getcwd()}/USD_files/parallell_spring_jumper/parallell_spring_jumper.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    max_depenetration_velocity=1.0, 
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0
                ),
        )
        
        super().__init__(
            prim_path="/World/envs/env_.*/robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
            soft_joint_pos_limit_factor=1.0, #add softening to joint limits to reduce impact of hard limits
        )
