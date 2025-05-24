import os
import numpy as np
import torch
from typing import Tuple, List

from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, ActuatorNetLSTMCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass

DEG2RAD = torch.pi / 180.0
RPM2RADPS = 2.0 * torch.pi / 60.0

# Define limits constants first
HIP_FLEXION_LIMITS: Tuple[float, float] = (-np.pi, np.pi)
KNEE_LIMITS: Tuple[float, float] = (0, 175*DEG2RAD)
ABDUCTION_LIMITS: Tuple[float, float] = (-np.pi, np.pi)

@configclass
class MarsJumperRobotCfg(ArticulationCfg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.HAA_REGEX: str = ".*_HAA" # NOTE: Mismatched names in comments vs. regex - assuming regex is correct
        self.HFE_REGEX: str = ".*_HFE" # NOTE: Mismatched names in comments vs. regex - assuming regex is correct
        self.KFE_REGEX: str = ".*_KFE"
        self.FEET_REGEX: str = ".*FOOT.*"
        self.AVOID_CONTACT_BODIES_REGEX: List[str] = [".*base.*", ".*HIP.*", ".*THIGH.*", ".*SHANK.*"]
        
        # Store limits as attributes if needed elsewhere, but primarily use constants
        self.hip_joint_limits = HIP_FLEXION_LIMITS 
        self.knee_joint_limits = KNEE_LIMITS
        self.abduction_joint_limits = ABDUCTION_LIMITS

        self.HIP_LINK_LENGTH: float = 0.11
        self.KNEE_LINK_LENGTH: float = 0.11
        
        self.abductor_angle = 0 * DEG2RAD
        self.hip_angle = -70 * DEG2RAD#-120 * DEG2RAD
        self.knee_angle = 140 * DEG2RAD#140 * DEG2RAD
        self.init_height = 0.08 #was 0.08 at -70 and 140, 0.06 at -120 and 140

        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, self.init_height),
            joint_pos={ #TODO: Check if these are correct
            "LF_HAA": self.abductor_angle, 
            "LH_HAA": self.abductor_angle, #postive angle is negative
            "RF_HAA": self.abductor_angle, #postive angle is negative
            "RH_HAA": self.abductor_angle,
            "LF_HFE": self.hip_angle,
            "LH_HFE": self.hip_angle,
            "RF_HFE": self.hip_angle, #postive angle is negative
            "RH_HFE": self.hip_angle,
            "LF_KFE": self.knee_angle,
            "LH_KFE": self.knee_angle,
            "RF_KFE": self.knee_angle,
            "RH_KFE": self.knee_angle,
        },
        )
        
        #old and too powerful
        # stall_torque = 2 #Nm
        # stiffness = 20 #Nm/rad (not sure if that's correct)
        # damping = 0.2 #Nm/rad/s #original 0.05
        
        #new more realistic
        stall_torque = 2 #Nm
        stiffness = 10 #Nm/rad (not sure if that's correct)
        damping = 0.2 #Nm/rad/s #original 0.05

        actuators = {
            "motors": ImplicitActuatorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                stiffness=stiffness,
                damping=damping,
                effort_limit=stall_torque,
                
            ),
        }
    
        spawn=sim_utils.UsdFileCfg(
                usd_path=f"{os.getcwd()}/USD_files/moved_motor_usd_limits/moved_motor_usd_limits.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    max_depenetration_velocity=1.0, 
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True, 
                    solver_position_iteration_count=4, 
                    solver_velocity_iteration_count=0 #if else then robot accumulates angular velocity for some reason
                ),
        )
        
        super().__init__(
            prim_path="/World/envs/env_.*/robot",
            #prim_path="/World/robot",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
            soft_joint_pos_limit_factor=1.0, #add softening to joint limits to reduce impact of hard limits
        )
