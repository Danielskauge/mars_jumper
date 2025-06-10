import os
import numpy as np
import torch
from typing import Tuple, List

from robot.actuators.actuators import TorqueSpeedServoCfg, ParallelElasticActuatorCfg
#from robot.actuators.gru_actuator_cfg import CombinedGRUAndPDActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


DEG2RAD = torch.pi / 180.0
RPM2RADPS = 2.0 * torch.pi / 60.0


@configclass
class MarsJumperRobotCfg(ArticulationCfg):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.HAA_REGEX: str = ".*_HAA" 
        self.HFE_REGEX: str = ".*_HFE" 
        self.KFE_REGEX: str = ".*_KFE"
        self.FEET_REGEX: str = ".*FOOT.*"
        self.AVOID_CONTACT_BODIES_REGEX: List[str] = [".*base.*", ".*HIP.*", ".*THIGH.*", ".*SHANK.*"]
        
        # Store limits as attributes if needed elsewhere, but primarily use constants
        # Define limits constants first
        self.HIP_FLEXION_LIMITS: Tuple[float, float] = (-np.pi, np.pi)
        self.KNEE_LIMITS: Tuple[float, float] = (20*DEG2RAD, 150*DEG2RAD)
        self.ABDUCTION_LIMITS: Tuple[float, float] = (-70*DEG2RAD, 70*DEG2RAD)

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
        

        actuators = {
            "knee_actuators": ParallelElasticActuatorCfg(
                joint_names_expr=[".*KFE"],
                stiffness=35, #original 35
                damping=0.05,
                effort_limit=1.82,
                velocity_limit=36,
                spring_stiffness=0.315, #0.315
            ),
            "hip_actuators": TorqueSpeedServoCfg(
                joint_names_expr=[".*HFE"],
                stiffness=35, #orignal 35
                damping=0.05, 
                effort_limit=1.82,
                velocity_limit=36,
            ),
            "abductor_actuators": TorqueSpeedServoCfg(
                joint_names_expr=[".*HAA"],
                stiffness=21, #original 21
                damping=0.073,
                effort_limit=1.12,
                velocity_limit=15.4,
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
