import os
import torch
from typing import Tuple, List

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from .motors import get_AK7010_cfg, get_AK809_cfg

from mars_jumper.robot.Articulation import Articulation

DEG2RAD = torch.pi / 180.0
RPM2RADPS = 2.0 * torch.pi / 60.0


class MarsJumperRobot(Articulation):
    """Mars Jumper Robot class."""
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        #IDS
        self.feet_idx: List[int] = self.find_bodies(cfg.FEET_REGEX)[0]
        self.avoid_contact_body_idx: List[int] = self.find_bodies(cfg.AVOID_CONTACT_BODIES_REGEX)[0]
        self.hip_abduction_joints_idx: List[int] = self.find_joints(cfg.HIP_ABDUCTION_JOINTS_REGEX)[0]
        self.hip_flexion_joints_idx: List[int] = self.find_joints(cfg.HIP_FLEXION_JOINTS_REGEX)[0]
        self.knee_joints_idx: List[int] = self.find_joints(cfg.KNEE_JOINTS_RGX)[0]
        self.joints_idx: List[int] = [self.hip_abduction_joints_idx, 
                                    self.hip_flexion_joints_idx, 
                                    self.knee_joints_idx]

    def clip_position_commands_to_joint_limits(self, position_commands: torch.Tensor) -> torch.Tensor:
        hip_abduction_commands = torch.clip(position_commands[:, self.hip_abduction_joints_idx], 
                          self.HIP_ABDUCTION_ANGLE_LIMITS_RAD[0], 
                          self.HIP_ABDUCTION_ANGLE_LIMITS_RAD[1])
        knee_commands = torch.clip(position_commands[:, self.knee_joints_idx], 
                          self.KNEE_ANGLE_LIMITS_RAD[0], 
                          self.KNEE_ANGLE_LIMITS_RAD[1])
        hip_flexion_commands = torch.clip(position_commands[:, self.hip_flexion_joints_idx], 
                          self.HIP_FLEXION_ANGLE_LIMITS_RAD[0], 
                          self.HIP_FLEXION_ANGLE_LIMITS_RAD[1])
        return torch.cat((hip_abduction_commands, knee_commands, hip_flexion_commands), dim=1)

class MarsJumperRobotConfig(ArticulationCfg):
    """Configuration of mars jumper robot using cube mars motor model."""
    HIP_FLEXION_JOINTS_REGEX: str = ".*_HAA*."
    HIP_ABDUCTION_JOINTS_REGEX: str = ".*_HFE*."
    KNEE_JOINTS_REGEX: str = ".*_KFE*."
    FEET_REGEX: str = ".*FOOT.*"
    AVOID_CONTACT_BODIES_REGEX: List[str] = ["base", ".*HIP.*", ".*THIGH.*", ".*SHANK.*"]

    HIP_ABDUCTION_ANGLE_LIMITS_RAD: Tuple[float, float] = (45 * DEG2RAD, 135 * DEG2RAD)
    KNEE_ANGLE_LIMITS_RAD: Tuple[float, float] = (0 * DEG2RAD, 135 * DEG2RAD)
    HIP_FLEXION_ANGLE_LIMITS_RAD: Tuple[float, float] = (-15 * DEG2RAD, 45 * DEG2RAD)

    def __init__(self):

        init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4086),
            joint_pos={
            "LF_HAA": 0.0,
            "LH_HAA": 0.0, 
            "RF_HAA": 0.0,
            "RH_HAA": 0.0,
            "LF_HFE": 45 * DEG2RAD,
            "LH_HFE": 45 * DEG2RAD,
            "RF_HFE": 45 * DEG2RAD,
            "RH_HFE": 45 * DEG2RAD,
            "LF_KFE": 70.4116 * DEG2RAD,
            "LH_KFE": 70.4116 * DEG2RAD,
            "RF_KFE": 70.8849 * DEG2RAD,
            "RH_KFE": 70.8849 * DEG2RAD,
        },
        )
        
        actuators = {
            "hip_motors": get_AK809_cfg(
                joint_names_expr=self.HIP_FLEXION_JOINTS_RGX, kp=20.0, kd=2.0
            ),
            "hip_abduction_motors": get_AK809_cfg(
                joint_names_expr=self.HIP_ABDUCTION_JOINTS_RGX, kp=20.0, kd=2.0
            ),
            "knee_motors": get_AK809_cfg(
                joint_names_expr=self.KNEE_JOINTS_RGX, kp=20.0, kd=2.0
            ),
            "knee_torsional_springs": ImplicitActuatorCfg(
                joint_names_expr=self.KNEE_JOINTS_RGX,
                effort_limit=10.0,
                stiffness=20.0, #TODO: Add correct values
                damping=0.001,
                friction=0.0,
            ),
            "hip_torsional_springs": ImplicitActuatorCfg(
                joint_names_expr=self.HIP_FLEXION_JOINTS_RGX,
                effort_limit=10.0,
                stiffness=20.0, #TODO: Add correct values
                damping=0.001,
                friction=0.0,
            ),
        }
        
        spawn=sim_utils.UsdFileCfg(
                usd_path=f"{os.getcwd()}/submodules/cad/simplified_robot/simplified_robot.usd",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    retain_accelerations=False,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                )
        )
        
        super().__init__(
            prim_path="/World/envs/env_.*/mars_jumper",
            spawn=spawn,
            init_state=init_state,
            actuators=actuators,
            soft_joint_pos_limit_factor=1.0, #add softening to joint limits to reduce impact of hard limits
        )
