import numpy as np
from typing import Tuple
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import rewards
from terms import events
from terms import curriculums
from terms import observations
from terms import terminations
from robot.robot_cfg import MarsJumperRobotCfg
from terms.phase import Phase
from dataclasses import field

MAX_EPISODE_LENGTH_S = 2
DEG2RAD = np.pi/180

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1, 
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MarsJumperRobotCfg()    
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/.*", history_length=1, track_air_time=True, force_threshold=0.1)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rotation_vector = ObservationTermCfg(func=observations.base_rotation_vector)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        previous_actions = ObservationTermCfg(func=mdp.last_action)        
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)
        command_vec = ObservationTermCfg(func=observations.takeoff_vel_vec_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_scene_to_default = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", debug_vis=True, joint_names=[".*"], use_default_offset=True) 

@configclass
class CommandRangesCfg:
    """Configuration for command ranges."""
    
    # beware that theese are in addition to the height at which the robot switches to flight phase
    initial_max_target_height: float = 0.6 #m 0.4 baseline
    initial_min_target_height: float = 0.6 #m
    
    final_max_target_height: float = 1.8 #m 
    final_min_target_height: float = 0.6 #m
    
    initial_pitch_range: Tuple[float, float] = (0.0, 0.0) # Initial pitch range
    initial_magnitude_range: Tuple[float, float] = field(init=False)

    final_pitch_range: Tuple[float, float] = (0.0, 0.0) # Final pitch range
    final_magnitude_range: Tuple[float, float] = field(init=False)

    def __post_init__(self):    
        mars_gravity_abs = 3.721
    
        initial_min_vel = float(np.sqrt(mars_gravity_abs * self.initial_min_target_height))
        initial_max_vel = float(np.sqrt(mars_gravity_abs * self.initial_max_target_height))
        
        final_min_vel = float(np.sqrt(mars_gravity_abs * self.final_min_target_height))
        final_max_vel = float(np.sqrt(mars_gravity_abs * self.final_max_target_height))

        self.initial_magnitude_range = (initial_min_vel, initial_max_vel)
        self.final_magnitude_range = (final_min_vel, final_max_vel) # Or adjust as needed based on curriculum logic

@configclass
class RewardsCfg:
    relative_cmd_error = RewardTermCfg(func=rewards.relative_cmd_error,
                                       params={"scale": 7.0, "kernel": "exponential"}, 
                                       weight=2.0,
    )
    # cmd_error = RewardTermCfg(func=rewards.cmd_error,
    #                           params={"scale": 7.0, "kernel": "exponential"}, 
    #                           weight=2.0,
    # )
    
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -1.0, params={"term_keys": ["base_contact", "bad_orientation"]})
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.01)
    undesired_contacts = RewardTermCfg(func=mdp.contact_forces,
                                       weight=-0.01, #-0.1,
                                       params={"sensor_cfg": 
                                           SceneEntityCfg(
                                               name="contact_forces", 
                                               body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), 
                                           "threshold": 0.01},)
    
@configclass
class TerminationsCfg:
    entered_flight = TerminationTermCfg(func=terminations.entered_flight, time_out=False)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), "threshold": 0.2})
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": np.pi/3})
    
# @configclass
# class CurriculumCfg:
#     command_range_progression = CurriculumTermCfg(
#         func=curriculums.progress_command_ranges,
#         params={
#             "num_curriculum_levels": 40,
#             "success_rate_threshold": 0.95,
#             "min_steps_above_threshold": 1,
#         },
#     )
    
@configclass
class SuccessCriteriaCfg:
    takeoff_magnitude_ratio_error_threshold: float = 0.1
    takeoff_angle_error_threshold: float = 10*DEG2RAD   

@configclass
class MarsJumperEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=2)
    command_ranges: CommandRangesCfg = CommandRangesCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()
    success_criteria: SuccessCriteriaCfg = SuccessCriteriaCfg()
    is_finite_horizon: bool = True
    
    def __post_init__(self):
        
        self.is_finite_horizon = True
                
        self.crouch_to_takeoff_height_trigger = 0.08 #0.22 #Threshold for transitioning from takeoff to flight phase
        self.takeoff_to_flight_height_trigger = 0.15 #robot max standing height is 22cm 
        self.flight_to_landing_height_trigger = 0.30 #robot might come in tiled, so we give it a bit more room
        
        self.real_time_control_dt = 1/100
        self.sim.dt = 1/400 #Physics time step, also the torque update rate
        self.episode_length_s = MAX_EPISODE_LENGTH_S
        self.viewer = ViewerCfg(
            eye=[0, 0.5, 0.04],
            lookat=[0, 0, 0],
            origin_type="asset_root",
            asset_name="robot",
            resolution=(1280, 720),
        )
        self.mars_gravity = -3.721
        self.earth_gravity = -9.81
        self.sim.gravity = (0.0, 0.0, self.earth_gravity)
        self.decimation = int(self.real_time_control_dt / self.sim.dt) #Number of physics steps per env step 
 
        self.sim.render_interval = self.decimation #Number of physics steps between render frames
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        