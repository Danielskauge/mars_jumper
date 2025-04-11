
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

MAX_EPISODE_LENGTH_S = 5.0
DEG2RAD = np.pi/180

@configclass
class CommandRangesCfg:
    """Configuration for command ranges."""
    mars_gravity_abs = 3.721
    earth_gravity_abs = 9.81
    initial_target_height = 0.5 #m
    initial_vel = 2.2147 #float(np.sqrt(earth_gravity_abs * initial_target_height)) #m/s
    
    final_target_height = 2 #m
    final_vel = float(np.sqrt(mars_gravity_abs * final_target_height)) #m/s
    
    initial_pitch_range: Tuple[float, float] = (0.0, 0.0) # Initial pitch range
    initial_magnitude_range: Tuple[float, float] = (initial_vel, initial_vel) # Initial magnitude range
    
    final_pitch_range: Tuple[float, float] = (0.0, 0.0) # Final pitch range
    final_magnitude_range: Tuple[float, float] = (initial_vel, 0.8*final_vel) # Final magnitude range
    
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
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
    # robots
    robot: ArticulationCfg = MarsJumperRobotCfg()    
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/.*", history_length=1, track_air_time=True, force_threshold=0.1)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    """Actions are resampled at each step of the environment"""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", debug_vis=True, joint_names=[".*"], use_default_offset=True) 
    
@configclass
class ObservationsCfg:
    """Observations are sampled at each step of the environment."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observation term configs for policy group."""
        base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        #base_quat = ObservationTermCfg(func=mdp.root_quat_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        previous_actions = ObservationTermCfg(func=mdp.last_action)        
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)
        command_vec = ObservationTermCfg(func=observations.takeoff_vel_vec_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """ 
    Event terms can modify anything about the environment. 
    They can be run at
    - startup: Once at the beginning of training
    - reset: When the episode for an environment ends and a new episode begins
    - interval: At a fixed interval of time
    """
    reset_scene_to_default = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    
@configclass
class RewardsCfg:
    """
    Rewards are computed at each step of the environment (which can include multiple physics steps). 
    There is no built in implementation of per-episode rewards.
    """
    #TAKEOFF PHASE
    # cmd_error = RewardTermCfg(func=rewards.cmd_error,
    #                           params={"scale": 1, "kernel": "exponential"},
    #                           weight=1,
    # )
    relative_cmd_error = RewardTermCfg(func=rewards.relative_cmd_error,
                                       params={"scale": 2.0, "kernel": "inverse_linear"},
                                       weight=2.0,
    )
    flat_orientation = RewardTermCfg(func=mdp.flat_orientation_l2, 
                                     weight= -0.1)
    
    joint_vel_l1 = RewardTermCfg(func=mdp.joint_vel_l1,
                                 params={"asset_cfg": SceneEntityCfg("robot")},
                                 weight= -0.001,
    )
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, 
                                   weight= -0.01)

    #FLIGHT PHASE
    attitude_rotation = RewardTermCfg(func=rewards.attitude_rotation_magnitude, 
                                      params={"kernel": "inverse_linear", "scale": 1},
                                      weight=2)
    
    root_ang_vel_l1 = RewardTermCfg(func=rewards.ang_vel_l1, 
                                    weight=-0.001)

    # is_alive_in_landing = RewardTermCfg(func=custom_rewards.is_alive, 
    #                                     params={"phases": [Phase.LANDING, Phase.FLIGHT]},
    #                                     weight=1)
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -0.1, params={"term_keys": ["base_contact", "bad_orientation", "time_out"]})
    
    #ALL PHASES
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.01)
    
    undesired_contacts = RewardTermCfg(func=mdp.contact_forces,
                                       weight=-0.01, #-0.1,
                                       params={"sensor_cfg": 
                                           SceneEntityCfg(
                                               name="contact_forces", 
                                               body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), 
                                           "threshold": 0.1},)
    
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    landed = TerminationTermCfg(func=terminations.landed, time_out=False)
    entered_flight = TerminationTermCfg(func=terminations.entered_flight, time_out=False)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), "threshold": 0.2})
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": np.pi/3})
    # self_collision = TerminationTermCfg(func=custom_terminations.self_collision, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0}) #TODO: implement
    
@configclass
class CurriculumCfg:
    """Curriculum terms."""

    # command_range_progression = CurriculumTermCfg(
    #     func=custom_curriculums.progress_command_ranges,
    #     params={
    #         "num_curriculum_levels": 20,
    #         "success_rate_threshold": 0.9,
    #     },
    # )
        
@configclass
class FullJumpEnvCfg(ManagerBasedRLEnvCfg):

    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=2)
    command_ranges: CommandRangesCfg = CommandRangesCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    is_finite_horizon: bool = True

    def __post_init__(self):

        #Takeoff success criteria
        self.takeoff_magnitude_ratio_error_threshold = 0.1
        self.takeoff_angle_error_threshold = 10*DEG2RAD
        
        self.crouch_to_takeoff_height_trigger = 0.08 #0.22 #Threshold for transitioning from takeoff to flight phase
        self.takeoff_to_flight_height_trigger = 0.25 #robot max standing height is 22cm
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
        self.sim.gravity = (0.0, 0.0, self.mars_gravity)
        self.decimation = int(self.real_time_control_dt / self.sim.dt) #Number of physics steps per env step 
 
        self.sim.render_interval = self.decimation #Number of physics steps between render frames
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        #if getattr(self.curriculum, "terrain_levels", None) is not None:
        #    if self.scene.terrain.terrain_generator is not None:
        #        self.scene.terrain.terrain_generator.curriculum = True
        #else:
        #    if self.scene.terrain.terrain_generator is not None:
        #        self.scene.terrain.terrain_generator.curriculum = False

