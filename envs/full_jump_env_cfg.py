import numpy as np
from typing import Tuple
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import rewards, events, curriculums, observations, terminations
from robot.robot_cfg import MarsJumperRobotCfg
from terms.phase import Phase

MAX_EPISODE_LENGTH_S = 3.0
DEG2RAD = np.pi/180

@configclass
class CommandRangesCfg:
    earth_gravity_abs = 9.81
    initial_target_height = 0.5 #m
    initial_vel = float(np.sqrt(earth_gravity_abs * initial_target_height)) #m/s
    
    final_target_height = 1.5 #m
    final_vel = float(np.sqrt(earth_gravity_abs * final_target_height)) #m/s
    
    initial_pitch_range: Tuple[float, float] = (0.0, 0.0) # Initial pitch range
    initial_magnitude_range: Tuple[float, float] = (initial_vel, initial_vel) # Initial magnitude range
    
    final_pitch_range: Tuple[float, float] = (0.0, 0.0) # Final pitch range
    final_magnitude_range: Tuple[float, float] = (initial_vel, final_vel) # Final magnitude range
    
@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Restore the original terrain definition
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

    contact_sensor = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot/.*", force_threshold=0.1)
    
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

@configclass
class ActionsCfg:
    #joint_pos = mdp.JointPositionActionCfg(asset_name="robot", debug_vis=True, joint_names=[".*"], use_default_offset=True) 
    # joint_pos = mdp.JointPositionToLimitsActionCfg(
    #     asset_name="robot", 
    #     joint_names=[".*"], # Apply to all joints
    #     scale=1.0, # Assuming policy output is already in a suitable range like [-1, 1] to be mapped to limits
    #     rescale_to_limits=True, # This is the key, and it's True by default
    #     # offset can be used if needed, but typically not if rescale_to_limits handles the full range.
    #     # If your policy outputs delta from default_joint_pos, you might add default_joint_pos as offset
    #     # AFTER the unscale_transform, or adjust how unscale_transform is used. 
    #     # For now, assume policy outputs absolute targets in normalized [-1,1] space.
    #     debug_vis=True
    # )
   joint_pos = mdp.JointPositionActionCfg(asset_name="robot", debug_vis=True, joint_names=[".*"], use_default_offset=True) 
    
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
class RewardsCfg:

    relative_cmd_error = RewardTermCfg(func=rewards.relative_cmd_error,
                                       params={"scale": 7.0, 
                                               "kernel": rewards.Kernel.EXPONENTIAL},
                                       weight=10.0,
    )

    landing_base_height = RewardTermCfg(func=rewards.landing_base_height,
                                   params={"target_height": 0.18, 
                                           "kernel": rewards.Kernel.INVERSE_SQUARE,
                                           "scale": 400},
                                   weight=0.2)
    
    landing_base_vertical_vel = RewardTermCfg(func=rewards.landing_base_vertical_vel_l1,
                                              weight=-0.001)
    
    attitude_error_on_way_down = RewardTermCfg(func=rewards.attitude_error_on_way_down, 
                                              params={"scale": 5.0},
                                              weight=1)
    
    attitude_rotation_flight = RewardTermCfg(func=rewards.attitude_rotation_magnitude, 
                                      params={"kernel": "inverse_quadratic", 
                                              "scale": 5.0,
                                              "phases": [Phase.FLIGHT]},
                                      weight=0.2)
    
    attitude_landing = RewardTermCfg(func=rewards.attitude_rotation_magnitude, 
                                      params={"kernel": "inverse_quadratic", 
                                              "scale": 11.0,
                                              "phases": [Phase.LANDING]},
                                      weight=0.3)
    
    # landing_foot_ground_contact = RewardTermCfg(func=rewards.landing_foot_ground_contact,
    #                                             weight=0.1)
    
    landing_joint_vel = RewardTermCfg(func=rewards.joint_vel_l1,
                                      params={"phases": [Phase.LANDING]},
                                      weight=-0.001)
    
    landing_abduction_zero_pos = RewardTermCfg(func=rewards.landing_abduction_zero_pos,
                                              weight=-0.01)
            
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -1, params={"term_keys": ["bad_orientation"]})
    
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.1)
    
    contact_forces_potential_based = RewardTermCfg(func=rewards.contact_forces_potential_based,
                                           weight=-0.01,
                                           params={"sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]),
                                                   "kernel": rewards.Kernel.LINEAR,
                                                   "phases": [Phase.LANDING, Phase.TAKEOFF, Phase.FLIGHT],
                                                   "potential_buffer_postfix": "landing_contact"})
    
    landing_contact_forces = RewardTermCfg(func=rewards.contact_forces,
                                       weight=-0.001,
                                       params={
                                           "kernel": rewards.Kernel.LINEAR,
                                           "phases": [Phase.FLIGHT, Phase.TAKEOFF]})
    
    # foot_contact_stability = RewardTermCfg(func=rewards.foot_contact_state_change_penalty,
    #                                          weight=-0.05, # Negative weight as the function returns a penalty
    #                                          params={"phases": [Phase.LANDING, Phase.TAKEOFF]} # Example phases
    #                                          )
    
@configclass
class TerminationsCfg:
    bad_takeoff_at_landing = TerminationTermCfg(func=terminations.bad_takeoff_at_landing, params={"relative_error_threshold": 0.1})
    #bad_flight_at_landing = TerminationTermCfg(func=terminations.bad_flight_at_landing, params={"angle_error_threshold": 15*DEG2RAD})
    #bad_takeoff_success_rate = TerminationTermCfg(func=terminations.bad_takeoff_success_rate, params={"success_rate_threshold": 0.9})
    #bad_flight_success_rate = TerminationTermCfg(func=terminations.bad_flight_success_rate, params={"success_rate_threshold": 0.9})
    #landed = TerminationTermCfg(func=terminations.landed, time_out=False)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    #bad_knee_angle = TerminationTermCfg(func=terminations.bad_knee_angle)
    bad_orientation = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/2, 
        "phases": [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING]
    })
    
    illegal_contact = TerminationTermCfg(func=mdp.terminations.illegal_contact, 
                                         params={
                                             "sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]),
                                             "threshold": 10})
    # self_collision = TerminationTermCfg(func=custom_terminations.self_collision, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0}) #TODO: implement
    
@configclass
class CurriculumCfg:
    command_range_progression = CurriculumTermCfg(
        func=curriculums.progress_command_ranges,
        params={
            "num_curriculum_levels": 50,
            "success_rate_threshold": 0.9,
        },
    )
        
@configclass
class FullJumpEnvCfg(ManagerBasedRLEnvCfg):

    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=2)
    command_ranges: CommandRangesCfg = CommandRangesCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    #curriculum: CurriculumCfg = CurriculumCfg()
    is_finite_horizon: bool = True

    def __post_init__(self):
        self.takeoff_magnitude_ratio_error_threshold = 0.1
        self.takeoff_angle_error_threshold_rad = 10*DEG2RAD
        
        #Flight success criteria
        self.flight_angle_error_threshold = 15*DEG2RAD
                
        self.real_time_control_dt = 1/120
        self.sim.dt = 1/360 #Physics time step, also the torque update rate
        self.episode_length_s = MAX_EPISODE_LENGTH_S
        self.viewer = ViewerCfg(
            eye=[0, 0.6, 0.03],
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
        # Use the physics material from the new ground_plane definition if needed elsewhere,
        # or define a shared material config. For now, assuming it's self-contained.
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        contact_sensor_period = self.decimation * self.sim.dt

        self.scene.contact_sensor.update_period = contact_sensor_period
        
