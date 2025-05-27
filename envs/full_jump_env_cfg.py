import numpy as np
from typing import Tuple
from isaaclab.assets.rigid_object.rigid_object_cfg import RigidObjectCfg
from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import rewards, events, curriculums, observations, terminations
from robot.robot_cfg import MarsJumperRobotCfg
from terms.utils import Phase
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

MAX_EPISODE_LENGTH_S = 2.0
DEG2RAD = np.pi/180
EARTH_GRAVITY = 9.81
MARS_GRAVITY = 3.721

@configclass
class MetricsBucketingCfg:
    """Configuration for command bucketing and metric tracking."""
    num_height_buckets: int = 3
    num_length_buckets: int = 3

@configclass
class CommandRangesCfg:
    # New primary interface: heighte and length ranges
    # Pitch 45deg from vertical is equivalent to H = 0.25L, which is the shallowest jump within the friction cone for a friction coefficient of 1. Thus <45deg might be optimal.

    min_target_length = 0.0  # m  
    max_target_length = 0.7  # m

    min_target_height = 0.2  # m
    max_target_height = 0.5  # m
    
    height_range: Tuple[float, float] = (min_target_height, max_target_height)
    length_range: Tuple[float, float] = (min_target_length, max_target_length)
    
    # Curriculum ranges for height/length
    curriculum_final_height_range: Tuple[float, float] = (min_target_height, max_target_height)
    curriculum_final_length_range: Tuple[float, float] = (min_target_length, max_target_length)
    
    # Keep these for backward compatibility and derived calculations
    @property
    def pitch_range(self) -> Tuple[float, float]:
        """Calculate pitch range from height/length ranges
        
        Convention: pitch = 0 is vertical, increases clockwise toward horizontal
        """
        # Calculate pitch from tan(pitch) = length/(4*height)
        # Use current range values, not original min/max values
        min_height, max_height = self.height_range
        min_length, max_length = self.length_range
        
        min_pitch = float(np.arctan(min_length / (4 * max_height)))
        max_pitch = float(np.arctan(max_length / (4 * min_height)))
        return (min_pitch, max_pitch)
    
    @property
    def magnitude_range(self) -> Tuple[float, float]:
        """Calculate magnitude range from height/length ranges using physics
        
        Convention: pitch = 0 is vertical, increases clockwise toward horizontal
        """
        gravity = EARTH_GRAVITY
        
        # Use current range values, not original min/max values
        min_height, max_height = self.height_range
        min_length, max_length = self.length_range
        
        # For each corner of height/length rectangle, calculate required velocity
        combinations = [
            (min_height, min_length),
            (min_height, max_length),
            (max_height, min_length),
            (max_height, max_length)
        ]
        
        magnitudes = []
        for h, l in combinations:
            pitch = np.arctan(l / (4 * h))  # Using corrected formula
            if np.sin(2 * pitch) > 0:  # Avoid division by zero
                v0 = np.sqrt(gravity * l / np.sin(2 * pitch))
                magnitudes.append(v0)
        
        if magnitudes:
            return (float(min(magnitudes)), float(max(magnitudes)))
        else:
            # Fallback to simple calculation based on pure vertical jumps
            min_magnitude = float(np.sqrt(2 * gravity * min_height))
            max_magnitude = float(np.sqrt(2 * gravity * max_height))
            return (min_magnitude, max_magnitude)

@configclass
class EventCfg:
    reset_scene_to_default = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_robot_pose_with_feet_on_ground = EventTermCfg(func=events.reset_robot_pose_with_feet_on_ground, mode="reset", params={
        "base_height_range": (0.08, 0.08), # Increased range from (0.05, 0.015) to allow deeper crouch
        "base_pitch_range_rad": (15*DEG2RAD, 15*DEG2RAD), # Restored wider range for more variety
        "front_foot_x_offset_range_cm": (0, 0), # Restored some variation from (0, 0)
        "hind_foot_x_offset_range_cm": (0, 0), # Restored variation from (0, 0)
        "base_vertical_vel_range": (-0, 0),  # Range for sampling base vertical velocity in m/s
    })

@configclass
class RewardsCfg:
    relative_cmd_error = RewardTermCfg(func=rewards.cmd_error,
                                       params={"kernel": rewards.Kernel.EXPONENTIAL,
                                               "scale": 7.0,
                                               "error_type": "relative"},
                                       weight=10.0, 
    )
    
    # Add takeoff angle error reward to shape angle precision
    takeoff_angle_error = RewardTermCfg(func=rewards.takeoff_angle_error,
                                        params={"scale": 7.0},
                                        weight=30.0,
    )
    
    liftoff_relative_cmd_error = RewardTermCfg(func=rewards.liftoff_relative_cmd_error,
        params={"kernel": rewards.Kernel.EXPONENTIAL,
                "scale": 7.0}, 
        weight=100.0, 
    )

    
    # relative_cmd_error_huber = RewardTermCfg(func=rewards.cmd_error,
    #                                    params={"kernel": rewards.Kernel.HUBER,
    #                                            "error_type": "relative",
    #                                            "delta": 0.01,
    #                                            "e_max": 0.6},
    #                                    weight=100.0, 
    # )
    

    
    # attitude_error_on_way_down = RewardTermCfg(func=rewards.attitude_error_on_way_down, 
    #                                           params={"scale": 5.0,
    #                                                   "kernel": rewards.Kernel.INVERSE_SQUARE,
    #                                                   "phases": [Phase.FLIGHT]},
    #                                           weight=0.8)
    
        
    attitude_error_on_way_down_huber = RewardTermCfg(func=rewards.attitude_error_on_way_down, 
                                              params={"scale": 5.0,
                                                      "kernel": rewards.Kernel.HUBER,
                                                      "delta": 0.01,
                                                      "e_max": 60*DEG2RAD,
                                                      "phases": [Phase.FLIGHT]},
                                              weight=1.5)
    
    # attitude_rotation_flight = RewardTermCfg(func=rewards.attitude_error, 
    #                                   params={"kernel": rewards.Kernel.INVERSE_SQUARE, 
    #                                           "scale": 5.0,
    #                                           "phases": [Phase.FLIGHT]},
    #                                   weight=0.08)
    
    # attitude_rotation_flight_huber = RewardTermCfg(func=rewards.attitude_error, 
    #                                   params={"kernel": rewards.Kernel.HUBER, 
    #                                           "delta": 0.1,
    #                                           "e_max": np.pi/3,
    #                                           "phases": [Phase.FLIGHT]},
    #                                   weight=0.08)
    
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -2, params={"term_keys": [#"bad_orientation_takeoff", 
                                                                                                #  "bad_yaw_takeoff",
                                                                                                 #"bad_orientation_flight",
                                                                                                #  "bad_yaw_flight",
                                                                                                #  "bad_roll_takeoff",
                                                                                                #  "bad_roll_flight",
                                                                                                 "bad_orientation_landing",
                                                                                                 #"failed_to_reach_target_height",
                                                                                                 "walking"
                                                                                                 ]})
    
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.01)

    # contact_forces = RewardTermCfg(func=rewards.contact_forces,
    #                                    weight=0.0001,
    #                                    params={
    #                                        "kernel": rewards.Kernel.HUBER,
    #                                        "delta": 1,
    #                                        "e_max": 100,
    #                                        "phases": [Phase.TAKEOFF, Phase.FLIGHT]})
    
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2,
                                   weight=-0.0001)
    
    landing_contact_forces = RewardTermCfg(func=rewards.contact_forces,
                                           weight=0.05,
                                           params={
                                               "kernel": rewards.Kernel.HUBER,
                                               "delta": 1,
                                               "e_max": 30,
                                               "phases": [Phase.LANDING]})
    
    feet_height = RewardTermCfg(func=rewards.feet_height,
                                    params={"kernel": rewards.Kernel.HUBER,
                                            "delta": 0.02,
                                            "e_max": 0.20,
                                            "phases": [Phase.LANDING]},
                                    weight=3)

    attitude_landing = RewardTermCfg(func=rewards.attitude_error, 
                                      params={"kernel": rewards.Kernel.HUBER, 
                                              "delta": 10*DEG2RAD,
                                              "e_max": 30*DEG2RAD,
                                              "phases": [Phase.LANDING]},
                                      weight=0.2)
    
    landing_base_height = RewardTermCfg(func=rewards.landing_base_height,
                                   params={"target_height": 0.16, 
                                           "kernel": rewards.Kernel.HUBER,
                                           "delta": 0.02,
                                           "e_max": 0.10},
                                   weight=0.5)
    
    
    # yaw_penalty_takeoff_ascent = RewardTermCfg(func=rewards.yaw_penalty,
    #                                            params={"kernel": rewards.Kernel.HUBER,
    #                                                    "delta": 20*DEG2RAD,  # 10 degrees threshold for quadratic to linear
    #                                                    "e_max": 30*DEG2RAD,  # 30 degrees max before reward=0
    #                                                    "phases": [Phase.TAKEOFF, Phase.FLIGHT]},
    #                                            weight=0.3)
    
    # roll_penalty_takeoff_ascent = RewardTermCfg(func=rewards.roll_penalty,
    #                                             params={"kernel": rewards.Kernel.HUBER,
    #                                                     "delta": 20*DEG2RAD,  # 5 degrees threshold for quadratic to linear
    #                                                     "e_max": 30*DEG2RAD,  # 15 degrees max before reward=0
    #                                                     "phases": [Phase.TAKEOFF, Phase.FLIGHT]},
    #                                             weight=0.3)
    
    # absolute_cmd_error = RewardTermCfg(func=rewards.cmd_error,
    #                                    params={"scale": 7/2.2, 
    #                                            "kernel": rewards.Kernel.EXPONENTIAL},
    #                                    weight=10.0,
    # )


    
    # takeoff_excess_rotation = RewardTermCfg(func=rewards.attitude_penalty_takeoff_threshold,
    #                                         params={"threshold_deg": 30},
    #                                         weight=-0.1)
    # relative_cmd_error_huber = RewardTermCfg(func=rewards.relative_cmd_error_huber,
    #                                         params={"delta": 0.1,      # Transition point (in relative error units)
    #                                                 "e_max": 0.6},     # Max error before reward=0 (in relative error units)
    #                                         weight=4.0,                # Use weight to control magnitude
    # )


    
    # takeoff_angle_error = RewardTermCfg(func=rewards.takeoff_angle_error,
    #                                    params={"scale": 3.0},
    #                                    weight=10.0)


    # landing_base_vertical_vel = RewardTermCfg(func=rewards.landing_base_vertical_vel_l1,
    #                                           weight=-0.001)
    
    # attitude_takeoff = RewardTermCfg(func=rewards.attitude_rotation_magnitude, 
    #                                   params={"kernel": "inverse_quadratic", 
    #                                           "scale": 10.0,
    #                                           "phases": [Phase.TAKEOFF]},
    #                                   weight=0.01)
    


    # landing_foot_ground_contact = RewardTermCfg(func=rewards.landing_foot_ground_contact,
    #                                             weight=0.1)
    
    # landing_joint_vel = RewardTermCfg(func=rewards.joint_vel_l1,
    #                                   params={"phases": [Phase.LANDING]},
    #                                   weight=-0.02)
    
    # landing_abduction_zero_pos = RewardTermCfg(func=rewards.landing_abduction_zero_pos,
    #                                           weight=-0.01)
            

    
    # joint_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2,
    #                                   weight=-0.0001)
    
    # foot_contact_stability = RewardTermCfg(func=rewards.foot_contact_state_change_penalty,
    #                                          weight=-0.05, # Negative weight as the function returns a penalty
    #                                          params={"phases": [Phase.LANDING, Phase.TAKEOFF]} # Example phases
    #                                          )
    
    # # Standing/landing foot positioning rewards

    # # Alternative penalty approach for foot height
    # feet_height_penalty = RewardTermCfg(func=rewards.feet_height_penalty,
    #                                     params={"ground_height": 0.0,
    #                                             "phases": [Phase.LANDING],
    #                                             "kernel": rewards.Kernel.LINEAR},
    #                                     weight=-0.05)  # Negative weight since it's a penalty

@configclass
class TerminationsCfg:
    #bad_takeoff_at_descent = TerminationTermCfg(func=terminations.bad_takeoff_at_descent, params={"relative_error_threshold": 0.05})
    bad_takeoff_success_rate = TerminationTermCfg(func=terminations.bad_takeoff_success_rate, params={"success_rate_threshold": 0.6, "phase": Phase.LANDING})
    #bad_takeoff_at_flight = TerminationTermCfg(func=terminations.bad_takeoff_at_flight, params={"relative_error_threshold": 0.05})
    bad_takeoff_at_landing = TerminationTermCfg(func=terminations.bad_takeoff_at_landing, params={"relative_error_threshold": 0.1})
    #bad_takeoff_at_flight = TerminationTermCfg(func=terminations.bad_takeoff_at_flight, params={"relative_error_threshold": 0.1})
    #landed = TerminationTermCfg(func=terminations.landed, time_out=False)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

    bad_orientation_takeoff = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/2, 
        "phases": [Phase.TAKEOFF]
    })
    bad_orientation_flight = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/2, 
        "phases": [Phase.FLIGHT]
    })
    
    # bad_yaw_takeoff = TerminationTermCfg(func=terminations.bad_yaw, params={"limit_angle": np.pi/4,
    #                                                                         "phases": [Phase.TAKEOFF]})
    # # bad_yaw_flight = TerminationTermCfg(func=terminations.bad_yaw, params={"limit_angle": np.pi/4,
    # #                                                                         "phases": [Phase.FLIGHT]})
    
    # bad_roll_takeoff = TerminationTermCfg(func=terminations.bad_roll, params={"limit_angle": np.pi/4,
    #                                                                           "phases": [Phase.TAKEOFF]})
    # bad_roll_flight = TerminationTermCfg(func=terminations.bad_roll, params={"limit_angle": np.pi/4,
    #                                                                          "phases": [Phase.FLIGHT]})
    bad_orientation_landing = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/6, 
        "phases": [Phase.LANDING]
    })
    
    #failed_to_reach_target_height = TerminationTermCfg(func=terminations.failed_to_reach_target_height, 
                                                    #    params={"height_threshold": 0.10})
    walking = TerminationTermCfg(func=terminations.walking)
    
    # illegal_contact = TerminationTermCfg(func=mdp.terminations.illegal_contact, 
    #                                      params={
    #                                          "sensor_cfg": SceneEntityCfg(name="contact_sensor", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]),
    #                                          "threshold": 100})
    # self_collision = TerminationTermCfg(func=custom_terminations.self_collision, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0}) #TODO: implement

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionToLimitsActionCfg(
        asset_name="robot", 
        joint_names=[".*"], # Apply to all joints
        scale=1.0, # Assuming policy output is already in a suitable range like [-1, 1] to be mapped to limits
        rescale_to_limits=True, # This is the key, and it's True by default
        # offset can be used if needed, but typically not if rescale_to_limits handles the full range.
        # If your policy outputs delta from default_joint_pos, you might add default_joint_pos as offset
        # AFTER the unscale_transform, or adjust how unscale_transform is used. 
        # For now, assume policy outputs absolute targets in normalized [-1,1] space.
        debug_vis=True
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
        command_vec = ObservationTermCfg(func=observations.takeoff_height_length_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg()

@configclass
class MySceneCfg(InteractiveSceneCfg):
    # Restore the original terrain definition
    terrain = TerrainImporterCfg(
       prim_path="/World/ground",
       terrain_type="plane",
       collision_group=-1,
       physics_material=sim_utils.RigidBodyMaterialCfg(
           friction_combine_mode="max",
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
class CurriculumCfg:
    command_range_progression = CurriculumTermCfg(
        func=curriculums.progress_command_ranges,
        params={
            "num_curriculum_levels": 50,
            "success_rate_threshold": 0.9,
            "min_steps_between_updates": 150,
            "enable_regression": False,
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
    metrics_bucketing: MetricsBucketingCfg | None = None

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
        
