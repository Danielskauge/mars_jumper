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

MAX_EPISODE_LENGTH_S = 4.0
DEG2RAD = np.pi/180
EARTH_GRAVITY = 9.81

@configclass
class MetricsBucketingCfg:
    """Configuration for command bucketing and metric tracking."""
    num_height_buckets: int = 1
    num_length_buckets: int = 1

@configclass
class CommandRangesCfg:
    # New primary interface: heighte and length ranges
    # Pitch 45deg from vertical is equivalent to H = 0.25L, which is the shallowest jump within the friction cone for a friction coefficient of 1. Thus <45deg might be optimal.

    min_target_length = 0.0  # m  
    max_target_length = 0.5  # m

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
    # relative_cmd_error = RewardTermCfg(func=rewards.cmd_error,
    #                                    params={"kernel": rewards.Kernel.EXPONENTIAL,
    #                                            "scale": 6.0,
    #                                            "error_type": "relative"},
    #                                    weight=20.0, 
    # )
    
    # relative_cmd_error_flight_trans = RewardTermCfg(func=rewards.liftoff_relative_cmd_error,
    #     params={"kernel": rewards.Kernel.EXPONENTIAL,
    #             "scale": 6.0}, 
    #     weight=150.0, 
    # )
    
    rel_cmd_error_huber = RewardTermCfg(func=rewards.cmd_error,
                                        params={"kernel": rewards.Kernel.HUBER,
                                                "delta": 0.05,
                                                "e_max": 0.8},
                                        weight=5.0,
    )
    
    relative_cmd_error_huber_flight_trans = RewardTermCfg(func=rewards.liftoff_relative_cmd_error,
        params={"kernel": rewards.Kernel.HUBER,
                "delta": 0.05,
                "e_max": 0.8},
        weight=150.0,
    )

    attitude_descent = RewardTermCfg(func=rewards.attitude_descent, 
                                              params={"kernel": rewards.Kernel.HUBER,
                                                      "delta": 15*DEG2RAD,
                                                      "e_max": 60*DEG2RAD},
                                              weight=0.5)
    
    attitude_landing_trans = RewardTermCfg(func=rewards.attitude_landing_trans, 
                                                        params={"kernel": rewards.Kernel.HUBER,
                                                        "delta": 15*DEG2RAD,
                                                        "e_max": 60*DEG2RAD,
                                              },
                                              weight=30)
    
    attitude_landing = RewardTermCfg(func=rewards.attitude, 
                                    params={"kernel": rewards.Kernel.HUBER, 
                                            "delta": 10*DEG2RAD,
                                            "e_max": 30*DEG2RAD,
                                            "phases": [Phase.LANDING]},
                                    weight=0.5)  # Reduced to match curriculum initial_weight
    
    
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -5, params={"term_keys": ["bad_orientation_takeoff", 
                                                                                                # "base_contact_force",
                                                                                                # "thigh_contact_force",
                                                                                                # "hip_contact_force",
                                                                                                 "bad_orientation_landing",
                                                                                                "landing_walking"
                                                                                                 ]})

    contact_forces = RewardTermCfg(func=rewards.contact_forces,
                                       weight=0.01,
                                       params={
                                           "kernel": rewards.Kernel.HUBER,
                                           "delta": 1,
                                           "e_max": 20,
                                           "phases": [Phase.TAKEOFF, Phase.FLIGHT, Phase.LANDING]})
    
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2,
                                   weight=-0.0001)
    
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.1)
    
    # landing_contact_forces = RewardTermCfg(func=rewards.contact_forces,
    #                                        params={
    #                                            "kernel": rewards.Kernel.HUBER,
    #                                            "delta": 1,
    #                                            "e_max": 20,
    #                                            "phases": [Phase.LANDING]},
    #                                        weight=0.1)
    
    # landing_max_feet_force_above_threshold = RewardTermCfg(func=rewards.landing_max_feet_force_above_threshold,
    #                                                        params={
    #                                                            "kernel": rewards.Kernel.HUBER,
    #                                                            "delta": 1,
    #                                                            "e_max": 80,
    #                                                            "threshold": 50.0},
    #                                                        weight=0.01)
    
    landing_com_vel = RewardTermCfg(func=rewards.landing_com_vel,
                                    params={
                                        "kernel": rewards.Kernel.HUBER,
                                        "delta": 0.04,
                                        "e_max": 1},
                                    weight=1,
                                    )
    
    # landing_joint_vel_delayed = RewardTermCfg(func=rewards.joint_vel_l1_delayed_landing,
    #                                          params={
    #                                              "delay_seconds": 0.5,
    #                                              "kernel": rewards.Kernel.HUBER,
    #                                              "delta": np.pi,
    #                                              "e_max": 2*np.pi}, #2 rad/s
    #                                          weight=0.01)
    
    # feet_height = RewardTermCfg(func=rewards.feet_height,
    #                                 params={"kernel": rewards.Kernel.HUBER,
    #                                         "delta": 0.02,
    #                                         "e_max": 0.1,
    #                                         "phases": [Phase.LANDING]},
    #                                 weight=2.0)  # Reduced to match curriculum initial_weight

    # landing_walking = RewardTermCfg(func=rewards.landing_walking,
    #                                         params={"kernel": rewards.Kernel.HUBER,
    #                                                 "delta": 0.5,
    #                                                 "e_max": 0.15},
    #                                         weight=0.5)

    
    # landing_base_height = RewardTermCfg(func=rewards.landing_base_height,
    #                                params={"target_height": 0.14, 
    #                                        "kernel": rewards.Kernel.HUBER,
    #                                        "delta": 0.02,
    #                                        "e_max": 0.10},
    #                                weight=0.05)  # Reduced to match curriculum initial_weight
    
    
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
    
    #landed = TerminationTermCfg(func=terminations.landed)
    #bad_takeoff_success_rate = TerminationTermCfg(func=terminations.bad_takeoff_success_rate, params={"success_rate_threshold": 0.6, "phase": Phase.LANDING})
    #bad_takeoff_at_landing = TerminationTermCfg(func=terminations.bad_takeoff_at_landing, params={"relative_error_threshold": 0.1})
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    landing_timeout = TerminationTermCfg(func=terminations.landing_timeout, time_out=True, params={"timeout": 2})

    bad_orientation_takeoff = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/4, 
        "phases": [Phase.TAKEOFF]
    })

    bad_orientation_landing = TerminationTermCfg(func=terminations.bad_orientation, params={
        "limit_angle": np.pi/4, 
        "phases": [Phase.LANDING]
    })
    landing_walking = TerminationTermCfg(func=terminations.landing_walking, params={"x_tolerance": 0.15, "y_tolerance": 0.15})
        
    # base_contact_force = TerminationTermCfg(func=terminations.illegal_contact, 
    #                                      params={"body": "base", "threshold": 100})
    
    # thigh_contact_force = TerminationTermCfg(func=terminations.illegal_contact, 
    #                                      params={"body": "thigh", "threshold": 100})
    
    # hip_contact_force = TerminationTermCfg(func=terminations.illegal_contact, 
    #                                      params={"body": "hip", "threshold": 100})

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
        # Standard observations (no latency)
        base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rotation_vector = ObservationTermCfg(func=observations.base_rotation_vector)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        previous_actions = ObservationTermCfg(func=mdp.last_action)        
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)
        command_vec = ObservationTermCfg(func=observations.takeoff_height_length_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Example of base height observation with latency (uncomment and modify latency_ms as needed)
        # base_height_delayed = ObservationTermCfg(func=observations.base_pos_z_with_latency, 
        #                                         params={"latency_ms": 50.0},  # 50ms latency
        #                                         noise=Unoise(n_min=-0.05, n_max=0.05))

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
    
    reward_weight_progression = CurriculumTermCfg(
        func=curriculums.progress_reward_weights_by_metric,
        params={
            "metric_name": "full_jump_success_rate",
            "reward_weight_configs": {
                "attitude_landing": {
                    "initial_weight": 0.1,
                    "target_weight": 0.5,
                    "metric_threshold": 0.8,
                    "metric_start": 0.2,
                },
                "attitude_landing_trans": {
                    "initial_weight": 30.0,
                    "target_weight": 100.0,
                    "metric_threshold": 0.8,
                    "metric_start": 0.2,
                },
                "feet_height": {
                    "initial_weight": 0.1,
                    "target_weight": 1.0,
                    "metric_threshold": 0.9,
                    "metric_start": 0.3,
                },
                "landing_base_height": {
                    "initial_weight": 0.05,
                    "target_weight": 0.3,
                    "metric_threshold": 0.9,
                    "metric_start": 0.3,
                },
            },
            "min_steps_between_updates": 200,
            "smoothing_factor": 0.05,
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
            eye=[-0.02, 0.55, -0.03],
            lookat=[-0.02, 0, -0.05],
            origin_type="asset_root",
            asset_name="robot",
            resolution=(1280, 720),
        )
        
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
        
