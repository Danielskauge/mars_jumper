# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
from typing import Tuple

from isaaclab.managers.manager_term_cfg import CurriculumTermCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import rewards as custom_rewards
from terms import events as custom_events
from terms import curriculums as custom_curriculums
from terms import observations as custom_observations
from terms.modifiers import RunningStatsNormalizerCfg
from terms import terminations as custom_terminations
#import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from robot.robot_cfg import MarsJumperRobotCfg  # isort: skip

MAX_EPISODE_LENGTH_S = 5.0
DEG2RAD = np.pi/180

@configclass
class CommandRangesCfg:
    """Configuration for command ranges."""
    initial_pitch_range: Tuple[float, float] = (0.0, 0.0) # Initial pitch range
    initial_magnitude_range: Tuple[float, float] = (1.0, 1.0) # Initial magnitude range
    
    final_pitch_range: Tuple[float, float] = (0.0, 0.0) # Final pitch range
    final_magnitude_range: Tuple[float, float] = (4.0, 5.0) # Final magnitude range
    
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
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/.*", history_length=5, track_air_time=True)
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
    """
    Actions are resampled at each step of the environment.
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", debug_vis=True, joint_names=[".*"], use_default_offset=True) 
    
@configclass
class ObservationsCfg:
    """
    Observations are sampled at each step of the environment.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observation term configs for policy group."""
        base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_quat = ObservationTermCfg(func=mdp.root_quat_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        
        previous_actions = ObservationTermCfg(func=mdp.last_action)
        #previous_torque = ObservationTermCfg(func=mdp.applied_torque) fix this maybe
        
        has_taken_off = ObservationTermCfg(func=custom_observations.has_taken_off)
        
        command_vec = ObservationTermCfg(func=custom_observations.takeoff_vel_vec_cmd, noise=Unoise(n_min=-0.01, n_max=0.01))

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

    # startup
   
    # randomize_rigid_body_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 0.8),
    #         "dynamic_friction_range": (0.5, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # randomize_base_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    
    # randomize_joint_friction = EventTermCfg(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "friction_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )
    # reset
    

    reset_robot_takeoff_state = EventTermCfg(
        func=custom_events.set_phase_to_takeoff,
        mode="reset",
    )

@configclass
class RewardsCfg:
    """
    Rewards are computed at each step of the environment (which can include multiple physics steps). 
    There is no built in implementation of per-episode rewards.
    """
    #LANDING PHASE ONLY
    landing_com_accel = RewardTermCfg(
        func=custom_rewards.landing_com_accel, 
        weight=0.0
    )
    feet_ground_impact_force = RewardTermCfg(
        func=custom_rewards.feet_ground_impact_force,
        weight=0.0,
    )
    #NOT FLIGHT PHASE
    left_right_joint_symmetry = RewardTermCfg(
        func=custom_rewards.left_right_joint_symmetry,
        weight=0.1,
    )
    #CROUCH PHASE ONLY
    crouch_knee_angle = RewardTermCfg(
        func=custom_rewards.crouch_knee_angle,
        weight=0.0, #positive
        params={"target_angle_rad": torch.pi*0.8, "reward_type": "cosine"},
    )
    crouch_hip_angle = RewardTermCfg(
        func=custom_rewards.crouch_hip_angle,
        weight=0.0, #positive
        params={"target_angle_rad": -torch.pi*0.4, "reward_type": "cosine"},
    )
    crouch_abductor_angle = RewardTermCfg(
        func=custom_rewards.crouch_abductor_angle,
        weight=0.0, #positive
        params={"target_angle_rad": 0.0, "reward_type": "cosine"},
    )
    #CROUCH AND LANDING PHASES
    feet_ground_contact = RewardTermCfg(
        func=custom_rewards.feet_ground_contact,
        weight=0, #0.01, #positive
    )
    flat_orientation = RewardTermCfg(
        func=custom_rewards.flat_orientation,
        weight=-0.1,
    )
    #TAKEOFF AND LANDING PHASES
    equal_force_distribution = RewardTermCfg(
        func=custom_rewards.equal_force_distribution,
        weight=0.0,
    )
    #TAKEOFF PHASE ONLY
    takeoff_vel_vec_magnitude = RewardTermCfg(
        func=custom_rewards.takeoff_vel_vec_magnitude,
        weight=0.0,
    )
    takeoff_vel_vec_angle = RewardTermCfg(
        func=custom_rewards.takeoff_vel_vec_angle,
        weight=0.0, 
    )
    #ALWAYS ACTIVE
    joint_acc_l2 = RewardTermCfg(
        func=mdp.joint_acc_l2, #TODO: should make a custom one not active in takeoff phase
        weight= 0, #-2.5e-7
    )
    joint_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,  #TODO: should make a custom one not active in takeoff phase
        weight=0.0, #-1e-5
    )
    action_rate_l2 = RewardTermCfg(
        func=mdp.action_rate_l2, #TODO: should make a custom one not active in takeoff phase
        weight= -0.1,
    )
    is_terminated = RewardTermCfg(
        func=mdp.is_terminated_term,
        weight= -0.0, #-0.1,
    )
    # feet_slide = RewardTermCfg(
    #     func=custom_rewards.feet_slide,
    #     weight=-0.5,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT.*")
    #     },
    # )
    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits, #returns stricly non-negative values
        weight=0, #-0.1
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.contact_forces,
        weight=0.0, #-0.1,
        params={"sensor_cfg": 
                SceneEntityCfg(
                    name="contact_forces", 
                    body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]
                ), 
                "threshold": 0.5
        },
    )
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0})
    reached_takeoff_height = TerminationTermCfg(func=custom_terminations.reached_takeoff_height, params={"height_threshold": 0.20})
    #bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": np.pi/4})
    # self_collision = TerminationTermCfg(func=custom_terminations.self_collision, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0}) #TODO: implement
    landed = TerminationTermCfg(func=custom_terminations.landed)
@configclass
class CurriculumCfg:
    """Curriculum terms."""

    command_range_progression = CurriculumTermCfg(
        func=custom_curriculums.progress_command_ranges,
        params={
            "num_curriculum_levels": 10,
        },
    )
        
@configclass
class MarsJumperEnvCfg(ManagerBasedRLEnvCfg):

    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=1)
    command_ranges: CommandRangesCfg = CommandRangesCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):

        #Takeoff success criteria
        self.takeoff_magnitude_ratio_error_threshold = 0.1
        self.takeoff_angle_error_threshold = 10*DEG2RAD
        
        self.real_time_control_dt = 1/120
        self.sim.dt = 1/480 #Physics time step, also the torque update rate
        self.episode_length_s = MAX_EPISODE_LENGTH_S
        self.viewer = ViewerCfg(
            eye=[0.5, 0.5, 0.5],
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

