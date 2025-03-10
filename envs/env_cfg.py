# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import curriculums
from terms import rewards as custom_rewards
from terms import events as custom_events
from terms import observations as custom_observations
from terms.command import TakeoffVelVecCommand, TakeoffVelVecCommandCfg
#import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from robot.robot_cfg import MarsJumperRobotCfg  # isort: skip

##
# Scene definition
##

MAX_EPISODE_LENGTH_S = 20.0

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
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
##
# MDP settings
##

@configclass
class CommandsCfg:
    """
    Commands are sampled at the beginning of each episode, but additionally resampled during episodes based on the resampling_time_range parameter in the CommandTermCfg. 
    The CommandTerm.compute() method is called at each env step and decrements the term-specific time_left counter every time it is called. 
    When time_left reaches zero, the command is resampled.
    """

    takeoff_vel_vec = TakeoffVelVecCommandCfg(
        class_type=TakeoffVelVecCommand,
        asset_name="robot",
        resampling_time_range=(MAX_EPISODE_LENGTH_S+1, MAX_EPISODE_LENGTH_S+1),
        debug_vis=True,
        ranges=TakeoffVelVecCommandCfg.Ranges(
            pitch_rad=(0.0, 0.0), #These will be the initial ranges used for curriculum
            magnitude=(1.0, 2.0), 
        )
    )

@configclass
class ActionsCfg:
    """
    Actions are resampled at each step of the environment.
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1, use_default_offset=True) #TODO understand clip
    

@configclass
class ObservationsCfg:
    """
    Observations are sampled at each step of the environment.
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observation term configs for policy group."""

        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_quat = ObservationTermCfg(func=mdp.root_quat_w, noise=Unoise(n_min=-0.01, n_max=0.01))
        
        # # projected_gravity = ObservationTermCfg(
        # #     func=mdp.projected_gravity,
        # #     noise=Unoise(n_min=-0.05, n_max=0.05),
        # # )
        # joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # actions = ObservationTermCfg(func=mdp.last_action)
        # height_scan = ObservationTermCfg( #TODO: what is this?
        #     func=mdp.base_pos_z,
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     )
        # has_taken_off = ObservationTermCfg(
        #     func=custom_observations.has_taken_off,
        #     params={"height_threshold": 0.15}
        # )
        takeoff_vel_vec_cmd = ObservationTermCfg( #TODO can add observation history direcltiy here
            func=custom_observations.takeoff_vel_vec_cmd,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

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
   
    # physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.7, 0.8),
    #         "dynamic_friction_range": (0.5, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTermCfg(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (0.8, 1.2),
    #         "operation": "scale",
    #     },
    # )
    
    initialize_buffers = EventTermCfg(
        func=custom_events.initialize_buffers,
        mode="startup",
    )

    # reset
    
    #This needs to be placed first among the reset events, as the cmd it samples is used in the state initialization terms
    reset_command = EventTermCfg(
        func=custom_events.reset_command,
        mode="reset",
    )

    reset_robot_initial_state = EventTermCfg(
        func=custom_events.reset_robot_initial_state,
        mode="reset",
        params={
            "crouch_flight_ratio": 1,
            "crouch_flexor_angle_range_rad": (-torch.pi/3, -torch.pi/4),
            "flight_base_euler_angles_range_rad": (-torch.pi/16, torch.pi/16),
            "flight_flexor_angles_range_rad": (-torch.pi/4, torch.pi/4),
            "flight_abductor_angles_range_rad": (-torch.pi/8, torch.pi/8),
        },
    )
    
    reset_episodic_buffers = EventTermCfg(
        func=custom_events.reset_episodic_buffers,
        mode="reset",
    )

    
@configclass
class RewardsCfg:
    """
    Rewards are computed at each step of the environment (which can include multiple physics steps). 
    There is no built in implementation of per-episode rewards.
    """
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, weight=0.0)
    
    undesired_contacts = RewardTermCfg(
        func=mdp.contact_forces,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    
    takeoff_vel_vec_magnitude_error = RewardTermCfg(
        func=custom_rewards.takeoff_vel_vec_magnitude_error,
        weight=1.0,
    )
    takeoff_vel_vec_angle_error = RewardTermCfg(
        func=custom_rewards.takeoff_vel_vec_angle_error,
        weight=0.0,
    )
        
    flat_orientation_l2 = RewardTermCfg(func=mdp.flat_orientation_l2, weight=0.0)
    joint_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=0.0)
    joint_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=0.0)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, weight=0.0)
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": np.pi/4},
    )
    # self_collision = TerminationTermCfg(
    #     func=custom_terminations.self_collision,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0},
    # ) #TODO: implement


@configclass
class CurriculumCfg:
    """Curriculum terms are run all at once by the curriculum manager compute() method, 
    which is called at the _reset_idx() method of the environment."""
    
    # command = CurriculumTermCfg(
    #     func=curriculums.change_command_ranges,
    #     params={
    #         "num_curriculum_steps": 10,
    #         "final_magnitude_range": (4.0, 5.0),
    #         "steps_per_increment": 10,
    #     },
    # )
        
    # domain_randomization_ranges = CurriculumTermCfg(
    #     func=custom_events.domain_randomization_ranges,
    #     params={
    #         "success_threshold": 0.5,
    #     },
    # )
    
##
# Environment configuration
##

@configclass
class MarsJumperEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4, env_spacing=2)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = MAX_EPISODE_LENGTH_S
        # simulation settings
        self.viewer = ViewerCfg(
            eye=[1, 1, 1],
            lookat=[0, 0, 0],
            resolution=(1280, 720),
            origin_type="world",
        )
        
        self.mars_gravity = -3.721
        self.sim.gravity = (0.0, 0.0, self.mars_gravity)
        self.sim.dt = 1/300
        self.sim.render_interval = self.decimation
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

