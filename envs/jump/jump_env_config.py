from __future__ import annotations
from typing import Dict

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg
from robot.mars_jumper_robot import MarsJumperRobotConfig
from initialization import InitializationScheme

DEG2RAD = torch.pi / 180.0

@configclass
class JumpEnvCfg(DirectRLEnvCfg):
    episode_length_s = 20.0
    decimation = 1
    action_scale = 0.1
    action_space = 12 #Number of joints. Needed for DirectRLEnv
    observation_space = 24
    state_space = 0 #TODO: Why set to 0?. Actually, fix all these.

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=1,
        disable_contact_processing=True,
        gravity=(0.0, 0.0, -3.72),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0, #TODO: Check if this is correct
        ),
    )
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1, #TODO: Check if this is correct
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0, #TODO: Check if this is correct
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot: ArticulationCfg = MarsJumperRobotConfig()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/mars_jumper_robot/.*",
        history_length=5, #TODO: Check if this is correct
        update_period=0.005, #TODO: Check if this is correct
        track_air_time=True,
    )

    # initializer
    scheme_fraqs: Dict[str, float] = {
        #InitializationScheme.STANDING.name: 0.71,
        InitializationScheme.DEFAULT.name: 0.04,
        #InitializationScheme.INFLIGHT.name: 0.025,
        #InitializationScheme.TOUCHDOWN.name: 0.20,
        #InitializationScheme.LANDED.name: 0.025,
    }

    # curriculum
    num_scheme_curriculums: Dict[str, int] = {
        #InitializationScheme.STANDING.name: 2,
        InitializationScheme.DEFAULT.name: 1,
        #InitializationScheme.INFLIGHT.name: 1,
        #InitializationScheme.TOUCHDOWN.name: 1,
        #InitializationScheme.LANDED.name: 2,
    }
    
    command_curriculum_limits: Dict[str, tuple[int, int]] = {
        InitializationScheme.STANDING.name: (0, 3),
        InitializationScheme.DEFAULT.name: (0, 3),
        InitializationScheme.INFLIGHT.name: (0, 3),
        InitializationScheme.TOUCHDOWN.name: (0, 3),
        InitializationScheme.LANDED.name: (0, 3),
    }

    num_command_curriculums: Dict[str, int] = {
        #InitializationScheme.STANDING.name: 4,
        InitializationScheme.DEFAULT.name: 3,
        #InitializationScheme.INFLIGHT.name: 4,
        #InitializationScheme.TOUCHDOWN.name: 4,
        #InitializationScheme.LANDED.name: 4,
    }

    curriculum_threshold: float = 0.15  # meters
    num_games_per_level: float = 5
    
    # reward scales
    goal_pos_error_large_reward_scale = 10.0
    goal_pos_error_small_reward_scale = 10.0
    speed_on_goal_reward_scale = 20.0
    angvel_reward_scale = 1.0
    break_torque_reward_scale = 2.0
    break_acc_reward_scale = 5.0
    takeoff_paws_pos_reward_scale = 5.0
    attitude_error_reward_scale = 1.0
    landstand_paws_reward_scale = 1.0
    landstand_lin_vel_reward_scale = 8.0
    landstand_joint_vel_reward_scale = 5.0
    landstand_orientation_reward_scale = 2.0
    landstand_joint_pos_reward_scale = 5.0
    landstand_joint_acc_reward_scale = 5.0
    # regularization rewards
    action_clip_reward_scale = -1e-1
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    contact_change_reward_scale = 0
    jerk_reward_scale = 0  # -1e-1
    symmetry_reward_scale = 8.0e-1
