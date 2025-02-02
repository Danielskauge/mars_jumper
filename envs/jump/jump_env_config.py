from __future__ import annotations
from typing import Dict

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets.articulation import ArticulationCfg
from mars_jumper.robot.MarsJumperRobot import MarsJumperConfig
from initialization import InitializationScheme

DEG2RAD = torch.pi / 180.0

@configclass
class JumpEnvCfg(DirectRLEnvCfg):
    episode_length_s = 5.0
    decimation = 8
    action_scale = 125 * DEG2RAD
    action_space = 12 #Number of joints. Needed for DirectRLEnv
    observation_space = 38
    state_space = 0 #TODO: Why set to 0?. Actually, fix all these.

    # simulation
    sim_cfg: SimulationCfg = SimulationCfg(
        dt=1 / 480,
        render_interval=decimation,
        disable_contact_processing=True,
        gravity=(0.0, 0.0, -3.72),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain_cfg: TerrainImporterCfg = TerrainImporterCfg(
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

    # scene
    scene_cfg: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )

    # robot
    robot_cfg: ArticulationCfg = MarsJumperConfig()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/mars-jumper/.*",
        history_length=5,
        update_period=0.005,
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
        #InitializationScheme.STANDING.name: 4,
        InitializationScheme.DEFAULT.name: 1,
        #InitializationScheme.INFLIGHT.name: 4,
        #InitializationScheme.TOUCHDOWN.name: 4,
        #InitializationScheme.LANDED.name: 2,
    }

    num_command_curriculums: Dict[str, int] = {
        #InitializationScheme.STANDING.name: 4,
        InitializationScheme.DEFAULT.name: 4,
        #InitializationScheme.INFLIGHT.name: 4,
        #InitializationScheme.TOUCHDOWN.name: 4,
        #InitializationScheme.LANDED.name: 4,
    }

    curriculum_threshold: float = 0.15  # meters
    num_games_per_level: float = 5

    # reward scales
    goal_pos_error_large_reward_scale = 1.0
    goal_pos_error_small_reward_scale = 2.0
    speed_on_goal_reward_scale = 60.0
    angvel_reward_scale = 1.0
    break_torque_reward_scale = 2.0
    takeoff_paw_pos_reward_scale = 1.0
    attitude_error_reward_scale = 1.0
    lanstand_paws_reward_scale = 5.0
    landstand_lin_vel_reward_scale = 5.0
    landstand_joint_vel_reward_scale = 5.0
    landstand_orientation_reward_scale = 5.0
    landstand_joint_pos_reward_scale = 5.0
    # regularization rewards
    action_clip_reward_scale = -1e-1
    joint_torque_reward_scale = -1e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    contact_change_reward_scale = -1e-2
    jerk_reward_scale = 0  # -1e-1
