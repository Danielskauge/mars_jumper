
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
        base_rotation_vector = ObservationTermCfg(func=observations.base_rotation_vector)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        previous_actions = ObservationTermCfg(func=mdp.last_action)        
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 

    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    reset_root_state_uniform = EventTermCfg(func=mdp.reset_root_state_uniform, 
                                          mode="reset", 
                                          params={"pose_range": {"x": (-0.0, 0.0), "y": (0.0, 0.0), "z": (0.5, 0.5)}, 
                                                  "velocity_range": {"x": (-0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)}, 
                                                  "asset_cfg": SceneEntityCfg("robot")})
    reset_joints_by_offset = EventTermCfg(func=mdp.reset_joints_by_offset, 
                                          mode="reset", 
                                          params={"position_range": (np.pi/2, np.pi/2), 
                                                  "velocity_range": (0.0, 0.0), 
                                                  "asset_cfg": SceneEntityCfg("robot")})

@configclass
class RewardsCfg:
    """
    Rewards are computed at each step of the environment (which can include multiple physics steps). 
    There is no built in implementation of per-episode rewards.
    """
    is_alive = RewardTermCfg(func=rewards.is_alive, weight=5.0)
    
    # flat_orientation = RewardTermCfg(func=rewards.flat_orientation,
    #                                  params={"phases": [Phase.LANDING]},
    #                                  weight= -0.1)

    # action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, 
    #                                weight= -0.1) #-0.01)
    
    # joint_torque_l2 = RewardTermCfg(func=mdp.joint_torques_l2,
    #                                 weight= -0.1)
    
    
    is_terminated = RewardTermCfg(func=mdp.is_terminated_term, weight= -1, params={"term_keys": ["base_contact", "bad_orientation"]})
    
    # dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
    #                                weight=-0.01)
    
    # undesired_contacts = RewardTermCfg(func=mdp.contact_forces,
    #                                    weight=-0.01, #-0.1,
    #                                    params={"sensor_cfg": 
    #                                        SceneEntityCfg(
    #                                            name="contact_forces", 
    #                                            body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), 
    #                                        "threshold": 0.01},)
    
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(func=mdp.illegal_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]), "threshold": 0.2})
    bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="base"), "limit_angle": np.pi/4})
    # self_collision = TerminationTermCfg(func=custom_terminations.self_collision, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "threshold": 1.0}) #TODO: implement

@configclass
class LandingEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=2)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    is_finite_horizon: bool = True

    def __post_init__(self):
        
        self.is_finite_horizon = True

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
        
