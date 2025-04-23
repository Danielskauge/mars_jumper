import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp
from isaaclab.envs.common import ViewerCfg
from terms import rewards as custom_rewards
from terms import observations
from robot.robot_cfg import MarsJumperRobotCfg
from terms import events as custom_events
from terms import terminations as custom_terminations

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the attitude stabilization scene."""
    robot: ArticulationCfg = MarsJumperRobotCfg()
    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/robot/.*", history_length=1, track_air_time=True)
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
        #base_height = ObservationTermCfg(func=mdp.base_pos_z, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        #base_quat = ObservationTermCfg(func=mdp.root_quat_w, params={"make_quat_unique": True}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_rotation_vector = ObservationTermCfg(func=observations.base_rotation_vector)
        joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        previous_actions = ObservationTermCfg(func=mdp.last_action)
        has_taken_off = ObservationTermCfg(func=observations.has_taken_off)
        

        def __post_init__(self):
            self.enable_corruption = False 
            self.concatenate_terms = True 

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Event terms can modify anything about the environment."""
    reset_scene_to_default = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
    reset_root_state_uniform = EventTermCfg(func=mdp.reset_root_state_uniform, mode="reset", 
                                        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                                                "pose_range": 
                                                    {
                                                        "z": (1.0, 1.0),
                                                        "roll": (-np.pi/2, np.pi/2),
                                                        "pitch": (-np.pi/2, np.pi/2),
                                                        "yaw": (-np.pi/2, np.pi/2)
                                                    },
                                                "velocity_range": 
                                                    {
                                                        "roll": (-np.pi/4, np.pi/4),
                                                        "pitch": (-np.pi/4, np.pi/4),
                                                        "yaw": (-np.pi/4, np.pi/4)
                                                    }})

    reset_joints_by_offset = EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", 
                                         params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                                                 "position_range": (-np.pi/2, np.pi/2),
                                                 "velocity_range": (-np.pi/4, np.pi/4)})

@configclass
class RewardsCfg:
    """Rewards are computed at each step of the environment."""
    attitude_rotation = RewardTermCfg(func=custom_rewards.attitude_rotation_magnitude, 
                                      params={"kernel": "inverse_quadratic", "scale": 11.0}, 
                                      weight=1)
    
    root_ang_vel_l1 = RewardTermCfg(func=custom_rewards.ang_vel_l1, 
                                    weight=-0.0) #-0.001
    #change_joint_direction = RewardTermCfg(func=custom_rewards.change_joint_direction_penalty, weight=-0.01)
    joint_vel_l2 = RewardTermCfg(func=mdp.joint_vel_l2,
                                 params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
                                 weight=-0.0,
    )
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, 
                                   weight=-0.0)
    
    dof_pos_limits = RewardTermCfg(func=mdp.joint_pos_limits, 
                                   weight=-0.001)
    
    undesired_contacts = RewardTermCfg(func=mdp.contact_forces,
                                       weight=-0.001, #-0.1,
                                       params={"sensor_cfg": 
                                        SceneEntityCfg(
                                            name="contact_forces", 
                                            body_names=[".*THIGH.*", ".*SHANK.*", ".*HIP.*", ".*base.*"]
                                        ), 
                                        "threshold": 0.1})
    
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
        
@configclass
class AttitudeEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for attitude stabilization environment."""
    scene: MySceneCfg = MySceneCfg(num_envs=1024*2, env_spacing=2)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    is_finite_horizon: bool = True
    

    def __post_init__(self):
        self.real_time_control_dt = 1/100
        self.sim.dt = 1/400
        self.viewer = ViewerCfg(
            eye=[0, 0.5, 0.04],
            lookat=[0, 0, 0],
            origin_type="asset_root",
            asset_name="robot",
            resolution=(1280, 720),
        )
        
        self.episode_length_s = 5
        self.success_angle_threshold = np.pi/16
        self.success_duration_threshold = 1.0   
        
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.decimation = int(self.real_time_control_dt / self.sim.dt)
 
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt 