"""Custom event functions for the Mars Jumper environment."""

from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING, Tuple
import numpy as np
from isaaclab.envs.mdp.events import reset_scene_to_default, reset_joints_by_offset, reset_root_state_uniform
from terms.utils import Phase
import torch
from terms.utils import convert_height_length_to_pitch_magnitude, convert_pitch_magnitude_to_vector
from isaaclab.utils.math import sample_uniform, quat_from_euler_xyz
import isaaclab.utils.math as math_utils
from typing import Dict
from isaaclab.envs import ManagerBasedRLEnv
from reset_crouch import sample_robot_crouch_pose
    
def reset_robot_pre_landing_state(env: ManagerBasedRLEnv, 
                                 env_ids: Sequence[int],
                                 height_range: Tuple[float, float] = (0., 0.0),
                                 root_lin_vel_range: Dict[str, Tuple[float, float]] = {"x": (-0.0, 0.0), "z": (-0.0, 0.0), "y": (0.0, 0.0)},
                                 base_euler_angle_max_deg: float = 0.0,
                                 base_ang_vel_max_deg: float = 0.0,
                                 joint_pos_max_limit_ratio: float = 0.0,
                                 joint_vel_max_deg: float = 0.0,
                                 ) -> None:
    
    reset_scene_to_default(env, env_ids)
    height = math_utils.sample_uniform(*height_range, (len(env_ids)), device=env.device) #Shape: (num_envs)
    reset_robot_attitude_state(env, env_ids, 
                               height=height, 
                               base_euler_angle_max_deg=base_euler_angle_max_deg,
                               base_ang_vel_max_deg=base_ang_vel_max_deg,
                               joint_pos_max_limit_ratio=joint_pos_max_limit_ratio,
                               joint_vel_max_deg=joint_vel_max_deg)
    
    x_vel_range, z_vel_range, y_vel_range = root_lin_vel_range["x"], root_lin_vel_range["z"], root_lin_vel_range["y"]
    x_vel = math_utils.sample_uniform(*x_vel_range, (len(env_ids)), device=env.device) #Shape: (num_envs)
    z_vel = math_utils.sample_uniform(*z_vel_range, (len(env_ids)), device=env.device) #Shape: (num_envs)
    y_vel = math_utils.sample_uniform(*y_vel_range, (len(env_ids)), device=env.device) #Shape: (num_envs)
    
    root_state = env.robot.data.root_state_w[env_ids].clone()
    root_state[:, 7] = x_vel
    root_state[:, 9] = z_vel
    root_state[:, 8] = y_vel
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    

    
def reset_robot_attitude_state(env: ManagerBasedRLEnv,
                               env_ids: Sequence[int],
                               height: float,
                               base_euler_angle_max_deg: float = 30.0, # max angle to sample from for roll, pitch, yaw
                               base_ang_vel_max_deg: float = 0.5, # max angular velocity (deg/s) to sample from for roll, pitch, yaw
                               joint_pos_max_limit_ratio: float = 0.1, # Ratio of joint range to sample from, centered on the range center
                               joint_vel_max_deg: float = 0.5, # Joint velocity range (rad/s)
                               ) -> None:
    """Resets the robot's base attitude state to a random orientation and angular velocity, and joints to random positions and velocities."""
    num_envs_to_reset = len(env_ids)

    # -- Base state --
    # Default root state (typically origin or initial height, zero velocity)
    root_state = env.robot.data.default_root_state[env_ids].clone()
    root_state[:, 2] = height  # Set z height

    # Random orientation
    max_angle_rad = torch.deg2rad(torch.tensor(base_euler_angle_max_deg, device=env.device))
    # Sample roll and pitch within limits
    roll = sample_uniform(-max_angle_rad, max_angle_rad, (num_envs_to_reset,), device=env.device)
    pitch = sample_uniform(-max_angle_rad, max_angle_rad, (num_envs_to_reset,), device=env.device)
    yaw = sample_uniform(-torch.pi, torch.pi, (num_envs_to_reset,), device=env.device)
    random_quat_w = quat_from_euler_xyz(roll, pitch, yaw)
    root_state[:, 3:7] = random_quat_w

    # Random angular velocity
    max_ang_vel_rad = torch.deg2rad(torch.tensor(base_ang_vel_max_deg, device=env.device))
    random_ang_vel = sample_uniform(-max_ang_vel_rad, max_ang_vel_rad, (num_envs_to_reset, 3), device=env.device)
    root_state[:, 10:13] = random_ang_vel

    # Set root state
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    # -- Joint state --
    # Default joint positions
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()

    # Random joint positions within ratio of limits
    lower_limits = env.robot.data.soft_joint_pos_limits[env_ids, :, 0]
    upper_limits = env.robot.data.soft_joint_pos_limits[env_ids, :, 1]
    joint_range = (upper_limits - lower_limits) * joint_pos_max_limit_ratio
    mid_pos = (upper_limits + lower_limits) / 2
    joint_pos = sample_uniform(mid_pos - joint_range/2, mid_pos + joint_range/2, joint_pos.shape, device=env.device)

    # Random joint velocities
    max_joint_vel_rad = torch.deg2rad(torch.tensor(joint_vel_max_deg, device=env.device))
    joint_vel = sample_uniform(-max_joint_vel_rad, max_joint_vel_rad, joint_vel.shape, device=env.device)

    # Set joint state
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def init_robot_in_takeoff_and_flight_phases(env: ManagerBasedRLEnv, 
                              env_ids: Sequence[int], 
                              flight_phase_ratio: float,
                              ) -> None:
    
    num_flight_envs = int(flight_phase_ratio * len(env_ids))
    flight_ids = env_ids[:num_flight_envs]
    takeoff_ids = env_ids[num_flight_envs:]
    
    reset_scene_to_default(env, takeoff_ids) #takeoff phase
    env.jump_phase[takeoff_ids] = Phase.TAKEOFF
    
    init_robot_in_flight_phase(env, flight_ids)
    env.jump_phase[flight_ids] = Phase.FLIGHT

def init_robot_in_flight_phase(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             ) -> None:
    """Initializes the robot state at a random point along its commanded ballistic trajectory."""
    num_envs_to_reset = len(env_ids)
    device = env.device

    # Convert height/length to pitch/magnitude for ballistic calculations
    height = env.target_height[env_ids]
    length = env.target_length[env_ids]
    pitch, magnitude = convert_height_length_to_pitch_magnitude(height, length, gravity=9.81)
    cmd_vel_cartesian = convert_pitch_magnitude_to_vector(pitch, magnitude) # Shape: (num_envs, 3)
    
    x_dot_0 = cmd_vel_cartesian[:, 0] # Shape: (num_envs)
    z_dot_0 = cmd_vel_cartesian[:, 2] # Shape: (num_envs)

    assert torch.all(z_dot_0 > 0), f"Commanded initial vertical velocity z_dot_0 must be positive: {z_dot_0}"
    
    g = env.cfg.sim.gravity[2] # Negative value
    z_start = math_utils.sample_uniform(0.1, 0.15, (num_envs_to_reset,), device=device) # Initial height at liftoff
    x_start = math_utils.sample_uniform(0.0,0.5, (num_envs_to_reset,), device=device)
    
    min_z_ascending = 0.20 
    min_z_descending = 0.30

    delta_z = z_start - min_z_ascending
    discriminant = z_dot_0**2 - 4 * (0.5 * g) * delta_z # b^2 - 4ac
    discriminant = torch.clamp(discriminant, min=1e-6) # Ensure positive for sqrt
    assert torch.all(discriminant > 0), f"Commanded velocity z_dot_0 {z_dot_0} from z_start {z_start} insufficient to reach min_sampling_z {min_z_ascending}. Discriminant: {discriminant}"
    t_sample_start = (-z_dot_0 + torch.sqrt(discriminant)) / g # Smaller positive root (g is negative)

    delta_z = z_start - min_z_descending
    discriminant = z_dot_0**2 - 4 * (0.5 * g) * delta_z # b^2 - 4ac
    discriminant = torch.clamp(discriminant, min=1e-6) # Ensure positive for sqrt
    assert torch.all(discriminant > 0), f"Commanded velocity z_dot_0 {z_dot_0} from z_start {z_start} insufficient to reach max_sampling_z {min_z_descending}. Discriminant: {discriminant}"
    t_sample_end = (-z_dot_0 - torch.sqrt(discriminant)) / g # Larger positive root (g is negative)

    assert t_sample_start > 0, f"Time range is invalid. t_start: {t_sample_start}, t_end: {t_sample_end}"
    assert t_sample_end > 0, f"Time range is invalid. t_start: {t_sample_start}, t_end: {t_sample_end}"
    assert t_sample_end > t_sample_start, f"Time range is invalid. t_start: {t_sample_start}, t_end: {t_sample_end}"
    
    sample_time = math_utils.sample_uniform(t_sample_start, t_sample_end, (num_envs_to_reset,), device=device) # Shape: (num_envs)

    x_sample = x_start + x_dot_0 * sample_time 
    z_sample = z_start + z_dot_0 * sample_time + 0.5 * g * sample_time**2 
    
    x_dot_sample = x_dot_0 # Horizontal velocity remains constant
    z_dot_sample = z_dot_0 + g * sample_time # Vertical velocity changes due to gravity
    assert torch.all(z_sample > 0)
    
    pose_range = {"x": (x_start, x_start), 
                  "z": (z_start, z_start),
                  "roll": (np.deg2rad(-10), np.deg2rad(10)),
                  "pitch": (np.deg2rad(-45), np.deg2rad(20)),
                  "yaw": (np.deg2rad(-10), np.deg2rad(10))}
    
    velocity_range = {"x": (x_dot_0, x_dot_0), 
                      "z": (z_dot_0, z_dot_0),
                      "roll": (np.deg2rad(-10), np.deg2rad(10)),
                      "pitch": (np.deg2rad(-20), np.deg2rad(20)),
                      "yaw": (np.deg2rad(-10), np.deg2rad(10))}

    
    reset_root_state_uniform(env, env_ids, pose_range=pose_range, velocity_range=velocity_range)
    reset_joints_by_offset(env, env_ids, position_range=(np.deg2rad(-20), np.deg2rad(20)), velocity_range=(np.deg2rad(-10), np.deg2rad(10)))    


def set_joint_limits_from_config(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> None:
    """Sets the joint position limits based on values defined in the robot's config."""
    
    # Ensure the robot config has the expected limit attributes
    if not hasattr(env.robot.cfg, "HIP_ABDUCTION_ANGLE_LIMITS_RAD") or \
       not hasattr(env.robot.cfg, "KNEE_ANGLE_LIMITS_RAD") or \
       not hasattr(env.robot.cfg, "HIP_FLEXION_ANGLE_LIMITS_RAD"):
        logger.error("Robot config is missing expected joint limit attributes. Skipping limit setting.")
        return

    # Get current joint limits (shape: (num_envs, num_joints, 2))
    current_limits = env.robot.data.joint_limits.clone()

    # Find joint indices
    haa_joint_idx, _ = env.robot.find_joints(env.robot.cfg.HIP_ABDUCTION_JOINTS_REGEX)
    hfe_joint_idx, _ = env.robot.find_joints(env.robot.cfg.HIP_FLEXION_JOINTS_REGEX) # Corrected regex source
    kfe_joint_idx, _ = env.robot.find_joints(env.robot.cfg.KNEE_JOINTS_REGEX)       # Corrected regex source

    # Apply limits from config
    # Note: Limits are often defined as (min, max), but PhysX expects (lower, upper)
    # Ensure the order matches PhysX expectations. The write_joint_limits_to_sim handles this internally usually.
    # We assume the config defines limits as (lower, upper) matching PhysX.
    if haa_joint_idx:
        current_limits[:, haa_joint_idx, 0] = env.robot.cfg.HIP_ABDUCTION_ANGLE_LIMITS_RAD[0]
        current_limits[:, haa_joint_idx, 1] = env.robot.cfg.HIP_ABDUCTION_ANGLE_LIMITS_RAD[1]
    if hfe_joint_idx:
        current_limits[:, hfe_joint_idx, 0] = env.robot.cfg.HIP_FLEXION_ANGLE_LIMITS_RAD[0]
        current_limits[:, hfe_joint_idx, 1] = env.robot.cfg.HIP_FLEXION_ANGLE_LIMITS_RAD[1]
    if kfe_joint_idx:
        # Be careful with KFE limits if they are defined (max, min) in config
        lower = min(env.robot.cfg.KNEE_ANGLE_LIMITS_RAD)
        upper = max(env.robot.cfg.KNEE_ANGLE_LIMITS_RAD)
        current_limits[:, kfe_joint_idx, 0] = lower
        current_limits[:, kfe_joint_idx, 1] = upper

    # Write the updated limits to the simulation
    # Using warn_limit_violation=False as this happens at startup before resets typically
    env.robot.write_joint_limits_to_sim(current_limits, warn_limit_violation=False)
    logger.info("Successfully set joint limits from robot configuration.")


def reset_robot_crouch_state(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             hip_angle_range_rad: Tuple[float, float],
                             ) -> None:
    
    
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()
    
    hip_joint_idx, _ = env.robot.find_joints(".*HFE")
    knee_joint_idx, _ = env.robot.find_joints(".*KFE")
    
    hip_flexor_angle = torch.empty(len(env_ids), device=env.device).uniform_(*hip_angle_range_rad) #Shape: (num_envs)
    knee_flexor_angle = -hip_flexor_angle * 2
    
    joint_pos[:, hip_joint_idx] = hip_flexor_angle.unsqueeze(-1) #Shape: (num_envs, 1)
    joint_pos[:, knee_joint_idx] = knee_flexor_angle.unsqueeze(-1) 
    
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    hip_link_len = env.robot.cfg.HIP_LINK_LENGTH
    knee_link_len = env.robot.cfg.KNEE_LINK_LENGTH
    
    root_z: torch.Tensor = hip_link_len * torch.cos(hip_flexor_angle) + \
            knee_link_len * torch.cos(knee_flexor_angle + hip_flexor_angle) + 0.01
             
    root_state = env.robot.data.default_root_state[env_ids].clone()
    
    root_state[:, :3] = env.scene.env_origins[env_ids]
    
    root_state[:, 2] = root_z
    
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    env._phase_buffer[env_ids] = Phase.CROUCH
    #print("Reset Event: Reset to crouch state of envs %s", env_ids)
    
def reset_robot_pose_with_feet_on_ground(env: ManagerBasedRLEnv, 
                             env_ids: Sequence[int],
                             base_height_range: Tuple[float, float],
                             base_pitch_range_rad: Tuple[float, float],
                             front_foot_x_offset_range_cm: Tuple[float, float],
                             hind_foot_x_offset_range_cm: Tuple[float, float],
                             base_vertical_vel_range: Tuple[float, float] = (0.0, 0.0),
                             ) -> None:
    
    # # Base pose (position x, y are from default, z is sampled height)
    base_height, base_pitch, \
    front_hip_angle, front_knee_angle, \
    hind_hip_angle, hind_knee_angle = sample_robot_crouch_pose(
                                                                base_height_range=base_height_range,
                                                                base_pitch_range_rad=base_pitch_range_rad,
                                                                front_foot_x_offset_range_cm=front_foot_x_offset_range_cm,
                                                                hind_foot_x_offset_range_cm=hind_foot_x_offset_range_cm,
                                                                device=env.device,
                                                                num_envs=len(env_ids)
                                                            )
    base_quat_w = quat_from_euler_xyz(torch.zeros_like(base_pitch), base_pitch, torch.zeros_like(base_pitch))
    
    # Sample base vertical velocity
    base_z_vel = math_utils.sample_uniform(*base_vertical_vel_range, (len(env_ids),), device=env.device)
    
    root_state = env.robot.data.default_root_state[env_ids].clone() 
    root_state[:, :2] = env.scene.env_origins[env_ids, :2] + env.robot.data.default_root_state[env_ids, :2]

    root_state[:, 2] = base_height
    root_state[:, 3:7] = base_quat_w
    root_state[:, 7:] = 0.0  # Zero linear and angular velocities
    root_state[:, 9] = base_z_vel  # Set base vertical velocity (z-component)
    
    joint_pos = env.robot.data.default_joint_pos[env_ids].clone()

    # Get joint indices using the robot's find_joints method
    # The [0] extracts the tensor of indices from the (indices, names) tuple.
    rf_hfe_idx = env.robot.find_joints("RF_HFE")[0]
    lf_hfe_idx = env.robot.find_joints("LF_HFE")[0]
    rh_hfe_idx = env.robot.find_joints("RH_HFE")[0]
    lh_hfe_idx = env.robot.find_joints("LH_HFE")[0]

    rf_kfe_idx = env.robot.find_joints("RF_KFE")[0]
    lf_kfe_idx = env.robot.find_joints("LF_KFE")[0]
    rh_kfe_idx = env.robot.find_joints("RH_KFE")[0]
    lh_kfe_idx = env.robot.find_joints("LH_KFE")[0]

    joint_pos[:, rf_hfe_idx] = front_hip_angle
    joint_pos[:, lf_hfe_idx] = front_hip_angle
    joint_pos[:, rh_hfe_idx] = hind_hip_angle
    joint_pos[:, lh_hfe_idx] = hind_hip_angle

    joint_pos[:, rf_kfe_idx] = front_knee_angle
    joint_pos[:, lf_kfe_idx] = front_knee_angle
    joint_pos[:, rh_kfe_idx] = hind_knee_angle
    joint_pos[:, lh_kfe_idx] = hind_knee_angle
    
    joint_vel = env.robot.data.default_joint_vel[env_ids].clone()
    
    env.robot.write_root_state_to_sim(root_state, env_ids=env_ids) 
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def randomize_actuator_gains(env, env_ids, asset_cfg,
                             stiffness_distribution_params=None,
                             damping_distribution_params=None,
                             operation="scale",
                             distribution="uniform"):
    """Randomize actuator PD gains by sampling one scale factor per actuator per env."""
    from isaaclab.assets import Articulation
    from isaaclab.actuators import ImplicitActuator
    asset: Articulation = env.scene[asset_cfg.name]
    # prepare environment ids tensor
    if env_ids is None:
        env_ids_tensor = torch.arange(env.num_envs, device=asset.device)
    else:
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=asset.device) if not torch.is_tensor(env_ids) else env_ids.to(asset.device)
    # loop through each actuator
    for actuator in asset.actuators.values():
        # determine joint indices for this actuator
        if isinstance(actuator.joint_indices, slice):
            idx = slice(None)
        else:
            idx = torch.tensor(actuator.joint_indices, device=asset.device)
        # randomize stiffness
        if stiffness_distribution_params is not None:
            low, high = stiffness_distribution_params
            factors = math_utils.sample_uniform(low, high, (len(env_ids_tensor), 1), device=asset.device)
            stiff = actuator.stiffness[env_ids_tensor].clone()
            stiff[:, idx] = asset.data.default_joint_stiffness[env_ids_tensor][:, idx]
            stiff[:, idx] *= factors
            actuator.stiffness[env_ids_tensor] = stiff
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiff, joint_ids=actuator.joint_indices, env_ids=env_ids_tensor)
        # randomize damping
        if damping_distribution_params is not None:
            low, high = damping_distribution_params
            factors = math_utils.sample_uniform(low, high, (len(env_ids_tensor), 1), device=asset.device)
            damp = actuator.damping[env_ids_tensor].clone()
            damp[:, idx] = asset.data.default_joint_damping[env_ids_tensor][:, idx]
            damp[:, idx] *= factors
            actuator.damping[env_ids_tensor] = damp
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damp, joint_ids=actuator.joint_indices, env_ids=env_ids_tensor)
    
def randomize_spring_stiffness(
    env, env_ids, asset_cfg,
    spring_distribution_params: tuple[float, float] = (0.8, 1.2)
) -> None:
    """Randomize spring stiffness for parallel elastic knee actuators."""
    from robot.actuators.actuators import ParallelElasticActuator
    asset = env.scene[asset_cfg.name]
    # sample one factor
    low, high = spring_distribution_params
    factor = float(math_utils.sample_uniform(low, high, (1,), device='cpu').item())
    # apply to each parallel elastic actuator
    for actuator in asset.actuators.values():
        if isinstance(actuator, ParallelElasticActuator):
            actuator.cfg.spring_stiffness *= factor
    
    