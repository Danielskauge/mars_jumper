"""Launch Isaac Sim Simulator first."""


import math
import torch
import isaaclab.utils.math as math_utils
from typing import Tuple

def sample_robot_crouch_pose(
    base_height_range: Tuple[float, float],
    base_pitch_range_rad: Tuple[float, float],
    front_foot_x_offset_range_cm: Tuple[float, float],
    hind_foot_x_offset_range_cm: Tuple[float, float],
    device: torch.device,
    num_envs: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates base pitch, and HFE/KFE joint angles for each leg
    such that the feet are on the ground (z=0 in world frame).
    The foot's x position relative to the hip's ground projection is also randomized.
    The base height and pitch are randomized.

    Angle conventions:
    - Returned HFE (hip_angle):
        - 0 degrees: thigh link is vertical, pointing downwards.
        - Positive angles: thigh link moves forward (in robot's +X direction or CCW).
    - Returned KFE (knee_angle):
        - This is the relative angle of the knee joint.
        - 0 degrees: shank is aligned with the thigh (leg is straight).
        - Positive angles: shank moves forward (CCW) relative to the thigh link.

    Args:
        env: The environment instance.
        env_ids: Sequence of environment IDs to process.
        base_height_range: (min, max) range for random base height.
        base_pitch_range_rad: (min, max) range for random base pitch in radians.
        front_foot_x_offset_range_cm: (min, max) range for random front foot x-offset
                                      relative to hip's ground projection, in centimeters.
        hind_foot_x_offset_range_cm: (min, max) range for random hind foot x-offset
                                     relative to hip's ground projection, in centimeters.
        device: PyTorch device.
        num_envs: Number of environments.

    Returns:
        Tuple containing:
        - base_height (torch.Tensor): Shape (num_envs,), base height.
        - base_pitch (torch.Tensor): Shape (num_envs,), base pitch angles.
        - front_hip_angle (torch.Tensor): Shape (num_envs,), HFE joint angle shared by front legs.
        - front_knee_angle (torch.Tensor): Shape (num_envs,), KFE joint angle shared by front legs.
        - hind_hip_angle (torch.Tensor): Shape (num_envs,), HFE joint angle shared by hind legs.
        - hind_knee_angle (torch.Tensor): Shape (num_envs,), KFE joint angle shared by hind legs.
    """

    # 1. Randomize Base Height and Pitch
    base_height = math_utils.sample_uniform(*base_height_range, (num_envs,), device=device)
    base_pitch = math_utils.sample_uniform(*base_pitch_range_rad, (num_envs,), device=device)

    # Check for hip-ground collision and attempt to fix
    # Constants based on HAA joint origins: abs(x_coord) = 0.042, z_coord = -0.006
    _HAA_X_ABS_VAL = 0.042
    _HAA_Z_VAL = -0.006

    def _calculate_min_hip_world_z(current_base_height, current_base_pitch):
        # This formula gives the z-coordinate of the lowest HAA joint origin in the world frame.
        # It's derived from: current_base_height + min_over_hips(-hip_x_in_base * sin(pitch) + hip_z_in_base * cos(pitch))
        # which simplifies to: current_base_height - abs(hip_x_in_base) * abs(sin(pitch)) + hip_z_in_base * cos(pitch)
        # (since hip_z_in_base is negative, and for typical pitch ranges cos(pitch) > 0)
        return current_base_height + (_HAA_Z_VAL * torch.cos(current_base_pitch) -
                                      _HAA_X_ABS_VAL * torch.abs(torch.sin(current_base_pitch)))

    min_hip_z_initial = _calculate_min_hip_world_z(base_height, base_pitch)
    is_touching_ground_initially = (min_hip_z_initial <= 0.0)

    if torch.any(is_touching_ground_initially):
        # Fix by setting pitch to 0 for these problematic environments
        base_pitch[is_touching_ground_initially] = base_pitch[is_touching_ground_initially] / 3.0 #dont zero it out completely, but reduce it, should likely be safe

    base_quat_w = math_utils.quat_from_euler_xyz(
        roll=torch.zeros_like(base_pitch),
        pitch=base_pitch,
        yaw=torch.zeros_like(base_pitch),
    )
    

    # URDF offsets (these are specific to a robot, kept as in original)
    haa_origins_in_base_frame = torch.tensor([
        [0.042, -0.05443, -0.006],  # RF_HAA
        [0.042,  0.05443, -0.006],  # LF_HAA
        [-0.042, -0.05443, -0.006], # RH_HAA
        [-0.042,  0.05443, -0.006], # LH_HAA
    ], device=device).unsqueeze(0).repeat(num_envs, 1, 1)

    hfe_origins_in_hip_link_frame = torch.tensor([
        [0.04538, -0.0194, 0.0],  # RF_HFE in RF_HIP
        [0.04538,  0.0194, 0.0],  # LF_HFE in LF_HIP
        [-0.04538, -0.0194, 0.0], # RH_HFE in RH_HIP
        [-0.04538,  0.0194, 0.0], # LH_HFE in LH_HIP
    ], device=device).unsqueeze(0).repeat(num_envs, 1, 1)

    base_pos_w = torch.zeros((num_envs, 3), device=device) #root_state[:, :3].unsqueeze(1)
    base_pos_w[:, 2] = base_height

    # Expand dims for broadcasting before use in rotations and additions
    _base_pos_w_expanded = base_pos_w.unsqueeze(1)  # Shape: (num_envs, 1, 3)
    _base_quat_w_expanded = base_quat_w.unsqueeze(1) # Shape: (num_envs, 1, 4)

    # Assuming HAA joint angle is 0 for this reset.
    # World position of HIP link's origin (where HAA connects base to HIP link)
    # q: (N,1,4), v: (N,4,3) -> out: (N,4,3)
    _rotated_haa_origins = math_utils.quat_rotate(_base_quat_w_expanded, haa_origins_in_base_frame)
    # (N,1,3) + (N,4,3) -> (N,4,3)
    haa_joint_origin_pos_w = _base_pos_w_expanded + _rotated_haa_origins
    
    # Check if any hip origin is too low
    hip_height_threshold = 0.03
    if torch.any(haa_joint_origin_pos_w[..., 2] < hip_height_threshold):
        print(f"Warning: At least one HAA joint origin is below {hip_height_threshold:.2f}m in world frame.")
        # Optionally, you can print more details, e.g., which envs or which hips are too low
        # for env_idx in range(num_envs):
        #     if torch.any(haa_joint_origin_pos_w[env_idx, :, 2] < hip_height_threshold):
        #         print(f"  Env {env_idx}: Hips below threshold: {haa_joint_origin_pos_w[env_idx, :, 2].tolist()}")
    
    # World orientation of HIP link (same as base if HAA is 0)
    # This _hip_link_quat_w_expanded is used for rotating hfe_origins_in_hip_link_frame
    _hip_link_quat_w_expanded = _base_quat_w_expanded # Shape (N, 1, 4)

    # World position of HFE joint axes
    # q: (N,1,4), v: (N,4,3) -> out: (N,4,3)
    _rotated_hfe_origins = math_utils.quat_rotate(_hip_link_quat_w_expanded, hfe_origins_in_hip_link_frame)
    # (N,4,3) + (N,4,3) -> (N,4,3)
    hfe_joint_axis_pos_w = haa_joint_origin_pos_w + _rotated_hfe_origins 

    # 3. Define Target Foot Positions
    # Sample foot x-offset for front and hind legs ensuring lateral symmetry (left/right)
    # Front legs: indices 0 (RF) & 1 (LF); Hind legs: indices 2 (RH) & 3 (LH)
    front_offset = math_utils.sample_uniform(
        front_foot_x_offset_range_cm[0] / 100.0,
        front_foot_x_offset_range_cm[1] / 100.0,
        (num_envs, 1),
        device=device
    )
    hind_offset = math_utils.sample_uniform(
        hind_foot_x_offset_range_cm[0] / 100.0,
        hind_foot_x_offset_range_cm[1] / 100.0,
        (num_envs, 1),
        device=device
    )
    # Combine offsets into full per-leg tensor [RF, LF, RH, LH]
    foot_x_offset_m = torch.cat(
        [front_offset, front_offset, hind_offset, hind_offset],
        dim=1
    )

    target_foot_pos_w = torch.zeros_like(hfe_joint_axis_pos_w)
    target_foot_pos_w[..., 0] = hfe_joint_axis_pos_w[..., 0] + foot_x_offset_m
    target_foot_pos_w[..., 1] = hfe_joint_axis_pos_w[..., 1] # Assuming foot Y aligns with HFE Y
    target_foot_pos_w[..., 2] = 0.0

    # 4. Inverse Kinematics for each leg (New Convention)
    foot_len = 0.02
    thigh_len = 0.10  # L1
    shank_len = 0.105 + foot_len # L2
    eps = 1e-6

    vec_w = target_foot_pos_w - hfe_joint_axis_pos_w # (num_envs, 4, 3)
    Dx_w = vec_w[..., 0]  # (num_envs, 4)
    Dz_w = vec_w[..., 2]  # (num_envs, 4)

    # Transform vec_hfe_to_foot from world to base frame
    # base_pitch is theta_b (num_envs,)
    theta_b = base_pitch.unsqueeze(1) # (num_envs, 1) for broadcasting
    cos_theta_b = torch.cos(theta_b)
    sin_theta_b = torch.sin(theta_b)

    Dx_b = Dx_w * cos_theta_b - Dz_w * sin_theta_b  # (num_envs, 4)
    Dz_b = Dx_w * sin_theta_b + Dz_w * cos_theta_b # (num_envs, 4)

    # Solve 2-link IK in base frame for (Dx_b, Dz_b)
    dist_sq_b = Dx_b**2 + Dz_b**2
    dist_b = torch.sqrt(dist_sq_b)

    # Clamp distance to be within reach
    if torch.any(dist_b > (thigh_len + shank_len) - eps):
        print("Warning: Distance too long for leg to reach target")
        
    dist_b = torch.clamp(dist_b, abs(thigh_len - shank_len) + eps, (thigh_len + shank_len) - eps)

    # KFE angle (alpha_kfe)
    # alpha_kfe = 0: shank aligned with thigh. Positive: shank moves forward CCW relative to thigh.
    cos_knee_angle_arg = (dist_b**2 - thigh_len**2 - shank_len**2) / (2 * thigh_len * shank_len)
    knee_angle = torch.acos(torch.clamp(cos_knee_angle_arg, -1.0 + eps, 1.0 - eps)) # (num_envs, 4)

    # HFE angle (alpha_hfe)
    # alpha_hfe = 0: thigh vertical down. Positive: thigh moves forward CCW.
    # phi_thigh_base is angle of thigh w.r.t. base's +X axis.
    # phi_thigh_base = atan2(Dz_b, Dx_b) - atan2(L2*sin(alpha_kfe), L1+L2*cos(alpha_kfe))
    foot_target_angle_in_base_frame = torch.atan2(Dz_b, Dx_b) # Angle of target vector in base frame
    
    gamma_b_num = shank_len * torch.sin(knee_angle)
    gamma_b_den = thigh_len + shank_len * torch.cos(knee_angle)
    # Add epsilon to denominator to avoid division by zero if L1+L2*cos(alpha_kfe) is zero
    # This can happen if L1=L2 and alpha_kfe=pi (target at origin, fully folded)
    gamma_b = torch.atan2(gamma_b_num, gamma_b_den + eps)
    
    thigh_base_angle = foot_target_angle_in_base_frame - gamma_b # (num_envs, 4)
    
    hip_angle = thigh_base_angle + (math.pi / 2.0)  # (num_envs, 4)
    
    front_hip_angle = hip_angle[:, 0]
    front_knee_angle = knee_angle[:, 0]
    hind_hip_angle = hip_angle[:, 2]
    hind_knee_angle = knee_angle[:, 2]
    
    return base_height, base_pitch, front_hip_angle.unsqueeze(-1), front_knee_angle.unsqueeze(-1), hind_hip_angle.unsqueeze(-1), hind_knee_angle.unsqueeze(-1)
    