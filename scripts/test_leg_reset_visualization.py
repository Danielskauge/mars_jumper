import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Sequence

# Attempt to import isaaclab math utilities, needed for HFE position reconstruction
try:
    import isaaclab.utils.math as math_utils
except ImportError:
    print("WARNING: isaaclab.utils.math could not be imported. HFE position reconstruction might fail.")
    print("Please ensure isaaclab is installed and in PYTHONPATH.")
    # As a fallback, define minimal quaternion functions if isaaclab is not available.
    # This is a simplified version and might not cover all edge cases like Isaac Lab's.
    def quat_from_euler_xyz(roll, pitch, yaw):
        cr = torch.cos(roll * 0.5)
        sr = torch.sin(roll * 0.5)
        cp = torch.cos(pitch * 0.5)
        sp = torch.sin(pitch * 0.5)
        cy = torch.cos(yaw * 0.5)
        sy = torch.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return torch.stack([qw, qx, qy, qz], dim=-1)

    def quat_rotate(quat, vec):
        # quat: (..., 4) w, x, y, z
        # vec: (..., 3) x, y, z
        # Simplified for batch of quats and batch of vecs if shapes match for broadcasting
        q_w = quat[..., 0]
        q_xyz = quat[..., 1:]
        
        t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
        rotated_vec = vec + q_w.unsqueeze(-1) * t + torch.cross(q_xyz, t, dim=-1)
        return rotated_vec

    class MathUtilsMock:
        def quat_from_euler_xyz(self, roll, pitch, yaw):
            return quat_from_euler_xyz(roll, pitch, yaw)
        def quat_rotate(self, quat, vec):
            return quat_rotate(quat, vec)
        def sample_uniform(self, low, high, shape, device):
            return torch.rand(shape, device=device) * (high - low) + low

    math_utils = MathUtilsMock()


# Assume mars_jumper is in PYTHONPATH or script is run from workspace root
from reset_crouch import sample_robot_crouch_pose
# --- Mock Isaac Lab structures and data (minimal for the function) ---
class MockRobotData:
    def __init__(self, device: str):
        # Batch size of 1 for simplicity in test
        self.default_root_state = torch.tensor([[
            0.0, 0.0, 0.5,  # pos x, y, z
            1.0, 0.0, 0.0, 0.0,  # quat w, x, y, z (identity)
            0.0, 0.0, 0.0,  # lin_vel x, y, z
            0.0, 0.0, 0.0   # ang_vel x, y, z
        ]], device=device)

class MockRobot:
    def __init__(self, device: str):
        self.data = MockRobotData(device=device)

class MockEnv:
    def __init__(self, device_str: str = "cpu"):
        self.device = torch.device(device_str)
        self.robot = MockRobot(device=self.device)
        # The function has a fallback if scene.env_origins is not present
        # For a more complete mock, you could add:
        # self.scene = type('MockScene', (), {'env_origins': torch.zeros((1,3), device=self.device)})


# --- Constants for visualization and testing ---
THIGH_LENGTH = 0.10  # Match value in reset_robot_with_feet_on_ground
SHANK_LENGTH = 0.105 # Match value in reset_robot_with_feet_on_ground
BASE_VIZ_LENGTH = 0.15 # Arbitrary length for visualizing the base segment

# URDF offsets copied from reset_crouch.py for HFE position reconstruction
# These are specific to the robot model used in reset_crouch.py
_HAA_ORIGINS_IN_BASE_FRAME_ROBOT = torch.tensor([
    [0.042, -0.05443, -0.006],  # RF_HAA
    [0.042,  0.05443, -0.006],  # LF_HAA
    [-0.042, -0.05443, -0.006], # RH_HAA
    [-0.042,  0.05443, -0.006], # LH_HAA
])

_HFE_ORIGINS_IN_HIP_LINK_FRAME_ROBOT = torch.tensor([
    [0.04538, -0.0194, 0.0],  # RF_HFE in RF_HIP
    [0.04538,  0.0194, 0.0],  # LF_HFE in LF_HIP
    [-0.04538, -0.0194, 0.0], # RH_HFE in RH_HIP
    [-0.04538,  0.0194, 0.0], # LH_HFE in LH_HIP
])

def visualize_leg_2d(
    base_pitch_rad: float,
    hfe_angle_rad: float,   # User convention: 0 = vertical down, +ve CCW
    kfe_angle_rad: float,   # User convention: 0 = vertical down, +ve CCW
    hfe_world_pos: np.ndarray, # Shape (2,) for (x, z)
    target_foot_world_pos: np.ndarray, # Shape (2,) for (x, z)
    thigh_len: float,
    shank_len: float,
    base_viz_len: float
):
    """
    Visualizes a single leg in 2D (XZ plane) based on provided angles and positions.
    Angles are interpreted as: world_link_angle = base_pitch + link_angle_param - pi/2
    """
    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # 1. Base
    # hfe_world_pos is the front of our visualized base segment
    base_rear_x = hfe_world_pos[0] - base_viz_len * math.cos(base_pitch_rad)
    base_rear_z = hfe_world_pos[1] - base_viz_len * math.sin(base_pitch_rad)
    ax.plot([base_rear_x, hfe_world_pos[0]], [base_rear_z, hfe_world_pos[1]], 'k-', lw=3, label='Base')
    ax.plot(hfe_world_pos[0], hfe_world_pos[1], 'ko', markersize=8, label='HFE Joint (World)')

    # 2. Thigh
    # Angle interpretation: world_thigh_angle = base_pitch + hfe_angle_rad - pi/2
    world_thigh_angle_rad = base_pitch_rad + hfe_angle_rad - (math.pi / 2.0)
    knee_world_x = hfe_world_pos[0] + thigh_len * math.cos(world_thigh_angle_rad)
    knee_world_z = hfe_world_pos[1] + thigh_len * math.sin(world_thigh_angle_rad)
    ax.plot([hfe_world_pos[0], knee_world_x], [hfe_world_pos[1], knee_world_z], 'b-', lw=2, label='Thigh')
    ax.plot(knee_world_x, knee_world_z, 'bo', markersize=6, label='Knee Joint')

    # 3. Shank
    # The kfe_angle_rad is an "absolute-style" angle (0 = shank vertical down, +ve CCW)
    # This is consistent with its docstring and how it's used here.
    # world_shank_angle = world_thigh_angle + (kfe_angle_rad - hfe_angle_rad) <- if kfe_angle_rad was absolute and hfe_angle_rad too, their diff is relative knee bend
    # The line below implies kfe_angle_rad is the angle of the shank in the base's coordinate system, adjusted by base_pitch.
    world_shank_angle_rad = base_pitch_rad + kfe_angle_rad - (math.pi / 2.0)
    
    foot_calc_x = knee_world_x + shank_len * math.cos(world_shank_angle_rad)
    foot_calc_z = knee_world_z + shank_len * math.sin(world_shank_angle_rad)
    ax.plot([knee_world_x, foot_calc_x], [knee_world_z, foot_calc_z], 'g-', lw=2, label='Shank')
    ax.plot(foot_calc_x, foot_calc_z, 'gx', markersize=10, mew=2, label='Calculated Foot Pos')

    # 4. Target Foot Position
    ax.plot(target_foot_world_pos[0], target_foot_world_pos[1], 'r*', markersize=12, label='Target Foot Pos (World)')

    # Plot settings
    all_x = [base_rear_x, hfe_world_pos[0], knee_world_x, foot_calc_x, target_foot_world_pos[0]]
    all_z = [base_rear_z, hfe_world_pos[1], knee_world_z, foot_calc_z, target_foot_world_pos[1]]
    ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.1)
    ax.set_ylim(min(all_z) - 0.1, max(all_z) + 0.1)
    
    ax.set_xlabel("World X")
    ax.set_ylabel("World Z")
    ax.set_title(f"Leg Visualization\nBase Pitch: {math.degrees(base_pitch_rad):.1f}\N{DEGREE SIGN}"
                 f", HFE: {math.degrees(hfe_angle_rad):.1f}\N{DEGREE SIGN}"
                 f", KFE: {math.degrees(kfe_angle_rad):.1f}\N{DEGREE SIGN}")
    ax.axhline(0, color='grey', linestyle='--', lw=0.8) # Ground line
    ax.axvline(hfe_world_pos[0], color='grey', linestyle=':', lw=0.5) # Vertical line at HFE
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.savefig("leg_visualization.png")

def main():
    # --- Test Parameters ---
    env_ids_to_reset = [0] # Test for one environment
    num_envs = len(env_ids_to_reset)
    device_str = "cpu" # Using CPU for this test
    test_device = torch.device(device_str)
    
    # Define ranges for the reset function
    # Keep base height positive, e.g., around 0.2-0.3m
    test_base_height_range: Tuple[float, float] = (0.1, 0.1)
    # Test with a specific pitch, e.g., 15 degrees CCW
    test_base_pitch_range_rad: Tuple[float, float] = (math.radians(-15.0), math.radians(15.0))
    # Foot X offset relative to hip's ground projection (in cm)
    # e.g., foot 5cm in front of hip's projection
    test_foot_x_offset_range_cm: Tuple[float, float] = (-5.0, 5.0) 


    print("Calling sample_robot_crouch_pose...")
    # Updated function call to match new signature from reset_crouch.py
    # Returns: base_height, base_pitch, hip_angle (HFE), knee_angle (KFE, relative)
    base_height_t, sampled_pitch_t, alpha_hfe_t, alpha_kfe_t = \
        sample_robot_crouch_pose(
            base_height_range=test_base_height_range,
            base_pitch_range_rad=test_base_pitch_range_rad,
            foot_x_offset_range_cm=test_foot_x_offset_range_cm,
            device=device_str, # Pass device string as per updated reset_crouch
            num_envs=num_envs
        )

    print("Function call successful.")
    print(f"  Returned base_height tensor shape: {base_height_t.shape}")
    print(f"  Returned base_pitch tensor shape: {sampled_pitch_t.shape}")
    print(f"  Returned alpha_hfe tensor shape: {alpha_hfe_t.shape}")
    print(f"  Returned alpha_kfe tensor shape: {alpha_kfe_t.shape}")

    # Select data for the first (and only) environment and the first leg (e.g., RF)
    env_idx = 0
    leg_idx = 0 # 0:RF, 1:LF, 2:RH, 3:LH (based on typical URDF parsing order)

    base_pitch_val = sampled_pitch_t[env_idx].item()
    # alpha_hfe_t is the absolute HFE angle (0=thigh vertical down, +CCW)
    hfe_angle_val = alpha_hfe_t[env_idx, leg_idx].item()
    # alpha_kfe_t is the relative KFE angle (0=shank aligned with thigh, +CCW relative to thigh)
    kfe_relative_val = alpha_kfe_t[env_idx, leg_idx].item()

    # Convert relative KFE angle to the "absolute-style" KFE angle expected by visualize_leg_2d
    # visualize_leg_2d expects kfe_angle_rad where 0 = shank vertical down.
    # kfe_absolute = hfe_absolute + kfe_relative
    kfe_absolute_val_for_viz = hfe_angle_val + kfe_relative_val
    
    # --- Reconstruct HFE world position ---
    # This logic is adapted from reset_crouch.py because hfe_pos_w is no longer returned.
    # 1. Prepare base state
    current_base_height = base_height_t[env_idx]
    current_base_pitch = sampled_pitch_t[env_idx]

    base_pos_w = torch.zeros((1, 3), device=test_device) # For a single env
    base_pos_w[0, 2] = current_base_height

    base_quat_w = math_utils.quat_from_euler_xyz(
        roll=torch.zeros_like(current_base_pitch),
        pitch=current_base_pitch,
        yaw=torch.zeros_like(current_base_pitch)
    ).unsqueeze(0) # Shape (1,4) for one env

    # URDF offsets for the specific robot, ensure tensors are on the correct device
    haa_origins_bf = _HAA_ORIGINS_IN_BASE_FRAME_ROBOT.to(test_device).unsqueeze(0) # (1, num_legs, 3)
    hfe_origins_hlf = _HFE_ORIGINS_IN_HIP_LINK_FRAME_ROBOT.to(test_device).unsqueeze(0) # (1, num_legs, 3)
    
    _base_pos_w_expanded = base_pos_w.unsqueeze(1)  # Shape: (1, 1, 3)
    _base_quat_w_expanded = base_quat_w # Shape: (1, 4)
                                         # math_utils.quat_rotate handles (N,4) and (N,M,3) -> (N,M,3)
                                         # or (N,1,4) and (N,M,3) -> (N,M,3)

    # Assuming HAA joint angle is 0 for this reset.
    # Reshape quaternion to (1,1,4) and vectors to (1,4,3) for quat_rotate
    # q: (1,1,4), v: (1,4,3) -> out: (1,4,3)
    _rotated_haa_origins = math_utils.quat_rotate(_base_quat_w_expanded.unsqueeze(1), haa_origins_bf)
    haa_joint_origin_pos_w = _base_pos_w_expanded + _rotated_haa_origins
    
    _hip_link_quat_w_expanded = _base_quat_w_expanded # Shape (1, 4), same as base_quat if HAA is 0

    # q: (1,1,4), v: (1,4,3) -> out: (1,4,3)
    _rotated_hfe_origins = math_utils.quat_rotate(_hip_link_quat_w_expanded.unsqueeze(1), hfe_origins_hlf)
    hfe_joint_axis_pos_w_calc = haa_joint_origin_pos_w + _rotated_hfe_origins
    
    hfe_pos_val = hfe_joint_axis_pos_w_calc[env_idx, leg_idx, [0, 2]].cpu().numpy() # Select X and Z

    # --- Reconstruct Target Foot world position ---
    # WARNING: This target is based on a new random sample for foot_x_offset_m,
    # which will likely differ from the one used inside sample_robot_crouch_pose
    # to calculate the joint angles. Thus, the visualized target (red star) may not
    # perfectly match the calculated foot position (green 'x'), even if IK is correct.
    foot_x_offset_m_sampled_in_test = math_utils.sample_uniform(
        test_foot_x_offset_range_cm[0] / 100.0,
        test_foot_x_offset_range_cm[1] / 100.0,
        (num_envs, 4), # num_envs, num_legs
        device=test_device
    )
    
    # Use the X and Y from the reconstructed HFE position for this leg
    # (original reset_crouch.py used hfe_joint_axis_pos_w for X and Y reference)
    # target_foot_pos_w[..., 0] = hfe_joint_axis_pos_w[..., 0] + foot_x_offset_m
    # target_foot_pos_w[..., 1] = hfe_joint_axis_pos_w[..., 1] # Assuming foot Y aligns with HFE Y
    # target_foot_pos_w[..., 2] = 0.0
    
    # For visualization (X,Z) plane:
    foot_target_x_calc = hfe_joint_axis_pos_w_calc[env_idx, leg_idx, 0] + foot_x_offset_m_sampled_in_test[env_idx, leg_idx]
    # Foot Z is on the ground
    foot_target_z_calc = 0.0
    foot_target_val = np.array([foot_target_x_calc.item(), foot_target_z_calc])


    print(f"\nVisualizing for leg {leg_idx} of env {env_idx}:")
    print(f"  Base Pitch (deg): {math.degrees(base_pitch_val):.2f}")
    print(f"  HFE Angle (deg, absolute, 0=down): {math.degrees(hfe_angle_val):.2f}")
    print(f"  KFE Angle (deg, relative to thigh, 0=straight): {math.degrees(kfe_relative_val):.2f}")
    print(f"  KFE Angle for Viz (deg, absolute, 0=shank down): {math.degrees(kfe_absolute_val_for_viz):.2f}")
    print(f"  Reconstructed HFE World Pos (X,Z): ({hfe_pos_val[0]:.3f}, {hfe_pos_val[1]:.3f})")
    print(f"  Reconstructed Target Foot World Pos (X,Z): ({foot_target_val[0]:.3f}, {foot_target_val[1]:.3f})")
    print("  NOTE: Target Foot World Pos is based on an independent random sample in this script.")

    visualize_leg_2d(
        base_pitch_rad=base_pitch_val,
        hfe_angle_rad=hfe_angle_val, # Pass the absolute HFE angle
        kfe_angle_rad=kfe_absolute_val_for_viz, # Pass the converted absolute KFE angle
        hfe_world_pos=hfe_pos_val,
        target_foot_world_pos=foot_target_val,
        thigh_len=THIGH_LENGTH,
        shank_len=SHANK_LENGTH,
        base_viz_len=BASE_VIZ_LENGTH
    )

if __name__ == "__main__":
    main() 