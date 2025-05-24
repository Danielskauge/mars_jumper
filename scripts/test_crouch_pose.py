"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="This script demonstrates Mars Jumper crouch pose sampling.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import numpy as np
import torch
from isaaclab.app import AppLauncher # Duplicate import removed
import isaaclab.sim as sim_utils
from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation # must be after applaunch
from reset_crouch import sample_robot_crouch_pose
from isaaclab.utils.math import quat_from_euler_xyz

# --- CONFIG ---
DEG2RAD = np.pi / 180.0
robot_cfg = MarsJumperRobotCfg()

gravity_on = False
simulate_physics = False
use_crouch_pose_sampling = True # Explicitly True for this script
base_height_range = [0.08, 0.08]
base_pitch_range_rad = [-0*DEG2RAD, 0*DEG2RAD]
front_foot_x_offset_range_cm = [-4, -4] # Added for separate front/hind offsets
hind_foot_x_offset_range_cm = [-4, -4]  # Added for separate front/hind offsets

static_pose_display_duration_s = 2.0 # Duration to display each static pose

# --- Scene Design ---
def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origin = [0.0, 0.0, 0.0]
    robot_cfg.prim_path = "/World/robot"
    mars_jumper_robot = Articulation(robot_cfg)
    return {"mars_jumper_robot": mars_jumper_robot}, [origin]

# --- Simulation Loop ---
def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    robot = entities["mars_jumper_robot"]
    
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    
    print(f"[INFO]: simulate_physics is set to: {simulate_physics}")
    last_sample_real_time = -static_pose_display_duration_s
    
    # Variables to store the latest sampled pose
    sampled_root_state = None
    sampled_joint_pos = None
    sampled_joint_vel = None # Will be initialized to zeros when first pose is sampled

    # Flag for initial direct joint state write when simulate_physics is True
    initial_physics_pose_written = False

    while simulation_app.is_running():
        current_real_time = time.perf_counter()
        
        new_pose_sampled_this_iteration = False # Flag to track if a new pose was sampled
        if current_real_time - last_sample_real_time >= static_pose_display_duration_s:
            last_sample_real_time = current_real_time
            new_pose_sampled_this_iteration = True
            print(f"\\n[INFO] Sampling new pose at real time: {current_real_time:.2f}s")

            # Sample crouch pose
            base_height_s, base_pitch_s, front_hip_angle_s, front_knee_angle_s, hind_hip_angle_s, hind_knee_angle_s = sample_robot_crouch_pose(
                base_height_range=base_height_range,
                base_pitch_range_rad=base_pitch_range_rad,
                front_foot_x_offset_range_cm=front_foot_x_offset_range_cm,
                hind_foot_x_offset_range_cm=hind_foot_x_offset_range_cm,
                device=sim.device,
                num_envs=1
            )
            base_quat_w_s = quat_from_euler_xyz(
                torch.zeros_like(base_pitch_s), base_pitch_s, torch.zeros_like(base_pitch_s)
            )

            # Initialize or update sampled pose variables
            if sampled_root_state is None: # First time allocation
                sampled_root_state = robot.data.default_root_state.clone()
                sampled_joint_pos = robot.data.default_joint_pos.clone()
                sampled_joint_vel = torch.zeros_like(robot.data.default_joint_vel) # Velocities are zero

            sampled_root_state[:, 2] = base_height_s
            sampled_root_state[:, 3:7] = base_quat_w_s 
            sampled_root_state[:, 7:] = 0.0 # Zero root linear and angular velocities

            # Apply sampled joint angles
            RF_HFE_idx = robot.find_joints("RF_HFE")[0]
            LF_HFE_idx = robot.find_joints("LF_HFE")[0]
            RH_HFE_idx = robot.find_joints("RH_HFE")[0]
            LH_HFE_idx = robot.find_joints("LH_HFE")[0]
            RF_KFE_idx = robot.find_joints("RF_KFE")[0]
            LF_KFE_idx = robot.find_joints("LF_KFE")[0]
            RH_KFE_idx = robot.find_joints("RH_KFE")[0]
            LH_KFE_idx = robot.find_joints("LH_KFE")[0]
            sampled_joint_pos[:, RF_HFE_idx] = front_hip_angle_s
            sampled_joint_pos[:, LF_HFE_idx] = front_hip_angle_s
            sampled_joint_pos[:, RH_HFE_idx] = hind_hip_angle_s
            sampled_joint_pos[:, LH_HFE_idx] = hind_hip_angle_s
            sampled_joint_pos[:, RF_KFE_idx] = front_knee_angle_s
            sampled_joint_pos[:, LF_KFE_idx] = front_knee_angle_s
            sampled_joint_pos[:, RH_KFE_idx] = hind_knee_angle_s
            sampled_joint_pos[:, LH_KFE_idx] = hind_knee_angle_s
            
            # All components of sampled_joint_vel remain zero

            print(f"[INFO]: New pose computed: base height {base_height_s.item():.4f} m")
            # DEBUG: Print sampled pitch and quaternion
            print(f"DEBUG: Sampled base_pitch_s: {base_pitch_s.item():.6f} rad")
            print(f"DEBUG: Calculated base_quat_w_s (w,x,y,z): [{base_quat_w_s[0,0].item():.6f}, {base_quat_w_s[0,1].item():.6f}, {base_quat_w_s[0,2].item():.6f}, {base_quat_w_s[0,3].item():.6f}]")
            # End DEBUG

            if simulate_physics:
                # Directly write the new root state.
                robot.write_root_state_to_sim(sampled_root_state)
                
                # Set the new joint positions and zero velocities as targets for the actuators.
                robot.set_joint_position_target(sampled_joint_pos)
                robot.set_joint_velocity_target(sampled_joint_vel) # Target zero velocity

                # If it's the very first pose with physics, also write joint state directly
                # to avoid the robot starting from default and then moving to the PD target.
                if not initial_physics_pose_written:
                    robot.write_joint_state_to_sim(sampled_joint_pos, sampled_joint_vel)
                    initial_physics_pose_written = True
        
        # --- Per-iteration update ---
        if simulate_physics:
            if sampled_root_state is not None: # Ensure targets have been set at least once
                robot.write_data_to_sim() # Apply actuator commands

            sim.step()
            sim_time += sim_dt
            step_count += 1
            robot.update(sim_dt)
        else: # Not simulating physics (original static display mode)
            if sampled_root_state is not None: # If a pose has been sampled
                # Continuously write the last sampled kinematic state to "freeze" it
                robot.write_root_state_to_sim(sampled_root_state)
                robot.write_joint_state_to_sim(sampled_joint_pos, sampled_joint_vel)
            
            time.sleep(0.01) # Original behavior for non-physics simulation to control update rate

        simulation_app.update()
       
    print("[INFO]: Pose sampling loop ended.")

# --- Main Function ---
def main():
    # Use a very small dt as we are mostly interested in static poses, but sim needs to step.
    # Gravity can be on or off, depending on whether you want to see the robot settle under gravity
    # or hold its exact sampled pose. For pure kinematic sampling, (0,0,0) is fine.
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/200, gravity=(0.0, 0.0, -9.81 if gravity_on else 0.0))) # Or (0,0,0)
    sim.set_camera_view(eye=[0, 1, 0.1], target=[0.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset() # Resets the simulation environment
    print("[INFO]: Setup complete for crouch pose testing...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close() 