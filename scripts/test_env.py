"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="This script demonstrates the Mars Jumper robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time # Added import
import numpy as np
import torch
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation  # must be after applaunch
from reset_crouch import sample_robot_crouch_pose
from isaaclab.utils.math import quat_from_euler_xyz


# --- CONFIG ---
DEG2RAD = np.pi / 180.0
robot_cfg = MarsJumperRobotCfg()

episode_duration_steps = 200

#For setting constant angles
use_constant_angles = True
constant_abductor_angle = 0
constant_hip_angle = 45 * DEG2RAD
constant_knee_angle = 45 * DEG2RAD
constant_base_pitch=np.pi/4
initial_angle_targets = {
    "LF_HAA": constant_abductor_angle, "LH_HAA": constant_abductor_angle, "RF_HAA": constant_abductor_angle, "RH_HAA": constant_abductor_angle,
    "LF_HFE": constant_hip_angle, "LH_HFE": constant_hip_angle, "RF_HFE": constant_hip_angle, "RH_HFE": constant_hip_angle,
    "LF_KFE": constant_knee_angle, "LH_KFE": constant_knee_angle, "RF_KFE": constant_knee_angle, "RH_KFE": constant_knee_angle,
}

#For sampling and testing static initial pose
use_crouch_pose_sampling = False
base_height_range = [0.1, 0.1]
base_pitch_range_rad = [np.pi/4, np.pi/4]
foot_x_offset_range_cm = [-0.0, 0.0]

#For transitioning to a jump
do_takeoff = False
takeoff_knee_angle = 0
takeoff_hip_angle = 0
takeoff_angle_targets = {
    "LF_HAA": constant_abductor_angle, "LH_HAA": constant_abductor_angle, "RF_HAA": constant_abductor_angle, "RH_HAA": constant_abductor_angle,
    "LF_HFE": constant_hip_angle, "LH_HFE": constant_hip_angle, "RF_HFE": constant_hip_angle, "RH_HFE": constant_hip_angle,
    "LF_KFE": constant_knee_angle, "LH_KFE": constant_knee_angle, "RF_KFE": constant_knee_angle, "RH_KFE": constant_knee_angle,
}

static_pose_display_duration_s = 5.0 # Duration to display each static pose

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

# --- Plotting Function ---
def plot_joint_data(time_log, target_logs, actual_logs, torque_logs, root_height_log, root_lin_acc_mag_log, joint_names, filename_prefix="joint"):
    num_joints = len(joint_names)
    num_plots = num_joints + 2
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3), sharex=True)
    if num_plots == 1:
        axs = [axs]
    fig.suptitle("Joint Angle Tracking, Applied Torque, Root Height, and Base Acceleration (First Cycle: 0-399 Steps)")
    for i, name in enumerate(joint_names):
        color_angle_target = 'tab:blue'
        color_angle_actual = 'tab:cyan'
        axs[i].set_ylabel("Angle (rad)", color=color_angle_target)
        axs[i].plot(time_log, target_logs[name], label=f'{name} Target Angle', linestyle='--', color=color_angle_target)
        axs[i].plot(time_log, actual_logs[name], label=f'{name} Actual Angle', color=color_angle_actual)
        axs[i].tick_params(axis='y', labelcolor=color_angle_target)
        axs[i].grid(True, axis='y', linestyle=':', alpha=0.6)
        ax_torque = axs[i].twinx()
        color_torque = 'tab:red'
        ax_torque.set_ylabel("Applied Torque (Nm)", color=color_torque)
        ax_torque.plot(time_log, torque_logs[name], label=f'{name} Applied Torque', color=color_torque, linestyle='-.')
        ax_torque.tick_params(axis='y', labelcolor=color_torque)
        lines, labels = axs[i].get_legend_handles_labels()
        lines2, labels2 = ax_torque.get_legend_handles_labels()
        ax_torque.legend(lines + lines2, labels + labels2, loc='upper right')
        axs[i].set_title(f"Joint: {name}")
        axs[i].grid(True, axis='x')
    root_height_ax = axs[num_joints]
    color_height = 'tab:green'
    root_height_ax.set_ylabel("Root Height (m)", color=color_height)
    root_height_ax.plot(time_log, root_height_log, label='Root Height', color=color_height)
    root_height_ax.tick_params(axis='y', labelcolor=color_height)
    root_height_ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    root_height_ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    root_height_ax.legend(loc='upper right')
    root_height_ax.set_title("Root Height")
    base_acc_ax = axs[num_joints + 1]
    color_acc = 'tab:purple'
    base_acc_ax.set_ylabel("Base Lin. Acc. Mag. (m/s^2)", color=color_acc)
    base_acc_ax.plot(time_log, root_lin_acc_mag_log, label='Base Lin. Acc. Mag.', color=color_acc)
    base_acc_ax.tick_params(axis='y', labelcolor=color_acc)
    base_acc_ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    base_acc_ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    base_acc_ax.set_ylim(0, 20)
    base_acc_ax.legend(loc='upper right')
    base_acc_ax.set_title("Base Linear Acceleration Magnitude")
    axs[-1].set_xlabel("Time (s)")
    combined_filename = f"scripts/plots/{filename_prefix}_position_tracking_height_and_accel.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(combined_filename)
    print(f"[INFO]: Combined plot saved to {combined_filename}")
    plt.close(fig)

# --- Simulation Loop ---
def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    time_log = []
    root_height_log = []
    root_lin_acc_mag_log = []
    joint_names_to_log = ["LF_HAA", "LF_HFE", "LF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]
    target_angle_logs = {name: [] for name in joint_names_to_log}
    actual_angle_logs = {name: [] for name in joint_names_to_log}
    applied_torque_logs = {name: [] for name in joint_names_to_log}
    plot_generated = False
    robot = entities["mars_jumper_robot"]
    
    try:
        joint_indices_to_log = {name: robot.find_joints(name)[0][0] for name in joint_names_to_log}
    except IndexError:
        print(f"[ERROR] Could not find one or more joints: {joint_names_to_log}")
        print(f"Available joints: {robot.joint_names}")
        return
    
    # --- Static Pose Validation Logic ---
    if use_crouch_pose_sampling:
        # Perform initial setup for crouch pose
        print("[INFO]: Entering static pose resampling mode.")
        last_sample_real_time = -static_pose_display_duration_s # Ensures first sample happens immediately
        
        while simulation_app.is_running():
            current_real_time = time.perf_counter()
            if current_real_time - last_sample_real_time >= static_pose_display_duration_s:
                last_sample_real_time = current_real_time
                print(f"\n[INFO] Resampling static pose at real time: {current_real_time:.2f}s")

                root_state_static = robot.data.default_root_state.clone()
                joint_pos_static = robot.data.default_joint_pos.clone()
                joint_vel_static = robot.data.default_joint_vel.clone()

                hip_joint_idx_static = robot.find_joints(robot.cfg.HIP_FLEXION_JOINTS_REGEX)[0]
                knee_joint_idx_static = robot.find_joints(robot.cfg.KNEE_JOINTS_REGEX)[0]

                base_height_s, base_pitch_s, sampled_hip_angles_s, sampled_knee_angles_s = sample_robot_crouch_pose(
                    base_height_range=base_height_range,
                    base_pitch_range_rad=base_pitch_range_rad,
                    foot_x_offset_range_cm=foot_x_offset_range_cm,
                    device=sim.device,
                    num_envs=1
                )
                base_quat_w_s = quat_from_euler_xyz(torch.zeros_like(base_pitch_s), base_pitch_s, torch.zeros_like(base_pitch_s))
                root_state_static[:, 2] = base_height_s
                root_state_static[:, 3:7] = base_quat_w_s
                root_state_static[:, 7:] = 0.0  # Zero velocity for static pose

                joint_pos_static[:, hip_joint_idx_static] = sampled_hip_angles_s
                joint_pos_static[:, knee_joint_idx_static] = sampled_knee_angles_s
                
                abductor_cfg_angle = robot_cfg.abductor_angle 
                for j_name in ["LF_HAA", "LH_HAA", "RF_HAA", "RH_HAA"]:
                    indices, _ = robot.find_joints(j_name)
                    if indices and len(indices) > 0: 
                        joint_pos_static[0, indices[0]] = abductor_cfg_angle
                    else:
                        print(f"[WARN] Static pose: Abductor joint {j_name} not found or no indices returned.")

                joint_vel_static[:] = 0.0 

                robot.write_root_state_to_sim(root_state_static)
                robot.write_joint_state_to_sim(joint_pos_static, joint_vel_static)
                print(f"[INFO]: Static crouch pose. Robot initial state written to sim.")

                static_pose_targets = {}
                all_joint_names_static = robot.joint_names
                for i, name in enumerate(all_joint_names_static):
                    target_val = joint_pos_static[0, i].item()
                    static_pose_targets[name] = target_val
                    indices, _ = robot.find_joints(name)
                    if indices and len(indices) > 0:
                        robot.set_joint_position_target(target_val, joint_ids=indices)
                
                robot.write_data_to_sim() 
                sim.step()               
                robot.update(sim_dt)     

                print(f"[INFO]: Static pose validation. Robot base height: {robot.data.root_pos_w[0, 2].item():.4f} m")
                try:
                    temp_joint_indices_to_log = {name: robot.find_joints(name)[0][0] for name in joint_names_to_log}
                    for name in joint_names_to_log:
                        target_val_log = static_pose_targets.get(name, float('nan')) # Use target_val from this scope
                        actual_val_log = robot.data.joint_pos[0, temp_joint_indices_to_log[name]].item()
                        print(f"  Joint {name}: Target: {target_val_log:.4f}, Actual: {actual_val_log:.4f} rad")
                except Exception as e:
                    print(f"  Warning: Could not print detailed joint info for static pose: {e}")
            
            simulation_app.update() 
            time.sleep(0.01) # Yield CPU

        return # Exit run_simulator for static pose case
    # --- End of Static Pose Validation Logic ---

    while simulation_app.is_running():
        if step_count % episode_duration_steps == 0:
            if step_count == episode_duration_steps and not plot_generated:
                plot_joint_data(time_log, target_angle_logs, actual_angle_logs, applied_torque_logs, root_height_log, root_lin_acc_mag_log, joint_names_to_log)
                plot_generated = True
                time_log = []
                root_height_log = []
                root_lin_acc_mag_log = []
                target_angle_logs = {name: [] for name in joint_names_to_log}
                actual_angle_logs = {name: [] for name in joint_names_to_log}
                applied_torque_logs = {name: [] for name in joint_names_to_log}
            sim_time = 0.0
 
            root_state = robot.data.default_root_state.clone()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            
            hip_joint_idx = robot.find_joints(robot.cfg.HIP_FLEXION_JOINTS_REGEX)[0]
            knee_joint_idx = robot.find_joints(robot.cfg.KNEE_JOINTS_REGEX)[0]

            if use_crouch_pose_sampling:
                sampled_base_height, sampled_base_pitch, sampled_hip_angles, sampled_knee_angles = sample_robot_crouch_pose(base_height_range=base_height_range, 
                                                                                            base_pitch_range_rad=base_pitch_range_rad, 
                                                                                            foot_x_offset_range_cm=foot_x_offset_range_cm, 
                                                                                            device=sim.device, 
                                                                                            num_envs=1)
                
                base_quat_w = quat_from_euler_xyz(torch.zeros_like(sampled_base_pitch), sampled_base_pitch, torch.zeros_like(sampled_base_pitch))
                root_state[:, 2] = sampled_base_height
                root_state[:, 3:7] = base_quat_w
                root_state[:, 7:] = 0.0
                
                joint_pos[:, hip_joint_idx] = sampled_hip_angles
                joint_pos[:, knee_joint_idx] = sampled_knee_angles
                                
            elif use_constant_angles:
                joint_pos[:, hip_joint_idx] = constant_hip_angle
                joint_pos[:, knee_joint_idx] = constant_knee_angle
                pitch_as_tensor = torch.tensor(constant_base_pitch, device=sim.device)
                base_quat_w = quat_from_euler_xyz(torch.zeros_like(pitch_as_tensor), pitch_as_tensor, torch.zeros_like(pitch_as_tensor))
                root_state[:, 3:7] = base_quat_w
            else:
                raise ValueError("Invalid mode selected. Please set use_crouch_pose_sampling or use_constant_angles to True.")
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            print(f"[INFO]: Resetting robot state at step {step_count}...")
            
        in_episode_step_count = step_count % episode_duration_steps
        
        # This 'joint_angle_targets' will be used for target setting and logging in the dynamic case.
        # The problematic 'if use_crouch_pose_sampling:' block that zeroed targets
        # is now correctly handled as this part of the code is only reached if use_crouch_pose_sampling is False.
        current_step_targets = {}
        if do_takeoff and in_episode_step_count > episode_duration_steps / 2:
            current_step_targets = takeoff_angle_targets
        else:
            current_step_targets = initial_angle_targets # Uses the global initial_angle_targets
            
        # for joint_name, target_pos in current_step_targets.items():
        #     joint_indices, _ = robot.find_joints(joint_name)
        #     if joint_indices: # Ensure joint exists
        #         robot.set_joint_position_target(target_pos, joint_ids=joint_indices)
                    
        # Logging
        if step_count < episode_duration_steps:
            time_log.append(sim_time)
            current_joint_pos = robot.data.joint_pos
            current_applied_torque = robot.data.applied_torque
            current_root_pos_w = robot.data.root_pos_w
            root_height_log.append(current_root_pos_w[0, 2].item())
            base_lin_acc_w = robot.data.body_lin_acc_w[0, 0, :]
            base_lin_acc_mag = torch.norm(base_lin_acc_w).item()
            root_lin_acc_mag_log.append(base_lin_acc_mag)
            
            for name in joint_names_to_log:
                target_angle_logs[name].append(current_step_targets[name]) # Use current_step_targets for logging
                joint_idx = joint_indices_to_log[name]
                actual_angle = current_joint_pos[0, joint_idx].item()
                actual_angle_logs[name].append(actual_angle)
                applied_torque = current_applied_torque[0, joint_idx].item()
                applied_torque_logs[name].append(applied_torque)
                
        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        step_count += 1
        robot.update(sim_dt)

# --- Main Function ---
def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/200, gravity=(0.0, 0.0, 0.0)))
    sim.set_camera_view(eye=[0, 1, 0.1], target=[0.0, 0.0, 0.0])
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()
