"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="This script demonstrates the Mars Jumper robot with constant angles and takeoff.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import matplotlib.pyplot as plt
import isaaclab.sim as sim_utils
from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation # must be after applaunch
from isaaclab.utils.math import quat_from_euler_xyz

# --- CONFIG ---
DEG2RAD = np.pi / 180.0
robot_cfg = MarsJumperRobotCfg()

episode_duration_steps = 150 # You might want to adjust this

gravity_on = True

root_height = 0.2

#For setting constant angles
use_constant_angles = True # Explicitly True for this script
abductor_angle = 0.0 # Use robot_cfg.abductor_angle if defined and preferred
hip_angle = 0 * DEG2RAD
knee_angle = 90 * DEG2RAD
base_pitch = 0 * DEG2RAD
initial_angle_targets = {
    "LF_HAA": abductor_angle, "LH_HAA": abductor_angle, "RF_HAA": abductor_angle, "RH_HAA": abductor_angle,
    "LF_HFE": hip_angle, "LH_HFE": hip_angle, "RF_HFE": hip_angle, "RH_HFE": hip_angle,
    "LF_KFE": knee_angle, "LH_KFE": knee_angle, "RF_KFE": knee_angle, "RH_KFE": knee_angle,
}

#For transitioning to a jump
do_takeoff = True # Set to True to test takeoff
takeoff_knee_angle = 0 * DEG2RAD # Example: more extended knee
takeoff_hip_angle = 0 * DEG2RAD   # Example: more extended hip
takeoff_angle_targets = {
    "LF_HAA": abductor_angle, "LH_HAA": abductor_angle, "RF_HAA": abductor_angle, "RH_HAA": abductor_angle,
    "LF_HFE": takeoff_hip_angle, "LH_HFE": takeoff_hip_angle, "RF_HFE": takeoff_hip_angle, "RH_HFE": takeoff_hip_angle,
    "LF_KFE": takeoff_knee_angle, "LH_KFE": takeoff_knee_angle, "RF_KFE": takeoff_knee_angle, "RH_KFE": takeoff_knee_angle,
}

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
def plot_joint_data(time_log, target_logs, actual_logs, torque_logs, root_height_log, root_lin_acc_mag_log, joint_names, filename_prefix="joint_dynamics"):
    num_joints = len(joint_names)
    num_plots = num_joints + 2 # For root height and base acceleration
    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3), sharex=True)
    if num_plots == 1: # Should not happen with root height/accel plots
        axs = [axs]
    fig.suptitle(f"Joint Angle Tracking, Torque, Root Height, Base Accel (Episode: {episode_duration_steps} steps)")
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

    # Root Height Plot
    root_height_ax = axs[num_joints]
    color_height = 'tab:green'
    root_height_ax.set_ylabel("Root Height (m)", color=color_height)
    root_height_ax.plot(time_log, root_height_log, label='Root Height', color=color_height)
    root_height_ax.tick_params(axis='y', labelcolor=color_height)
    root_height_ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    root_height_ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    root_height_ax.legend(loc='upper right')
    root_height_ax.set_title("Root Height")

    # Base Linear Acceleration Magnitude Plot
    base_acc_ax = axs[num_joints + 1]
    color_acc = 'tab:purple'
    base_acc_ax.set_ylabel("Base Lin. Acc. Mag. (m/s^2)", color=color_acc)
    base_acc_ax.plot(time_log, root_lin_acc_mag_log, label='Base Lin. Acc. Mag.', color=color_acc)
    base_acc_ax.tick_params(axis='y', labelcolor=color_acc)
    base_acc_ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    base_acc_ax.yaxis.set_major_locator(plt.MaxNLocator(20))
    base_acc_ax.set_ylim(0, 20) # Optional: set y-limit for acceleration
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
    # Define which joints to log and plot
    joint_names_to_log = ["LF_HAA", "LF_HFE", "LF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]
    target_angle_logs = {name: [] for name in joint_names_to_log}
    actual_angle_logs = {name: [] for name in joint_names_to_log}
    applied_torque_logs = {name: [] for name in joint_names_to_log}
    plot_generated = False

    robot = entities["mars_jumper_robot"]
    
    try:
        joint_indices_to_log = {name: robot.find_joints(name)[0][0] for name in joint_names_to_log}
    except IndexError:
        print(f"[ERROR] Could not find one or more joints for logging: {joint_names_to_log}")
        print(f"Available joints: {robot.joint_names}")
        return

    # Main simulation loop
    while simulation_app.is_running():
        # Reset and plot logic
        if step_count % episode_duration_steps == 0:
            if step_count == episode_duration_steps and not plot_generated: # Generate plot only after the first full episode
                plot_joint_data(time_log, target_angle_logs, actual_angle_logs, applied_torque_logs, root_height_log, root_lin_acc_mag_log, joint_names_to_log)
                plot_generated = True # Ensure plot is generated only once per run or reset data for new plots
                # Option to clear logs for next episode if you want separate plots per episode:
                # time_log = []
                # root_height_log = []
                # root_lin_acc_mag_log = []
                # target_angle_logs = {name: [] for name in joint_names_to_log}
                # actual_angle_logs = {name: [] for name in joint_names_to_log}
                # applied_torque_logs = {name: [] for name in joint_names_to_log}
            
            # Reset simulation time for logging this episode (if logging per episode)
            # sim_time = 0.0 # Reset if logging starts from 0 for each episode plot

            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            
            # Renaming to align with full_jump_env.py's convention:
            # full_jump_env.py: self.hip_joint_idx is HAA (abduction via ".*HAA.*")
            #                   self.abduction_joint_idx is HFE (flexion via ".*HFE.*")
            #                   self.knee_joint_idx is KFE (via ".*KFE.*")
            # We assume robot.cfg.HIP_ABDUCTION_JOINTS_REGEX corresponds to HAA joints,
            # robot.cfg.HIP_FLEXION_JOINTS_REGEX to HFE joints, and robot.cfg.KNEE_JOINTS_REGEX to KFE joints.
            hip_joint_idx_env = robot.find_joints(".*HAA.*")[0] # For HAA joints
            abduction_joint_idx_env = robot.find_joints(".*HFE.*")[0] # For HFE joints
            knee_joint_idx_env = robot.find_joints(".*KFE.*")[0] # For KFE joints

            # Set joint positions according to the script's angle variables and env-style index names:
            # abductor_angle is for HAA joints (indexed by hip_joint_idx_env)
            # hip_angle is for HFE joints (indexed by abduction_joint_idx_env)
            # knee_angle is for KFE joints (indexed by knee_joint_idx_env)
            joint_pos[:, hip_joint_idx_env] = abductor_angle
            joint_pos[:, abduction_joint_idx_env] = hip_angle
            joint_pos[:, knee_joint_idx_env] = knee_angle
            
            pitch_as_tensor = torch.tensor(base_pitch, device=sim.device)
            base_quat_w = quat_from_euler_xyz(torch.zeros_like(pitch_as_tensor), pitch_as_tensor, torch.zeros_like(pitch_as_tensor))
            root_state[:, 2] = root_height # Set a default spawn height, adjust as needed
            root_state[:, 3:7] = base_quat_w
            root_state[:, 7:] = 0.0 # Zero velocity

            robot.write_root_state_to_sim(root_state)
            robot.write_joint_position_to_sim(joint_pos)
            robot.write_joint_velocity_to_sim(joint_vel)
            robot.update(sim_dt) # Update once after setting initial state
            print(f"[INFO]: Resetting robot state at step {step_count}...")
            # After reset, if it's the very first step of the simulation, allow one plot generation cycle
            if step_count == 0:
                 plot_generated = False # Allow plotting for the first episode collected
            
        in_episode_step_count = step_count % episode_duration_steps
        
        current_step_targets = {}
        if do_takeoff and (in_episode_step_count > episode_duration_steps / 2):
            current_step_targets = takeoff_angle_targets
        else:
            current_step_targets = initial_angle_targets
            
        for joint_name, target_pos in current_step_targets.items():
            joint_indices, _ = robot.find_joints(joint_name)
            if joint_indices: # Ensure joint exists
                robot.set_joint_position_target(target_pos, joint_ids=joint_indices)
                robot.set_joint_velocity_target(0.0, joint_ids=joint_indices) # Add this for static poses
                    
        # Logging (only for the first episode or if logs are not cleared)
        if not plot_generated: # Log data only if the plot for the current episode hasn't been generated yet
            time_log.append(sim_time)
            current_joint_pos = robot.data.joint_pos
            current_applied_torque = robot.data.applied_torque
            current_root_pos_w = robot.data.root_pos_w
            root_height_log.append(current_root_pos_w[0, 2].item())
            
            # Ensure we are getting the base link's acceleration (index 0 of the first body)
            base_lin_acc_w = robot.data.body_lin_acc_w[0, 0, :] 
            base_lin_acc_mag = torch.norm(base_lin_acc_w).item()
            root_lin_acc_mag_log.append(base_lin_acc_mag)
            
            for name in joint_names_to_log:
                target_angle_logs[name].append(current_step_targets[name])
                joint_idx = joint_indices_to_log[name]
                actual_angle = current_joint_pos[0, joint_idx].item()
                actual_angle_logs[name].append(actual_angle)
                applied_torque = current_applied_torque[0, joint_idx].item()
                applied_torque_logs[name].append(applied_torque)
                
        robot.write_data_to_sim()
        sim.step()            # Perform simulation step
        sim_time += sim_dt    # Advance simulation time
        step_count += 1       # Increment step counter
        robot.update(sim_dt)  # Update robot state from sim

        simulation_app.update() # Update App/GUI

# --- Main Function ---
def main():
    # Configure simulation context (e.g., dt, gravity)
    # Gravity should be enabled for takeoff dynamics
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/360, gravity=(0.0, 0.0, -9.81 if gravity_on else 0.0)))
    sim.set_camera_view(eye=[0, 1.5, 0.5], target=[0.0, 0.0, 0.2]) # Adjusted camera for better view
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset() # Reset environment before starting
    print("[INFO]: Setup complete for takeoff dynamics testing...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close() 