"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the Mars Jumper robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils

from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation #must be after applaunch

import matplotlib.pyplot as plt

DEG2RAD = np.pi/180.0

# Removed unused angle definition variables: abductor_angle, crouch_knee_angle, crouch_hip_angle

def plot_root_position(time_log, x_pos_log, y_pos_log, z_pos_log, filename="root_position_tracking.png"):
    """Plots the root position (x, y, z) over time."""
    fig, ax = plt.subplots(figsize=(10, 6)) # Changed from ax1 to ax
    fig.suptitle("Robot Root Position Over Time (First Cycle)")

    # Plot root position on primary y-axis
    color_pos = 'tab:blue' # Using one color base for position axis
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)", color=color_pos)
    ax.plot(time_log, x_pos_log, label='Root X', color='tab:red', linestyle='-')
    ax.plot(time_log, y_pos_log, label='Root Y', color='tab:green', linestyle='--')
    ax.plot(time_log, z_pos_log, label='Root Z', color='tab:blue', linestyle=':')
    ax.tick_params(axis='y', labelcolor=color_pos)
    ax.grid(True)
    ax.legend(loc='upper left') # Combine legend

    # Removed secondary axis and related plotting/legend code

    full_filename = f"scripts/plots/{filename}"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
    plt.savefig(full_filename)
    print(f"[INFO]: Root position plot saved to {full_filename}")
    plt.close(fig)

# New function similar to the original plot_joint_data
def plot_lf_joint_data(time_log, target_logs, actual_logs, torque_logs, joint_names, filename="lf_joint_tracking.png"):
    """Plots target vs actual joint angles and applied torques for specified LF joints."""
    num_joints = len(joint_names)
    fig, axs = plt.subplots(num_joints, 1, figsize=(12, num_joints * 3), sharex=True)
    if num_joints == 1:
        axs = [axs]
    fig.suptitle("LF Joint Angle Tracking and Applied Torque (First Cycle)")

    for i, name in enumerate(joint_names):
        # Plot angles on the primary y-axis (left)
        color_angle_target = 'tab:blue'
        color_angle_actual = 'tab:cyan'
        axs[i].set_ylabel("Angle (rad)", color=color_angle_target)
        axs[i].plot(time_log, target_logs[name], label=f'{name} Target Angle', linestyle='--', color=color_angle_target)
        axs[i].plot(time_log, actual_logs[name], label=f'{name} Actual Angle', color=color_angle_actual)
        axs[i].tick_params(axis='y', labelcolor=color_angle_target)
        axs[i].grid(True, axis='y', linestyle=':', alpha=0.6) # Angle grid

        # Create a secondary y-axis (right) for torque
        ax_torque = axs[i].twinx()
        color_torque = 'tab:red'
        ax_torque.set_ylabel("Torque (Nm)", color=color_torque)
        # Ensure torque_logs[name] exists and has data
        if name in torque_logs and torque_logs[name]:
             ax_torque.plot(time_log, torque_logs[name], label=f'{name} Applied Torque', color=color_torque, linestyle='-.')
        ax_torque.tick_params(axis='y', labelcolor=color_torque)

        # Combine legends
        lines, labels = axs[i].get_legend_handles_labels()
        lines2, labels2 = ax_torque.get_legend_handles_labels()
        ax_torque.legend(lines + lines2, labels + labels2, loc='upper right')

        axs[i].set_title(f"Joint: {name}")
        axs[i].grid(True, axis='x') # Shared X-axis grid

    axs[-1].set_xlabel("Time (s)")
    full_filename = f"scripts/plots/{filename}"
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout
    plt.savefig(full_filename)
    print(f"[INFO]: LF Joint plot saved to {full_filename}")
    plt.close(fig)

def run_simulator(sim: sim_utils.SimulationContext):
    """Runs the simulation loop."""
    # -- Scene Setup --
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Define origin directly here
    origin = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device) # Make it a tensor on the correct device

    # Create Mars Jumper Robot
    robot_cfg = MarsJumperRobotCfg()
    robot_cfg.prim_path = "/World/robot"
    robot = Articulation(robot_cfg) # Assign directly to 'robot'

    # Find LF joint indices once
    lf_joint_names = ["LF_HAA", "LF_HFE", "LF_KFE"]
    try:
        # Store indices in a dictionary for easier access
        lf_joint_indices = {name: robot.find_joints(name)[0][0] for name in lf_joint_names}
    except IndexError:
        print(f"[ERROR] Could not find one or more LF joints: {lf_joint_names}")
        print(f"Available joints: {robot.joint_names}")
        return
    # -- End Scene Setup --

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # --- Logging Setup ---
    log_duration_steps = 200 # Keep the cycle duration
    time_log = []
    # Root position logs
    root_pos_x_log = []
    root_pos_y_log = []
    root_pos_z_log = []
    # LF Joint logs - using dictionaries
    lf_target_angle_logs = {name: [] for name in lf_joint_names}
    lf_actual_angle_logs = {name: [] for name in lf_joint_names}
    lf_applied_torque_logs = {name: [] for name in lf_joint_names}
    plot_generated = False
    # 'robot' is now defined above, no need for entities["mars_jumper_robot"]
    # --- End Logging Setup ---

    # Start simulation (Reset needs to happen after scene setup)
    sim.reset()
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        # --- Logging Logic (for first cycle: steps 0 to log_duration_steps - 1) ---
        if count < log_duration_steps:
            time_log.append(sim_time)
            # Log root position
            current_root_pos = robot.data.root_pos_w[0, :3].cpu().numpy()
            root_pos_x_log.append(current_root_pos[0])
            root_pos_y_log.append(current_root_pos[1])
            root_pos_z_log.append(current_root_pos[2])

            # Log LF joint data
            current_joint_pos = robot.data.joint_pos[0] # Get positions for the first robot
            current_applied_torque = robot.data.applied_torque[0] # Get torques for the first robot
            default_joint_pos = robot.data.default_joint_pos[0] # Get default positions

            for name in lf_joint_names:
                joint_idx = lf_joint_indices[name]
                # Log target angle (from default positions)
                target_angle = default_joint_pos[joint_idx].item()
                lf_target_angle_logs[name].append(target_angle)
                # Log actual angle
                actual_angle = current_joint_pos[joint_idx].item()
                lf_actual_angle_logs[name].append(actual_angle)
                # Log applied torque
                applied_torque = current_applied_torque[joint_idx].item()
                lf_applied_torque_logs[name].append(applied_torque)
        # --- End Logging Logic ---

        # Reset condition (triggers at steps 0, 400, 800, ...)
        if count % log_duration_steps == 0:
            # --- Plotting (triggers only once after the first cycle completes) ---
            if count == log_duration_steps and not plot_generated:
                 # Plot root position
                 plot_root_position(time_log, root_pos_x_log, root_pos_y_log, root_pos_z_log)
                 # Plot LF joint data
                 plot_lf_joint_data(time_log, lf_target_angle_logs, lf_actual_angle_logs, lf_applied_torque_logs, lf_joint_names)
                 plot_generated = True
            # --- End Plotting ---

            # Reset time and robot state for the next cycle
            sim_time = 0.0 # Reset sim_time for logging consistency if needed, though plot uses the log

            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            # Use the locally defined origin tensor
            root_state[:, :3] += origin[0] # Apply origin offset from the defined tensor
            robot.write_root_state_to_sim(root_state) # Use combined write function

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset() # Internal buffer reset
            print(f"[INFO]: Resetting robot state at step {count}...")

        # Set joint position targets to default positions (likely zero or near-zero)
        # Note: This sets targets for ALL joints, not just LF
        robot.set_joint_position_target(robot.data.default_joint_pos)
        
        # Write all commands to sim
        robot.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        # Update buffers
        robot.update(sim_dt)

def main():
    """Main function."""
    # Initialize simulation
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/200))
    sim.set_camera_view(eye=[0, 0.5, 0.2], target=[0.0, 0.0, 0.0])
    mars_g = 3.721
    earth_g = 9.81
    sim.gravity = (0.0, 0.0, -earth_g)
    sim.reset()
    # Scene setup is now inside run_simulator
    # scene_entities, scene_origins =  design_scene() # Removed
    # scene_origins = torch.tensor(scene_origins, device=sim.device) # Removed

    # Run simulation without passing entities or origins
    run_simulator(sim) # Modified call

if __name__ == "__main__":
    main()
    simulation_app.close()
