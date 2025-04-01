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

abductor_angle = 0 * DEG2RAD
hip_angle = -70 * DEG2RAD
knee_angle = 140 * DEG2RAD

takeoff_knee_angle = 0 * DEG2RAD
takeoff_hip_angle = 0 * DEG2RAD

initial_angle_targets = {
    "LF_HAA": abductor_angle, 
    "LH_HAA": abductor_angle, #postive angle is negative
    "RF_HAA": abductor_angle, #postive angle is negative
    "RH_HAA": abductor_angle,
    "LF_HFE": hip_angle,
    "LH_HFE": hip_angle,
    "RF_HFE": hip_angle, #postive angle is negative
    "RH_HFE": hip_angle,
    "LF_KFE": knee_angle,
    "LH_KFE": knee_angle,
    "RF_KFE": knee_angle,
    "RH_KFE": knee_angle,
}

takeoff_angle_targets = {
    "LF_HAA": abductor_angle, 
    "LH_HAA": abductor_angle, #postive angle is negative
    "RF_HAA": abductor_angle, #postive angle is negative
    "RH_HAA": abductor_angle,
    "LF_HFE": takeoff_hip_angle,
    "LH_HFE": takeoff_hip_angle,
    "RF_HFE": takeoff_hip_angle, #postive angle is negative
    "RH_HFE": takeoff_hip_angle,
    "LF_KFE": takeoff_knee_angle,
    "LH_KFE": takeoff_knee_angle,
    "RF_KFE": takeoff_knee_angle,
    "RH_KFE": takeoff_knee_angle,
}

def design_scene():
    """Designs the scene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origin = [0.0, 0.0, 0.0]
    
    # Create Mars Jumper Robot
    robot_cfg = MarsJumperRobotCfg()
    robot_cfg.prim_path = "/World/robot"
    mars_jumper_robot = Articulation(robot_cfg)
    #mars_jumper_robot.print_usd_info()

    return {"mars_jumper_robot": mars_jumper_robot}, [origin]

def plot_joint_data(time_log, target_logs, actual_logs, torque_logs, root_height_log, joint_names, filename_prefix="joint"):
    """Plots target vs actual joint angles, actual joint torques, and root height."""
    num_joints = len(joint_names)
    # Add 1 subplot for root height
    num_plots = num_joints + 1

    fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3), sharex=True)
    # Ensure axs is always a list, even with one joint + root height (2 plots)
    if num_plots == 1:
         axs = [axs]

    fig.suptitle("Joint Angle Tracking, Applied Torque, and Root Height (First Cycle: 0-399 Steps)")

    # Plot joint data
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
        ax_torque.plot(time_log, torque_logs[name], label=f'{name} Applied Torque', color=color_torque, linestyle='-.')
        ax_torque.tick_params(axis='y', labelcolor=color_torque)
        # ax_torque.grid(True, axis='y', linestyle='--', alpha=0.6) # Optional torque grid

        # Combine legends
        lines, labels = axs[i].get_legend_handles_labels()
        lines2, labels2 = ax_torque.get_legend_handles_labels()
        ax_torque.legend(lines + lines2, labels + labels2, loc='upper right')

        # Add title to subplot
        axs[i].set_title(f"Joint: {name}")
        axs[i].grid(True, axis='x') # Shared X-axis grid

    # Plot root height on the last subplot
    root_height_ax = axs[num_joints] # Index of the last subplot
    color_height = 'tab:green'
    root_height_ax.set_ylabel("Root Height (m)", color=color_height)
    root_height_ax.plot(time_log, root_height_log, label='Root Height', color=color_height)
    root_height_ax.tick_params(axis='y', labelcolor=color_height)
    root_height_ax.grid(True, axis='y', linestyle=':', alpha=0.6)
    root_height_ax.yaxis.set_major_locator(plt.MaxNLocator(20))  # Increase number of y-axis ticks
    root_height_ax.set_ylim(0, 0.2)  # Set y-axis limits
    root_height_ax.legend(loc='upper right')
    root_height_ax.set_title("Root Height")

    # Set common x-axis label on the very last subplot
    axs[-1].set_xlabel("Time (s)")
    combined_filename = f"scripts/plots/{filename_prefix}_position_tracking_and_height.png" # Updated filename
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.savefig(combined_filename)
    print(f"[INFO]: Combined plot saved to {combined_filename}")
    plt.close(fig) # Close the figure to free memory

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # --- Logging Setup ---
    log_duration_steps = 400
    time_log = []
    root_height_log = [] # Initialize root_height_log here
    # Log Left-Front (LF) and Right-Hind (RH) legs
    joint_names_to_log = ["LF_HAA", "LF_HFE", "LF_KFE", "RH_HAA", "RH_HFE", "RH_KFE"]
    target_angle_logs = {name: [] for name in joint_names_to_log}
    actual_angle_logs = {name: [] for name in joint_names_to_log}
    applied_torque_logs = {name: [] for name in joint_names_to_log} # New dict for torque logs
    plot_generated = False
    robot = entities["mars_jumper_robot"] # Get robot handle
    # Ensure default state is read before finding joints if needed, but Articulation likely handles this
    # Find joint indices once for efficiency
    try:
        joint_indices_to_log = {name: robot.find_joints(name)[0][0] for name in joint_names_to_log}
    except IndexError:
        print(f"[ERROR] Could not find one or more joints: {joint_names_to_log}")
        print(f"Available joints: {robot.joint_names}")
        return
    # --- End Logging Setup ---

    while simulation_app.is_running():
        # --- Logging Logic (for steps 0 to log_duration_steps - 1) ---
        # Reset condition (triggers at steps 0, 400, 800, ...)
        if count % log_duration_steps == 0:
            # --- Plotting (triggers only once when count hits log_duration_steps) ---
            # Check count > 0 to avoid plotting at step 0 if logs were somehow filled
            if count == log_duration_steps and not plot_generated:
                 plot_joint_data(time_log, target_angle_logs, actual_angle_logs, applied_torque_logs, root_height_log, joint_names_to_log)
                 plot_generated = True
                 # Reset logs after plotting the first cycle
                 time_log = []
                 root_height_log = []
                 target_angle_logs = {name: [] for name in joint_names_to_log}
                 actual_angle_logs = {name: [] for name in joint_names_to_log}
                 applied_torque_logs = {name: [] for name in joint_names_to_log}
            # --- End Plotting ---

            # Reset time and robot state for the next cycle
            sim_time = 0.0
            # count = 0 # Keep count incrementing across resets for simplicity unless cycle count matters

            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins[0] # Apply origin offset
            robot.write_root_state_to_sim(root_state) # Use combined write function

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset() # Internal buffer reset
            print(f"[INFO]: Resetting robot state at step {count}...")

        # Determine targets based on the count *within the current cycle*
        effective_count = count % log_duration_steps
        targets = initial_angle_targets
       
        if effective_count > int(log_duration_steps/2):
            targets = takeoff_angle_targets

        # Set joint position targets for all joints based on the current phase
        for joint_name, target_pos in targets.items():
            # Find joint index dynamically - less efficient but robust if indices change (unlikely here)
            # Using pre-calculated indices might be faster if performance critical
            joint_indices, _ = robot.find_joints(joint_name)
            if len(joint_indices) > 0:
                 robot.set_joint_position_target(target_pos, joint_ids=joint_indices)
            # else: print warning?
            
        if count < log_duration_steps:
            time_log.append(sim_time)
            
            # Log target and actual angles for selected joints
            current_joint_pos = robot.data.joint_pos # Get current positions once
            current_applied_torque = robot.data.applied_torque # Get current applied torques once
            current_root_pos_w = robot.data.root_pos_w # Get current root position once
            root_height_log.append(current_root_pos_w[0, 2].item()) # Log root height

            for name in joint_names_to_log:
                 target_angle_logs[name].append(targets[name])
                 joint_idx = joint_indices_to_log[name]
                 # Assuming only one robot instance (index 0)
                 actual_angle = current_joint_pos[0, joint_idx].item()
                 actual_angle_logs[name].append(actual_angle)
                 # Log applied torque
                 applied_torque = current_applied_torque[0, joint_idx].item()
                 applied_torque_logs[name].append(applied_torque)
        # --- End Logging Logic ---

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
    sim.set_camera_view(eye=[0, 1, 0.1], target=[0.0, 0.0, 0.0])
    sim.gravity = (0.0, 0.0, -3.721)
    # Setup scene
    scene_entities, scene_origins =  design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Start simulation
    sim.reset()
    print("[INFO]: Setup complete...") 
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()
