# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint for a single episode of an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import shutil

from networkx import complement
import numpy as np
import wandb
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games for a single episode.")
parser.add_argument("--cmd_magnitude", type=float, default=None, help="Fixed command magnitude.")
parser.add_argument("--cmd_pitch", type=float, default=None, help="Fixed command pitch.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log video and plots to WandB.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default="full-jump", help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the episode.")
parser.add_argument("--plot", action="store_true", default=False, help="Generate plots of episode data.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Force settings for single episode playback
args_cli.num_envs = 1
#args_cli.video = True # Always record video
args_cli.enable_cameras = True # Enable cameras for video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import math
import os
import time
import torch
import matplotlib.pyplot as plt
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.contact_sensor import ContactSensor # Import ContactSensor

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import envs
from terms.utils import get_center_of_mass_lin_vel, all_feet_off_the_ground, any_feet_on_the_ground
from terms.phase import Phase # Import Phase enum
from terms.observations import has_taken_off


def plot_episode_data(robot, clipped_actions, scaled_actions, joint_angles, joint_torques, com_lin_vel, base_height, jump_phase, feet_off_ground, dt, log_dir, cmd_filename_suffix, cmd_wandb_suffix, actual_cmd_magnitude, episode_any_feet_on_ground, episode_takeoff_toggle, episode_contact_forces, episode_rewards, episode_foot_ground_forces):
    """Plots the recorded actions, joint angles, torques, target positions, COM velocity, base height, feet status, contact forces, and rewards."""
    print("[INFO] Plotting episode data...")
    plots_dir = os.path.join(log_dir, "plots") # Define plots directory path
    os.makedirs(plots_dir, exist_ok=True) # Create plots directory if it doesn't exist

    # --- Convert data to numpy ---
    actions_np = torch.stack(clipped_actions).cpu().numpy()
    joint_angles_np = torch.stack(joint_angles).cpu().numpy()
    joint_torques_np = torch.stack(joint_torques).cpu().numpy()
    target_positions_np = torch.stack(scaled_actions).cpu().numpy()
    com_lin_vel_np = torch.stack(com_lin_vel).cpu().numpy()
    base_height_np = torch.stack(base_height).cpu().numpy()
    jump_phase_np = torch.stack(jump_phase).cpu().numpy().flatten() # Ensure flat
    feet_off_ground_np = torch.stack(feet_off_ground).cpu().numpy().astype(int) # Boolean status based on filtered ground contact
    any_feet_on_ground_np = torch.stack(episode_any_feet_on_ground).cpu().numpy().astype(int) # Boolean status based on filtered ground contact
    takeoff_toggle_np = torch.stack(episode_takeoff_toggle).cpu().numpy().astype(int)
    time_np = np.arange(len(actions_np)) * dt

    # Convert foot ground forces
    foot_ground_forces_np = {}
    if episode_foot_ground_forces:
        pass # Remove logic as episode_foot_ground_forces is removed
    # Convert general contact forces if available
    contact_forces_available = bool(episode_contact_forces)
    contact_forces_np = None
    if contact_forces_available:
        contact_forces_np = torch.stack(episode_contact_forces).cpu().numpy()
        if contact_forces_np.size == 0:
            contact_forces_available = False
            print("[Warning] General contact force data was collected but appears empty.")

    # Convert rewards if available
    rewards_available = bool(episode_rewards)
    rewards_np = None
    future_returns_np = None
    if rewards_available:
        rewards_np = torch.stack(episode_rewards).cpu().numpy()
        if rewards_np.size > 0:
            num_steps = len(rewards_np)
            future_returns_np = np.zeros(num_steps)
            cumulative_reward = 0.0
            for i in range(num_steps - 1, -1, -1):
                cumulative_reward += rewards_np[i]
                future_returns_np[i] = cumulative_reward
        else:
            rewards_available = False # Mark as unavailable if empty
            print("[Warning] Reward data was collected but appears empty.")


    # --- Helper Functions ---
    def add_all_phase_shading(ax, time_data, phase_data, add_legend_labels=True):
        phase_colors = {
            Phase.CROUCH: ('lightblue', 0.3),
            Phase.TAKEOFF: ('lightgreen', 0.3),
            Phase.FLIGHT: ('gold', 0.3),
            Phase.LANDING: ('lightcoral', 0.3)
        }
        phases = np.asarray(phase_data).flatten()
        labeled_phases = set()
        start_idx = 0
        legend_handles = [] # To store handles for the legend

        for i in range(1, len(phases)):
            if phases[i] != phases[start_idx]:
                phase_val = int(phases[start_idx])
                try:
                    phase_enum = Phase(phase_val)
                    color, alpha = phase_colors.get(phase_enum, ('gray', 0.1))
                    # Always create the span, label conditionally
                    span = ax.axvspan(time_data[start_idx], time_data[i], color=color, alpha=alpha)
                    if add_legend_labels and phase_enum not in labeled_phases:
                        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha)) # Create handle for legend
                        labeled_phases.add((phase_enum, plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha))) # Store enum and handle
                except ValueError: pass
                start_idx = i

        # Shade the last segment
        if start_idx < len(phases):
             phase_val = int(phases[start_idx])
             try:
                 phase_enum = Phase(phase_val)
                 color, alpha = phase_colors.get(phase_enum, ('gray', 0.1))
                 span = ax.axvspan(time_data[start_idx], time_data[-1], color=color, alpha=alpha)
                 if add_legend_labels and phase_enum not in labeled_phases:
                     legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha))
                     labeled_phases.add((phase_enum, plt.Rectangle((0, 0), 1, 1, fc=color, alpha=alpha)))
             except ValueError: pass

        # Add legend outside loop if labels were added
        if add_legend_labels and labeled_phases:
             # Sort phases by enum value for consistent legend order
             sorted_phases = sorted(list(labeled_phases), key=lambda item: item[0].value)
             handles = [item[1] for item in sorted_phases]
             labels = [item[0].name for item in sorted_phases]
             ax.legend(handles, labels, title="Phase", loc='upper right', fontsize='small')


    joint_groups = {
        'Abductors': ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA'],
        'Hip': ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE'],
        'Knee': ['LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
    }

    # Dictionary to store plot paths for WandB
    wandb_plots = {}

    # --- Plotting Sections ---

    # 1. Joint Control (Target vs Actual Angle)
    print("[INFO] Plotting Joint Control (Target vs Actual)...")
    fig_ctrl, axs_ctrl = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_ctrl.suptitle('Joint Control: Target Position vs Actual Angle', fontsize=16)
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs_ctrl[i]
        joint_names = joint_groups[group_name]
        lines = [] # Store lines for legend
        labels = [] # Store labels for legend
        for name in joint_names:
            try:
                # Use find_joints correctly - it returns (indices, names)
                indices_list, resolved_names = robot.find_joints(name)
                if not indices_list: # Check if list is empty
                   print(f"[Warning] Joint '{name}' not found. Skipping.")
                   continue
                idx = indices_list[0] # Take the first index found
                resolved_name = resolved_names[0] # Get the corresponding resolved name

                target_pos_deg = target_positions_np[:, idx] * 180.0 / np.pi
                actual_angle_deg = joint_angles_np[:, idx] * 180.0 / np.pi
                line1, = ax.plot(time_np, target_pos_deg, linestyle='--')
                line2, = ax.plot(time_np, actual_angle_deg)
                # Alternate colors for target/actual pairs for better distinction if needed
                # Use resolved_name for clarity in legend
                lines.extend([line1, line2])
                labels.extend([f'{resolved_name} Target', f'{resolved_name} Actual'])

            except (IndexError, ValueError) as e: # Handle case where joint might not be found or index out of bounds
                print(f"[Warning] Could not find or plot joint '{name}'. Error: {e}")
                continue
        ax.set_title(f"{group_name}")
        ax.set_ylabel("Angle (deg)")
        # Create legend with 2 columns if many joints
        ncol = 2 if len(labels) > 4 else 1
        ax.legend(lines, labels, fontsize='small', ncol=ncol)
        add_all_phase_shading(ax, time_np, jump_phase_np, add_legend_labels=(i==0)) # Only add phase legend to first subplot
        ax.grid(True)
    axs_ctrl[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plot_path = os.path.join(plots_dir, f"joint_control_plot_{cmd_filename_suffix}.png")
    plt.savefig(plot_path)
    plt.close(fig_ctrl)
    wandb_plots[f"joint_control_plot_{cmd_wandb_suffix}"] = plot_path

    # 2. Base Kinematics (Height, COM Vel Components, COM Vel Mag)
    print("[INFO] Plotting Base Kinematics...")
    fig_kin, axs_kin = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_kin.suptitle('Base Kinematics', fontsize=16)
    # Height
    axs_kin[0].plot(time_np, base_height_np, label="Base Height")
    axs_kin[0].set_title("Base Height")
    axs_kin[0].set_ylabel("Height (m)")
    axs_kin[0].legend()
    axs_kin[0].grid(True)
    add_all_phase_shading(axs_kin[0], time_np, jump_phase_np, add_legend_labels=True) # Add phase legend here
    # COM Vel Components
    vel_components = ['X', 'Y', 'Z']
    for i in range(3):
        axs_kin[1].plot(time_np, com_lin_vel_np[:, i], label=f"COM Vel {vel_components[i]}")
    axs_kin[1].set_title("COM Linear Velocity Components")
    axs_kin[1].set_ylabel("Velocity (m/s)")
    axs_kin[1].legend()
    axs_kin[1].grid(True)
    add_all_phase_shading(axs_kin[1], time_np, jump_phase_np, add_legend_labels=False)
    # COM Vel Magnitude
    com_lin_vel_mag = np.linalg.norm(com_lin_vel_np, axis=1)
    axs_kin[2].plot(time_np, com_lin_vel_mag, label="COM Velocity Magnitude")
    axs_kin[2].axhline(y=actual_cmd_magnitude, color='r', linestyle='--', label=f"Command Mag ({actual_cmd_magnitude:.2f})")
    axs_kin[2].set_title("COM Linear Velocity Magnitude")
    axs_kin[2].set_xlabel("Time (s)")
    axs_kin[2].set_ylabel("Velocity (m/s)")
    axs_kin[2].legend()
    axs_kin[2].grid(True)
    add_all_phase_shading(axs_kin[2], time_np, jump_phase_np, add_legend_labels=False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(plots_dir, f"base_kinematics_plot_{cmd_filename_suffix}.png")
    plt.savefig(plot_path)
    plt.close(fig_kin)
    wandb_plots[f"base_kinematics_plot_{cmd_wandb_suffix}"] = plot_path

    # 3. Contact Status & Filtered Forces
    print("[INFO] Plotting Contact Status & Forces...")
    fig_contact, axs_contact = plt.subplots(2, 1, figsize=(15, 10), sharex=True) # Changed from 3,1 to 2,1
    fig_contact.suptitle('Contact Status & Forces', fontsize=16)
    # Status Signals
    axs_contact[0].step(time_np, any_feet_on_ground_np, where='post', label='any_feet_on_the_ground() (General Sensor)', linewidth=1.5) # Updated label
    axs_contact[0].step(time_np, feet_off_ground_np, where='post', label='all_feet_off_the_ground() (General Sensor)', linestyle=':', linewidth=1.5) # Updated label
    axs_contact[0].step(time_np, takeoff_toggle_np, where='post', label='has_taken_off()', linestyle='--', linewidth=1.5) # From terms.observations
    axs_contact[0].set_title("Contact & Takeoff Status")
    axs_contact[0].set_ylabel("Status (0 or 1)")
    axs_contact[0].set_yticks([0, 1])
    axs_contact[0].set_ylim([-0.1, 1.1])
    axs_contact[0].legend(loc='center right', fontsize='small')
    axs_contact[0].grid(True)
    add_all_phase_shading(axs_contact[0], time_np, jump_phase_np, add_legend_labels=True) # Give this plot the phase legend
    # Plotting General Contact Forces (excluding feet) in the second subplot
    ax_gen_cf = axs_contact[1] # Use the second axis now
    if contact_forces_available:
        print("[INFO] Plotting General Contact Forces (Non-Foot)...")
        # Define body groups excluding feet explicitly
        # Trying to match indices from the general sensor based on patterns
        body_groups_non_foot = {
            'Shanks': ["LF_SHANK", "RF_SHANK", "LH_SHANK", "RH_SHANK"],
            'Thighs': ["LF_THIGH", "RF_THIGH", "LH_THIGH", "RH_THIGH"],
            'Base/Hip': ["base", "LF_HIP", "RF_HIP", "LH_HIP", "RH_HIP"] # Group base and hip
        }
        num_bodies = contact_forces_np.shape[1]

        for i, (group_name, body_names_patterns) in enumerate(body_groups_non_foot.items()):
            ax = ax_gen_cf
            any_plotted = False
            lines = []
            labels = []
            plotted_indices = set() # Keep track of plotted indices to avoid duplicates if patterns overlap

            for name_pattern in body_names_patterns:
                try:
                    # Get indices/names matching the pattern from the *robot asset*
                    indices_list, resolved_names = robot.find_bodies(name_pattern)
                    if not indices_list:
                        # print(f"[Debug] No bodies found for pattern '{name_pattern}' in group '{group_name}'.")
                        continue

                    # Now, we need to map these robot body indices to the indices used
                    # by the *general contact sensor*. This requires knowing how the
                    # general sensor was configured (which bodies it includes).
                    # Assuming the general sensor includes *all* robot bodies,
                    # the indices might align, but this is fragile.
                    # A safer approach (requires env changes) is to have the sensor
                    # provide a mapping or use SceneEntityCfg within the plot function.
                    # For now, assume indices align with the full robot body list:
                    for idx, resolved_name in zip(indices_list, resolved_names):
                         # Check if this index is within the bounds of the collected force data
                         # AND if we haven't plotted this specific index already
                        if idx < num_bodies and idx not in plotted_indices:
                            force_magnitude = np.linalg.norm(contact_forces_np[:, idx, :], axis=1)
                            line, = ax.plot(time_np, force_magnitude)
                            lines.append(line)
                            labels.append(resolved_name)
                            any_plotted = True
                            plotted_indices.add(idx) # Mark this index as plotted
                        # else: print(f"[Debug] Index {idx} for {resolved_name} out of bounds ({num_bodies}) or already plotted.")

                except Exception as e:
                    print(f"[Warning] Error processing body pattern '{name_pattern}' in group '{group_name}': {e}")

            # Set titles and labels on the shared axis ax_gen_cf
            ax_gen_cf.set_title(f"General Contact Forces (Non-Foot)")
            ax_gen_cf.set_ylabel("Force Mag (N)")
            if any_plotted:
                ax_gen_cf.legend(lines, labels, fontsize='small', ncol=2 if len(labels) > 4 else 1)
            else:
                ax_gen_cf.text(0.5, 0.5, 'No data found/plotted', ha='center', va='center', transform=ax_gen_cf.transAxes)
            ax_gen_cf.grid(True)
            add_all_phase_shading(ax_gen_cf, time_np, jump_phase_np, add_legend_labels=False) # No phase legend here

    # 4. Joint Torques
    print("[INFO] Plotting Joint Torques...")
    fig_torq, axs_torq = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_torq.suptitle('Joint Torques', fontsize=16)
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs_torq[i]
        joint_names = joint_groups[group_name]
        lines = []
        labels = []
        for name in joint_names:
            try:
                indices_list, resolved_names = robot.find_joints(name)
                if not indices_list: continue
                idx = indices_list[0]
                resolved_name = resolved_names[0]

                line, = ax.plot(time_np, joint_torques_np[:, idx])
                lines.append(line)
                labels.append(resolved_name)
            except (IndexError, ValueError) as e:
                print(f"[Warning] Could not find or plot torque for joint '{name}'. Error: {e}")
                continue
        ax.set_title(f"{group_name}")
        ax.set_ylabel("Torque (Nm)")
        ax.legend(lines, labels, fontsize='small', ncol=2 if len(labels) > 4 else 1)
        add_all_phase_shading(ax, time_np, jump_phase_np, add_legend_labels=(i==0)) # Phase legend on first
        ax.grid(True)
    axs_torq[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(plots_dir, f"joint_torques_plot_{cmd_filename_suffix}.png")
    plt.savefig(plot_path)
    plt.close(fig_torq)
    wandb_plots[f"joint_torques_plot_{cmd_wandb_suffix}"] = plot_path

    # 5. Rewards & Returns
    if rewards_available:
        print("[INFO] Plotting Rewards & Returns...")
        fig_rew, axs_rew = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        fig_rew.suptitle('Rewards and Returns', fontsize=16)
        # Instantaneous Rewards
        axs_rew[0].plot(time_np, rewards_np, label="Instantaneous Reward", linewidth=1.5)
        axs_rew[0].set_title("Instantaneous Reward")
        axs_rew[0].set_ylabel("Reward")
        axs_rew[0].legend()
        axs_rew[0].grid(True)
        add_all_phase_shading(axs_rew[0], time_np, jump_phase_np, add_legend_labels=True) # Phase legend here
        # Returns
        if future_returns_np is not None:
            axs_rew[1].plot(time_np, future_returns_np, label="Summed Future Rewards (Return)", linewidth=1.5)
            axs_rew[1].set_title("Return (Summed Future Rewards)")
            axs_rew[1].set_ylabel("Return")
            axs_rew[1].legend()
            axs_rew[1].grid(True)
            add_all_phase_shading(axs_rew[1], time_np, jump_phase_np, add_legend_labels=False)
        else:
             axs_rew[1].text(0.5, 0.5, 'No return data available', ha='center', va='center', transform=axs_rew[1].transAxes)
             axs_rew[1].set_title("Return (Summed Future Rewards)")
             axs_rew[1].grid(True)

        axs_rew[1].set_xlabel("Time (s)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(plots_dir, f"rewards_plot_{cmd_filename_suffix}.png")
        plt.savefig(plot_path)
        plt.close(fig_rew)
        wandb_plots[f"rewards_plot_{cmd_wandb_suffix}"] = plot_path
    else:
        print("[INFO] Skipping Rewards plot (no data).")

    # 6. Clipped Actions
    print("[INFO] Plotting Clipped Actions...")
    fig_act, axs_act = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    fig_act.suptitle('Clipped Actions (Policy Output)', fontsize=16)
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs_act[i]
        joint_names = joint_groups[group_name]
        lines = []
        labels = []
        for name in joint_names:
            try:
                indices_list, resolved_names = robot.find_joints(name)
                if not indices_list: continue
                idx = indices_list[0]
                resolved_name = resolved_names[0]

                line, = ax.plot(time_np, actions_np[:, idx])
                lines.append(line)
                labels.append(resolved_name)
            except (IndexError, ValueError) as e:
                print(f"[Warning] Could not find or plot action for joint '{name}'. Error: {e}")
                continue
        ax.set_title(f"{group_name}")
        ax.set_ylabel("Action Value")
        ax.legend(lines, labels, fontsize='small', ncol=2 if len(labels) > 4 else 1)
        add_all_phase_shading(ax, time_np, jump_phase_np, add_legend_labels=(i==0)) # Phase legend on first
        ax.grid(True)
    axs_act[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(plots_dir, f"clipped_actions_plot_{cmd_filename_suffix}.png")
    plt.savefig(plot_path)
    plt.close(fig_act)
    wandb_plots[f"clipped_actions_plot_{cmd_wandb_suffix}"] = plot_path

    # --- Log all plots to WandB ---
    if args_cli.wandb:
        print("[INFO] Logging plots to WandB...")
        wandb_log_data = {key: wandb.Image(path) for key, path in wandb_plots.items() if path and os.path.exists(path)}
        if wandb_log_data:
            wandb.log(wandb_log_data)
            print(f"[INFO] Logged {len(wandb_log_data)} plots to WandB.")
        else:
            print("[Warning] No valid plot paths found to log to WandB.")

def main():
    """Play with RL-Games agent for a single episode."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    
    if args_cli.cmd_magnitude is not None and args_cli.cmd_pitch is not None:
        # Override the command ranges to be sampled from with fixed values
        env_cfg.command_ranges.initial_magnitude_range = (args_cli.cmd_magnitude, args_cli.cmd_magnitude)
        env_cfg.command_ranges.initial_pitch_range = (args_cli.cmd_pitch, args_cli.cmd_pitch)
    
    # Determine the actual command values used (either from CLI or defaults)
    actual_cmd_magnitude = env_cfg.command_ranges.initial_magnitude_range[0]
    actual_cmd_pitch = env_cfg.command_ranges.initial_pitch_range[0]
    # Simplified format for filenames/logging
    cmd_filename_suffix = "episode" # Generic suffix
    cmd_wandb_suffix = "" # No command info in wandb key

    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # --- Clear existing plots directory ---
    plots_dir = os.path.join(log_dir, "plots")
    if os.path.exists(plots_dir):
        print(f"[INFO] Removing existing plots directory: {plots_dir}")
        shutil.rmtree(plots_dir)
    # We don't need to recreate it here, as plot_episode_data will do it.
    # os.makedirs(plots_dir, exist_ok=True) 

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment (render_mode is forced to rgb_array)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording if enabled
    if args_cli.video:
        video_folder = os.path.join(log_root_path, log_dir, "videos", "play_single")
        # Delete existing video folder to ensure a new one is created
        if os.path.exists(video_folder):
            shutil.rmtree(video_folder)
        # Ensure the base directory exists
        os.makedirs(video_folder, exist_ok=True)

        video_kwargs = {
            "video_folder": video_folder,
            "name_prefix": f"{cmd_filename_suffix}_", # Use simplified suffix
            "step_trigger": lambda step: step == 0,
            "video_length": env.unwrapped.max_episode_length, # Record the whole episode
            "disable_logger": True,
        }
        print("[INFO] Recording single episode video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config (always 1)
    agent_cfg["params"]["config"]["num_actors"] = 1
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    
    # Use lists to store data for the single episode
    episode_actions = []
    episode_joint_angles = []
    episode_joint_torques = []
    episode_target_positions = []
    episode_com_lin_vel = []
    episode_base_height = []
    episode_jump_phase = []
    episode_feet_off_ground = []
    episode_takeoff_toggle = []
    episode_any_feet_on_ground = []
    episode_contact_forces = [] # Initialize list for contact forces
    episode_rewards = [] # Initialize list for rewards
    dt = env.unwrapped.physics_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
        

    # Wrap the simulation loop in a try...finally block
    try:
        # simulate environment for a single episode
        while simulation_app.is_running():
            start_time = time.time()
            with torch.inference_mode():
                obs = agent.obs_to_torch(obs)
                actions = agent.get_action(obs, is_deterministic=True)
                
                episode_actions.append(actions.clone().squeeze(0))
                episode_joint_angles.append(env.unwrapped.robot.data.joint_pos.clone().squeeze(0))
                episode_joint_torques.append(env.unwrapped.robot.data.applied_torque.clone().squeeze(0))
                # Retrieve target joint positions (these are the scaled actions for ImplicitActuator)
                target_positions = env.unwrapped.robot.data.joint_pos_target.clone().squeeze(0)
                episode_target_positions.append(target_positions)
                current_com_vel = get_center_of_mass_lin_vel(env.unwrapped).squeeze(0) # Squeeze batch dim
                current_base_height = env.unwrapped.robot.data.root_pos_w[:, 2].squeeze(0) # Squeeze batch dim
                episode_com_lin_vel.append(current_com_vel.clone())
                episode_base_height.append(current_base_height.clone())
                current_jump_phase = env.unwrapped.jump_phase.clone().squeeze(0) # Squeeze batch dim
                episode_jump_phase.append(current_jump_phase.clone())
                current_feet_off_ground = all_feet_off_the_ground(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_feet_off_ground.append(current_feet_off_ground.clone())
                
                takeoff_toggle = has_taken_off(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_takeoff_toggle.append(takeoff_toggle.clone()) # Store takeoff toggle

                any_feet = any_feet_on_the_ground(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_any_feet_on_ground.append(any_feet.clone()) # Store any feet status
                
                # Read from the general contact sensor (if needed for other plots)
                general_contact_sensor: ContactSensor = env.unwrapped.scene["contact_sensor"]
                current_contact_forces = general_contact_sensor.data.net_forces_w.clone().squeeze(0) # Squeeze batch dim
                episode_contact_forces.append(current_contact_forces)
                

                obs, rewards, dones, _ = env.step(actions)
                episode_rewards.append(rewards.clone().squeeze(0)) # Store rewards

                # perform operations for terminated episodes
                if len(dones) > 0 and dones[0]: # Check the first (and only) env
                    print("dones[0]: ", dones[0])
                    #print("base_contact: ", env.unwrapped.termination_manager.get_term("base_contact"))
                    print("bad_orientation: ", env.unwrapped.termination_manager.get_term("bad_orientation"))
                    print("time_out: ", env.unwrapped.termination_manager.get_term("time_out"))
                    print(f"[INFO] Episode finished after {len(episode_actions)} steps.")
                    # Don't reset RNN state here, episode is over
                    break # Exit loop after first episode finishes

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        env.close()
        if args_cli.wandb:
            with open(os.path.join(log_dir, "params/agent.yaml"), "r") as f:
                data = yaml.safe_load(f)

            run_id = data["params"]["config"]["wandb_run_id"]
            run_name = data["params"]["config"]["wandb_run_name"]
            run_project = data["params"]["config"]["wandb_project"]
            # Initialize wandb here before logging video/plots
            # Ensure project, entity etc. match your setup if needed
            wandb.init(id=run_id, project=run_project, resume="must")
            
            if args_cli.video:
                save_video_to_wandb(video_folder, log_dir, run_id, run_project, dt, cmd_wandb_suffix="") 

        if args_cli.plot:
            print("[INFO] Preparing to plot data...") # Add print statement
            plot_episode_data(robot=env.unwrapped.robot, 
                                          clipped_actions=episode_actions, 
                                          scaled_actions=episode_target_positions,
                                          joint_angles=episode_joint_angles, 
                                          joint_torques=episode_joint_torques, 
                                          com_lin_vel=episode_com_lin_vel,
                                          base_height=episode_base_height,
                                          jump_phase=episode_jump_phase,
                                          feet_off_ground=episode_feet_off_ground,
                                          dt=dt, 
                                          log_dir=log_dir,
                                          cmd_filename_suffix=cmd_filename_suffix,
                                          cmd_wandb_suffix=cmd_wandb_suffix,
                                          actual_cmd_magnitude=actual_cmd_magnitude,
                                          episode_any_feet_on_ground=episode_any_feet_on_ground,
                                          episode_takeoff_toggle=episode_takeoff_toggle,
                                          episode_contact_forces=episode_contact_forces,
                                          episode_rewards=episode_rewards) # Pass None as foot forces are removed 

        if args_cli.wandb:
            wandb.finish()
            
def save_video_to_wandb(video_folder, log_dir, run_id, run_project, dt, cmd_wandb_suffix):
    print(f"Logging video to WandB run id: {run_id} in project: {run_project}")
    import glob
    # Use the simplified suffix in the video name pattern if needed, or adjust glob
    # If cmd_filename_suffix is just "episode", the pattern is "episode_*.mp4"
    video_files = glob.glob(os.path.join(video_folder, f"{cmd_filename_suffix}_*.mp4"))
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    # Log with simplified key
    wandb.log({f"single_jump_video": wandb.Video(video_files[0], fps=int(1/dt), format="mp4")})

if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        print(f"An error occurred during main execution: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for debugging
    finally:
        if simulation_app.is_running():
             simulation_app.close()
