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


def plot_episode_data(robot, clipped_actions, scaled_actions, joint_angles, joint_torques, com_lin_vel, base_height, jump_phase, feet_off_ground, dt, log_dir, cmd_filename_suffix, cmd_wandb_suffix, actual_cmd_magnitude, episode_any_feet_on_ground, episode_takeoff_toggle, episode_contact_forces):
    """Plots the recorded actions, joint angles, torques, target positions, COM velocity, base height, feet status, and contact forces.

    Args:
        clipped_actions: A list of action tensors for the episode (clipped, [-1, 1]).
        scaled_actions: A list of target joint position tensors for the episode (scaled actions).
        joint_angles: A list of joint angle tensors for the episode.
        joint_torques: A list of joint torque tensors for the episode.
        com_lin_vel: A list of COM linear velocity tensors for the episode.
        base_height: A list of base height tensors for the episode.
        jump_phase: A list of jump phase integers for the episode.
        feet_off_ground: A list of boolean tensors indicating if feet are off ground.
        dt: The simulation timestep.
        log_dir: Directory to save the plot.
        cmd_filename_suffix: Suffix string for filenames based on command values.
        cmd_wandb_suffix: Suffix string for WandB logging keys based on command values.
        actual_cmd_magnitude: The actual command magnitude used for the episode.
        episode_any_feet_on_ground: List of booleans indicating if any feet are on the ground.
        episode_takeoff_toggle: List of booleans indicating takeoff toggle status.
        episode_contact_forces: List of contact force tensors for the episode.
    """
    print("[INFO] Plotting episode data including actions, joints, COM, base, phase, status, and contact forces...")
    plots_dir = os.path.join(log_dir, "plots") # Define plots directory path
    os.makedirs(plots_dir, exist_ok=True) # Create plots directory if it doesn't exist

    actions = torch.stack(clipped_actions).cpu().numpy()
    joint_angles = torch.stack(joint_angles).cpu().numpy()
    joint_torques = torch.stack(joint_torques).cpu().numpy()
    target_positions = torch.stack(scaled_actions).cpu().numpy()
    com_lin_vel = torch.stack(com_lin_vel).cpu().numpy()
    base_height = torch.stack(base_height).cpu().numpy()
    jump_phase = torch.stack(jump_phase).cpu().numpy() # Convert jump_phase to numpy
    feet_off_ground = torch.stack(feet_off_ground).cpu().numpy().astype(int) # Convert feet status to numpy int (0 or 1)
    time = np.arange(len(actions)) * dt

    # Convert contact forces if available
    contact_forces_available = bool(episode_contact_forces) # Check if the list is non-empty
    if contact_forces_available:
        contact_forces = torch.stack(episode_contact_forces).cpu().numpy()
        # Check if the tensor is empty (might happen if collected but always zero)
        if contact_forces.size == 0:
            contact_forces_available = False
            print("[Warning] Contact force data was collected but appears empty. Skipping contact force plotting.")

    # Find flight phase intervals
    flight_intervals = []
    in_flight = False
    start_time = None
    for t_idx, phase in enumerate(jump_phase):
        current_time = time[t_idx]
        if phase == Phase.FLIGHT and not in_flight:
            start_time = current_time
            in_flight = True
        elif phase != Phase.FLIGHT and in_flight:
            flight_intervals.append((start_time, current_time))
            in_flight = False
    if in_flight: # Handle case where episode ends during flight
        flight_intervals.append((start_time, time[-1]))

    # Helper function to add flight phase shading
    def add_flight_shading(ax):
        for start, end in flight_intervals:
            ax.axvspan(start, end, color='yellow', alpha=0.3, label='Flight Phase' if start == flight_intervals[0][0] else "") # Only label first span

    # Helper function to add phase shading for all phases
    def add_all_phase_shading(ax, time, jump_phase_data, add_legend_labels=True):
        phase_colors = {
            Phase.CROUCH: ('lightblue', 0.3),
            Phase.TAKEOFF: ('lightgreen', 0.3),
            Phase.FLIGHT: ('gold', 0.3),
            Phase.LANDING: ('lightcoral', 0.3)
        }
        phases = np.asarray(jump_phase_data).flatten() # Ensure it's a flat numpy array
        unique_phases = np.unique(phases)
        labeled_phases = set()

        start_idx = 0
        for i in range(1, len(phases)):
            if phases[i] != phases[start_idx]:
                phase_val = int(phases[start_idx])
                try:
                    phase_enum = Phase(phase_val)
                    color, alpha = phase_colors.get(phase_enum, ('gray', 0.1)) # Default to gray if phase not in map
                    # Only add label if add_legend_labels is True and phase hasn't been labeled yet
                    label = f"{phase_enum.name} Phase" if add_legend_labels and phase_enum not in labeled_phases else ""
                    ax.axvspan(time[start_idx], time[i], color=color, alpha=alpha, label=label)
                    if label:
                        labeled_phases.add(phase_enum)
                except ValueError:
                    print(f"Warning: Encountered unknown phase value {phase_val}") # Handle unexpected phase values
                start_idx = i

        # Shade the last segment
        if start_idx < len(phases):
             phase_val = int(phases[start_idx])
             try:
                 phase_enum = Phase(phase_val)
                 color, alpha = phase_colors.get(phase_enum, ('gray', 0.1))
                 # Only add label if add_legend_labels is True and phase hasn't been labeled yet
                 label = f"{phase_enum.name} Phase" if add_legend_labels and phase_enum not in labeled_phases else ""
                 ax.axvspan(time[start_idx], time[-1], color=color, alpha=alpha, label=label)
                 if label:
                    labeled_phases.add(phase_enum)
             except ValueError:
                 print(f"Warning: Encountered unknown phase value {phase_val}")

    # Group joints by type
    joint_groups = {
        'Abductors': ['LF_HAA', 'LH_HAA', 'RF_HAA', 'RH_HAA'],
        'Hip': ['LF_HFE', 'LH_HFE', 'RF_HFE', 'RH_HFE'],
        'Knee': ['LF_KFE', 'LH_KFE', 'RF_KFE', 'RH_KFE']
    }

    # Plot clipped actions
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True) # Use subplots for easier sharing and shading
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs[i]
        joint_names = joint_groups[group_name]
        for name in joint_names:
            idx = robot.find_joints(name)[0]
            ax.plot(time, actions[:, idx], label=name)
        ax.set_title(f"{group_name} Clipped Actions vs Time")
        ax.set_ylabel("Action Value")
        ax.legend()
        add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend
        ax.grid(True) # Add grid
    axs[-1].set_xlabel("Time (s)") # Set xlabel only on the bottom plot
    plt.tight_layout()
    # Save clipped actions plot
    clipped_actions_path = os.path.join(plots_dir, f"clipped_actions_plot_{cmd_filename_suffix}.png") # Use plots_dir
    plt.savefig(clipped_actions_path)
    plt.close(fig) # Close the figure

    # Plot scaled actions (target positions)
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True) # Use subplots
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs[i]
        joint_names = joint_groups[group_name]
        for name in joint_names:
            idx = robot.find_joints(name)[0]
            # Convert radians to degrees for plotting
            target_pos_deg = target_positions[:, idx] * 180.0 / np.pi 
            ax.plot(time, target_pos_deg, label=name)
        ax.set_title(f"{group_name} Target Positions vs Time")
        ax.set_ylabel("Position (deg)") # Update label
        ax.legend()
        add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend
        ax.grid(True) # Add grid
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    # Save scaled actions plot
    scaled_actions_path = os.path.join(plots_dir, f"scaled_actions_plot_{cmd_filename_suffix}.png") # Use plots_dir
    plt.savefig(scaled_actions_path)
    plt.close(fig) # Close the figure

    # Plot joint angles
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True) # Use subplots
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs[i]
        joint_names = joint_groups[group_name]
        for name in joint_names:
            idx = robot.find_joints(name)[0]
            ax.plot(time, joint_angles[:, idx], label=name)
        ax.set_title(f"{group_name} Joint Angles vs Time")
        ax.set_ylabel("Joint Angle (rad)")
        ax.legend()
        add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend
        ax.grid(True) # Add grid
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    # Save angles plot
    angles_path = os.path.join(plots_dir, f"angles_plot_{cmd_filename_suffix}.png") # Use plots_dir
    plt.savefig(angles_path)
    plt.close(fig) # Close the figure

    # Plot joint torques
    fig, axs = plt.subplots(3, 1, figsize=(15, 12), sharex=True) # Use subplots
    for i, group_name in enumerate(['Abductors', 'Hip', 'Knee']):
        ax = axs[i]
        joint_names = joint_groups[group_name]
        for name in joint_names:
            idx = robot.find_joints(name)[0]
            ax.plot(time, joint_torques[:, idx], label=name)
        ax.set_title(f"{group_name} Joint Torques vs Time")
        ax.set_ylabel("Joint Torque (Nm)")
        ax.legend()
        add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend
        ax.grid(True) # Add grid
    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    # Save torques plot
    torques_path = os.path.join(plots_dir, f"torques_plot_{cmd_filename_suffix}.png") # Use plots_dir
    plt.savefig(torques_path)
    plt.close(fig) # Close the figure

    # Plot COM Linear Velocity Components and Magnitude
    fig, axs = plt.subplots(2, 1, figsize=(15, 12), sharex=True) # Use subplots

    # Subplot 1: Components
    ax = axs[0]
    vel_components = ['X', 'Y', 'Z']
    for i in range(3):
        ax.plot(time, com_lin_vel[:, i], label=f"COM Vel {vel_components[i]}")
    ax.set_title("COM Linear Velocity Components vs Time")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend

    # Subplot 2: Magnitude
    ax = axs[1]
    com_lin_vel_mag = np.linalg.norm(com_lin_vel, axis=1)
    ax.plot(time, com_lin_vel_mag, label="COM Velocity Magnitude")
    # Add horizontal line for command magnitude
    ax.axhline(y=actual_cmd_magnitude, color='r', linestyle='--', label=f"Command Magnitude ({actual_cmd_magnitude:.2f})")
    ax.set_title("COM Linear Velocity Magnitude vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.legend()
    add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend

    plt.tight_layout()
    com_vel_path = os.path.join(plots_dir, f"com_velocity_plot_{cmd_filename_suffix}.png") # Use plots_dir
    plt.savefig(com_vel_path)
    plt.close(fig) # Close the figure

    # Plot Base Height and Feet Off Ground Status together
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True) # Use subplots, share x-axis

    # Subplot 1: Base Height
    ax = axs[0]
    ax.plot(time, base_height, label="Base Height")
    ax.set_title("Base Height vs Time")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True) # Add grid
    add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend

    # Subplot 2: Feet Off Ground Status
    ax = axs[1]
    ax.step(time, feet_off_ground, where='post', label='All Feet Off Ground (1=True, 0=False)')
    ax.set_title("Feet Contact Status vs Time")
    ax.set_xlabel("Time (s)") # Set xlabel only on the bottom plot
    ax.set_ylabel("Status")
    ax.set_yticks([0, 1]) # Set y-ticks to 0 and 1
    ax.set_ylim([-0.1, 1.1]) # Set y-limits for clarity
    ax.legend()
    ax.grid(True) # Add grid
    add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use all phases, no legend

    plt.tight_layout()
    base_feet_path = os.path.join(plots_dir, f"base_feet_plot_{cmd_filename_suffix}.png")
    plt.savefig(base_feet_path)
    plt.close(fig)

    # Plot Jump Phase, Takeoff Toggle, and Ground Contact
    fig, ax = plt.subplots(1, 1, figsize=(15, 6)) # Single subplot

    # Add phase shading first so lines are on top
    add_all_phase_shading(ax, time, jump_phase) # Keep legend labels for this plot

    # Convert boolean/int data to numpy arrays
    any_feet_on_ground_np = torch.stack(episode_any_feet_on_ground).cpu().numpy().astype(int)
    takeoff_toggle_np = torch.stack(episode_takeoff_toggle).cpu().numpy().astype(int)

    # Plot signals
    ax.step(time, any_feet_on_ground_np, where='post', label='Any Feet On Ground (1=True)', linewidth=2)
    ax.step(time, takeoff_toggle_np, where='post', label='Takeoff Toggle (1=True)', linestyle='--', linewidth=2)

    ax.set_title("Jump Phase, Ground Contact, and Takeoff Status vs Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Status (0 or 1)")
    ax.set_yticks([0, 1])
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc='upper right') # Adjust legend location if needed
    ax.grid(True, which='both', linestyle='-', linewidth=0.5) # Add grid

    plt.tight_layout()
    phase_status_path = os.path.join(plots_dir, f"phase_status_plot_{cmd_filename_suffix}.png")
    plt.savefig(phase_status_path)
    plt.close(fig)

    if args_cli.wandb:
        # Log all plots to wandb
        wandb_log_data = {
            f"clipped_actions_plot_{cmd_wandb_suffix}": wandb.Image(clipped_actions_path),
            f"scaled_actions_plot_{cmd_wandb_suffix}": wandb.Image(scaled_actions_path),
            f"angles_plot_{cmd_wandb_suffix}": wandb.Image(angles_path),
            f"torques_plot_{cmd_wandb_suffix}": wandb.Image(torques_path),
            f"com_velocity_plot_{cmd_wandb_suffix}": wandb.Image(com_vel_path), # Updated wandb log key and path
            f"base_feet_plot_{cmd_wandb_suffix}": wandb.Image(base_feet_path), # Add combined base/feet plot
            f"phase_status_plot_{cmd_wandb_suffix}": wandb.Image(phase_status_path), # Add phase status plot
            f"contact_forces_plot_{cmd_wandb_suffix}": wandb.Image(episode_contact_forces[0]) if episode_contact_forces else wandb.Image(None) # Add contact forces plot
        }
        wandb.log(wandb_log_data)

    # Plot Contact Forces if available
    contact_forces_plot_path = None
    if contact_forces_available:
        print("[INFO] Plotting contact forces...")
        fig_cf, axs_cf = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

        body_groups = {
            'Feet': ['LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'],
            'Shanks': ['LF_SHANK', 'RF_SHANK', 'LH_SHANK', 'RH_SHANK'],
            'Thighs': ['LF_THIGH', 'RF_THIGH', 'LH_THIGH', 'RH_THIGH'],
            'Base': ['base'] # Assuming 'base' is the name
        }
        
        num_bodies = contact_forces.shape[1]

        for i, (group_name, body_names) in enumerate(body_groups.items()):
            ax = axs_cf[i]
            body_indices = []
            valid_body_names_in_group = []
            for name in body_names:
                try:
                    # find_bodies returns a tuple of (indices, names)
                    # We want the list of indices (first element of tuple)
                    indices_list = robot.find_bodies(name)[0]
                    
                    # Check if any bodies were found for this name
                    if not indices_list:
                         print(f"[Warning] No body found for name pattern '{name}'. Skipping.")
                         continue # Skip to the next name in body_names
                    
                    # Take the first index found
                    idx = indices_list[0] 

                    if idx < num_bodies: # Ensure index is valid
                         body_indices.append(idx)
                         valid_body_names_in_group.append(name) # Store the original name pattern used for lookup
                    else:
                        print(f"[Warning] Body index {idx} for name pattern '{name}' out of range ({num_bodies}). Skipping.")
                except (IndexError, ValueError, AttributeError) as e:
                    # Handle cases where find_bodies might fail (though primary check is indices_list)
                    print(f"[Warning] Error processing body name pattern '{name}'. Skipping. Error: {e}")

            if not body_indices:
                print(f"[Warning] No valid body indices found for group '{group_name}'. Skipping plot for this group.")
                ax.set_title(f"{group_name} Contact Forces (No bodies found)")
                # Add dummy elements to maintain structure if needed, or just skip
                ax.text(0.5, 0.5, 'No data found for this group', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.grid(True)
                continue # Skip to next group

            ax.set_title(f"{group_name} Contact Force Magnitudes vs Time")
            for j, idx in enumerate(body_indices):
                # Calculate magnitude (norm) along the force vector axis (axis=2)
                force_magnitude = np.linalg.norm(contact_forces[:, idx, :], axis=1)
                ax.plot(time, force_magnitude, label=f"{valid_body_names_in_group[j]}") # Use j for indexing valid_body_names_in_group
            
            ax.set_ylabel("Force Magnitude (N)")
            ax.legend()
            ax.grid(True)
            # Add phase shading if desired
            add_all_phase_shading(ax, time, jump_phase, add_legend_labels=False) # Use the existing shading function, no legend
        
        axs_cf[-1].set_xlabel("Time (s)") # Set xlabel only on the bottom plot

        plt.tight_layout()
        contact_forces_plot_path = os.path.join(plots_dir, f"contact_forces_plot_{cmd_filename_suffix}.png")
        plt.savefig(contact_forces_plot_path)

        if args_cli.wandb:
            # Check if the plot was actually created before logging
            if contact_forces_plot_path and os.path.exists(contact_forces_plot_path):
                 wandb.log({f"contact_forces_plot_{cmd_wandb_suffix}": wandb.Image(contact_forces_plot_path)})
            else:
                 print("[Warning] Contact force plot was not generated or saved, skipping WandB log.")

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
    # Format for filenames/logging
    cmd_filename_suffix = f"mag{actual_cmd_magnitude:.1f}_pitch{actual_cmd_pitch:.1f}"
    cmd_wandb_suffix = f"(mag:{actual_cmd_magnitude:.1f}, pitch:{actual_cmd_pitch:.1f})"

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
            "name_prefix": f"{cmd_filename_suffix}_", # Use actual command values
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
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                #print("obs: ", obs)
                obs = agent.obs_to_torch(obs)
                #print("obs torch: ", obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=True)
                
                # Store actions and joint angles
                # Squeeze is okay since num_envs is always 1
                episode_actions.append(actions.clone().squeeze(0))
                episode_joint_angles.append(env.unwrapped.robot.data.joint_pos.clone().squeeze(0))
                episode_joint_torques.append(env.unwrapped.robot.data.applied_torque.clone().squeeze(0))
                # Retrieve target joint positions (these are the scaled actions for ImplicitActuator)
                target_positions = env.unwrapped.robot.data.joint_pos_target.clone().squeeze(0)
                episode_target_positions.append(target_positions)
                # Get and store COM linear velocity and base height
                current_com_vel = get_center_of_mass_lin_vel(env.unwrapped).squeeze(0) # Squeeze batch dim
                current_base_height = env.unwrapped.robot.data.root_pos_w[:, 2].squeeze(0) # Squeeze batch dim
                episode_com_lin_vel.append(current_com_vel.clone())
                episode_base_height.append(current_base_height.clone())
                # Get and store jump phase
                current_jump_phase = env.unwrapped.jump_phase.clone().squeeze(0) # Squeeze batch dim
                episode_jump_phase.append(current_jump_phase.clone())
                # Get and store feet status
                current_feet_off_ground = all_feet_off_the_ground(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_feet_off_ground.append(current_feet_off_ground.clone())
                
                # Get and store takenoff toggle
                takeoff_toggle = has_taken_off(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_takeoff_toggle.append(takeoff_toggle.clone()) # Store takeoff toggle

                # Get and store any feet on ground status
                any_feet = any_feet_on_the_ground(env.unwrapped).squeeze(0) # Squeeze batch dim
                episode_any_feet_on_ground.append(any_feet.clone()) # Store any feet status
                
                # Get and store net contact forces using the scene's contact sensor
                try:
                    # Define the contact sensor config name (assuming it's 'contact_forces')
                    contact_sensor_cfg_name = SceneEntityCfg("contact_forces").name 
                    contact_sensor: ContactSensor = env.unwrapped.scene[contact_sensor_cfg_name]
                    # Get net forces (shape: [num_envs, num_bodies, 3])
                    current_contact_forces = contact_sensor.data.net_forces_w.clone().squeeze(0) # Squeeze batch dim
                    episode_contact_forces.append(current_contact_forces)
                except KeyError:
                    # Handle case where contact sensor is not found in the scene
                    if not episode_contact_forces: # Log only once
                        print(f"[Warning] Contact sensor '{contact_sensor_cfg_name}' not found in env.scene. Skipping contact force plotting.")
                    # Append None or handle appropriately in plotting
                    pass 
                except AttributeError:
                     # Handle case where contact sensor data or net_forces_w attribute doesn't exist
                    if not episode_contact_forces: # Log only once
                        print(f"[Warning] Attribute 'data.net_forces_w' not found for contact sensor '{contact_sensor_cfg_name}'. Skipping contact force plotting.")
                    # Append None or handle appropriately in plotting
                    pass

                obs, _, dones, _ = env.step(actions)

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
        # close the environment
        env.close()
        if args_cli.wandb:
            with open(os.path.join(log_dir, "params/agent.yaml"), "r") as f:
                data = yaml.safe_load(f)

            run_id = data["params"]["config"]["wandb_run_id"]
            run_name = data["params"]["config"]["wandb_run_name"]
            run_project = data["params"]["config"]["wandb_project"]
            wandb.init(id=run_id, project=run_project, resume="must")
            
            if args_cli.video:
                save_video_to_wandb(video_folder, log_dir, run_id, run_project, dt, cmd_wandb_suffix=cmd_wandb_suffix)
        # Always plot regardless of wandb
        # Plot only if --plot flag is set
        if args_cli.plot:
            plot_episode_data(env.unwrapped.robot, 
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
                                          episode_contact_forces=episode_contact_forces) # Pass contact forces

        if args_cli.wandb:
            wandb.finish()
            
def save_video_to_wandb(video_folder, log_dir, run_id, run_project, dt, cmd_wandb_suffix):
    print(f"Logging video to WandB run id: {run_id} in project: {run_project}")
    import glob
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    wandb.log({f"single_jump_video_{cmd_wandb_suffix}": wandb.Video(video_files[0], fps=1/dt, format="mp4")})

if __name__ == "__main__":
    # run the main function
    try:
        main()
    except Exception as e:
        print(f"An error occurred during main execution: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback for debugging
    finally:
        # close sim app
        if simulation_app.is_running():
             simulation_app.close()
