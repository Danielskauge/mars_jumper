# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint for multiple episodes using RL-Games, with command ranges."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import shutil

import wandb
import yaml

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games for multiple episodes.")
parser.add_argument("--video", action="store_true", default=True, help="Record a video spanning all episodes.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log video to WandB (if video is enabled).")
parser.add_argument("--video_length_s", type=int, default=10, help="Length of the recorded video (in seconds).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="full-jump", required=False, help="Name of the task.")
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
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import math
import os
import time
import torch

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import envs

def main():
    """Play with RL-Games agent for multiple episodes."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Construct video_name_prefix based on command ranges
    video_name_prefix = "episode"  # Default prefix as cmd_ args are removed

    # wrap for video recording if enabled
    video_folder = os.path.join(log_root_path, log_dir, "videos", "play_multiple")
    # Delete existing video folder first if it exists
    if os.path.exists(video_folder):
        shutil.rmtree(video_folder)
    
    # Ensure the base directory exists (always do this)
    os.makedirs(video_folder, exist_ok=True)

    # Define video arguments and apply the wrapper (always do this if video is enabled)
    video_kwargs = {
        "video_folder": video_folder,
        "step_trigger": lambda step: step == 0,
        "video_length": int(args_cli.video_length_s / env.unwrapped.step_dt), # Ensure integer steps
        "name_prefix": video_name_prefix,
        "disable_logger": True,
    }
    print(f"[INFO] Recording video for {args_cli.video_length_s} seconds with prefix '{video_name_prefix}'.")
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

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    
    dt = env.unwrapped.physics_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    total_episodes_done = 0 # Track total episodes completed across all envs

    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()

    # Wrap the simulation loop in a try...finally block
    try:
        # simulate environment
        # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
        #   attempt to have complete control over environment stepping. However, this removes other
        #   operations such as masking that is used for multi-agent learning by RL-Games.
        while simulation_app.is_running():
            # Check if max_steps reached
            if timestep >= args_cli.video_length_s / env.unwrapped.step_dt:
                print(f"[INFO] Reached max_steps ({args_cli.video_length_s / env.unwrapped.step_dt}). Stopping simulation.")
                break

            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                # convert obs to agent format
                obs = agent.obs_to_torch(obs)
                # agent stepping
                actions = agent.get_action(obs, is_deterministic=True)

                # env stepping
                obs, _, dones, _ = env.step(actions)

                # perform operations for terminated episodes
                terminated_envs = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(terminated_envs) > 0:
                    total_episodes_done += len(terminated_envs)
                    # reset rnn state for terminated episodes
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:, terminated_envs, :] = 0.0 # Use terminated_envs indices
            timestep += 1

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # close the environment
        env.close()
        print(f"[INFO] Completed {total_episodes_done} episodes in {timestep} steps.")
        # Save video to wandb if enabled and video was recorded
        if args_cli.wandb and video_folder:
            save_video_to_wandb(video_folder, log_dir, video_name_prefix, env)

def save_video_to_wandb(video_folder, log_dir, video_name_prefix, env):
    with open(os.path.join(log_dir, "params/agent.yaml"), "r") as f:
        data = yaml.safe_load(f)

    run_id = data["params"]["config"]["wandb_run_id"]
    run_name = data["params"]["config"]["wandb_run_name"]
    run_project = data["params"]["config"]["wandb_project"]
    
    wandb.init(id=run_id, project=run_project, resume="must")
    
    print(f"Logging video to WandB run: {run_name}, id: {run_id} in project: {run_project}")
    import glob
    # Use the prefix to find the correct video file
    video_files = glob.glob(os.path.join(video_folder, f"{video_name_prefix}*.mp4"))
    if not video_files:
        print(f"No video files found in {video_folder} with prefix {video_name_prefix}")
        # Fallback: try finding any mp4 if the prefixed one wasn't found (maybe RecordVideo changed behavior)
        video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        if not video_files:
             print(f"No video files found in {video_folder} at all.")
             wandb.finish()
             return
        else:
             print(f"Warning: Could not find video with prefix '{video_name_prefix}', logging the first video found: {video_files[0]}")

    # Use the prefix in the WandB log key
    wandb_log_key = f"video_{video_name_prefix}"
    wandb.log({wandb_log_key: wandb.Video(video_files[0], fps=int(1/env.unwrapped.step_dt), format="mp4")}) # Use env dt for fps
    wandb.finish()



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
