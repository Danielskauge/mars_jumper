# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import atexit
import signal
import sys
import subprocess
import logging

from isaaclab.app import AppLauncher

# Suppress INFO messages from the TensorBoard logger
# Set level to WARNING to only show warnings and errors
logging.getLogger('tensorboard').setLevel(logging.WARNING)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--wandb", action="store_true", default=False, help="Whether to use wandb for logging.")
parser.add_argument("--run", type=str, default=None, help="Name of the run.")
parser.add_argument("--project", type=str, default=None, help="Name of the project.")
# Add new agent config arguments
parser.add_argument("--horizon_length", type=int, default=None, help="Horizon length for PPO.")
parser.add_argument("--num_minibatches", type=int, default=None, help="Number of minibatches for PPO.")
parser.add_argument("--minibatch_size", type=int, default=None, help="Minibatch size for PPO.")
parser.add_argument("--mini_epochs", type=int, default=None, help="Number of mini-epochs for PPO.")
parser.add_argument("--lr_schedule", type=str, default=None, help="Learning rate schedule (e.g., 'linear', 'adaptive').")
parser.add_argument("--learning_rate", type=float, default=None, help="Initial learning rate.")
parser.add_argument("--gamma", type=float, default=None, help="Discount factor (gamma).")
parser.add_argument("--entropy_coef", type=float, default=None, help="Entropy coefficient.")
parser.add_argument("--e_clip", type=float, default=None, help="Clipping parameter for PPO.")
parser.add_argument("--kl_threshold", type=float, default=None, help="KL threshold for adaptive LR schedule.")
parser.add_argument("--regularizer", type=str, default=None, help="Regularizer name (e.g., 'kl').")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import class_to_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import envs  # noqa: F401
import wandb  # Import wandb

@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    
    print(f"args_cli.task: {args_cli.task}")
    print(f"args_cli.project: {args_cli.project}")

    # Initialize wandb *before* configuring env/agent if using sweeps
    run = None
    init_wandb = False
    if args_cli.wandb: # Check if --wandb flag is set (needed for sweeps too)
        if not hasattr(app_launcher, "global_rank") or app_launcher.global_rank == 0:
            init_wandb = True

    if init_wandb:
        run = wandb.init(
            project= args_cli.project,
            # config is automatically populated by sweep agent, or use agent_cfg if not a sweep
            sync_tensorboard=True,
            monitor_gym=args_cli.video,
            save_code=True,
            name=args_cli.run , # Use provided name or let wandb generate default
            resume="allow",
            settings=wandb.Settings(start_method="thread")
        )
    else:
        # Disable wandb if not rank 0 or --wandb not specified
        os.environ["WANDB_MODE"] = "disabled"
        
    print()
        
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.device is not None:
        agent_cfg["params"]["config"]["device"] = args_cli.device
        agent_cfg["params"]["config"]["device_name"] = args_cli.device
    
    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # override agent config with CLI arguments
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.horizon_length is not None:
        agent_cfg["params"]["config"]["horizon_length"] = args_cli.horizon_length
    if args_cli.num_minibatches is not None:
        agent_cfg["params"]["config"]["num_minibatches"] = args_cli.num_minibatches
    if args_cli.minibatch_size is not None:
        agent_cfg["params"]["config"]["minibatch_size"] = args_cli.minibatch_size
    if args_cli.mini_epochs is not None:
        agent_cfg["params"]["config"]["mini_epochs"] = args_cli.mini_epochs
    if args_cli.lr_schedule is not None:
        agent_cfg["params"]["config"]["lr_schedule"] = args_cli.lr_schedule
    if args_cli.learning_rate is not None:
        agent_cfg["params"]["config"]["learning_rate"] = args_cli.learning_rate
    if args_cli.gamma is not None:
        agent_cfg["params"]["config"]["gamma"] = args_cli.gamma
    if args_cli.entropy_coef is not None:
        agent_cfg["params"]["config"]["entropy_coef"] = args_cli.entropy_coef
    if args_cli.e_clip is not None:
        agent_cfg["params"]["config"]["e_clip"] = args_cli.e_clip
    if args_cli.kl_threshold is not None:
        agent_cfg["params"]["config"]["kl_threshold"] = args_cli.kl_threshold
    if args_cli.regularizer is not None:
        agent_cfg["params"]["config"]["regularizer"]["name"] = args_cli.regularizer
        
    if args_cli.wandb:
        agent_cfg["params"]["config"]["wandb"] = True
        if run:
            agent_cfg["params"]["config"]["wandb_run_id"] = run.id
            agent_cfg["params"]["config"]["wandb_run_name"] = run.name
        agent_cfg["params"]["config"]["wandb_project"] = args_cli.project

    # Check if minibatch size is correct
    minibatch_size = agent_cfg["params"]["config"]["minibatch_size"]
    horizon_length = agent_cfg["params"]["config"]["horizon_length"]
    num_minibatches = agent_cfg["params"]["config"]["num_minibatches"]
    num_envs = env_cfg.scene.num_envs
    calculated_minibatch_size = horizon_length * num_envs / num_minibatches
    assert minibatch_size == calculated_minibatch_size, f"Minibatch size mismatch: (in yaml/sweep) {minibatch_size} != (calculated) {calculated_minibatch_size}"
       
    
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs (use run name if available from wandb/cli)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.wandb:
        log_dir += "_" + run.name

    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # Log final configs to WandB if initialized
    if init_wandb and run:
            # Convert env_cfg to a nested dictionary before logging
            rewards_dict = class_to_dict(env_cfg.rewards)
            termination_dict = class_to_dict(env_cfg.terminations)
            network_dict = class_to_dict(agent_cfg["params"]["network"])
            if hasattr(env_cfg, "curriculum"):
                curriculum_dict = class_to_dict(env_cfg.curriculum)
                run.config.update({"curriculum": curriculum_dict}, allow_val_change=True) # Log curriculum
            if hasattr(env_cfg, "success_criteria"):
                success_criteria_dict = class_to_dict(env_cfg.success_criteria)
                run.config.update({"success_criteria": success_criteria_dict}, allow_val_change=True) # Log success criteria
            if hasattr(env_cfg, "command_ranges"):
                command_ranges_dict = class_to_dict(env_cfg.command_ranges)
                run.config.update({"command_ranges": command_ranges_dict}, allow_val_change=True) # Log command ranges

            run.config.update(agent_cfg["params"]["config"], allow_val_change=True) # Allow changes from sweep
            run.config.update({"sim_freq": int(1/env_cfg.sim.dt)})
            run.config.update({"real_time_control_dt": int(1/env_cfg.real_time_control_dt)})
            run.config.update({"num_envs": env_cfg.scene.num_envs})
            run.config.update({"rewards": rewards_dict}, allow_val_change=True) # Allow changes from sweep
            run.config.update({"terminations": termination_dict}, allow_val_change=True) # Allow changes from sweep
            run.config.update({"network": network_dict}, allow_val_change=True) # Log network
    def signal_handler(sig, frame):
        print("[INFO] Ctrl+C detected. Finishing WandB run.")
        if init_wandb and run: # Use init_wandb flag
            run.finish()
        print("[INFO] Exiting.")
        simulation_app.close()  # Close the app to avoid hanging
        os._exit(0)  # Exit immediately

    signal.signal(signal.SIGINT, signal_handler)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg) # Dumps potentially modified env_cfg
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg) # Dumps potentially modified agent_cfg
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,  # Important: Disable default logger
        }
        print("[INFO] Recording videos during training.")
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

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path, "track": True})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "track": True})

    if init_wandb and run: # Use init_wandb flag
        run.finish()
    # close the simulator
    env.close()
    
    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
