

import gymnasium as gym

from envs.attitude_env_cfg import AttitudeEnvCfg
from envs.full_jump_env_cfg import FullJumpEnvCfg
from envs.landing_env_cfg import LandingEnvCfg
from . import agents
from .env_cfg import MarsJumperEnvCfg
from .env import MarsJumperEnv
from .attitude_env import AttitudeEnv


gym.register(
    id="attitude",
    entry_point="envs.attitude_env:AttitudeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": AttitudeEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="takeoff",
    entry_point="envs.env:MarsJumperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MarsJumperEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
) 

gym.register(
    id="landing",
    entry_point="envs.landing_env:LandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": LandingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)


gym.register(
    id="full-jump",
    entry_point="envs.full_jump_env:FullJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FullJumpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

