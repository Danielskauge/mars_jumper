

import gymnasium as gym

from envs.attitude_env_cfg import AttitudeEnvCfg
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
    id="full-jump",
    entry_point="envs.env:MarsJumperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MarsJumperEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
) 

# gym.register(
#     id="full-jump-v2",
#     entry_point="envs.full_jump_env:MarsJumperEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": FullJumpEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#     },
# )

