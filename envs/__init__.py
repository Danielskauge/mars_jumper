

import gymnasium as gym

from envs.full_jump_env_cfg import FullJumpEnvCfg
from . import agents


gym.register(
    id="full-jump",
    entry_point="envs.full_jump_env:FullJumpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FullJumpEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)

