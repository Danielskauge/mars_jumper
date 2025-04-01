

import gymnasium as gym
from . import agents
from .env_cfg import MarsJumperEnvCfg
from .env import MarsJumperEnv

gym.register(
    id="mars-jumper-manager-based",
    entry_point="envs.env:MarsJumperEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MarsJumperEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
) 

print(f"\nRegistered mars-jumper-manager-based environment with gymnasium\n")  