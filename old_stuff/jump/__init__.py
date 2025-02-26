# import gymnasium as gym

# from . import agents
# from .jump_env import JumpEnv
# from .jump_env_config import JumpEnvCfg

# gym.register(
#     id="mars-jumper",
#     entry_point="envs.jump:JumpEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": JumpEnvCfg,
#         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
#         "rsl_rl_cfg_entry_point": None,
#         "skrl_cfg_entry_point": None,
#     },
# ) 

# print(f"\nRegistered MarsJumper environment with gymnasium\n")