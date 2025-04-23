from envs.env_cfg import MarsJumperEnvCfg
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn
from collections.abc import Sequence
import torch
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from terms.phase import Phase
from terms.utils import update_jump_phase, log_phase_info
import logging

logger = logging.getLogger(__name__)

class LandingEnv(ManagerBasedRLEnv):
    cfg: "MarsJumperEnvCfg"

    def __init__(self, cfg, **kwargs):
        # Initialize internal variables for tracking success
        #Adjust the threshold as needed for your definition of success
        
        super().__init__(cfg=cfg, **kwargs)
        # Need to potentially re-initialize buffers after super().__init__ determines num_envs

        self.jump_phase = torch.full((self.num_envs,), Phase.TAKEOFF, dtype=torch.int32, device=self.device)
        
        self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
        self.episode_length_s = cfg.episode_length_s

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)
        self.episode_length += self.real_time_control_dt
        return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

    def _reset_idx(self, env_ids: Sequence[int]):
        # Log success rate for the environments being reset
        # Call the parent's reset method to handle standard resets and logging
        if len(env_ids) > 0:
            mean_episode_length = torch.mean(self.episode_length[env_ids])
            # --- Call super()._reset_idx ---
            # This triggers event manager's reset mode, including reset_robot_initial_state -> reset_robot_flight_state
            super()._reset_idx(env_ids)
            # --- After super()._reset_idx ---
            self.extras["log"].update({"mean_episode_length": mean_episode_length})
            
            # Reset tracking buffers for these environments
            self.jump_phase[env_ids] = Phase.LANDING
            self.episode_length[env_ids] = 0.0