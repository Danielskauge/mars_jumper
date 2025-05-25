# from envs.attitude_env_cfg import AttitudeEnvCfg
# from isaaclab.assets.articulation.articulation import Articulation
# from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn, VecEnvObs
# from collections.abc import Sequence
# import torch
# from isaaclab.managers import SceneEntityCfg
# import logging

# from terms.phase import Phase

# logger = logging.getLogger(__name__)

# class AttitudeEnv(ManagerBasedRLEnv):
#     cfg: "AttitudeEnvCfg"

#     def __init__(self, cfg, **kwargs):
#         super().__init__(cfg=cfg, **kwargs)
#         self.stable_duration = torch.zeros(self.num_envs, device=self.device)
#         self.angle_error = torch.zeros(self.num_envs, device=self.device)
#         self.robot: Articulation = self.scene[SceneEntityCfg("robot").name]
#         self.prev_joint_vel = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)
#         self.jump_phase = torch.full((self.num_envs,), Phase.FLIGHT, device=self.device)

#     def step(self, action: torch.Tensor) -> VecEnvStepReturn:
#         obs_buf, reward_buf, terminated_buf, truncated_buf, extras = super().step(action)  
        
#         self.prev_joint_vel = self.robot.data.joint_vel.clone()
        
#         quat = self.robot.data.root_quat_w
#         w = quat[:, 0]
#         angle = 2 * torch.acos(torch.clamp(w, min=-1.0, max=1.0))
#         self.angle_error = torch.abs(angle)

#         is_stable = self.angle_error < self.cfg.success_angle_threshold
#         self.stable_duration[is_stable] += self.cfg.real_time_control_dt
#         self.stable_duration[~is_stable] = 0

#         return obs_buf, reward_buf, terminated_buf, truncated_buf, extras

#     def _reset_idx(self, env_ids: Sequence[int]):
#         if len(env_ids) > 0:
#             # Calculate metrics before reset
#             final_angle_error_mean = self.angle_error[env_ids].mean().item() 
            
#             #env is successful if the robot is stable for the duration threshold before the episode ends
#             success_rate = torch.mean((self.stable_duration[env_ids] > self.cfg.success_duration_threshold).float()).item()  
                      
#             # Reset state for these envs
#             self.stable_duration[env_ids] = 0
            
#             super()._reset_idx(env_ids)
            
#             # Log metrics
#             self.extras["log"].update({"success_rate": success_rate, 
#                                        "final_angle_error": final_angle_error_mean})
            
