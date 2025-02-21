from typing import Tuple, List, Dict

import numpy as np
from isaaclab.utils.math import sample_uniform
from mars_jumper.robot.robot_cfg import MarsJumperRobot
from torch import Tensor

import omni.log
import torch
from torch.nn.functional import normalize

import isaaclab.sim as sim_utils
from isaaclab.utils.buffers import TimestampedBuffer

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_rotate_inverse,
    quat_error_magnitude,
)
from isaaclab.sensors import ContactSensor

from initialization import InitializationScheme, JumpInitializerBase
from .jump_env_config import JumpEnvCfg
from .curriculum import make_initializer, get_next_curriculum


class JumpEnv(DirectRLEnv):
    cfg: JumpEnvCfg

    def __init__(self, cfg: JumpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Initialize episode reward tracking
        self._episode_reward_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "goal_pos_error_large",
                "goal_pos_error_small", 
                "terminate_goal_pos_error",
                "orientation_error",
                "speed_on_goal",
                "angvel",
                "takeoff_paws",
                "break_torque",
                "landstand_paws",
                "landstand_lin_vel",
                "landstand_joint_vel",
                "landstand_orientation", 
                "landstand_joint_pos",
                "landstand_joint_acc",
                "action_clip",
                "dof_torques",
                "dof_acc_l2",
                "action_rate_l2",
                "contact_change",
                "jerk",
                "joint_symmetry", #TODO: CHECK IF THIS IS NEEDED
            ]
        }
        
        self._init_buffers()
        self._init_indices()
        self._init_initializers()

    def _init_buffers(self):
        """Initialize buffers for storing robot state, commands and other data."""
        with torch.device(self.device):
            # Action buffers
            self._actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._clipped_actions = torch.zeros_like(self._actions)
            self._scaled_and_shifted_actions = torch.zeros_like(self._actions)
            self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space)
            self._previous_torques = torch.zeros_like(self._actions)
            
            # Command and goal tracking
            self._command_position = torch.zeros(self.num_envs, 3)
            self._close_to_goal_count = torch.zeros(self.num_envs, dtype=torch.int)
            self._idle_count = torch.zeros(self.num_envs, dtype=torch.int)
            
            # Termination flags
            self._terminate_on_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_touchdown = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_takeoff = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_walking = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_landed = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_collision = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_idle = torch.zeros(self.num_envs, dtype=torch.bool)
            self._episode_end = torch.zeros(self.num_envs, dtype=torch.bool)

            # Robot state tracking
            self._init_schemes = torch.zeros(self.num_envs, dtype=torch.int)
            self._is_airborne = torch.zeros(self.num_envs, dtype=torch.bool)
            self._has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool)
            
            self._steps_since_takeoff = torch.zeros(self.num_envs, dtype=torch.int)
            

            # Timestamped buffers for caching computed values
            self._contact_states = TimestampedBuffer(
                torch.zeros(self.num_envs, 4, dtype=torch.bool)
            )
            self._collision_states = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.bool)
            )
            self._airborne = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.bool)
            )
            self._takeoff = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.bool)
            )
            self._touchdown = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.bool)
            )
            self._goal_vec = TimestampedBuffer(
                torch.zeros(self.num_envs, 3, dtype=torch.float)
            )
            self._start_vec = TimestampedBuffer(
                torch.zeros(self.num_envs, 3, dtype=torch.float)
            )
            self._speed_on_goal = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.float)
            )
            self._rot_error_rad = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.float)
            )
            self._takeoff_pos = TimestampedBuffer(
                torch.zeros(self.num_envs, 3, dtype=torch.float)
            )
            self._touchdown_pos = TimestampedBuffer(
                torch.zeros(self.num_envs, 3, dtype=torch.float)
            )

            # curriculums
            self._max_command_curriculum = torch.zeros(self.num_envs, dtype=torch.int)
            self._min_command_curriculum = torch.zeros(self.num_envs, dtype=torch.int)
            self._num_scheme_curriculums = torch.zeros(self.num_envs, dtype=torch.int)
            self._scheme_curriculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._command_curriculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._curriculum_progress = torch.zeros(self.num_envs, dtype=torch.int)
            self._game_won = torch.zeros(self.num_envs, dtype=torch.bool)
            self._balanced_scheme = False
            self._scheme_count: Dict[InitializationScheme, int] = {}

    def _init_indices(self):
        """Get specific body indices for the robot."""
        # self._feet_idx = self._robot.feet_idx
        # self._avoid_contact_body_idx = self._robot.avoid_contact_body_idx
        # self._joint_idx = self._robot.joints_idx
        
        #must be here instead of in MarsJumperRobot class as it does not find the _physx_view attribute otherwise for some reason
        self.hip_abduction_joints_idx: List[int] = self._robot.find_joints(self._robot.cfg.HIP_ABDUCTION_JOINTS_REGEX)[0] #TODO: should be tensor as in olympus?
        self.avoid_contact_body_idx: List[int] = self._robot.find_bodies(self._robot.cfg.AVOID_CONTACT_BODIES_REGEX)[0]
        self.feet_idx: List[int] = self._robot.find_bodies(self._robot.cfg.FEET_REGEX)[0] #Find bodies returns a tuple of two lists, first is the body indices, second is the body names
        self.hip_flexion_joints_idx: List[int] = self._robot.find_joints(self._robot.cfg.HIP_FLEXION_JOINTS_REGEX)[0]
        self.knee_joints_idx: List[int] = self._robot.find_joints(self._robot.cfg.KNEE_JOINTS_REGEX)[0]
        self.joints_idx: List[int] = (self.hip_abduction_joints_idx +
                                    self.hip_flexion_joints_idx +
                                    self.knee_joints_idx)
        

    def _init_initializers(self):
        """Initialize the robot's state based on different schemes and curriculums."""
        self._initializers: Dict[
            InitializationScheme, List[List[JumpInitializerBase]]
        ] = {}

        for scheme in self.cfg.num_scheme_curriculums.keys():
            self._initializers[InitializationScheme[scheme]] = [
                [
                    make_initializer(
                        InitializationScheme[scheme],
                        scheme_curriculum,
                        command_curriculum,
                        self._robot,
                    )
                    for command_curriculum in range(
                        self.cfg.command_curriculum_limits[scheme][0],
                        self.cfg.command_curriculum_limits[scheme][1],
                    )
                ]
                for scheme_curriculum in range(self.cfg.num_scheme_curriculums[scheme])
            ]

        start_idx = 0
        schemes = list(self.cfg.scheme_fraqs.keys())
        for scheme in schemes:
            fraction = self.cfg.scheme_fraqs[scheme]
            num_scheme_curriculums = self.cfg.num_scheme_curriculums[scheme]
            num_command_curriculums = self.cfg.num_command_curriculums[scheme]
            end_idx = (
                start_idx + int(fraction * self.num_envs)
                if scheme != schemes[-1]
                else self.num_envs
            )
            self._init_schemes[start_idx:end_idx] = InitializationScheme[scheme]
            self._num_scheme_curriculums[start_idx:end_idx] = num_scheme_curriculums
            self._min_command_curriculum[start_idx:end_idx] = self.cfg.command_curriculum_limits[scheme][0]
            self._max_command_curriculum[start_idx:end_idx] = self.cfg.command_curriculum_limits[scheme][1]
            self._scheme_count[InitializationScheme[scheme]] = end_idx - start_idx
            start_idx = end_idx

        assert start_idx == self.num_envs

        self._command_curriculum_level[:] = self._min_command_curriculum

    def _setup_domain_randomization(self):
        """Setup domain randomization for the robot."""
        pass

    def _setup_scene(self):
        """Setup the simulation scene."""
        print("[INFO] Setting up scene...")
        self.scene.clone_environments(copy_from_source=False)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self._robot = MarsJumperRobot(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
                
    def _pre_physics_step(self, actions: torch.Tensor):
        self.verify_robot_data()
        self._actions[:] = actions
        self._scaled_and_shifted_actions[:] = (
            self.cfg.action_scale * self._actions
            + self._robot.data.default_joint_pos[:, : self.cfg.action_space]
        )
        self._clipped_actions[:] = self._robot.clip_position_commands_to_joint_limits(
            self._scaled_and_shifted_actions,
            self.hip_abduction_joints_idx,
            self.knee_joints_idx,
            self.hip_flexion_joints_idx,
        )

        assert torch.isfinite(self._actions).all(), "NaN/Inf in actions"
        assert torch.isfinite(self._scaled_and_shifted_actions).all(), "NaN/Inf in scaled_and_shifted_actions"
        assert torch.isfinite(self._clipped_actions).all(), "NaN/Inf in clipped_actions"

    def _apply_action(self):
        self.verify_robot_data()
        self._robot.set_joint_position_target(
            self._clipped_actions, self.joints_idx #TODO: make sure its consistnt
        )

    def _get_observations(self) -> dict:
        self.verify_robot_data()
        self._previous_actions[:] = self._actions
        self._previous_torques[:] = self._robot.data.applied_torque[
            :, : self.cfg.action_space
        ]
        
        assert torch.isfinite(self._previous_actions).all(), "NaN/Inf in previous_actions"
        assert torch.isfinite(self._previous_torques).all(), "NaN/Inf in previous_torques"
        
        if not torch.isfinite(self._robot.data.joint_vel).all():
            nan_inf_indices = torch.nonzero(~torch.isfinite(self._robot.data.joint_vel))
            raise ValueError(f"NaN/Inf in joint_vel at indices: {nan_inf_indices}, in total {nan_inf_indices.numel()} NaN/Inf values, out of {self._robot.data.joint_vel.numel()} total values")
        
        assert torch.isfinite(self._robot.data.joint_vel).all(), "NaN/Inf in joint_vel"
        assert torch.isfinite(self._robot.data.default_joint_pos).all(), "NaN/Inf in default_joint_pos"

        obs = torch.cat(
            [
                self._robot.data.joint_pos[:, : self.cfg.action_space] 
                - self._robot.data.default_joint_pos[:, : self.cfg.action_space],
                self._robot.data.joint_vel[:, : self.cfg.action_space],
            ],
            dim=-1,
        )
        
        assert torch.isfinite(obs).all(), "NaN/Inf in observations"
        
        if self.cfg.observation_space != obs.shape[1]:
            raise ValueError(f"obs.shape[1] != self.cfg.observation_space, {obs.shape[1]} != {self.cfg.observation_space}")
        
        return {"policy": obs}        

    def _get_rewards(self) -> torch.Tensor:
        self.verify_robot_data()
        goal_pos_error_squared = self.goal_vec.square().sum(dim=1)

        goal_pos_error_large_reward = torch.exp(-goal_pos_error_squared / (1.5**2))
        goal_pos_error_small_reward = torch.exp(
            -(
                self.goal_vec.square()
                / (torch.tensor([[0.5**2, 0.5**2, 1**2]], device=self.device))
            ).sum(dim=1)
        )
        goal_pos_error_terminate = torch.where(
            self._terminate_touchdown,
            torch.exp(-goal_pos_error_squared / (2.0**2)),
            0.0,
        )

        orientation_reward = torch.where(
            self.speed_on_goal > 0.5,
            torch.exp(-self.rot_error_rad.square() / ((30 * torch.pi / 180) ** 2)),
            0.0,
        )

        speed_on_goal_reward = torch.where(
            self.airborne
            * (self._robot.data.root_pos_w[:, 2] > 0.5)
            * (self.goal_vec[:, :2].norm(dim=1) > 0.3),
            self.speed_on_goal.clamp(min=0.0).square()
            * torch.exp(-self.rot_error_rad.square() / ((30 * torch.pi / 180) ** 2)),
            0.0,
        )

        angvel = self._robot.data.root_ang_vel_w.square().sum(dim=1)
        angvel_reward = torch.exp(-angvel / (10.0**2))

        motor_torque = self._robot.data.applied_torque[:, : self.cfg.action_space]
        joint_vel = self._robot.data.joint_vel[:, : self.cfg.action_space]
        break_torque_reward = (
            torch.exp((motor_torque * joint_vel).clamp(max=0).mean(dim=1) / 20.0) - 1
        )

        start_vec = self._robot.data.root_pos_w - self._terrain.env_origins
        paw_heights = self._robot.data.body_pos_w[:, self.feet_idx, 2]

        takeoff_paws_rewards = torch.where(
            start_vec[:, :2].norm(dim=1) < 0.3,
            torch.exp(-paw_heights.square().mean(dim=-1) / (0.2**2)),
            0.0,
        )

        landstand_paws_reward = (
            torch.exp(-paw_heights.square().mean(dim=-1) / (0.15**2))
            * goal_pos_error_small_reward
        )
        landstand_joint_vel = (
            torch.exp(
                -self._robot.data.joint_vel[:, : self.cfg.action_space]
                .square()
                .mean(dim=1)
                / (2**2)
            )
            * goal_pos_error_small_reward
        )
        landstand_lin_vel = (
            torch.exp(-self._robot.data.root_lin_vel_w.square().sum(dim=1) / 0.5**2)
            * goal_pos_error_small_reward
        )

        landstand_orientation = torch.exp(
            -self.rot_error_rad.square()
            / ((20 * torch.pi / 180) ** 2)
            * goal_pos_error_small_reward
        )

        landstand_joint_pos = (
            torch.exp(
                -(
                    self._robot.data.joint_pos[:, : self.cfg.action_space]
                    - self._robot.data.default_joint_pos[:, : self.cfg.action_space]
                )
                .square()
                .mean(dim=1)
                / 0.5**2
            )
            * goal_pos_error_small_reward
        )

        landstand_joint_acc = (
            -0.0
            + torch.exp(
                -(self._robot.data.joint_acc[:, : self.cfg.action_space])
                .square()
                .mean(dim=1)
                / 8**2
            )
        ) * goal_pos_error_small_reward

        rewards = {
            "goal_pos_error_large": goal_pos_error_large_reward
            * self.cfg.goal_pos_error_large_reward_scale
            * self.step_dt,
            "goal_pos_error_small": goal_pos_error_small_reward
            * self.cfg.goal_pos_error_small_reward_scale
            * self.step_dt,
            "terminate_goal_pos_error": goal_pos_error_terminate * 0,
            "orientation_error": orientation_reward
            * self.cfg.attitude_error_reward_scale
            * self.step_dt,
            "speed_on_goal": speed_on_goal_reward
            * self.cfg.speed_on_goal_reward_scale
            * self.step_dt,
            "angvel": angvel_reward * self.cfg.angvel_reward_scale * self.step_dt,
            "takeoff_paws": takeoff_paws_rewards
            * self.cfg.takeoff_paws_pos_reward_scale
            * self.step_dt,
            "break_torque": break_torque_reward
            * self.cfg.break_torque_reward_scale
            * self.step_dt,
            "landstand_paws": landstand_paws_reward
            * self.cfg.landstand_paws_reward_scale
            * self.step_dt,
            "landstand_lin_vel": landstand_lin_vel
            * self.cfg.landstand_lin_vel_reward_scale
            * self.step_dt,
            "landstand_joint_vel": landstand_joint_vel
            * self.cfg.landstand_joint_vel_reward_scale
            * self.step_dt,
            "landstand_orientation": landstand_orientation
            * self.cfg.landstand_orientation_reward_scale
            * self.step_dt,
            "landstand_joint_pos": landstand_joint_pos
            * self.cfg.landstand_joint_pos_reward_scale
            * self.step_dt,
            "landstand_joint_acc": landstand_joint_acc
            * self.cfg.landstand_joint_acc_reward_scale
            * self.step_dt,
        }

        rewards.update(self._calculate_regularization_rewards())

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        assert torch.isfinite(reward).all(), "NaN/Inf in reward"

        # Logging
        for key, value in rewards.items():
            self._episode_reward_sums[key] += value

        metrics = dict()
        if self.reset_time_outs.any():
            metrics["terminal_goal_pos_error"] = (
                goal_pos_error_squared[self.reset_time_outs].sqrt().mean().item()
            )
        metrics["num_airborne"] = self.airborne.count_nonzero().item()
        metrics["num_airborne_standing"] = (
            self.airborne[self._init_schemes == InitializationScheme.STANDING]
            .count_nonzero()
            .item()
        )

        self.extras.update({f"metrics/{key}": value for key, value in metrics.items()})

        # bookkeeping
        close_to_goal = goal_pos_error_squared < 0.25**2
        self._close_to_goal_count[close_to_goal] += 1
        self._close_to_goal_count[~close_to_goal] = 0

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine if the episode is done based on various conditions."""
        self.verify_robot_data()
        self._episode_end[:] = self.episode_length_buf >= self.max_episode_length - 1
        self._terminate_on_goal[:] = self._close_to_goal_count >= 2.0 / self.step_dt

        time_out = self._episode_end | self._terminate_on_goal

        self._terminate_collision[:] = self.collision_state

        self._terminate_touchdown[:] = self.touchdown * (
            (self.goal_vec[:, :2].norm(dim=1) > 0.5)
            | (self._robot.data.root_pos_w[:, 2] < 0.35)
        )

        self._terminate_walking[:] = (
            self.in_contact
            * (self.start_vec[:, :2].norm(dim=-1) > 0.3)
            * (self.goal_vec[:, :2].norm(dim=-1) > 0.5)
        )

        self._terminate_idle[:] = (self.start_vec.norm(dim=1) < 0.5) * (
            self.episode_length_buf > 1 / self.step_dt
        )

        died = (
            self._terminate_collision
            | (self._terminate_walking)
            | self._terminate_touchdown
            | (self._robot.data.root_pos_w[:, 2] < 0.2)
            | self._terminate_idle
        )

        assert torch.isfinite(died).all(), "NaN/Inf in died"
        assert torch.isfinite(time_out).all(), "NaN/Inf in time_out"

        # curriculums
        within_curriculum_thresh = (
            self.goal_vec.norm(dim=1) < self.cfg.curriculum_threshold
        )
        self._curriculum_progress[within_curriculum_thresh * time_out] += 1
        self._curriculum_progress[(~within_curriculum_thresh) * time_out] = 0
        self._curriculum_progress[died] = 0

        (
            self._scheme_curriculum_level[time_out],
            self._command_curriculum_level[time_out],
            self._curriculum_progress[time_out],
            new_game_won,
        ) = get_next_curriculum(
            self._scheme_curriculum_level[time_out],
            self._command_curriculum_level[time_out],
            self._curriculum_progress[time_out],
            self._num_scheme_curriculums[time_out],
            self._max_command_curriculum[time_out],
            self.cfg.num_games_per_level,
        )

        timeout_idx = time_out.nonzero(as_tuple=True)[0]

        self._game_won[timeout_idx] |= new_game_won

        random_curriculum_mask = time_out * self._game_won
        num_random_curriculum = int(random_curriculum_mask.count_nonzero().item())

        if num_random_curriculum > 0:

            self._command_curriculum_level[random_curriculum_mask] = sample_uniform(
                self._min_command_curriculum[random_curriculum_mask],
                self._max_command_curriculum[random_curriculum_mask],
                num_random_curriculum,
                self.device,
            ).int()

            self._scheme_curriculum_level[random_curriculum_mask] = sample_uniform(
                0,
                self._num_scheme_curriculums[random_curriculum_mask],
                num_random_curriculum,
                self.device,
            ).int()

        # Logging
        curriculums = dict()
        for scheme in list(self.cfg.scheme_fraqs.keys()):
            mask = self._init_schemes == InitializationScheme[scheme]
            curriculums[f"{scheme}/mean_scheme_curriculum"] = (
                self._scheme_curriculum_level[mask].float().mean().item()
            )
            curriculums[f"{scheme}/mean_command_curriculum"] = (
                self._command_curriculum_level[mask].float().mean().item()
            )
            curriculums[f"{scheme}/game_won_fraq"] = (
                self._game_won[mask].float().mean().item()
            )

        self.extras.update(curriculums)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the environment for the given indices."""
        self.verify_robot_data()
        num_resets = len(env_ids)
        if env_ids is None or num_resets == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length)
            )

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._close_to_goal_count[env_ids] = 0
        self._idle_count[env_ids] = 0
        self._steps_since_takeoff.data[env_ids] = 0

        # Reset robot state and command
        state = torch.zeros(
            num_resets, 13 + 2 * self._robot.num_joints, device=self.device
        )
        command = torch.zeros(num_resets, 3, device=self.device)
        air_borne = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        for scheme, initializers in self._initializers.items():
            scheme_mask = self._init_schemes[env_ids] == scheme
            for sc in range(self.cfg.num_scheme_curriculums[scheme.name]):
                sc_mask = self._scheme_curriculum_level[env_ids] == sc
                for cc in range(self.cfg.command_curriculum_limits[scheme.name][1] - self.cfg.command_curriculum_limits[scheme.name][0]):
                    cc_mask = self._command_curriculum_level[env_ids] == (cc + self.cfg.command_curriculum_limits[scheme.name][0])
                    mask = scheme_mask * sc_mask * cc_mask
                    count = torch.count_nonzero(mask).item()
                    if count > 0:
                        command[mask], state[mask] = initializers[sc][cc].get_initial_states(count)

            if scheme in [
                InitializationScheme.TOUCHDOWN,
                InitializationScheme.INFLIGHT,
            ]:
                air_borne[scheme_mask] = True

        command += self._terrain.env_origins[env_ids]
        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]

        assert torch.isfinite(command).all(), "NaN/Inf in command"
        assert torch.isfinite(state).all(), "NaN/Inf in state"
        assert torch.isfinite(root_pose).all(), "NaN/Inf in root_pose"
        assert torch.isfinite(joint_pos).all(), "NaN/Inf in joint_pos"
        assert torch.isfinite(root_vel).all(), "NaN/Inf in root_vel"
        assert torch.isfinite(joint_vel).all(), "NaN/Inf in joint_vel"

        # update buffers
        self._airborne.data[env_ids] = air_borne
        self._goal_vec.data[env_ids] = command - root_pose[:, :3]
        self._start_vec.data[env_ids] = (
            self._terrain.env_origins[env_ids] - root_pose[:, :3]
        )
        self._takeoff_pos.data[env_ids[air_borne]] = self._terrain.env_origins[
            env_ids[air_borne]
        ] + torch.tensor([[0, 0, 0.45]], device=self.device)
        self._touchdown_pos.data[env_ids[~air_borne]] = root_pose[~air_borne, :3]

        self._command_position[env_ids] = command
        self._scaled_and_shifted_actions[env_ids] = joint_pos[:, : self.cfg.action_space]
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_reward_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_reward_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_reward_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/collision"] = torch.count_nonzero(
            self._terminate_collision[env_ids]
        ).item()
        extras["Episode_Termination/episode_end"] = torch.count_nonzero(
            self._episode_end[env_ids]
        ).item()
        extras["Episode_Termination/on_goal"] = torch.count_nonzero(
            self._terminate_on_goal[env_ids]
        ).item()
        extras["Episode_Termination/walking"] = torch.count_nonzero(
            self._terminate_walking[env_ids]
        ).item()
        extras["Episode_Termination/touchdown"] = torch.count_nonzero(
            self._terminate_touchdown[env_ids]
        ).item()
        extras["Episode_Termination/idle"] = torch.count_nonzero(
            self._terminate_idle[env_ids]
        ).item()
        extras["Episode_Termination/takeoff"] = torch.count_nonzero(
            self._terminate_takeoff[env_ids]
        ).item()
        extras["Episode_Termination/landed"] = torch.count_nonzero(
            self._terminate_landed[env_ids]
        ).item()
        self.extras["log"].update(extras)
        
        
    def validate_tensor(self, tensor: torch.Tensor, name: str):
        non_finite = ~torch.isfinite(tensor)
        if non_finite.any():
            bad_indices = torch.nonzero(non_finite)
            msg = f"\n NaN/Inf in {name}: {non_finite.sum()}/{tensor.numel()} values at indices {bad_indices.tolist()} \n"
            for idx in bad_indices[:10]:  # Only show first 10 bad values
                msg += f"Value at indices {idx.tolist()}: {tensor[tuple(idx)]}\n"
            assert False, msg
    
    def verify_robot_data(self):
        self.validate_tensor(self._robot.data.root_state_w, "root_state_w")
        self.validate_tensor(self._robot.data.root_link_state_w, "root_link_state_w")
        self.validate_tensor(self._robot.data.root_com_state_w, "root_com_state_w")
        self.validate_tensor(self._robot.data.body_state_w, "body_state_w")
        self.validate_tensor(self._robot.data.body_link_state_w, "body_link_state_w")
        self.validate_tensor(self._robot.data.body_com_state_w, "body_com_state_w")
        self.validate_tensor(self._robot.data.body_acc_w, "body_acc_w")
        self.validate_tensor(self._robot.data.joint_pos, "joint_pos")
        self.validate_tensor(self._robot.data.joint_vel, "joint_vel")
        self.validate_tensor(self._robot.data.joint_acc, "joint_acc")
        
    def _calculate_regularization_rewards(self) -> dict[str, Tensor]:
        self.verify_robot_data()
        """Calculate regularization rewards to encourage smooth and stable behavior."""
        joint_torques = (
            self._robot.data.applied_torque[:, : self.cfg.action_space]
            .square()
            .sum(dim=1)
        )
        jerk = (
            torch.isclose(
                self._robot.data.applied_torque[:, : self.cfg.action_space].sgn()
                * self._previous_torques.sgn(),
                torch.tensor(-1.0, device=self.device)
                .view(1, 1)
                .expand_as(self._previous_torques),
            )
            .float()
            .sum(dim=1)
        )

        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )

        action_clip = (
            (self._scaled_and_shifted_actions - self._clipped_actions).square().sum(dim=1)
        )

        contact_state = torch.any(
            torch.abs(
                self._contact_sensor.data.net_forces_w_history[:, 0, self.feet_idx]
            )
            > 0.1,
            dim=1,
        ).float()

        prev_contact_state = torch.any(
            torch.abs(
                self._contact_sensor.data.net_forces_w_history[:, 1, self.feet_idx]
            )
            > 0.1,
            dim=1,
        ).float()

        contact_change = (contact_state - prev_contact_state).abs().sum(dim=1)
        rewards = {
            "action_clip": action_clip
            * self.cfg.action_clip_reward_scale
            * self.step_dt,
            "dof_torques": joint_torques
            * self.cfg.joint_torque_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            "contact_change": contact_change
            * self.cfg.contact_change_reward_scale
            * self.step_dt,
            "jerk": jerk * self.cfg.jerk_reward_scale * self.step_dt,
        }

        assert (~torch.isfinite(joint_torques)).sum() == 0, f"NaN/Inf in joint_torques: {(~torch.isfinite(joint_torques)).sum()}/{joint_torques.numel()} values"
        assert (~torch.isfinite(jerk)).sum() == 0, f"NaN/Inf in jerk: {(~torch.isfinite(jerk)).sum()}/{jerk.numel()} values"
        assert (~torch.isfinite(action_rate)).sum() == 0, f"NaN/Inf in action_rate: {(~torch.isfinite(action_rate)).sum()}/{action_rate.numel()} values"
        assert (~torch.isfinite(action_clip)).sum() == 0, f"NaN/Inf in action_clip: {(~torch.isfinite(action_clip)).sum()}/{action_clip.numel()} values"
        assert (~torch.isfinite(contact_state)).sum() == 0, f"NaN/Inf in contact_state: {(~torch.isfinite(contact_state)).sum()}/{contact_state.numel()} values"
        assert (~torch.isfinite(prev_contact_state)).sum() == 0, f"NaN/Inf in prev_contact_state: {(~torch.isfinite(prev_contact_state)).sum()}/{prev_contact_state.numel()} values"
        assert (~torch.isfinite(contact_change)).sum() == 0, f"NaN/Inf in contact_change: {(~torch.isfinite(contact_change)).sum()}/{contact_change.numel()} values"


        return rewards

    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split the state tensor into root pose, joint positions, root velocity, and joint velocities."""
        return torch.split(
            state, [7, self._robot.num_joints, 6, self._robot.num_joints], dim=1
        )

    @property
    def goal_vec(self) -> torch.Tensor:
        if self._goal_vec.timestamp < self.common_step_counter:
            self._goal_vec.data[:] = self._command_position - self._robot.data.root_pos_w
            self._goal_vec.timestamp = self.common_step_counter
        assert torch.isfinite(self._goal_vec.data).all(), "NaN/Inf in goal_vec"
        return self._goal_vec.data

    @property
    def start_vec(self) -> torch.Tensor:
        if self._start_vec.timestamp < self.common_step_counter:
            self._start_vec.data[:] = (
                self._terrain.env_origins - self._robot.data.root_pos_w
            )
            self._start_vec.timestamp = self.common_step_counter
        assert torch.isfinite(self._start_vec.data).all(), "NaN/Inf in start_vec"
        return self._start_vec.data

    @property
    def speed_on_goal(self) -> torch.Tensor:
        if self._speed_on_goal.timestamp < self.common_step_counter:
            self._speed_on_goal.data[:] = (
                self._robot.data.root_lin_vel_w[:, :2] * normalize(self.goal_vec[:, :2])
            ).sum(dim=1)
            self._speed_on_goal.timestamp = self.common_step_counter
        assert torch.isfinite(self._speed_on_goal.data).all(), "NaN/Inf in speed_on_goal"
        return self._speed_on_goal.data

    @property
    def rot_error_rad(self) -> torch.Tensor:
        if self._rot_error_rad.timestamp < self.common_step_counter:
            self._rot_error_rad.data[:] = quat_error_magnitude(
                self._robot.data.root_quat_w,
                self._robot.data.default_root_state[:, 3:7],
            )
            self._rot_error_rad.timestamp = self.common_step_counter
        assert torch.isfinite(self._rot_error_rad.data).all(), "NaN/Inf in rot_error_rad"
        return self._rot_error_rad.data

    @property
    def contact_state(self) -> torch.Tensor:
        if self._contact_states.timestamp < self.common_step_counter:
            self._contact_states.data[:] = (
                self._contact_sensor.data.net_forces_w_history[
                    :, 0, self.feet_idx
                ].norm(dim=-1)
                > 0.1
            )
            self._contact_states.timestamp = self.common_step_counter
        assert torch.isfinite(self._contact_states.data).all(), "NaN/Inf in contact_states"
        return self._contact_states.data

    @property
    def in_contact(self) -> torch.Tensor:
        return torch.any(self.contact_state, dim=1)

    @property
    def collision_state(self) -> torch.Tensor:
        if self._collision_states.timestamp < self.common_step_counter:
            self._collision_states.data[:] = torch.any(
                torch.max(
                    torch.norm(
                        self._contact_sensor.data.net_forces_w_history[
                            :, :, self.avoid_contact_body_idx
                        ],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
            self._collision_states.timestamp = self.common_step_counter
        assert torch.isfinite(self._collision_states.data).all(), "NaN/Inf in collision_states"
        return self._collision_states.data

    @property
    def airborne(self) -> torch.Tensor:
        if self._airborne.timestamp < self.common_step_counter:
            self._update_flight_states()
        assert torch.isfinite(self._airborne.data).all(), "NaN/Inf in airborne"
        return self._airborne.data

    @property
    def takeoff(self) -> torch.Tensor:
        if self._takeoff.timestamp < self.common_step_counter:
            self._update_flight_states()
        assert torch.isfinite(self._takeoff.data).all(), "NaN/Inf in takeoff"
        return self._takeoff.data

    @property
    def touchdown(self) -> torch.Tensor:
        if self._touchdown.timestamp < self.common_step_counter:
            self._update_flight_states()
        assert torch.isfinite(self._touchdown.data).all(), "NaN/Inf in touchdown"
        return self._touchdown.data

    def _update_flight_states(self) -> None:
        self._takeoff.data[:] = (
            (~self.in_contact)
            * (~self._airborne.data)
            * (self._robot.data.root_lin_vel_w[:, 2] > 0.5)
            * (self._robot.data.root_lin_vel_w[:, 0] > 0.5)
        )
        self._touchdown.data[:] = self._airborne.data * self.in_contact
        self._airborne.data[self._takeoff.data] = True
        self._airborne.data[self._touchdown.data] = False

        self._airborne.timestamp = self.common_step_counter
        self._takeoff.timestamp = self.common_step_counter
        self._touchdown.timestamp = self.common_step_counter

        assert torch.isfinite(self._airborne.data).all(), "NaN/Inf in airborne"
        assert torch.isfinite(self._takeoff.data).all(), "NaN/Inf in takeoff"
        assert torch.isfinite(self._touchdown.data).all(), "NaN/Inf in touchdown"
