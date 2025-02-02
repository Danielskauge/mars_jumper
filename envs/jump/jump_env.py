from __future__ import annotations
from typing import Tuple, List, Dict
from torch import Tensor

import torch
from torch.nn.functional import normalize

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.buffers import TimestampedBuffer
from omni.isaac.lab.assets import Articulation

from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.utils.math import (
    quat_rotate_inverse,
    quat_error_magnitude,
)
from omni.isaac.lab.sensors import ContactSensor

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
                "orientaion_error",
                "speed_on_goal",
                "angvel",
                "takeoff_paws",
                "break_torque",
                "landstand_paws",
                "landstand_lin_vel",
                "landstand_joint_vel",
                "landstand_orientation", 
                "landstand_joint_pos",
                "action_clip",
                "dof_torques",
                "dof_acc_l2",
                "action_rate_l2",
                "contact_change",
                "jerk",
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
            self._target_positions = torch.zeros(self.num_envs, 3)
            self._close_to_goal_counter = torch.zeros(self.num_envs, dtype=torch.int)
            self._idle_counter = torch.zeros(self.num_envs, dtype=torch.int)
            
            # Termination flags
            self._terminate_on_goal = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_touchdown = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_walking = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_collision = torch.zeros(self.num_envs, dtype=torch.bool)
            self._terminate_idle = torch.zeros(self.num_envs, dtype=torch.bool)
            self._episode_end = torch.zeros(self.num_envs, dtype=torch.bool)

            # Robot state tracking
            self._initialization_schemes = torch.zeros(self.num_envs, dtype=torch.int)
            self._is_airbornee = torch.zeros(self.num_envs, dtype=torch.bool)
            self._has_taken_off = torch.zeros(self.num_envs, dtype=torch.bool)

            # Timestamped buffers for caching computed values
            self._contact_states = TimestampedBuffer(
                torch.zeros(self.num_envs, 4, dtype=torch.bool)
            )
            self._collision_states = TimestampedBuffer(
                torch.zeros(self.num_envs, dtype=torch.bool)
            )
            self._airbornee = TimestampedBuffer(
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

            # Curriculum learning state
            self._num_command_curriculum_levels = torch.zeros(self.num_envs, dtype=torch.int)
            self._num_scheme_curriculum_levels = torch.zeros(self.num_envs, dtype=torch.int)
            self._scheme_curriculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._command_curriculum_level = torch.zeros(self.num_envs, dtype=torch.int)
            self._curriculum_progress = torch.zeros(self.num_envs, dtype=torch.float)

    def _init_indices(self):
        """Get specific body indices for the robot."""
        self._feet_idx = self._robot.feet_idx
        self._avoid_contact_body_idx = self._robot.avoid_contact_body_idx
        self._joint_idx = self._robot.joints_idx

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
                    for command_curriculum in range(self.cfg.num_command_curriculums[scheme])
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
            self._num_scheme_curriculum_levels[start_idx:end_idx] = num_scheme_curriculums
            self._num_command_curriculum_levels[start_idx:end_idx] = num_command_curriculums
            start_idx = end_idx

        assert start_idx == self.num_envs

    def _setup_domain_randomization(self):
        """Setup domain randomization for the robot."""
        pass

    def _setup_scene(self):
        """Setup the simulation scene."""
        print("[INFO] Setting up scene...")
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions[:] = actions
        self._scaled_and_shifted_actions[:] = (
            self.cfg.action_scale * self._actions
            + self._robot.data.default_joint_pos[:, :] 
        ) #TODO: Need to shift actions?
        
        self._clipped_actions[:] = self._robot.clip_position_commands_to_joint_limits(
            self._scaled_and_shifted_actions
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(
            self._clipped_actions, self._joint_idx
        )

    def _get_observations(self) -> dict:
        self._previous_actions[:] = self._actions
        self._previous_torque[:] = self._robot.data.applied_torque[
            :, : self.cfg.action_space
        ]

        goal_pos_error_b = quat_rotate_inverse(
            self._robot.data.root_quat_w, self._commands - self._robot.data.root_pos_w
        )

        obs = torch.cat(
            [
                goal_pos_error_b,
                self._robot.data.root_pos_w[:, [2]],
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_pos[:, : self.cfg.action_space]
                - self._robot.data.default_joint_pos[:, : self.cfg.action_space],
                self._robot.data.joint_vel[:, : self.cfg.action_space],
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        return observations

    def _get_rewards(self) -> torch.Tensor:
        goal_pos_error_squared = self.goal_vec.square().sum(dim=1)

        goal_pos_error_large_reward = torch.exp(-goal_pos_error_squared / (1.5**2))
        goal_pos_error_small_reward = torch.exp(-goal_pos_error_squared / (0.5**2))
        goal_pos_error_terminate = torch.where(
            self._terminate_touchdown,
            torch.exp(-goal_pos_error_squared / (2.0**2)),
            0.0,
        )

        oreientation_reward = torch.exp(
            -self.rot_error_rad.square() / ((30 * torch.pi / 180) ** 2)
        )

        speed_on_goal_reward = torch.where(
            self.airborn, self.speed_on_goal.clamp(min=0.0), 0.0
        )

        ang_vel = self._robot.data.root_ang_vel_w.square().sum(dim=1)
        ang_vel_reward = torch.exp(-ang_vel / (10.0**2))

        motor_torque = self._robot.data.applied_torque[:, : self.cfg.action_space]
        joint_vel = self._robot.data.joint_vel[:, : self.cfg.action_space]
        break_torque_reward = (
            torch.exp((motor_torque * joint_vel).clamp(max=0).mean(dim=1) / 20.0) - 1
        )

        start_vec = self._robot.data.root_pos_w - self._terrain.env_origins
        paw_heights = self._robot.data.body_pos_w[:, self._feet_ids, 2]

        takeoff_paws_rewards = torch.where(
            start_vec[:, :2].norm(dim=1) < 0.3,
            torch.exp(-paw_heights.square().mean(dim=-1) / (0.2**2)),
            0.0,
        )

        land_stand_paws_reward = (
            torch.exp(-paw_heights.square().mean(dim=-1) / (0.15**2))
            * goal_pos_error_small_reward
        )
        land_stand_joint_vel = (
            torch.exp(
                -self._robot.data.joint_vel[:, : self.cfg.action_space]
                .square()
                .mean(dim=1)
                / (2**2)
            )
            * goal_pos_error_small_reward
        )
        land_stand_lin_vel = (
            torch.exp(-self._robot.data.root_lin_vel_w.square().sum(dim=1) / 0.5**2)
            * goal_pos_error_small_reward
        )

        land_stand_orientation = torch.exp(
            -self.rot_error_rad.square()
            / ((20 * torch.pi / 180) ** 2)
            * goal_pos_error_small_reward
        )

        land_stand_joint_pos = (
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

        rewards = {
            "goal_pos_error_large": goal_pos_error_large_reward
            * self.cfg.goal_pos_error_large_reward_scale
            * self.step_dt,
            "goal_pos_error_small": goal_pos_error_small_reward
            * self.cfg.goal_pos_error_small_reward_scale
            * self.step_dt,
            "terminate_goal_pos_error": goal_pos_error_terminate * 0,
            "orientaion_error": oreientation_reward
            * self.cfg.attitude_error_reward_scale
            * self.step_dt,
            "speed_on_goal": speed_on_goal_reward
            * self.cfg.speed_on_goal_reward_scale
            * self.step_dt,
            "angvel": ang_vel_reward * self.cfg.angvel_reward_scale * self.step_dt,
            "takeoff_paws": takeoff_paws_rewards
            * self.cfg.takeoff_paw_pos_reward_scale
            * self.step_dt,
            "break_torque": break_torque_reward
            * self.cfg.break_torque_reward_scale
            * self.step_dt,
            "landstand_paws": land_stand_paws_reward
            * self.cfg.lanstand_paws_reward_scale
            * self.step_dt,
            "landstand_lin_vel": land_stand_lin_vel
            * self.cfg.landstand_lin_vel_reward_scale
            * self.step_dt,
            "landstand_joint_vel": land_stand_joint_vel
            * self.cfg.landstand_joint_vel_reward_scale
            * self.step_dt,
            "landstand_orientation": land_stand_orientation
            * self.cfg.landstand_orientation_reward_scale
            * self.step_dt,
            "landstand_joint_pos": land_stand_joint_pos
            * self.cfg.landstand_joint_pos_reward_scale
            * self.step_dt,
        }

        rewards.update(self._calculate_regularization_rewards())

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        metrics = dict()
        if self.reset_time_outs.any():
            metrics["terminal_goal_pos_error"] = (
                goal_pos_error_squared[self.reset_time_outs].sqrt().mean().item()
            )
        metrics["num_airborne"] = self.airborn.count_nonzero().item()
        metrics["num_airborne_standing"] = (
            self.airborn[self._init_schemes == InitializationScheme.STANDING]
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

        # curriculums
        within_curriculum_thresh = (
            self.goal_vec.norm(dim=1) < self.cfg.curriculum_threshold
        )
        self._curriculum_progress[within_curriculum_thresh * time_out] += 1
        self._curriculum_progress[(~within_curriculum_thresh) * time_out] = 0
        self._curriculum_progress[died] = 0

        (
            self._scheme_curiculum_level[time_out],
            self._command_curiculum_level[time_out],
            self._curriculum_progress[time_out],
        ) = get_next_curriculum(
            self._scheme_curiculum_level[time_out],
            self._command_curiculum_level[time_out],
            self._curriculum_progress[time_out],
            self._num_scheme_curriculums[time_out],
            self._num_command_curriculums[time_out],
            self.cfg.num_games_per_level,
        )

        # Logging
        curriculums = dict()
        for scheme in list(self.cfg.scheme_fraqs.keys()):
            mask = self._init_schemes == InitializationScheme[scheme]
            curriculums[f"{scheme}/mean_scheme_curriclum"] = (
                self._scheme_curiculum_level[mask].float().mean().item()
            )
            curriculums[f"{scheme}/mean_command_curriclum"] = (
                self._command_curiculum_level[mask].float().mean().item()
            )

        self.extras.update(curriculums)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset the environment for the given indices."""
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

        # Reset robot state and command
        state = torch.zeros(
            num_resets, 13 + 2 * self._robot.num_joints, device=self.device
        )
        command = torch.zeros(num_resets, 3, device=self.device)
        air_born = torch.zeros(num_resets, dtype=torch.bool, device=self.device)
        for scheme, initializers in self._initializers.items():
            scheme_mask = self._init_schemes[env_ids] == scheme
            for sc in range(self.cfg.num_scheme_curriculums[scheme.name]):
                sc_mask = self._scheme_curiculum_level[env_ids] == sc
                for cc in range(self.cfg.num_command_curriculums[scheme.name]):
                    cc_mask = self._command_curiculum_level[env_ids] == cc
                    mask = scheme_mask * sc_mask * cc_mask
                    count = torch.count_nonzero(mask).item()
                    if count > 0:
                        command[mask], state[mask] = initializers[sc][cc].draw(count)

            if scheme in [
                InitializationScheme.TOUCHDOWN,
                InitializationScheme.INFLIGHT,
            ]:
                air_born[mask] = True

        command += self._terrain.env_origins[env_ids]
        root_pose, joint_pos, root_vel, joint_vel = self._split_state(state)
        root_pose[:, :3] += self._terrain.env_origins[env_ids]

        # update buffers
        self._airborne.data[env_ids] = air_born
        self._goal_vec.data[env_ids] = command - root_pose[:, :3]
        self._start_vec.data[env_ids] = (
            self._terrain.env_origins[env_ids] - root_pose[:, :3]
        )

        self._commands[env_ids] = command
        self._filtered_action[env_ids] = joint_pos[:, : self.cfg.action_space]
        self._robot.write_root_pose_to_sim(root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(root_vel, env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
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
        self.extras["log"].update(extras)

    def _calculate_regularization_rewards(self) -> dict[str, Tensor]:
        """Calculate regularization rewards to encourage smooth and stable behavior."""
        joint_torques = (
            self._robot.data.applied_torque[:, : self.cfg.action_space]
            .square()
            .sum(dim=1)
        )

        joint_accel = torch.sum(
            torch.square(self._robot.data.joint_acc[:, : self.cfg.action_space]), dim=1
        )

        jerk = (
            torch.isclose(
                self._robot.data.applied_torque[:, : self.cfg.action_space].sgn()
                * self._previous_torque.sgn(),
                torch.tensor(-1.0, device=self.device)
                .view(1, 1)
                .expand_as(self._previous_torque),
            )
            .float()
            .sum(dim=1)
        )

        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )

        action_clip = (
            (self._processed_actions - self._filtered_action).square().sum(dim=1)
        )

        contact_state = torch.any(
            torch.abs(
                self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids]
            )
            > 0.1,
            dim=1,
        ).float()

        prev_contact_state = torch.any(
            torch.abs(
                self._contact_sensor.data.net_forces_w_history[:, 1, self._feet_ids]
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
            "dof_acc_l2": joint_accel
            * self.cfg.joint_accel_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            "contact_change": contact_change
            * self.cfg.contact_change_reward_scale
            * self.step_dt,
            "jerk": jerk * self.cfg.jerk_reward_scale * self.step_dt,
        }

        return rewards

    def _split_state(self, state: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split the state tensor into root pose, joint positions, root velocity, and joint velocities."""
        return torch.split(
            state, [7, self._robot.num_joints, 6, self._robot.num_joints], dim=1
        )

    @property
    def goal_vec(self) -> torch.Tensor:
        if self._goal_vec.timestamp < self.common_step_counter:
            self._goal_vec.data[:] = self._commands - self._robot.data.root_pos_w
            self._goal_vec.timestamp = self.common_step_counter
        return self._goal_vec.data

    @property
    def start_vec(self) -> torch.Tensor:
        if self._start_vec.timestamp < self.common_step_counter:
            self._start_vec.data[:] = (
                self._terrain.env_origins - self._robot.data.root_pos_w
            )
            self._start_vec.timestamp = self.common_step_counter
        return self._start_vec.data

    @property
    def speed_on_goal(self) -> torch.Tensor:
        if self._speed_on_goal.timestamp < self.common_step_counter:
            self._speed_on_goal.data[:] = (
                self._robot.data.root_lin_vel_w[:, :2] * normalize(self.goal_vec[:, :2])
            ).sum(dim=1)
            self._speed_on_goal.timestamp = self.common_step_counter
        return self._speed_on_goal.data

    @property
    def rot_error_rad(self) -> torch.Tensor:
        if self._rot_error_rad.timestamp < self.common_step_counter:
            self._rot_error_rad.data[:] = quat_error_magnitude(
                self._robot.data.root_quat_w,
                self._robot.data.default_root_state[:, 3:7],
            )
            self._rot_error_rad.timestamp = self.common_step_counter
        return self._rot_error_rad.data

    @property
    def contact_state(self) -> torch.Tensor:
        if self._contact_state.timestamp < self.common_step_counter:
            self._contact_state.data[:] = (
                self._contact_sensor.data.net_forces_w_history[
                    :, 0, self._feet_ids
                ].norm(dim=-1)
                > 0.1
            )
            self._contact_state.timestamp = self.common_step_counter
        return self._contact_state.data

    @property
    def in_contact(self) -> torch.Tensor:
        return torch.any(self.contact_state, dim=1)

    @property
    def collision_state(self) -> torch.Tensor:
        if self._collision_state.timestamp < self.common_step_counter:
            self._collision_state.data[:] = torch.any(
                torch.max(
                    torch.norm(
                        self._contact_sensor.data.net_forces_w_history[
                            :, :, self._avoid_contact_body_ids
                        ],
                        dim=-1,
                    ),
                    dim=1,
                )[0]
                > 1.0,
                dim=1,
            )
            self._collision_state.timestamp = self.common_step_counter
        return self._collision_state.data

    @property
    def airborn(self) -> torch.Tensor:
        if self._airborne.timestamp < self.common_step_counter:
            self._update_flight_states()
        return self._airborne.data

    @property
    def takeoff(self) -> torch.Tensor:
        if self._takeoff.timestamp < self.common_step_counter:
            self._update_flight_states()
        return self._takeoff.data

    @property
    def touchdown(self) -> torch.Tensor:
        if self._touchdown.timestamp < self.common_step_counter:
            self._update_flight_states()
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
