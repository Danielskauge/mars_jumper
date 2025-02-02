from typing import Tuple, Literal, List
from torch import Tensor
from abc import abstractmethod, ABC
from typing_extensions import override
import torch
from torch.nn.functional import normalize
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.utils.math import (
    quat_from_euler_xyz,
    quat_rotate,
    random_orientation,
    quat_from_angle_axis,
)
from utilities import uniform_sample, Stack
from .configs import FlightTrajectoryCfg, CommandCfg
from .flight_trajectory import FlightTrajectory

class InitializerBase(ABC):
    """
    Base class for initializing environments with a specific robot state.
    """

    def __init__(self, robot_articulation: Articulation):
        """
        Initializes the base state and configuration for the environments.

        Args:
            robot_articulation: The robot articulation containing environment data.
        """
        self._num_envs = robot_articulation.num_instances
        self._device = robot_articulation.device
        self._default_base_state = robot_articulation.data.default_root_state[0]
        self._default_joint_positions = robot_articulation.data.default_joint_pos[0]
        self._default_joint_velocities = robot_articulation.data.default_joint_vel[0]
        self._num_states = 13 + 2 * robot_articulation.num_joints
        self._state_stack: Stack = None

    @abstractmethod
    def _create_state_stack(self) -> Stack:
        """
        Abstract method to create a stack of initial states.
        """
        pass

    def _get_initial_states(self, num_envs: int) -> Tensor:
        """
        Draws a batch of initial states from the stack.

        Args:
            num_envs: Number of environments to draw states for.

        Returns:
            A tensor containing the initial states for the environments.
        """
        if self._state_stack is None:
            self._state_stack = self.create_state_stack()
        return self._state_stack.pop(num_envs)

    @staticmethod
    def generate_random_orientation(
        num_envs: int, device: str, euler_limits: Tuple[Tensor, Tensor] | None = None
    ) -> Tensor:
        """
        Generates random orientations for the given number of environments.

        Args:
            num_envs: Number of environments.
            device: Device to perform computations on.
            euler_limits: Optional limits for Euler angles.

        Returns:
            A tensor of random orientations.
        """
        if euler_limits is not None:
            euler_angles = torch.split(
                uniform_sample(*euler_limits, num_envs).deg2rad(), 1, dim=1
            )
            return normalize(quat_from_euler_xyz(*euler_angles).squeeze(1))

        phi = torch.rand(num_envs, device=device) * torch.pi
        theta = torch.rand(num_envs, device=device) * 2 * torch.pi
        sin_phi = torch.sin(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)

        axis = torch.stack((sin_phi * cos_theta, sin_phi * sin_theta, torch.cos(phi)), dim=1)
        angle = torch.rand(num_envs, device=device) * 2 * torch.pi - torch.pi

        return quat_from_angle_axis(angle, axis)

class JumpInitializerBase(InitializerBase):
    """
    Base class for jump initializers, extending InitializerBase with jump-specific configurations.
    """

    def __init__(
        self,
        robot_articulation: Articulation,
        flight_trajectory_cfg: FlightTrajectoryCfg,
        command_cfg: CommandCfg,
    ) -> None:
        """
        Initializes the jump initializer with flight trajectory and command configurations.

        Args:
            robot_articulation: The robot articulation.
            flight_trajectory_cfg: Configuration for flight trajectory.
            command_cfg: Configuration for command generation.
        """
        super().__init__(robot_articulation)
        self._flight_trajectory_cfg = flight_trajectory_cfg
        self._command_cfg = command_cfg
        self._command_dim = 3
        self._set_command_limits(command_cfg)
        self._set_flight_trajectory_limits(flight_trajectory_cfg)

    @override
    def get_initial_states(self, num_envs: int) -> Tuple[Tensor, Tensor]:
        """
        Draws a batch of commands and states from the stack.

        Args:
            num_envs: Number of environments to draw data for.

        Returns:
            A tuple containing commands and states for the environments.
        """
        if self._state_stack is None:
            self._state_stack = self.create_state_stack()
        return torch.split(
            self._state_stack.pop(num_envs), [self._command_dim, self._num_states], dim=1
        )

    def _convert_tuple_to_tensor(self, tuple_data: Tuple[List[float] | float]) -> Tuple[Tensor, Tensor]:
        """
        Converts a tuple of lists or floats to a tuple of tensors.

        Args:
            tuple_data: Tuple containing lists or floats.

        Returns:
            A tuple of tensors.
        """
        return tuple(torch.tensor(data, device=self._device) for data in tuple_data)

    def _set_command_limits(self, command_cfg: CommandCfg) -> None:
        """
        Sets the landing position limits from the command configuration.

        Args:
            command_cfg: Configuration for command generation.
        """
        self._landing_pos_limits = self._convert_tuple_to_tensor(command_cfg.landing_pos_limits)
        assert self._landing_pos_limits[0].shape[0] == self._command_dim
        assert self._landing_pos_limits[1].shape[0] == self._command_dim

    def _set_flight_trajectory_limits(
        self, flight_trajectory_cfg: FlightTrajectoryCfg
    ) -> None:
        """
        Sets the takeoff and jump limits from the flight trajectory configuration.

        Args:
            flight_trajectory_cfg: Configuration for flight trajectory.
        """
        if flight_trajectory_cfg is None:
            return
        self._takeoff_pos_limits = self._convert_tuple_to_tensor(
            flight_trajectory_cfg.takeoff_pos_limits
        )
        self._takeoff_angle_limits = self._convert_tuple_to_tensor(
            flight_trajectory_cfg.takeoff_angle_limits
        )
        self._jump_length_limits = self._convert_tuple_to_tensor(
            flight_trajectory_cfg.jump_length_limits
        )

    def _sample_command(self) -> Tensor:
        """
        Samples a random command within the landing position limits.

        Returns:
            A tensor containing the sampled command.
        """
        return uniform_sample(*self._landing_pos_limits, self._num_envs)

    def _sample_flight_trajectory(
        self,
        command: Tensor,
        clip_mode: Literal["landing_pos", "takeoff_pos"] = "landing_pos",
    ) -> FlightTrajectory:
        """
        Samples a flight trajectory based on the command and clip mode.

        Args:
            command: The command tensor.
            clip_mode: Mode to clip the trajectory, either 'landing_pos' or 'takeoff_pos'.

        Returns:
            A FlightTrajectory object representing the sampled trajectory.
        """
        if self._flight_trajectory_cfg is None:
            raise ValueError("Flight trajectory config is not provided")

        landing_position = command.clone()
        takeoff_position = uniform_sample(*self._takeoff_pos_limits, self._num_envs)
        takeoff_angle = uniform_sample(*self._takeoff_angle_limits, self._num_envs)

        commanded_jump_length = command - takeoff_position
        too_long = (
            commanded_jump_length[:, :2] - self._jump_length_limits[1].unsqueeze(0)
        ).clip(min=0)
        too_short = (
            commanded_jump_length[:, :2] - self._jump_length_limits[0].unsqueeze(0)
        ).clip(max=0)

        if clip_mode == "takeoff_pos":
            takeoff_position[:, :2] += too_long
            takeoff_position[:, :2] += too_short

        elif clip_mode == "landing_pos":
            landing_position[:, :2] -= too_long
            landing_position[:, :2] -= too_short
        else:
            raise ValueError("clip_mode must be either 'landing_pos' or 'takeoff_pos'")

        return FlightTrajectory(takeoff_position, landing_position, takeoff_angle)
