from typing_extensions import override
import torch
from omni.isaac.lab.assets.articulation import Articulation
from utilities import Stack
from .configs import CommandCfg
from .initializer_base import JumpInitializerBase

class DefaultInitializer(JumpInitializerBase):
    """
    DefaultInitializer is responsible for setting up the initial state of the robot
    in the simulation environment using default configurations.

    Inherits from JumpInitializerBase to utilize common initialization functionalities.
    """

    def __init__(
        self,
        robot_articulation: Articulation,
        command_cfg: CommandCfg,
    ):
        """
        Initializes the DefaultInitializer with the given robot articulation and command configuration.

        Args:
            robot_articulation: The robot articulation containing environment data.
            command_cfg: Configuration for command generation.
        """
        super().__init__(robot_articulation, None, command_cfg)

    @override
    def _create_state_stack(self) -> Stack:
        """
        Creates a stack of initial states for the environments.

        Returns:
            A Stack object containing the initial states.
        """
        def generator():
            """
            Generates a batch of initial states by combining command, base state, joint positions,
            and velocities.

            Returns:
                A tensor containing the concatenated initial states for the environments.
            """
            command = self._sample_command()

            return torch.cat(
                (
                    command,
                    self._default_base_state[:7]
                    .unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_joint_pos.unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_base_state[7:]
                    .unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                    self._default_joint_vel.unsqueeze(0)
                    .expand(self._num_envs, -1)
                    .clone(),
                ),
                dim=1,
            )

        return Stack(generator)
