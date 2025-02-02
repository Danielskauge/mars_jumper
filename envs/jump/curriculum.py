from typing import Tuple

from omni.isaac.lab.assets.articulation import Articulation

from initialization import (
    DefaultInitializer,
    CommandCfg,
    FlightTrajectoryCfg,
    InitializationScheme,
    JumpInitializerBase,
)

# Default paw positions for the Mars Jumper robot
# These will need adjustment based on your robot's geometry
DEFAULT_PAW_POS_LIMITS = {
    "Paw_BL": (
        [-0.25, 0.20, 0.0],  # min x,y,z
        [-0.23, 0.22, 0.0],  # max x,y,z
    ),
    "Paw_BR": (
        [-0.25, -0.22, 0.0],
        [-0.23, -0.20, 0.0],
    ),
    "Paw_FL": (
        [0.23, 0.15, 0.0],
        [0.25, 0.17, 0.0],
    ),
    "Paw_FR": (
        [0.23, -0.17, 0.0],
        [0.25, -0.15, 0.0],
    ),
}

# Curriculum constants
MAX_NUM_COMMAND_CURRICULUMS = 4
MAX_NUM_DEFAULT_CURRICULUMS = 1
MAX_NUM_STANDING_CURRICULUMS = 4
MAX_NUM_INFLIGHT_CURRICULUMS = 4
MAX_NUM_TOUCHDOWN_CURRICULUMS = 4
MAX_NUM_LANDED_CURRICULUMS = 2

# Default flight trajectory configuration
DEFAULT_FLIGHT_TRAJECTORY_CFG = FlightTrajectoryCfg(
    takeoff_pos_limits=(
        [0.0, 0.0, 0.4],
        [0.1, 0.0, 0.45],
    ),
    takeoff_angle_limits=(40.0, 50.0),
    jump_length_limits=(
        [1.0, -1.0],
        [4.0, 1.0],
    ),
)

def make_initializer(
    scheme: InitializationScheme,
    scheme_curriculum: int,
    command_curriculum: int,
    mars_jumper: Articulation,
) -> JumpInitializerBase:
    """Create an initializer based on the scheme and curriculum level."""
    match scheme:
        case InitializationScheme.STANDING:
            return StandingInitializer(
                cfg=_get_standing_cfg(scheme_curriculum),
                robot_articulation=mars_jumper,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.INFLIGHT:
            return InflightInitializer(
                cfg=_get_inflight_cfg(scheme_curriculum),
                robot_articulation=mars_jumper,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.TOUCHDOWN:
            return TouchdownInitializer(
                cfg=_get_touchdown_cfg(scheme_curriculum),
                robot_articulation=mars_jumper,
                flight_trajectory_cfg=DEFAULT_FLIGHT_TRAJECTORY_CFG,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case InitializationScheme.LANDED:
            return LandedInitializer(
                cfg=_get_landed_cfg(scheme_curriculum),
                robot_articulation=mars_jumper,
                command_cfg=_get_command_cfg(command_curriculum),
            )
        case _:
            return DefaultInitializer(
                robot_articulation=mars_jumper,
                command_cfg=_get_command_cfg(command_curriculum),
            ) 

def _get_command_cfg(curriculum: int) -> CommandCfg:
    """Get command configuration for the given curriculum level."""
    match curriculum:
        case 0:
            return CommandCfg(
                landing_pos_limits=(
                    [1.0, -0.2, 0.0],
                    [1.5, 0.2, 0.0],
                ),
            )
        case 1:
            return CommandCfg(
                landing_pos_limits=(
                    [1.0, -0.5, 0.0],
                    [2.0, 0.5, 0.0],
                ),
            )
        case 2:
            return CommandCfg(
                landing_pos_limits=(
                    [1.0, -1.0, 0.0],
                    [3.0, 1.0, 0.0],
                ),
            )
        case 3:
            return CommandCfg(
                landing_pos_limits=(
                    [1.0, -1.0, 0.0],
                    [4.0, 1.0, 0.0],
                ),
            )
        case _:
            raise ValueError(f"Invalid curriculum level: {curriculum}")

# def _get_standing_cfg(curriculum: int) -> StandingInitializerCfg:
#     """Get standing initialization config for the given curriculum level."""
#     match curriculum:
#         case 0:
#             return StandingInitializerCfg(
#                 base_pos_limits=([-0.0, 0.0, 0.4], [0.0, 0.0, 0.5]),
#                 base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case 1:
#             return StandingInitializerCfg(
#                 base_pos_limits=([-0.1, -0.1, 0.4], [0.1, 0.1, 0.5]),
#                 base_euler_limits=([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case 2:
#             return StandingInitializerCfg(
#                 base_pos_limits=([-0.2, -0.2, 0.4], [0.2, 0.2, 0.5]),
#                 base_euler_limits=([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case 3:
#             return StandingInitializerCfg(
#                 base_pos_limits=([-0.3, -0.3, 0.4], [0.3, 0.3, 0.5]),
#                 base_euler_limits=([-15.0, -15.0, -15.0], [15.0, 15.0, 15.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case _:
#             raise ValueError(f"Invalid curriculum level: {curriculum}")

# def _get_inflight_cfg(curriculum: int) -> InflightInitializerCfg:
#     """Get in-flight initialization config for the given curriculum level."""
#     match curriculum:
#         case 0:
#             return InflightInitializerCfg(
#                 lateral_joints_limits=(0, 80),
#                 transversal_joints_limits=(0, 120),
#                 base_euler_limits=([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
#                 base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
#                 normalized_time_limits=(0.01, 0.1),
#             )
#         case 1:
#             return InflightInitializerCfg(
#                 lateral_joints_limits=(0, 80),
#                 transversal_joints_limits=(0, 120),
#                 base_euler_limits=([-5.0, -10.0, -1.0], [5.0, 10.0, 1.0]),
#                 base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
#                 normalized_time_limits=(0.1, 0.2),
#             )
#         case 2:
#             return InflightInitializerCfg(
#                 lateral_joints_limits=(0, 80),
#                 transversal_joints_limits=(0, 120),
#                 base_euler_limits=([-5.0, -20.0, -5.0], [5.0, 20.0, 5.0]),
#                 base_ang_vel_limits=([-0.5, -0.5, -0.5], [0.5, 1.0, 0.5]),
#                 normalized_time_limits=(0.1, 0.2),
#             )
#         case 3:
#             return InflightInitializerCfg(
#                 lateral_joints_limits=(0, 80),
#                 transversal_joints_limits=(0, 120),
#                 base_euler_limits=([-20.0, -30.0, -20.0], [20.0, 30.0, 20.0]),
#                 base_ang_vel_limits=([-1.0, -2.0, -1.0], [1.0, 2.0, 1.0]),
#                 normalized_time_limits=(0.1, 0.3),
#             )
#         case _:
#             raise ValueError(f"Invalid curriculum level: {curriculum}")

# def _get_touchdown_cfg(curriculum: int) -> TouchdownInitializerCfg:
#     """Get touchdown initialization config for the given curriculum level."""
#     match curriculum:
#         case 0:
#             return TouchdownInitializerCfg(
#                 base_euler_limits=([-1.0, -5.0, -2.0], [1, 5.0, 2.0]),
#                 base_ang_vel_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
#                 lateral_joints_limits=(-5, 5),
#                 front_transversal_joints_limits=(90, 30),
#                 back_transversal_joints_limits=(-20, -30),
#                 time_to_land_limits=(0.1, 0.3),
#             )
#         case 1:
#             return TouchdownInitializerCfg(
#                 base_euler_limits=([-1.0, -10, -2.0], [1, 10.0, 2.0]),
#                 base_ang_vel_limits=([0.0, -1.0, 0.0], [0.0, 1.0, 0.0]),
#                 lateral_joints_limits=(-5, 5),
#                 front_transversal_joints_limits=(90, 30),
#                 back_transversal_joints_limits=(-20, -30),
#                 time_to_land_limits=(0.1, 0.3),
#             )
#         case 2:
#             return TouchdownInitializerCfg(
#                 base_euler_limits=([-1.0, -10.0, -2.0], [1, 10.0, 2.0]),
#                 base_ang_vel_limits=([-0.5, -1.0, -0.5], [0.5, 1.0, 0.5]),
#                 lateral_joints_limits=(-5, 5),
#                 front_transversal_joints_limits=(90, 30),
#                 back_transversal_joints_limits=(-20, -30),
#                 time_to_land_limits=(0.3, 0.5),
#             )
#         case 3:
#             return TouchdownInitializerCfg(
#                 base_euler_limits=([-10.0, -20.0, -10.0], [10, 20.0, 10.0]),
#                 base_ang_vel_limits=([-0.5, -1.0, -0.5], [0.5, 1.0, 0.5]),
#                 lateral_joints_limits=(-5, 5),
#                 front_transversal_joints_limits=(90, 30),
#                 back_transversal_joints_limits=(-20, -30),
#                 time_to_land_limits=(0.3, 0.5),
#             )
#         case _:
#             raise ValueError(f"Invalid curriculum level: {curriculum}")

# def _get_landed_cfg(curriculum: int) -> LandedInitializerCfg:
#     """Get landed initialization config for the given curriculum level."""
#     match curriculum:
#         case 0:
#             return LandedInitializerCfg(
#                 base_pos_limits=([-0.0, 0.0, 0.4], [0.0, 0.0, 0.5]),
#                 base_euler_limits=([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case 1:
#             return LandedInitializerCfg(
#                 base_pos_limits=([-0.20, 0.0, 0.4], [0.10, 0.0, 0.5]),
#                 base_euler_limits=([-1.0, -5.0, -1.0], [1.0, 5.0, 1.0]),
#                 paw_pos_limits=DEFAULT_PAW_POS_LIMITS,
#             )
#         case _:
#             raise ValueError(f"Invalid curriculum level: {curriculum}")

def get_next_curriculum(
    current_scheme_curriculum: int,
    current_command_curriculum: int,
    scheme: InitializationScheme,
    success_rate: float,
    curriculum_threshold: float,
) -> Tuple[int, int]:
    """Get the next curriculum level based on current performance."""
    next_scheme_curriculum = current_scheme_curriculum
    next_command_curriculum = current_command_curriculum

    if success_rate >= curriculum_threshold:
        if current_command_curriculum < MAX_NUM_COMMAND_CURRICULUMS - 1:
            next_command_curriculum = current_command_curriculum + 1
        elif current_scheme_curriculum < _get_max_scheme_curriculum(scheme) - 1:
            next_scheme_curriculum = current_scheme_curriculum + 1
            next_command_curriculum = 0

    return next_scheme_curriculum, next_command_curriculum

def _get_max_scheme_curriculum(scheme: InitializationScheme) -> int:
    """Get maximum curriculum level for a given scheme."""
    match scheme:
        case InitializationScheme.STANDING:
            return MAX_NUM_STANDING_CURRICULUMS
        case InitializationScheme.DEFAULT:
            return MAX_NUM_DEFAULT_CURRICULUMS
        case InitializationScheme.INFLIGHT:
            return MAX_NUM_INFLIGHT_CURRICULUMS
        case InitializationScheme.TOUCHDOWN:
            return MAX_NUM_TOUCHDOWN_CURRICULUMS
        case InitializationScheme.LANDED:
            return MAX_NUM_LANDED_CURRICULUMS
        case _:
            raise ValueError(f"Invalid scheme: {scheme}") 