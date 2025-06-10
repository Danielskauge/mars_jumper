#!/usr/bin/env python3

"""
Debug script for testing actuator performance and identifying input mismatches.
"""

import argparse
import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional

from isaaclab.app import AppLauncher

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="Debug script for validating actuator model performance.")
parser.add_argument("--trajectory", type=str, choices=["sine", "constant", "step"], default="constant", help="Type of target trajectory")
parser.add_argument("--amplitude", type=float, default=math.pi/8, help="Amplitude for sine/step trajectories in radians")
parser.add_argument("--frequency", type=float, default=1.0, help="Frequency for sine trajectory in Hz")
parser.add_argument("--constant-value", type=float, default=math.pi/8, help="Target value for constant trajectory in radians")
parser.add_argument("--step-time", type=float, default=2.5, help="Time of step change for step trajectory in seconds")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# must be after applaunch
from isaaclab.assets.articulation import Articulation, ArticulationCfg 
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, ActuatorBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

# Import debug actuator
from robot.actuators.gru_actuator_debug import GruActuatorDebugCfg

# --- CONFIGURATION CLASSES ---
@configclass
class Simple1DoFRobotCfg(ArticulationCfg):
    usd_path: str = ""
    actuator_cfg_instance: Optional[ActuatorBaseCfg] = None
    
    def __post_init__(self):
        self.prim_path = "/World/TestRobot"
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        )
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={".*joint1.*": 0.0},
            joint_vel={".*joint1.*": 0.0},
        )
        if self.actuator_cfg_instance is not None:
            self.actuators = {"single_joint": self.actuator_cfg_instance}
        self.soft_joint_pos_limit_factor = 1.0

# --- CONSTANTS ---
SIM_DT = 1 / 240
CONTROL_DECIMATION = 1
EPISODE_DURATION_S = 5.0
NUM_STEPS_PER_EPISODE = int(EPISODE_DURATION_S / (SIM_DT * CONTROL_DECIMATION))
ANGLE_OFFSET_RAD = math.radians(40.0)

# --- HELPER FUNCTIONS ---
def generate_targets(time_array, trajectory_type, **kwargs):
    if trajectory_type == "sine":
        amplitude = kwargs.get("amplitude", math.pi/8)
        frequency = kwargs.get("frequency", 1.0)
        phase_offset = kwargs.get("phase_offset", 0.0)
        pos_targets = amplitude * np.array([math.sin(2 * math.pi * frequency * t + phase_offset) for t in time_array])
        vel_targets = amplitude * 2 * math.pi * frequency * np.array([math.cos(2 * math.pi * frequency * t + phase_offset) for t in time_array])
    elif trajectory_type == "constant":
        constant_value = kwargs.get("constant_value", math.pi/8)
        pos_targets = np.full_like(time_array, constant_value)
        vel_targets = np.zeros_like(time_array)
    elif trajectory_type == "step":
        amplitude = kwargs.get("amplitude", math.pi/8)
        step_time = kwargs.get("step_time", 2.5)
        pos_targets = np.where(time_array < step_time, 0.0, amplitude)
        vel_targets = np.zeros_like(time_array)
    else:
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")
    
    return torch.tensor(pos_targets, dtype=torch.float32), torch.tensor(vel_targets, dtype=torch.float32)

def setup_common_elements():
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    cfg_light.func("/World/Light", cfg_light)

def spawn_robot_in_scene(robot_cfg_instance: ArticulationCfg) -> Articulation:
    robot = Articulation(robot_cfg_instance)
    return robot

def run_simulation_episode(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target_pos_trajectory: torch.Tensor,
    target_vel_trajectory: torch.Tensor,
    episode_label: str
):
    print(f"[INFO] Running debug episode: {episode_label}")
    sim_time = 0.0
    
    joint_name = robot.joint_names[0]
    joint_idx = robot.find_joints(joint_name)[0][0]

    # Reset robot to initial state
    root_state = robot.data.default_root_state.clone()
    joint_pos_init = robot.data.default_joint_pos.clone()
    joint_pos_init += ANGLE_OFFSET_RAD
    joint_vel_init = robot.data.default_joint_vel.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_position_to_sim(joint_pos_init)
    robot.write_joint_velocity_to_sim(joint_vel_init)
    robot.update(SIM_DT)

    for step in range(NUM_STEPS_PER_EPISODE):
        current_target_pos = target_pos_trajectory[step].item()
        current_target_vel = target_vel_trajectory[step].item()

        robot.set_joint_position_target(current_target_pos, joint_ids=[joint_idx])
        robot.set_joint_velocity_target(current_target_vel, joint_ids=[joint_idx])

        for _ in range(CONTROL_DECIMATION):
            robot.write_data_to_sim()
            sim.step(render=True)
            robot.update(SIM_DT)

        sim_time += SIM_DT * CONTROL_DECIMATION

        if simulation_app.is_running() is False:
            break

    print(f"[INFO] Finished debug episode: {episode_label}")

def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.0, 1.0, 0.5), target=(0.0, 0.0, 0.0))

    setup_common_elements()

    # File paths
    usd_file_path = os.path.join(os.getcwd(), "test_actuators/simple_1dof_inertia_0.0134.usd")
    network_file_path = os.path.join(os.getcwd(), "test_actuators/gru-global_debug_best_jit.pt")
    hydra_config_path = os.path.join(os.getcwd(), "test_actuators/training_config.yaml")
    normalization_stats_path = os.path.join(os.getcwd(), "test_actuators/normalization_stats.json")

    # Check files exist
    for file_path, name in [(usd_file_path, "USD"), (network_file_path, "Network"), 
                           (hydra_config_path, "Hydra config"), (normalization_stats_path, "Normalization stats")]:
        if not os.path.exists(file_path):
            print(f"[ERROR] {name} file not found: {file_path}")
            simulation_app.close()
            return

    print(f"[INFO] All required files found. Running debug analysis...")

    # Create debug GRU actuator config
    debug_nn_act_cfg = GruActuatorDebugCfg(
        joint_names_expr=[".*joint1.*"],
        network_file=network_file_path,
        hydra_config_path=hydra_config_path,
        normalization_stats_path=normalization_stats_path,
        effort_limit=3.0,
        velocity_limit=34.91,
        saturation_effort=3.0
    )
    
    robot_cfg_debug = Simple1DoFRobotCfg(
        usd_path=usd_file_path,
        actuator_cfg_instance=debug_nn_act_cfg,
    )

    # Generate target trajectory
    time_points = np.linspace(0, EPISODE_DURATION_S, NUM_STEPS_PER_EPISODE, endpoint=False)
    
    trajectory_kwargs = {
        "amplitude": args_cli.amplitude,
        "frequency": args_cli.frequency,
        "constant_value": args_cli.constant_value,
        "step_time": args_cli.step_time
    }
    
    target_pos_traj, target_vel_traj = generate_targets(
        time_points, 
        args_cli.trajectory, 
        **trajectory_kwargs
    )
    
    # Shift trajectory into network training domain
    target_pos_traj = target_pos_traj + ANGLE_OFFSET_RAD
    target_pos_traj = target_pos_traj.to(sim.device)
    target_vel_traj = target_vel_traj.to(sim.device)
    
    print(f"[INFO] Using {args_cli.trajectory} trajectory")

    # Run debug simulation
    robot_debug = spawn_robot_in_scene(robot_cfg_debug)
    sim.reset()
    
    # Set joint position limits
    joint_ids, _ = robot_debug.find_joints([".*joint1.*"])
    limits = torch.tensor([[[ANGLE_OFFSET_RAD, math.radians(150.0)]]], dtype=torch.float32, device=sim.device)
    robot_debug.write_joint_position_limit_to_sim(limits=limits, joint_ids=joint_ids)
    
    if simulation_app.is_running() and sim.has_gui():
        for _ in range(50):
            sim.render()

    # Check simulation frequency matches training frequency
    actuator_debug = next(iter(robot_debug.actuators.values()))
    training_freq = getattr(actuator_debug, "training_frequency_hz", None)
    sim_freq = 1.0 / (SIM_DT * CONTROL_DECIMATION)
    
    print(f"[INFO] Training frequency: {training_freq:.2f} Hz")
    print(f"[INFO] Simulation frequency: {sim_freq:.2f} Hz")
    
    if abs(sim_freq - training_freq) > 1e-6:
        print(f"[WARNING] Frequency mismatch detected!")
    else:
        print(f"[INFO] Frequencies match.")

    # Run the debug episode
    run_simulation_episode(sim, robot_debug, target_pos_traj, target_vel_traj, "debug")

    print("[INFO] Debug analysis complete. Check debug_gru_inputs_outputs.json for detailed logs.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close() 