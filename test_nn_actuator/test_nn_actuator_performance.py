#!/usr/bin/env python3

"""
Script for testing actuator performance comparison between different actuator types.

Required files for this test:
1. test_nn_actuator/simple_1dof_robot.usd - USD file defining the robot
2. test_nn_actuator/gru-global_debug_best_jit.pt - TorchScript GRU model file  
3. test_nn_actuator/training_config.yaml - Hydra config file from training
4. test_nn_actuator/normalization_stats.json - Normalization statistics from training

The GRU actuator will determine its operating mode (residual vs non-residual) from the 
training_config.yaml file. In residual mode, PD parameters are loaded from the config.
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
parser = argparse.ArgumentParser(description="Test script for validating actuator model performance.")

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

# Assuming GruActuator and its Cfg are in this path
# You might need to adjust this import based on your project structure
from robot.actuators.gru_actuator import GruActuatorCfg

# --- CONFIGURATION CLASSES ---
"""
This robot will have a fixed base, a single revolute joint, and a link.
The link is a sphere with radius 1.0 m and mass 5.0 kg.
Its center of mass (CoM) is at the joint origin (0, 0, 0).

Principal moments of inertia (about CoM, which is also the joint origin):
I_xx = 2.0 kg*m^2
I_yy = 2.0 kg*m^2
I_zz = 2.0 kg*m^2

The joint will be controlled by a PD controller, and the link will have a simple sine wave target trajectory.

The robot will be tested with both the implicit PD actuator and the combined GRU and PD actuator.
"""

@configclass
class Simple1DoFRobotCfg(ArticulationCfg):
    # Make these dataclass fields instead of constructor arguments
    usd_path: str = ""
    actuator_cfg_instance: Optional[ActuatorBaseCfg] = None
    
    def __post_init__(self):
        """Post-initialization to set up the configuration."""
        # Don't call super().__init__() here - dataclass handles initialization
        self.prim_path = "/World/TestRobot"
        self.spawn = sim_utils.UsdFileCfg(
            usd_path=self.usd_path,
            activate_contact_sensors=False, # No contact sensors needed for this test
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        )
        self.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), # At ground level
            joint_pos={".*joint1.*": 0.0},
            joint_vel={".*joint1.*": 0.0},
        )
        if self.actuator_cfg_instance is not None:
            self.actuators = {"single_joint": self.actuator_cfg_instance}
        self.soft_joint_pos_limit_factor = 1.0


# --- CONSTANTS AND SIMULATION PARAMETERS ---
SIM_DT = 1 / 100  # Simulation physics timestep
CONTROL_DECIMATION = 1
EPISODE_DURATION_S = 5.0  # Duration of each test episode in seconds
NUM_STEPS_PER_EPISODE = int(EPISODE_DURATION_S / (SIM_DT * CONTROL_DECIMATION))

# Sine wave parameters for target trajectory
SINE_AMPLITUDE_POS = math.pi / 4  # rad
SINE_FREQUENCY_POS = 0.5  # Hz

# --- HELPER FUNCTIONS ---
def generate_sine_wave_targets(time_array, amplitude, frequency, phase_offset=0.0):
    """Generates position and velocity targets for a sine wave."""
    pos_targets = amplitude * np.array([math.sin(2 * math.pi * frequency * t + phase_offset) for t in time_array])
    vel_targets = amplitude * 2 * math.pi * frequency * np.array([math.cos(2 * math.pi * frequency * t + phase_offset) for t in time_array])
    return torch.tensor(pos_targets, dtype=torch.float32), torch.tensor(vel_targets, dtype=torch.float32)

def plot_comparison_data(time_log, data_dict, joint_name, filename_prefix="actuator_comparison"):
    """Plots comparison data for different actuators."""
    num_metrics = 3 # pos, vel, torque
    fig, axs = plt.subplots(num_metrics, 1, figsize=(15, num_metrics * 4), sharex=True)
    fig.suptitle(f"Actuator Performance Comparison for Joint: {joint_name}")

    plot_infos = [
        {"label": "Position", "unit": "rad", "target_key": "target_pos", "actual_keys": ["actual_pos_implicit", "actual_pos_nn"]},
        {"label": "Velocity", "unit": "rad/s", "target_key": "target_vel", "actual_keys": ["actual_vel_implicit", "actual_vel_nn"]},
        {"label": "Applied Torque", "unit": "Nm", "target_key": None, "actual_keys": ["torque_implicit", "torque_nn"]}
    ]

    colors = {
        "target": "black",
        "implicit": "tab:blue",
        "nn": "tab:red"
    }
    line_styles = {
        "target": "--",
        "implicit": "-",
        "nn": ":"
    }

    for i, p_info in enumerate(plot_infos):
        axs[i].set_ylabel(f"{p_info['label']} ({p_info['unit']})")
        axs[i].grid(True, linestyle=':', alpha=0.7)

        if p_info["target_key"] and p_info["target_key"] in data_dict:
            axs[i].plot(time_log, data_dict[p_info["target_key"]], label=f"Target {p_info['label']}", color=colors["target"], linestyle=line_styles["target"])

        if "actual_pos_implicit" in p_info["actual_keys"] and "actual_pos_implicit" in data_dict:
             axs[i].plot(time_log, data_dict["actual_pos_implicit"], label="Actual Pos (Implicit PD)", color=colors["implicit"], linestyle=line_styles["implicit"])
        if "actual_pos_nn" in p_info["actual_keys"] and "actual_pos_nn" in data_dict:
             axs[i].plot(time_log, data_dict["actual_pos_nn"], label="Actual Pos (NN)", color=colors["nn"], linestyle=line_styles["nn"])

        if "actual_vel_implicit" in p_info["actual_keys"] and "actual_vel_implicit" in data_dict:
             axs[i].plot(time_log, data_dict["actual_vel_implicit"], label="Actual Vel (Implicit PD)", color=colors["implicit"], linestyle=line_styles["implicit"])
        if "actual_vel_nn" in p_info["actual_keys"] and "actual_vel_nn" in data_dict:
             axs[i].plot(time_log, data_dict["actual_vel_nn"], label="Actual Vel (NN)", color=colors["nn"], linestyle=line_styles["nn"])

        if "torque_implicit" in p_info["actual_keys"] and "torque_implicit" in data_dict:
            axs[i].plot(time_log, data_dict["torque_implicit"], label="Torque (Implicit PD)", color=colors["implicit"], linestyle=line_styles["implicit"])
        if "torque_nn" in p_info["actual_keys"] and "torque_nn" in data_dict:
            axs[i].plot(time_log, data_dict["torque_nn"], label="Torque (NN)", color=colors["nn"], linestyle=line_styles["nn"])

        axs[i].legend(loc='upper right')

    axs[-1].set_xlabel("Time (s)")
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = f"plots/{filename_prefix}_{joint_name}.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(save_path)
    print(f"[INFO]: Plot saved to {save_path}")
    plt.close(fig)

# --- SCENE DESIGN AND SIMULATION LOGIC ---
def setup_common_elements():
    """Sets up common scene elements like ground plane and light."""
    # Ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    # Distant light
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    cfg_light.func("/World/Light", cfg_light)

def spawn_robot_in_scene(robot_cfg_instance: ArticulationCfg) -> Articulation:
    """Spawns the robot based on its configuration."""
    # Spawn robot
    robot = Articulation(robot_cfg_instance)
    return robot

def run_simulation_episode(
    sim: sim_utils.SimulationContext,
    robot: Articulation,
    target_pos_trajectory: torch.Tensor,
    target_vel_trajectory: torch.Tensor,
    episode_label: str # For logging and data storage
):
    """Runs one episode of simulation for the given robot and targets."""
    print(f"[INFO] Running episode: {episode_label}")
    sim_time = 0.0
    log_data = {
        f"actual_pos_{episode_label}": [],
        f"actual_vel_{episode_label}": [],
        f"torque_{episode_label}": [],
    }

    # Find the joint index (assuming only one joint for this robot)
    joint_name = robot.joint_names[0]
    joint_idx = robot.find_joints(joint_name)[0][0]

    # Reset robot to initial state
    root_state = robot.data.default_root_state.clone()
    joint_pos_init = robot.data.default_joint_pos.clone()
    joint_vel_init = robot.data.default_joint_vel.clone()
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_position_to_sim(joint_pos_init)
    robot.write_joint_velocity_to_sim(joint_vel_init)
    robot.update(SIM_DT) # Initial update

    for step in range(NUM_STEPS_PER_EPISODE):
        # Get current target for this step
        current_target_pos = target_pos_trajectory[step].item()
        current_target_vel = target_vel_trajectory[step].item()

        # Set joint targets
        # For ImplicitActuator, setting pos/vel targets directly translates to PD control by PhysX.
        # For GruActuator, these targets are used as inputs to the neural network.
        # In residual mode, the GRU predicts a residual that gets added to a PD controller output.
        # In non-residual mode, the GRU predicts the full torque directly.
        robot.set_joint_position_target(current_target_pos, joint_ids=[joint_idx])
        robot.set_joint_velocity_target(current_target_vel, joint_ids=[joint_idx])
        # For GruActuator, effort targets are computed internally by the neural network
        # and don't need to be set explicitly via set_joint_effort_target
        # robot.set_joint_effort_target(0.0, joint_ids=[joint_idx])

        # Simulate for CONTROL_DECIMATION steps
        for _ in range(CONTROL_DECIMATION):
            robot.write_data_to_sim()
            sim.step(render=True) # Render at each physics step for smoother visuals
            robot.update(SIM_DT)

        sim_time += SIM_DT * CONTROL_DECIMATION

        # Log data
        current_joint_pos = robot.data.joint_pos[0, joint_idx].item()
        current_joint_vel = robot.data.joint_vel[0, joint_idx].item()
        # `applied_torque` is generally the final torque sent to sim after actuator model processing.
        # For ImplicitActuator, `computed_torque` is an approximation, `applied_torque` should be what physx uses.
        # For GruActuator, `applied_torque` is the final torque after neural network computation and clipping.
        # `articulation.py` line 1403 for implicit: self.applied_effort = self._clip_effort(self.computed_effort)
        # `articulation.py` line 187: self.root_physx_view.set_dof_actuation_forces(self._joint_effort_target_sim, self._ALL_INDICES)
        # self._joint_effort_target_sim is populated by actuator.compute()
        # Let's log applied_torque, which should be available for both implicit and GRU actuators.
        current_torque = robot.data.applied_torque[0, joint_idx].item()

        log_data[f"actual_pos_{episode_label}"].append(current_joint_pos)
        log_data[f"actual_vel_{episode_label}"].append(current_joint_vel)
        log_data[f"torque_{episode_label}"].append(current_torque)

        if simulation_app.is_running() is False:
            break

    print(f"[INFO] Finished episode: {episode_label}")
    return log_data

# --- MAIN FUNCTION ---
def main():
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=(0.0, 0.0, 0.0))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.0, 1.0, 0.5), target=(0.0, 0.0, 0.0))

    # Setup common scene elements (ground plane, light) once
    setup_common_elements()

    # USD path for the robot (ensure this file exists)
    usd_file_path = os.path.join(os.getcwd(), "test_nn_actuator/simple_1dof_robot.usd")
    if not os.path.exists(usd_file_path):
        print(f"[ERROR] USD file not found: {usd_file_path}")
        simulation_app.close()
        return

    # Placeholder network file (ensure this file exists)
    network_file_path = os.path.join(os.getcwd(), "test_nn_actuator/gru-global_debug_best_jit.pt")
    if not os.path.exists(network_file_path):
        print(f"[ERROR] Network file not found: {network_file_path}. Please create it or replace with your model.")
        simulation_app.close()
        return

    # Placeholder Hydra config file (ensure this file exists)
    hydra_config_path = os.path.join(os.getcwd(), "test_nn_actuator/training_config.yaml")
    if not os.path.exists(hydra_config_path):
        print(f"[ERROR] Hydra config file not found: {hydra_config_path}. Please create it or replace with your config.")
        simulation_app.close()
        return

    # Placeholder normalization stats file (ensure this file exists)
    normalization_stats_path = os.path.join(os.getcwd(), "test_nn_actuator/normalization_stats.json")
    if not os.path.exists(normalization_stats_path):
        print(f"[ERROR] Normalization stats file not found: {normalization_stats_path}. Please create it.")
        simulation_app.close()
        return

    # --- Actuator Configurations ---
    # 1. Implicit PD Actuator Config
    implicit_pd_act_cfg = ImplicitActuatorCfg(
        joint_names_expr=[".*joint1.*"],
        stiffness=10.0, # P-gain
        damping=1,    # D-gain
        effort_limit_sim=10.0 # Max torque PhysX can apply
    )
    robot_cfg_implicit = Simple1DoFRobotCfg(
        usd_path=usd_file_path,
        actuator_cfg_instance=implicit_pd_act_cfg,
    )

    # 2. GRU Actuator Config (NN-based)
    # The GRU actuator determines its mode (residual vs non-residual) from the Hydra config file
    # PD parameters are loaded from the Hydra config in residual mode, not from this cfg
    nn_act_cfg = GruActuatorCfg(
        joint_names_expr=[".*joint1.*"],
        # Required files from training pipeline
        network_file=network_file_path,
        hydra_config_path=hydra_config_path,
        normalization_stats_path=normalization_stats_path,

        effort_limit=10.0, # Max output effort of the actuator model
        velocity_limit=100.0, # Velocity limit for the actuator model
        saturation_effort=10.0 # Peak motor torque for DCMotor's torque-speed curve
    )
    robot_cfg_nn = Simple1DoFRobotCfg(
        usd_path=usd_file_path,
        actuator_cfg_instance=nn_act_cfg,
    )

    # --- Generate Target Trajectory ---
    time_points = np.linspace(0, EPISODE_DURATION_S, NUM_STEPS_PER_EPISODE, endpoint=False)
    target_pos_traj, target_vel_traj = generate_sine_wave_targets(time_points, SINE_AMPLITUDE_POS, SINE_FREQUENCY_POS)
    target_pos_traj = target_pos_traj.to(sim.device)
    target_vel_traj = target_vel_traj.to(sim.device)

    # Store all data for plotting
    all_plot_data = {
        "target_pos": target_pos_traj.cpu().numpy(),
        "target_vel": target_vel_traj.cpu().numpy(),
    }
    time_log_for_plot = time_points

    # --- Run Simulation for Implicit PD ---
    robot_implicit = spawn_robot_in_scene(robot_cfg_implicit)
    sim.reset() # Initialize simulation with the new robot
    if simulation_app.is_running() and sim.has_gui():
        # Render a few frames to ensure the scene is loaded visually
        for _ in range(50): sim.render()
    data_implicit = run_simulation_episode(sim, robot_implicit, target_pos_traj, target_vel_traj, "implicit")
    all_plot_data.update(data_implicit)

    # --- Run Simulation for NN Actuator ---
    robot_nn = spawn_robot_in_scene(robot_cfg_nn)
    sim.reset() # Initialize simulation with the new robot
    if simulation_app.is_running() and sim.has_gui():
        # Render a few frames to ensure the scene is loaded visually
        for _ in range(50): sim.render()
    data_nn = run_simulation_episode(sim, robot_nn, target_pos_traj, target_vel_traj, "nn")
    all_plot_data.update(data_nn)

    # --- Plotting ---
    # Assuming the single joint is named 'joint1' as per the USD
    plot_comparison_data(time_log_for_plot, all_plot_data, "joint1")

    print("[INFO] Test script finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close() 