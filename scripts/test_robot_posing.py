"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher

# --- CLI Setup ---
parser = argparse.ArgumentParser(description="This script demonstrates setting and testing static poses for the Mars Jumper robot.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation # must be after applaunch
from isaaclab.utils.math import quat_from_euler_xyz

# --- CONFIG ---
DEG2RAD = np.pi / 180.0
robot_cfg = MarsJumperRobotCfg()

# For pure kinematic pose testing, disable actuator interference
# by setting stiffness and damping to zero.
# This prevents actuators from fighting the directly set joint positions.

gravity_on = False # Set to False for pure pose testing without gravity interference

root_height = 0.5 # Desired root height of the robot's base

# Define the desired static pose using joint angles
abductor_angle = 0.0
hip_angle = 45 * DEG2RAD
knee_angle = 45 * DEG2RAD
base_pitch = 0 * DEG2RAD # Desired pitch of the robot's base


# --- Scene Design ---
def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)
    origin = [0.0, 0.0, 0.0]
    robot_cfg.prim_path = "/World/robot"
    mars_jumper_robot = Articulation(robot_cfg)
    return {"mars_jumper_robot": mars_jumper_robot}, [origin]

# --- Simulation Loop ---
def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    sim_dt = sim.get_physics_dt()
    robot = entities["mars_jumper_robot"]

    target_root_state = robot.data.default_root_state.clone()
    target_joint_pos = torch.zeros_like(robot.data.default_joint_pos)
    target_joint_vel = torch.zeros_like(robot.data.default_joint_vel)
    
    knee_joint_idx = robot.find_joints(robot_cfg.KFE_REGEX)[0]
    hip_joint_idx = robot.find_joints(robot_cfg.HFE_REGEX)[0]
    
    target_joint_pos[:, knee_joint_idx] = knee_angle
    target_joint_pos[:, hip_joint_idx] = hip_angle

    pitch_as_tensor = torch.tensor(base_pitch, device=sim.device)
    base_quat_w = quat_from_euler_xyz(torch.zeros_like(pitch_as_tensor), pitch_as_tensor, torch.zeros_like(pitch_as_tensor))
    
    target_root_state[0, 2] = root_height      # Set z (height)
    target_root_state[0, 3:7] = base_quat_w    # Set orientation (qx, qy, qz, qw)
    target_root_state[0, 7:] = 0.0             # Set root linear and angular velocities to zero
    
    robot.write_root_state_to_sim(target_root_state)
    robot.write_joint_position_to_sim(target_joint_pos)
    robot.write_joint_velocity_to_sim(target_joint_vel) 

    print("[INFO]: Starting simulation. Robot pose will be set at each step.")
    while simulation_app.is_running():

        # Use actuator commands to set joint targets
        robot.set_joint_position_target(target_joint_pos)
        robot.set_joint_velocity_target(target_joint_vel)
        # Apply the joint position/velocity targets to the simulation
        robot.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)
        simulation_app.update()

    print("[INFO]: Simulation loop ended.")

def main():

    sim_cfg = sim_utils.SimulationCfg(
        dt=1/100.0, # Using a slightly larger dt than original for posing
        gravity=(0.0, 0.0, -9.81 if gravity_on else 0.0)
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3]) 
    
    scene_entities, scene_origins = design_scene()

    sim.reset()
    print("[INFO]: Setup complete. Starting robot posing script...")
    
    run_simulator(sim, scene_entities, torch.tensor(scene_origins, device=sim.device))

if __name__ == "__main__":
    main()
    simulation_app.close()
    print("[INFO]: Application closed.") 