"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates the Mars Jumper robot.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from robot.robot_cfg import MarsJumperRobotCfg
from isaaclab.assets.articulation import Articulation #must be after applaunch
from isaaclab.actuators.actuator_cfg import ActuatorNetLSTMCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
import gymnasium as gym # Import gymnasium

ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)

RPM2RADPS = 2.0 * torch.pi / 60.0
IdealPD_ACTUATOR_CFG = IdealPDActuatorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    stiffness=20.0,
    damping=1.0,
    velocity_limit=470*RPM2RADPS,
    effort_limit=20
)


ANYMAL_B_CFG = ArticulationCfg(
    prim_path="/World/robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-B/anymal_b.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.0,  # both front HFE
            ".*H_HFE": 0.0,  # both hind HFE
            ".*F_KFE": 0.0,  # both front KFE
            ".*H_KFE": 0.0,  # both hind KFE
        },
    ),
    actuators={"legs": IdealPD_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)

def design_scene():
    """Designs the scene."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origin = [0.0, 0.0, 0.0]
    
    # Create Mars Jumper Robot
    mars_jumper_robot_cfg = MarsJumperRobotCfg()
    mars_jumper_robot_cfg.prim_path = "/World/robot"
    mars_jumper_robot = Articulation(mars_jumper_robot_cfg)

    return {"mars_jumper_robot": mars_jumper_robot}, [origin]

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    # Define maximum simulation time (in seconds)
    max_sim_duration = 3.0

    # Logging buffers
    timestamps = []
    all_joint_pos = []
    all_joint_vel = []
    all_joint_efforts = [] # Buffer for joint efforts
    all_target_angles = [] # Buffer for target angles

    # Get the robot articulation
    robot = entities["mars_jumper_robot"]

    # Reset robot state once at the beginning
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += origins[0]
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    print("[INFO]: Starting simulation...")

    while simulation_app.is_running() and sim_time < max_sim_duration:

        # Example: Apply a target angle after a certain time
        if sim_time > 1.0:
            target_angle = torch.full((1, 12), torch.pi * 0.1, device=sim.device) # Ensure target is on the correct device
        else:
            target_angle = torch.zeros((1, 12), device=sim.device) # Ensure target is on the correct device

        robot.set_joint_position_target(target_angle)

        # Log target angle *before* stepping
        all_target_angles.append(target_angle.cpu().numpy().copy()) # Log target angles

        robot.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        robot.update(sim_dt)

        # Log data
        timestamps.append(sim_time)
        all_joint_pos.append(robot.data.joint_pos.cpu().numpy().copy()) # Store data on CPU
        all_joint_vel.append(robot.data.joint_vel.cpu().numpy().copy())
        all_joint_efforts.append(robot.data.applied_torque.cpu().numpy().copy()) # Log applied joint efforts

    return timestamps, all_joint_pos, all_joint_vel, all_joint_efforts, all_target_angles, robot.joint_names

def plot_data(timestamps, all_joint_pos, all_joint_vel, all_joint_efforts, all_target_angles, joint_names):
    """Plots the joint positions, velocities, and efforts over time for selected joints."""
    # Convert lists to numpy arrays - REMOVED, conversion happens in main() now
    # all_joint_pos = np.array(all_joint_pos).squeeze()
    # all_joint_vel = np.array(all_joint_vel).squeeze()
    # all_joint_efforts = np.array(all_joint_efforts).squeeze()
    # timestamps = np.array(timestamps)

    # Handle case where only one joint exists after squeeze - REMOVED, handled in main()
    # if all_joint_pos.ndim == 1:
    #     all_joint_pos = all_joint_pos[:, np.newaxis]
    #     all_joint_vel = all_joint_vel[:, np.newaxis]
    #     all_joint_efforts = all_joint_efforts[:, np.newaxis]

    # --- Select specific joints ---
    # Find the index of the first occurrence of each joint type identifier
    try:
        haa_idx = next(i for i, name in enumerate(joint_names) if 'HAA' in name)
        hfe_idx = next(i for i, name in enumerate(joint_names) if 'HFE' in name)
        kfe_idx = next(i for i, name in enumerate(joint_names) if 'KFE' in name)
        selected_indices = [haa_idx, hfe_idx, kfe_idx]
        selected_joint_names = [joint_names[i] for i in selected_indices]
    except StopIteration:
        print("Warning: Could not find all joint types (HAA, HFE, KFE). Plotting all joints.")
        # Check if data is available before determining shape
        if all_joint_pos.size == 0:
             print("Error: No joint position data to plot.")
             return # Exit if no data
        num_joints = all_joint_pos.shape[1] if all_joint_pos.ndim == 2 else 1
        selected_indices = range(num_joints)
        selected_joint_names = joint_names
    # --- End selection ---


    # Create position plot
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1) # Create subplot for positions
    plt.title("Selected Joint Positions Over Time")
    for i in selected_indices: # Plot only selected joints
        plt.plot(timestamps, all_joint_pos[:, i], label=joint_names[i])
        # Plot the corresponding target angle with a dashed line
        plt.plot(timestamps, all_target_angles[:, i], '--', label=f"{joint_names[i]} (Target)")
    plt.ylabel("Position (rad)")
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.xlim(0, timestamps[-1]) # Optional: Force x-axis limit for debugging

    # Create velocity plot
    plt.subplot(3, 1, 2) # Create subplot for velocities
    plt.title("Selected Joint Velocities Over Time")
    for i in selected_indices: # Plot only selected joints
        plt.plot(timestamps, all_joint_vel[:, i], label=joint_names[i])
    plt.ylabel("Velocity (rad/s)")
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.xlim(0, timestamps[-1]) # Optional: Force x-axis limit for debugging

    # Create effort plot
    plt.subplot(3, 1, 3) # Create subplot for efforts
    plt.title("Selected Joint Efforts (Applied Torque) Over Time")
    for i in selected_indices: # Plot only selected joints
        plt.plot(timestamps, all_joint_efforts[:, i], label=joint_names[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Effort (Nm)")
    plt.legend(loc='upper right')
    plt.grid(True)
    # plt.xlim(0, timestamps[-1]) # Optional: Force x-axis limit for debugging

    plt.tight_layout() # Adjust layout to prevent overlap
    plt.savefig("selected_joint_data.png") # Save to a different file name
    print("Plot saved to selected_joint_data.png") # Added print statement

def main():
    """Main function."""
    # Initialize simulation
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/600))
    # Set camera view before designing scene might be more reliable
    sim.set_camera_view(eye=[1, 1, 1], target=[0.0, 0.0, 0.0])

    # Setup scene
    scene_entities, scene_origins =  design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)


    # Start simulation
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run simulation and get logged data
    timestamps, all_joint_pos, all_joint_vel, all_joint_efforts, all_target_angles, joint_names = run_simulator(sim, scene_entities, scene_origins)
    
    # Print summary statistics
    print("\nSimulation Summary:")
    print(f"Duration: {timestamps[-1]:.2f} seconds")
    print(f"Number of timesteps: {len(timestamps)}")
    

    # --- Convert data lists to NumPy arrays ---
    timestamps = np.array(timestamps)
    # Squeeze to remove env dim if num_envs=1, handle potential single timestep case
    all_joint_pos = np.array(all_joint_pos).squeeze()
    all_joint_vel = np.array(all_joint_vel).squeeze()
    all_joint_efforts = np.array(all_joint_efforts).squeeze()
    all_target_angles = np.array(all_target_angles).squeeze() # Convert target angles

    # Ensure arrays are 2D even if only one timestep or one joint was logged
    if all_joint_pos.ndim == 0: # Single value (1 timestep, 1 joint)
        all_joint_pos = all_joint_pos.reshape(1, 1)
        all_joint_vel = all_joint_vel.reshape(1, 1)
        all_joint_efforts = all_joint_efforts.reshape(1, 1)
        all_target_angles = all_target_angles.reshape(1, 1) # Reshape target angles
    elif all_joint_pos.ndim == 1: # Either 1 joint over time, or multiple joints for 1 timestep
        if len(timestamps) > 1: # 1 joint over time
            all_joint_pos = all_joint_pos[:, np.newaxis]
            all_joint_vel = all_joint_vel[:, np.newaxis]
            all_joint_efforts = all_joint_efforts[:, np.newaxis]
            all_target_angles = all_target_angles[:, np.newaxis] # Reshape target angles
        else: # Multiple joints for 1 timestep
             all_joint_pos = all_joint_pos[np.newaxis, :]
             all_joint_vel = all_joint_vel[np.newaxis, :]
             all_joint_efforts = all_joint_efforts[np.newaxis, :]
             all_target_angles = all_target_angles[np.newaxis, :] # Reshape target angles
    # --- End conversion ---

    # Plot the data after simulation finishes
    # Pass the already converted numpy arrays to plot_data
    plot_data(timestamps, all_joint_pos, all_joint_vel, all_joint_efforts, all_target_angles, joint_names)

    # Save data to CSV
    import pandas as pd

    # Check if data arrays are valid before creating DataFrame
    if all_joint_pos.ndim != 2 or all_joint_vel.ndim != 2 or all_joint_efforts.ndim != 2 or all_target_angles.ndim != 2:
         print("Error: Data arrays do not have the expected 2 dimensions after processing. Skipping CSV save.")
    else:
        data = {
            'timestamp': timestamps,
            **{f'{name}_pos': all_joint_pos[:, i] for i, name in enumerate(joint_names)},
            **{f'{name}_vel': all_joint_vel[:, i] for i, name in enumerate(joint_names)},
            **{f'{name}_effort': all_joint_efforts[:, i] for i, name in enumerate(joint_names)},
            **{f'{name}_target': all_target_angles[:, i] for i, name in enumerate(joint_names)} # Add targets to CSV data
        }

        df = pd.DataFrame(data)
        df.to_csv('joint_data.csv', index=False)
        print("Joint data saved to joint_data.csv")

if __name__ == "__main__":
    main()
    simulation_app.close()
