
"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np

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

from mars_jumper.robot.robot_cfg import MarsJumperRobotConfig, MarsJumperRobot

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create origin for robot
    origin = [0.0, 0.0, 0.0]
    
    # Create Mars Jumper Robot
    robot_cfg = MarsJumperRobotConfig()
    robot_cfg.prim_path = "/World/robot"
    mars_jumper_robot = MarsJumperRobot(robot_cfg)
    #mars_jumper_robot.print_usd_info()

    return {"mars_jumper_robot": mars_jumper_robot}, [origin]

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    while simulation_app.is_running():
        # reset every 200 steps
        if count % 200 == 0:
            sim_time = 0.0
            count = 0
            robot = entities["mars_jumper_robot"]
            
            # Reset robot state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins[0]
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting robot state...")
 
        robot = entities["mars_jumper_robot"]
        
        DEG2RAD = np.pi/180.0
        
        abductor_angle = 45 * DEG2RAD
        flexion_angle = 45 * DEG2RAD
        knee_angle = 45 * DEG2RAD
        
        targets = {
            "LF_HAA": abductor_angle, 
            "LH_HAA": -abductor_angle, #postive angle is negative
            "RF_HAA": -abductor_angle, #postive angle is negative
            "RH_HAA": abductor_angle,
            "LF_HFE": flexion_angle,
            "LH_HFE": flexion_angle,
            "RF_HFE": -flexion_angle, #postive angle is negative
            "RH_HFE": flexion_angle,
            "LF_KFE": knee_angle,
            "LH_KFE": knee_angle,
            "RF_KFE": knee_angle,
            "RH_KFE": knee_angle,
        }
        
        for joint_name, joint_pos in targets.items():
            joint_idx = robot.find_joints(joint_name)[0]
            robot.set_joint_position_target(joint_pos, joint_ids=joint_idx)
        
        robot.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1
        robot.update(sim_dt)

def main():
    """Main function."""
    # Initialize simulation
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/100))
    sim.set_camera_view(eye=[1, 1, 1], target=[0.0, 0.0, 0.0])
    # Add camera that follows robot
    # sim.add_camera_sensor(
    #     prim_path="/World/camera",
    #     position=[0, 0, 2],
    #     target=[0, 0, 0],
    #     clipping_range=(0.01, 100.0),
    #     focal_length=24.0,
    #     focus_distance=2.0,
    #     horizontal_aperture=20.955,
    #     horizontal_fov=90.0,
    #     follow_target="/World/robot",
    #     follow_offset=[0, -2, 1]
    # )
    
    
    
    # Setup scene
    scene_entities, scene_origins =  design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    
    # Start simulation
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins)

if __name__ == "__main__":
    main()
    simulation_app.close()
