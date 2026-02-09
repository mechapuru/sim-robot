"""
Script to perform and record a pick and place task.
"""

import os
import sys
import time
import numpy as np
import pybullet as p
from termcolor import cprint
from typing import List, Optional

# Ensure package is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sim import SimpleEnv
from wrapper import PointCloudWrapper
from motion_planning.motion_planning_utils import motion_planning
from utils.state_io import save_trajectory

def move_robot(env, path, record_trajectory=None):
    """Execute path and optionally record trajectory."""
    for q in path:
        env.env.robot.control(env.env.robot.arm_joint_indices, q)
        
        # Stabilize and record
        for _ in range(25):
            p.stepSimulation(physicsClientId=env.env.id)
            if env.env.gui:
                time.sleep(1/240)
                
        if record_trajectory is not None:
            # Capture observation
            raw_state = env.env.robot.get_state()
            obs = env.observation(raw_state)
            record_trajectory.append(obs)

def main():
    # 1. Initialize Environment
    print("Initializing environment...")
    config_path = os.path.join(os.path.dirname(__file__), "configs", "example_scene.yaml")
    
    env = SimpleEnv(
        config_path=config_path,
        gui=True,
        robot_base_pos=[0, 0, 0]
    )
    
    # Use 'scene' mode to capture everything as requested
    env = PointCloudWrapper(env, num_points=1000)
    
    # 2. Reset
    print("Resetting environment...")
    env.reset()
    
    # Define Positions
    # Note: These depend on your setup. You might need to adjust based on cube location.
    # Current cube config: center: (0.4, 0.0, 0.025)
    
    # Adjusted for centered robot base [0, 0, 0]
    # We must account for the gripper length since we control the FLANGE (Link 7)
    # but want the FINGERTIPS to be at the object.
    GRIPPER_OFFSET = 0.13 # ~13cm from flange to grip center
    
    pick_pos = [-0.3, 0.0, 0.035 + GRIPPER_OFFSET] 
    place_pos = [0.0, 0.2, 0.05 + GRIPPER_OFFSET] 
    
    # Orientation: Downward facing gripper
    down_orn = p.getQuaternionFromEuler([0, np.pi, 0])
    
    trajectory = []
    
    # -------------------------------------------------------------------------
    # Phase 1: Pre-Grasp (Move above object)
    # -------------------------------------------------------------------------
    cprint("Phase 1: Moving to Pre-Grasp...", "cyan")
    pre_grasp_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.15]
    
    success, path, _, _ = motion_planning(env.env, pre_grasp_pos, down_orn)
    if not success:
        cprint("Planning to Pre-Grasp Failed!", "red")
        return
        
    move_robot(env, path, trajectory)
    
    # -------------------------------------------------------------------------
    # Phase 2: Open Gripper & Approach
    # -------------------------------------------------------------------------
    cprint("Phase 2: Opening Gripper & Approaching...", "cyan")
    env.env.robot.open_gripper()
    for _ in range(20): 
        p.stepSimulation(physicsClientId=env.env.id)
        if env.env.gui: time.sleep(0.01)
        
    # Cartesian move down (simple linear interpolation for approach)
    # For simplicity, we use motion planner for safety, but IK direct control is also an option
    success, path, _, _ = motion_planning(env.env, pick_pos, down_orn)
    if not success:
        cprint("Approach Failed!", "red")
        return
    move_robot(env, path, trajectory)
    
    # -------------------------------------------------------------------------
    # Phase 3: Grasp
    # -------------------------------------------------------------------------
    cprint("Phase 3: Grasping...", "cyan")
    env.env.robot.close_gripper()
    for _ in range(50): # Wait for grasp to settle
        p.stepSimulation(physicsClientId=env.env.id)
        if env.env.gui: time.sleep(0.01)
        # Record holding state
        obs = env.observation(env.env.robot.get_state())
        trajectory.append(obs)
        
    # -------------------------------------------------------------------------
    # Phase 4: Lift
    # -------------------------------------------------------------------------
    cprint("Phase 4: Lifting...", "cyan")
    lift_pos = [pick_pos[0], pick_pos[1], pick_pos[2] + 0.2]
    success, path, _, _ = motion_planning(env.env, lift_pos, down_orn)
    if not success:
        cprint("Lift Failed!", "red")
        return
    move_robot(env, path, trajectory)
    
    # -------------------------------------------------------------------------
    # Phase 5: Move to Place
    # -------------------------------------------------------------------------
    cprint("Phase 5: Moving to Place Location...", "cyan")
    pre_place_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.2]
    success, path, _, _ = motion_planning(env.env, pre_place_pos, down_orn)
    if not success:
        cprint("Move to Place Failed!", "red")
        return
    move_robot(env, path, trajectory)
    
    # -------------------------------------------------------------------------
    # Phase 6: Lower & Release
    # -------------------------------------------------------------------------
    cprint("Phase 6: Lowering & Releasing...", "cyan")
    success, path, _, _ = motion_planning(env.env, place_pos, down_orn)
    if not success:
        cprint("Lower Failed!", "red")
        return
    move_robot(env, path, trajectory)
    
    env.env.robot.open_gripper()
    for _ in range(50):
        p.stepSimulation(physicsClientId=env.env.id)
        if env.env.gui: time.sleep(0.01)
        obs = env.observation(env.env.robot.get_state())
        trajectory.append(obs)
        
    # -------------------------------------------------------------------------
    # Phase 7: Retreat
    # -------------------------------------------------------------------------
    cprint("Phase 7: Retreating...", "cyan")
    success, path, _, _ = motion_planning(env.env, pre_place_pos, down_orn)
    move_robot(env, path, trajectory)
    
    # Save
    cprint(f"Task Complete! Saving {len(trajectory)} frames...", "green")
    save_path = os.path.join(os.path.dirname(__file__), "pick_place_scenedata.pkl")
    save_trajectory(save_path, trajectory)
    cprint(f"Saved to {save_path}", "green")
    
    # Visualize final
    time.sleep(2)

if __name__ == "__main__":
    main()
