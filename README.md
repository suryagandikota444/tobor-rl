# Two-Link Robot Arm Reinforcement Learning

This repository contains code for training and controlling a two-link robotic arm using reinforcement learning. The robot consists of a base, a first arm segment, and a second arm segment extending from the first.

## File Structure

### Core Environment Files
- **`mujoco_working_code/tobor_mujoco_env.py`** - Main RL environment class that creates a simulated world using Mujoco physics engine. Designed for training reinforcement learning algorithms like PPO to control the robotic arm.
- **`old_test/simple_fk_environment.py`** - Previous version of the environment that uses simple forward kinematics instead of full physics simulation.
- **`mujoco_working_code/tobor.xml`** - XML file that defines the robot's structure, joints, and physical properties.

### Training and Evaluation
- **`mujoco_working_code/train.py`** - Trains a PPO (Proximal Policy Optimization) algorithm using either the simple forward kinematics or full physics environment.
- **`mujoco_working_code/sim_dynamic_marker.py`** - Visualizes training performance and metrics.
- **`mujoco_working_code/control.py`** - Inference script that loads a trained model and sends control commands to the physical robot.

### Additional Components
- **`old_test/stereo_depth_pipeline/`** - Future perception module (not currently in use) that will enable the robot to perceive its environment using stereo depth cameras.
- **`old_test/scratch.py`** - Development/testing file.

## Robot Description

The robot is a two-link arm system with the following structure:
- **Base**: Fixed mounting point
- **Link 1**: First arm segment connected to the base
- **Link 2**: Second arm segment connected to Link 1

This configuration allows for planar movement and is commonly used for learning fundamental robotic control concepts.

## Usage

1. **Training**: Use `train.py` to train RL models on the simulated environment
2. **Visualization**: Use `sim_dynamic_marker.py` to monitor training progress and performance
3. **Deployment**: Use `control.py` to deploy trained models to the physical robot

## Dependencies

- Mujoco (physics simulation)
- PPO implementation (reinforcement learning)
- Standard robotics libraries for XML parsing and control 