# Two-Link Robot Arm Reinforcement Learning

This repository contains code for training and controlling a two-link robotic arm using reinforcement learning. The robot consists of a base, a first arm segment, and a second arm segment extending from the first.

## Automated Clothes Steaming (`steamer/`)

The `steamer/` directory contains the firmware for Tobor's flagship daily-use application: **autonomous garment steaming**. The idea is simple — mount a handheld steamer to the robot's end effector, hang a shirt on a rack, and let the arm do the rest every morning.

### How It Works

The ESP32-based firmware (`steamer.ino`) runs a WiFi access point and exposes HTTP endpoints to orchestrate the arm. Under the hood it:

1. **Controls a dual two-link arm system** (left and right, 6 servos total) via a PCA9685 PWM driver over I2C, with smooth non-blocking motion interpolation at 20 ms intervals.
2. **Computes inverse kinematics** on-board to translate desired (x, y) end-effector positions into shoulder and elbow joint angles, keeping the steamer head precisely on target.
3. **Executes a predefined sweep pattern** — the arm starts in an inverted-V pose, sweeps the steamer downward across the fabric, rotates the base to the next column, and repeats — covering the full width of the garment in a series of vertical passes.
4. **Accepts real-time commands** over HTTP (`/set_angle`, `/set_actuators`, `/move_ik`) so the sweep can be tuned, paused, or overridden from a phone or the companion app.

### Typical Morning Routine

- Hang your shirt on the rack the night before.
- Fill and power on the steamer.
- Tobor connects to your network (or hosts its own AP), starts the sweep sequence, and finishes in a few minutes — wrinkle-free clothes with zero effort.

### Hardware

- ESP32 microcontroller
- PCA9685 16-channel 12-bit PWM servo driver
- Two-link arm (3 DOF per side) — 3D-printed with 270° servos
- Handheld garment steamer mounted at the end effector

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