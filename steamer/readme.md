# Steamer Application

## Overview
This application is designed to control a 2-link robot arm with 3 degrees of freedom (3DOF) using an ESP32. The primary purpose is to automate the steaming of clothes by holding a steamer and sweeping across a shirt/article of clothing. The application calculates inverse kinematics (IK) to determine the necessary angles for the arm's joints and sends pulse signals to motor drivers to achieve the desired movements.

## Functionality
- **Individual Angle Control:** Allows for precise control of each joint by sending specific angles to the motor drivers.
- **Inverse Kinematics (IK):** Computes the required joint angles to position the end effector at a desired location.
- **Predefined Sweep Motion:** Executes a series of movements to perform a "sweep" across a garment. The robot arm starts in an upside-down V position, moves the end effector down and up, rotates the base, and repeats the motion for a specified number of steps.

## Setup
- **Hardware:**
  - ESP32 microcontroller
  - PCA9685 16-channel 12-bit PWM servo driver
  - 2-link robot arm with 3DOF (3D printed)
  - 3 servo motors (270 degree rotation)
