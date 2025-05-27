import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import random

# This class creates a simulated world or task where an RL agent can 
# learn to control the robotic arm. This uses forward kinematics to compute the end-effector 
# position from the joint angles. It does not use any physics engine or simulation.

# The simple_fk_environment class simulates a 3-DOF, 2-link robotic arm tasked
# with reaching a target in 3D space. It's designed to be used with 
# reinforcement learning algorithms like PPO.

def forward_kinematics_3d(theta0, theta1, theta2, L1=21.0, L2=21.0):
    # Convert to radians
    t0 = np.radians(theta0)
    t1 = np.radians(theta1)
    t2 = np.radians(theta2)

    # In the Y-Z plane, compute position from shoulder and elbow
    y = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    r = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)  # radial distance from base

    # Project onto 3D space using base rotation
    x = r * np.cos(t0)
    z = r * np.sin(t0)

    return np.array([x, y, z])

THETA0_MIN_ANGLE = 0
THETA0_MAX_ANGLE = 360
THETA1_MIN_ANGLE = 0
THETA1_MAX_ANGLE = 180
THETA2_MIN_ANGLE = 0
THETA2_MAX_ANGLE = 180

class simple_fk_environment(gym.Env):
    def __init__(self):
        super(simple_fk_environment, self).__init__()
        self.L1 = 21.0
        self.L2 = 21.0
        # Angle limits for theta1 and theta2
        self.theta1_min_angle = THETA1_MIN_ANGLE
        self.theta1_max_angle = THETA1_MAX_ANGLE
        self.theta2_max_angle = THETA2_MAX_ANGLE
        self.theta2_min_angle = THETA2_MIN_ANGLE
        # Angle limits for theta0
        self.theta0_min_angle = THETA0_MIN_ANGLE
        self.theta0_max_angle = THETA0_MAX_ANGLE

        self.epsilon = 1.0
        self.max_steps = 50

        # Action space: Δθ₀, Δθ₁, Δθ₂
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(3,), dtype=np.float32)

        # Observation space: [θ₀, θ₁, θ₂, x_target, y_target, z_target]
        obs_high = np.array([
            self.theta0_max_angle, self.theta1_max_angle, self.theta2_max_angle,
            (self.L1 + self.L2), (self.L1 + self.L2), (self.L1 + self.L2)
        ], dtype=np.float32)
        obs_low = np.array([
            self.theta0_min_angle, self.theta1_min_angle, self.theta2_min_angle,
            -(self.L1+self.L2),0,-(self.L1+self.L2)
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        self.reset() 

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self.theta = np.zeros(3, dtype=np.float32)
        self.theta[0] = np.random.uniform(self.theta0_min_angle, self.theta0_max_angle)
        self.theta[1] = np.random.uniform(self.theta1_min_angle, self.theta1_max_angle)
        self.theta[2] = np.random.uniform(self.theta2_min_angle, self.theta2_max_angle) 
        
        self.target = np.random.uniform(low=[-(self.L1+self.L2),0,-(self.L1+self.L2)], high=[(self.L1 + self.L2), self.L1 + self.L2, (self.L1 + self.L2)])
        self.steps = 0
        obs = self._get_state()
        return obs, {}

    def _get_state(self):
        return np.concatenate([self.theta, self.target]).astype(np.float32)

    def step(self, action):
        self.steps += 1
        self.theta += action
        
        # Clip angles according to their specific limits
        self.theta[0] = np.clip(self.theta[0], self.theta0_min_angle, self.theta0_max_angle)
        self.theta[1] = np.clip(self.theta[1], self.theta1_min_angle, self.theta1_max_angle)
        self.theta[2] = np.clip(self.theta[2], self.theta2_min_angle, self.theta2_max_angle)

        ee_pos = forward_kinematics_3d(*self.theta, self.L1, self.L2)
        
        # reward function is linearly proportional to the distance to the target
        dist = np.linalg.norm(ee_pos - self.target)
        reward = -dist / (self.L1 + self.L2)
            
        terminated = dist < self.epsilon
        truncated = self.steps >= self.max_steps

        if terminated:
            reward += 6  

        elif truncated:
            reward -= 1 # Penalty for timeout

        obs = self._get_state()
        return obs, reward, terminated, truncated, {}

    def render(self, mode='human'):
        if mode == 'human':
            plt.ion() # Turn on interactive mode

            # Calculate current end-effector position using the full kinematics
            current_ee_world = forward_kinematics_3d(*self.theta, L1=self.L1, L2=self.L2)

            # --- Setup for Subplots ---
            fig, axs = plt.subplots(1, 2, figsize=(16, 8)) # 1 row, 2 columns
            fig.suptitle(f"Step: {getattr(self, 'steps', 0)}, θ0: {self.theta[0]:.1f}, θ1: {self.theta[1]:.1f}, θ2: {self.theta[2]:.1f}")

            # --- Plot 1: Side View in Arm's Vertical Plane ---
            ax1 = axs[0]
            t1_rad = np.radians(self.theta[1])
            t2_rad = np.radians(self.theta[2]) # This is t2 relative to t1 (elbow angle)
            
            # Coordinates in the arm's own vertical plane (radial distance vs height)
            r_link1 = self.L1 * np.cos(t1_rad)
            y_link1 = self.L1 * np.sin(t1_rad)
            r_link2 = r_link1 + self.L2 * np.cos(t1_rad + t2_rad)
            y_link2 = y_link1 + self.L2 * np.sin(t1_rad + t2_rad)

            ax1.plot([0, r_link1, r_link2], [0, y_link1, y_link2], '-o', label='Arm Configuration')
            
            # Target projected onto this radial distance vs height view
            target_r_world = np.sqrt(self.target[0]**2 + self.target[2]**2)
            ax1.plot(target_r_world, self.target[1], 'rx', markersize=10, label=f'Target (r={target_r_world:.1f}, y={self.target[1]:.1f})')
            
            # Current EE projected onto this radial distance vs height view
            current_ee_r_world = np.sqrt(current_ee_world[0]**2 + current_ee_world[2]**2)
            ax1.plot(current_ee_r_world, current_ee_world[1], 'g+', markersize=10, label=f'Current EE (r={current_ee_r_world:.1f}, y={current_ee_world[1]:.1f})')
            
            ax1.set_xlabel("Radial Distance (in arm's plane / world XZ projection)")
            ax1.set_ylabel("Height (World Y)")
            ax1.set_title("Side View (Arm's Vertical Plane)")
            # ax1.axis('equal')
            ax1.set_xlim(-(self.L1 + self.L2) * 1.1, (self.L1 + self.L2) * 1.1)
            ax1.set_ylim(-(self.L1 + self.L2) * 0.2, (self.L1 + self.L2) * 1.1) # Adjusted Y limit for better view
            ax1.legend()
            ax1.grid(True)

            # --- Plot 2: Top-Down View (World XZ Plane) ---
            ax2 = axs[1]
            t0_rad = np.radians(self.theta[0])
            # t1_rad is already defined

            # Joint 1 (elbow) position in world XZ plane
            # r_L1_horizontal_projection is the projection of L1 onto the horizontal in the arm's vertical plane
            r_L1_horizontal_projection = self.L1 * np.cos(t1_rad)
            x_j1_world = r_L1_horizontal_projection * np.cos(t0_rad)
            z_j1_world = r_L1_horizontal_projection * np.sin(t0_rad)

            # End-effector world XZ (from full kinematics)
            x_ee_world = current_ee_world[0]
            z_ee_world = current_ee_world[2]

            ax2.plot([0, x_j1_world, x_ee_world], [0, z_j1_world, z_ee_world], '-o', label='Arm Configuration')
            ax2.plot(self.target[0], self.target[2], 'rx', markersize=10, label=f'Target (x={self.target[0]:.1f}, z={self.target[2]:.1f})')
            ax2.plot(x_ee_world, z_ee_world, 'g+', markersize=10, label=f'Current EE (x={x_ee_world:.1f}, z={z_ee_world:.1f})')
            
            # Plot a line indicating the arm's base rotation direction (theta0)
            dir_len = (self.L1 + self.L2) * 0.25
            ax2.plot([0, dir_len * np.cos(t0_rad)], [0, dir_len * np.sin(t0_rad)], 'b--', alpha=0.5, label=f'θ0 Direction ({self.theta[0]:.1f}°)')


            ax2.set_xlabel("World X")
            ax2.set_ylabel("World Z")
            ax2.set_title("Top-Down View (World XZ Plane)")
            ax2.axis('equal')
            ax2.set_xlim(-(self.L1 + self.L2) * 1.1, (self.L1 + self.L2) * 1.1)
            ax2.set_ylim(-(self.L1 + self.L2) * 1.1, (self.L1 + self.L2) * 1.1)
            ax2.legend()
            ax2.grid(True)
            
            # --- Display and Close ---
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.draw()
            plt.gcf().canvas.flush_events()
            plt.pause(0.05) # Keep plot visible for 0.5 seconds
            
            plt.close(fig) # Close the specific figure

        elif mode == 'rgb_array':
            # This mode is not implemented
            pass