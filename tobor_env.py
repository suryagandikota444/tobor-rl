import pybullet as p
import pybullet_data
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class tobor_env(gym.Env):
    def __init__(self, urdf_path="tobor.urdf", gui=False, action_repeat=10): # Added action_repeat from previous context
        super().__init__()
        self.gui = gui
        self.urdf_path = urdf_path
        self.physics_client = p.connect(p.GUI if self.gui else p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True, physicsClientId=self.physics_client)
        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)
        
        self.previous_dist_to_target = None # Will be properly set in reset

        spaces_box_low = -0.2
        spaces_box_high = 0.2
        self.action_space = spaces.Box(low=spaces_box_low, high=spaces_box_high, shape=(self.num_joints,), dtype=np.float32)

        # Observation space is the joint angles and the target position: [x, y, z, theta0, theta1, theta2]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -.28, -.28, 0]),
            high=np.array([6.28, 3.14, 2.7, .28, .28, .28]), # URDF joint limits + target space
            dtype=np.float32
        )

        self.current_success_threshold = 0.04  # Curriculum-based success threshold for termination & big bonus
        self.action_repeat = action_repeat # Number of physics steps per agent action

        self.target_zone_threshold = 0.06  # Radius for being "in the zone" (larger than success threshold)
        self.target_zone_reward_bonus = 50 # Reward per step for being in the target zone
        
        self.k_decay_exp_dense = 15.0       # Decay factor for exponential dense reward
        self.max_dense_reward_val_exp = 2.0 # Max value of exponential dense reward (at dist=0)
        
        self.action_penalty_scale = 0.01    # Scale for action magnitude penalty
        self.progress_reward_scale = 5.0     # Scale for progress reward

        self.max_steps = 200 # Max agent steps
        self.step_counter = 0 # Counts agent steps
        self._reset_target() # Sets target_position for the first time
        self.target_debug_item_id = -1 # For render visualization

        print("Environment Initialized:")
        print(f"  Action space high: {spaces_box_high}, low: {spaces_box_low}")
        print(f"  Initial success_threshold (for termination): {self.current_success_threshold}")
        print(f"  Target zone threshold (for staying reward): {self.target_zone_threshold}")
        print(f"  Max agent steps: {self.max_steps}")
        print(f"  Action repeat (physics steps per agent step): {self.action_repeat}")
        print(f"  Target zone reward bonus: {self.target_zone_reward_bonus}")
        print(f"  target zone reward bonus scale: {self.target_zone_reward_bonus}")
        print(f"  k_decay_exp_dense: {self.k_decay_exp_dense}")
        print(f"  max_dense_reward_val_exp: {self.max_dense_reward_val_exp}")
        print(f"  action_penalty_scale: {self.action_penalty_scale}")
        print(f"  progress_reward_scale: {self.progress_reward_scale}")


    def _reset_target(self):
        self.target_position = np.random.uniform(low=[-.28, -.28, 0], high=[.28, .28, .28])

    def set_success_threshold(self, new_threshold):
        """Allows changing the success threshold during training (for curriculum)."""
        print(f"Updating success threshold (for termination/big bonus) from {self.current_success_threshold} to {new_threshold}")
        self.current_success_threshold = new_threshold

    def set_zone_threshold(self, new_threshold):
        """Allows changing the target zone threshold during training (for curriculum)."""
        print(f"Updating target zone threshold (for staying reward) from {self.target_zone_threshold} to {new_threshold}")
        self.target_zone_threshold = new_threshold

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True, physicsClientId=self.physics_client)
        self.target_debug_item_id = -1 

        joint_limits_low = [0, 0, 0]
        joint_limits_high = [6.28, 3.14, 2.7]
        for i in range(self.num_joints):
            angle = np.random.uniform(joint_limits_low[i], joint_limits_high[i])
            p.resetJointState(self.robot_id, i, angle, physicsClientId=self.physics_client)
        
        self.step_counter = 0
        self._reset_target() # Set the new target for this episode

        # Initialize previous_dist_to_target AFTER new target is set and joints are positioned
        ee_pos_tuple = p.getLinkState(self.robot_id, self.num_joints - 1, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_pos_tuple[0]) # Assuming URDF CoM is at the tip
        self.previous_dist_to_target = np.linalg.norm(ee_pos - self.target_position)
        
        return self._get_obs(), {}

    def step(self, action):
        self.step_counter += 1 # Agent step
        
        joint_states_before_action = p.getJointStates(self.robot_id, range(self.num_joints), physicsClientId=self.physics_client)
        current_positions = np.array([s[0] for s in joint_states_before_action])

        for i in range(self.num_joints):
            new_target_angle = current_positions[i] + action[i]
            p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=i,
                controlMode=p.POSITION_CONTROL,
                targetPosition=new_target_angle,
                force=50000, 
                physicsClientId=self.physics_client
            )
        
        # num_physics_steps_per_agent_step = 10 # Using self.action_repeat now
        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)

        ee_pos_tuple = p.getLinkState(self.robot_id, self.num_joints - 1, physicsClientId=self.physics_client)
        ee_pos = np.array(ee_pos_tuple[0]) # Assuming URDF CoM is at the tip
        dist = np.linalg.norm(ee_pos - self.target_position)
        
        # Initialize reward for this step
        current_step_reward = 0.0

        # dense_reward = -10 * (dist / .63) # Reward is e.g. [-1, 0]
        # current_step_reward = dense_reward 

        exp_dense_reward = -1 * self.max_dense_reward_val_exp * np.exp(-self.k_decay_exp_dense * dist)
        current_step_reward += exp_dense_reward

        # Penalizes large actions to encourage smoother control.
        action_mag_penalty = -1 * self.action_penalty_scale * np.linalg.norm(action)
        current_step_reward += action_mag_penalty

        # Rewards the agent for reducing its distance to the target compared to the previous step.
        if self.previous_dist_to_target is not None:
            distance_reduction = self.previous_dist_to_target - dist
            progress_rwd = -1 * self.progress_reward_scale * distance_reduction
            current_step_reward += progress_rwd
        
        # Rewards the agent for being within a certain radius of the target.
        # This can encourage it to stay near the target even if it doesn't hit the precise success threshold.
        if dist < self.target_zone_threshold:
            current_step_reward += self.target_zone_reward_bonus

        if dist < self.current_success_threshold: 
            current_step_reward += 100.0 # Large bonus for precise success

        terminated = dist < self.current_success_threshold # Termination based on precise success
        
        truncated = False
        if self.step_counter >= self.max_steps:
            current_step_reward -= 10.0 
            truncated = True
            
        # Update previous_dist_to_target for the next agent step's progress calculation
        self.previous_dist_to_target = dist

        print("--------------------------------")
        print("current position: ", current_positions)
        print("action: ", action)
        print("ee_pos: ", ee_pos)
        print("target_position: ", self.target_position)
        print("dist: ", dist)
        print("self.step_counter: ", self.step_counter)
        print("reward: ", current_step_reward) # Print the final reward for the step
        print("--------------------------------")
            
        return self._get_obs(), current_step_reward, terminated, truncated, {}

    def _get_obs(self):
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints), physicsClientId=self.physics_client)
        joint_positions = np.array([s[0] for s in joint_states])
        # Ensure target_position is an np.array for concatenation
        return np.concatenate([joint_positions, np.asarray(self.target_position)]).astype(np.float32)

    def render(self, mode="human"):
        if self.gui and hasattr(self, 'target_position') and self.target_position is not None:
            point_color = [1, 0, 0] 
            point_size = 15          
            self.target_debug_item_id = p.addUserDebugPoints(
                [self.target_position.tolist()], 
                [point_color],                  
                pointSize=point_size,
                replaceItemUniqueId=self.target_debug_item_id,
                physicsClientId=self.physics_client
            )

        if mode == "rgb_array":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1.0, 0.7, 0.5],
                cameraTargetPosition=[0, 0, 0.2],
                cameraUpVector=[0, 0, 1],
                physicsClientId=self.physics_client
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=2.0,
                physicsClientId=self.physics_client
            )
            width, height, rgb_img, _, _ = p.getCameraImage(
                width=320,
                height=240,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL, 
                physicsClientId=self.physics_client
            )
            return np.array(rgb_img)[:, :, :3]
        
        elif mode == "human":
            if not self.gui:
                img_array = self.render(mode="rgb_array") 
                if img_array is not None:
                    import cv2
                    cv2.imshow("tobor_env_cv_render", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
            return None
        
        return None 
    
    def close(self):
        if self.physics_client >= 0: 
            p.disconnect(physicsClientId=self.physics_client)
            self.physics_client = -1 
