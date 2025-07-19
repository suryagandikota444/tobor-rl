import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
from gymnasium.envs.registration import register
import random
from scipy.optimize import minimize  # For IK reachability check


class ToborEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("tobor.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.data.qpos[:] = [0, 0, 0]
        self.frame_skip = 1
        self.sustained_position_steps_threshold = 10
        self.sustained_position_steps = 0
        self.in_success_threshold = False
        # Joint limits in radians
        self.joint_limits_low = np.deg2rad(np.array([0.0, 0.0, 0.0]))
        self.joint_limits_high = np.deg2rad(np.array([360.0, 180.0, 135.0]))
        self.success_threshold = 0.075  # Initial loose threshold for curriculum start

        # Sets -10 - 10 as the max/min angles that the base/arms can move in one step
        self.action_space = spaces.Box(
            low=np.array([-5, -5, -5]),
            high=np.array([5, 5, 5]),
            dtype=np.float32
        )

        # Sets the range of the target position that the robot can reach
        pos_low = np.array([0, 0, 0], dtype=np.float32)
        pos_high = np.array([360, 180, 135], dtype=np.float32)  # In degrees
        target_low = np.array([-0.20, -0.20, 0.01], dtype=np.float32)
        target_high = np.array([0.20, 0.20, 0.20], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate([pos_low, target_low]),
            high=np.concatenate([pos_high, target_high]),
            dtype=np.float32
        )

        mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        self.data.qpos[:] += np.deg2rad(action)
        self.data.qpos = np.clip(self.data.qpos, self.joint_limits_low, self.joint_limits_high)

        for _ in range(self.frame_skip):
            mujoco.mj_forward(self.model, self.data)

        obs = np.concatenate([np.rad2deg(self.data.qpos), self._target])
        reward = -np.linalg.norm(self._get_ee_pos() - self._target)

        # enourages motion to stay at a certain location after reaching and penalizes for moving away
        if self.in_success_threshold and -reward > self.success_threshold:
            reward -= 1.5
            self.sustained_position_steps = 0
            
        if -reward < self.success_threshold:
            self.sustained_position_steps += 1
            self.in_success_threshold = True
            reward += 0.5

        terminated = self.sustained_position_steps >= self.sustained_position_steps_threshold
        truncated = False

        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None, starting_pos=None):
        super().reset(seed=seed)
        self.data.qvel[:] = np.zeros(self.model.nv)

        # Sample random joint angles within limits to generate a reachable target
        random_qpos = np.random.uniform(self.joint_limits_low, self.joint_limits_high)
        self.data.qpos[:] = random_qpos
        mujoco.mj_forward(self.model, self.data)
        self._target = self._get_ee_pos().copy()
        self.sustained_position_steps = 0

        # Reset qpos to a random position
        random_qpos = np.random.uniform(self.joint_limits_low, self.joint_limits_high)
        self.data.qpos[:] = random_qpos
        if starting_pos:
            self.data.qpos[:] = starting_pos
        mujoco.mj_forward(self.model, self.data)

        # print(self._target)
        # print("reset")

        obs = np.concatenate([np.rad2deg(self.data.qpos), self._target])
        return obs, {}

    def set_success_threshold(self, new_threshold):
        self.success_threshold = new_threshold

    def render(self):
        if self.render_mode == "human":
            if not hasattr(self, "_viewer"):
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if hasattr(self, "_viewer"):
            self._viewer.close()

    def _get_ee_pos(self):
        return self.data.site_xpos[self.model.site("end_effector").id]

    # Optional: Keep is_reachable for testing, but not used in reset
    def is_reachable(self, target, tol=0.01):
        def fk(q):
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            return self._get_ee_pos()

        def objective(q):
            return np.linalg.norm(fk(q) - target)

        bounds = list(zip(self.joint_limits_low, self.joint_limits_high))
        initial_guess = self._get_initial_guess(target)

        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': 50})
        return result.fun < tol

    def _get_initial_guess(self, target):
        q1 = np.arctan2(target[1], target[0])
        if q1 < 0:
            q1 += 2 * np.pi
        q1 = np.clip(q1, self.joint_limits_low[0], self.joint_limits_high[0])
        q2 = np.pi / 2
        q3 = 0.0
        return np.array([q1, q2, q3])

    def test_reachability(self, target):
        self.data.qpos[:] = np.deg2rad([90, 0, 0])
        mujoco.mj_forward(self.model, self.data)
        ee_pos = self._get_ee_pos()
        dist = np.linalg.norm(ee_pos - target)
        reachable = self.is_reachable(target)
        print(f"Target: {target}, Example EE: {ee_pos}, Dist: {dist}, Reachable in limits: {reachable}")


register(
    id="tobor-v0",
    entry_point="tobor_mujoco_env:ToborEnv",
    max_episode_steps=200,
)