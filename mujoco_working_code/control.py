import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gym  # For subclassing
import gym.spaces as spaces
import urllib.request
import time
import mujoco
import gymnasium as gym
import tobor_mujoco_env  # Registers env

class RobotController:
    def __init__(self, ip='192.168.4.1'):
        self.ip = ip

    def set_angle(self, servo, angle):
        url = f"http://{self.ip}/set_angle?servo={servo}&angle={int(angle)}"
        try:
            urllib.request.urlopen(url).read()
            print(f"Sent servo={servo}, angle={angle:.2f}")
        except Exception as e:
            print(f"Error: {e}")
    
    def set_angles(self, servos):
        url = f"http://{self.ip}/set_angles?servo0=0&angle0={servos[0]}&servo1=1&angle1={servos[1]}&servo2=2&angle2={servos[2]}"
        try:
            urllib.request.urlopen(url).read()
            print(f"Sent servos={servos}")
        except Exception as e:
            print(f"Error: {e}")

class DummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))
        self.action_space = spaces.Box(low=-10, high=10, shape=(3,))

    def reset(self):
        return np.zeros(6)

    def step(self, action):
        return np.zeros(6), 0, False, {}

# Load normalization and model (base to match sim performance)
vec_env = DummyVecEnv([lambda: DummyEnv()])
norm_env = VecNormalize.load("vec_normalize_stats_curriculum.pkl", vec_env)
model = PPO.load("ppo_tobor_arm_normalized_curriculum.zip", env=norm_env)
# For penalize (after fixing/retraining): 
# norm_env = VecNormalize.load("vec_normalize_stats_curriculum_penalize_when_out_threshold_5_deg.pkl", vec_env)
# model = PPO.load("ppo_tobor_arm_normalized_curriculum_penalize_when_out_threshold_5_deg.zip", env=norm_env)

# Load env for exact FK (no render)
fk_env = gym.make("tobor-v0").env.env.env  # Access model/data

def get_ee_pos(qpos_rad):
    fk_env.data.qpos[:] = qpos_rad
    mujoco.mj_forward(fk_env.model, fk_env.data)
    return fk_env._get_ee_pos().copy()

robot = RobotController()

targets = np.array([
    [-.3, 0, .2],
    [-.3, 0, 0],
    [-.3, .1, .2],
    [-.3, .1, 0],
    [-.3, .3, .2],
    [-.3, .3, 0],
], dtype=np.float32)

joint_limits_low = np.deg2rad([0, 0, 0])
joint_limits_high = np.deg2rad([180, 180, 135])

for target_idx, target in enumerate(targets):
    print(f"\n=== TARGET {target_idx + 1}/{len(targets)}: {target} ===")
    qpos = np.zeros(3)  # radians
    step_count = 0
    max_steps = 50
    done = False

    while not done and step_count < max_steps:
        raw_obs = np.concatenate([np.rad2deg(qpos), target])
        obs = norm_env.normalize_obs(np.array([raw_obs]))[0]
        action, _ = model.predict(obs, deterministic=True)
        print(f"action: {action}")
        qpos += np.deg2rad(action)
        qpos = np.clip(qpos, joint_limits_low, joint_limits_high)
        robot.set_angles(np.rad2deg(qpos))  # Uncomment for real
        time.sleep(.1)
        ee_pos = get_ee_pos(qpos)
        dist = np.linalg.norm(ee_pos - target)
        print(f"Step {step_count}: Dist={dist:.4f}")
        if dist < 0.032:
            done = True
        step_count += 1

    final_dist = np.linalg.norm(get_ee_pos(qpos) - target)
    print(f"Final distance: {final_dist:.4f}")
    if final_dist < 0.032:
        print("✅ Success!")
    else:
        print("❌ Failed.")

print("Control finished.")