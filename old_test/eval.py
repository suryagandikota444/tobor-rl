import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("tobor-v0", render_mode="human")
model = PPO.load("mujoco_tobor_arm")

obs, _ = env.reset()
for _ in range(300):
    action, _ = model.predict(obs)
    obs, reward, done, truncated, _ = env.step(action)
    if done or truncated:
        obs, _ = env.reset()
env.close()