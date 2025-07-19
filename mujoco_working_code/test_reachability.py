import numpy as np
from tobor_mujoco_env import ToborEnv

# Instantiate the environment
env = ToborEnv()

# Define a target point (example from your output)
target = np.array([-0.22682046, 0.1616805, 0.192629])  # Example target

# Call the test_reachability function
env.reset()
env.step(np.array([10, 10, 10], dtype=np.float32))
env.step(np.array([20, 20, 20], dtype=np.float32))
env.step(np.array([30, 30, 30], dtype=np.float32))
env.step(np.array([40, 40, 40], dtype=np.float32))
env.step(np.array([30, 30, 30], dtype=np.float32))
env.step(np.array([20, 20, 20], dtype=np.float32))
env.step(np.array([20, 20, 20], dtype=np.float32))
print(np.rad2deg(env.data.qpos)) 

# Optionally, close the environment
env.close()