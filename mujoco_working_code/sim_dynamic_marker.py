import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import time
import mujoco
import tobor_mujoco_env  # Registers the environment

# Load the base environment with rendering
base_env = gym.make("tobor-v0", render_mode="human").env.env.env
vec_env = DummyVecEnv([lambda: base_env])

# Load normalization stats and trained model
norm_env = VecNormalize.load("vec_normalize_stats_curriculum_penalize_when_out_threshold_5_deg_rand_start_pos.pkl", vec_env)
# norm_env = VecNormalize.load("vec_normalize_stats_curriculum.pkl", vec_env)
model = PPO.load("ppo_tobor_arm_normalized_curriculum_penalize_when_out_threshold_5_deg_rand_start_pos.zip", env=norm_env)
# model = PPO.load("ppo_tobor_arm_normalized_curriculum.zip", env=norm_env)

# Optional: enforce success criteria used during curriculum
base_env.set_success_threshold(0.005)

# --- Dynamic Sphere Update Setup ---
marker_body_id = base_env.model.body("marker_sphere").id

def update_marker_position(new_pos):
    base_env.model.body_pos[marker_body_id] = new_pos
    mujoco.mj_forward(base_env.model, base_env.data)
# -----------------------------------

# Define 10 targets
targets = np.array([
    [0.18, 0.12, 0.05],
    [0.10, 0.15, 0.15],
    [0.00, 0.18, 0.12],
    [-0.12, 0.14, 0.14],
    [-0.18, 0.10, 0.10],
    [-0.10, 0.06, 0.18],
    [0.00, 0.00, 0.18],
    [0.15, 0.05, 0.14],
    [0.12, 0.18, 0.12],
    [-0.08, 0.10, 0.17],
], dtype=np.float32)

# Loop over each target
for target_idx, target in enumerate(targets):
    print(f"\n=== TARGET {target_idx + 1} / {len(targets)}: {target} ===")

    # if not base_env.is_reachable(target):
    #     print(f"Target {target} is NOT reachable!")
    #     continue

    # Reset and override the target
    norm_env.reset()
    # norm_env.data.qpos[:] = [0,0,0]
    base_env.data.qpos[:] = [0,0,0]
    base_env._target = target.copy()
    update_marker_position(target.copy())

    # Update normalized obs with new target
    raw_obs = np.concatenate([np.rad2deg(base_env.data.qpos), base_env._target])
    obs = norm_env.normalize_obs(np.array([raw_obs]))

    norm_env.render()

    done = False
    total_reward = 0.0
    step_count = 0
    max_steps = 200

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        # print(f"action: {action}")
        obs, reward, dones, infos = norm_env.step(action)
        ee_pos = base_env._get_ee_pos()
        dist = np.linalg.norm(ee_pos - base_env._target)

        print(f"Step {step_count}: Dist={dist:.4f}, Reward={reward[0]:.4f}")

        total_reward += reward[0]
        done = dones[0]
        step_count += 1

        if dist < 0.02:
            done = True

        norm_env.render()
        time.sleep(0.1)

    final_dist = np.linalg.norm(base_env._get_ee_pos() - base_env._target)
    print(f"Final distance: {final_dist:.4f}, Total reward: {total_reward:.2f}")
    if final_dist < 0.0275:
        print("✅ Success!")
    else:
        print("❌ Failed.")
        print(f"\n=== TARGET {target_idx + 1} / {len(targets)}: {target} ===")
        wait = input("waiting...")

# Keep viewer open
print("Press Ctrl+C or close viewer to exit.")
while hasattr(base_env, "_viewer") and base_env._viewer.is_running():
    with base_env._viewer.lock():
        pass
    base_env._viewer.sync()
    time.sleep(0.01)

norm_env.close()