import numpy as np
import time
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

# --- Assume these functions exist and work ---
from simple_fk_environment import simple_fk_environment, THETA0_MAX_ANGLE, THETA0_MIN_ANGLE, THETA1_MAX_ANGLE, THETA1_MIN_ANGLE, THETA2_MAX_ANGLE, THETA2_MIN_ANGLE
from tobor_env import tobor_env
# --- Load Model and Normalization Stats ---
MODEL_PATH = "ppo_robot_arm_normalized_curriculum.zip"
STATS_PATH = "vec_normalize_stats_curriculum.pkl"
ROBOT_API_BASE_URL = "http://192.168.4.1/set_angle"
SEND_COMMANDS = False

vec_env_dummy = make_vec_env(lambda: tobor_env(gui=True), n_envs=1)
vec_normalize_stats = VecNormalize.load(STATS_PATH, vec_env_dummy)

# Not updating stats based on real robot data here
vec_normalize_stats.training = False

# Don't need reward normalization during deployment
vec_normalize_stats.norm_reward = False

model = PPO.load(MODEL_PATH)
first_pred = True
previous_action = [0,0,0]
target_coords = [.28,0,0]
print("target_coords", target_coords)
print("Starting robot control loop...")

# assumes zero difference between predictions and robot movement
try:
    while True:

        # a. Read Current State
        if first_pred:
            current_angles = [0,0,0] # Returns np.array([theta0, theta1, theta2])
            first_pred = False
        else:
            current_angles = previous_action # Returns np.array([theta0, theta1, theta2])

        # raw observation format
        obs_raw = np.concatenate([current_angles, target_coords]).astype(np.float32)

        # reshape observation
        obs_reshaped = obs_raw.reshape(1, -1)

        # normalize observation
        normalized_obs = vec_normalize_stats.normalize_obs(obs_raw)

        # get prediction         
        action_from_model, _ = model.predict(normalized_obs, deterministic=True)                                                
        if hasattr(vec_normalize_stats, 'norm_action') and vec_normalize_stats.norm_action:

            action_unnormalized = vec_normalize_stats.unnormalize_action(action_from_model.reshape(1, -1))
            action = action_unnormalized[0]
        else:

            action = action_from_model
        
        print("action_from_model (potentially normalized)", action_from_model)


        target_angles = current_angles + action

        ROBOT_JOINT_LIMITS_LOW_RAD = np.array([0, 0, 0])
        ROBOT_JOINT_LIMITS_HIGH_RAD = np.array([6.28, 3.14, 2.7]) 

        safe_target_angles0 = np.clip(target_angles[0:1], 0, ROBOT_JOINT_LIMITS_HIGH_RAD[0])
        safe_target_angles1 = np.clip(target_angles[1:2], 0, ROBOT_JOINT_LIMITS_HIGH_RAD[1])
        safe_target_angles2 = np.clip(target_angles[2:3], 0, ROBOT_JOINT_LIMITS_HIGH_RAD[2])

        safe_target_angles = np.concatenate([safe_target_angles0, safe_target_angles1, safe_target_angles2])

        print(f"Sending target angles to robot: {safe_target_angles}")
        if SEND_COMMANDS:
            for i, angle in enumerate(safe_target_angles):
                try:
                    url = f"{ROBOT_API_BASE_URL}?servo={i}&angle={int(round(angle))}"
                    response = requests.get(url, timeout=1)
                    response.raise_for_status()
                    print(f"Successfully sent command for servo {i}: angle {int(round(angle))}. Response: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Error sending command for servo {i}: {e}")

        previous_action = safe_target_angles

        time.sleep(.1)
except KeyboardInterrupt:
    print("Stopping control loop.")

finally:
    print("Cleanup complete.")
