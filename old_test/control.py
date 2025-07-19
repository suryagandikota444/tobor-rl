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
MODEL_PATH = "logs/PPO_6/ppo_robot_arm_normalized_curriculum.zip"
STATS_PATH = "logs/PPO_6/vec_normalize_stats_curriculum.pkl"
ROBOT_API_BASE_URL = "http://192.168.4.1/set_angle"
SEND_COMMANDS = False

vec_env_dummy = make_vec_env(lambda: tobor_env(gui=False), n_envs=1)
vec_normalize_stats = VecNormalize.load(STATS_PATH, vec_env_dummy)

# Not updating stats based on real robot data here
vec_normalize_stats.training = False

# Don't need reward normalization during deployment
vec_normalize_stats.norm_reward = False

model = PPO.load(MODEL_PATH)
first_pred = True
# previous_action should store the last commanded safe angles, as a 1D numpy array
previous_action = np.array([0.0, 0.0, 0.0])
target_coords_1d = np.array([0.27922098, 0.09583922, 0.23196395]) # Keep target as 1D
print("target_coords", target_coords_1d)
print("Starting robot control loop...")

# assumes zero difference between predictions and robot movement
try:
    while True:

        # a. Read Current State
        if first_pred:
            current_angles_1d = np.array([0.0, 0.0, 0.0]) # Current angles as 1D
            first_pred = False
        else:
            current_angles_1d = previous_action # previous_action is 1D

        # Observation for the model: concatenate 1D current angles and 1D target coordinates
        obs_components = [current_angles_1d, target_coords_1d]
        obs_raw_flat = np.concatenate(obs_components).astype(np.float32)
        
        # Reshape for VecNormalize and model prediction (expects batch dimension)
        obs_for_model = obs_raw_flat.reshape(1, -1)

        # normalize observation
        normalized_obs = vec_normalize_stats.normalize_obs(obs_for_model)

        # get prediction
        action_from_model, _ = model.predict(normalized_obs, deterministic=True)
        
        action_scaled = action_from_model # This is likely (1, num_actions) or (num_actions,)

        if hasattr(vec_normalize_stats, 'norm_action') and vec_normalize_stats.norm_action:
            # action_from_model is (1, num_actions) if from predict, unnormalize_action expects (n_envs, n_actions)
            action_unnormalized = vec_normalize_stats.unnormalize_action(action_from_model) 
            action_scaled = action_unnormalized[0] # Get the (num_actions,) array
        else:
            # If not norm_action, action_from_model is already scaled to env's action space.
            # It might be (1, num_actions) or (num_actions,). Ensure it's 1D.
            if action_from_model.ndim > 1:
                action_scaled = action_from_model[0]
            else:
                action_scaled = action_from_model

        print("action_scaled (should be in env's action_space, e.g., [-0.1, 0.1]):", action_scaled)

        # Apply the division if you've confirmed it's necessary due to action space mismatch
        # For now, let's assume action_scaled is what the model intends for the *current* env action space
        # If you find the model was trained with a wider space, this is where adjustment might be needed,
        # OR better, ensure the dummy env for loading matches training.
        # final_action_delta = action_scaled / 2.0 # Your current adjustment
        final_action_delta = action_scaled # Try without /2 first if env action space matches training

        print("final_action_delta:", final_action_delta)

        # Target angles are 1D
        target_angles_1d = current_angles_1d + final_action_delta

        ROBOT_JOINT_LIMITS_LOW_RAD = np.array([0.0, 0.0, 0.0])
        ROBOT_JOINT_LIMITS_HIGH_RAD = np.array([6.28, 3.14, 2.7])

        # Clip 1D target angles correctly
        safe_target_angles_1d = np.clip(target_angles_1d,
                                        ROBOT_JOINT_LIMITS_LOW_RAD,
                                        ROBOT_JOINT_LIMITS_HIGH_RAD)
        
        # Convert safe angles from radians to degrees
        safe_target_angles_deg = np.degrees(safe_target_angles_1d)
        
        print(f"Sending target angles to robot(rad): {safe_target_angles_1d}")
        print(f"Sending target angles to robot(Deg): {safe_target_angles_deg}")
        if SEND_COMMANDS:
            for i, angle in enumerate(safe_target_angles_deg):
                try:
                    url = f"{ROBOT_API_BASE_URL}?servo={i}&angle={int(round(angle))}"
                    response = requests.get(url, timeout=1)
                    response.raise_for_status()
                    print(f"Successfully sent command for servo {i}: angle {int(round(angle))}. Response: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"Error sending command for servo {i}: {e}")

        # Store the 1D array for the next iteration
        previous_action = safe_target_angles_1d

        time.sleep(.1)
except KeyboardInterrupt:
    print("Stopping control loop.")

finally:
    print("Cleanup complete.")
