# In a new script or after training in train.py
from simple_fk_environment import simple_fk_environment
from tobor_env import tobor_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from time import sleep

# Load the normalization statistics
# eval_env = make_vec_env(simple_fk_environment, n_envs=1)
MODEL_PATH = "logs/PPO_6/ppo_robot_arm_normalized_curriculum.zip"
STATS_PATH = "logs/PPO_6/vec_normalize_stats_curriculum.pkl"
eval_env = make_vec_env(lambda: tobor_env(gui=True), n_envs=1)
eval_env = VecNormalize.load(STATS_PATH, eval_env)

# We are in evaluation mode, so we don't want to update the statistics
eval_env.training = False
# We want to see the unnormalized rewards, so we set norm_reward to False
eval_env.norm_reward = False 

# Load the trained model
model = PPO.load(MODEL_PATH, env=eval_env)

print("Evaluating trained agent...")
obs = eval_env.reset()

for i in range(5000): # Run for 5000 steps or a few episodes
    action, _states = model.predict(obs, deterministic=True) # Use deterministic=True for evaluation
    obs, rewards, dones, info = eval_env.step(action)

    print(f"Step: {i}, Action: {action}, Reward: {rewards[0]:.2f}, Done: {dones[0]}")
    if hasattr(eval_env.envs[0], 'render'): # If your single env has a render method
        eval_env.envs[0].render()

    if dones[0]:
        print(f"Episode finished after {info[0]['episode']['l']} steps with total reward {info[0]['episode']['r']:.2f}")
        obs = eval_env.reset()
    sleep(.1)
eval_env.close()