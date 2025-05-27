from tobor_env import tobor_env # Ensure this points to your modified env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    A custom callback to implement curriculum learning by adjusting
    the environment's success threshold.
    """
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)

        # Define curriculum stages (timesteps, new_threshold)
        self.curriculum_stages = [
            (1_000_000, 0.03, 0.05), 
            (2_000_000, 0.02, 0.04),
        ]
        self.current_stage_index = 0
        print(self.curriculum_stages)

    def _on_step(self) -> bool:

        # Check if it's time to move to the next curriculum stage
        if self.current_stage_index < len(self.curriculum_stages):
            threshold_timestep, new_threshold, zone_threshold = self.curriculum_stages[self.current_stage_index]
            
            if self.num_timesteps >= threshold_timestep:

                # Get current threshold from the environment
                # For VecEnv, get_attr returns a list
                current_env_thresholds = self.training_env.get_attr("current_success_threshold")
                if not current_env_thresholds or current_env_thresholds[0] > new_threshold: # Only update if new is tighter or different
                    self.training_env.env_method("set_success_threshold", new_threshold)
                    self.training_env.env_method("set_zone_threshold", zone_threshold)

                    if self.verbose > 0:
                        print(f"Callback: Timestep {self.num_timesteps}, advancing curriculum.")
                        print(f"Callback: Set success threshold to {new_threshold}")
                    self.current_stage_index += 1
        return True
    
env = make_vec_env(lambda: tobor_env(gui=False), n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True, gamma=0.99)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs", ent_coef=0.01)
curriculum_callback = CurriculumCallback(verbose=1)

# Train the model, now with the callback
print("Starting training with curriculum...")
total_timesteps = 5_000_000
print("total_timesteps = ", total_timesteps)
model.learn(total_timesteps=total_timesteps, callback=curriculum_callback)

# Save the trained model AND the normalization statistics
model.save("ppo_robot_arm_normalized_curriculum")
env.save("vec_normalize_stats_curriculum.pkl")

print("Training complete. Model and normalization stats saved.")
