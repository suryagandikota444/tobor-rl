from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import tobor_mujoco_env  # Registers tobor-v0

class CurriculumCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CurriculumCallback, self).__init__(verbose)
        self.curriculum_stages = [
            (1_000, 0.1),
            (500_000, 0.05),
            (1_000_000, 0.025),
            (2_000_000, 0.01),
            (3_000_000, 0.005),
        ]
        self.current_stage_index = 0

    def _on_step(self) -> bool:
        if self.current_stage_index < len(self.curriculum_stages):
            threshold_timestep, new_threshold = self.curriculum_stages[self.current_stage_index]
            if self.num_timesteps >= threshold_timestep:
                current_thresholds = self.training_env.get_attr("success_threshold")
                if not current_thresholds or current_thresholds[0] > new_threshold:
                    self.training_env.env_method("set_success_threshold", new_threshold)
                    if self.verbose > 0:
                        print(f"Timestep {self.num_timesteps}: Updated success threshold to {new_threshold}")
                    self.current_stage_index += 1
        return True

env = make_vec_env("tobor-v0", n_envs=1, env_kwargs={'render_mode': None})
env = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=0.99)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs", ent_coef=0.005)
curriculum_callback = CurriculumCallback(verbose=1)

print("Starting training with curriculum...")
total_timesteps = 4_000_000
model.learn(total_timesteps=total_timesteps, callback=curriculum_callback)

model.save("ppo_tobor_arm_normalized_curriculum")
env.save("vec_normalize_stats_curriculum.pkl")

print("Training complete.")