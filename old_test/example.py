#Importing OpenAI gym package and MuJoCo engine
import gymnasium as gym
import mujoco
#Setting MountainCar-v0 as the environment
env = gym.make('MountainCar-v0', render_mode="human")
#Sets an initial state
env.reset()
# Rendering our instance 300 times
for _ in range(300):
  #renders the environment
  env.render()
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  env.step(env.action_space.sample())
env.close()