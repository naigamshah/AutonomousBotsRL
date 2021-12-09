import sys
import gym
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from DDPG import DDPGAgent
from utils import *


env = NormalizedEnv(gym.make("Pendulum-v0"))

agent = DDPGAgent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []
num_episodes = 50
num_steps = 500

for episode in range(num_episodes):
	state = env.reset()
	noise.reset()
	episode_reward = 0

	for step in range(num_steps):
		env.render()
		action = agent.get_action(state)
		action = noise.get_action(action, step)
		new_state, reward, done, _ = env.step(action)
		agent.memory.push(state, action, reward, new_state, done)

		if len(agent.memory) > batch_size:
			agent.update(batch_size)

		state = new_state
		episode_reward += reward

		if done or step==(num_steps-1):
			sys.stdout.write("episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
			break

	rewards.append(reward)
	avg_rewards.append(np.mean(rewards[-10:]))		

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

