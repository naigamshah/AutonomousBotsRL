import sys
import gym
import torch
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from A2C import A2CAgent
from utils import *

env = gym.make("CartPole-v1")
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.n

agent = A2CAgent(env)

all_lengths = []
average_lengths = []
all_rewards = []
entropy_term = 0
num_episodes = 1000
num_steps = 300

for episode in range(num_episodes):
	log_probs = []
	values = []
	rewards = []

	state = env.reset()
	for step in range(num_steps):
		env.render()
		value, policy_dist = agent.get_action(state)
		dist = policy_dist.detach().numpy()

		action = np.random.choice(num_outputs, p=np.squeeze(dist))
		log_prob = torch.log(policy_dist.squeeze(0)[action])
		entropy = -np.sum(np.mean(dist) * np.log(dist))
		new_state, reward, done, _ = env.step(action)

		rewards.append(reward)
		values.append(value)
		log_probs.append(log_prob)
		entropy_term += entropy
		state = new_state

		if done or step==(num_steps-1):
			Qval, _ = agent.get_action(new_state)
			all_rewards.append(np.sum(reward))
			all_lengths.append(step)
			average_lengths.append(np.mean(all_lengths[-10:]))
			if episode%10==0:
				sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode, np.sum(rewards), step, average_lengths[-1]))
			break

	agent.update(log_probs, values, rewards, Qval, entropy_term)

# Plot results
smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
smoothed_rewards = [elem for elem in smoothed_rewards]
plt.plot(all_rewards)
plt.plot(smoothed_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

plt.plot(all_lengths)
plt.plot(average_lengths)
plt.xlabel('Episode')
plt.ylabel('Episode length')
plt.show()