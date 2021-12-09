import torch
import gym
import time
import random
import numpy as np
from collections import deque
from DQN import DQNAgent

num_episodes = 2000
num_steps = 300
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.99
scores = []
scores_avg_window = 100
solved_score = 195

env = gym.make("CartPole-v0")

action_size = env.action_space.n
state_size = env.observation_space.shape[0]

agent = DQNAgent(state_size=state_size, action_size=action_size, dqn_type="DQN")

for episode in range(1, num_episodes+1):
	state = env.reset()
	score = 0

	for step in range(num_steps):
		#env.render()
		action = agent.act(state, epsilon)

		next_state, reward, done, _ = env.step(action)

		agent.step(state, action, reward, next_state, done)

		state = next_state

		score += reward

		if done:
			break

	scores.append(score)
	average_score = np.mean(scores[episode - min(episode,scores_avg_window):episode+1])
	epsilon = max(min_epsilon, epsilon_decay*epsilon)

	print('\rEpisode {}\tEpsilon {}\tAverage Score: {:.2f}'.format(episode, epsilon, average_score), end="")

	if episode % scores_avg_window == 0:
		print('\rEpisode {}\tEpsilon {}\tAverage Score: {:.2f}'.format(episode, epsilon, average_score))

	if average_score >= solved_score:
		print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, average_score))

		timestr = time.strftime("%Y%m%d-%H%M%S")
		nn_filename = "../models/DQNAgent_Trained_Model_Banana_" + timestr + ".pth"
		torch.save(agent.network.state_dict(), nn_filename)

		scores_filename = "../scores/DQNAgent_scores_Banana_" + timestr + ".csv"
		np.savetxt(scores_filename, scores, delimiter=",")
		break

env.close()