import torch
import gym
import time
import random
import numpy as np
from collections import deque
from DQN import DQNAgent
from envs.unityagents import UnityEnvironment

num_episodes = 2000
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 0.99
scores = []
scores_avg_window = 100
solved_score = 14

env = UnityEnvironment(file_name="Banana_Linux/Banana.x86_64")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

action_size = brain.vector_action_space_size
state_size = brain.vector.observation_space_size

agent = Agent(state_size=state_size, action_size=action_size, dqn_type="DQN")

for episode in range(1, num_episodes+1):
	env_info = env.reset(train_mode=True)[brain_name]

	state = env_info.vector_observations[0]

	score = 0

	while True:
		action = agent.act(state, epsilon)

		env_info = env.step(action)[brain_name]

		next_state = env_info.vector_observations[0]
		reward = env_info.rewards[0]
		done = env_info.local_done[0]

		agent.step(state, action, reward, next_state, done)

		state = next_state

		score += reward

		if done:
			break

	scores.append(score)
	average_score = np.mean(scores[episode - min(episode,scores_avg_window):episode+1])
	epsilon = max(min_epsilon, epsilon_decay*epsilon)

	print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, average_score), end="")

	if episode % scores_avg_window == 0:
		print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, average_score))

	if average_score >= solved_score:
		print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, average_score))

		timestr = time.strftime("%Y%m%d-%H%M%S")
		nn_filename = "../models/DQNAgent_Trained_Model_Banana_" + timestr + ".pth"
		torch.save(agent.network.state_dict(), nn_filename)

		scores_filename = "../scores/DQNAgent_scores_Banana_" + timestr + ".csv"
		np.savetxt(scores_filename, scores, delimiter=",")
		break

env.close()