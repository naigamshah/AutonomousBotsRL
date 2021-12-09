import gym
import gym_arobo
import torch
import gym
import time
import random
import numpy as np
from collections import deque
from DQN import DQNAgent
import pygame
import os

#os.environ['SDL_VIDEODRIVER']='dummy'
#pygame.display.set_mode((700,710))
height = 200
width = 200

num_episodes = 300
num_steps = 2000
epsilon = 1.0
min_epsilon = 0.01
epsilon_decay = 5e-3
scores = []
scores_avg_window = 10
solved_score = 4
good_score = 4

env = gym.make("ARobo-v0")

action_size = env.action_space.n
state_size = 21
#print(action_size, state_size)

agent = DQNAgent(state_size=state_size, action_size=action_size, dqn_type="DQN")

for episode in range(1, num_episodes+1):
	sensors,_,_,info = env.reset()
	state = np.asarray(np.asarray([sensors['sensorRE'][0],
										sensors['sensorDE'][0],
										sensors['sensorLE'][0],
										sensors['sensorUE'][0],
										sensors['sensorURD'][0][0],
										sensors['sensorRDD'][0][0],
										sensors['sensorDLD'][0][0],
										sensors['sensorLUD'][0][0],
										sensors['sensorRE'][1],
										sensors['sensorDE'][1],
										sensors['sensorLE'][1],
										sensors['sensorUE'][1],
										sensors['sensorURD'][1][0],
										sensors['sensorRDD'][1][0],
										sensors['sensorDLD'][1][0],
										sensors['sensorLUD'][1][0],
										info['State'],
										info['Bot'][0],
										info['Bot'][1],
										info['Last_Target'][0],
										info['Last_Target'][1]]))
	score = 0
	
	for step in range(num_steps):
		env.render()
		action = agent.act(state, epsilon)

		sensors, reward, done, info = env.step(action)
		next_state = np.asarray(np.asarray([sensors['sensorRE'][0],
										sensors['sensorDE'][0],
										sensors['sensorLE'][0],
										sensors['sensorUE'][0],
										sensors['sensorURD'][0][0],
										sensors['sensorRDD'][0][0],
										sensors['sensorDLD'][0][0],
										sensors['sensorLUD'][0][0],
										sensors['sensorRE'][1],
										sensors['sensorDE'][1],
										sensors['sensorLE'][1],
										sensors['sensorUE'][1],
										sensors['sensorURD'][1][0],
										sensors['sensorRDD'][1][0],
										sensors['sensorDLD'][1][0],
										sensors['sensorLUD'][1][0],
										info['State'],
										info['Bot'][0],
										info['Bot'][1],
										info['Last_Target'][0],
										info['Last_Target'][1]]))

		agent.step(state, action, reward, next_state, done)

		state = next_state

		score += reward

		if done:
			break

	scores.append(score)
	average_score = np.mean(scores[episode - min(episode,scores_avg_window):episode+1])
	epsilon = max(min_epsilon, epsilon-epsilon_decay)

	print('\rEpisode {}\tEpsilon {}\tAverage Score: {:.2f}'.format(episode, epsilon, average_score), end="")

	if episode % scores_avg_window == 0:
		print('\rEpisode {}\tEpsilon {}\tAverage Score: {:.2f}'.format(episode, epsilon, average_score))

	if average_score >= good_score:
		print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, average_score))

		timestr = time.strftime("%Y%m%d-%H%M%S")
		nn_filename = "../models/DQNAgent_Trained_Model_ARobo_(" + str(height) + "x" + (width) + ")_" + timestr + ".pth"
		torch.save(agent.network.state_dict(), nn_filename)

		scores_filename = "../scores/DQNAgent_scores_Arobo_" + timestr + ".csv"
		np.savetxt(scores_filename, scores, delimiter=",")
		break

env.close()