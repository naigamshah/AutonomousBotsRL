import gym
import gym_arobo
import random 
import numpy as np

env = gym.make('ARobo-v0')

totals = []
for episode in range(10):
	episode_rewards = 0
	obs = env.reset()
	action = 1
	for step in range(1000):
		action = env.action_space.sample()
		#action = (random.randrange(0,1000)%5)
		
		env.render()
		sensors, reward, done, info = env.step(action)
		
		#####	Obstacles	#####
		
		print("------------Obstacle-----------------")
		print("Sensor R: " + str(sensors['sensorRE'][0]))
		print("Sensor D: " + str(sensors['sensorDE'][0]))
		print("Sensor L: " + str(sensors['sensorLE'][0]))
		print("Sensor U: " + str(sensors['sensorUE'][0]))
		print("Sensor UR: " + str(sensors['sensorURD'][0][0]))
		print("Sensor UL: " + str(sensors['sensorLUD'][0][0]))
		print("Sensor DR: " + str(sensors['sensorRDD'][0][0]))
		print("Sensor DL: " + str(sensors['sensorDLD'][0][0]))
		
		#####    Target	  ######
		
		print("------------Target-----------------")
		print("Sensor R: " + str(sensors['sensorRE'][1]))
		print("Sensor D: " + str(sensors['sensorDE'][1]))
		print("Sensor L: " + str(sensors['sensorLE'][1]))
		print("Sensor U: " + str(sensors['sensorUE'][1]))
		print("Sensor UR: " + str(sensors['sensorURD'][1][0]))
		print("Sensor UL: " + str(sensors['sensorLUD'][1][0]))
		print("Sensor DR: " + str(sensors['sensorRDD'][1][0]))
		print("Sensor DL: " + str(sensors['sensorDLD'][1][0]))
			
		episode_rewards += reward
		
		if done:
			totals.append(episode_rewards)
			break
			
print(totals)
print('The largest number of time steps covered: ' + str(max(totals)))
