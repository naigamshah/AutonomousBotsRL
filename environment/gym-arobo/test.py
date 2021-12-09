import gym
import gym_arobo
import random 
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.models import load_model

env = gym.make('ARobo-v0')
env.reset()

model = load_model("First.h5")
goal_steps = 10000
score_requirement = -10
initial_games = 100

scores = []
choices = []
for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()

        if len(prev_obs)==0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs)))[0])
            #print("Action: " + str(action))
            #print(prev_obs.shape)

        choices.append(action)
                
        sensors_new, reward, done, info = env.step(action)
        #prev_obs = sensors_new
        game_memory.append([sensors_new, action,info])
        prev_obs = np.asarray([sensors_new['sensorRE'][0],
								sensors_new['sensorDE'][0],
								sensors_new['sensorLE'][0],
								sensors_new['sensorUE'][0],
								sensors_new['sensorURD'][0][0],
								sensors_new['sensorRDD'][0][0],
								sensors_new['sensorDLD'][0][0],
								sensors_new['sensorLUD'][0][0],
								sensors_new['sensorRE'][1],
								sensors_new['sensorDE'][1],
								sensors_new['sensorLE'][1],
								sensors_new['sensorUE'][1],
								sensors_new['sensorURD'][1][0],
								sensors_new['sensorRDD'][1][0],
								sensors_new['sensorDLD'][1][0],
								sensors_new['sensorLUD'][1][0],
								info['State']])
								
        print(prev_obs)
		#input_nn = input_nn.reshape(-1,len(input_nn))
        score+=reward
        if done: 
        	break
	
	scores.append(score)

print('Average Score:',sum(scores)/len(scores))
print(score_requirement)


