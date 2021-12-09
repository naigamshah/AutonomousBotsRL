import gym
import gym_arobo
import random 
import numpy as np
from statistics import median, mean
from collections import Counter
from keras.models import Sequential
from keras.layers import Dense,Dropout

env = gym.make('ARobo-v0')
LR = 1e-3
env.reset()
goal_steps = 1000
score_requirement = -10
initial_games = 1000

def bot_velocity(attrt,x,y):

	x_reach = attrt[0] + 5
	y_reach = attrt[1] + 5
	dir_x = 1 #directions to travel for bot
	dir_y = 1
	dist_x = abs(x_reach - x)
	dist_y = abs(y_reach - y)
	
	if x_reach-x >= 0:
		dir_x = 1
	else:
		dir_x = -1
	
	if y_reach-y >= 0:
		dir_y = 1
	else:
		dir_y = -1 
	
	"""
	ACTIONS:
	
	Nothing ----> 0
	Left -------> 1
	Right ------> 2
	Up ---------> 3
	Down -------> 4
	"""
	
	if dist_x==0:
		if dir_y==1:
			action = 4
		else:
			action = 3
	elif dist_y==0:
		if dir_x==1:
			action = 2
		else:
			action = 1
	else:
		if dist_x>=dist_y:
			if dir_x==1:
				action = 2
			else:
				action = 1
		else:
			if dir_y==1:
				action = 4
			else:
				action = 3
			
	return action

def basic_policy(sensors,info):
	dist_re = sensors['sensorRE'][0]
	dist_de = sensors['sensorDE'][0]
	dist_le = sensors['sensorLE'][0]
	dist_ue = sensors['sensorUE'][0]
	dist_urd = sensors['sensorURD'][0][0]
	dist_rdd = sensors['sensorRDD'][0][0]
	dist_dld = sensors['sensorDLD'][0][0]
	dist_lud = sensors['sensorLUD'][0][0]
	
	dist_t_re = sensors['sensorRE'][1]
	dist_t_de = sensors['sensorDE'][1]
	dist_t_le = sensors['sensorLE'][1]
	dist_t_ue = sensors['sensorUE'][1]
	dist_t_urd = sensors['sensorURD'][1][0]
	dist_t_rdd = sensors['sensorRDD'][1][0]
	dist_t_dld = sensors['sensorDLD'][1][0]
	dist_t_lud = sensors['sensorLUD'][1][0]
	
	obstacle_distances = np.asarray([dist_re,dist_de,dist_le,dist_ue,dist_urd,dist_rdd,dist_dld,dist_lud])
	obstacle_distances_edge = np.asarray([dist_re,dist_de,dist_le,dist_ue])
	obstacle_distances_diagonal = np.asarray([dist_urd,dist_rdd,dist_dld,dist_lud])
	
	min_dist_target = min(abs(dist_t_re),abs(dist_t_de),abs(dist_t_le),abs(dist_t_ue),abs(dist_t_urd),abs(dist_t_lud),abs(dist_t_rdd),abs(dist_t_dld))
	min_dist_obstacle = np.min(obstacle_distances)
	collision_direction = np.argmin(obstacle_distances)
	run_direction = np.argmax(obstacle_distances_edge)
	
	state = info['State']
	
	"""
	STATE INFORMATION:
	
	Finish State(FS)----->4
	Winning State(WS)---->3
	Safe State(SS)------->2
	Non-Safe State(NS)--->1
	Collision State(CS)-->0		
	"""
	"""
	ACTIONS:
	
	Nothing ----> 0
	Left -------> 1
	Right ------> 2
	Up ---------> 3
	Down -------> 4
	"""
	
	toss = random.randrange(0,1000)%2
	action = 0
	if state==3:
		if dist_t_re==min_dist_target:
			action = 2
		elif dist_t_de == min_dist_target:
			action = 4
		elif dist_t_le == min_dist_target:
			action = 1
		elif dist_t_ue == min_dist_target:
			action = 3
		elif dist_t_urd == min_dist_target:
			if toss==0:
				action = 3
			else:
				action = 2
		elif dist_t_rdd == min_dist_target:
			if toss==0:
				action = 2
			else:
				action = 4
		elif dist_t_dld == min_dist_target:
			if toss==0:
				action = 4
			else:
				action = 1
		elif dist_t_lud == min_dist_target:
			if toss==0:
				action = 1
			else:
				action = 3	
	elif state==2:
		if min_dist_target<5000:
			if dist_t_re==min_dist_target:
				action = 2
			elif dist_t_de == min_dist_target:
				action = 4
			elif dist_t_le == min_dist_target:
				action = 1
			elif dist_t_ue == min_dist_target:
				action = 3
			elif dist_t_urd == min_dist_target:
				if toss==0:
					action = 3
				else:
					action = 2
			elif dist_t_rdd == min_dist_target:
				if toss==0:
					action = 2
				else:
					action = 4
			elif dist_t_dld == min_dist_target:
				if toss==0:
					action = 4
				else:
					action = 1
			elif dist_t_lud == min_dist_target:
				if toss==0:
					action = 1
				else:
					action = 3
		else:
			action = bot_velocity(info['Target'],info['Bot'][0],info['Bot'][1])
			#action = random.randrange(0,1000)%4 + 1
	elif state==1:
		if run_direction==0:
			action = 2
		elif run_direction==1:
			action = 4
		elif run_direction==2:
			action = 1
		elif run_direction==3:
			action = 3
	
	return action

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(1000):
            env.render()
            
            action = (random.randrange(0,1000)%2) + 2
            
            sensors, reward, done, info = env.step(action)
            print("-------------------Reward = " + str(reward) + "-------------------")
            if done:
                break
                
#some_random_games_first()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for i in range(initial_games):
        score = 0
        game_memory = []
        prev_sensors_output = []
        start = True
        for _ in range(goal_steps):
            #env.render()
            #action = random.randrange(0,1000)%5
            
            if start:
            	action = random.randrange(0,1000)%2 + 2
            	start = False
            else:
            	action = basic_policy(sensors,info)
            	#print("STATE INFORMATION: " + str(info['State']))
            	#print("Action_taken = " + str(action))
			
            sensors, reward, done, info = env.step(action)
            
            if len(prev_sensors_output) > 0 :
                game_memory.append([prev_sensors_output, action,info])
            prev_sensors_output = sensors
            score+=reward
            if done: 
            	break
            	
        print("----"+str(i)+ ".)-------------Score = " + str(score) + "-------------------")

        if score >= score_requirement and info['State']==4:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 0:
                    output = [1,0,0,0,0]
               	elif data[1] == 1:
                    output = [0,1,0,0,0]
                elif data[1] == 2:
                	output = [0,0,1,0,0]
                elif data[1] == 3:
                	output = [0,0,0,1,0]
                elif data[1] == 4:
                	output = [0,0,0,0,1]
                
                input_nn = np.asarray([data[0]['sensorRE'][0],
                						data[0]['sensorDE'][0],
                						data[0]['sensorLE'][0],
                						data[0]['sensorUE'][0],
                						data[0]['sensorURD'][0][0],
                						data[0]['sensorRDD'][0][0],
                						data[0]['sensorDLD'][0][0],
                						data[0]['sensorLUD'][0][0],
                						data[0]['sensorRE'][1],
                						data[0]['sensorDE'][1],
                						data[0]['sensorLE'][1],
                						data[0]['sensorUE'][1],
                						data[0]['sensorURD'][1][0],
                						data[0]['sensorRDD'][1][0],
                						data[0]['sensorDLD'][1][0],
                						data[0]['sensorLUD'][1][0],
                						data[2]['State']])		
                    
                training_data.append([input_nn, output])

        env.reset()
        scores.append(score)
    
    #training_data_save = np.array(training_data)
    #np.save('saved.npy',training_data_save)
    
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    x = Counter(accepted_scores)
    print(sorted(x.items()))
    
    return training_data
    
training_data = initial_population()

X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]))
print(X.shape)
y = np.array([i[1] for i in training_data])
print(y.shape)

def build_model():
    model = Sequential()
    model.add(Dense(32,input_dim = 17, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(5, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model
    
model1 = build_model()
model1.fit(X,y,validation_split=0.2,epochs=3,batch_size=16,verbose=1)
model1.save("First.h5")
