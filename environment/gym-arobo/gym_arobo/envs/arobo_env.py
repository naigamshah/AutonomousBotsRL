import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pygame
import time
import random
import pygame.surfarray as surfarray
from pygame.locals import *
from itertools import cycle
import numpy as np 
from math import sqrt
 
class ARoboEnv(gym.Env):  
	metadata = {'render.modes': ['human']}  
     
	def __init__(self,succ=0,total=0):
		#For full screen make (w,h) = (1366,710)
		#For Half screen make (w,h) = (700,710)
		self.display_width = 200
		self.display_height = 200
		
		pygame.init()
		
		self.max_num_obstacles = 1
		self.max_speed_obstacles = 0
		self.max_speed_target = 0
		self.min_obstacle_width, self.max_obstacle_width = 50, 120
		self.min_obstacle_height, self.max_obstacle_height = 50, 120
 
		self.robot_width = 20 
		self.robot_height = 20
		self.corners = {"bottom_left":[5,(self.display_height - self.robot_height - 5)], 
						"top_left":[5,5],
						"bottom_right":[(self.display_width - self.robot_width - 5),(self.display_height - self.robot_height - 5)],
						"top_right":[(self.display_width - self.robot_width - 5),5]}
		self.random_start_position = random.randint(0,3)
		self.initial_x_bot = list(self.corners.values())[self.random_start_position][0]
		self.initial_y_bot = list(self.corners.values())[self.random_start_position][1]
		self.max_speed_bot = 3
		
		#######################
		###		Regions		###
		#######################
		self.non_safe_radius = 30 #from center of bot
		self.winning_radius = 20  #from center of bot	
		
		self.non_safe_distance_edge = self.non_safe_radius - (min(self.robot_width,self.robot_height)/2)
		self.winning_distance_edge = self.winning_radius - (min(self.robot_width,self.robot_height)/2)
		
		self.non_safe_distance_diagonal = self.non_safe_radius - (sqrt((self.robot_width/2)**2 + (self.robot_height/2)**2)) 
		self.winning_distance_diagonal = self.winning_radius - (sqrt((self.robot_width/2)**2 + (self.robot_height/2)**2))
		
		#######################
		 
		self.black = (0,0,0)
		self.white = (255,255,255)
		self.red = (255,0,0)
		self.green = (0,255,0)
		self.blue = (0,0,255) 

		self.gameDisplay = pygame.display.set_mode((self.display_width,self.display_height))
		pygame.display.set_caption("Robot Simulation Environment")
		self.clock = pygame.time.Clock()
		
		self.robot_image = pygame.image.load('/home/naigam/Desktop/Projects/RL/Autonomous Bots/gym-arobo/gym_arobo/envs/robot.jpeg')


		self.action_space = spaces.Discrete(5)
		self.observation_space = spaces.Box(low=0, high=255, shape=(self.display_height, self.display_width, 3), dtype=np.uint8)

	def step(self, action):
		pygame.event.pump()
		reward = 0
		done = False
		
		#############################
		##### State Information #####
		#############################
		"""
		Finish State(FS)----->4
		Winning State(WS)---->3
		Safe State(SS)------->2
		Non-Safe State(NS)--->1
		Collision State(CS)-->0		
		"""
		old_state = None
		new_state = None
		#############################
		
		#######################################
		###         SENSOR-DATA OLD         ###
		#######################################
		
		sensors_old = self.sensor_detection(self.x,self.y,self.attro,self.attrt)
		
		dist_re = sensors_old['sensorRE'][0]
		dist_de = sensors_old['sensorDE'][0]
		dist_le = sensors_old['sensorLE'][0]
		dist_ue = sensors_old['sensorUE'][0]
		dist_urd = sensors_old['sensorURD'][0][0]
		dist_lud = sensors_old['sensorLUD'][0][0]
		dist_rdd = sensors_old['sensorRDD'][0][0]
		dist_dld = sensors_old['sensorDLD'][0][0]
		
		dist_t_re = sensors_old['sensorRE'][1]
		dist_t_de = sensors_old['sensorDE'][1]
		dist_t_le = sensors_old['sensorLE'][1]
		dist_t_ue = sensors_old['sensorUE'][1]
		dist_t_urd = sensors_old['sensorURD'][1][0]
		dist_t_lud = sensors_old['sensorLUD'][1][0]
		dist_t_rdd = sensors_old['sensorRDD'][1][0]
		dist_t_dld = sensors_old['sensorDLD'][1][0]
		
		de = self.robot_width/2
		dd = sqrt((self.robot_width)**2+(self.robot_height)**2)
		min_dist_old = min(dist_re+de,dist_de+de,dist_le+de,dist_ue+de,dist_urd+dd,dist_lud+dd,dist_rdd+dd,dist_dld+dd)
		
		if (abs(dist_t_re)<=self.winning_distance_edge) or (abs(dist_t_de)<=self.winning_distance_edge) or (abs(dist_t_le)<=self.winning_distance_edge) or (abs(dist_t_ue)<=self.winning_distance_edge) or (abs(dist_t_urd)<=self.winning_distance_diagonal) or (abs(dist_t_lud)<=self.winning_distance_diagonal) or (abs(dist_t_rdd)<=self.winning_distance_diagonal) or (abs(dist_t_dld)<=self.winning_distance_diagonal):
			old_state = 3 #WS
		elif (dist_re<=self.non_safe_distance_edge) or (dist_de<=self.non_safe_distance_edge) or (dist_le<=self.non_safe_distance_edge) or (dist_ue<=self.non_safe_distance_edge) or (dist_urd<=self.non_safe_distance_diagonal) or (dist_lud<=self.non_safe_distance_diagonal) or (dist_rdd<=self.non_safe_distance_diagonal) or (dist_dld<=self.non_safe_distance_diagonal):
			old_state = 1 #NS
		else:
			old_state = 2 #SS
			
		#print("Old state: " + str(old_state))
		#######################################
		
		if action==0: #do nothing
			self.x_change = 0
			self.y_change = 0
		if action==1: #left
			self.x_change = -1*self.max_speed_bot
			self.y_change = 0
		if action==2: #right
			self.x_change = self.max_speed_bot
			self.y_change = 0
		if action==3: #up
			self.y_change = -1*self.max_speed_bot
			self.x_change = 0
		if action==4: #down
			self.y_change = self.max_speed_bot
			self.x_change = 0
		
		self.x += self.x_change
		self.y += self.y_change
			
		####
		self.gameDisplay.fill(self.white)
		
		for i in range(0,len(self.attro)):
			self.obstacles(self.attro[i],self.blue)
			
		self.target(self.attrt,self.green)
		
		self.robot(self.x,self.y)
		self.reach_count(self.success,self.total_attempts)
 		####
 
 
		self.tosst = random.randrange(0,1000) % 4
		if self.tosst==0:
			self.attrt[0] += self.attrt[4]
			self.attrt[1] += self.attrt[5]
		elif self.tosst==1:
			self.attrt[0] -= self.attrt[4]
			self.attrt[1] += self.attrt[5]
		elif self.tosst==2:
			self.attrt[0] += self.attrt[4]
			self.attrt[1] -= self.attrt[5]
		elif self.tosst==3:
			self.attrt[0] -= self.attrt[4]
			self.attrt[1] -= self.attrt[5]
		
		if self.attrt[0] + self.attrt[2] > self.display_width:
			self.attrt[0] = self.display_width - self.attrt[2]
		elif self.attrt[0]<0:
			self.attrt[0] = 0	
		if self.attrt[1] + self.attrt[3] > self.display_height:
			self.attrt[1] = self.display_height - self.attrt[3]
		elif self.attrt[1]<0:
			self.attrt[1]=0
			
		self.tosso = []
		for i in range(len(self.attro)):
			self.tosso.append(random.randrange(0,1000) % 4)
			if self.tosso[i]==0:
				self.attro[i][0] += self.attro[i][4]
				self.attro[i][1] += self.attro[i][5]
			elif self.tosso[i]==1:
				self.attro[i][0] -= self.attro[i][4]
				self.attro[i][1] += self.attro[i][5]
			elif self.tosso[i]==2:
				self.attro[i][0] += self.attro[i][4]
				self.attro[i][1] -= self.attro[i][5]
			elif self.tosso[i]==3:
				self.attro[i][0] -= self.attro[i][4]
				self.attro[i][1] -= self.attro[i][5]
		
		for i in range(0,len(self.attro)):
			if self.attrt[1] > self.attro[i][1] and self.attrt[1] < self.attro[i][1]+self.attro[i][3] or self.attrt[1]+self.attrt[3] > self.attro[i][1] and self.attrt[1]+self.attrt[3] < self.attro[i][1]+self.attro[i][3] :
		
				if self.attrt[0]>self.attro[i][0] and self.attrt[0]<self.attro[i][0]+self.attro[i][2] or self.attrt[0]+self.attrt[2] > self.attro[i][0] and self.attrt[0]+self.attrt[2] < self.attro[i][0]+self.attro[i][2]:
					#print("collision avoided")
					self.avoid_collision_target(self.attrt,self.attro,self.tosst,self.tosso,i)
		
		flagc = 0
		flagw = 0		
		for i in range(0,len(self.attro)):
			if self.y > self.attro[i][1] and self.y < self.attro[i][1]+self.attro[i][3] or self.y+self.robot_height > self.attro[i][1] and self.y+self.robot_height < self.attro[i][1]+self.attro[i][3] :
		
				if self.x>self.attro[i][0] and self.x<self.attro[i][0]+self.attro[i][2] or self.x+self.robot_width > self.attro[i][0] and self.x+self.robot_width < self.attro[i][0]+self.attro[i][2]:
					flagc = 1
					break
					
		if self.y<0 or self.y+self.robot_height>self.display_height or self.x<0 or self.x+self.robot_width>self.display_width:
			flagc=1 
					
		if (self.y > self.attrt[1] and self.y < self.attrt[1]+self.attrt[3]) or (self.y+self.robot_height > self.attrt[1] and self.y+self.robot_height < self.attrt[1]+self.attrt[3]):
				
			if (self.x>self.attrt[0] and self.x<self.attrt[0]+self.attrt[2]) or (self.x+self.robot_width > self.attrt[0] and self.x+self.robot_width < self.attrt[0]+self.attrt[2]):
				flagw = 1
				self.success += 1
		
		#######################################
		###         SENSOR-DATA NEW         ###
		#######################################
		
		sensors_new = self.sensor_detection(self.x,self.y,self.attro,self.attrt)
		
		dist_re = sensors_new['sensorRE'][0]
		dist_de = sensors_new['sensorDE'][0]
		dist_le = sensors_new['sensorLE'][0]
		dist_ue = sensors_new['sensorUE'][0]
		dist_urd = sensors_new['sensorURD'][0][0]
		dist_lud = sensors_new['sensorLUD'][0][0]
		dist_rdd = sensors_new['sensorRDD'][0][0]
		dist_dld = sensors_new['sensorDLD'][0][0]
		
		dist_t_re = sensors_new['sensorRE'][1]
		dist_t_de = sensors_new['sensorDE'][1]
		dist_t_le = sensors_new['sensorLE'][1]
		dist_t_ue = sensors_new['sensorUE'][1]
		dist_t_urd = sensors_new['sensorURD'][1][0]
		dist_t_lud = sensors_new['sensorLUD'][1][0]
		dist_t_rdd = sensors_new['sensorRDD'][1][0]
		dist_t_dld = sensors_new['sensorDLD'][1][0]
		
		min_dist_new = min(dist_re+de,dist_de+de,dist_le+de,dist_ue+de,dist_urd+dd,dist_lud+dd,dist_rdd+dd,dist_dld+dd)
		
		if flagc==1:
			new_state = 0 #CS
		elif flagw==1:
			new_state = 4 #FS
		elif (abs(dist_t_re)<=self.winning_distance_edge) or (abs(dist_t_de)<=self.winning_distance_edge) or (abs(dist_t_le)<=self.winning_distance_edge) or (abs(dist_t_ue)<=self.winning_distance_edge) or (abs(dist_t_urd)<=self.winning_distance_diagonal) or (abs(dist_t_lud)<=self.winning_distance_diagonal) or (abs(dist_t_rdd)<=self.winning_distance_diagonal) or (abs(dist_t_dld)<=self.winning_distance_diagonal):
			new_state = 3 #WS
		elif (dist_re<=self.non_safe_distance_edge) or (dist_de<=self.non_safe_distance_edge) or (dist_le<=self.non_safe_distance_edge) or (dist_ue<=self.non_safe_distance_edge) or (dist_urd<=self.non_safe_distance_diagonal) or (dist_lud<=self.non_safe_distance_diagonal) or (dist_rdd<=self.non_safe_distance_diagonal) or (dist_dld<=self.non_safe_distance_diagonal):
			new_state = 1 #NS
		else:
			new_state = 2 #SS
		
		#print("New state: " + str(new_state))
		#######################################
		####		REWARD FUNCTION			###
		#######################################
		"""
		Finish State(FS)----->4
		Winning State(WS)---->3
		Safe State(SS)------->2
		Non-Safe State(NS)--->1
		Collision State(CS)-->0		
		"""
		"""
		if old_state==2 and new_state==3:   # SS ----> WS
			reward = 2
		elif old_state==3 and new_state==4: # WS ----> FS
			reward = 4
		elif old_state==1 and new_state==2: # NS ----> SS
			reward = 1
		elif old_state==2 and new_state==1: # SS ----> NS
			reward = -2
		elif old_state==1 and new_state==1 and (min_dist_new < min_dist_old): # NS ----> NS
			reward = -1
		elif old_state==1 and new_state==1 and (min_dist_new >= min_dist_old): # NS ----> NS
			reward = 0
		elif old_state==1 and new_state==0: # NS ----> CS
			reward = -4
		elif old_state==2 and new_state==2: # SS ----> SS
			reward = 0
		elif old_state==3 and new_state==1: # WS ----> NS
			reward = -2
		elif old_state==3 and new_state==0: # WS ----> CS
			reward = -3
		elif old_state==3 and new_state==2: # WS ----> SS
			reward = -1
		else: # any other scenario
			reward = 0
		"""
		if old_state==2 and new_state==3:   # SS ----> WS
			reward = 0
		elif old_state==3 and new_state==4: # WS ----> FS
			reward = 4
		elif old_state==1 and new_state==2: # NS ----> SS
			reward = 0
		elif old_state==2 and new_state==1: # SS ----> NS
			reward = -2
		elif old_state==1 and new_state==1 and (min_dist_new < min_dist_old): # NS ----> NS
			reward = -1
		elif old_state==1 and new_state==1 and (min_dist_new >= min_dist_old): # NS ----> NS
			reward = 0
		elif old_state==1 and new_state==0: # NS ----> CS
			reward = -4
		elif old_state==2 and new_state==2: # SS ----> SS
			reward = 0
		elif old_state==3 and new_state==1: # WS ----> NS
			reward = -2
		elif old_state==3 and new_state==0: # WS ----> CS
			reward = -3
		elif old_state==3 and new_state==2: # WS ----> SS
			reward = -1
		else: # any other scenario
			reward = 0

		
		#######################################
		self.sensors = sensors_new
		#####	Obstacles	#####
		"""
		print("------------Obstacle-----------------")
		print("Sensor R: " + str(self.sensors['sensorRE'][0]))
		print("Sensor D: " + str(self.sensors['sensorDE'][0]))
		print("Sensor L: " + str(self.sensors['sensorLE'][0]))
		print("Sensor U: " + str(self.sensors['sensorUE'][0]))
		print("Sensor UR: " + str(self.sensors['sensorURD'][0][0]))
		print("Sensor UL: " + str(self.sensors['sensorLUD'][0][0]))
		print("Sensor DR: " + str(self.sensors['sensorRDD'][0][0]))
		print("Sensor DL: " + str(self.sensors['sensorDLD'][0][0]))
		"""
		#####    Target	  ######
		"""
		print("------------Target-----------------")
		print("Sensor R: " + str(self.sensors['sensorRE'][1]))
		print("Sensor D: " + str(self.sensors['sensorDE'][1]))
		print("Sensor L: " + str(self.sensors['sensorLE'][1]))
		print("Sensor U: " + str(self.sensors['sensorUE'][1]))
		print("Sensor UR: " + str(self.sensors['sensorURD'][1][0]))
		print("Sensor UL: " + str(self.sensors['sensorLUD'][1][0]))
		print("Sensor DR: " + str(self.sensors['sensorRDD'][1][0]))
		print("Sensor DL: " + str(self.sensors['sensorDLD'][1][0]))
		"""		
		#image_data = pygame.surfarray.array3d(pygame.display.get_surface())
		
		#return image_data, reward, done,{}
		
		if flagw==1:
			self.total_attempts +=1
			self.won(self.success,self.total_attempts)
			done = True
		if flagc==1:
			self.total_attempts +=1
			self.collision(self.success,self.total_attempts)
			done = True
		
		min_dist_target = min(abs(dist_t_re),abs(dist_t_de),abs(dist_t_le),abs(dist_t_ue),abs(dist_t_urd),abs(dist_t_lud),abs(dist_t_rdd),abs(dist_t_dld))
		
		if min_dist_target<5000:
			if abs(dist_t_re)==min_dist_target:
				self.last_target_x = self.x + self.robot_width + abs(dist_t_re)
				self.last_target_y = self.y + (self.robot_height/2)
			if abs(dist_t_de)==min_dist_target:
				self.last_target_x = self.x + (self.robot_width/2)
				self.last_target_y = self.y + (self.robot_height) + abs(dist_t_de)
			if abs(dist_t_le)==min_dist_target:
				self.last_target_x = self.x - abs(dist_t_le)
				self.last_target_y = self.y + (self.robot_height/2) 
			if abs(dist_t_ue)==min_dist_target:
				self.last_target_x = self.x + (self.robot_width/2)
				self.last_target_y = self.y - abs(dist_t_ue)
			if abs(dist_t_urd)==min_dist_target:
				self.last_target_x = self.x + (self.robot_width) + (abs(dist_t_urd)/sqrt(2))
				self.last_target_y = self.y - (abs(dist_t_urd)/sqrt(2))
			if abs(dist_t_lud)==min_dist_target:
				self.last_target_x = self.x - (abs(dist_t_lud)/sqrt(2))
				self.last_target_y = self.y - (abs(dist_t_urd)/sqrt(2))
			if abs(dist_t_rdd)==min_dist_target:
				self.last_target_x = self.x + (self.robot_width) + (abs(dist_t_rdd)/sqrt(2))
				self.last_target_y = self.y + (self.robot_height) + (abs(dist_t_rdd)/sqrt(2))
			if abs(dist_t_dld)==min_dist_target:
				self.last_target_x = self.x - (abs(dist_t_dld)/sqrt(2))
				self.last_target_y = self.y + (self.robot_height) + (abs(dist_t_dld)/sqrt(2))
		
		return self.sensors, reward, done,{'State': new_state,'Bot':[self.x,self.y],'Target':self.attrt,'Last_Target':[self.last_target_x,self.last_target_y]}
 	
	def reset(self,succ=0,total=0):
		self.random_start_position = random.randint(0,3)
		self.initial_x_bot = list(self.corners.values())[self.random_start_position][0]
		self.initial_y_bot = list(self.corners.values())[self.random_start_position][1]
		self.x = self.initial_x_bot
		self.y = self.initial_y_bot
		
		self.x_change = 0
		self.y_change = 0
		
		self.total_attempts = total
		self.success = succ
		
		####Gear Box####
		#gear = 0
		
		self.targetx = random.randrange(40,self.display_width-30) 
		self.targety = random.randrange(40,self.display_height-30)
		self.targetw = 30
		self.targeth = 30
		self.target_speedx = random.randrange(0,self.max_speed_target+1) #unlock these to give
		self.target_speedy = random.randrange(0,self.max_speed_target+1) #target some velocity
		#self.target_speedx = 0
		#self.target_speedy = 0
		self.attrt = [self.targetx,self.targety,self.targetw,self.targeth,self.target_speedx,self.target_speedy]
		
		self.last_target_x = self.attrt[0]
		self.last_target_y = self.attrt[1]
		
		self.attro = []
		for i in range(0,random.randrange(1,self.max_num_obstacles+1)):
			flag_obs = 0
			while flag_obs==0:
				self.obstaclex = random.randrange(40,self.display_width)
				self.obstacley = random.randrange(40,self.display_height)
				self.obstaclew = random.randrange(self.min_obstacle_width,self.max_obstacle_width)
				self.obstacleh = random.randrange(self.min_obstacle_height,self.max_obstacle_height)
				if (self.targetx + self.targetw) < self.obstaclex or self.targetx > (self.obstaclex + self.obstaclew) or (self.targety + self.targeth) < self.obstacley or self.targety > (self.obstacley + self.obstacleh):
					flag_obs = 1 
					  
			self.obstacle_speedx =random.randrange(0,self.max_speed_obstacles+1)#unlock these to give
			self.obstacle_speedy =random.randrange(0,self.max_speed_obstacles+1)#obstacle some velocity
			#self.obstacle_speedx = 0
			#self.obstacle_speedy = 0
			self.attro.append([self.obstaclex,self.obstacley,self.obstaclew,self.obstacleh,self.obstacle_speedx,self.obstacle_speedy])
		 
		#image_data,_,_,_ = self.step(0)
		sensors, reward, done, info = self.step(0)
		
		#return image_data
		return sensors, reward, done, info
	 
	def render(self, mode='human', close=False):
		pygame.display.update()
		self.clock.tick(60)
    
	def sensor_detection(self,x,y,attro,attrt):
		center_x = x + (self.robot_width/2)
		center_y = y + (self.robot_height/2)
		
		ser_x,ser_y = center_x + (self.robot_width/2), center_y 
		sed_x,sed_y = center_x, center_y + (self.robot_height/2)
		sel_x,sel_y = center_x - (self.robot_width/2), center_y
		seu_x,seu_y = center_x, center_y - (self.robot_height/2)
		
		sdur_x, sdur_y = x + self.robot_width, y
		sdrd_x, sdrd_y = x + self.robot_width, y + self.robot_height
		sddl_x, sddl_y = x, y + self.robot_height
		sdlu_x, sdlu_y = x, y
		
		#Initialize distances
		dist_ser = self.display_width - ser_x
		dist_sed = self.display_height - sed_y
		dist_sel = sel_x
		dist_seu = seu_y
		dist_sdur_x, dist_sdur_y = self.display_width - sdur_x, sdur_y 
		dist_sdrd_x, dist_sdrd_y = self.display_width - sdrd_x, self.display_height - sdrd_y
		dist_sddl_x, dist_sddl_y = sddl_x, self.display_height - sddl_y
		dist_sdlu_x, dist_sdlu_y = sdlu_x, sdlu_y
		
		dist_sdur = min(self.display_width - sdur_x, sdur_y) * sqrt(2)
		dist_sdrd = min(self.display_width - sdrd_x, self.display_height - sdrd_y) * sqrt(2)
		dist_sddl = min(sddl_x, self.display_height - sddl_y) * sqrt(2)
		dist_sdlu = min(sdlu_x, sdlu_y) * sqrt(2)
		
		#Obstacle distance calulation
		for i in range(len(attro)):
			#RightEdge sensor
			if attro[i][0] >= ser_x and ser_y >= attro[i][1] and ser_y <= (attro[i][1]+attro[i][3]):
				if dist_ser >= (attro[i][0]-ser_x):
					dist_ser = attro[i][0]-ser_x
			
			#DownEdge sensor
			if attro[i][1] >= sed_y and sed_x >= attro[i][0] and sed_x <= (attro[i][0]+attro[i][2]):
				if dist_sed >= (attro[i][1]-sed_y):
					dist_sed = attro[i][1]-sed_y
					
			#LeftEdge sensor
			if attro[i][0] <= sel_x and ser_y >= attro[i][1] and ser_y <= (attro[i][1]+attro[i][3]):
				if dist_sel >= (sel_x-(attro[i][0]+attro[i][2])):
					dist_sel = sel_x-(attro[i][0] + attro[i][2])
					
			#UpEdge sensor
			if attro[i][1] <= seu_y and seu_x >= attro[i][0] and seu_x <= (attro[i][0]+attro[i][2]):
				if dist_seu >= (seu_y-(attro[i][1]+attro[i][3])):
					dist_seu = seu_y-(attro[i][1]+attro[i][3])
					
			#UpRightDiagonal sensor
			if (attro[i][0]+attro[i][2])>=sdur_x and attro[i][1]<=sdur_y:
				for xcord in range(attro[i][0],attro[i][0]+attro[i][2]+1):
					ycord = attro[i][1] + attro[i][3]
					if (xcord-sdur_x)==(sdur_y-ycord):
						dist = sqrt((xcord-sdur_x)**2 + (sdur_y-ycord)**2)
						if dist_sdur>=dist:
							dist_sdur = dist
							dist_sdur_x,dist_sdur_y = (xcord-sdur_x),(sdur_y-ycord)
							
				for ycord in range(attro[i][1],attro[i][1]+attro[i][3]+1):
					xcord = attro[i][0]
					if (xcord-sdur_x)==(sdur_y-ycord):
						dist = sqrt((xcord-sdur_x)**2 + (sdur_y-ycord)**2)
						if dist_sdur>=dist:
							dist_sdur = dist
							dist_sdur_x,dist_sdur_y = (xcord-sdur_x),(sdur_y-ycord)
							
			#RightDownDiagonal sensor
			if (attro[i][0]+attro[i][2])>=sdrd_x and (attro[i][1]+attro[i][3])>=sdrd_y:
				for xcord in range(attro[i][0],attro[i][0]+attro[i][2]+1):
					ycord = attro[i][1]
					if (xcord-sdrd_x)==(ycord-sdrd_y):
						dist = sqrt((xcord-sdrd_x)**2 + (ycord-sdrd_y)**2)
						if dist_sdrd>=dist:
							dist_sdrd = dist
							dist_sdrd_x,dist_sdrd_y = (xcord-sdrd_x),(ycord-sdrd_y)
							
				for ycord in range(attro[i][1],attro[i][1]+attro[i][3]+1):
					xcord = attro[i][0]
					if (xcord-sdrd_x)==(ycord-sdrd_y):
						dist = sqrt((xcord-sdrd_x)**2 + (ycord-sdrd_y)**2)
						if dist_sdrd>=dist:
							dist_sdrd = dist
							dist_sdrd_x,dist_sdrd_y = (xcord-sdrd_x),(ycord-sdrd_y)
							
			#DownLeftDiagonal sensor
			if attro[i][0]<=sddl_x and (attro[i][1]+attro[i][3])>=sddl_y:
				for xcord in range(attro[i][0],attro[i][0]+attro[i][2]+1):
					ycord = attro[i][1]
					if (sddl_x-xcord)==(ycord-sddl_y):
						dist = sqrt((sddl_x-xcord)**2 + (ycord-sddl_y)**2)
						if dist_sddl>=dist:
							dist_sddl = dist
							dist_sddl_x,dist_sddl_y = (sddl_x-xcord),(ycord-sddl_y)
							
				for ycord in range(attro[i][1],attro[i][1]+attro[i][3]+1):
					xcord = attro[i][0] + attro[i][2]
					if (sddl_x-xcord)==(ycord-sddl_y):
						dist = sqrt((sddl_x-xcord)**2 + (ycord-sddl_y)**2)
						if dist_sddl>=dist:
							dist_sddl = dist
							dist_sddl_x,dist_sddl_y = (sddl_x-xcord),(ycord-sddl_y)
							
			#LeftUpDiagonal sensor
			if attro[i][0]<=sdlu_x and attro[i][1]<=sdlu_y:
				for xcord in range(attro[i][0],attro[i][0]+attro[i][2]+1):
					ycord = attro[i][1] + attro[i][3]
					if (sdlu_x-xcord)==(sdlu_y-ycord):
						dist = sqrt((sdlu_x-xcord)**2 + (sdlu_y-ycord)**2)
						if dist_sdlu>=dist:
							dist_sdlu = dist
							dist_sdlu_x,dist_sdlu_y = (sdlu_x-xcord),(sdlu_y-ycord)
							
				for ycord in range(attro[i][1],attro[i][1]+attro[i][3]+1):
					xcord = attro[i][0] + attro[i][2]
					if (sdlu_x-xcord)==(sdlu_y-ycord):
						dist = sqrt((sdlu_x-xcord)**2 + (sdlu_y-ycord)**2)
						if dist_sdlu>=dist:
							dist_sdlu = dist
							dist_sdlu_x,dist_sdlu_y = (sdlu_x-xcord),(sdlu_y-ycord)
							
							
		#Target distance calulation
		dist_t_ser = -10000
		dist_t_sed = -10000
		dist_t_sel = -10000
		dist_t_seu = -10000
		dist_t_sdur_x, dist_t_sdur_y = -10000, -10000 
		dist_t_sdrd_x, dist_t_sdrd_y = -10000, -10000
		dist_t_sddl_x, dist_t_sddl_y = -10000, -10000
		dist_t_sdlu_x, dist_t_sdlu_y = -10000, -10000
		
		dist_t_sdur = -10000
		dist_t_sdrd = -10000
		dist_t_sddl = -10000
		dist_t_sdlu = -10000
		
		#RightEdge sensor
		if attrt[0] >= ser_x and ser_y >= attrt[1] and ser_y <= (attrt[1]+attrt[3]):
			if abs(dist_t_ser) >= (attrt[0]-ser_x):
				dist_t_ser = attrt[0]-ser_x
		
		#DownEdge sensor
		if attrt[1] >= sed_y and sed_x >= attrt[0] and sed_x <= (attrt[0]+attrt[2]):
			if abs(dist_t_sed) >= (attrt[1]-sed_y):
				dist_t_sed = attrt[1]-sed_y
				
		#LeftEdge sensor
		if attrt[0] <= sel_x and ser_y >= attrt[1] and ser_y <= (attrt[1]+attrt[3]):
			if abs(dist_t_sel) >= (sel_x-(attrt[0]+attrt[2])):
				dist_t_sel = sel_x-(attrt[0]+attrt[2]) 
				
		#UpEdge sensor
		if attrt[1] <= seu_y and seu_x >= attrt[0] and seu_x <= (attrt[0]+attrt[2]):
			if abs(dist_t_seu) >= (seu_y-(attrt[1]+attrt[3])):
				dist_t_seu = seu_y-(attrt[1]+attrt[3])					
							
		#UpRightDiagonal sensor
		if (attrt[0]+attrt[2])>=sdur_x and attrt[1]<=sdur_y:
			for xcord in range(attrt[0],attrt[0]+attrt[2]+1):
				ycord = attrt[1] + attrt[3]
				if (xcord-sdur_x)==(sdur_y-ycord):
					dist = sqrt((xcord-sdur_x)**2 + (sdur_y-ycord)**2)
					if abs(dist_t_sdur)>=dist:
						dist_t_sdur = dist
						dist_t_sdur_x,dist_t_sdur_y = (xcord-sdur_x),(sdur_y-ycord)
						
			for ycord in range(attrt[1],attrt[1]+attrt[3]+1):
				xcord = attrt[0]
				if (xcord-sdur_x)==(sdur_y-ycord):
					dist = sqrt((xcord-sdur_x)**2 + (sdur_y-ycord)**2)
					if abs(dist_t_sdur)>=dist:
						dist_t_sdur = dist
						dist_t_sdur_x,dist_t_sdur_y = (xcord-sdur_x),(sdur_y-ycord)
						
		#RightDownDiagonal sensor
		if (attrt[0]+attrt[2])>=sdrd_x and (attrt[1]+attrt[3])>=sdrd_y:
			for xcord in range(attrt[0],attrt[0]+attrt[2]+1):
				ycord = attrt[1]
				if (xcord-sdrd_x)==(ycord-sdrd_y):
					dist = sqrt((xcord-sdrd_x)**2 + (ycord-sdrd_y)**2)
					if abs(dist_t_sdrd)>=dist:
						dist_t_sdrd = dist
						dist_t_sdrd_x,dist_t_sdrd_y = (xcord-sdrd_x),(ycord-sdrd_y)
						
			for ycord in range(attrt[1],attrt[1]+attrt[3]+1):
				xcord = attrt[0]
				if (xcord-sdrd_x)==(ycord-sdrd_y):
					dist = sqrt((xcord-sdrd_x)**2 + (ycord-sdrd_y)**2)
					if abs(dist_t_sdrd)>=dist:
						dist_t_sdrd = dist
						dist_t_sdrd_x,dist_t_sdrd_y = (xcord-sdrd_x),(ycord-sdrd_y)
						
		#DownLeftDiagonal sensor
		if attrt[0]<=sddl_x and (attrt[1]+attrt[3])>=sddl_y:
			for xcord in range(attrt[0],attrt[0]+attrt[2]+1):
				ycord = attrt[1]
				if (sddl_x-xcord)==(ycord-sddl_y):
					dist = sqrt((sddl_x-xcord)**2 + (ycord-sddl_y)**2)
					if abs(dist_t_sddl)>=dist:
						dist_t_sddl = dist
						dist_t_sddl_x,dist_t_sddl_y = (sddl_x-xcord),(ycord-sddl_y)
						
			for ycord in range(attrt[1],attrt[1]+attrt[3]+1):
				xcord = attrt[0] + attrt[2]
				if (sddl_x-xcord)==(ycord-sddl_y):
					dist = sqrt((sddl_x-xcord)**2 + (ycord-sddl_y)**2)
					if abs(dist_t_sddl)>=dist:
						dist_t_sddl = dist
						dist_t_sddl_x,dist_t_sddl_y = (sddl_x-xcord),(ycord-sddl_y)
						
		#LeftUpDiagonal sensor
		if attrt[0]<=sdlu_x and attrt[1]<=sdlu_y:
			for xcord in range(attrt[0],attrt[0]+attrt[2]+1):
				ycord = attrt[1] + attrt[3]
				if (sdlu_x-xcord)==(sdlu_y-ycord):
					dist = sqrt((sdlu_x-xcord)**2 + (sdlu_y-ycord)**2)
					if abs(dist_t_sdlu)>=dist:
						dist_t_sdlu = dist
						dist_t_sdlu_x,dist_t_sdlu_y = (sdlu_x-xcord),(sdlu_y-ycord)
						
			for ycord in range(attrt[1],attrt[1]+attrt[3]+1):
				xcord = attrt[0] + attrt[2]
				if (sdlu_x-xcord)==(sdlu_y-ycord):
					dist = sqrt((sdlu_x-xcord)**2 + (sdlu_y-ycord)**2)
					if abs(dist_t_sdlu)>=dist:
						dist_t_sdlu = dist
						dist_t_sdlu_x,dist_t_sdlu_y = (sdlu_x-xcord),(sdlu_y-ycord)
							
		
		######################################
		### Obstacle Blocks view of target ###   
		######################################
		
		if dist_t_ser>dist_ser:
			dist_t_ser = -5000
		if dist_t_sed>dist_sed:
			dist_t_sed = -5000
		if dist_t_sel>dist_sel:
			dist_t_sel = -5000
		if dist_t_seu>dist_seu:
			dist_t_seu = -5000
		if dist_t_sdur>dist_sdur:
			dist_t_sdur = -5000
			dist_t_sdur_x, dist_t_sdur_y = -5000, -5000
		if dist_t_sdrd>dist_sdrd:
			dist_t_sdrd = -5000
			dist_t_sdrd_x, dist_t_sdrd_y = -5000, -5000
		if dist_t_sddl>dist_sddl:
			dist_t_sddl = -5000
			dist_t_sddl_x, dist_t_sddl_y = -5000, -5000
		if dist_t_sdlu>dist_sdlu:
			dist_t_sdlu = -5000
			dist_t_sdlu_x, dist_t_sdlu_y = -5000, -5000
		######################################
		
		
		sensor_re = [dist_ser,dist_t_ser]
		sensor_de = [dist_sed,dist_t_sed]
		sensor_le = [dist_sel,dist_t_sel]
		sensor_ue = [dist_seu,dist_t_seu]
		sensor_urd = [[dist_sdur,[dist_sdur_x,dist_sdur_y]],[dist_t_sdur,[dist_t_sdur_x,dist_t_sdur_y]]]
		sensor_rdd = [[dist_sdrd,[dist_sdrd_x,dist_sdrd_y]],[dist_t_sdrd,[dist_t_sdrd_x,dist_t_sdrd_y]]]
		sensor_dld = [[dist_sddl,[dist_sddl_x,dist_sddl_y]],[dist_t_sddl,[dist_t_sddl_x,dist_t_sddl_y]]]
		sensor_lud = [[dist_sdlu,[dist_sdlu_x,dist_sdlu_y]],[dist_t_sdlu,[dist_t_sdlu_x,dist_t_sdlu_y]]]
		
		sensors = {'sensorRE': sensor_re,
		'sensorDE': sensor_de,
		'sensorLE': sensor_le,
		'sensorUE': sensor_ue,
		'sensorURD': sensor_urd,
		'sensorRDD': sensor_rdd,
		'sensorDLD': sensor_dld,
		'sensorLUD': sensor_lud				
		}
							
		return sensors
        
	def avoid_collision_target(self,attrt,attro,tosst,tosso,i):
		if tosst==0:
			if tosso[i]==0:
				attrt[0] = attrt[0] - attrt[4] + attro[i][4] 
				attrt[1] = attrt[1] - attrt[5] + attro[i][5]
			elif tosso[i]==1:
				attrt[0] = attrt[0] - attrt[4] - attro[i][4]
				attrt[1] = attrt[1] - attrt[5] + attro[i][5]
			elif tosso[i]==2:
				attrt[0] = attrt[0] - attrt[4] + attro[i][4]
				attrt[1] = attrt[1] - attrt[5] - attro[i][5]
			elif tosso[i]==3:
				attrt[0] = attrt[0] - attrt[4] - attro[i][4]
				attrt[1] = attrt[1] - attrt[5] - attro[i][5]
		elif tosst==1:
			if tosso[i]==0:
				attrt[0] = attrt[0] + attrt[4] + attro[i][4]
				attrt[1] = attrt[1] - attrt[5] + attro[i][5]
			elif tosso[i]==1:
				attrt[0] = attrt[0] + attrt[4] - attro[i][4]
				attrt[1] = attrt[1] - attrt[5] + attro[i][5]
			elif tosso[i]==2:
				attrt[0] = attrt[0] + attrt[4] + attro[i][4]
				attrt[1] = attrt[1] - attrt[5] - attro[i][5]
			elif tosso[i]==3:
				attrt[0] = attrt[0] + attrt[4] - attro[i][4]
				attrt[1] = attrt[1] - attrt[5] - attro[i][5]
		elif tosst==2:
			if tosso[i]==0:
				attrt[0] = attrt[0] - attrt[4] + attro[i][4]
				attrt[1] = attrt[1] + attrt[5] + attro[i][5]
			elif tosso[i]==1:
				attrt[0] = attrt[0] - attrt[4] - attro[i][4]
				attrt[1] = attrt[1] + attrt[5] + attro[i][5]
			elif tosso[i]==2:
				attrt[0] = attrt[0] - attrt[4] + attro[i][4]
				attrt[1] = attrt[1] + attrt[5] - attro[i][5]
			elif tosso[i]==3:
				attrt[0] = attrt[0] - attrt[4] - attro[i][4]
				attrt[1] = attrt[1] + attrt[5] - attro[i][5]
		elif tosst==3:
			if tosso[i]==0:
				attrt[0] = attrt[0] + attrt[4] + attro[i][4]
				attrt[1] = attrt[1] + attrt[5] + attro[i][5]
			elif tosso[i]==1:
				attrt[0] = attrt[0] + attrt[4] - attro[i][4]
				attrt[1] = attrt[1] + attrt[5] + attro[i][5]
			elif tosso[i]==2:
				attrt[0] = attrt[0] + attrt[4] + attro[i][4]
				attrt[1] = attrt[1] + attrt[5] - attro[i][5]
			elif tosso[i]==3:
				attrt[0] = attrt[0] + attrt[4] - attro[i][4]
				attrt[1] = attrt[1] + attrt[5] - attro[i][5]

	def bot_velocity(self,attrt,x,y):

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
		
		if dist_x==0:
			x_change = 0
			y_change = dir_y * self.max_speed_bot
		elif dist_y==0:
			x_change = dir_x * self.max_speed_bot
			y_change = 0
		else:
			if dist_x <= dist_y:
				ratio = float(dist_x)/float(dist_y)
				min_r = 2
				for i in range(self.max_speed_bot,0,-1):
					for j in range(i,0,-1):
						r = float(j)/float(i)
						if abs(r - ratio) < min_r:
							x_change = dir_x * j
							y_change = dir_y * i
							min_r = abs(r - ratio)
						if min_r==0:
							break 
			else:
				ratio = float(dist_y)/float(dist_x)
				min_r = 2
				for i in range(self.max_speed_bot,0,-1):
					for j in range(i,0,-1):
						r = float(j)/float(i)
						if abs(r - ratio) < min_r:
							x_change = dir_x * i
							y_change = dir_y * j
							min_r = abs(r - ratio)
						if min_r==0:
							break 
				
		return x_change,y_change

	def reach_count(self,count,total):
		font = pygame.font.SysFont(None, 25)
		text = font.render("Successfull attempts: " + str(count) + "/" + str(total), True, self.black)
		self.gameDisplay.blit(text,(0,0))

	def text_objects(self,text,font):
		textSurface = font.render(text,True,self.black)
		return textSurface,textSurface.get_rect()

	def message_display(self,text,succ,total):
		largeText = pygame.font.Font('freesansbold.ttf',30)
		TextSurf, TextRect = self.text_objects(text,largeText)
		TextRect.center = ((self.display_width/2),self.display_height/2)
		self.gameDisplay.blit(TextSurf,TextRect)
		
		pygame.display.update()
		
		time.sleep(2)
		#self.__init__(succ,total)
		#sim_env(succ,total) #############################
		
	def won(self,succ,total):
		self.message_display("You have reached the target!!",succ,total)

	def collision(self,succ,total):
		self.message_display("Bot collided with an Obstacle!",succ,total)

	def robot(self,x,y):
		self.gameDisplay.blit(self.robot_image,(x,y))
		
	def target(self,attr,color):
		pygame.draw.rect(self.gameDisplay,color,attr[0:4])
		
	def obstacles(self,attr,color):
		pygame.draw.rect(self.gameDisplay,color,attr[:][0:4])
