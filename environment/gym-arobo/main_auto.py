import pygame
import time
import random
from math import sqrt

pygame.init()

#For full screen make (w,h) = (1366,710)
#For Half screen make (w,h) = (700,710)
display_width = 700      
display_height = 710

max_num_obstacles = 15
max_speed_obstacles = 3
max_speed_target = 5

robot_width = 20 
robot_height = 20
initial_x_bot = 5
initial_y_bot = (display_height - robot_height - 5)
max_speed_bot = 5

#######################
###		Regions		###
#######################
non_safe_radius = 30 #from center of bot
winning_radius = 20  #from center of bot	

non_safe_distance_edge = non_safe_radius - (min(robot_width,robot_height)/2)
winning_distance_edge = winning_radius - (min(robot_width,robot_height)/2)

non_safe_distance_diagonal = non_safe_radius - (sqrt((robot_width/2)**2 + (robot_height/2)**2)) 
winning_distance_diagonal = winning_radius - (sqrt((robot_width/2)**2 + (robot_height/2)**2))

#######################

black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)



gameDisplay = pygame.display.set_mode((display_width,display_height))
#gameDisplay.fill(white)
pygame.display.set_caption("Robot Simulation Environment")
clock = pygame.time.Clock()

robot_image = pygame.image.load('robot.jpeg')

def sensor_detection(x,y,attro,attrt):
	center_x = x + (robot_width/2)
	center_y = y + (robot_height/2)
	
	ser_x,ser_y = center_x + (robot_width/2), center_y 
	sed_x,sed_y = center_x, center_y + (robot_height/2)
	sel_x,sel_y = center_x - (robot_width/2), center_y
	seu_x,seu_y = center_x, center_y - (robot_height/2)
	
	sdur_x, sdur_y = x + robot_width, y
	sdrd_x, sdrd_y = x + robot_width, y + robot_height
	sddl_x, sddl_y = x, y + robot_height
	sdlu_x, sdlu_y = x, y
	
	#Initialize distances
	dist_ser = display_width - ser_x
	dist_sed = display_height - sed_y
	dist_sel = sel_x
	dist_seu = seu_y
	dist_sdur_x, dist_sdur_y = display_width - sdur_x, sdur_y 
	dist_sdrd_x, dist_sdrd_y = display_width - sdrd_x, display_height - sdrd_y
	dist_sddl_x, dist_sddl_y = sddl_x, display_height - sddl_y
	dist_sdlu_x, dist_sdlu_y = sdlu_x, sdlu_y
	
	dist_sdur = min(display_width - sdur_x, sdur_y) * sqrt(2)
	dist_sdrd = min(display_width - sdrd_x, display_height - sdrd_y) * sqrt(2)
	dist_sddl = min(sddl_x, display_height - sddl_y) * sqrt(2)
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
						
def avoid_collision_target(attrt,attro,tosst,tosso,i):
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
	
	if dist_x==0:
		x_change = 0
		y_change = dir_y * max_speed_bot
	elif dist_y==0:
		x_change = dir_x * max_speed_bot
		y_change = 0
	else:
		if dist_x <= dist_y:
			ratio = float(dist_x)/float(dist_y)
			min_r = 2
			for i in range(max_speed_bot,0,-1):
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
			for i in range(max_speed_bot,0,-1):
				for j in range(i,0,-1):
					r = float(j)/float(i)
					if abs(r - ratio) < min_r:
						x_change = dir_x * i
						y_change = dir_y * j
						min_r = abs(r - ratio)
					if min_r==0:
						break 
			
	return x_change,y_change

def reach_count(count,total):
	font = pygame.font.SysFont(None, 25)
	text = font.render("Successfull attempts: " + str(count) + "/" + str(total), True, black)
	gameDisplay.blit(text,(0,0))

def text_objects(text,font):
	textSurface = font.render(text,True,black)
	return textSurface,textSurface.get_rect()

def message_display(text,succ,total):
	largeText = pygame.font.Font('freesansbold.ttf',30)
	TextSurf, TextRect = text_objects(text,largeText)
	TextRect.center = ((display_width/2),display_height/2)
	gameDisplay.blit(TextSurf,TextRect)
	
	pygame.display.update()
	
	time.sleep(2)
	
	sim_env(succ,total)
	
def won(succ,total):
	message_display("You have reached the target!!",succ,total)

def collision(succ,total):
	message_display("Bot collided with an Obstacle!",succ,total)

def robot(x,y):
	gameDisplay.blit(robot_image,(x,y))
	
def target(attr,color):
	pygame.draw.rect(gameDisplay,color,attr[0:4])
	
def obstacles(attr,color):
	pygame.draw.rect(gameDisplay,color,attr[:][0:4])

def sim_env(succ,total):
	x = initial_x_bot
	y = initial_y_bot
	
	x_change = 0
	y_change = 0
	
	total_attempts = total
	success = succ
	
	####Gear Box####
	gear = 0
	
	targetx = random.randrange(40,display_width-30) 
	targety = random.randrange(40,display_height-30)
	targetw = 30
	targeth = 30
	#target_speedx = random.randrange(0,max_speed_target+1) #unlock these to give
	#target_speedy = random.randrange(0,max_speed_target+1) #target some velocity
	target_speedx = 0
	target_speedy = 0
	attrt = [targetx,targety,targetw,targeth,target_speedx,target_speedy]
	
	attro = []
	for i in range(0,random.randrange(1,max_num_obstacles+1)):
		flag_obs = 0
		while flag_obs==0:
			obstaclex = random.randrange(40,display_width)
			obstacley = random.randrange(40,display_height)
			obstaclew = random.randrange(50,120)
			obstacleh = random.randrange(50,120)
			if (targetx + targetw) < obstaclex or targetx > (obstaclex + obstaclew) or (targety + targeth) < obstacley or targety > (obstacley + obstacleh):
				flag_obs = 1   
		#obstacle_speedx =random.randrange(0,max_speed_obstacles+1)#unlock these to give
		#obstacle_speedy =random.randrange(0,max_speed_obstacles+1)#obstacle some velocity
		obstacle_speedx = 0
		obstacle_speedy = 0
		attro.append([obstaclex,obstacley,obstaclew,obstacleh,obstacle_speedx,obstacle_speedy])
	
	envExit = False
	
	while not envExit:
		
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
		
		sensors_old = sensor_detection(x,y,attro,attrt)
		
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
		
		de = robot_width/2
		dd = sqrt((robot_width)**2+(robot_height)**2)
		min_dist_old = min(dist_re+de,dist_de+de,dist_le+de,dist_ue+de,dist_urd+dd,dist_lud+dd,dist_rdd+dd,dist_dld+dd)
		
		if (abs(dist_t_re)<=winning_distance_edge) or (abs(dist_t_de)<=winning_distance_edge) or (abs(dist_t_le)<=winning_distance_edge) or (abs(dist_t_ue)<=winning_distance_edge) or (abs(dist_t_urd)<=winning_distance_diagonal) or (abs(dist_t_lud)<=winning_distance_diagonal) or (abs(dist_t_rdd)<=winning_distance_diagonal) or (abs(dist_t_dld)<=winning_distance_diagonal):
			old_state = 3 #WS
		elif (dist_re<=non_safe_distance_edge) or (dist_de<=non_safe_distance_edge) or (dist_le<=non_safe_distance_edge) or (dist_ue<=non_safe_distance_edge) or (dist_urd<=non_safe_distance_diagonal) or (dist_lud<=non_safe_distance_diagonal) or (dist_rdd<=non_safe_distance_diagonal) or (dist_dld<=non_safe_distance_diagonal):
			old_state = 1 #NS
		else:
			old_state = 2 #SS
			
		print("Old state: " + str(old_state))
		
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				quit()
			
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					if gear < 6:
						gear+=1
				if event.key == pygame.K_s:
					if gear > 0:
						gear-=1
				if event.key == pygame.K_LEFT:
					x_change = -1 * max_speed_bot * gear
				if event.key == pygame.K_RIGHT:
					x_change = max_speed_bot * gear
				if event.key == pygame.K_UP:
					y_change = -1 * max_speed_bot * gear
				if event.key == pygame.K_DOWN:
					y_change = max_speed_bot  * gear
			
			if event.type == pygame.KEYUP:
				if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT or event.key == pygame.K_UP or event.key == pygame.K_DOWN:
					x_change = 0
					y_change = 0
				
			
		#x_change,y_change = bot_velocity(attrt,x,y) #automate bot_velocity
		
		x += x_change
		y += y_change
		
		##########################
		gameDisplay.fill(white)
		
		for i in range(0,len(attro)):
			obstacles(attro[i],blue)
			
		target(attrt,green)
		
		robot(x,y)
		reach_count(success,total_attempts)
		##########################
		tosst = random.randrange(0,1000) % 4
		if tosst==0:
			attrt[0] += attrt[4]
			attrt[1] += attrt[5]
		elif tosst==1:
			attrt[0] -= attrt[4]
			attrt[1] += attrt[5]
		elif tosst==2:
			attrt[0] += attrt[4]
			attrt[1] -= attrt[5]
		elif tosst==3:
			attrt[0] -= attrt[4]
			attrt[1] -= attrt[5]
		
		if attrt[0] + attrt[2] > display_width:
			attrt[0] = display_width-attrt[2]
		elif attrt[0]<0:
			attrt[0] = 0	
		if attrt[1] + attrt[3] > display_height:
			attrt[1] = display_height-attrt[3]
		elif attrt[1]<0:
			attrt[1]=0
			
		tosso = []
		for i in range(len(attro)):
			tosso.append(random.randrange(0,1000) % 4)
			if tosso[i]==0:
				attro[i][0] += attro[i][4]
				attro[i][1] += attro[i][5]
			elif tosso[i]==1:
				attro[i][0] -= attro[i][4]
				attro[i][1] += attro[i][5]
			elif tosso[i]==2:
				attro[i][0] += attro[i][4]
				attro[i][1] -= attro[i][5]
			elif tosso[i]==3:
				attro[i][0] -= attro[i][4]
				attro[i][1] -= attro[i][5]
			
		##########################
		"""if x > display_width - robot_width or x<0:
			if x<0:
				x=0
			else:
				x = display_width - robot_width
				
		if y > display_height - robot_height or y<0:
			if y<0:
				y=0
			else:
				y = display_height - robot_height
				
		#Uncommenting this block means that the robot cannot fall off the edge of the environment but in real scenario it has to learn not to fall off of the edges
		
		"""
		#######################################
		####        COLLISION OR WON       ####
		#######################################
		
		for i in range(0,len(attro)):
			if attrt[1] > attro[i][1] and attrt[1] < attro[i][1]+attro[i][3] or attrt[1]+attrt[3] > attro[i][1] and attrt[1]+attrt[3] < attro[i][1]+attro[i][3] :
		
				if attrt[0]>attro[i][0] and attrt[0]<attro[i][0]+attro[i][2] or attrt[0]+attrt[2] > attro[i][0] and attrt[0]+attrt[2] < attro[i][0]+attro[i][2]:
					#print("collision avoided")
					avoid_collision_target(attrt,attro,tosst,tosso,i)
		
		flagc = 0
		flagw = 0		
		for i in range(0,len(attro)):
			if y > attro[i][1] and y < attro[i][1]+attro[i][3] or y+robot_height > attro[i][1] and y+robot_height < attro[i][1]+attro[i][3] :
		
				if x>attro[i][0] and x<attro[i][0]+attro[i][2] or x+robot_width > attro[i][0] and x+robot_width < attro[i][0]+attro[i][2]:
					flagc = 1
					break
					
		if y<0 or y+robot_height>display_height or x<0 or x+robot_width>display_width:
			flagc=1 
					
		if (y > attrt[1] and y < attrt[1]+attrt[3]) or (y+robot_height > attrt[1] and y+robot_height < attrt[1]+attrt[3]):
				
			if (x>attrt[0] and x<attrt[0]+attrt[2]) or (x+robot_width > attrt[0] and x+robot_width < attrt[0]+attrt[2]):
				flagw = 1
				success += 1
				
		"""
		if flagw==1:
			total_attempts +=1
			print("-----------*****Won*****---------------")
			won(success,total_attempts)
			
		if flagc==1:
			total_attempts +=1
			print("-----------*****Collided*****---------------")
			collision(success,total_attempts)
		"""	
		#######################################
		
		#######################################
		###         SENSOR-DATA NEW         ###
		#######################################
		
		sensors_new = sensor_detection(x,y,attro,attrt)
		
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
		elif (abs(dist_t_re)<=winning_distance_edge) or (abs(dist_t_de)<=winning_distance_edge) or (abs(dist_t_le)<=winning_distance_edge) or (abs(dist_t_ue)<=winning_distance_edge) or (abs(dist_t_urd)<=winning_distance_diagonal) or (abs(dist_t_lud)<=winning_distance_diagonal) or (abs(dist_t_rdd)<=winning_distance_diagonal) or (abs(dist_t_dld)<=winning_distance_diagonal):
			new_state = 3 #WS
		elif (dist_re<=non_safe_distance_edge) or (dist_de<=non_safe_distance_edge) or (dist_le<=non_safe_distance_edge) or (dist_ue<=non_safe_distance_edge) or (dist_urd<=non_safe_distance_diagonal) or (dist_lud<=non_safe_distance_diagonal) or (dist_rdd<=non_safe_distance_diagonal) or (dist_dld<=non_safe_distance_diagonal):
			new_state = 1 #NS
		else:
			new_state = 2 #SS
		
		print("New state: " + str(new_state))
		
			
		#######################################
		###           SENSOR-DATA           ###
		#######################################
		sensors = sensors_new
		#sensors = sensor_detection(x,y,attro,attrt)
		#####	Obstacles	#####
		"""
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
		"""
		#######################################
		
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
		if old_state==2 and new_state==3:   # SS ----> WS
			reward = 2
		elif old_state==3 and new_state==4: # WS ----> FS
			reward = 3
		elif old_state==1 and new_state==2: # NS ----> SS
			reward = 1
		elif old_state==2 and new_state==1: # SS ----> NS
			reward = -2
		elif old_state==1 and new_state==1 and (min_dist_new < min_dist_old): # NS ----> NS
			reward = -1
		elif old_state==1 and new_state==1 and (min_dist_new >= min_dist_old): # NS ----> NS
			reward = 0
		elif old_state==1 and new_state==0: # NS ----> CS
			reward = -3
		elif old_state==2 and new_state==2: # SS ----> SS
			reward = 0
		else: # WS ----> SS || any other scenario
			reward = 0 
		
		print("-------------------Reward = " + str(reward) + "-------------------")
		#######################################
		
		if flagw==1:
			total_attempts +=1
			#print("-----------*****Won*****---------------")
			won(success,total_attempts)
			
		if flagc==1:
			total_attempts +=1
			#print("-----------*****Collided*****---------------")
			collision(success,total_attempts)
			
		pygame.display.update()
		clock.tick(60)		
				
						
sim_env(0,0)
