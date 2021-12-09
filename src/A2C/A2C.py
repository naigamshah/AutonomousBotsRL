import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
import random
from models import *
from utils import *

class A2CAgent:
	def __init__(self, env, hidden_size=256, learning_rate=3e-4, gamma=0.99, max_memory_size=50000, seed=7):
		self.num_states = env.observation_space.shape[0]
		self.num_actions = env.action_space.n
		self.gamma = gamma
		self.seed = random.seed(seed)

		self.actor_critic = ActorCritic(self.num_states, hidden_size, self.num_actions, seed)
		self.ac_optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

		self.memory = Memory(max_memory_size, seed)

	def get_action(self, state):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		value, policy_dist = self.actor_critic.forward(state)
		value = value.detach().numpy()[0,0]
		return value, policy_dist

	def update(self, log_probs, values, rewards, Qval, entropy_term):
		Qvals = np.zeros_like(values)
		for t in reversed(range(len(rewards))):
			Qval = rewards[t] + self.gamma * Qval
			Qvals[t] = Qval
		values = torch.FloatTensor(values)
		Qvals = torch.FloatTensor(Qvals)
		log_probs = torch.stack(log_probs)

		#Loss calculations
		advantage = Qvals - values
		actor_loss = (-log_probs * advantage).mean()
		critic_loss = 0.5 * advantage.pow(2).mean()
		ac_loss = actor_loss + critic_loss + 0.001 * entropy_term

		#Update network
		self.ac_optimizer.zero_grad()
		ac_loss.backward()
		self.ac_optimizer.step()


