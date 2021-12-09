import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np 
import random
from collections import namedtuple, deque
from models import QNetwork
from memory_utils import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQNAgent():
	def __init__(self, state_size, action_size, dqn_type="DQN", max_memory_size=1e5, batch_size=64, gamma=0.99, learning_rate=1e-3, tau=2e-3, update_rate=4, seed=7):
		self.dqn_type = dqn_type
		self.state_size = state_size
		self.action_size = action_size
		self.buffer_size = int(max_memory_size)
		self.batch_size = batch_size
		self.gamma = gamma
		self.learning_rate = learning_rate
		self.tau = tau
		self.update_rate = update_rate
		self.seed = random.seed(seed)

		self.network = QNetwork(state_size, action_size, seed).to(device)
		self.target_network = QNetwork(state_size, action_size, seed).to(device)
		self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

		self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, seed)

		#Inititalize time step for updating every UPDATE_EVERY steps
		self.t_step = 0

	def step(self, state, action, reward, next_state, done):
		self.memory.push(state, action, reward, next_state, done)

		self.t_step = (self.t_step + 1) % self.update_rate
		if self.t_step == 0:
			if len(self.memory) > self.batch_size:
				experiences = self.memory.sample()
				self.learn(experiences, self.gamma)

	def act(self, state, eps=0.0):
		"""Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
		state = torch.from_numpy(state).float().unsqueeze(0).to(device)
		self.network.eval()
		with torch.no_grad():
			action_values = self.network(state)
		self.network.train()

		#Epsilon-greedy action selection
		if random.random() > eps:
			return np.argmax(action_values.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_size))

	def learn(self, experiences, DQN=True):
		"""
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            self.gamma (float): discount factor
        """

		states, actions, rewards, next_states, terminals = experiences

        # Get Q values from current observations (s, a) using model nextwork
		Qsa = self.network(states).gather(1, actions)

		if self.dqn_type == "DDQN":
			Qsa_prime_actions = self.network(next_states).detach().max(1)[1].unsqueeze(1)
			Qsa_prime_targets = self.target_network(next_states)[Qsa_prime_actions].unsqueeze(1)
		else:
        	#  self.dqn_type == "DQN" ----> Vanilla DQN
			Qsa_prime_targets_values = self.target_network(next_states).detach()
			Qsa_prime_targets = Qsa_prime_targets_values.max(1)[0].unsqueeze(1)

        # Compute Q targets for current states
		Qsa_targets = rewards + (self.gamma * Qsa_prime_targets * (1 - terminals))

		loss = F.mse_loss(Qsa, Qsa_targets)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

        #Update target network
		self.soft_update(self.network, self.target_network, self.tau)

	"""
	Soft update model parameters.
	θ_target = τ*θ_local + (1 - τ)*θ_target
	"""
	def soft_update(self, local_model, target_model, tau):
		for target_param, param in zip(target_model.parameters(), local_model.parameters()):
			target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    		
