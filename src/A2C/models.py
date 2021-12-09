import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

class ActorCritic(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, seed):
		super(ActorCritic, self).__init__()
		self.seed = torch.manual_seed(seed)

		self.critic_linear1 = nn.Linear(input_size, hidden_size)
		self.critic_linear2 = nn.Linear(hidden_size, 1)

		self.actor_linear1 = nn.Linear(input_size,hidden_size)
		self.actor_linear2 = nn.Linear(hidden_size, output_size)

	def forward(self, state):
		"""
		Param state is a torch tensor
		"""
		value = F.relu(self.critic_linear1(state))
		value = F.relu(self.critic_linear2(value))

		policy_dist = F.relu(self.actor_linear1(state))
		policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

		return value, policy_dist