import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

class RewardModel(jit.ScriptModule):
  def __init__(self, state_size, belief_size, hidden_size, activation_function='relu'):
    super().__init__()
    self.act_fn = getattr(F, activation_function)
    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)
    # self.sigmoid = nn.Sigmoid()

  @jit.script_method
  def forward(self, state, belief):
    hidden = self.act_fn(self.fc1(torch.cat([state, belief], dim=1)))
    hidden = self.act_fn(self.fc2(hidden))
    # reward = self.fc3(hidden).squeeze(dim=1)
    reward = self.fc3(hidden)
    # reward = self.sigmoid(self.fc3(hidden))
    return reward