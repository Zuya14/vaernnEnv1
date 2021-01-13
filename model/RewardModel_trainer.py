import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

from .RewardModel import RewardModel

class RewardModel_trainer:

    def __init__(self, state_size, belief_size, hidden_size, device='cpu'):
        self.rewardModel = RewardModel(state_size, belief_size, hidden_size)
        self.rewardModel = self.rewardModel.to(device, non_blocking=True)
        self.device = device

        self.optimizer = optim.Adam(self.rewardModel.parameters())

    def train(self, predicts, hiddens, rewards):
        self.rewardModel.train()

        avg_loss = 0

        # for batch_idx, (predict, hidden, reward) in enumerate(zip(predicts, hiddens, rewards)):
        predicts = predicts.to(self.device)
        hiddens  = hiddens.to(self.device)
        rewards  = rewards.to(self.device)

        self.optimizer.zero_grad()
        
        out = self.rewardModel(predicts, hiddens)

        loss = F.mse_loss(out, rewards)
        # loss = F.binary_cross_entropy(out, rewards)
        
        loss.backward()
        
        avg_loss += loss.item()
        
        self.optimizer.step()

        avg_loss /= len(predicts)

        return avg_loss

    def test(self, predicts, hiddens, rewards):
        self.rewardModel.eval()

        avg_loss = 0
        
        with torch.no_grad():
            # for batch_idx, (predict, hidden, reward) in enumerate(zip(predicts, hiddens, rewards)):
            predicts = predicts.to(self.device)
            hiddens  = hiddens.to(self.device)
            rewards  = rewards.to(self.device)
            
            out = self.rewardModel(predicts, hiddens)
            loss = F.mse_loss(out, rewards)
            # loss = F.binary_cross_entropy(out, rewards)
            
            avg_loss += loss.item()
                
        avg_loss /= len(predicts)

        return avg_loss

    def save(self, path):
        torch.save(self.rewardModel.to('cpu').state_dict(), path)
        self.rewardModel.to(self.device)

    def load(self, path):
        self.rewardModel.load_state_dict(torch.load(path))