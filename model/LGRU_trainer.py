import torch
import torch.nn as nn
import torch.optim as optim

from .LGRU import LGRU

class LGRU_trainer:

    def __init__(self, latent_size, action_size, hidden_size, activation_function='relu', drop_prob=0, num_layers=1, device='cpu'):

        self.rnn = LGRU(latent_size, action_size, hidden_size, activation_function, drop_prob, num_layers)          

        self.rnn = self.rnn.to(device)
        self.device = device

        self.optimizer = optim.Adam(self.rnn.parameters())

    def train(self, observations, actions, next_observations):
        self.rnn.train()
        train_loss = 0
        criterion = nn.MSELoss()

        obs = observations.to(self.device)
        act = actions.to(self.device)
        next_obs = next_observations.to(self.device)

        for param in self.rnn.parameters():
            param.grad = None

        predict, hidden = self.rnn(act, obs)
        # print(predict.size(), next_obs.size())

        loss = criterion(predict, next_obs)
        loss.backward()
        train_loss += loss.item()   
        self.optimizer.step()

        train_loss /= actions[0].size()[0]

        return train_loss, predict, hidden

    def test(self, observations, actions, next_observations):
        self.rnn.eval()
        test_loss = 0
        criterion = nn.MSELoss()

        with torch.no_grad():

            obs = observations.to(self.device)
            act = actions.to(self.device)
            next_obs = next_observations.to(self.device)

            predict, hidden = self.rnn(act, obs)

            loss = criterion(predict, next_obs)

            test_loss += loss.item()
                        
        test_loss /= actions[0].size()[0]

        return test_loss, predict, hidden

    def save(self, path):
        torch.save(self.rnn.to('cpu').state_dict(), path)
        self.rnn.to(self.device)

    def load(self, path):
        self.rnn.load_state_dict(torch.load(path))