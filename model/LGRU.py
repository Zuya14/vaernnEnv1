import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import jit

class LGRU(nn.Module):
    __constants__ = ['latent_size', 'action_size', 'hidden_size']

    def __init__(self, latent_size, action_size, hidden_size, activation_function='relu', drop_prob=0, num_layers=1):
        super().__init__()
        self.latent_size = latent_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.act_fn = getattr(F, activation_function)

        self.fc = nn.Linear(hidden_size, latent_size)

        self.rnn = nn.GRU(latent_size + action_size, hidden_size, dropout=drop_prob, batch_first=True, num_layers=num_layers)

    def forward(self, actions, latents, hidden=None):
        """ MULTI STEPS forward.

        :args actions: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args latents: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        """
        if hidden is None:
            return self.forward1(actions, latents)
        else:
            return self.forward2(actions, latents, hidden)

    def forward1(self, actions, latents):
        state = torch.cat([actions, latents], dim=-1)

        self.rnn.flatten_parameters()
        outs, hidden = self.rnn(state)
            
        predict = self.fc(self.act_fn(outs[:, -1, :])).reshape(-1, 1, self.latent_size)

        return predict, hidden

    def forward2(self, actions, latents, hidden):
        state = torch.cat([actions, latents], dim=-1)

        self.rnn.flatten_parameters()

        outs, hidden = self.rnn(state, hidden)
            
        predict = self.fc(self.act_fn(outs[:, -1, :])).reshape(-1, 1, self.latent_size)

        return predict, hidden