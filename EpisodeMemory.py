import random
import numpy as np
import torch
import torch.nn.utils.rnn as rnn


'''
    [
        [observations, actions, rewards, dones],
        [observations, actions, rewards, dones],
        [observations, actions, rewards, dones],
        [
            [observation, observation, observation], 
            [actions, actions, actions], 
            [rewards, rewards, rewards], 
            [dones, dones, dones]
        ]
    ]
'''

class Episode:

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def append(self, observation, action, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def size(self):
        return len(self.observations)

    def sample(self, L):
        if L < self.size():
            index = random.randint(0, self.size()-1-L)
            return self.observations[index:index+L], self.actions[index:index+L], self.rewards[index:index+L], self.dones[index:index+L]
        else:
            return self.observations, self.actions, self.rewards, self.dones

    def get(self):
        return self.observations, self.actions, self.rewards, self.dones


class EpisodeMemory:

    def __init__(self, mem_size):
        self.mem_size = mem_size
        self.size = 0
        self.episodes = []

    def append(self, episode):
        self.size += episode.size()
        self.episodes.append(episode)

        while self.size > self.mem_size:
            self.size -= self.episodes[0].size()
            del self.episodes[0]

    def extend(self, memory):
        self.size += memory.size
        self.episodes.extend(memory.episodes)

        while self.size > self.mem_size:
            self.size -= self.episodes[0].size()
            del self.episodes[0]

    def get(self, index):
        datas = [self.episodes[index].get()]

        datas_observations = rnn.pad_sequence([torch.tensor(data[0]) for data in datas], batch_first=True).float()
        datas_actions      = rnn.pad_sequence([torch.tensor(data[1]) for data in datas], batch_first=True).float()

        return datas_observations, datas_actions

    def sample(self, n, L):
        indeces = np.random.randint(0, len(self.episodes), n)

        datas = [self.episodes[i].sample(L+1) for i in indeces]

        datas_observations  = rnn.pad_sequence([torch.tensor(data[0][:L]) for data in datas], batch_first=True).float()
        next_observations = rnn.pad_sequence([torch.tensor(data[0][L]) for data in datas], batch_first=True).float()
        datas_actions      = rnn.pad_sequence([torch.tensor(data[1][:L]) for data in datas], batch_first=True).float()
        datas_rewards      = rnn.pad_sequence([torch.tensor(data[2][:L]) for data in datas], batch_first=True).float()
        datas_dones        = rnn.pad_sequence([torch.tensor(data[3][:L]) for data in datas], batch_first=True).float()

        return datas_observations, datas_actions, datas_rewards, datas_dones, next_observations

    def sample_indeces(self, n, L, indeces):
        datas = [self.episodes[i].sample(L+1) for i in indeces]

        datas_observations  = rnn.pad_sequence([torch.tensor(data[0][:L]) for data in datas], batch_first=True).float()
        next_observations = rnn.pad_sequence([torch.tensor(data[0][L]) for data in datas], batch_first=True).float()
        datas_actions      = rnn.pad_sequence([torch.tensor(data[1][:L]) for data in datas], batch_first=True).float()
        datas_rewards      = rnn.pad_sequence([torch.tensor(data[2][:L]) for data in datas], batch_first=True).float()
        datas_dones        = rnn.pad_sequence([torch.tensor(data[3][:L]) for data in datas], batch_first=True).float()

        return datas_observations, datas_actions, datas_rewards, datas_dones, next_observations

if __name__ == '__main__':

    memory = EpisodeMemory(mem_size=100000)

    while True:
        episode = Episode()
        count = 0
        while True:
            observation = np.random.rand(10) 
            action = np.random.rand(3)
            reward = random.random()
            done = True if random.randint(0,100) >= 98 else False

            episode.append(observation, action, reward, done)

            count += 1

            if done:
                break

        memory.append(episode)

        datas_observations, datas_actions, datas_rewards, datas_dones = memory.sample2(n=10, L=100)

        print(datas_observations.size())
        print(datas_actions.size())
        print(datas_rewards.size())
        print(datas_dones.size())

        print(torch.cat([datas_observations, datas_actions], dim=2).shape)
        print()
