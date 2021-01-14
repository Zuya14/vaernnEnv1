import sys

import gym
import numpy as np
import gym.spaces
from gym.utils import seeding
import torch

import pybullet as p
import cv2
import math

import random
import copy

import bullet_lidar
import robot_sim  

class vaernnEnv1(gym.Env):
    global_id = 0

    def __init__(self):
        super().__init__()
        self.seed(seed=random.randrange(10000))
        self.sim = None

    def setting(self, _id=-1, mode=p.DIRECT, sec=0.01):
        if _id == -1:
            self.sim = robot_sim.robot_sim(vaernnEnv1.global_id, mode, sec)
            vaernnEnv1.global_id += 1
        else:
            self.sim = robot_sim.robot_sim(_id, mode, sec)

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))

        self.lidar = self.createLidar()

        self.sec = sec
        self.reset()

    def copy(self, _id=-1):
        new_env = vaernnEnv1()
        
        if _id == -1:
            new_env.sim = self.sim.copy(vaernnEnv1.global_id)
            vaernnEnv1.global_id += 1
        else:
            new_env.sim = self.sim.copy(_id)

        new_env.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))
        new_env.lidar = new_env.createLidar()
        new_env.sec = self.sec

        return new_env

    def reset(self, x=0.0, y=0.0, theta=0.0, vx=0.0, vy=0.0, w=0.0, action=None, clientReset=False):
        assert self.sim is not None, print("call setting!!") 
        self.sim.reset(x=x, y=y, theta=theta, vx=vx, vy=vy, w=w, sec=self.sec, action=action, clientReset=clientReset)
        return None

    def createLidar(self):
        deg_offset = 90.
        startDeg = -135. + deg_offset
        endDeg = 135. + deg_offset
        resolusion = 0.25
        maxLen = 8.
        minLen = 0.
        return bullet_lidar.bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    def step(self, action):

        done = self.sim.step(action)

        observation = self.sim.observe(self.lidar)

        reward = self.get_reward()

        return observation, reward, done, {}

    def observe(self):
        return self.sim.observe(self.lidar)

    def observe2d(self):
        return self.sim.observe2d(self.lidar)

    def get_reward(self):
        isComtact = self.sim.isContacts()

        rewardContact = -1.0 if isComtact else 0.0
        # rewardContact = -100.0 if isComtact else 0.0
        # rewardContact = -10.0 if isComtact else 0.0

        rewardMove = self.sim.vy

        reward = rewardContact + rewardMove

        return reward

    def render(self, mode='human', close=False):
        pass

    def close(self):
        self.sim.close()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_random_action(self):
        return self.action_space.sample()
        # return torch.from_numpy(self.action_space.sample())

    def getState(self):
        return self.sim.getState()