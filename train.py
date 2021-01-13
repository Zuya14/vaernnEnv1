import argparse

import math
import numpy as np
import random
import os
import datetime
from concurrent import futures

from StateBuffer import StateMem, StateBuffer
from EpisodeMemory import Episode, EpisodeMemory
from lidar_util import imshowLocalDistance

import pybullet as p
import gym
import cv2

import torch
from torchvision.utils import save_image

from model.VAE_trainer import VAE_trainer
from model.LGRU_trainer import LGRU_trainer
from model.RewardModel_trainer import RewardModel_trainer
import plot_graph

from gym.envs.registration import register

register(
    id='vaernn-v1',
    entry_point='vaernnEnv1:vaernnEnv1'
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def collect_init_episode(memory_size, collect_num, min_step, clientReset=False, sample_rate=0.001, sec=0.01):
    env = gym.make('vaernn-v1')
    env.setting(sec=sec)
    memory = EpisodeMemory(mem_size=memory_size)

    collect_count = 0

    while collect_count < collect_num:
        episode = Episode()

        env.reset(sec=sec, clientReset=clientReset)
        observation = env.observe()

        step = 0
        while True:

            pre_action = env.sample_random_action()
            next_observation, reward, done, _ = env.step(pre_action)

            action = env.sim.action
            episode.append(observation[:1080], action, reward, done)
            observation = next_observation

            step += 1
            if done:
                break
        
        if step >= min_step:
            collect_count += 1
            memory.append(episode)

    return memory

if __name__ == '__main__':

    s_time = datetime.datetime.now()

    print("start:", s_time)

    parser = argparse.ArgumentParser(description='train')

    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--init-episode", type=int, default=10)

    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument("--rnn-hidden", type=int, default=18)

    parser.add_argument('--chunk-size', type=int, default=10)

    parser.add_argument('--models-vae', type=str, default='')
    parser.add_argument('--models-rnn', type=str, default='')
    parser.add_argument('--models-reward', type=str, default='')
    parser.add_argument('--id', type=str, default='')

    parser.add_argument('--threads', type=int, default=min(os.cpu_count(), 10))

    parser.add_argument("--sec", type=float, default=0.01)

    args = parser.parse_args()

    out_dir = './result' 

    if args.id != '':
        out_dir += '/' + args.id

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ''' ---- Models ---- '''

    latent_size = 18
    action_size = 3

    cnn_outsize = 34560
    first_channel = 8
    red_times = 1
    repeat = 0
    channel_inc = 2

    vae_train =  VAE_trainer(
        first_channel=first_channel,
        latent_size=latent_size, 
        red_times = red_times,
        repeat=repeat,
        channel_inc=channel_inc, 
        device=device)
    rnn_train =  LGRU_trainer(
        latent_size=latent_size,
        action_size=action_size, 
        hidden_size=args.rnn_hidden, 
        activation_function='relu', 
        drop_prob=args.drop_prob, 
        num_layers=args.num_layers, 
        device=device)
    reward_train = RewardModel_trainer(
        state_size=latent_size,
        belief_size=args.rnn_hidden,
        hidden_size=latent_size+args.rnn_hidden,
        device=device
    )

    
    if args.models_vae is not '' and os.path.exists(args.models_vae):
        vae_train.load(args.models_vae)
        print("load:", args.models_vae)

    if args.models_rnn != '' and os.path.exists(args.models_rnn):
        rnn_train.load(args.models_rnn)
        print("load:", args.models_rnn)

    if args.models_reward != '' and os.path.exists(args.models_reward):
        reward_train.load(args.models_reward)
        print("load:", args.models_reward)


    train_plot_data = plot_graph.Plot_Graph_Data(out_dir, 'train_loss', {'vae_loss': [], 'mse_loss': [], 'KLD_loss': [], 'rnn_loss': [], 'reward_loss': []})
    plotGraph = plot_graph.Plot_Graph([train_plot_data])

    ''' ---- Initialize EpisodeMemory ---- '''

    memory = EpisodeMemory(mem_size=args.memory_size)

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_list = [executor.submit(collect_init_episode, memory_size=args.memory_size, collect_num=args.init_episode, min_step=args.chunk_size+args.test_predict_step+1, clientReset=True, sample_rate=sample_rate, sec=args.sec) for i in range(args.threads)]
        future_return = futures.as_completed(fs=future_list)

    for future in future_return:
        mem = future.result()
        memory.extend(mem)

    ''' ---- Model fitting ---- '''

    for i in range(args.epochs):

        datas_observations, datas_actions, datas_rewards, datas_dones, next_observations = memory.sample(n=args.batch_size, L=args.chunk_size)
        mse_loss, KLD_loss, z  = vae_train.train2(datas_observations)
        _, _, z2 = vae_train.train2(next_observations)
        vae_loss = mse_loss+KLD_loss

        z  = z.detach()
        z2 = z2.detach()

        rnn_loss, predicts, hiddens = rnn_train.train(z.view(args.batch_size, args.chunk_size, latent_size), datas_actions, z2.view(args.batch_size, 1, latent_size))
        reward_loss = reward_train.train(predicts.view(args.batch_size, -1).detach(), hiddens.view(args.batch_size, -1).detach(), datas_rewards[:, -1].view(args.batch_size, 1)) 

        plotGraph.addDatas('train_loss', ['vae_loss', 'mse_loss', 'KLD_loss', 'rnn_loss', 'reward_loss'], [vae_loss, mse_loss, KLD_loss, rnn_loss, reward_loss])

        if epoch%10 == 0:
            vae_train.save(out_dir+'/vae.pth')
            rnn_train.save(out_dir+'/rnn.pth')
            reward_train.save(out_dir+'/reward.pth')

            plotGraph.plot('train_loss')

            print('epoch [{}/{}], vae_loss: {:.4f}, rnn_loss: {:.4f} reward_loss: {:.4f} '.format(
                epoch + 1,
                num_epochs,
                vae_loss,
                rnn_loss,
                reward_loss)
                )       