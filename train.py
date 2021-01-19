import argparse

import math
import numpy as np
import random
import os
import datetime
from concurrent import futures

from EpisodeMemory import Episode, EpisodeMemory
from lidar_util import imshowLocalDistance

import pybullet as p
import gym
import cv2

import torch
from torch import jit
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

max_step = 200

def collect_init_episode(memory_size, collect_num, min_step, clientReset=False, sample_rate=0.001, sec=0.01):
    env = gym.make('vaernn-v1')
    env.setting(sec=sec)
    memory = EpisodeMemory(mem_size=memory_size)

    collect_count = 0

    while collect_count < collect_num:
        episode = Episode()

        env.reset(clientReset=clientReset)
        observation = env.observe()

        step = 0
        while True:
        # for _ in range(max_step):
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

def planner(rnn_model, reward_model, old_actions, old_states, state, planning_horizon, optimisation_iters, candidates, top_candidates, sec):
    action0 = old_actions[:, -1]
    action_size = action0.size()[-1]
    state_size = state.size()[-1]

    diff_action_mean = torch.zeros(1, planning_horizon, action_size, device=device)
    diff_action_std_dev = torch.ones(1, planning_horizon, action_size, device=device)

    for _ in range(optimisation_iters):
        diff_actions = (diff_action_mean + diff_action_std_dev * torch.randn(candidates, planning_horizon, action_size, device=device))
        diff_actions.clamp_(min=-1.0, max=1.0) 

        # actions = calcAction2(action0, diff_actions, sec)
        actions = calcAction3(action0, diff_actions*sec)

        # if old_states is not None:
        #     act = torch.cat([old_actions.view(1, -1, action_size), actions[:, 0].view(-1, 1, action_size)], dim=1).expand(candidates, -1, -1) 
        #     inp = torch.cat([old_states.view(1, -1, state_size), state.view(1, 1, state_size)], dim=1).expand(candidates, -1, -1)
        # else:
        #     act = actions[:, 0].view(-1, 1, action_size).expand(candidates, -1, -1) 
        #     inp = state.view(1, 1, state_size).expand(candidates, -1, -1)   
        act = actions[:, 0].view(-1, 1, action_size).expand(candidates, -1, -1) 
        inp = state.view(1, 1, state_size).expand(candidates, -1, -1)   

        reward_sum = torch.zeros(candidates, planning_horizon, 1, device=device)

        for t in range(1, planning_horizon):
            predicts, hiddens = rnn_model(act, inp)
            reward = reward_model(predicts, hiddens.view(candidates, 1, -1))
            reward_sum[:, t-1] += reward.view(candidates, 1)

            act = torch.cat([act, actions[:, t, :].view(candidates, 1, action_size)], dim=1)
            inp = torch.cat([inp, predicts.view(candidates, 1, -1)], dim=1)

        predicts, hiddens = rnn_model(act, inp)
        reward = reward_model(predicts, hiddens.view(candidates, 1, -1))
        reward_sum[:, planning_horizon-1] += reward.view(candidates, 1)

        returns = reward_sum.view(candidates, planning_horizon).sum(dim=1)
        # print(reward_sum)

        _, topk = returns.topk(top_candidates, dim=0, largest=True, sorted=False)
        best_diff_actions = diff_actions[topk].reshape(top_candidates, planning_horizon, action_size)
        diff_action_mean, diff_action_std_dev = best_diff_actions.mean(dim=0, keepdim=True), best_diff_actions.std(dim=0, unbiased=False, keepdim=True)

    return diff_action_mean[0, 0, :].view(action_size)

# horizon * candidates * action_size
def calcAction(action0, diff_actions, sec):
    # !! --- diff_actions \in [-1, 1] --- !! 

    T = diff_actions.size()[0]
    candidates = diff_actions.size()[1]

    actions = torch.empty_like(diff_actions)
    v_s     = actions[:, :, 0]
    theta_s = actions[:, :, 1]
    w_s     = actions[:, :, 2]

    v_scale     = 1.0
    theta_scale = math.pi / 2.0
    w_scale     = math.pi / 4.0

    limit_v     = v_scale     * sec
    limit_theta = theta_scale * sec
    limit_w     = w_scale     * sec

    delta_v     = diff_actions[0, :, 0] * limit_v
    delta_theta = diff_actions[0, :, 1] * limit_theta
    delta_w     = diff_actions[0, :, 2] * limit_w

    v_s[0]     = (action0[0] + delta_v[0]).clamp_(-v_scale, v_scale)

    theta_s[0] = action0[1] + delta_theta[0]

    for c in range(candidates):
        if theta_s[0, c] > math.pi:
            theta_s[0, c] -= 2*math.pi
        elif theta_s[0, c] < -math.pi:
            theta_s[0, c] += 2*math.pi

    w_s[0]     = (action0[2] + delta_w[0]).clamp_(-w_scale, w_scale)

    for t in range(1, T):
        v_s[t]     = (v_s[t-1] + delta_v[t]).clamp_(-v_scale, v_scale)

        theta_s[t] = theta_s[t-1] + delta_theta[t]

        for c in range(candidates):
            if theta_s[t, c] > math.pi:
                theta_s[t, c] -= 2*math.pi
            elif theta_s[t, c] < -math.pi:
                theta_s[t, c] += 2*math.pi

        w_s[t]     = (w_s[t-1] + delta_w[t]).clamp_(-w_scale, w_scale)
        
    return torch.cat([v_s, theta_s, w_s], dim=-1)

# candidates * horizon * action_size
def calcAction2(action0, diff_actions, sec):
    # !! --- diff_actions \in [-1, 1] --- !! 

    T = diff_actions.size()[1]
    candidates = diff_actions.size()[0]

    actions = torch.zeros_like(diff_actions)
    v_s     = actions[:, :, 0]
    theta_s = actions[:, :, 1]
    w_s     = actions[:, :, 2]

    v_scale     = 1.0
    theta_scale = math.pi / 2.0
    w_scale     = math.pi / 4.0

    limit_v     = v_scale     * sec
    limit_theta = theta_scale * sec
    limit_w     = w_scale     * sec

    delta_v     = diff_actions[:, :, 0] * limit_v
    delta_theta = diff_actions[:, :, 1] * limit_theta
    delta_w     = diff_actions[:, :, 2] * limit_w

    v_s[:, 0]     = (action0[:, 0] + delta_v[:, 0]).clamp_(-v_scale, v_scale)

    theta_s[:, 0] = action0[:, 1] + delta_theta[:, 0]

    for c in range(candidates):
        if theta_s[c, 0] > math.pi:
            theta_s[c, 0] -= 2*math.pi
        elif theta_s[c, 0] < -math.pi:
            theta_s[c, 0] += 2*math.pi

    w_s[:, 0]     = (action0[:, 2] + delta_w[:, 0]).clamp_(-w_scale, w_scale)

    for t in range(1, T):
        v_s[:, t]     = (v_s[:, t-1] + delta_v[:, t]).clamp_(-v_scale, v_scale)

        theta_s[:, t] = theta_s[:, t-1] + delta_theta[:, t]

        for c in range(candidates):
            if theta_s[c, t] > math.pi:
                theta_s[c, t] -= 2*math.pi
            elif theta_s[c, t] < -math.pi:
                theta_s[c, t] += 2*math.pi

        w_s[:, t]     = (w_s[:, t-1] + delta_w[:, t]).clamp_(-w_scale, w_scale)
        
    return torch.cat([v_s.view(candidates, T, 1), theta_s.view(candidates, T, 1), w_s.view(candidates, T, 1)], dim=-1)

@jit.script
# candidates * horizon * action_size
def calcAction3(action0, diff_actions):
    # !! --- diff_actions \in [-1, 1] --- !! 

    T = diff_actions.size()[1]
    candidates = diff_actions.size()[0]

    actions = torch.zeros_like(diff_actions)
    v_s     = actions[:, :, 0]
    theta_s = actions[:, :, 1]
    w_s     = actions[:, :, 2]

    v_scale     = 1.0
    theta_scale = math.pi / 2.0
    w_scale     = math.pi / 4.0

    limit_v     = v_scale     
    limit_theta = theta_scale 
    limit_w     = w_scale     

    delta_v     = diff_actions[:, :, 0] * limit_v
    delta_theta = diff_actions[:, :, 1] * limit_theta
    delta_w     = diff_actions[:, :, 2] * limit_w

    v_s[:, 0]     = (action0[:, 0] + delta_v[:, 0]).clamp_(-v_scale, v_scale)

    theta_s[:, 0] = action0[:, 1] + delta_theta[:, 0]

    for c in range(candidates):
        if theta_s[c, 0] > math.pi:
            theta_s[c, 0] -= 2*math.pi
        elif theta_s[c, 0] < -math.pi:
            theta_s[c, 0] += 2*math.pi

    w_s[:, 0]     = (action0[:, 2] + delta_w[:, 0]).clamp_(-w_scale, w_scale)

    for t in range(1, T):
        v_s[:, t]     = (v_s[:, t-1] + delta_v[:, t]).clamp_(-v_scale, v_scale)

        theta_s[:, t] = theta_s[:, t-1] + delta_theta[:, t]

        for c in range(candidates):
            if theta_s[c, t] > math.pi:
                theta_s[c, t] -= 2*math.pi
            elif theta_s[c, t] < -math.pi:
                theta_s[c, t] += 2*math.pi

        w_s[:, t]     = (w_s[:, t-1] + delta_w[:, t]).clamp_(-w_scale, w_scale)
        
    return torch.cat([v_s.view(candidates, T, 1), theta_s.view(candidates, T, 1), w_s.view(candidates, T, 1)], dim=-1)


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
    parser.add_argument('--test-predict-step', type=int, default=10)

    parser.add_argument('--planning-horizon', type=int, default=10)
    parser.add_argument('--max-iters', type=int, default=10)
    parser.add_argument('--candidates', type=int, default=100)
    parser.add_argument('--top_candidates', type=int, default=10)

    parser.add_argument('--action-noise', type=float, default=0.3)

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
        drop_prob=0.0, 
        num_layers=1, 
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
    reward_plot_data = plot_graph.Plot_Graph_Data(out_dir, 'reward', {'reward': [], 'reward_per_step': []})
    plotGraph = plot_graph.Plot_Graph([train_plot_data, reward_plot_data])

    ''' ---- Initialize EpisodeMemory ---- '''

    print("Initialize EpisodeMemory")

    memory = EpisodeMemory(mem_size=args.memory_size)

    with futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_list = [executor.submit(collect_init_episode, memory_size=args.memory_size, collect_num=args.init_episode, min_step=args.chunk_size+1, clientReset=True, sec=args.sec) for i in range(args.threads)]
        future_return = futures.as_completed(fs=future_list)

    for future in future_return:
        mem = future.result()
        memory.extend(mem)

    ''' ---- Start Training ---- '''

    print("Start Training")

    env = gym.make('vaernn-v1')
    env.setting(sec=args.sec)

    f = open(out_dir + '/done_position.txt', 'w')

    for epoch in range(1, args.epochs+1):

        ''' ---- Model fitting ---- '''

        datas_observations, datas_actions, datas_rewards, datas_dones, next_observations = memory.sample(n=args.batch_size, L=args.chunk_size)
        mse_loss, KLD_loss, z  = vae_train.train2(datas_observations.view(-1, args.batch_size, 1080))
        _, _, z2 = vae_train.train2(next_observations)
        vae_loss = mse_loss+KLD_loss

        z  = z.detach()
        z2 = z2.detach()

        rnn_loss, predicts, hiddens = rnn_train.train(z.view(args.batch_size, args.chunk_size, latent_size), datas_actions, z2.view(args.batch_size, 1, latent_size))
        reward_loss = reward_train.train(predicts.view(args.batch_size, -1).detach(), hiddens.view(args.batch_size, -1).detach(), datas_rewards[:, -1].view(args.batch_size, 1)) 

        plotGraph.addDatas('train_loss', ['vae_loss', 'mse_loss', 'KLD_loss', 'rnn_loss', 'reward_loss'], [vae_loss, mse_loss, KLD_loss, rnn_loss, reward_loss])

        ''' ---- Data collection ---- '''

        with torch.no_grad():
            reward_sum = 0.0
            vae_train.vae.eval()
            rnn_train.rnn.eval()
            reward_train.rewardModel.eval()

            episode = Episode()

            env.reset(clientReset=False)
            observation = env.observe()[:1080]

            old_actions = torch.tensor([env.sim.action], device=device).view(1, 1, -1)
            old_states  = vae_train.vae(torch.tensor([observation], device=device).view(-1, 1, 1080))[1].view(1, 1, -1)

            step = 1
            # while True:
            for i in range(1, max_step+1):

                state = vae_train.vae(torch.tensor([observation], device=device).view(-1, 1, 1080))[1].view(1, 1, -1)

                diff_action = planner(rnn_train.rnn, reward_train.rewardModel, old_actions, old_states, state, args.planning_horizon, args.max_iters, args.candidates, args.top_candidates, args.sec)

                diff_action = diff_action + args.action_noise * torch.randn_like(diff_action)
                diff_action.clamp_(min=-1.0, max=1.0)
                
                next_observation, reward, done, _ = env.step(diff_action.cpu().numpy())

                action = torch.tensor([env.sim.action.astype(np.float32)], device=device).view(1, 1, -1).float()
                old_actions = torch.cat([old_actions, action], dim=1)
                old_states  = torch.cat([old_states, state], dim=1)

                if old_actions.size()[1] > args.chunk_size:
                    old_actions = old_actions[:, -args.chunk_size:, :]
                    old_states  = old_states[:, -args.chunk_size:, :]

                episode.append(observation, env.sim.action, reward, done)
                observation = next_observation[:1080]

                reward_sum += reward

                step = i

                if done:
                    robotPos, robotOri = env.sim.getRobotPosInfo()
                    f.write('{:4d}: x:{:2.4f}, y:{:2.4f}, t:{:2.4f}\n'.format(epoch, robotPos[0], robotPos[1], robotOri[2])) 
                    break

            memory.append(episode)

            plotGraph.addDatas('reward', ['reward', 'reward_per_step'], [reward_sum, reward_sum/step])

        ''' ---- Save Model ---- '''
        
        if epoch%10 == 0:
            vae_train.save(out_dir+'/vae.pth')
            rnn_train.save(out_dir+'/rnn.pth')
            reward_train.save(out_dir+'/reward.pth')

            plotGraph.plot('train_loss')
            plotGraph.plot('reward')

        print('epoch [{}/{}], vae_loss: {:.4f}, rnn_loss: {:.4f} reward_loss: {} '.format(
            epoch,
            args.epochs,
            vae_loss,
            rnn_loss,
            reward_loss)
            )       

        ''' ---- Test ---- '''

        if epoch % (args.epochs//10) == 0:
            vae_train.vae.eval()
            rnn_train.rnn.eval()
            reward_train.rewardModel.eval()

            with torch.no_grad():
                reward_sum = 0.0

                env.reset(clientReset=False)
                observation = env.observe()[:1080]

                old_actions = torch.tensor([env.sim.action], device=device).view(1, 1, -1)
                old_states  = vae_train.vae(torch.tensor([observation], device=device).view(-1, 1, 1080))[1].view(1, 1, -1)

                height = 800
                width = 800
                
                fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  
                video = cv2.VideoWriter(out_dir + '/test{}.mp4'.format(epoch), fourcc, 10, (height, width))
    
                img = imshowLocalDistance('vaernn_v0', height, width, env.lidar, next_observation, maxLen=1.0, show=False, line=True)
                video.write(img)

                # while True:
                for _ in range(max_step):

                    state = vae_train.vae(torch.tensor([observation], device=device).view(-1, 1, 1080))[1].view(1, 1, -1)

                    diff_action = planner(rnn_train.rnn, reward_train.rewardModel, old_actions, old_states, state, args.planning_horizon, args.max_iters, args.candidates, args.top_candidates, args.sec)
                    
                    next_observation, reward, done, _ = env.step(diff_action.cpu().numpy())

                    action = torch.tensor([env.sim.action.astype(np.float32)], device=device).view(1, 1, -1).float()
                    old_actions = torch.cat([old_actions, action], dim=1)
                    old_states  = torch.cat([old_states, state], dim=1)

                    if old_actions.size()[1] > args.chunk_size:
                        old_actions = old_actions[:, -args.chunk_size:, :]
                        old_states  = old_states[:, -args.chunk_size:, :]

                    observation = next_observation[:1080]

                    img = imshowLocalDistance('vaernn_v0', height, width, env.lidar, next_observation, maxLen=1.0, show=False, line=True)
                    video.write(img)

                    reward_sum += reward

                    if done:
                        break

                video.release()
                
            with torch.no_grad():
                datas_observations, datas_actions, datas_rewards, datas_dones, next_observations = memory.sample(n=args.test_batch_size, L=args.chunk_size+args.test_predict_step)

                obs = datas_observations.to(device)
                act = datas_actions.to(device)
                obs2 = next_observations.to(device)

                recon_x,  z,  _ = vae_train.vae(obs.view(-1, 1, 1080))
                z       = z.view(args.test_batch_size, args.chunk_size+args.test_predict_step, -1)
                recon_x = recon_x.view(args.test_batch_size, args.chunk_size+args.test_predict_step, -1)

                recon_x2,  z2,  _ = vae_train.vae(obs2.view(-1, 1, 1080))
                z2       = z2.view(args.test_batch_size, 1, -1)
                recon_x2 = recon_x2.view(args.test_batch_size, 1, -1)

                inp = z[:, 0:args.chunk_size, :]
                out = inp

                for i in range(args.test_predict_step):

                    predict, hidden = rnn_train.rnn(act[:, i:i+args.chunk_size, :].view(args.test_batch_size, args.chunk_size, -1), inp.view(args.test_batch_size, args.chunk_size, -1))

                    out = torch.cat([out, predict], dim=1)

                    new_inp = torch.cat([inp, predict], dim=1)
                    inp = new_inp[:, 1:, :]

                recon_predict = vae_train.vae.decoder(out)

            zs = torch.cat([z, z2], dim=1)
            recon_xs = torch.cat([recon_x, recon_x2], dim=1)

            save_image(torch.cat([obs.view(-1, 1080), recon_x.view(-1, 1080),], dim=1), '{}/result-vae{}.png'.format(out_dir, epoch))
            save_image(torch.cat([zs[:, 1:, :].reshape(-1, rnn_train.rnn.latent_size), out.view(-1, rnn_train.rnn.latent_size)], dim=1), '{}/result-rnn{}.png'.format(out_dir, epoch))
            save_image(torch.cat([recon_xs[:, 1:, :].reshape(-1, 1080), recon_predict.reshape(-1, 1080)], dim=1), '{}/result-rnn_recon{}.png'.format(out_dir, epoch))

    f.close()

    print("save:epoch", epoch)
    print(datetime.datetime.now())