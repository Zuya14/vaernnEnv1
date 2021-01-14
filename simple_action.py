import numpy as np
import cv2
import gym
from gym.envs.registration import register
import pybullet as p
import lidar_util
import torch
import math

torch.set_printoptions(precision=3, sci_mode=False)

def predict(points, action, dt):

    v     = action[0]
    theta = action[1]
    w     = action[2]

    dv = v*dt
    dd = np.array([dv*np.cos(theta), dv*np.sin(theta)])
    cos = np.cos(-w*dt)
    sin = np.sin(-w*dt)
    next_points = np.array([[cos*p[0]-sin*p[1], sin*p[0]+cos*p[1]] for p in points - dd])

    return next_points

def predict_tensor(points, action, dt):
    _act = action*dt

    _move_x = _act[:, 0] * torch.cos(action[:, 1])
    _move_y = _act[:, 0] * torch.sin(action[:, 1])
    cos = torch.cos(-_act[:, 2]).view(-1, 1)
    sin = torch.sin(-_act[:, 2]).view(-1, 1)
    next_points = points - torch.stack([_move_x, _move_y], dim=-1).view(-1, 1, 2)
    next_points = torch.stack([cos*next_points[:, :, 0]-sin*next_points[:, :, 1], sin*next_points[:, :, 0]+cos*next_points[:, :, 1]], dim=-1).view(-1, 1, 1080, 2)

    return next_points

def planner(action0, points, yaw):

    action0 = torch.tensor(action0).view(1, action_size)

    diff_action_mean = torch.zeros(1, planning_horizon, action_size)
    diff_action_std_dev = torch.ones(1, planning_horizon, action_size)

    for _ in range(optimisation_iters):
        diff_actions = (diff_action_mean + diff_action_std_dev * torch.randn(candidates, planning_horizon, action_size))
        diff_actions.clamp_(min=-1.0, max=1.0) 

        actions = calcAction(action0, diff_actions*sec)
        pt = points.view(1, 1, 1080, 2).repeat(candidates, 1, 1, 1) 

        vys = calcVy(yaw, actions, sec)
        # print(vys.size())

        rewards= torch.zeros(candidates, planning_horizon)

        for t in range(planning_horizon):
            pt = predict_tensor(pt[:, 0, :, :], actions[:, t, :], sec)
            # print(pt[:, :, 0, :])
            dist = pt[:, 0, :, 0]**2 + pt[:, 0, :, 1]**2
            dist = torch.sqrt(dist)
            min, min_indices = torch.min(dist, dim=1)

            reward_contact = torch.where(min <= 0.25, -100.0*torch.ones_like(min), torch.zeros_like(min))
            # reward_contact = torch.where(min <= 0.5, -torch.ones_like(min), torch.zeros_like(min))
            # reward_contact = torch.where(min <= 0.25, -torch.ones_like(min)*(1.0 + torch.abs(actions[:, t, 0])), torch.zeros_like(min))
            reward_vy = vys[:, t]
            # reward_vy = torch.where(vys[:, t] < 0, -torch.ones_like(min), torch.zeros_like(min))

            # print(reward_contact.size(), reward_vy.size())

            reward = reward_contact + reward_vy
            rewards[:, t] = reward

        # print(rewards[0, :])
        # print(rewards)

        returns = rewards.view(candidates, planning_horizon).sum(dim=1)
        _, topk = returns.topk(top_candidates, dim=0, largest=True, sorted=False)
        best_diff_actions = diff_actions[topk].reshape(top_candidates, planning_horizon, action_size)
        diff_action_mean, diff_action_std_dev = best_diff_actions.mean(dim=0, keepdim=True), best_diff_actions.std(dim=0, unbiased=False, keepdim=True)
      
    return diff_action_mean[0, 0, :].view(action_size)

def calcAction(action0, diff_actions):
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

def calcVy(yaw, actions, sec):

    candidates = actions.size()[0]
    planning_horizon = actions.size()[1]
    
    vys = torch.empty(candidates, planning_horizon)

    yaws = torch.tensor([yaw]).view(1, 1).repeat(candidates, planning_horizon)

    yaws[:, 0] = yaws[:, 0] + actions[:, 0, 2]*sec 
    vys[:, 0] = actions[:, 0, 0] * torch.cos(yaws[:, 0] + actions[:, 0, 1])

    for t in range(1, planning_horizon):
        yaws[:, t] = yaws[:, t-1] + actions[:, t, 2]*sec
        vys[:, t] = actions[:, t, 0] * torch.cos(yaws[:, t] + actions[:, t, 1])

    return vys

register(
    id='vaernn-v1',
    entry_point='vaernnEnv1:vaernnEnv1'
)

# sec = 0.01
sec = 0.1
# sec = 1

action_size = 3

optimisation_iters = 10
candidates = 100
top_candidates = 10
planning_horizon = 10

env = gym.make('vaernn-v1')
env.setting(sec=sec)

action = env.sim.action
observation = env.observe2d()[:1080]

pos, ori = env.sim.getRobotPosInfo()
yaw = p.getEulerFromQuaternion(ori)[2]

while True:

    pos, ori = env.sim.getRobotPosInfo()
    yaw = p.getEulerFromQuaternion(ori)[2]
    diff_action = planner(action, torch.tensor(observation), yaw)
    print(diff_action)

    # predict_action = env.sim.calcAction(diff_action)
    
    # predict_observation = predict(observation, predict_action, sec)
    # predict_observation_tensor = predict_tensor(, torch.tensor(predict_action), sec)

    # dist = np.array([np.sqrt(p[0]**2+p[1]**2) for p in observation])
    # predict_dist = np.array([np.sqrt(p[0]**2+p[1]**2) for p in predict_observation])

    _, reward, done, _ = env.step(diff_action)
    next_observation = env.observe2d()[:1080]

    img = lidar_util.imshowLocal(name="simple_action", h=800, w=800, points=observation, maxLen=env.lidar.maxLen, show=False, line=False)
    # img = lidar_util.imshowLocal(name="simple_action", h=800, w=800, points=predict_observation, maxLen=env.lidar.maxLen, show=False, line=False)
    # img = lidar_util.imshowLocal(name="simple_action", h=800, w=800, points=next_observation, maxLen=env.lidar.maxLen, show=False, line=False, point_color=(0,255,0), preimg=img)
    cv2.imshow("simple_action", img)

    observation = next_observation[:1080]
    action = env.sim.action

    if done:
        print("done")
        cv2.waitKey(0)
        break

    if cv2.waitKey(1) >= 0:
        break