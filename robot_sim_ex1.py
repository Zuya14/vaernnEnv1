import cv2
import numpy as np
import pybullet as p
import pybullet_utils.bullet_client as bc
import pybullet_data

import math
import copy

import lidar_util

import random

robot_name = "urdf/robot.urdf"
dynamic_body0 = "urdf/door.urdf"
dynamic_body1 = "urdf/door2.urdf"

class robot_sim:

    def __init__(self, _id, mode=p.DIRECT, sec=0.01):
        self._id = _id
        self.mode = mode
        self.phisicsClient = bc.BulletClient(connection_mode=mode)
        # print(self.phisicsClient)
        self.reset(sec=sec)
        
    def copy(self, _id):
        new_sim = robot_sim(_id, self.mode)
        new_sim.robotPos = copy.deepcopy(self.robotPos)
        new_sim.robotOri = copy.deepcopy(self.robotOri)

        new_sim.reset(
            x=new_sim.robotPos[0], 
            y=new_sim.robotPos[1], 
            theta=p.getEulerFromQuaternion(new_sim.robotOri)[2], 
            vx = self.vx,
            vy = self.vy,
            w = self.w,
            sec=self.sec,
            dynamic_counter=self.dynamic_counter, 
            interval=self.dynamic_interval
            )

        new_sim.updateRobotInfo()

        return new_sim

    def reset(self, x=0.0, y=0.0, theta=0.0, vx=0.0, vy=0.0, w=0.0, sec=0.01, dynamic_counter=0.0, interval=3.0, action=None, clientReset=False):
        if clientReset:
            self.phisicsClient = bc.BulletClient(connection_mode=self.mode)

        self.sec = sec
        if dynamic_counter < 0.0:
            self.dynamic_counter = random.uniform(0.0, interval)
            # self.dynamic_counter2 = random.uniform(0.0, interval)
        else:
            self.dynamic_counter = dynamic_counter
            # self.dynamic_counter2 = dynamic_counter
        self.dynamic_interval = interval

        self.vx = vx
        self.vy = vy
        self.w = w

        self.phisicsClient.resetSimulation()
        self.robotUniqueId = 0 
        self.bodyUniqueIds = []
        self.dynamic_bodyUniqueIds = []
        # self.phisicsClient.setGravity(0, 0, -9.8)
        self.phisicsClient.setTimeStep(sec)

        self.action = action if action is not None else [0.0, 0.0, 0.0]
        
        self.done = False

        self.loadBodys(x, y, theta)

    def getId(self):
        return self._id

    def loadBodys(self, x, y, theta):
        self.robotPos = (x,y,0)
        self.robotOri = p.getQuaternionFromEuler([0, 0, theta])

        self.robotUniqueId = self.phisicsClient.loadURDF(
            robot_name,
            basePosition=self.robotPos,
            baseOrientation = self.robotOri
            )
        
        self.bodyUniqueIds += [self.phisicsClient.loadURDF("urdf/wall_ex.urdf")]
        self.dynamic_bodyUniqueIds += [self.phisicsClient.loadURDF(dynamic_body0)]
        # self.dynamic_bodyUniqueIds += [self.phisicsClient.loadURDF(dynamic_body1)]

    def step(self, action):

        if not self.done:
            self.action = self.calcAction(action)

            _, ori = self.getRobotPosInfo()
            yaw = p.getEulerFromQuaternion(ori)[2]
            dtheta = yaw + self.action[1]

            sin = np.sin(dtheta)
            cos = np.cos(dtheta)

            self.vx = self.action[0] * -sin
            self.vy = self.action[0] * cos

            self.w = self.action[2]

            self.updateRobotInfo()

            self.phisicsClient.stepSimulation()

            if self.isContacts():
                self.done = True

            if self.robotPos[1] > 6.0:
                self.done = True
        else:
            self.vx = 0
            self.vy = 0
            self.w = 0

        self.dynamic_counter += self.sec
        if self.dynamic_counter > self.dynamic_interval:
            self.dynamic_counter -= self.dynamic_interval

        # self.dynamic_counter2 += self.sec
        # if self.dynamic_counter2 > self.dynamic_interval:
        #     self.dynamic_counter2 -= self.dynamic_interval
        self.updateDynamicBody()

        return self.done
        
    def calcAction(self, action):
        # tmp = 2.0 * (action - 0.5)
        tmp = action

        v_scale     = 1.0
        theta_scale = math.pi / 2.0
        w_scale     = math.pi / 4.0

        limit_v     = v_scale     * self.sec
        limit_theta = theta_scale * self.sec
        limit_w     = w_scale     * self.sec

        delta_v     = tmp[0] * limit_v
        delta_theta = tmp[1] * limit_theta
        delta_w     = tmp[2] * limit_w

        v     = np.clip(self.action[0] + delta_v,     -v_scale,     v_scale)

        # theta = np.clip(self.action[1] + delta_theta, -theta_scale, theta_scale)
        theta = self.action[1] + delta_theta
        if theta > math.pi:
            theta -= 2*math.pi
        elif theta < -math.pi:
            theta += 2*math.pi

        w     = np.clip(self.action[2] + delta_w,     -w_scale,     w_scale)

        return np.array([v, theta, w])

    def observe(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanDist = bullet_lidar.scanDistance(self.phisicsClient, pos, yaw, height=0.9)
        self.scanDist = scanDist / bullet_lidar.maxLen
        # self.scanDist = (scanDist + np.random.normal(0, 0.03, scanDist.shape[0])) / bullet_lidar.maxLen
        self.scanDist = self.scanDist.astype(np.float32)
        # self.scanDist = np.clip(self.scanDist.astype(np.float32), 0.0, 1.0)
        
        return self.scanDist

    def observe2d(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanPoints = bullet_lidar.scanPosLocal(self.phisicsClient, pos, yaw, height=0.9)[:, :2]
        
        return scanPoints

    def render(self, bullet_lidar):
        pos, ori = self.getRobotPosInfo()
        yaw = p.getEulerFromQuaternion(ori)[2]
        scanPosLocal = bullet_lidar.scanPosLocal(self.phisicsClient, pos, yaw, height=0.9)

        # img = imshowLocal(name="render"+str(self.phisicsClient), h=480, w=640, points=scanPosLocal, maxLen=bullet_lidar.maxLen, show=False, line=True)
        img = lidar_util.imshowLocal(name="render"+str(self.phisicsClient), h=800, w=800, points=scanPosLocal, maxLen=bullet_lidar.maxLen, show=False, line=True)

        return img

    def updateDynamicBody(self):

        if self.dynamic_counter < self.dynamic_interval/2.0:
            x = 1.5 * (2.0 * self.dynamic_counter / self.dynamic_interval)
        else:
            x = 1.5 * (2.0 * (self.dynamic_interval-self.dynamic_counter) / self.dynamic_interval)

        _id = self.phisicsClient.loadURDF(
            dynamic_body0,
            basePosition=[x, 0, 0]
            )
        self.phisicsClient.removeBody(self.dynamic_bodyUniqueIds[0])
        self.dynamic_bodyUniqueIds[0] = _id


        # if self.dynamic_counter2 < self.dynamic_interval/2.0:
        #     x = 1.5 * (2.0 * self.dynamic_counter2 / self.dynamic_interval)
        # else:
        #     x = 1.5 * (2.0 * (self.dynamic_interval-self.dynamic_counter2) / self.dynamic_interval)

        # _id = self.phisicsClient.loadURDF(
        #     dynamic_body1,
        #     basePosition=[x, 0, 0]
        #     )
        # self.phisicsClient.removeBody(self.dynamic_bodyUniqueIds[1])
        # self.dynamic_bodyUniqueIds[1] = _id

    def updateRobotInfo(self):

        self.phisicsClient.resetBaseVelocity(
            self.robotUniqueId,
            linearVelocity=[self.vx, self.vy, 0],
            angularVelocity=[0, 0, self.w]
            )

        self.robotPos, self.robotOri = self.phisicsClient.getBasePositionAndOrientation(self.robotUniqueId)

    def getRobotPosInfo(self):
        return self.robotPos, self.robotOri

    def getState(self):
        pos, ori = self.getRobotPosInfo()
        return pos[0], pos[1], p.getEulerFromQuaternion(ori)[2], self.vx, self.vy, self.w 

    def close(self):
        self.phisicsClient.disconnect()

    def contacts(self):
        list = []
        for i in self.bodyUniqueIds: # 接触判定
            list += self.phisicsClient.getContactPoints(self.robotUniqueId, i)
        for i in self.dynamic_bodyUniqueIds: # 接触判定
            list += self.phisicsClient.getContactPoints(self.robotUniqueId, i)
        return list

    def isContacts(self):
        return len(self.contacts()) > 0

    def isDone(self):
        return self.done

if __name__ == '__main__':

    # sim = robot_sim(0, mode=p.GUI)
    sim = robot_sim(0, mode=p.GUI, sec=0.05)
    # sim = robot_sim(0, mode=p.DIRECT)
    sim.reset(dynamic_counter=-1.0)
    from bullet_lidar import bullet_lidar

    deg_offset = 90.
    rad_offset = deg_offset*(math.pi/180.0)
    startDeg = -135. + deg_offset
    endDeg = 135. + deg_offset
    resolusion = 0.25
    maxLen = 8.
    minLen = 0.
    lidar = bullet_lidar(startDeg, endDeg, resolusion, maxLen, minLen)

    action = np.array([0.0]*3)

    while True:
        sim.step(action=action)
        cv2.imshow("robot_sim", sim.render(lidar))

        if sim.isContacts():
            print("done")
            cv2.waitKey(0)
            break

        if cv2.waitKey(1) >= 0:
            break