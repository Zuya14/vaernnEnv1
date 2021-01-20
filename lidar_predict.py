import numpy as np
import cv2
import math

def predict(points, v, theta_world, w, dt):
    
    dv = v * sec
    d = np.array([dv*np.cos(theta_world), dv*np.sin(theta_world)])
    cos = np.cos(-w*sec)
    sin = np.sin(-w*sec)
    next_points = np.array([[cos*p[0]-sin*p[1], sin*p[0]+cos*p[1]] for p in points - d])

    return next_points