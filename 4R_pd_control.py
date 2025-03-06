import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from roboticstoolbox import DHRobot, RevoluteDH

L1 = RevoluteDH(a=1, alpha=0)
L2 = RevoluteDH(a=1, alpha=0)
L3= RevoluteDH(a=1, alpha=0)
L4= RevoluteDH(a=1, alpha=0)

robot = DHRobot([L1, L2,L3,L4], name='2R Robot')

dt=0.1
steps=500

center_x = 2  
center_y = 2     
radius = 1

theta = np.linspace(0, 2 * np.pi, steps)
x_d = center_x + radius * np.cos(theta)
y_d = center_y + radius * np.sin(theta)

x_dot_d = np.gradient(x_d, dt)
y_dot_d = np.gradient(y_d, dt)
x_ddot_d = np.gradient(x_dot_d, dt)
y_ddot_d = np.gradient(y_dot_d, dt)

q = np.array([0, 0], dtype=float) 
q_dot = np.array([0, 0], dtype=float) 
q_ddot = np.array([0, 0], dtype=float)  
Kp, Kd = 15, 4
max_q_dot, max_q_ddot = 1, 5