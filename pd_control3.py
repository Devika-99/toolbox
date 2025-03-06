import roboticstoolbox as rb
import numpy as np
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from sympy import symbols, Function, diff, Matrix, cos, sin
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

l1, l2, l3, l4 = 1, 1, 1, 1

fourR_planar = rb.DHRobot([
    rb.RevoluteDH(a=l1, alpha=0),  
    rb.RevoluteDH(a=l2, alpha=0),
    rb.RevoluteDH(a=l3, alpha=0),  
    rb.RevoluteDH(a=l4, alpha=0)   
])

dt=0.1
centre_x=2
centre_y=2
radius=1
steps=100
theta=np.linspace(0,2*np.pi,steps)

x_d=centre_x+radius*np.cos(theta)
y_d=centre_y+radius*np.sin(theta)

desired_positions = np.column_stack((x_d, y_d))

x_dot_d=np.diff(x_d)/dt
y_dot_d=np.diff(y_d)/dt
x_ddot_d=np.diff(x_dot_d)/dt
y_ddot_d=np.diff(y_dot_d)/dt

t = symbols("t")
q1 = Function("q1")(t)
q2 = Function("q2")(t)
q3 = Function("q3")(t)
q4 = Function("q4")(t)
q1_dot, q2_dot, q3_dot, q4_dot = symbols("q1_dot q2_dot q3_dot q4_dot")  
q1_ddot, q2_ddot, q3_ddot, q4_ddot = symbols("q1_ddot q2_ddot q3_ddot q4_ddot")  
q_sym=[q1,q2,q3,q4]
q_dot_sym=[q1_dot,q2_dot,q3_dot,q4_dot]

x_end=cos(q1)+cos(q1+q2)+cos(q1+q2+q3)+cos(q1+q2+q3+q4)
y_end=sin(q1)+sin(q1+q2)+sin(q1+q2+q3)+sin(q1+q2+q3+q4)

pos_end=np.array([x_end,y_end])
print(pos_end)

# J_sym=np.array([-sin(q1)-sin(q1+q2)-sin(q1+q2+q3)-sin(q1+q2+q3+q4)],-sin(q1+q2)-sin(q1+q2+q3)-sin(q1+q2+q3+q4),-sin(q1+q2+q3)-sin(q1+q2+q3+q4),-sin(q1+q2+q3+q4)
#                [cos(q1)+cos(q1+q2)+cos(q1+q2+q3)+cos(q1+q2+q3+q4),cos(q1+q2)+cos(q1+q2+q3)+cos(q1+q2+q3+q4),cos(q1+q2+q3)+cos(q1+q2+q3+q4),cos(q1+q2+q3+q4)],
#                )
J_sym = Matrix(pos_end).jacobian(Matrix(q_sym))
J_dot_sym=diff(J_sym,t)
print(J_dot_sym)