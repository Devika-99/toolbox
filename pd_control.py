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
steps=50
theta=np.linspace(0,2*np.pi,steps)

x_d=centre_x+radius*np.cos(theta)
y_d=centre_y+radius*np.sin(theta)
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

J_sym = Matrix(pos_end).jacobian(Matrix(q_sym))
J_dot_sym=diff(J_sym,t)
J_dot_sym = J_dot_sym.subs({
    diff(q1, t): q1_dot,
    diff(q2, t): q2_dot,
    diff(q3, t): q3_dot,
    diff(q4, t): q4_dot
})

q_vals = np.array([[0], [0], [0], [0]])    
q_dot_vals = np.array([[0], [0], [0], [0]])          
q_ddot_vals = np.array([[0], [0], [0], [0]]) 

Kp=5
Kd=2

ee_positions = [] 

for i in range(steps-2):
    q1_val=float(q_vals[0])
    q2_val=float(q_vals[1])
    q3_val=float(q_vals[2])
    q4_val=float(q_vals[3])

    q1_dot_val=q_dot_vals[0]
    q2_dot_val=q_dot_vals[1]
    q3_dot_val=q_dot_vals[2]
    q4_dot_val=q_dot_vals[3]

    J= np.array(J_sym.subs({q1: q1_val, q2: q2_val, q3: q3_val, q4: q4_val}),dtype=np.float64)

    J_dot = np.array(J_dot_sym.subs({
        q1: q1_val, 
        q2: q2_val, 
        q3: q3_val, 
        q4: q4_val,
        q1_dot: q1_dot_val.item(),  
        q2_dot: q2_dot_val.item(),  
        q3_dot: q3_dot_val.item(),  
        q4_dot: q4_dot_val.item()
    }))

        
    x_e=x_end.subs({q1:q1_val,q2:q2_val,q3:q3_val,q4:q4_val})
    y_e=y_end.subs({q1:q1_val,q2:q2_val,q3:q3_val,q4:q4_val})
    ee_positions.append((x_e, y_e))
    x_e_dot,y_e_dot=np.matmul(J,q_dot_vals)
    e=np.array([[x_d[i]-x_e],[y_d[i]-y_e]])
    e_dot=np.array([[x_dot_d[i]-x_e_dot],[y_dot_d[i]-y_e_dot]]).reshape(2,1)
    e_ddot=x_ddot_d[i]-np.matmul(J,q_ddot_vals)-np.matmul(J_dot,q_dot_vals)
    # q_ddot=np.matmul(np.linalg.pinv(J),x_ddot_d[i]+Kd*e_dot+Kp*e-np.matmul(J_dot,q_dot_vals))
    q_ddot=np.matmul(np.linalg.pinv(J),Kd*e_dot+Kp*e+e_ddot)

    tspan = (i * dt, (i + 1) * dt)
    q_dot_vals=q_dot_vals+q_ddot*dt
    q_vals=q_vals+q_dot_vals*dt
    fourR_planar.plot(q_vals.reshape(-1,1).flatten())


ee_positions = np.array(ee_positions)
print(ee_positions)
plt.plot(ee_positions[:, 0], ee_positions[:, 1], 'ro-', label="End-Effector Path")

# Labels and grid
plt.xlabel("X position")
plt.ylabel("Y position")
plt.legend()
plt.grid()
plt.show()