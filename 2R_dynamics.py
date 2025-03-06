import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from roboticstoolbox import DHRobot, RevoluteDH
from scipy.integrate import solve_ivp
from roboticstoolbox import DHLink
import rospy


L1 = DHLink(a=1, alpha=0, d=0, offset=0, m=1, r=[0.5, 0, 0], I=[0.1, 0.1, 0.1, 0, 0, 0], Jm=0.01)
L2 = DHLink(a=1, alpha=0, d=0, offset=0, m=1, r=[0.5, 0, 0], I=[0.1, 0.1, 0.1, 0, 0, 0], Jm=0.01)


robot = DHRobot([L1, L2], name='2R Robot')

dt=0.1
steps=500

centre_x=1
centre_y=1
radius=0.5

theta=np.linspace(0,2*np.pi,steps)
x_d= centre_x+radius*np.cos(theta)
y_d= centre_y+radius*np.sin(theta)

x_dot_d=np.gradient(x_d,dt)
y_dot_d=np.gradient(y_d,dt)

x_ddot_d=np.gradient(x_dot_d,dt)
y_ddot_d=np.gradient(y_dot_d,dt)

q = np.array([0.1, 0.1], dtype=float) 
q_dot = np.array([0, 0], dtype=float) 
q_ddot = np.array([0, 0], dtype=float)  

y0 = np.concatenate([q.flatten(), q_dot.flatten()])
Kp, Kd = 2, 3
max_q_dot, max_q_ddot = 1, 1

q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot', real=True)
x_end = sp.cos(q1) + sp.cos(q1 + q2)
y_end = sp.sin(q1) + sp.sin(q1 + q2)
pos_end = sp.Matrix([x_end, y_end])

J_sym = pos_end.jacobian([q1, q2])
J_dot_sym = J_sym.diff(q1) * q1_dot + J_sym.diff(q2) * q2_dot

J_func = sp.lambdify((q1, q2), J_sym, 'numpy')
J_dot_func = sp.lambdify((q1, q2, q1_dot, q2_dot), J_dot_sym, 'numpy')

trajectory = np.zeros((steps, 2))
end_effector_vel = np.zeros((steps, 2))
end_effector_acc = np.zeros((steps, 2))
position_errors = np.zeros((steps, 2))
velocity_errors = np.zeros((steps, 2))
acceleration_errors = np.zeros((steps, 2))

plt.ion()
fig, ax = plt.subplots()
ax.plot(x_d, y_d, 'b--', label='Desired Trajectory')

def integ(t, y, robot, u):
    q = y[:2]     
    q_dot = y[2:]

    B = robot.inertia(q)
    C = robot.coriolis(q, q_dot)
    G = robot.gravload(q)
    q_ddot = np.matmul(np.linalg.pinv(B), u - C @ q_dot - G)
    dydt = np.concatenate([q_dot, q_ddot])  
    return dydt


for i in range(steps):

    q1_val, q2_val = q
    q1_dot_val, q2_dot_val = q_dot
    J = np.array(J_func(q1_val, q2_val), dtype=float)
    J_dot = np.array(J_dot_func(q1_val, q2_val, q1_dot_val, q2_dot_val), dtype=float)
    x_observed = np.cos(q1_val) + np.cos(q1_val + q2_val) 
    y_observed = np.sin(q1_val) + np.sin(q1_val + q2_val) 

    trajectory[i] = [x_observed, y_observed]
    end_effector_vel[i] = (J @ q_dot).flatten()
    end_effector_acc[i] = (J @ q_ddot + J_dot @ q_dot).flatten()
    position_errors[i] = [x_d[i] - x_observed, y_d[i] - y_observed]
    velocity_errors[i] = [x_dot_d[i] - end_effector_vel[i, 0], y_dot_d[i] - end_effector_vel[i, 1]]
    e = position_errors[i]
    e_dot = velocity_errors[i]
    q_ddot = np.linalg.pinv(J) @ (Kp * e + Kd * e_dot + np.array([x_ddot_d[i], y_ddot_d[i]]) - J_dot @ q_dot)
    print(f"q_ddot:{q_ddot}")
    y=np.matmul(np.linalg.pinv(J),[x_ddot_d[i],y_ddot_d[i]]+Kd*e_dot+Kp*e-np.matmul(J_dot,q_dot))
    print(f"y:{y}")
    B=robot.inertia(q)
    C=robot.coriolis(q,q_dot)
    G=robot.gravload(q)
    print(f"B:{B}")
    print(f"C:{C}")
    print(f"G:{G}")
    tspan=[i*dt,(i+1)*dt]
    u=np.matmul(B,y)+np.matmul(C,q_dot)+G
    print(f"y:{y}")
    print(f"u:{u}")
    sol = solve_ivp(integ, tspan, y0, args=(robot, u), method="RK45")
    # sol = solve_ivp(integ, tspan, y0, args=(robot, u), method="RK45", atol=1e-6, rtol=1e-6)

    print(f"sol:{sol}")
    q_new = sol.y[:2, -1]  
    q_dot_new = sol.y[2:, -1] 
    print(f"q_new:{q_new}")
    print(f"q_dot_new:{q_dot_new}")
    q=q_new
    q_dot=q_dot_new
    ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], 'r', label='Actual Trajectory' if i == 0 else '')
    robot.plot(q)
    y0 = np.concatenate([q.flatten(), q_dot.flatten()])
    plt.pause(0.01)



    

    


