import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from roboticstoolbox import DHRobot, DHLink
from scipy.integrate import solve_ivp

def initialize_robot():
    L1 = DHLink(a=1, alpha=0, d=0, offset=0, m=1, r=[0.5, 0, 0], I=[0.1, 0.1, 0.1, 0, 0, 0], Jm=0.01)
    L2 = DHLink(a=1, alpha=0, d=0, offset=0, m=1, r=[0.5, 0, 0], I=[0.1, 0.1, 0.1, 0, 0, 0], Jm=0.01)
    return DHRobot([L1, L2], name='2R Robot')

def generate_trajectory(centre_x=1, centre_y=1, radius=0.5, steps=500, dt=0.1):
    theta = np.linspace(0, 2*np.pi, steps)
    x_d = centre_x + radius * np.cos(theta)
    y_d = centre_y + radius * np.sin(theta)
    x_dot_d = np.gradient(x_d, dt)
    y_dot_d = np.gradient(y_d, dt)
    x_ddot_d = np.gradient(x_dot_d, dt)
    y_ddot_d = np.gradient(y_dot_d, dt)
    return x_d, y_d, x_dot_d, y_dot_d, x_ddot_d, y_ddot_d

def compute_jacobian():
    q1, q2, q1_dot, q2_dot = sp.symbols('q1 q2 q1_dot q2_dot', real=True)
    x_end = sp.cos(q1) + sp.cos(q1 + q2)
    y_end = sp.sin(q1) + sp.sin(q1 + q2)
    pos_end = sp.Matrix([x_end, y_end])
    
    J_sym = pos_end.jacobian([q1, q2])
    J_dot_sym = J_sym.diff(q1) * q1_dot + J_sym.diff(q2) * q2_dot
    
    J_func = sp.lambdify((q1, q2), J_sym, 'numpy')
    J_dot_func = sp.lambdify((q1, q2, q1_dot, q2_dot), J_dot_sym, 'numpy')
    return J_func, J_dot_func

def compute_control(robot, q, q_dot, J_func, J_dot_func, x_d, y_d, x_dot_d, y_dot_d, x_ddot_d, y_ddot_d, i, Kp=2, Kd=3):
    q1_val, q2_val = q
    q1_dot_val, q2_dot_val = q_dot
    J = np.array(J_func(q1_val, q2_val), dtype=float)
    J_dot = np.array(J_dot_func(q1_val, q2_val, q1_dot_val, q2_dot_val), dtype=float)
    
    x_observed = np.cos(q1_val) + np.cos(q1_val + q2_val)
    y_observed = np.sin(q1_val) + np.sin(q1_val + q2_val)
    
    e = np.array([x_d[i] - x_observed, y_d[i] - y_observed])
    e_dot = np.array([x_dot_d[i] - (J @ q_dot)[0], y_dot_d[i] - (J @ q_dot)[1]])
    
    q_ddot = np.linalg.pinv(J) @ (Kp * e + Kd * e_dot + np.array([x_ddot_d[i], y_ddot_d[i]]) - J_dot @ q_dot)
    
    return q_ddot, e, e_dot, J, J_dot

def simulate_dynamics(robot, q, q_dot, q_ddot, J, J_dot, e, e_dot, i, dt):
    B = robot.inertia(q)
    C = robot.coriolis(q, q_dot)
    G = robot.gravload(q)
    
    u = B @ q_ddot + C @ q_dot + G

    def integ(t, y, robot, u):
        q = y[:2]
        q_dot = y[2:]
        B = robot.inertia(q)
        C = robot.coriolis(q, q_dot)
        G = robot.gravload(q)
        q_ddot = np.linalg.pinv(B) @ (u - C @ q_dot - G)
        dydt = np.concatenate([q_dot, q_ddot])
        return dydt

    tspan = [i*dt, (i+1)*dt]
    y0 = np.concatenate([q.flatten(), q_dot.flatten()])
    sol = solve_ivp(integ, tspan, y0, args=(robot, u), method="RK45")

    q_new = sol.y[:2, -1]
    q_dot_new = sol.y[2:, -1]

    return q_new, q_dot_new

def main():
    robot = initialize_robot()
    x_d, y_d, x_dot_d, y_dot_d, x_ddot_d, y_ddot_d = generate_trajectory()
    J_func, J_dot_func = compute_jacobian()
    
    dt = 0.1
    steps = 500
    q = np.array([0.1, 0.1], dtype=float)
    q_dot = np.array([0, 0], dtype=float)

    plt.ion()
    fig, ax = plt.subplots()
    ax.plot(x_d, y_d, 'b--', label='Desired Trajectory')

    trajectory = np.zeros((steps, 2))
    
    for i in range(steps):
        q_ddot, e, e_dot, J, J_dot = compute_control(robot, q, q_dot, J_func, J_dot_func, x_d, y_d, x_dot_d, y_dot_d, x_ddot_d, y_ddot_d, i)
        
        q_new, q_dot_new = simulate_dynamics(robot, q, q_dot, q_ddot, J, J_dot, e, e_dot, i, dt)

        trajectory[i] = [np.cos(q_new[0]) + np.cos(q_new[0] + q_new[1]), np.sin(q_new[0]) + np.sin(q_new[0] + q_new[1])]
        ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 1], 'r', label='Actual Trajectory' if i == 0 else '')

        robot.plot(q)
        plt.pause(0.01)

        q, q_dot = q_new, q_dot_new  

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
