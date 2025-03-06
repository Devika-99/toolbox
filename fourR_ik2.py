#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64
import sympy as sp

# Define symbolic variables for joint angles q1, q2, q3, q4
q1, q2, q3, q4 = sp.symbols('q1 q2 q3 q4')

# Goal point (2D position + orientation)
goal_point = np.array([1.0, 1.0, 0.3])  # Goal point in x, z, orientation space
xf = goal_point  # Goal coordinates

# Initial configuration (joint angles in radians)
init_config = np.array([0.1, 0.1, 0.1, 0.1])  # Angles in radians
q0 = init_config.T  # Initial joint angles in radians

# Link lengths for the 4-DOF robot arm
a1, a2, a3, a4 = 0.4, 0.4, 0.4, 0.4  # Lengths in meters

# Forward kinematics for the 4-DOF 2D robot arm (x, z, orientation)
kinematics = np.array([
    a1 * sp.cos(q1) + a2 * sp.cos(q1 + q2) + a3 * sp.cos(q1 + q2 + q3) + a4 * sp.cos(q1 + q2 + q3 + q4),  # x
    a1 * sp.sin(q1) + a2 * sp.sin(q1 + q2) + a3 * sp.sin(q1 + q2 + q3) + a4 * sp.sin(q1 + q2 + q3 + q4),  # z
    q1 + q2 + q3 + q4  # Orientation (sum of joint angles)
])

# Compute the Jacobian matrix (partial derivatives of kinematics w.r.t. q1, q2, q3, q4)
Jacobian = sp.Matrix([
    [sp.diff(kinematics[0], q1), sp.diff(kinematics[0], q2), sp.diff(kinematics[0], q3), sp.diff(kinematics[0], q4)],
    [sp.diff(kinematics[1], q1), sp.diff(kinematics[1], q2), sp.diff(kinematics[1], q3), sp.diff(kinematics[1], q4)],
    [sp.diff(kinematics[2], q1), sp.diff(kinematics[2], q2), sp.diff(kinematics[2], q3), sp.diff(kinematics[2], q4)]
])

# Initialize ROS node
rospy.init_node('inverse_kinematics_publisher', anonymous=True)


# Learning rate for the IK solver
xi = 0.01  # Increased learning rate for faster convergence
error_margin = 0.05  # A bit larger error margin for faster convergence

# Initial joint positions (set to initial configuration)
q = np.array(q0)

# Debug: Check the Jacobian at the initial configuration
Jacobian_at_q0 = np.array([
    [float(Jacobian[0, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[0, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[0, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[0, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))],
    [float(Jacobian[1, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[1, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[1, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[1, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))],
    [float(Jacobian[2, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[2, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[2, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
     float(Jacobian[2, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))]
])

print(f"Jacobian at initial configuration:\n{Jacobian_at_q0}")

# Iterate to solve the IK problem using pseudo-inverse of the Jacobian
max_iterations = 1000  # Reduced number of iterations

for i in range(max_iterations):
    # Substitute current joint angles into the kinematics and Jacobian
    kinematics_ = np.array([
        float(kinematics[0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
        float(kinematics[1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
        float(kinematics[2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))
    ])

    # Evaluate the Jacobian at the current joint angles
    J_hash_ = np.array([
        [float(Jacobian[0, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[0, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[0, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[0, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))],
        [float(Jacobian[1, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[1, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[1, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[1, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))],
        [float(Jacobian[2, 0].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[2, 1].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[2, 2].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]})),
         float(Jacobian[2, 3].subs({q1: q[0], q2: q[1], q3: q[2], q4: q[3]}))]
    ])

    # Calculate the error between the current end-effector position and the goal point
    dist = np.linalg.norm(xf - kinematics_)

    # Compute the change in joint angles using the pseudo-inverse of the Jacobian
    Jt_inv = np.linalg.pinv(J_hash_)  # Pseudo-inverse for numerical stability
    q_dot = xi * np.dot(Jt_inv, xf - kinematics_)

    # Update joint positions
    q = q + q_dot

    # Break if the error is smaller than the margin
    if dist < error_margin:
        rospy.loginfo(f"Converged in {i+1} iterations")
        break

