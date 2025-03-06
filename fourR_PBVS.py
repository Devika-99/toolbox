import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

l1=1
l2=1
l3=1
l4=1
fourR_planar = rb.DHRobot([
    rb.RevoluteDH(a=l1, alpha=0),  # First joint
    rb.RevoluteDH(a=l2, alpha=0),  # Second joint
    rb.RevoluteDH(a=l3, alpha=0),  # Third joint
    rb.RevoluteDH(a=l4, alpha=0)   # Fourth joint
])

qr=[np.pi/6,np.pi/6,np.pi/6,np.pi/6]
Tee = fourR_planar.fkine(qr)
print(Tee)

camera = CentralCamera.Default(pose=Tee)
print(f"Camera initial pose:{camera.pose}")

def forward_kinematics(q, l):
    """ Compute forward kinematics """
    x = l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]) + l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    z = l[0] * np.sin(q[0]) + l[1] * np.sin(q[0] + q[1]) + l[2] * np.sin(q[0] + q[1] + q[2]) + l[3] * np.sin(np.sum(q))
    theta = np.sum(q)
    return np.array([x, z, theta])

def jacobian(q, l):
    """ Compute Jacobian matrix """
    J = np.zeros((3, 4))
    
    J[0, 0] = -l[0] * np.sin(q[0]) - l[1] * np.sin(q[0] + q[1]) - l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 1] = -l[1] * np.sin(q[0] + q[1]) - l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 2] = -l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 3] = -l[3] * np.sin(np.sum(q))
    
    J[1, 0] = l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]) + l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 1] = l[1] * np.cos(q[0] + q[1]) + l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 2] = l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 3] = l[3] * np.cos(np.sum(q))
    
    J[2, :] = 1  # Orientation dependency
    
    return J

def inverse_kinematics_4R(goal, q_init, l, alpha=0.1, tol=1e-3, max_iters=1000):
    q = np.array(q_init, dtype=np.float64)
    for _ in range(max_iters):
        kinematics = forward_kinematics(q, l)
        error = goal - kinematics

        if np.linalg.norm(error) < tol:
            break
        
        J = jacobian(q, l)
        J_pseudo_inv = np.linalg.pinv(J)
        q += alpha * J_pseudo_inv @ error
    
    return q



centre_point=np.array([3,2,0.1])

P=np.array([[centre_point[0]-0.2,centre_point[0]-0.2,centre_point[0]+0.2,centre_point[0]+0.2],
            [centre_point[1]-0.2,centre_point[1]+0.2,centre_point[1]+0.2,centre_point[1]-0.2],
            [0,0,0,0]])

print(P)

T_Cd_G = SE3.Tz(0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
l = [1.0, 1.0, 1.0, 1.0] 
ax.scatter(P[0, :], P[1, :], P[2, :], c='g', marker='o')
theta1=np.pi/6
theta2=np.pi/6
theta3=np.pi/6
theta4=np.pi/6
for i in range(200):
    
    objpose_in_camera = camera.pose.inv()*SE3.Trans(centre_point)
    p = camera.project_point(P, objpose=objpose_in_camera)
    Te_C_G = camera.estpose(P, p, frame="camera")
    T_delta = Te_C_G * T_Cd_G.inv()
    camera.pose = camera.pose * T_delta.interp1(0.05)
    print(f"Camera pose:{camera.pose}")
    x, y, z = camera.pose.t
    print(f"x,y,z:{x,y,z}")
    goal=[x,y,theta1+theta2+theta3+theta4]
    theta_1, theta_2,theta_3,theta_4 = inverse_kinematics_4R(goal,qr,l)
    print(f"theta:{theta_1,theta_2,theta_3,theta_4}")
    fourR_planar.q = [theta_1, theta_2,theta_3,theta_4]
    theta1=theta_1
    theta2=theta_2
    theta3=theta_3
    theta4=theta_4
    ax.cla()
    fourR_planar.plot(fourR_planar.q)  
    print(f"ee position:{fourR_planar.fkine(fourR_planar.q)}")
    ax.scatter(camera.pose.t[0], camera.pose.t[1], camera.pose.t[2], c='r', marker='x')  # Mark camera pose

    ax.scatter(P[0, :], P[1, :], P[2, :], c='g', marker='o')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.pause(0.01)  
