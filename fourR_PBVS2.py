import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import logm 

l1=1
l2=1
l3=1
l4=1
fourR_planar = rb.DHRobot([
    rb.RevoluteDH(a=l1, alpha=0), 
    rb.RevoluteDH(a=l2, alpha=0), 
    rb.RevoluteDH(a=l3, alpha=0), 
    rb.RevoluteDH(a=l4, alpha=0)   
])

qr=[np.pi/6,np.pi/6,np.pi/6,np.pi/6]
Tee = fourR_planar.fkine(qr)
print(Tee)
X_start=Tee.t[0]
Y_start=Tee.t[1]

camera = CentralCamera.Default(pose=Tee)
print(f"Camera initial pose:{camera.pose}")
l = [1.0, 1.0, 1.0, 1.0] 
def jacobian(q, l):
    """ Compute Jacobian matrix in the x-y plane """
    J = np.zeros((6, 4))
    
    J[0, 0] = -l[0] * np.sin(q[0]) - l[1] * np.sin(q[0] + q[1]) - l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 1] = -l[1] * np.sin(q[0] + q[1]) - l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 2] = -l[2] * np.sin(q[0] + q[1] + q[2]) - l[3] * np.sin(np.sum(q))
    J[0, 3] = -l[3] * np.sin(np.sum(q))
    
    J[1, 0] = l[0] * np.cos(q[0]) + l[1] * np.cos(q[0] + q[1]) + l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 1] = l[1] * np.cos(q[0] + q[1]) + l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 2] = l[2] * np.cos(q[0] + q[1] + q[2]) + l[3] * np.cos(np.sum(q))
    J[1, 3] = l[3] * np.cos(np.sum(q))
    J[2, :] = 0
    J[3, :] = 0
    J[4, :] = 0
    J[5, :] = 1

    return J

centre_point=np.array([4,2,0.1])
P=np.array([[centre_point[0]-0.2,centre_point[0]-0.2,centre_point[0]+0.2,centre_point[0]+0.2],
            [centre_point[1]-0.2,centre_point[1]+0.2,centre_point[1]+0.2,centre_point[1]-0.2],
            [0,0,0,0]])

print(P)

T_Cd_G = SE3.Tz(0.1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[0, :], P[1, :], P[2, :], c='g', marker='o')
theta1=np.pi/6
theta2=np.pi/6
theta3=np.pi/6
theta4=np.pi/6
x_prev=X_start
y_prev=Y_start
theta=[theta1,theta2,theta3,theta4]
theta_dot=[0,0,0,0]
R_current=camera.pose.R
for i in range(300):
    
    objpose_in_camera = camera.pose.inv()*SE3.Trans(centre_point)
    p = camera.project_point(P, objpose=objpose_in_camera)
    Te_C_G = camera.estpose(P, p, frame="camera")
    T_delta = Te_C_G * T_Cd_G.inv()
    camera.pose = camera.pose * T_delta.interp1(0.01)
    print(f"Camera pose:{camera.pose}")
    x, y, z = camera.pose.t
    R_new=camera.pose.R
    angular_velocity = logm(R_new @ R_current.T) / 0.01 # 3x3 matrix

# Extract angular velocity components
    omega_x, omega_y, omega_z = angular_velocity[2, 1], angular_velocity[0, 2], angular_velocity[1, 0]

    x_vector=(x-x_prev)/0.01
    y_vector=(y-y_prev)/0.01
    vector = np.array([x_vector, y_vector,0,omega_x,omega_y,omega_z]).reshape(6, 1)  

    theta_dot = np.matmul(np.linalg.pinv(jacobian(theta,l)), vector).flatten()  

    theta_new=np.transpose(theta)+theta_dot*0.01
    fourR_planar.q = theta_new.flatten()  
    ax.cla()
    fourR_planar.plot(fourR_planar.q) 
    theta=theta_new
    x_prev=x
    y_prev=y
    R_current=R_new
    print(f"ee position:{fourR_planar.fkine(fourR_planar.q)}")
    ax.scatter(camera.pose.t[0], camera.pose.t[1], camera.pose.t[2], c='r', marker='x')  
    ax.scatter(P[0, :], P[1, :], P[2, :], c='g', marker='o')

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.pause(0.01)  
