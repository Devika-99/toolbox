import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from roboticstoolbox.models.DH import Planar2

twoR = Planar2()
Tee = twoR.fkine(twoR.qr)
print(Tee)

camera = CentralCamera.Default(pose=Tee)
print(f"Camera initial pose:{camera.pose}")
def inverse_kinematics(x, y):
    l1 = 1
    l2 = 1

    r = np.sqrt(x**2 + y**2)
    if r > l1 + l2:  # Check if the point is reachable
        raise ValueError("Target is unreachable")
    cos_theta2 = (r**2 - l1**2 - l2**2) / (2 * l1 * l2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)

    k1 = l1 + l2 * cos_theta2
    k2 = l2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return theta1, theta2

# Define the target grid


# center_point = np.mean(P, axis=0)
# print(center_point)

# center_point=np.array([0.5,0.6,0])
# P=np.array([])

# centre_point=np.array([1,0.7,0.1])
# P=np.array([[centre_point[0]-0.2,centre_point[0]-0.2,centre_point[0]+0.2,centre_point[0]+0.2],
#             [centre_point[1]-0.2,centre_point[1]+0.2,centre_point[1]+0.2,centre_point[1]-0.2],
#             [0,0,0,0]])

P = mkgrid(2, side=0.5)
center_point = np.mean(P, axis=0)
print(f"Centre point:{center_point}")
print(P)
T_Cd_G = SE3.Tz(0.1)

# Create a figure and 3D axis for plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(P[0, :], P[1, :], P[2, :], c='r', marker='o', label="Grid Points")
for i in range(200):
    p = camera.project_point(P, objpose=SE3.Tz(0.1))
    Te_C_G = camera.estpose(P, p, frame="camera")
    T_delta = Te_C_G * T_Cd_G.inv()
    camera.pose = camera.pose * T_delta.interp1(0.05)
    print(f"Camera pose:{camera.pose}")
    x, y, z = camera.pose.t
    theta1, theta2 = inverse_kinematics(x, y)
    
    # Update the manipulator's joint configuration
    twoR.q = [theta1, theta2]
    
    # Clear the previous plot
    ax.cla()
    twoR.plot(twoR.q)  
    print(f"ee position:{twoR.fkine(twoR.q)}")
    # Plot camera position
    ax.scatter(camera.pose.t[0], camera.pose.t[1], camera.pose.t[2], c='r', marker='x')  # Mark camera pose
    
    # Plot the target grid
    ax.scatter(P[0, :], P[1, :], P[2, :], c='g', marker='o')


    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-3, 2.5])

    # Pause to update the plot
    plt.pause(0.01)  # Adjust pause time for smoother visualization
