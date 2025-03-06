import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create camera pose
camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2))

# Create a grid of points
# centre_point=np.array([0.5,0.5,0.5])

# P=np.array([[centre_point[0]-0.2,centre_point[0]-0.2,centre_point[0]+0.2,centre_point[0]+0.2],
#             [centre_point[1]-0.2,centre_point[1]+0.2,centre_point[1]+0.2,centre_point[1]-0.2],
#             [0,0,0,0]])

P = mkgrid(2, 0.5)

# Project points onto the camera plane
p = camera.project_point(P, objpose=SE3.Tz(1))

# Estimate camera pose
Te_C_G = camera.estpose(P, p, frame="camera")
Te_C_G.printline()

# Transformation from camera to goal frame
T_Cd_G = SE3.Tz(1)

# Calculate the difference transformation
T_delta = Te_C_G * T_Cd_G.inv()
T_delta.printline()

# Update camera pose
camera.pose = camera.pose * T_delta.interp1(0.05)
# print(f"Camera_pose:{camera.pose}")
# x,y,z=camera.pose.t
# print(f"X,Y,Z:{x,y,z}")
# Re-initialize the camera with the new pose
camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2))

# Re-initialize goal transformation
T_Cd_G = SE3.Tz(1)

# Initialize and run PBVS (Position-Based Visual Servoing)
pbvs = PBVS(camera, P=P, pose_g=SE3.Trans(-1, -1, 2), pose_d=T_Cd_G, plotvol=[-1, 2, -1, 2, -3, 2.5])

# Run the visual servoing loop
pbvs.run(200)

# Plot results
pbvs.plot_p()    # Plot image plane trajectory
pbvs.plot_vel()   # Plot camera velocity
pbvs.plot_pose()  # Plot camera trajectory

# Add 3D plotting for additional feedback

# Generate a 3D cylinder for visualization
theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the cylinder
z = np.linspace(-1, 1, 100)  # Height of the cylinder
Z, T = np.meshgrid(z, theta)  # Create meshgrid for Z and theta

# Parametric equations for a cylinder
X = np.cos(T)  # X coordinates
Y = np.sin(T)  # Y coordinates

# Plot the cylinder using wireframe
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot wireframe (ensure X, Y, Z all have the same shape)
ax.plot_wireframe(X, Y, Z, color='b')

# Set plot labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# Show the plot
plt.show()
