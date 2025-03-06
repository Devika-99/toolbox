import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from machinevisiontoolbox import CentralCamera
from machinevisiontoolbox import IBVS
import matplotlib.pyplot as plt

# Step 1: Define the Manipulator (2R Planar Robot)
L1 = rtb.RevoluteDH(a=1, alpha=0)  # First link, length 1
L2 = rtb.RevoluteDH(a=1, alpha=0)  # Second link, length 1

# Create the SerialLink robot
manipulator = rtb.DHRobot([L1, L2], name='2R Planar')

# Step 2: Define the Camera
# Initial camera pose is the same as the manipulator's end-effector pose
camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -3) * SE3.Rz(0.6))

# Step 3: Define the Target Point in 3D Space (World Coordinates)
# The target point where the camera should align in the image plane
target_point_world = np.array([1.5, 1.5, 3]).reshape(3, 1)  # Reshaped to 3x1 array

# Step 4: Create the IBVS Object
# Create the target image points in pixel coordinates (ideal image points)
pd = camera.project_point(target_point_world)

# Initial joint angles for the manipulator (can be modified)
theta_initial = [0.5, 0.5]

# Step 5: Setup the IBVS for the Manipulator (Camera mounted on end-effector)
ibvs = IBVS(camera, P=target_point_world, p_d=pd)

# Step 6: Control Loop for IBVS
iterations = 100  # Number of control loop iterations
lmbda = 0.1  # Gain for velocity command

# Run the IBVS control loop
for i in range(iterations):
    # Calculate the end-effector pose (camera pose)
    ee_pose = manipulator.fkine(theta_initial)
    
    # Update the camera pose based on the manipulator's end-effector pose
    camera.pose = ee_pose
    
    # Project the current position of the target in the image plane
    p = camera.project_point(target_point_world)
    
    # Calculate the error in the image plane
    e = pd - p

    # Calculate the Jacobian of the visual servoing system (image Jacobian)
    J = camera.visjac_p(p, depth=1)
    
    # Calculate the velocity command using the pseudoinverse of the Jacobian
    v = lmbda * np.linalg.pinv(J) @ e.flatten(order="F")
    
    # Update the camera pose based on the computed velocity
    camera.pose = camera.pose @ SE3.Delta(v)
    
    # Perform Inverse Kinematics (IK) to move the manipulator based on the velocity
    J_robot = manipulator.jacobe(theta_initial)  # Robot Jacobian
    q_dot = np.linalg.pinv(J_robot) @ v  # Joint velocity for manipulator
    theta_initial += q_dot * 0.1  # Update joint angles (small step)

    # Optionally, you can visualize the robot configuration at each step
    manipulator.plot(theta_initial)

# Step 7: Plot Results (Optional)
ibvs.plot_p()  # Plot the trajectory of image points
ibvs.plot_vel()  # Plot the velocity of the camera
ibvs.plot_pose()  # Plot the pose of the camera
ibvs.plot_jcond()  # Plot the Jacobian condition number

# Final Visualization (Target in Image)
plt.show()
