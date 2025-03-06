import roboticstoolbox as rb
import numpy as np

# Load a predefined robot model (Puma 560)
robot = rb.models.Puma560()

# Plot the robot at a given configuration (e.g., all joint angles at zero)
robot.plot([0, 0, 0, 0, 0, 0])

# Define joint angles
theta = [np.pi/4, np.pi/6, 0, 0, 0, 0]

# Plot robot at these joint angles
robot.plot(theta)
