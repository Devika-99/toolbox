import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the second-order ODE as a system of first-order ODEs
# y'' = -y is equivalent to:
# y' = z
# z' = -y
def system(Y, t):
    y, z = Y
    dydt = z
    dzdt = -y
    return [dydt, dzdt]

# Initial conditions: y(0) = 1, y'(0) = 0
Y0 = [1, 0]

# Time points to solve for
t = np.linspace(0, 10, 100)

# Solve the ODE using odeint
sol = odeint(system, Y0, t)

# Extract y and y'
y = sol[:, 0]
z = sol[:, 1]

# Plot the results
plt.figure(figsize=(10, 5))

# Plot y(t)
plt.subplot(2, 1, 1)
plt.plot(t, y, label='y(t)')
plt.title('Solution of y\'\' = -y (Simple Harmonic Oscillator)')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.grid(True)

# Plot y'(t) which is the velocity
plt.subplot(2, 1, 2)
plt.plot(t, z, label="y'(t) = z(t)", color='orange')
plt.xlabel('Time (t)')
plt.ylabel('y\'(t)')
plt.grid(True)

plt.tight_layout()
plt.show()
