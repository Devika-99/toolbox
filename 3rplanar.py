import numpy as np

def inverse_kinematics_3r(x, y, l1, l2, l3):
    """
    Compute inverse kinematics for a 3R planar manipulator.
    Assuming the fourth joint is ignored (θ4 = 0).
    """
    # Compute distance from base to target
    r2 = x**2 + y**2
    
    # Compute θ2 using cosine rule
    cos_theta2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    if abs(cos_theta2) > 1:
        raise ValueError("No valid solution: Point out of reach")
    
    theta2 = np.arctan2(np.sqrt(1 - cos_theta2**2), cos_theta2)  # Elbow up
    k1 = l1 + l2 * np.cos(theta2)
    k2 = l2 * np.sin(theta2)
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    # Compute θ3
    x3 = x - l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2)
    y3 = y - l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    theta3 = np.arctan2(y3, x3)
        
    return theta1,theta2,theta3,0

# Example usage:
x, y = 2.5, 1.5  # Desired end-effector position
l1, l2, l3 = 1.0, 1.0, 1.0  # Link lengths

