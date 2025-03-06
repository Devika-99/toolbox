import numpy as np
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

def inverse_kinematics(goal, q_init, l, alpha=0.1, tol=1e-3, max_iters=1000):
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

l = [1.0, 1.0, 1.0, 1.0] 

q_init = [0, 0, 0, 0]   
goal = np.array([4,3, q_init[0]+q_init[1]+q_init[2]+q_init[3]])  # Desired end-effector position and orientation
q_solution = inverse_kinematics(goal, q_init, l)
print("IK Solution:", q_solution)
print(f"Forward Kinematics:{forward_kinematics(q_solution,l)}")

