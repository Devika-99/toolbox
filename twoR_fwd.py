import numpy as np
l1=1
l2=1
theta1=-0.669505925908312
theta2=0.7292947459007837
x=l1*np.cos(theta1)+l2*np.cos(theta1+theta2)
y=l1*np.sin(theta1)+l2*np.sin(theta1+theta2)
print(f"x:{x}")
print(f"y:{y}")

x_value=1.782   
y_value=-0.5608  

camera_pose=np.array([[0,-1,0,1],
                     [1,0,0,1],
                     [0,0,1,0],
                     [0,0,0,1]])
camera_pose_inv=np.linalg.inv(camera_pose)
print(camera_pose_inv)

SE3_Trans=np.array([[1,0,0,1.6],
                    [0,1,0,0.5],
                    [0,0,1,0.1],
                    [0,0,0,1]])
objpose=np.matmul(camera_pose_inv,SE3_Trans)
print(objpose)

K = np.array([[800, 0, 500],
              [0, 800, 500],
              [0, 0, 1]])
X_c=[0,1,0,-0.5]
Y_c=[-1,0,0,-0.6]
Z_c=[0,0,1,0.1]
P_image = K @ np.vstack((X_c, Y_c, np.array(Z_c)+1 ))  # Add 1 to Z to avoid division by zero
P_image /= P_image[2]  # Normalize by depth
u, v = P_image[:2]

print("Pixel Coordinates (u, v):")
print(np.vstack((u, v)))

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

camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2))
print(camera)
Pworld=np.matmul(objpose,)