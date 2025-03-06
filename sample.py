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

centre_point=np.array([1,0.7,0.1])

camera = CentralCamera.Default(pose=Tee)
print(f"Camera pose:{camera.pose}")
print(f"Camera pose inv:{camera.pose.inv()}")
print(f"Is it identity :{camera.pose.A@camera.pose.inv().A}")
print(SE3.Trans(centre_point))

centre_point=np.array([1.5,0.1,0.1])
P=np.array([[centre_point[0]-0.2,centre_point[0]-0.2,centre_point[0]+0.2,centre_point[0]+0.2],
            [centre_point[1]-0.2,centre_point[1]+0.2,centre_point[1]+0.2,centre_point[1]-0.2],
            [0,0,0,0]])
print(P)
print((P[:,0]+P[:,1]+P[:,2]+P[:,3])/4)
