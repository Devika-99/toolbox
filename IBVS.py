import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2))
P = mkgrid(2, side=0.5, pose=SE3.Tz(3))
pd = 200 * np.array([[-1, -1, 1, 1], [-1, 1, 1, -1]]) + np.c_[camera.pp]
p = camera.project_point(P)
e = pd - p
J = camera.visjac_p(p, depth=1)
lmbda = 0.1
v = lmbda * np.linalg.pinv(J) @ e.flatten(order="F")
camera.pose = camera.pose @ SE3.Delta(v)
camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -3) * SE3.Rz(0.6))
ibvs = IBVS(camera, P=P, p_d=pd)
ibvs.run(100)
ibvs.plot_p()   
ibvs.plot_vel()  
ibvs.plot_pose()
ibvs.plot_jcond()
