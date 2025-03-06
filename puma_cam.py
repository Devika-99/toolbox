import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from roboticstoolbox.models.DH import Puma560
puma=Puma560()
camera = CentralCamera.Default(pose=SE3.Trans(1, 1, -2))
print(camera)
q=puma.qr
T_end_eff=puma.fkine(q)
camera.T=T_end_eff
print(camera.T)