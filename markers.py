import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


scene = Image.Read("lab-scene.png", rgb=False)
plt.imshow(scene.image, cmap='gray')  # Use `scene.image` to access pixel data
plt.title("Scene")
plt.show()

camera = CentralCamera(f=3045, imagesize=scene.shape,
pp=(2016, 1512), rho=1.4e-6)
markers = scene.fiducial(dict="4x4_50", K=camera.K, side=67e-3)
print(markers)
for marker in markers:
    marker.draw(scene, length=0.10, thick=20)