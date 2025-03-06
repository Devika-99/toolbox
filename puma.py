import roboticstoolbox as rb
import numpy as np
from machinevisiontoolbox.base import *
from machinevisiontoolbox import *
from spatialmath.base import *
from spatialmath import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import models from roboticstoolbox
from roboticstoolbox.models.DH import Puma560

# Instantiate the Puma560 model
puma = Puma560()

# Print model details (optional)
print(puma)

# Use the teach method with the ready configuration
puma.teach(q=puma.qr)
