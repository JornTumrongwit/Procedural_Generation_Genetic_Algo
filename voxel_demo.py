import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.colors
import math
class Tower:
    def __init__(self, height, width, depth, x, y, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.y = y
        self.rotation = rotation

def midpoints(x):
    sl = ()
    for i in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x

tt = Tower(300, 10, 10, 0, 0, 30)
# and plot everything
ax = plt.figure().add_subplot(projection='3d')
square = np.array([[0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]])
defwidth = square[0] * tt.width
defdepth = square[1] * tt.depth
for i in range(len(square[0])):
    square[0][i] = defwidth[i]*math.cos(math.radians(tt.rotation)) - defdepth[i]*math.sin(math.radians(tt.rotation))
    square[1][i] = defwidth[i]*math.sin(math.radians(tt.rotation)) + defdepth[i]*math.cos(math.radians(tt.rotation))
print(square)
x = np.array([[[square[0][0], square[0][1]],
            [square[0][2], square[0][3]]],
            [[square[0][0], square[0][1]],
            [square[0][2], square[0][3]]]])
y = np.array([[[square[1][0], square[1][1]],
            [square[1][2], square[1][3]]],
            [[square[1][0], square[1][1]],
            [square[1][2], square[1][3]]]])
z = np.array([[[0, 0],
               [0, 0]],
               [[1, 1],
               [1, 1]]])
faces = np.array([[[True]]])
ax.voxels(x, y, z, faces, edgecolors=(0, 1, 1), linewidth=0.5)

plt.show()