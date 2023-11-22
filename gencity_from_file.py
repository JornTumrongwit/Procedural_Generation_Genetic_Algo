import tower_demo_d_tree as dtree
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import time
import cv2
from enum import Enum
import parameters
from statistics import mean
# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glm
    import glfw
except:
    print("OpenGL wrapper for python not found")

file1 = open('townfile.txt', 'r')
Lines = file1.readlines()

eye_param = list(map(float, Lines[0].split()))
eye = glm.vec3(eye_param[0], eye_param[1], eye_param[2])
up_param = list(map(float, Lines[1].split()))
up = glm.vec3(up_param[0], up_param[1], up_param[2])
center_param = list(map(float, Lines[2].split()))
center = glm.vec3(center_param[0], center_param[1], center_param[2])

dim = list(map(int, Lines[3].split()))

tower = []
for line in Lines[4:]:
    tower_param = line.split()
    tower_param = list(map(float, tower_param))
    tower.append(dtree.Tower(tower_param[0], tower_param[1], tower_param[2], tower_param[3], tower_param[4], tower_param[5]))

dtree.display_towers(tower, eye, up, center, dim[0], dim[1])