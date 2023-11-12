import tower_demo_d_tree as dtree
from matplotlib import pyplot as plt
import numpy as np
import random
import math
import time
import cv2
from enum import Enum
import imageio.v3 as iio
import ipympl
# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glm
    import glfw
except:
    print("OpenGL wrapper for python not found")

image = 'badapple.png'
#image
img = cv2.imread(image)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv2.blur(img,(50,50))
cv2.imwrite(r"tester.png", img)
img = np.divide(img, 255/2)
img = np.add(img, -1)

im = cv2.imread(image)
im = cv2.blur(im, (25, 25))
im = np.divide(im, 255/2)
im = np.add(im, -1)
im = np.around(im)
bestscore = np.sum(np.multiply(im, img))
print(bestscore)
d_width = len(im[0])
d_height = len(im)

#GA parameters
p_mu_grow = 0.125
p_mu_cut = 0.1 
p_mu_alter = 0.18
nodes = 3000
minnodes = 1
popsize = 180
generations = 15
k_tournament = 10
elitism = popsize//5

# Initialize the library
glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(d_width, d_height, "hidden window", None, None)
if not window:
    glfw.terminate()

# Make the window's context current
glfw.make_context_current(window)

glutInit(sys.argv)
cube = dtree.Cube()
cube.init()

def render_And_Score(towers):
    # Instantiate the cube
    cube = dtree.Cube()

    cube.init()

    cube.changetower(towers)
    cube.reshape(d_width, d_height)
    cube.display()
    
    image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
    imagearr = np.flip(imagearr, 0)
    im2 = cv2.blur(imagearr, (25, 25))
    cv2.imwrite(r"testresult.png", im2)
    im2 = np.divide(imagearr, 255/2)
    im2 = np.add(im2, -1)
    im2 = np.around(im2)
    score = np.sum(np.multiply(img, im2))
    return score/bestscore + 1

def save_result(towers):
    # Instantiate the cube
    cube = dtree.Cube()

    cube.init()

    cube.changetower(towers)
    cube.reshape(d_width, d_height)
    cube.display()
    
    image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_BGR, OpenGL.GL.GL_UNSIGNED_BYTE)
    imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
    imagearr = np.flip(imagearr, 0)
    cv2.imwrite(r"testfinal.png", imagearr)

class City:
    def __init__(self, tree):
       self.tree = tree
       self.tower = dtree.maketowers(tree)
       self.score = render_And_Score(self.tower) 

def getscore(city):
    return city.score

def getlen(city):
    return city.tree.nodes

#SELECTING PARENTS
#ROULETTE
def roulette_selection(towns):
    totalfit = 0
    fitarr = []
    for town in towns:
        totalfit += town.score
        fitarr.append(totalfit)
    rng = random.uniform(0, totalfit)
    for i in range(len(fitarr)):
        if rng <= fitarr[i]:
            parent1 = towns[i]
            break
    return parent1

#TOURNAMENT
def tournament(towns):
    indices = set()
    chosen = []
    for i in range(k_tournament):
        added = False
        while not added:
            rng = random.randint(0, len(towns)-1)
            if rng not in indices:
                indices.add(rng)
                chosen.append(towns[rng])
                added = True
    chosen.sort(key=getscore)
    return chosen[0]

cities = []
for i in range(popsize):
    test_tree = dtree.Tree(None, minnodes, nodes)
    cities.append(City(test_tree))

cities.sort(key=getlen, reverse = True)
cities.sort(key=getscore, reverse = True)
cities = cities[:elitism]
for gen in range(generations):
    print("GENERATION:", gen)
    print("BEST 3:", cities[0].score, ",", cities[1].score, ",", cities[2].score)
    while len(cities) < popsize:
        par1 = roulette_selection(cities)
        par2 = tournament(cities)
        while par1 == par2:
            par2 = tournament(cities)
        test_child_tree, test_child_tree2 = dtree.crossover(par1.tree, par2.tree)
        rng = random.uniform(0, 1)
        test_child_tree.grow_mutate(p_mu_grow)
        if rng < p_mu_cut:
            test_child_tree.cut()
        if rng < p_mu_alter:
            test_child_tree.alter()
        rng = random.uniform(0, 1)
        test_child_tree2.grow_mutate(p_mu_grow)
        if rng < p_mu_cut:
            test_child_tree2.cut()
        if rng < p_mu_alter:
            test_child_tree2.alter()
        cities.append(City(test_child_tree))
        cities.append(City(test_child_tree2))
    cities.sort(key=getlen, reverse = True)
    cities.sort(key=getscore, reverse = True)
    cities = cities[:elitism]
    print()

towers = dtree.maketowers(cities[0].tree)
print(render_And_Score(dtree.maketowers(cities[0].tree)), cities[0].tree.nodes)
save_result(dtree.maketowers(cities[0].tree))
glfw.destroy_window(window)
glfw.terminate()
