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
import parameters
# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glm
    import glfw
except:
    print("OpenGL wrapper for python not found")

#GA parameters
p_mu_grow = 0.125
p_mu_cut = 0.1 + p_mu_grow
p_mu_alter = 0.18 + p_mu_cut
nodes = parameters.nodecount
minnodes = 1
popsize = 180
generations = 50
k_tournament = 10
elitism = popsize//5
refblur = 30
resblur = 10

image = parameters.image
#image
img = cv2.imread(image)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv2.blur(img,(refblur, refblur))
cv2.imwrite(r"tester.png", img)
img = np.divide(img, 255/(parameters.good - parameters.bad))
img = np.add(img, parameters.bad)

im = cv2.imread(image)
im = cv2.blur(im, (resblur, resblur))
im = np.divide(im, 255)
bestscore = np.sum(np.multiply(im, img))
d_width = len(im[0])
d_height = len(im)
parameters.unit_mod /= len(im)
parameters.expandedness *= len(im)/len(im[0])
parameters.aspect = len(im[0])/len(im)


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
    cube.display_nowire()
    
    image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
    imagearr = np.flip(imagearr, 0)
    im2 = cv2.blur(imagearr, (resblur, resblur))
    #cv2.imwrite(r"testresult.png", im2)
    im2 = np.divide(imagearr, 255)
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
for gen in range(generations):
    print("GENERATION:", gen)
    print("BEST 3:", cities[0].score, ",", cities[1].score, ",", cities[2].score)
    elites = cities[:elitism]
    while len(elites) < popsize:
        par1 = roulette_selection(cities)
        par2 = tournament(cities)
        while par1 == par2:
            par2 = tournament(cities)
        test_child_tree, test_child_tree2 = dtree.crossover(par1.tree, par2.tree)
        rng = random.uniform(0, 1)
        if rng < p_mu_grow:
            test_child_tree.grow()
        elif rng < p_mu_cut:
            test_child_tree.cut()
        elif rng < p_mu_alter:
            test_child_tree.alter()
        rng = random.uniform(0, 1)
        if rng < p_mu_grow:
            test_child_tree2.grow()
        elif rng < p_mu_cut:
            test_child_tree2.cut()
        elif rng < p_mu_alter:
            test_child_tree2.alter()
        elites.append(City(test_child_tree))
        elites.append(City(test_child_tree2))
    elites.sort(key=getlen, reverse = True)
    elites.sort(key=getscore, reverse = True)
    cities = elites
    print()

towers = dtree.maketowers(cities[0].tree)
print(render_And_Score(dtree.maketowers(cities[0].tree)), cities[0].tree.nodes)
save_result(dtree.maketowers(cities[0].tree))
glfw.destroy_window(window)
glfw.terminate()
