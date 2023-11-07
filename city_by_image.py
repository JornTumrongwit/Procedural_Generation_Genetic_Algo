import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import cv2
# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import glm
    import glfw
except:
    print("OpenGL wrapper for python not found")

#image
image = 'testimage.png'

im = cv2.imread(image)
im = np.divide(im, 255/2)
im = np.add(im, -1)
bestscore = np.sum(np.multiply(im, im))
# set up the figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

start = time.time()

d_width = len(im[0])
d_height = len(im)

#class for tower generation
class Tower:
    def __init__(self, height, width, depth, x, z, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.z = z
        self.rotation = rotation

#clamp helper
def clamp(num, min, max):
    low = max(num, min)
    return min(max, low)

def generate_tower():
    z = random.uniform(-18, 18)
    x = random.uniform(-(27-z/1.5), 27-z/1.5)
    height = random.uniform(heightmin/50, heightmax/50)
    width = random.uniform(widthmin, widthmax)
    depth = random.uniform(depthmin, depthmax)*width
    rotation = random.uniform(0, 2*math.pi)
    return Tower(height, width, depth, x, z, rotation)

tower_amt = 30
heightmin = 3
heightmax = 600
widthmin = 0.5
widthmax = 4
depthmin = 0.5
depthmax = 2
xlen = 600
zlen = 100
xmod = 15
heightmod = 200
height = xlen * (len(im[0])/len(im)) #setting aspect ratio

#procedural generation parameters
generations = 25
p_mu_grow = 0.125
p_mu_cut = 0.1 + p_mu_grow
p_mu_alter = 0.18 + p_mu_cut
p_crossover = 0.9
popsize = 180
elitism = 0.2
k_tournament = 10

d_width = len(im[0])
d_height = len(im)
# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self):
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.width = 1.0
        self.height = 2.0
        self.depth = 1.0
        self.pos_x = 5.0
        self.pos_z = 0.0
        self.zoom = 0.2
        self.aspect = len(im[0])/len(im)
        self.eye = glm.vec3(0.0, 3.0, 5.0)
        self.def_eye = self.eye
        self.center = glm.vec3(0.0, 0.0, 0.0)
        self.def_center = self.center
        self.up = glm.vec3(0.0, 1.0, 0.0)
        x = glm.cross(self.up, self.eye-self.center)
        y = glm.cross(self.eye-self.center, x)
        self.up = glm.normalize(y)
        self.def_up = self.up

    # Initialize
    def init(self):
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Set the shade model to flat
        glShadeModel(GL_FLAT)
        

    # Draw cube
    def draw(self, towers):

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[0], self.up[0], self.up[1], self.up[2])
        glScalef(self.zoom, self.zoom, self.zoom)
        # Draw solid cube
        for tower in towers:
            #object transforms
            glPushMatrix()
            glColor3f(1.0, 1.0, 1.0)
            glRotatef(tower.rotation, 0.0, 1.0, 0.0)
            glTranslatef(tower.x, 0.0, tower.z)
            glScalef(tower.width, tower.height, tower.depth)
            glTranslatef(0.5, 0.5, 0.5)
            glutSolidCube(1.0)
            glPopMatrix()
            

        glFlush()

    # The display function
    def display(self, towers):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw cube
        self.draw(towers)

    # The reshape function
    def reshape(self, w, h):
        glViewport(0, 0, GLsizei(w), GLsizei(h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1.0*w/h, 1.0*w/h, -1.0, 1.0, 1.5, 50.0)
        self.aspect = w/h
        glMatrixMode(GL_MODELVIEW)

    def rotate_left(self, degree):
        rot = glm.rotate(glm.mat4(1.0), degree, glm.normalize(self.up))
        self.eye = rot * (self.eye)
    
    def rotate_up(self, degree):
        axis = glm.cross(self.eye, self.up)
        rot = glm.rotate(glm.mat4(1.0), degree, glm.normalize(axis))
        self.up = rot * self.up
        self.eye = rot * (self.eye)  

    # The keyboard controls
    def special(self, key, x, y):

        # Rotate cube according to keys pressed
        if key == GLUT_KEY_RIGHT:
            self.rotate_left(-0.1)
        if key == GLUT_KEY_LEFT:
            self.rotate_left(0.1)
        if key == GLUT_KEY_UP:
            self.rotate_up(0.1)
        if key == GLUT_KEY_DOWN:
            self.rotate_up(-0.1)
        glutPostRedisplay()
    
    # The normal keyboard controls
    def keyb(self, key, x, y):
        if key == b'+' or key == b'=':
            self.zoom *= 1.1
        elif key == b'-' or key == b'_':
            self.zoom /= 1.1
        elif key == b'r':
            self.eye = self.def_eye
            self.center = self.def_center
            self.up = self.def_up
        glutPostRedisplay()

# The main function
def render_And_Score(towers):
    cube.display(towers)
    
    image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
    imagearr = np.flip(imagearr, 0)
    im2 = np.divide(imagearr, 255/2)
    im2 = np.add(im2, -1)
    score = np.sum(np.multiply(im, im2))
    return score/bestscore

# Initialize the library
glfw.init()

# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(d_width, d_height, "hidden window", None, None)
# Make the window's context current
glfw.make_context_current(window)

glutInit(sys.argv)

# Instantiate the cube
cube = Cube()

cube.init()

cube.reshape(d_width, d_height)
    
class Town:
    def __init__(self, towers, score):
        self.towers = towers
        self.score = score

towns = []
#MAKING THE TOWNS
st = time.time()
for i in range(popsize):
    tower_count = np.random.normal(loc=100, scale=20)
    towers = np.array([])
    for tower in range(int(tower_count)):
        towers = np.append(towers, generate_tower())
    score = render_And_Score(towers) + 1
    towns.append(Town(towers, score))
ed = time.time()

def town_rank(town):
    return town.score

towns.sort(key=town_rank, reverse=True)

#SELECTING PARENTS
#ROULETTE
def roulette_selection(towns):
    towns_copy = np.flip(towns)
    totalfit = 0
    fitarr = []
    for town in towns_copy:
        totalfit += town.score
        fitarr.append(totalfit)
    rng = random.uniform(0, totalfit)
    for i in range(len(fitarr)):
        if rng <= fitarr[i]:
            parent1 = towns_copy[i]
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
    chosen.sort(key=town_rank)
    return chosen[0]

#MUTATION
def mutate(child):
    rng = random.uniform(0, 1)
    if rng < p_mu_cut:
        while True:
            child = np.append(child, generate_tower())
            consider = random.uniform(0, 1)
            if consider > tower_amt/(tower_amt +len(child)):
                return child
    elif rng < p_mu_grow:
        while True:
            child = child[:len(child)-1]
            consider = random.uniform(0, 1)
            if consider < tower_amt/(tower_amt +len(child)):
                return child
    elif rng < p_mu_alter:
        consider = random.randint(0, len(child)-1)
        child[consider] = generate_tower()
        return child
    else:
        return child

#Genetic algo
for gen in range(generations):
    print("GENERATION:", gen)
    print("BEST 3:", towns[0].score, ",", towns[1].score, ",", towns[2].score)

    #crossovers. In this case we're just doubling the initial population
    for children in range(int(popsize * elitism /2)):
        #get parents
        par1 = roulette_selection(towns)
        par2 = tournament(towns)
        print("PARENT:", par1.score, ",", par2.score)
        #crossover
        cut_parent_1 = random.randint(0, len(par1.towers)-1)
        cut_parent_2 = random.randint(0, len(par2.towers)-1)
        child1 = Town(np.append(par1.towers[:cut_parent_1], par2.towers[cut_parent_2:]), -1)
        child2 = Town(np.append(par2.towers[:cut_parent_2], par1.towers[cut_parent_1:]), -1)
        #mutation
        child1.towers = mutate(child1.towers)
        child2.towers = mutate(child2.towers)
        child1.score = render_And_Score(child1.towers)
        child2.score = render_And_Score(child2.towers)
        print("Children,", child1.score, ",", child2.score)
        towns.append(child1)
        towns.append(child2)
    
    #sort city, pick best popsize cities
    towns.sort(key=town_rank,reverse=True)
    towns = towns[:popsize]

end = time.time()
glfw.destroy_window(window)
glfw.terminate()
print("TIME =", end-start)

f = open("townfile.txt", "w")
town = towns[0]
print("BEST SCORE:", town.score)
for tower in town.towers:
    f.write(f"{tower.height} {tower.width} {tower.depth} {tower.x} {tower.z} {tower.rotation}\n")
f.close()
