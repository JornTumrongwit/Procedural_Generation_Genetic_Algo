import pygame
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
# OpenGL imports for python
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
except:
    print("OpenGL wrapper for python not found")

#image
image = 'testimage.png'

asurf = pygame.image.load('testimage.png')
bitmap = pygame.surfarray.array2d(asurf)
start = time.time()
# set up the figure and axes
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

#class for tower types
class Tower:
    def __init__(self, height, width, depth, x, y, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.y = y
        self.rotation = rotation

#clamp helper
def clamp(num, min, max):
    low = max(num, min)
    return min(max, low)

towcount = 64
heightmin = 3
xlen = 300
zlen = xlen * (len(bitmap[0])/len(bitmap)) #setting aspect ratio

#procedural generation parameters
generations = 25
p_mu = 0.001
p_crossover = 0.9
popsize = 10

#tower generation
towers = []
for tower in range(towcount):
    x = random.uniform(0, xlen)
    y = random.uniform(0, xlen)
    height = random.uniform(0, zlen)
    width = random.uniform(0, xlen)
    depth = random.uniform(0, xlen)
    towers.append(Tower(height, width, depth, x, y))

# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self):
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.scale = 1.0
        self.width = 1.0
        self.height = 3.0
        self.depth = 1.0
        self.pos_x = 0.0
        self.pos_z = 0.0
        self.pos_y = 0.0

    # Initialize
    def init(self):
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Set the shade model to flat
        glShadeModel(GL_FLAT)

    # Draw cube
    def draw(self):

        # Set the color to white
        glColor3f(1.0, 1.0, 1.0)

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        
        #object transforms
        glRotatef(self.rotate_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotate_x, 1.0, 0.0, 0.0)
        glScalef(self.scale*self.width, self.scale*self.height, self.scale*self.depth)
        glTranslatef(self.pos_x, self.pos_y, self.pos_z)

        # Draw solid cube
        for i in range(300):
            glutSolidCube(1.0)
            glRotatef(math.radians(i), 1.0, 0.0, 0.0)
            glTranslatef(i/300, 0.0, 0.0)

        glFlush()

    # The display function
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT)

        # Draw cube
        self.draw()

    # The reshape function
    def reshape(self, w, h):
        glViewport(0, 0, GLsizei(w), GLsizei(h))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
        glMatrixMode(GL_MODELVIEW)

    # The keyboard controls
    def special(self, key, x, y):

        # Rotate cube according to keys pressed
        if key == GLUT_KEY_RIGHT:
            self.rotate_y += 5
        if key == GLUT_KEY_LEFT:
            self.rotate_y -= 5
        if key == GLUT_KEY_UP:
            self.rotate_x += 5
        if key == GLUT_KEY_DOWN:
            self.rotate_x -= 5
        glutPostRedisplay()


# The main function
def main():

    # Initialize OpenGL
    glutInit(sys.argv)

    # Set display mode
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

    # Set size and position of window size
    glutInitWindowSize(400, 400)
    glutInitWindowPosition(100, 100)

    # Create window with given title
    glutCreateWindow("Cube")

    # Instantiate the cube
    cube = Cube()

    cube.init()

    # The callback for display function
    glutDisplayFunc(cube.display)

    # The callback for reshape function
    glutReshapeFunc(cube.reshape)

    # The callback function for keyboard controls
    glutSpecialFunc(cube.special)

    # Start the main loop
    glutMainLoop()

# Call the main function
if __name__ == '__main__':
    main()
'''
#MAKING THE SKYLINE
top = []
currentskyscrapers = 0
currentmidtowers = 0

for i in range(lim):
    height = random.uniform(heightmin, heightmax)
    added = False
    for tower in towers:
        if height >= tower.height:
            if tower.built == tower.amt:
                while height >= tower.height:
                    height = random.uniform(heightmin, tower.height)
            else:
                tower.built = tower.built + 1
                top.append(height)
                added = True
                break
    if not added:
        top.append(height)

while(len(top)<xamt*yamt):
    top.append(0)

cap = heightmax
for tower in towers:
    while tower.built < tower.amt:
        index = random.randint(0, len(top)-1)
        while top[index] >= tower.height:
            index = random.randint(0, len(top)-1)
        height = tower.height-1
        while height < tower.height:
            height = random.uniform(tower.height, cap)
        top[index] = height
        tower.built = tower.built+1
    cap = tower.height

random.shuffle(top)
bottom = np.zeros_like(top)

#GA

#fitness of a city is based on how tall the buildings are in relation to its distance from the stations
#Aaaaand I just realized this is technically MOO, yay!
def fitness(city):
    score = 0
    x, y = makegrid(city.x, city.y)
    #adding scores based on stations close by, and deducting based on height if it is far
    for i in range(len(city.top)):
        dist = 0
        for station in stations:
            #using manhattan distance, since we're in a city and you have blocks 
            #instead of just a straight line through
            if dist == 0:
                dist = abs(station[0] - x[i]) + abs(station[1] - y[i])
            else:
                dist = min(abs(station[0] - x[i]) + abs(station[1] - y[i]), dist)
        mod = (stationproxim-dist)
        if mod > 0:
            mod = mod / 10 #less intensity for buildings away from that area
        score = score + city.top[i]*(stationproxim-dist)
    return score

cities = []
#populating 
for pop in range(popsize):
    cityx = defx.copy()
    cityy = defy.copy()
    random.shuffle(cityx)
    random.shuffle(cityy)
    cities.append(City(cityx, cityy, top))

#ordered crossover
def order(parent1, parent2, cutoff):
    child = parent1[:cutoff]
    for char in parent2:
        if char not in child:
            child = np.append(child, char)
    return child

#roulette selection
def roulettegen(paramt):
    fit = []
    for city in cities:
        fit.append(fitness(city))
    print(fit[:3])
    amt = 0
    for i in range(len(fit)):
        fit[i] = fit[i] + amt
        amt = fit[i]
    parents = []
    for p in range(paramt):
        rng = random.uniform(0, amt)
        for i in range(len(fit)):
            if rng <= fit[i]:
                parents.append(i)
    return parents

#mutation (change to a random height for that tower range)
def mutate(input):
    if input >= towers[0].height:
        return random.uniform(towers[0].height, heightmax)
    for i in range(1, len(towers)):
        if input >= towers[i].height:
            return random.uniform(towers[i].height, towers[i-1].height)
    if input >= heightmin:
        return random.uniform(heightmin, towers[len(towers)-1].height)
    else:
        return 0
    
#Genetic algo
for gen in range(generations):
    print("GENERATION:", gen)

    #get parents
    parents = roulettegen(popsize)

    #crossovers. In this case we're just doubling the initial population
    for children in range(popsize):
        #choose parents
        par1 = random.randint(0, len(parents)-1)
        par2 = random.randint(0, len(parents)-1)
        while par2 == par1:
            par2 = random.randint(0, len(parents)-1)
        city1 = cities[parents[par1]]
        city2 = cities[parents[par2]]
        if(random.uniform(0, 1) < p_crossover):
            x_cutpoint = random.randint(0, xamt-1)
            y_cutpoint = random.randint(0, xamt-1)
            top_cutpoint = random.randint(0, len(top)-1)
            #crossing x
            newx = order(city1.x, city2.x, x_cutpoint)
            #crossing y
            newy = order(city1.y, city2.y, y_cutpoint)
            newtop = city1.top[:top_cutpoint] + city2.top[top_cutpoint:]
            #to keep the same amount of towers, just use combine city1's tower arrangements with city 2's
            child = City(newx, newy, newtop)
            for tower in range(len(child.top)):
                rng = random.uniform(0, 1)
                if rng < p_mu:
                    child.top[tower] = mutate(child.top[tower])
            cities.append(child)
    
    #sort city, pick best popsize cities
    cities.sort(key=fitness,reverse=True)
    cities = cities[:popsize]
'''

x = []
y = []
height = []
for tower in towers:
    x.append(tower.x)
    y.append(tower.y)
end = time.time()
print("TIME =", end-start)
#"Modelling" the city
ax1.set_xlim(0, xlen)
ax1.set_ylim(0, xlen)
ax1.set_zlim(0, xlen)

ax1.bar3d(x, y, bottom, width, depth, city.top, shade=True)
ax1.set_title('Shaded')

plt.show()
