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
    import glm
except:
    print("OpenGL wrapper for python not found")

#image
image = 'testimage.png'

asurf = pygame.image.load('testimage.png')
bitmap = pygame.surfarray.array2d(asurf)
start = time.time()

#class for tower generation
class Tower:
    def __init__(self, height, width, depth, x, y, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.z = z
        self.rotation = rotation
    
towcount = 64
heightmin = 3
xlen = 300
height = xlen * (len(bitmap[0])/len(bitmap)) #setting aspect ratio

#tower generation
towers = []
x = random.uniform(0, xlen)
z = random.uniform(0, xlen)
height = random.uniform(0, height)
width = random.uniform(0, xlen)
depth = random.uniform(0, xlen)
towers.append(Tower(height, width, depth, x, z, 0.0))
for tower in range(towcount-1):
    x = random.uniform(0, xlen)
    z = random.uniform(0, xlen)
    height = random.uniform(0, height)
    width = random.uniform(0, xlen)
    depth = random.uniform(0, xlen)
    towers.append(Tower(height, width, depth, x, z, 0.0))

# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self):
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.viewrotate_y = 0.0
        self.viewrotate_x = 0.0
        self.scale = 1.0
        self.width = 1.0
        self.height = 3.0
        self.depth = 1.0
        self.pos_x = 0.0
        self.pos_z = 10.0
        self.pos_y = 0.0
        self.zoom = 0.3
        self.aspect = len(bitmap[0])/len(bitmap)
        self.eye = glm.vec3(0.0, 3.0, 5.0)
        self.up = glm.vec3(0.0, 1.0, 0.0)

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
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], 0.0, 0.0, 0.0, self.up[0], self.up[1], self.up[2])

        #object transforms
        glRotatef(self.rotate_y, 0.0, 1.0, 0.0)
        glRotatef(self.rotate_x, 1.0, 0.0, 0.0)
        glScalef(self.scale*self.width *self.zoom, self.scale*self.height*self.zoom, self.scale*self.depth*self.zoom)
        glTranslatef(self.pos_x, self.pos_y, self.pos_z)

        # Draw solid cube
        for i in range(10):
            glutSolidCube(1.0)
            glTranslatef(i, 0.0, i)

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
        glFrustum(-1.0*w/h, 1.0*w/h, -1.0, 1.0, 1.5, 50.0)
        self.aspect = w/h
        glMatrixMode(GL_MODELVIEW)

    # The keyboard controls
    def special(self, key, x, y):

        # Rotate cube according to keys pressed
        if key == GLUT_KEY_RIGHT:
            rot = glm.rotate(glm.mat4(1.0), -0.1, glm.normalize(self.up))
            self.eye = rot * self.eye
        if key == GLUT_KEY_LEFT:
            rot = glm.rotate(glm.mat4(1.0), 0.1, glm.normalize(self.up))
            self.eye = rot * self.eye
        if key == GLUT_KEY_UP:
            axis = glm.cross(self.eye, self.up)
            rot = glm.rotate(glm.mat4(1.0), 0.1, glm.normalize(axis))
            self.up = rot * self.up
            self.eye = rot * self.eye
        if key == GLUT_KEY_DOWN:
            axis = glm.cross(self.eye, self.up)
            rot = glm.rotate(glm.mat4(1.0), -0.1, glm.normalize(axis))
            self.up = rot * self.up
            self.eye = rot * self.eye
        glutPostRedisplay()
    
    # The normal keyboard controls
    def keyb(self, key, x, y):
        if key == b'+' or key == b'=':
            self.zoom *= 1.1
        elif key == b'-' or key == b'_':
            self.zoom /= 1.1
        glutPostRedisplay()



# The main function
def main():
    # Initialize OpenGL
    glutInit(sys.argv)

    # Set display mode
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

    # Set size and position of window size
    glutInitWindowSize(len(bitmap[0]), len(bitmap))
    glutInitWindowPosition(0, 10)

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

    # The callback function for normal keyboard controls
    glutKeyboardFunc(cube.keyb)

    # Start the main loop
    glutMainLoop()

# Call the main function
if __name__ == '__main__':
    main()