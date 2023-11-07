import pygame
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

asurf = pygame.image.load('testimage.png')
bitmap = pygame.surfarray.array2d(asurf)
im = cv2.imread(image)
start = time.time()

#class for tower generation
class Tower:
    def __init__(self, height, width, depth, x, z, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.z = z
        self.rotation = rotation
    
towcount = 200
heightmin = 3
heightmax = 1200
widthmin = 0.5
widthmax = 4
depthmin = 0.5
depthmax = 4
xlen = 600
zlen = 100
xmod = 15
heightmod = 200
height = xlen * (len(bitmap[0])/len(bitmap)) #setting aspect ratio

#tower generation
towers = []
for tower in range(towcount):
    z = random.uniform(-18, 18)
    x = random.uniform(-(27-z/1.5), 27-z/1.5)
    height = random.uniform(heightmin/50, heightmax/50)
    width = random.uniform(widthmin, widthmax)
    depth = random.uniform(depthmin, depthmax)*width
    rotation = random.uniform(0, 2*math.pi)
    towers.append(Tower(height, width, depth, x, z, rotation))

# The cube class
class Cube:

    # Constructor for the cube class
    def __init__(self):
        self.rotate_y = 0.0
        self.rotate_x = 0.0
        self.viewrotate_y = 0.0
        self.viewrotate_x = 0.0
        self.width = 1.0
        self.height = 2.0
        self.depth = 1.0
        self.pos_x = 5.0
        self.pos_z = 0.0
        self.pos_y = 0.5
        self.zoom = 0.2
        self.aspect = len(bitmap[0])/len(bitmap)
        self.eye = glm.vec3(0.0, 3.0, 5.0)
        self.def_eye = self.eye
        self.center = glm.vec3(0.0, -1.0, 0.0)
        self.def_center = self.center
        self.up = glm.vec3(0.0, 1.0, 0.0)
        x = glm.cross(self.up, self.eye)
        y = glm.cross(self.eye, x)
        self.up = glm.normalize(y)
        self.def_up = self.up

    # Initialize
    def init(self):
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Set the shade model to flat
        glShadeModel(GL_FLAT)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)

    # Draw cube
    def draw(self):

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], 0.0, self.center[1], 0.0, self.up[0], self.up[1], self.up[2])
        glScalef(self.zoom, self.zoom, self.zoom)
        # Draw solid cube
        for tower in towers:
            #object transforms
            glPushMatrix()
            glColor3f(1.0, 1.0, 1.0)
            glRotatef(tower.rotation, 1.0, 0.0, 0.0)
            glTranslatef(tower.x, self.pos_y*tower.height, tower.z)
            glScalef(tower.width, tower.height, tower.depth)
            glutSolidCube(1.0)
            glColor3f(1.0, 0.0, 0.0)
            glutWireCube(1.0)
            glPopMatrix()
            

        glFlush()

    # The display function
    def display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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

    def rotate_left(self, degree):
        rot = glm.rotate(glm.mat4(1.0), degree, glm.normalize(self.up))
        self.eye = rot * (self.eye-self.center) + self.center
    
    def rotate_up(self, degree):
        axis = glm.cross(self.eye, self.up)
        rot = glm.rotate(glm.mat4(1.0), degree, glm.normalize(axis))
        self.up = rot * self.up
        self.eye = rot * (self.eye-self.center) + self.center      

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
def main():
    d_width = len(im[0])
    d_height = len(im)
    if display:
        # Initialize OpenGL
        glutInit(sys.argv)
        
        # Set display mode
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)

        # Set size and position of window size
        glutInitWindowSize(d_width, d_height)
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
        glutMainLoop()
    else:
        # Initialize the library
        if not glfw.init():
            return
        # Set window hint NOT visible
        glfw.window_hint(glfw.VISIBLE, False)
        # Create a windowed mode window and its OpenGL context
        window = glfw.create_window(d_width, d_height, "hidden window", None, None)
        if not window:
            glfw.terminate()
            return

        # Make the window's context current
        glfw.make_context_current(window)

        glutInit(sys.argv)
        
        # Instantiate the cube
        cube = Cube()

        cube.init()
        
        cube.reshape(d_width, d_height)
        cube.display()
        
        image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
        imagearr = np.flip(imagearr, 0)

        cv2.imwrite(r"testresult.png", imagearr)

        glfw.destroy_window(window)
        glfw.terminate()
    
display = True
# Call the main function
if __name__ == '__main__':
    main()