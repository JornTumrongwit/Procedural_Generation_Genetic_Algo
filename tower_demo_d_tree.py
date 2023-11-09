import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import cv2
from enum import Enum
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
start = time.time()

#mode parameters
depth_test = True
display = True
wire = True

#growth parameters
p_diagonal = 0.8
p_straight = 0.9
p_building = 0.3
nodes_max = 1000
straight_distance = 100
diagonal_distance = 500
unit_mod = 10
start_offset_x = 0
start_offset_z = -10
expandedness = 0.75 #basically idk what to call the span to x vs span to z
nodecount = 1000

class Direction(Enum):
    X = 0
    Y = 1

class Node(Enum):
    Center = 0 #offset_x, offset_y
    Diagonal = 1 #distance, angle
    Straight = 2 #distance, direction
    Building = 3 #rotation, width, depth, height

#Tower parameters
heightmin = 3
heightmax = 600
widthmin = 0.5
widthmax = 2
depthmin = 0.5
depthmax = 2
heightmod = 50

#Node structure: Type, parent offset, parameters, child offsets
#The tree
class Tree:
    def __init__(self, d_tree, diags, straights, blds, diag_slot, straight_slot, bld_slot, nodes) -> None:
        self.d_tree = d_tree
        self.diags = diags
        self.straights = straights
        self.blds = blds
        self.diag_slot = diag_slot
        self.straight_slot = straight_slot
        self.bld_slot = bld_slot
        self.nodes = nodes
        if d_tree == None:
            self.d_tree = []
            self.center()
            self.diag_slot = 1
            self.straight_slot = 0
            self.bld_slot = 0
            self.diags = 0
            self.straights = 0
            self.blds = 0
            self.nodes = random.randint(0, nodecount)
            for i in range(self.nodes):
                self.grow()

    #offset = 7
    def diagonal(self, slot_ind, par_ind) -> None:
        this_distance = random.uniform(0, diagonal_distance)/unit_mod
        this_angle = random.uniform(0, 2*math.pi)
        next_diagonal = None
        next_straight = None
        next_bld = None 
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Diagonal)
        self.d_tree.append(-offset)
        self.d_tree.append(this_distance)
        self.d_tree.append(this_angle)
        self.d_tree.append(next_diagonal)
        self.d_tree.append(next_straight)
        self.d_tree.append(next_bld)
        self.diags += 1
        self.diag_slot += 1
        self.straight_slot += 1
        self.bld_slot += 1
    
    #offset = 6
    def straight(self, slot_ind, par_ind) -> None:
        this_distance = random.uniform(-straight_distance, straight_distance)/unit_mod
        this_direction = random.choice(list(Direction))
        next_straight = None
        next_bld = None
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Straight)
        self.d_tree.append(-offset)
        self.d_tree.append(this_distance)
        self.d_tree.append(this_direction)
        self.d_tree.append(next_straight)
        self.d_tree.append(next_bld)
        self.straights += 1
        self.straight_slot += 1
        self.bld_slot += 1

    #offset = 5 
    def center(self) -> None:
        this_x_offset = start_offset_x
        this_z_offset = start_offset_z
        next_diagonal = None
        self.d_tree.append(Node.Center)
        self.d_tree.append(0)
        self.d_tree.append(this_x_offset)
        self.d_tree.append(this_z_offset)
        self.d_tree.append(next_diagonal)
    
    #offset = 6
    def building(self, slot_ind, par_ind) -> None:
        this_rotation = random.uniform(0, 360)
        this_width = random.uniform(widthmin, widthmax)
        this_depth = random.uniform(depthmin, depthmax)*this_width
        this_height = random.uniform(heightmin/heightmod, heightmax/heightmod)
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Building)
        self.d_tree.append(-offset)
        self.d_tree.append(this_rotation)
        self.d_tree.append(this_width)
        self.d_tree.append(this_depth)
        self.d_tree.append(this_height)
        self.blds += 1

    #Growing
    def grow(self) -> None:
        prob_d = p_diagonal*self.diag_slot
        prob_s = p_straight*self.straight_slot
        prob_b = p_building*self.bld_slot
        totalprob = prob_d + prob_s + prob_b
        num_b = prob_b/totalprob
        num_s = prob_s/totalprob + num_b
        num_d = prob_d/totalprob + num_s
        choice = random.uniform(0, 1)
        candidates = []
        index = 0
        #generate
        if choice < num_b:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+6] == None:
                        candidates.append((index+6, index))
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    if self.d_tree[index+5] == None:
                        candidates.append((index+5, index))
                    index += 6
            pick = random.choice(candidates)
            self.building(pick[0], pick[1])
            self.bld_slot -= 1

        elif choice < num_s:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+5] == None:
                        candidates.append((index+5, index))
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    if self.d_tree[index+4] == None:
                        candidates.append((index+4, index))
                    index += 6
            pick = random.choice(candidates)
            self.straight(pick[0], pick[1])
            self.straight_slot -= 1

        elif choice < num_d:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    if self.d_tree[index+4] == None:
                        candidates.append((index+4, index))
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+4] == None:
                        candidates.append((index+4, index))
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    index += 6
            pick = random.choice(candidates)
            self.diagonal(pick[0], pick[1])
            self.diag_slot -= 1

        #in the VERY LOW CHANCE that for some reason the randomizer hits that perfect decimal 
        #inaccuracy from 100 just say it's a dud mutation
    
    #alter
    def alter(self) -> None:
        diag = self.diags
        straight = self.straights + diag
        blds = self.blds + straight
        rng = random.randint(0, blds)
        candidate = []
        i = 5
        if rng <= diag:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            candidate.append(i)
                            i += 7
                        case Node.Straight:
                            i += 6
                        case Node.Building:
                            i += 6
                        case _:
                            raise Exception("NOT A NODE")
            index = random.choice(candidate)
            self.d_tree[index+2] = random.uniform(0, diagonal_distance)/unit_mod
            self.d_tree[index+3] = random.uniform(0, 2*math.pi)

        elif rng <= straight:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            i += 7
                        case Node.Straight:
                            candidate.append(i)
                            i += 6
                        case Node.Building:
                            i += 6
                        case _:
                            raise Exception("NOT A NODE")
                    
            index = random.choice(candidate)
            self.d_tree[index+2] = random.uniform(-straight_distance, straight_distance)/unit_mod
            self.d_tree[index+3] = random.choice(list(Direction))
            
        else:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            i += 7
                        case Node.Straight:
                            i += 6
                        case Node.Building:
                            candidate.append(i)
                            i += 6
                        case _:
                            raise Exception("NOT A NODE")
            index = random.choice(candidate)
            self.d_tree[index+2] = random.uniform(0, 360)
            self.d_tree[index+3] = random.uniform(widthmin, widthmax)
            self.d_tree[index+4] = random.uniform(depthmin, depthmax)*self.d_tree[index+3]
            self.d_tree[index+5] = random.uniform(heightmin/heightmod, heightmax/heightmod)
    
    def remove_branch(self, index) -> None:
        if index == 5:
            return #please don't just completely disappear
        slices = [(0, index)] #array of indices for each individual slices
        removal = set() #array of indices to remove
        removal.add(index)
        tail = 0
        removed = 0
        parent = [0] * len(self.d_tree)
        while index < len(self.d_tree):
            if index not in removal:
                #Not removing, adjust the offsets
                #Change parent's offset, and change child's offset
                parent_ind = self.d_tree[index + 1] + index
                parent_removed = parent[parent_ind]
                end_offset = self.d_tree[index + 1] + removed - parent_removed
                parent[index] = removed
                self.d_tree[index + 1] = end_offset
                match self.d_tree[index]:
                    case Node.Diagonal:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+4] = -end_offset
                            case Node.Center:
                                self.d_tree[parent_ind+4] = -end_offset
                        index += 7
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Straight:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+5] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+4] = -end_offset
                        index += 6
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Building:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+6] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+5] = -end_offset
                        index += 6
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
            else:
                #remove this item
                removal.remove(index)
                self.nodes -= 1
                parent_ind = self.d_tree[index + 1] + index
                match self.d_tree[index]:
                    case Node.Diagonal:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+4] = None
                            case Node.Center:
                                self.d_tree[parent_ind+4] = None
                        self.diags -= 1
                        self.diag_slot += 1
                        if self.d_tree[index+4] is not None:
                            removal.add(self.d_tree[index+4] + index)
                        if self.d_tree[index+5] is not None:
                            removal.add(self.d_tree[index+5] + index)
                        if self.d_tree[index+6] is not None:
                            removal.add(self.d_tree[index+6] + index)
                        index += 7
                        removed += 7
                        tail = index
                        slices.append((tail, tail))
                    case Node.Straight:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+5] = None
                            case Node.Straight:
                                self.d_tree[parent_ind+4] = None
                        self.straights -=1
                        self.straight_slot += 1
                        if self.d_tree[index+4] is not None:
                            removal.add(self.d_tree[index+4] + index)
                        if self.d_tree[index+5] is not None:
                            removal.add(self.d_tree[index+5] + index)
                        index += 6
                        removed += 6
                        tail = index
                        slices.append((tail, tail))
                    case Node.Building:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+6] = None
                            case Node.Straight:
                                self.d_tree[parent_ind+5] = None
                        self.blds -=1
                        self.bld_slot += 1
                        index += 6
                        removed += 6
                        tail = index
                        slices.append((tail, tail))
        result = []
        for slice in slices:
            result += self.d_tree[slice[0]: slice[1]]
        self.d_tree = result

    #PAIN
    def cut(self) -> None:
        prob_d = (1-p_diagonal)*self.diags
        prob_s = (1-p_straight)*self.straights
        prob_b = (1-p_building)*self.blds
        totalprob = prob_d + prob_s + prob_b
        num_b = prob_b/totalprob
        num_s = prob_s/totalprob + num_b
        num_d = prob_d/totalprob + num_s
        choice = random.uniform(0, 1)
        candidates = []
        index = 0
        #choose
        if choice < num_b:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    candidates.append(index)
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    index += 6
            pick = random.choice(candidates)
            self.remove_branch(pick)

        elif choice < num_s:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    candidates.append(index)
                    index += 6
            pick = random.choice(candidates)
            self.remove_branch(pick)

        elif choice < num_d:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += 5
                elif self.d_tree[index] == Node.Building:
                    index += 6
                elif self.d_tree[index] == Node.Diagonal:
                    candidates.append(index)
                    index += 7
                elif self.d_tree[index] == Node.Straight:
                    index += 6
            pick = random.choice(candidates)
            self.remove_branch(pick)

    #copying, useful for child
    def copy(self) -> None:
        d_tree = self.d_tree.copy()
        return Tree(d_tree,
                    self.diags, self.straights, self.blds, 
                    self.diag_slot, self.straight_slot, self.bld_slot, 
                    self.nodes)
    
test_tree = Tree(None, None, None, None, None, None, None, None)
test_tree.cut()
#testing
alters = 30
for i in range(alters):
    test_tree.alter()
grow = 10
for i in range(grow):
    test_tree.grow()
print(f"NODES: {test_tree.nodes}")
#class for tower generation
class Tower:
    def __init__(self, height, width, depth, x, z, rotation) -> None:
        self.height = height
        self.width = width
        self.depth = depth
        self.x = x
        self.z = z
        self.rotation = rotation

def maketowers(tree):
    return traverse_tree(tree, 5, tree.d_tree[2], tree.d_tree[3])

def traverse_tree(tree, index, x, z):
    result = []
    match tree.d_tree[index]:
        case Node.Diagonal:
            if tree.d_tree[index+4] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+4], x, z)
            if tree.d_tree[index+5] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+5], 
                                        x + (tree.d_tree[index+2] * math.sin(tree.d_tree[index+3])), 
                                        z + (tree.d_tree[index+2] * math.cos(tree.d_tree[index+3])))
            if tree.d_tree[index+6] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+6], 
                                        x + (tree.d_tree[index+2] * math.sin(tree.d_tree[index+3])), 
                                        z + (tree.d_tree[index+2] * math.cos(tree.d_tree[index+3])))
            return result
        case Node.Straight:
            match tree.d_tree[index+3]:
                case Direction.X:
                    x += tree.d_tree[index+2]
                case Direction.Y:
                    z += tree.d_tree[index+2]
            if tree.d_tree[index+4] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+4], x, z)
            if tree.d_tree[index+5] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+5], x, z)
            return result
        case Node.Building:
            return [Tower(tree.d_tree[index+5], tree.d_tree[index+3], 
                          tree.d_tree[index+4], x, z*expandedness, tree.d_tree[index+2])]

towers = maketowers(test_tree)

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
        self.eye = glm.vec3(0.0, 6.0, 5.0)
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

        if depth_test == True:
            #Depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
        

    # Draw cube
    def draw(self):

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
            glTranslatef(tower.x, 0.0, tower.z)
            glRotatef(tower.rotation, 0.0, 1.0, 0.0)
            glScalef(tower.width, tower.height, tower.depth)
            glTranslatef(0.5, 0.5, 0.5)
            glutSolidCube(1.0)
            if wire:
                glColor3f(1.0, 0.0, 0.0)
                glScalef(1.01, 1.01, 1.01)
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
        glutInitWindowPosition(0, 0)

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
        im2 = np.divide(imagearr, 255/2)
        im2 = np.add(im2, -1)
        score = np.sum(np.multiply(im, im2))

        cv2.imwrite(r"testresult.png", imagearr)

        glfw.destroy_window(window)
        glfw.terminate()

# Call the main function
if __name__ == '__main__':
    main()