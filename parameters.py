import glm
#growth parameters
p_diagonal = 10
p_straight = 3
p_building = 1
nodes_max = 2000
straight_distance = 75
min_straight_distance = 100
diagonal_distance = 40
min_diagonal_distance = 1
unit_mod = 6480
start_offset_x = 0
start_offset_z = -0
expandedness = 1 #basically idk what to call the span to x vs span to z
nodecount = 300

h_edge = 150/unit_mod
v_edge = 100/(unit_mod*expandedness)

#Tower parameters
heightmin = 3
heightmax = 600
widthmin = 8
widthmax = 16
depthmin = 0.5
depthmax = 2
heightmod = 50

horizontal_max = 100
vertical_max = 100

aspect = 1
image = 'badapple.png'

#Fitness parameters
bad = -10
good = 6

#camera
eye = glm.vec3(0.0, 6.0, 5.0)
center = glm.vec3(0.0, -1.0, 0.0)
up = glm.vec3(0.0, 1.0, 0.0)

def clamp(value, input, offset):
    return min(value+offset, max(-value+offset, input))