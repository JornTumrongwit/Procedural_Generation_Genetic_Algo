import glm
#growth parameters
p_diagonal = 5
p_straight = 5
p_building = 7
nodecount = 500
min_nodes = 50
unit_mod = 6480
straight_distance = 30 
min_straight_distance = 10 
diagonal_distance = 30 
min_diagonal_distance = 10 
start_offset_x = 0
start_offset_z = 20
expandedness = 1 #basically idk what to call the span to x vs span to z

max_diags = 10
max_straights = 10

h_edge = 150/unit_mod
v_edge = 100/(unit_mod*expandedness)

#Tower parameters
heightmin = 3
heightmax = 1200
widthmin = 3
widthmax = 10
depthmin = 0.5
depthmax = 2
heightmod = 50

max_x = 200
max_z = 30

min_x = -200
min_z = -200

horizontal_max = 100
vertical_max = 100

aspect = 1
image = 'testimage.png'

#Fitness parameters
bad = 10
good = 2

#camera
eye = glm.vec3(0.0, 3.0, 10.0)
center = glm.vec3(0.0, 1.0, -3.0)
up = glm.vec3(0.0, 3.0, 0.0)

#GA parameters
p_mu_grow = 0.125
p_mu_cut = 0.1 + p_mu_grow
p_mu_alter = 0.18 + p_mu_cut
nodes = nodecount
minnodes = 1
popsize = 180
generations = 15
k_tournament = 10
elitism = 40
refblur = 10
resblur = 30


def clamp(value, input, offset):
    return min(value+offset, max(-value+offset, input))