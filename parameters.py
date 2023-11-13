import glm
#growth parameters
p_diagonal = 10
p_straight = 10
p_building = 3
nodecount = 200
min_nodes = 50
straight_distance = 40
min_straight_distance = 20
diagonal_distance = 40
min_diagonal_distance = 20
unit_mod = 6480
start_offset_x = 0
start_offset_z = 10
expandedness = 1 #basically idk what to call the span to x vs span to z

max_diags = 10
max_straights = 10

h_edge = 150/unit_mod
v_edge = 100/(unit_mod*expandedness)

#Tower parameters
heightmin = 3
heightmax = 1200
widthmin = 10
widthmax = 25
depthmin = 0.5
depthmax = 2
heightmod = 50

horizontal_max = 100
vertical_max = 100

aspect = 1
image = 'testimage.png'

#Fitness parameters
bad = -10
good = 2

#camera
eye = glm.vec3(0.0, 3.0, 12.0)
center = glm.vec3(0.0, 1.0, 0.0)
up = glm.vec3(0.0, 1.0, 0.0)

#GA parameters
p_mu_grow = 0.125
p_mu_cut = 0.1 + p_mu_grow
p_mu_alter = 0.18 + p_mu_cut
nodes = nodecount
minnodes = 1
popsize = 180
generations = 130
k_tournament = 10
elitism = 40
refblur = 25
resblur = 50


def clamp(value, input, offset):
    return min(value+offset, max(-value+offset, input))