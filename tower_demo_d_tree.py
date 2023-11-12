import matplotlib.pyplot as plt
import numpy as np
import random
import math
import time
import cv2
from enum import Enum
from enum import IntEnum
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

#mode parameters
depth_test = True
display = True
wire = False

class Direction(IntEnum):
    X = 0
    Y = 1

class Node(Enum):
    Center = 0 #offset_x, offset_y
    Diagonal = 1 #distance, angle
    Straight = 2 #distance, direction
    Building = 3 #rotation, width, depth, height

#"grammar"
diag_offset = 9
straight_offset = 8
building_offset = 8
center_offset = 5

class Branch:
    def __init__(self, branch, nodes, diag_slot, straight_slot, bld_slot, diags, straights, blds):
        self.branch = branch.copy()
        self.diags = diags
        self.straights = straights
        self.blds = blds
        self.diag_slot = diag_slot
        self.straight_slot = straight_slot
        self.bld_slot = bld_slot
        self.nodes = nodes

#Node structure: Type, parent offset, parameters, child offsets
#The tree
class Tree:
    def __init__(self, origin, minnodes, maxnodes) -> None:
        if origin == None:
            self.d_tree = []
            self.center()
            self.diag_slot = 1
            self.straight_slot = 0
            self.bld_slot = 0
            self.diags = 0
            self.straights = 0
            self.blds = 0
            self.nodes = 0
            create = random.randint(minnodes, maxnodes)
            while self.nodes < create:
                self.grow()
        else:
            self.d_tree = origin.d_tree.copy()
            self.diags = origin.diags
            self.straights = origin.straights
            self.blds = origin.blds
            self.diag_slot = origin.diag_slot
            self.straight_slot = origin.straight_slot
            self.bld_slot = origin.bld_slot
            self.nodes = origin.nodes
    
    def diag_param(self, x, z) -> None:
        this_distance = random.uniform(parameters.min_diagonal_distance, parameters.diagonal_distance)/parameters.unit_mod
        this_angle = random.uniform(0, 2*math.pi)
        params = []
        params.append(this_distance)
        params.append(this_angle)
        params.append(x)
        params.append(z)
        return params

    def diagonal(self, slot_ind, par_ind, x, z) -> None:
        next_diagonal = None
        next_straight = None
        next_bld = None 
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Diagonal)
        self.d_tree.append(-offset)
        params = self.diag_param(x, z)
        for item in params:
            self.d_tree.append(item)
        self.d_tree.append(next_diagonal)
        self.d_tree.append(next_straight)
        self.d_tree.append(next_bld)
        self.diags += 1
        self.diag_slot += 1
        self.straight_slot += 1
        self.bld_slot += 1
    
    def straight_param(self, x, z, plus_x, plus_z) -> None:
        this_distance = random.uniform(parameters.min_straight_distance, parameters.straight_distance)/parameters.unit_mod * random.choice([-1, 1])
        this_direction = random.choice(list(Direction))
        x += plus_x
        z += plus_z
        params = []
        params.append(this_distance)
        params.append(this_direction)
        params.append(x)
        params.append(z)
        return params
    
    #offset = 6
    def straight(self, slot_ind, par_ind, x, z, plus_x, plus_z) -> None:
        next_straight = None
        next_bld = None
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Straight)
        self.d_tree.append(-offset)
        params = self.straight_param(x, z, plus_x, plus_z)
        for item in params:
            self.d_tree.append(item)
        self.d_tree.append(next_straight)
        self.d_tree.append(next_bld)
        self.straights += 1
        self.straight_slot += 1
        self.bld_slot += 1
    
    def center(self) -> None:
        this_x_offset = parameters.start_offset_x
        this_z_offset = parameters.start_offset_z
        next_diagonal = None
        self.d_tree.append(Node.Center)
        self.d_tree.append(0)
        self.d_tree.append(this_x_offset)
        self.d_tree.append(this_z_offset)
        self.d_tree.append(next_diagonal)
    
    def bld_param(self, x, z):
        this_rotation = random.uniform(0, 360)
        this_width = random.uniform(parameters.widthmin, parameters.widthmax)/parameters.unit_mod
        this_depth = random.uniform(parameters.depthmin, parameters.depthmax)*this_width
        this_height = random.uniform(parameters.heightmin/parameters.heightmod, parameters.heightmax/parameters.heightmod)
        params = []
        params.append(this_rotation)
        params.append(this_width)
        params.append(this_depth)
        params.append(this_height)
        params.append(x)
        params.append(z)
        return params

    #offset = 6
    def building(self, slot_ind, par_ind, x, z) -> None:
        offset = len(self.d_tree) - par_ind
        self.d_tree[slot_ind] = offset
        self.d_tree.append(Node.Building)
        self.d_tree.append(-offset)
        params = self.bld_param(x, z)
        for item in params:
            self.d_tree.append(item)
        self.blds += 1

    #Growing
    def grow(self) -> None:
        prob_d = parameters.p_diagonal*self.diag_slot
        prob_s = parameters.p_straight*self.straight_slot
        prob_b = parameters.p_building*self.bld_slot
        totalprob = prob_d + prob_s + prob_b
        num_b = prob_b/totalprob
        num_s = prob_s/totalprob + num_b
        num_d = prob_d/totalprob + num_s
        choice = random.uniform(0, 1)
        candidates = []
        index = 0
        self.nodes += 1
        #generate
        if choice < num_b:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+diag_offset-1] == None:
                        candidates.append((index+diag_offset-1, index, 
                                        self.d_tree[index+4] + self.d_tree[index+2] * math.sin(self.d_tree[index+3]),
                                        self.d_tree[index+5] + self.d_tree[index+2] * math.cos(self.d_tree[index+3])))
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    if self.d_tree[index+straight_offset-1] == None:
                        candidates.append((index+straight_offset-1, index,
                                           self.d_tree[index+4] + (1-int(self.d_tree[index+3]))*self.d_tree[index+2],
                                           self.d_tree[index+5] + int(self.d_tree[index+3])*self.d_tree[index+2]))
                    index += straight_offset
            pick = random.choice(candidates)
            self.building(pick[0], pick[1], pick[2], pick[3])
            self.bld_slot -= 1

        elif choice < num_s:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+diag_offset-2] == None:
                        candidates.append((index+diag_offset-2, index, self.d_tree[index+4], self.d_tree[index+5],
                                           self.d_tree[index+2] * math.sin(self.d_tree[index+3]), self.d_tree[index+2] * math.cos(self.d_tree[index+3])))
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    if self.d_tree[index+straight_offset-2] == None:
                        candidates.append((index+straight_offset-2, index, self.d_tree[index+4], self.d_tree[index+5],
                                           (1-int(self.d_tree[index+3]))*self.d_tree[index+2], int(self.d_tree[index+3])*self.d_tree[index+2]))
                    index += straight_offset
            pick = random.choice(candidates)
            self.straight(pick[0], pick[1], pick[2], pick[3], pick[4], pick[5])
            self.straight_slot -= 1

        elif choice < num_d:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    if self.d_tree[index+center_offset-1] == None:
                        candidates.append((index+center_offset-1, index, self.d_tree[index+2], self.d_tree[index+3]))
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    if self.d_tree[index+diag_offset-3] == None:
                        candidates.append((index+diag_offset-3, index, self.d_tree[index+4], self.d_tree[index+5]))
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    index += straight_offset
            pick = random.choice(candidates)
            self.diagonal(pick[0], pick[1], pick[2], pick[3])
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
        i = center_offset
        if rng <= diag:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            candidate.append((i, self.d_tree[i+4], self.d_tree[i+5]))
                            i += diag_offset
                        case Node.Straight:
                            i += straight_offset
                        case Node.Building:
                            i += building_offset
                        case _:
                            raise Exception("NOT A NODE")
            index = random.choice(candidate)
            params = self.diag_param(index[1], index[2])
            adding = 2
            for item in params:
                self.d_tree[index[0]+adding] = item
                adding += 1
            
        elif rng <= straight:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            i += diag_offset
                        case Node.Straight:
                            candidate.append((i, self.d_tree[i+4], self.d_tree[i+5]))
                            i += straight_offset
                        case Node.Building:
                            i += building_offset
                        case _:
                            raise Exception("NOT A NODE")
                    
            index = random.choice(candidate)
            params = self.straight_param(index[1], index[2], 0, 0)
            adding = 2
            for item in params:
                self.d_tree[index[0]+adding] = item
                adding += 1
            
        else:
            while i < len(self.d_tree):
                if type(self.d_tree[i]) == Node:
                    match self.d_tree[i]:
                        case Node.Diagonal:
                            i += diag_offset
                        case Node.Straight:
                            i += straight_offset
                        case Node.Building:
                            candidate.append((i, self.d_tree[i+5], self.d_tree[i+6]))
                            i += building_offset
                        case _:
                            raise Exception("NOT A NODE")
            index = random.choice(candidate)
            params = self.bld_param(index[1], index[2])
            adding = 2
            for item in params:
                self.d_tree[index[0]+adding] = item
                adding += 1
    
    def remove_branch(self, index) -> None:
        if index == center_offset:
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
                                self.d_tree[parent_ind+diag_offset-3] = -end_offset
                            case Node.Center:
                                self.d_tree[parent_ind+center_offset-1] = -end_offset
                        index += diag_offset
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Straight:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-2] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-2] = -end_offset
                        index += straight_offset
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Building:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-1] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-1] = -end_offset
                        index += building_offset
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
                                self.d_tree[parent_ind+diag_offset-3] = None
                            case Node.Center:
                                self.d_tree[parent_ind+center_offset-1] = None
                        self.diags -= 1
                        if self.d_tree[index+diag_offset-3] is not None:
                            removal.add(self.d_tree[index+diag_offset-3] + index)
                        else:
                            self.diag_slot -= 1
                        if self.d_tree[index+diag_offset-2] is not None:
                            removal.add(self.d_tree[index+diag_offset-2] + index)
                        else:
                            self.straight_slot -= 1
                        if self.d_tree[index+diag_offset-1] is not None:
                            removal.add(self.d_tree[index+diag_offset-1] + index)
                        else:
                            self.bld_slot -= 1
                        index += diag_offset
                        removed += diag_offset
                        tail = index
                        slices.append((tail, tail))
                    case Node.Straight:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-2] = None
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-2] = None
                        self.straights -=1
                        if self.d_tree[index+straight_offset-2] is not None:
                            removal.add(self.d_tree[index+straight_offset-2] + index)
                        else:
                            self.straight_slot -= 1
                        if self.d_tree[index+straight_offset-1] is not None:
                            removal.add(self.d_tree[index+straight_offset-1] + index)
                        else:
                            self.bld_slot -= 1
                        index += straight_offset
                        removed += straight_offset
                        tail = index
                        slices.append((tail, tail))
                    case Node.Building:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-1] = None
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-1] = None
                        self.blds -=1
                        index += building_offset
                        removed += building_offset
                        tail = index
                        slices.append((tail, tail))
        result = []
        for slice in slices:
            result += self.d_tree[slice[0]: slice[1]]
        self.d_tree = result.copy()

    def cut(self) -> None:
        prob_d = (1-parameters.p_diagonal)*self.diags
        prob_s = (1-parameters.p_straight)*self.straights
        prob_b = (1-parameters.p_building)*self.blds
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
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    candidates.append(index)
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    index += straight_offset
            pick = random.choice(candidates)
            self.remove_branch(pick)
            self.bld_slot += 1

        elif choice < num_s:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    candidates.append(index)
                    index += straight_offset
            pick = random.choice(candidates)
            self.remove_branch(pick)
            self.straight_slot += 1

        elif choice < num_d:
            while index < len(self.d_tree):
                if self.d_tree[index] == Node.Center:
                    index += center_offset
                elif self.d_tree[index] == Node.Building:
                    index += building_offset
                elif self.d_tree[index] == Node.Diagonal:
                    candidates.append(index)
                    index += diag_offset
                elif self.d_tree[index] == Node.Straight:
                    index += straight_offset
            pick = random.choice(candidates)
            self.remove_branch(pick)
            self.diag_slot += 1


    #A different remove can result in an extract, takes in index of the child to cut, and the parent's slot
    def extract(self, index, slot) -> Branch:
        b_diags = 0
        b_straights = 0
        b_blds = 0
        b_diag_slot = 0
        b_straight_slot = 0
        b_bld_slot = 0
        b_nodes = 0
        match self.d_tree[index]:
            case Node.Diagonal:
                self.diag_slot += 1
            case Node.Straight:
                self.straight_slot += 1
        extractlst = []
        slices = [(0, index)] #array of indices for each individual slices
        removal = set() #array of indices to remove
        removal.add(index)
        tail = 0
        removed = 0
        parent = [0] * len(self.d_tree)
        self.d_tree[slot] = None
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
                                self.d_tree[parent_ind+diag_offset-3] = -end_offset
                            case Node.Center:
                                self.d_tree[parent_ind+center_offset-1] = -end_offset
                        index += diag_offset
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Straight:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-2] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-2] = -end_offset
                        index += straight_offset
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
                    case Node.Building:
                        match self.d_tree[parent_ind]:
                            case Node.Diagonal:
                                self.d_tree[parent_ind+diag_offset-1] = -end_offset
                            case Node.Straight:
                                self.d_tree[parent_ind+straight_offset-1] = -end_offset
                        index += building_offset
                        slices[len(slices)-1] = (slices[len(slices)-1][0], index)
            else:
                #remove this item, add it to the extracted list
                b_nodes += 1
                removal.remove(index)
                parent[index] = len(extractlst) # the index in the extracted list
                self.nodes -= 1
                parent_ind = self.d_tree[index + 1] + index
                parent_extract = parent[parent_ind]
                offset = parent_extract - parent[index]
                match self.d_tree[index]:
                    case Node.Diagonal:
                        b_diags += 1
                        self.diags -= 1
                        if self.d_tree[index+diag_offset-3] is not None:
                            removal.add(self.d_tree[index+diag_offset-3] + index)
                        else:
                            self.diag_slot -= 1
                            b_diag_slot += 1
                        if self.d_tree[index+diag_offset-2] is not None:
                            removal.add(self.d_tree[index+diag_offset-2] + index)
                        else:
                            self.straight_slot -= 1
                            b_straight_slot += 1
                        if self.d_tree[index+diag_offset-1] is not None:
                            removal.add(self.d_tree[index+diag_offset-1] + index)
                        else:
                            self.bld_slot -= 1
                            b_bld_slot += 1
                        #Adding stuffs to extracted list
                        item_ind = len(extractlst)
                        extractlst += self.d_tree[index:index+diag_offset]
                        if item_ind > 0:
                            extractlst[item_ind+1] = offset
                            match self.d_tree[parent_ind]:
                                case Node.Diagonal:
                                    extractlst[parent_extract+diag_offset-3] = -offset
                                case Node.Center:
                                    extractlst[parent_extract+center_offset-1] = -offset
                        else:
                            extractlst[1] = 0
                        index += diag_offset
                        removed += diag_offset
                        tail = index
                        slices.append((tail, tail))
                    case Node.Straight:
                        self.straights -=1
                        b_straights += 1
                        if self.d_tree[index+straight_offset-2] is not None:
                            removal.add(self.d_tree[index+straight_offset-2] + index)
                        else:
                            self.straight_slot -= 1
                            b_straight_slot += 1
                        if self.d_tree[index+straight_offset-1] is not None:
                            removal.add(self.d_tree[index+straight_offset-1] + index)
                        else:
                            self.bld_slot -= 1
                            b_bld_slot += 1
                        #Adding stuffs to extracted list
                        item_ind = len(extractlst)
                        extractlst += self.d_tree[index:index+straight_offset]
                        if item_ind > 0:
                            extractlst[item_ind+1] = offset
                            match self.d_tree[parent_ind]:
                                case Node.Diagonal:
                                    extractlst[parent_extract+diag_offset-2] = -offset
                                case Node.Straight:
                                    extractlst[parent_extract+straight_offset-2] = -offset
                        else:
                            extractlst[1] = 0
                        index += straight_offset
                        removed += straight_offset
                        tail = index
                        slices.append((tail, tail))
                    case Node.Building:
                        self.blds -=1
                        b_blds += 1
                        #Adding stuffs to extracted list
                        item_ind = len(extractlst)
                        extractlst += self.d_tree[index:index+building_offset]
                        if item_ind > 0:
                            extractlst[item_ind+1] = offset
                            match self.d_tree[parent_ind]:
                                case Node.Diagonal:
                                    extractlst[parent_extract+diag_offset-1] = -offset
                                case Node.Straight:
                                    extractlst[parent_extract+straight_offset-1] = -offset
                        else:
                            extractlst[1] = 0
                        index += building_offset
                        removed += building_offset
                        tail = index
                        slices.append((tail, tail))
        result = []
        for slice in slices:
            result += self.d_tree[slice[0]: slice[1]]
        self.d_tree = result.copy()
        return Branch(extractlst, b_nodes, b_diag_slot, b_straight_slot,
                      b_bld_slot, b_diags, b_straights, b_blds)
    
    '''
    #returns: index of child to cut, parent's slot, index of parent, type of parent, type of slot
    def random_crosspoint(self) -> (int, int, Node, Node):
        candidates = []
        index = 0
        while index < len(self.d_tree):
            match self.d_tree[index]:
                case Node.Center:
                    index += center_offset
                case Node.Diagonal:
                    if self.d_tree[index+diag_offset-3] is not None:
                        candidates.append((index + self.d_tree[index+diag_offset-3], index+diag_offset-3, index,
                                           Node.Diagonal, Node.Diagonal))
                    if self.d_tree[index+diag_offset-2] is not None:
                        candidates.append((index + self.d_tree[index+diag_offset-2], index+diag_offset-2, index,
                                           Node.Diagonal, Node.Straight))
                    if self.d_tree[index+diag_offset-1] is not None:
                        candidates.append((index + self.d_tree[index+diag_offset-1], index+diag_offset-1, index,
                                           Node.Diagonal, Node.Building))
                    index += diag_offset
                case Node.Straight:
                    if self.d_tree[index+straight_offset-2] is not None:
                        candidates.append((index + self.d_tree[index+straight_offset-2], index+straight_offset-2, index, 
                                           Node.Straight, Node.Straight))
                    if self.d_tree[index+straight_offset-1] is not None:
                        candidates.append((index + self.d_tree[index+straight_offset-1], index+straight_offset-1, index, 
                                           Node.Straight, Node.Building))
                    index += straight_offset
                case Node.Building:
                    index += building_offset
        if len(candidates) == 0:
            return (None, None, None, None)
        return random.choice(candidates)
    
    '''
    def pick_crosspoint(self, nodetype, connectType) -> (int, int, int):
        candidate = []
        index = 0
        match nodetype:
            case Node.Diagonal:
                while index < len(self.d_tree):
                    match self.d_tree[index]:
                        case Node.Center:
                            index += center_offset
                        case Node.Diagonal:
                            match connectType:
                                case Node.Diagonal:
                                    slot = diag_offset-3
                                case Node.Straight:
                                    slot = diag_offset-2
                                case Node.Building:
                                    slot = diag_offset-1
                            if self.d_tree[index+slot] is not None:
                                candidate.append((index+self.d_tree[index+slot], index + slot, index))
                            index += diag_offset
                        case Node.Straight:
                            index += straight_offset
                        case Node.Building:
                            index += building_offset

            case Node.Straight:
                while index < len(self.d_tree):
                    match self.d_tree[index]:
                        case Node.Center:
                            index += center_offset
                        case Node.Diagonal:
                            index += diag_offset
                        case Node.Straight:
                            match connectType:
                                case Node.Straight:
                                    slot = straight_offset-2
                                case Node.Building:
                                    slot = straight_offset-1
                            if self.d_tree[index+slot] is not None:
                                candidate.append((index+self.d_tree[index+slot], index + slot, index))
                            index += straight_offset
                        case Node.Building:
                            index += building_offset
        if len(candidate) == 0:
            return None
        else:
            return random.choice(candidate)
        
    def stick(self, branch, par_slot, par_ind):
        child_ind = len(self.d_tree)
        offset = len(self.d_tree) - par_ind
        self.d_tree+=branch.branch.copy()
        self.d_tree[par_slot] = offset
        self.d_tree[child_ind+1] = -offset
        self.nodes += branch.nodes
        self.diags += branch.diags
        self.blds += branch.blds
        self.straights += branch.straights
        self.diag_slot += branch.diag_slot
        self.straight_slot += branch.straight_slot
        self.diag_slot += branch.diag_slot
        self.bld_slot += branch.bld_slot
    
def crossover(parent1, parent2) -> (Tree, Tree):
    test_child = Tree(parent1, None, None)
    test_child2 = Tree(parent2, None, None)
    tree_cross = test_child.pick_crosspoint(Node.Diagonal, Node.Diagonal)
    if tree_cross is None:
        return (test_child, test_child2)
    index, slot, par_index = tree_cross
    tree2_cross = test_child2.pick_crosspoint(Node.Diagonal, Node.Diagonal)
    if tree2_cross is None:
        return (test_child, test_child2)
    index2, slot2, par_index2 = tree2_cross
    test_branch = test_child.extract(index, slot)
    test_branch2 = test_child2.extract(index2, slot2)
    test_child.stick(test_branch2, slot, par_index)
    test_child2.stick(test_branch, slot2, par_index2)
    return (test_child, test_child2)

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
    return traverse_tree(tree, center_offset)

def traverse_tree(tree, index):
    result = []
    match tree.d_tree[index]:
        case Node.Diagonal:
            if tree.d_tree[index+diag_offset-3] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+diag_offset-3])
            if tree.d_tree[index+diag_offset-2] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+diag_offset-2])
            if tree.d_tree[index+diag_offset-1] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+diag_offset-1])
            return result
        case Node.Straight:
            if tree.d_tree[index+straight_offset-2] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+straight_offset-2])
            if tree.d_tree[index+straight_offset-1] is not None:
                result += traverse_tree(tree, index + tree.d_tree[index+straight_offset-1])
            return result
        case Node.Building:
            return [Tower(tree.d_tree[index+5], tree.d_tree[index+3], 
                          tree.d_tree[index+4], tree.d_tree[index+6], tree.d_tree[index+7]*parameters.expandedness, tree.d_tree[index+2])]

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
        self.aspect = parameters.aspect
        self.eye = parameters.eye
        self.def_eye = self.eye
        self.center = parameters.center
        self.def_center = self.center
        self.up = parameters.up
        x = glm.cross(self.up, self.eye-self.center)
        y = glm.cross(self.eye-self.center, x)
        self.up = glm.normalize(y)
        self.def_up = self.up

    # Initialize
    def init(self):
        self.towers = None
        # Set background to black
        glClearColor(0.0, 0.0, 0.0, 0.0)

        # Set the shade model to flat
        glShadeModel(GL_FLAT)

        if depth_test == True:
            #Depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
    
    def changetower(self, towers):
        self.towers = towers

    # Draw cube
    def draw(self):

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[0], self.up[0], self.up[1], self.up[2])
        glScalef(self.zoom, self.zoom, self.zoom)

        # Draw solid cube
        for tower in self.towers:
            #object transforms
            glPushMatrix()
            glColor3f(1.0, 1.0, 1.0)
            glTranslatef(tower.x, 0.0, tower.z)
            glRotatef(tower.rotation, 0.0, 1.0, 0.0)
            glScalef(tower.width, tower.height, tower.depth)
            glTranslatef(0.5, 0.5, 0.5)
            glutSolidCube(1.0)
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

    # Draw cube
    def draw_nowire(self):

        # Reset the matrix
        glLoadIdentity()

        # Set the camera
        gluLookAt(self.eye[0], self.eye[1], self.eye[2], self.center[0], self.center[1], self.center[0], self.up[0], self.up[1], self.up[2])
        glScalef(self.zoom, self.zoom, self.zoom)

        # Draw solid cube
        for tower in self.towers:
            #object transforms
            glPushMatrix()
            glColor3f(1.0, 1.0, 1.0)
            glTranslatef(tower.x, 0.0, tower.z)
            glRotatef(tower.rotation, 0.0, 1.0, 0.0)
            glScalef(tower.width, tower.height, tower.depth)
            glTranslatef(0.5, 0.5, 0.5)
            glutSolidCube(1.0)
            glPopMatrix()
            

        glFlush()

    # The display function
    def display_nowire(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Draw cube
        self.draw_nowire()

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
def display_towers(towers):
    wire = True
    d_width = len(im[0])
    d_height = len(im)
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
    cube.changetower(towers)

    # The callback for display function
    glutDisplayFunc(cube.display)

    # The callback for reshape function
    glutReshapeFunc(cube.reshape)

    # The callback function for keyboard controls
    glutSpecialFunc(cube.special)

    # The callback function for normal keyboard controls
    glutKeyboardFunc(cube.keyb)
    glutMainLoop()

def save_towers(towers):
    d_width = len(im[0])
    d_height = len(im)
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
    cube.changetower(towers)
    
    cube.reshape(d_width, d_height)
    cube.display_nowire()
    
    image_buffer = glReadPixels(0, 0, d_width, d_height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    imagearr = np.frombuffer(image_buffer, dtype=np.uint8).reshape(d_height, d_width, 3)
    imagearr = np.flip(imagearr, 0)

    cv2.imwrite(r"testresult.png", imagearr)

    glfw.destroy_window(window)
    glfw.terminate()

# Call the main function
if __name__ == '__main__':
    im = cv2.imread(parameters.image)
    im = np.divide(im, 255/2)
    im = np.add(im, -1)
    parameters.h_edge *= len(im)
    parameters.v_edge *= len(im[0])
    parameters.unit_mod /= len(im)
    #parameters.expandedness *= len(im)/len(im[0])
    test_tree = Tree(None, parameters.min_nodes, parameters.nodecount)
    test_tree.cut()
    for i in range(30):
        test_tree.alter()
    test_tree2 = Tree(None, parameters.min_nodes, parameters.nodecount)
    test_child, test_child2 = crossover(test_tree, test_tree2)
    towers = maketowers(test_child2)
    display_towers(towers)