# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:42:55 2021

@author: Ashima
"""

import numpy as np
import torch
import torch.nn as nn

trees = [
	[0, 4, 5, 3, 0],
    [1, 1, 6, 0, 0],
    [2, 14, 3, 0, 0],
    [3, 8, 3, 0, 0],
    [4, 0, 6, 0, 0],
    [5, 6, 0, 1, 1],    # 0
    [6, 7, 4, 0, 0],
    [7, 7, 4, 0, 0],
    [8, 18, 7, 1, 1],
    [9, 3, 0, 1, 1],    # 0 
    [10, 3, 0, 1, 1],   # 0
    [11, 14, 3, 0, 0],
    [12, 9, 1, 1, 1],
    [13, 18, 7, 1, 1],
    [14, 7, 4, 0, 0],
    [15, 11, 3, 0, 0],
    [16, 3, 0, 1, 1],   # 0
    [17, 9, 1, 1, 1],
    [18, 7, 4, 0, 0],
    [19, 11, 3, 0, 0],
    [20, 6, 0, 1, 1],   # 0 
    [21, 11, 3, 0, 0],
    [22, 5, 0, 1, 1],   # 0
    [23, 10, 2, 2, 1],
    [24, 7, 4, 0, 0],
    [25, 6, 0, 1, 1],   # 0
    [26, 13, 4, 0, 0],
    [27, 15, 4, 0, 0],
    [28, 3, 0, 1, 1],     # 0
    [29, 15, 4, 0, 0],
    [30, 0, 6, 0, 0],
    [31, 11, 3, 0, 0],
    [32, 1, 6, 0, 0],
    [33, 10, 2, 2, 1],
    [34, 12, 3, 0, 0],
    [35, 14, 3, 0, 0],
    [36, 16, 3, 0, 0],
    [37, 9, 1, 1, 1],
    [38, 11, 3, 0, 0],
    [39, 5, 0, 1, 1],     # 0
    [40, 5, 0, 1, 1],     # 0
    [41, 19, 7, 1, 1],
    [42, 8, 3, 0, 0],
    [43, 8, 3, 0, 0],
    [44, 15, 4, 0, 0],
    [45, 13, 4, 0, 0],
    [46, 14, 3, 0, 0],
    [47, 17, 5, 3, 0],
    [48, 18, 7, 1, 1],
    [49, 10, 2, 2, 1],
    [50, 16, 3, 0, 0],
    [51, 4, 5, 3, 0],
    [52, 17, 5, 3, 0],
    [53, 4, 5, 3, 0],
    [54, 2, 5, 3, 0],
    [55, 0, 6, 0, 0],
    [56, 17, 5, 3, 0],
    [57, 4, 5, 3, 0],
    [58, 18, 7, 1, 1],
    [59, 17, 5, 3, 0],
    [60, 10, 2, 2, 1],
    [61, 3, 0, 1, 1],     # 0
    [62, 2, 5, 3, 0],
    [63, 12, 3, 0, 0],
    [64, 12, 3, 0, 0],
    [65, 16, 3, 0, 0],
    [66, 12, 3, 0, 0],
    [67, 1, 6, 0, 0],
    [68, 9, 1, 1, 1],
    [69, 19, 7, 1, 1],
    [70, 2, 5, 3, 0],
    [71, 10, 2, 2, 1],
    [72, 0, 6, 0, 0],
    [73, 1, 6, 0, 0],
    [74, 16, 3, 0, 0],
    [75, 12, 3, 0, 0],
    [76, 9, 1, 1, 1],
    [77, 13, 4, 0, 0],
    [78, 15, 4, 0, 0],
    [79, 13, 4, 0, 0],
    [80, 16, 3, 0, 0],
    [81, 19, 7, 1, 1],
    [82, 2, 5, 3, 0],
    [83, 4, 5, 3, 0],
    [84, 6, 0, 1, 1],
    [85, 19, 7, 1, 1],
    [86, 5, 0, 1, 1],        # 0
    [87, 5, 0, 1, 1],         # 0 
    [88, 8, 3, 0, 0],
    [89, 19, 7, 1, 1],
    [90, 18, 7, 1, 1],
    [91, 1, 6, 0, 0],
    [92, 2, 5, 3, 0],
    [93, 15, 4, 0, 0],
    [94, 6, 0, 1, 1],         # 0
    [95, 0, 6, 0, 0],
    [96, 17, 5, 3, 0],
    [97, 8, 3, 0, 0],
    [98, 14, 3, 0, 0],
    [99, 13, 4, 0, 0]]

def get_targets(targets):
    global trees
    l1_target_list = []
    l2_target_list = []
    l3_target_list = []
    l4_target_list = []

    for i in range(targets.size(0)):
        l1_target_list.append(trees[targets[i]][4])
        l2_target_list.append(trees[targets[i]][3])
        l3_target_list.append(trees[targets[i]][2])
        l4_target_list.append(trees[targets[i]][1])
    l1_target_list = torch.from_numpy(np.array(l1_target_list))
    l2_target_list = torch.from_numpy(np.array(l2_target_list))
    l3_target_list = torch.from_numpy(np.array(l3_target_list))
    l4_target_list = torch.from_numpy(np.array(l4_target_list))
    return l1_target_list, l2_target_list, l3_target_list, l4_target_list

def map_l4_to_l5():
    global trees
    species_list = []
    trees = np.array(trees)
    for l4 in np.unique(trees[:, 1]):
        idxs = np.where(trees[:, 1] == l4)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l3_to_l4():
    global trees
    species_list = []
    trees = np.array(trees)
    for l3 in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == l3)[0]
        species_list.append(list(np.unique(trees[idxs][:, 1])))
    return species_list

def map_l2_to_l3():
    global trees
    species_list = []
    trees = np.array(trees)
    for l2 in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == l2)[0]
        species_list.append(list(np.unique(trees[idxs][:, 2])))
    return species_list

def map_l1_to_l2():
    global trees
    species_list = []
    trees = np.array(trees)
    for l1 in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == l1)[0]
        species_list.append(list(np.unique(trees[idxs][:, 3])))
    return species_list

def map_l3_to_l5():
    global trees
    species_list = []
    trees = np.array(trees)
    for l3 in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == l3)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list    

def map_l2_to_l5():
    global trees
    species_list = []
    trees = np.array(trees)
    for l2 in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == l2)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list    

def map_l1_to_l5():
    global trees
    species_list = []
    trees = np.array(trees)
    for l1 in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == l1)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list
