# -*- coding: utf-8 -*-
"""
Created on Thu May  6 11:13:13 2021

@author: Ashima
"""

import pdb
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pickle


def get_target_l4(targets):

    save_path = 'inat19_tree_list_level5.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    order_target_list = []
    family_target_list = []
    genus_target_list = []

    for i in range(targets.size(0)): 
        order_target_list.append(trees[targets[i]][3])
        family_target_list.append(trees[targets[i]][2])
        genus_target_list.append(trees[targets[i]][1])

    order_target_list = Variable(torch.from_numpy(np.array(order_target_list))).cuda()
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())   
    genus_target_list = Variable(torch.from_numpy(np.array(genus_target_list)).cuda())

    return order_target_list, family_target_list, genus_target_list

# def temp():
#     save_path = 'tiered_tree_list_level13.pkl'
#     import pdb; pdb.set_trace()
#     with open(save_path, 'rb') as file:
#         trees = pickle.load(file)
# temp()

def get_target_l12(targets):

    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    l1_target_list = []
    l2_target_list = []
    l3_target_list = []
    l4_target_list = []
    l5_target_list = []
    l6_target_list = []
    l7_target_list = []
    l8_target_list = []
    l9_target_list = []
    l10_target_list = []
    l11_target_list = []

    for i in range(targets.size(0)): 
        l1_target_list.append(trees[targets[i]][11])
        l2_target_list.append(trees[targets[i]][10])
        l3_target_list.append(trees[targets[i]][9])
        l4_target_list.append(trees[targets[i]][8])
        l5_target_list.append(trees[targets[i]][7])
        l6_target_list.append(trees[targets[i]][6])
        l7_target_list.append(trees[targets[i]][5])
        l8_target_list.append(trees[targets[i]][4])
        l9_target_list.append(trees[targets[i]][3])
        l10_target_list.append(trees[targets[i]][2])
        l11_target_list.append(trees[targets[i]][1])

    l1_target_list = Variable(torch.from_numpy(np.array(l1_target_list)).cuda())
    l2_target_list = Variable(torch.from_numpy(np.array(l2_target_list)).cuda())
    l3_target_list = Variable(torch.from_numpy(np.array(l3_target_list)).cuda())
    l4_target_list = Variable(torch.from_numpy(np.array(l4_target_list)).cuda()) 
    l5_target_list = Variable(torch.from_numpy(np.array(l5_target_list)).cuda())
    l6_target_list = Variable(torch.from_numpy(np.array(l6_target_list)).cuda())
    l7_target_list = Variable(torch.from_numpy(np.array(l7_target_list)).cuda())
    l8_target_list = Variable(torch.from_numpy(np.array(l8_target_list)).cuda())
    l9_target_list = Variable(torch.from_numpy(np.array(l9_target_list)).cuda())
    l10_target_list = Variable(torch.from_numpy(np.array(l10_target_list)).cuda()) 
    l11_target_list = Variable(torch.from_numpy(np.array(l11_target_list)).cuda())

    return l1_target_list, l2_target_list, l3_target_list, l4_target_list, l5_target_list, \
            l6_target_list, l7_target_list, l8_target_list, l9_target_list, l10_target_list, l11_target_list



def map_l11_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l11 in np.unique(trees[:, 1]):
        idxs = np.where(trees[:, 1] == l11)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l10_to_l11():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l10 in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == l10)[0]
        species_list.append(list(np.unique(trees[idxs][:, 1])))
    return species_list

def map_l9_to_l10():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l9 in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == l9)[0]
        species_list.append(list(np.unique(trees[idxs][:, 2])))
    return species_list

def map_l8_to_l9():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l8 in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == l8)[0]
        species_list.append(list(np.unique(trees[idxs][:, 3])))
    return species_list

def map_l7_to_l8(): 
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l7 in np.unique(trees[:, 5]):
        idxs = np.where(trees[:, 5] == l7)[0]
        species_list.append(list(np.unique(trees[idxs][:, 4])))
    return species_list

def map_l6_to_l7():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l6 in np.unique(trees[:, 6]):
        idxs = np.where(trees[:, 6] == l6)[0]
        species_list.append(list(np.unique(trees[idxs][:, 5])))
    return species_list

def map_l5_to_l6():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l5 in np.unique(trees[:, 7]):
        idxs = np.where(trees[:, 7] == l5)[0]
        species_list.append(list(np.unique(trees[idxs][:, 6])))
    return species_list

def map_l4_to_l5():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l4 in np.unique(trees[:, 8]):
        idxs = np.where(trees[:, 8] == l4)[0]
        species_list.append(list(np.unique(trees[idxs][:, 7])))
    return species_list

def map_l3_to_l4():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l3 in np.unique(trees[:, 9]):
        idxs = np.where(trees[:, 9] == l3)[0]
        species_list.append(list(np.unique(trees[idxs][:, 8])))
    return species_list

def map_l2_to_l3():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l2 in np.unique(trees[:, 10]):
        idxs = np.where(trees[:, 10] == l2)[0]
        species_list.append(list(np.unique(trees[idxs][:, 9])))
    return species_list

def map_l1_to_l2():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l1 in np.unique(trees[:, 11]):
        idxs = np.where(trees[:, 11] == l1)[0]
        species_list.append(list(np.unique(trees[idxs][:, 10])))
    return species_list



def map_l10_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l10 in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == l10)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l9_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l9 in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == l9)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l8_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l8 in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == l8)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l7_to_l12(): 
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l7 in np.unique(trees[:, 5]):
        idxs = np.where(trees[:, 5] == l7)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l6_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l6 in np.unique(trees[:, 6]):
        idxs = np.where(trees[:, 6] == l6)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l5_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l5 in np.unique(trees[:, 7]):
        idxs = np.where(trees[:, 7] == l5)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l4_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l4 in np.unique(trees[:, 8]):
        idxs = np.where(trees[:, 8] == l4)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l3_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l3 in np.unique(trees[:, 9]):
        idxs = np.where(trees[:, 9] == l3)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l2_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l2 in np.unique(trees[:, 10]):
        idxs = np.where(trees[:, 10] == l2)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_l1_to_l12():
    save_path = 'tiered_imagenet/tiered_tree_list_level13.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for l1 in np.unique(trees[:, 11]):
        idxs = np.where(trees[:, 11] == l1)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

















