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

def get_target_l3(targets):

    save_path = 'iNat19/inat19_tree_list_level3.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    family_target_list = []
    genus_target_list = []

    for i in range(targets.size(0)): 
        family_target_list.append(trees[targets[i]][2])
        genus_target_list.append(trees[targets[i]][1])

    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())   
    genus_target_list = Variable(torch.from_numpy(np.array(genus_target_list)).cuda())

    return family_target_list, genus_target_list

def get_target_l4(targets):

    save_path = 'iNat19/inat19_tree_list_level5.pkl'
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

def get_target_l5(targets):

    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    # with open('iNat19/inat19_tree_list_level7.pkl', 'rb') as file:
    #     trees7 = pickle.load(file)

    # with open('iNat19/inat19_tree_list_level3.pkl', 'rb') as file:
    #     trees3 = pickle.load(file)

    class_target_list = []
    order_target_list = []
    genus_target_list = []
    family_target_list = []

    for i in range(targets.size(0)): 
        class_target_list.append(trees[targets[i]][4])
        order_target_list.append(trees[targets[i]][3])
        family_target_list.append(trees[targets[i]][2])
        genus_target_list.append(trees[targets[i]][1])

    class_target_list = Variable(torch.from_numpy(np.array(class_target_list)).cuda())
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())
    genus_target_list = Variable(torch.from_numpy(np.array(genus_target_list)).cuda())

    return class_target_list, order_target_list, family_target_list, genus_target_list

def get_target_l6(targets):
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)
        trees = np.array(trees)

    with open('iNat19/inat19_tree_list_level5.pkl', 'rb') as file:
        trees5 = pickle.load(file)
        trees5 = np.array(trees5)

    with open('iNat19/inat19_tree_list_level3.pkl', 'rb') as file:
        trees3 = pickle.load(file)
        trees3 = np.array(trees3)

    phylum_target_list = []
    class_target_list = []
    order_target_list = []
    genus_target_list = []
    family_target_list = []

    for i in range(targets.size(0)): 
        phylum_target_list.append(trees[targets[i]][5])
        class_target_list.append(trees[targets[i]][4])
        order_target_list.append(trees[targets[i]][3])
        family_target_list.append(trees[targets[i]][2])
        genus_target_list.append(trees[targets[i]][1])

    phylum_target_list = Variable(torch.from_numpy(np.array(phylum_target_list)).cuda()) 
    class_target_list = Variable(torch.from_numpy(np.array(class_target_list)).cuda())
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())
    genus_target_list = Variable(torch.from_numpy(np.array(genus_target_list)).cuda())

    return phylum_target_list, class_target_list, order_target_list, family_target_list, genus_target_list    

def get_target_l7(targets):

    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    kingdom_target_list = []
    phylum_target_list = []
    class_target_list = []
    order_target_list = []
    genus_target_list = []
    family_target_list = []
    # import pdb; pdb.set_trace()
    for i in range(targets.size(0)): 
        kingdom_target_list.append(trees[targets[i]][6])
        phylum_target_list.append(trees[targets[i]][5])
        class_target_list.append(trees[targets[i]][4])
        order_target_list.append(trees[targets[i]][3])
        family_target_list.append(trees[targets[i]][2])
        genus_target_list.append(trees[targets[i]][1])

    kingdom_target_list = Variable(torch.from_numpy(np.array(kingdom_target_list)).cuda())
    phylum_target_list = Variable(torch.from_numpy(np.array(phylum_target_list)).cuda())
    class_target_list = Variable(torch.from_numpy(np.array(class_target_list)).cuda())
    order_target_list = Variable(torch.from_numpy(np.array(order_target_list)).cuda())   
    family_target_list = Variable(torch.from_numpy(np.array(family_target_list)).cuda())
    genus_target_list = Variable(torch.from_numpy(np.array(genus_target_list)).cuda())

    return kingdom_target_list, phylum_target_list, class_target_list, order_target_list, family_target_list, genus_target_list

# Map every coarse category level to their next finer-level category.
def map_genus_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for genus in np.unique(trees[:, 1]):
        idxs = np.where(trees[:, 1] == genus)[0]
        species_list.append(list(trees[idxs][:, 0]))
    return species_list

def map_family_to_genus():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    genus_list = []
    trees = np.array(trees)
    for family in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == family)[0]
        genus_list.append(list(np.unique(trees[idxs][:, 1])))
    return genus_list

def map_order_to_family():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    family_list = []
    trees = np.array(trees)
    for order in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == order)[0]
        family_list.append(list(np.unique(trees[idxs][:, 2])))
    return family_list

def map_class_to_order():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    order_list = []
    trees = np.array(trees)
    for class_ in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == class_)[0]
        order_list.append(list(np.unique(trees[idxs][:, 3])))
    return order_list

def map_phylum_to_class():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    class_list = []
    trees = np.array(trees)
    for phylum in np.unique(trees[:, 5]):
        idxs = np.where(trees[:, 5] == phylum)[0]
        class_list.append(list(np.unique(trees[idxs][:, 4])))
    return class_list

def map_kingdom_to_phylum():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    phylum_list = []
    trees = np.array(trees)
    for kingdom in np.unique(trees[:, 6]):
        idxs = np.where(trees[:, 6] == kingdom)[0]
        phylum_list.append(list(np.unique(trees[idxs][:, 5])))
    return phylum_list

# Map every coarse category level to species category.
def map_family_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for family in np.unique(trees[:, 2]):
        idxs = np.where(trees[:, 2] == family)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_order_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for order in np.unique(trees[:, 3]):
        idxs = np.where(trees[:, 3] == order)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_class_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for class_ in np.unique(trees[:, 4]):
        idxs = np.where(trees[:, 4] == class_)[0]
    species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list    

def map_phylum_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for phylum in np.unique(trees[:, 5]):
        idxs = np.where(trees[:, 5] == phylum)[0]
    species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list

def map_kingdom_to_species():
    save_path = 'iNat19/inat19_tree_list_level7.pkl'
    with open(save_path, 'rb') as file:
        trees = pickle.load(file)

    species_list = []
    trees = np.array(trees)
    for kingdom in np.unique(trees[:, 6]):
        idxs = np.where(trees[:, 6] == kingdom)[0]
        species_list.append(list(np.unique(trees[idxs][:, 0])))
    return species_list
