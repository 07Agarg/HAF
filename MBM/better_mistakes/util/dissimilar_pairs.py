import numpy as np
import torch
import random

def create_dissimilar_pairs(targets, probabs, device):
    limit = 256
    count = 0
    temp = list(zip(targets, probabs))
    random.shuffle(temp)
    targets, probabs = zip(*temp)
    targets = torch.stack(targets)
    probabs = torch.stack(probabs)
    dissimilar_pairs = torch.zeros((limit, 2, probabs.size()[1])).to(device)
    for i in range(len(targets)):
        if count >= limit: 
            break
        for j in range(len(targets) - 1):
            if count >= limit: 
                break
            if targets[j] != targets[j+1]:
                dissimilar_pairs[count][0] = probabs[j]
                dissimilar_pairs[count][1] = probabs[j+1]
                count = count + 1
    return dissimilar_pairs
