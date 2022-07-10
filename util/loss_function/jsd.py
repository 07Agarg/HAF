import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

class JSDSimilarLoss(nn.Module):
    def __init__(self):
        super(JSDSimilarLoss, self).__init__()
        self.margin = 1.0

    def forward(self, p, q):
        mix = torch.clamp(0.5 * (p + q), 1e-7, 1).log()
        loss = F.kl_div(mix, p, reduction="batchmean")
        loss += F.kl_div(mix, q, reduction="batchmean")
        loss = 0.5 * loss
        return loss

class JSDDissimilarLoss(nn.Module):
    def __init__(self, margin):
        super(JSDDissimilarLoss, self).__init__()
        self.margin = margin

    def forward(self, p, q, device, margin):
        mix = torch.clamp(0.5 * (p + q), 1e-7, 1).log()
        clip = torch.from_numpy(np.array([0])).float().to(device)
        kl_p = F.kl_div(mix, p, reduction="none")
        kl_q = F.kl_div(mix, q, reduction="none")
        loss = torch.max((margin - kl_p.sum(dim=1)), clip).mean()
        loss += torch.max((margin - kl_q.sum(dim=1)), clip).mean()
        loss = 0.5 * loss
        return loss

