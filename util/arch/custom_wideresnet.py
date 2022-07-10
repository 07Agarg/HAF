import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from util.loss_function.jsd import JSDSimilarLoss, JSDDissimilarLoss
from MBM.better_mistakes.model.performance import accuracy
from MBM.better_mistakes.util.dissimilar_pairs import create_dissimilar_pairs
from cifar100.cifar100_get_tree_target_level5 import *

simloss = JSDSimilarLoss()
dissimiloss = JSDDissimilarLoss(margin=3.0)
criterion = nn.CrossEntropyLoss()
l1_to_l2 = map_l1_to_l2()
l2_to_l3 = map_l2_to_l3()
l3_to_l4 = map_l3_to_l4()
l4_to_l5 = map_l4_to_l5()

class WideResNet(nn.Module):
    def __init__(self, model, feature_size=128, num_classes=100):
        super(WideResNet, self).__init__()

        self.features_2 = model
        self.num_ftrs = 128 * 1 * 1
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes),
        )

    def forward(self, x, targets="ignored"):
        x = self.features_2(x)
        species_input = x
        # ---------------------------------------------------------------------------------------
        species_out = self.classifier_3(species_input)

        return species_out


class WideResNet_ours_l5_cejsd_wtconst_dissim(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(WideResNet_ours_l5_cejsd_wtconst_dissim, self).__init__()
        self.num_classes = num_classes
        self.features_2 = model
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size , num_classes[4]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)
        self.device = gpu

    def forward(self, x, targets):
        x = self.features_2(x)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_targets, l2_targets, l3_targets, l4_targets = get_targets(targets)
        l1_targets, l2_targets, l3_targets, l4_targets = l1_targets.to(self.device), l2_targets.to(self.device), l3_targets.to(self.device), l4_targets.to(self.device)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        ce_loss = criterion(l5_out, targets)
        acc_l5, _ = accuracy(l5_out, targets)
        acc_l5 = acc_l5[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        dissim_loss_l4 = torch.tensor(0).to(self.device)
        dissim_loss_l3 = torch.tensor(0).to(self.device)
        dissim_loss_l2 = torch.tensor(0).to(self.device)
        dissim_loss_l1 = torch.tensor(0).to(self.device)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        dissimilar_pairs_l1 = create_dissimilar_pairs(l1_targets, l1_out, self.device)
        dissim_loss_l1 = dissimiloss(dissimilar_pairs_l1[:, 0, :], dissimilar_pairs_l1[:, 1, :], self.device, margin=3.0)

        dissimilar_pairs_l2 = create_dissimilar_pairs(l2_targets, l2_out, self.device)
        dissim_loss_l2 = dissimiloss(dissimilar_pairs_l2[:, 0, :], dissimilar_pairs_l2[:, 1, :], self.device, margin=3.0)

        dissimilar_pairs_l3 = create_dissimilar_pairs(l3_targets, l3_out, self.device)
        dissim_loss_l3 = dissimiloss(dissimilar_pairs_l3[:, 0, :], dissimilar_pairs_l3[:, 1, :], self.device, margin=3.0)

        dissimilar_pairs_l4 = create_dissimilar_pairs(l4_targets, l4_out, self.device)
        dissim_loss_l4 = dissimiloss(dissimilar_pairs_l4[:, 0, :], dissimilar_pairs_l4[:, 1, :], self.device, margin=3.0)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # class and phylum.
        cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
        for l4 in range(len(l4_to_l5)):
            cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[l4_to_l5[l4], :], dim=0)
        cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
        l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # class and phylum.
        cl4_weight_hat = torch.zeros_like(classifier_3_weight).to(self.device)
        for l3 in range(len(l3_to_l4)):
            cl4_weight_hat[l3, :] = torch.sum(classifier_4_weight[l3_to_l4[l3], :], dim=0)
        cl4_weight_hat = F.normalize(cl4_weight_hat, p=2, dim=1)
        l2_weight_3_4 = torch.mean(cosine_sim(classifier_3_weight, cl4_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_2_3 = torch.tensor(0).to(self.device)  # class and phylum.
        cl3_weight_hat = torch.zeros_like(classifier_2_weight).to(self.device)
        for l2 in range(len(l2_to_l3)):
            cl3_weight_hat[l2, :] = torch.sum(classifier_3_weight[l2_to_l3[l2], :], dim=0)
        cl3_weight_hat = F.normalize(cl3_weight_hat, p=2, dim=1)
        l2_weight_2_3 = torch.mean(cosine_sim(classifier_2_weight, cl3_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_1_2 = torch.tensor(0).to(self.device)  # phylum and kingdom.
        cl2_weight_hat = torch.zeros_like(classifier_1_weight).to(self.device)
        for l1 in range(len(l1_to_l2)):
            cl2_weight_hat[l1, :] = torch.sum(classifier_2_weight[l1_to_l2[l1], :], dim=0)
        cl2_weight_hat = F.normalize(cl2_weight_hat, p=2, dim=1)
        l2_weight_1_2 = torch.mean(cosine_sim(classifier_1_weight, cl2_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
        for l4 in range(len(l4_to_l5)):
            p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
        jsd_level_4_5 = simloss(p_hat_l5, l4_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
        for l3 in range(len(l3_to_l4)):
            p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
        jsd_level_3_4 = simloss(p_hat_l4, l3_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
        for l2 in range(len(l2_to_l3)):
            p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
        jsd_level_2_3 = simloss(p_hat_l3, l2_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
        for l1 in range(len(l1_to_l2)):
            p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
        jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5
        dissim_loss = dissim_loss_l4 + dissim_loss_l3 + dissim_loss_l2 + dissim_loss_l1
        self.dissimiloss_list = [dissim_loss_l4.item(), dissim_loss_l3.item(), dissim_loss_l2.item(), dissim_loss_l1.item()]
        self.ce_loss_list = [ce_loss.item()]
        self.jsd_loss_list = [jsd_level_4_5.item(), jsd_level_3_4.item(), jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_4_5.item(), l2_weight_3_4.item(), l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l1, acc_l2, acc_l3, acc_l4, acc_l5]
        self.loss = ce_loss + jsd_loss - l2_weight_loss + dissim_loss
        return l5_out


class WideResNet_flamingo_l5(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(WideResNet_flamingo_l5, self).__init__()

        self.features_2 = model
        self.num_ftrs = 128 * 1 * 1
        self.classifier_1 = nn.Sequential(
            nn.Linear(512, num_classes[4]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(410, num_classes[3]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(308, num_classes[2]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(206, num_classes[1]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(103, num_classes[0]),)
        self.layer_outputs = {}
        self.device = gpu

    def forward(self, x, targets):
        x = self.features_2(x)
        x_1 = x[:, 0:102]
        x_2 = x[:, 102:204]
        x_3 = x[:, 204:306]
        x_4 = x[:, 306:409]
        x_5 = x[:, 409:512]
        x1_input = torch.cat([x_1, x_2.detach(), x_3.detach(), x_4.detach(), x_5.detach()], 1)
        x2_input = torch.cat([x_2, x_3.detach(), x_4.detach(), x_5.detach()], 1)
        x3_input = torch.cat([x_3, x_4.detach(), x_5.detach()], 1)
        x4_input = torch.cat([x_4, x_5.detach()], 1)
        x5_input = x_5
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_targets, l2_targets, l3_targets, l4_targets = get_targets(targets)
        l1_targets, l2_targets, l3_targets, l4_targets = l1_targets.to(self.device), l2_targets.to(self.device), l3_targets.to(self.device), l4_targets.to(self.device)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = self.classifier_1(x1_input)        # torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        ce_loss_l1 = criterion(l1_out, l1_targets)   # 3
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = self.classifier_2(x2_input)        # torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        ce_loss_l2 = criterion(l2_out, l2_targets)      # 4
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = self.classifier_3(x3_input)        #torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        ce_loss_l3 = criterion(l3_out, l3_targets)
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = self.classifier_4(x4_input)        # torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        ce_loss_l4 = criterion(l4_out, l4_targets)
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = self.classifier_5(x5_input)        # torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        ce_loss_l5 = criterion(l5_out, targets)
        acc_l5, _ = accuracy(l5_out, targets)
        acc_l5 = acc_l5[0].item()
        #---------------------------------------------------------------------------------------
        self.loss = ce_loss_l1 + ce_loss_l2 + ce_loss_l3 + ce_loss_l4 + ce_loss_l5
        self.ce_loss_list = [ce_loss_l5.item(), ce_loss_l4.item(), ce_loss_l3.item(), ce_loss_l2.item(), ce_loss_l1.item()]
        self.acc_list = [acc_l1, acc_l2, acc_l3, acc_l4, acc_l5]
        return l5_out


class WideResNet_ours_l5_cejsd(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):

        super(WideResNet_ours_l5_cejsd, self).__init__()
        self.num_classes = num_classes
        self.features_2 = model
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size , num_classes[4]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)
        self.device = gpu

    def forward(self, x, targets):
        x = self.features_2(x)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_targets, l2_targets, l3_targets, l4_targets = get_targets(targets)
        l1_targets, l2_targets, l3_targets, l4_targets = l1_targets.to(self.device), l2_targets.to(self.device), l3_targets.to(self.device), l4_targets.to(self.device)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = self.classifier_1(x)
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = self.classifier_2(x)
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = self.classifier_3(x)
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = self.classifier_4(x)
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = self.classifier_5(x)
        ce_loss = criterion(l5_out, targets)
        acc_l5, _ = accuracy(l5_out, targets)
        acc_l5 = acc_l5[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
        for l4 in range(len(l4_to_l5)):
            p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
        jsd_level_4_5 = simloss(p_hat_l5, l4_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
        for l3 in range(len(l3_to_l4)):
            p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
        jsd_level_3_4 = simloss(p_hat_l4, l3_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
        for l2 in range(len(l2_to_l3)):
            p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
        jsd_level_2_3 = simloss(p_hat_l3, l2_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
        for l1 in range(len(l1_to_l2)):
            p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
        jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        self.ce_loss_list = [ce_loss.item()]
        self.jsd_loss_list = [jsd_level_4_5.item(), jsd_level_3_4.item(), jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.acc_list = [acc_l1, acc_l2, acc_l3, acc_l4, acc_l5]
        self.loss = ce_loss + jsd_loss
        return l5_out


class WideResNet_ours_l5_cejsd_wtconst(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(WideResNet_ours_l5_cejsd_wtconst, self).__init__()
        self.num_classes = num_classes
        self.features_2 = model
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size , num_classes[4]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)
        self.device = gpu

    def forward(self, x, targets):
        x = self.features_2(x)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_targets, l2_targets, l3_targets, l4_targets = get_targets(targets)
        l1_targets, l2_targets, l3_targets, l4_targets = l1_targets.to(self.device), l2_targets.to(self.device), l3_targets.to(self.device), l4_targets.to(self.device)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        ce_loss = criterion(l5_out, targets)
        acc_l5, _ = accuracy(l5_out, targets)
        acc_l5 = acc_l5[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # class and phylum.
        cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
        for l4 in range(len(l4_to_l5)):
            cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[l4_to_l5[l4], :], dim=0)
        cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
        l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # class and phylum.
        cl4_weight_hat = torch.zeros_like(classifier_3_weight).to(self.device)
        for l3 in range(len(l3_to_l4)):
            cl4_weight_hat[l3, :] = torch.sum(classifier_4_weight[l3_to_l4[l3], :], dim=0)
        cl4_weight_hat = F.normalize(cl4_weight_hat, p=2, dim=1)
        l2_weight_3_4 = torch.mean(cosine_sim(classifier_3_weight, cl4_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_2_3 = torch.tensor(0).to(self.device)  # class and phylum.
        cl3_weight_hat = torch.zeros_like(classifier_2_weight).to(self.device)
        for l2 in range(len(l2_to_l3)):
            cl3_weight_hat[l2, :] = torch.sum(classifier_3_weight[l2_to_l3[l2], :], dim=0)
        cl3_weight_hat = F.normalize(cl3_weight_hat, p=2, dim=1)
        l2_weight_2_3 = torch.mean(cosine_sim(classifier_2_weight, cl3_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_1_2 = torch.tensor(0).to(self.device)  # phylum and kingdom.
        cl2_weight_hat = torch.zeros_like(classifier_1_weight).to(self.device)
        for l1 in range(len(l1_to_l2)):
            cl2_weight_hat[l1, :] = torch.sum(classifier_2_weight[l1_to_l2[l1], :], dim=0)
        cl2_weight_hat = F.normalize(cl2_weight_hat, p=2, dim=1)
        l2_weight_1_2 = torch.mean(cosine_sim(classifier_1_weight, cl2_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
        for l4 in range(len(l4_to_l5)):
            p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
        jsd_level_4_5 = simloss(p_hat_l5, l4_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
        for l3 in range(len(l3_to_l4)):
            p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
        jsd_level_3_4 = simloss(p_hat_l4, l3_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
        for l2 in range(len(l2_to_l3)):
            p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
        jsd_level_2_3 = simloss(p_hat_l3, l2_out)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
        for l1 in range(len(l1_to_l2)):
            p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
        jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5
        self.ce_loss_list = [ce_loss.item()]
        self.jsd_loss_list = [jsd_level_4_5.item(), jsd_level_3_4.item(), jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_4_5.item(), l2_weight_3_4.item(), l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l1, acc_l2, acc_l3, acc_l4, acc_l5]
        self.loss = ce_loss + jsd_loss - l2_weight_loss
        return l5_out
