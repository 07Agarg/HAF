import pickle

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torchvision.models as models

from iNat19.inat_get_target_tree import *
from tiered_imagenet.tiered_get_target_tree import *
from MBM.better_mistakes.util.dissimilar_pairs import create_dissimilar_pairs
from util.loss_function.jsd import JSDSimilarLoss, JSDDissimilarLoss
from MBM.better_mistakes.model.performance import accuracy

import torch.nn.functional as F

simloss = JSDSimilarLoss()
dissimiloss = JSDDissimilarLoss(margin=3.0)
criterion = nn.CrossEntropyLoss()

# inaturalist-19
num_classes = [1010, 72, 57, 34, 9, 4, 3]        # 7-level setup
genus_to_species = map_genus_to_species()
family_to_genus = map_family_to_genus()
order_to_family = map_order_to_family()
class_to_order = map_class_to_order()
phylum_to_class = map_phylum_to_class()
kingdom_to_phylum = map_kingdom_to_phylum()

# tiered-imagenet
num_classes = [608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2]         # 12-level setup
l11_to_l12 = map_l11_to_l12()
l10_to_l11 = map_l10_to_l11()
l9_to_l10 = map_l9_to_l10()
l8_to_l9 = map_l8_to_l9()
l7_to_l8 = map_l7_to_l8()
l6_to_l7 = map_l6_to_l7()
l5_to_l6 = map_l5_to_l6()
l4_to_l5 = map_l4_to_l5()
l3_to_l4 = map_l3_to_l4()
l2_to_l3 = map_l2_to_l3()
l1_to_l2 = map_l1_to_l2()


class ResNet18(nn.Module):
    """Used for Cross-entropy, CRM, Making-better-mistakes."""

    def __init__(self, model, feature_size, num_classes):
        super(ResNet18, self).__init__()

        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes), )

    def forward(self, x, target="ignored"):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x)  # N * 512
        #x = F.normalize(x, p=2, dim=1)
        species_input = x
        species_out = self.classifier_3(species_input)
        return species_out


class ResNet18_flamingo_l7(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_flamingo_l7, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[6]), )  # number of kingdom
        self.classifier_2 = nn.Sequential(
            nn.Linear(515, self.num_classes[5]), )  # num of phylum
        self.classifier_3 = nn.Sequential(
            nn.Linear(430, self.num_classes[4]), )  # num of order
        self.classifier_4 = nn.Sequential(
            nn.Linear(345, self.num_classes[3]), )  # num of class
        self.classifier_5 = nn.Sequential(
            nn.Linear(260, self.num_classes[2]), )  # num of family
        self.classifier_6 = nn.Sequential(
            nn.Linear(175, self.num_classes[1]), )  # num of genus
        self.classifier_7 = nn.Sequential(
            nn.Linear(90, self.num_classes[0]), )  # num of species
        # self.layer_outputs = {}
        self.device = gpu

    # def forward(self, x, targets):
    def forward(self, x, targets):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x)  # N * 512

        x_1 = x[:, 0:self.feature_size // 7]  # [:,0: 85]
        x_2 = x[:, self.feature_size // 7:self.feature_size // 7 * 2]  # [:,85: 170]
        x_3 = x[:, self.feature_size // 7 * 2:self.feature_size // 7 * 3]  # [:,170: 255]
        x_4 = x[:, self.feature_size // 7 * 3:self.feature_size // 7 * 4]  # [:,170: 340]
        x_5 = x[:, self.feature_size // 7 * 4:self.feature_size // 7 * 5]  # [:,340: 425]
        x_6 = x[:, self.feature_size // 7 * 5:self.feature_size // 7 * 6]  # [:,425: 510]
        x_7 = x[:, self.feature_size // 7 * 6:self.feature_size]  # [:,510: 600]

        kingdom_input = torch.cat(
            [x_1, x_2.detach(), x_3.detach(), x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach()], 1)  # (b, 600)
        phylum_input = torch.cat([x_2, x_3.detach(), x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach()],
                                 1)  # (b, 515)
        class_input = torch.cat([x_3, x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach()], 1)  # (b, 430)
        order_input = torch.cat([x_4, x_5.detach(), x_6.detach(), x_7.detach()], 1)  # (b, 345)
        family_input = torch.cat([x_5, x_6.detach(), x_7.detach()], 1)  # (b, 260)
        genus_input = torch.cat([x_6, x_7.detach()],
                                1)  # (b, 175)
        species_input = x_7  # (b, 90)
        # ---------------------------------------------------------------------------------------
        [kingdom_targets, phylum_targets, class_targets, \
         order_targets, family_targets, genus_targets] = get_target_l7(targets)
        # ---------------------------------------------------------------------------------------
        kingdom_out = self.classifier_1(kingdom_input)
        ce_loss_kingdom = criterion(kingdom_out, kingdom_targets)  # 3
        acc_kingdom, _ = accuracy(kingdom_out, kingdom_targets)
        acc_kingdom = acc_kingdom[0].item()
        # ---------------------------------------------------------------------------------------
        phylum_out = self.classifier_2(phylum_input)
        ce_loss_phylum = criterion(phylum_out, phylum_targets)  # 4
        acc_phylum, _ = accuracy(phylum_out, phylum_targets)
        acc_phylum = acc_phylum[0].item()
        # ---------------------------------------------------------------------------------------
        class_out = self.classifier_3(class_input)
        ce_loss_class = criterion(class_out, class_targets)  # 9
        acc_class, _ = accuracy(class_out, class_targets)
        acc_class = acc_class[0].item()
        # ---------------------------------------------------------------------------------------
        order_out = self.classifier_4(order_input)
        ce_loss_order = criterion(order_out, order_targets)  # 34
        acc_order, _ = accuracy(order_out, order_targets)
        acc_order = acc_order[0].item()
        # ---------------------------------------------------------------------------------------
        family_out = self.classifier_5(family_input)
        ce_loss_family = criterion(family_out, family_targets)  # 57
        acc_family, _ = accuracy(family_out, family_targets)
        acc_family = acc_family[0].item()
        # ---------------------------------------------------------------------------------------
        genus_out = self.classifier_6(genus_input)
        ce_loss_genus = criterion(genus_out, genus_targets)  # 72
        acc_genus, _ = accuracy(genus_out, genus_targets)
        acc_genus = acc_genus[0].item()
        # ---------------------------------------------------------------------------------------
        species_out = self.classifier_7(species_input)
        ce_loss_species = criterion(species_out, targets)  # 1010
        acc_species, _ = accuracy(species_out, targets)
        acc_species = acc_species[0].item()
        # ---------------------------------------------------------------------------------------
        ce_loss = ce_loss_kingdom + ce_loss_phylum + ce_loss_class + ce_loss_order + ce_loss_family + ce_loss_genus + ce_loss_species
        self.ce_loss_list = [ce_loss_species.item(), ce_loss_genus.item(), ce_loss_family.item(),
         ce_loss_order.item(), ce_loss_class.item(), ce_loss_phylum.item(), ce_loss_kingdom.item()]
        self.acc_list = [acc_species, acc_genus, acc_family, acc_order, acc_class, acc_phylum, acc_kingdom]
        self.loss = ce_loss
        return species_out


class ResNet18_flamingo_l12(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):

        super(ResNet18_flamingo_l12, self).__init__() 

        self.features_2 =  nn.Sequential(*list(model.children())[:-2])
        self.feature_size = feature_size
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1                  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )

        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, num_classes[11]),)              # coareset level
        self.classifier_2 = nn.Sequential(
            nn.Linear(550, num_classes[10]),)                       
        self.classifier_3 = nn.Sequential(
            nn.Linear(500, num_classes[9]),)                        
        self.classifier_4 = nn.Sequential(
            nn.Linear(450, num_classes[8]),)                        
        self.classifier_5 = nn.Sequential(
            nn.Linear(400, num_classes[7]),)                        
        self.classifier_6 = nn.Sequential(
            nn.Linear(350, num_classes[6]),)                       
        self.classifier_7 = nn.Sequential(
            nn.Linear(300, num_classes[5]),)                         
        self.classifier_8 = nn.Sequential(
            nn.Linear(250, num_classes[4]),)                         
        self.classifier_9 = nn.Sequential(
            nn.Linear(200, num_classes[3]),)                         
        self.classifier_10 = nn.Sequential(
            nn.Linear(150, num_classes[2]),)                         
        self.classifier_11 = nn.Sequential(
            nn.Linear(100, num_classes[1]),)                         
        self.classifier_12 = nn.Sequential(
            nn.Linear(50, num_classes[0]),)                         # finest level
  
    def forward(self, x, targets):
        x = self.features_2(x)   
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512

        x_1 = x[:, 0:self.feature_size//12]
        x_2 = x[:,self.feature_size// 12    :self.feature_size// 12*2]
        x_3 = x[:,self.feature_size// 12 * 2:self.feature_size// 12*3]    
        x_4 = x[:,self.feature_size// 12 * 3:self.feature_size// 12*4]
        x_5 = x[:,self.feature_size// 12 * 4:self.feature_size// 12*5]     
        x_6 = x[:,self.feature_size// 12 * 5:self.feature_size// 12*6]     
        x_7 = x[:,self.feature_size// 12 * 6:self.feature_size// 12*7]     
        x_8 = x[:,self.feature_size// 12 * 7:self.feature_size// 12*8]    
        x_9 = x[:,self.feature_size// 12 * 8:self.feature_size// 12*9]
        x_10 = x[:,self.feature_size// 12 * 9:self.feature_size// 12*10]   
        x_11 = x[:,self.feature_size// 12 * 10:self.feature_size// 12*11]  
        x_12 = x[:,self.feature_size// 12 * 11:self.feature_size]
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_input = torch.cat([x_1, x_2.detach(), x_3.detach(), x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l2_input = torch.cat([x_2, x_3.detach(), x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l3_input = torch.cat([x_3, x_4.detach(), x_5.detach(), x_6.detach(), x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l4_input = torch.cat([x_4, x_5.detach(), x_6.detach(), x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l5_input = torch.cat([x_5, x_6.detach(), x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l6_input = torch.cat([x_6, x_7.detach(), x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l7_input = torch.cat([x_7, x_8.detach(), x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l8_input = torch.cat([x_8, x_9.detach(), x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l9_input = torch.cat([x_9, x_10.detach(), x_11.detach(), x_12.detach()], 1)
        l10_input = torch.cat([x_10, x_11.detach(), x_12.detach()], 1)
        l11_input = torch.cat([x_11, x_12.detach()], 1)
        l12_input = x_12
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [l1_targets, l2_targets, l3_targets, l4_targets, l5_targets, l6_targets, l7_targets, \
        l8_targets, l9_targets, l10_targets, l11_targets] = get_target_l12(targets)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = self.classifier_1(l1_input)
        ce_loss_l1 = criterion(l1_out, l1_targets)              # 2
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = self.classifier_2(l2_input)
        ce_loss_l2 = criterion(l2_out, l2_targets)              # 5
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = self.classifier_3(l3_input)
        ce_loss_l3 = criterion(l3_out, l3_targets)              # 21
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = self.classifier_4(l4_input)
        ce_loss_l4 = criterion(l4_out, l4_targets)              # 35
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = self.classifier_5(l5_input)
        ce_loss_l5 = criterion(l5_out, l5_targets)              # 86
        acc_l5, _ = accuracy(l5_out, l5_targets)
        acc_l5 = acc_l5[0].item()
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = self.classifier_6(l6_input)
        ce_loss_l6 = criterion(l6_out, l6_targets)              # 159 
        acc_l6, _ = accuracy(l6_out, l6_targets)
        acc_l6 = acc_l6[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = self.classifier_7(l7_input)
        ce_loss_l7 = criterion(l7_out, l7_targets)              # 270
        acc_l7, _ = accuracy(l7_out, l7_targets)
        acc_l7 = acc_l7[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l8_out = self.classifier_8(l8_input)
        ce_loss_l8 = criterion(l8_out, l8_targets)              # 422
        acc_l8, _ = accuracy(l8_out, l8_targets)
        acc_l8 = acc_l8[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l9_out = self.classifier_9(l9_input)
        ce_loss_l9 = criterion(l9_out, l9_targets)              # 510
        acc_l9, _ = accuracy(l9_out, l9_targets)
        acc_l9 = acc_l9[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l10_out = self.classifier_10(l10_input)
        ce_loss_l10 = criterion(l10_out, l10_targets)           # 584
        acc_l10, _ = accuracy(l10_out, l10_targets)
        acc_l10 = acc_l10[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l11_out = self.classifier_11(l11_input)
        ce_loss_l11 = criterion(l11_out, l11_targets)           # 607
        acc_l11, _ = accuracy(l11_out, l11_targets)
        acc_l11 = acc_l11[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l12_out = self.classifier_12(l12_input)
        ce_loss_l12 = criterion(l12_out, targets)           # 608
        acc_l12, _ = accuracy(l12_out, targets)
        acc_l12 = acc_l12[0].item()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
        ce_loss =  ce_loss_l12 + ce_loss_l11 + ce_loss_l10 + ce_loss_l9 + ce_loss_l8 + ce_loss_l7 + ce_loss_l6 + ce_loss_l5 + ce_loss_l4 + ce_loss_l3 + ce_loss_l2 + ce_loss_l1
        self.ce_loss_list = [ce_loss_l12.item(), ce_loss_l11.item(), ce_loss_l10.item(), ce_loss_l9.item(), ce_loss_l8.item(),
                            ce_loss_l7.item(), ce_loss_l6.item(), ce_loss_l5.item(), ce_loss_l4.item(), ce_loss_l3.item(),
                            ce_loss_l2.item(), ce_loss_l1.item()]
        self.acc_list = [acc_l12, acc_l11, acc_l10, acc_l9, acc_l8, acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]
        self.loss = ce_loss
        return l12_out
        

class ResNet18_ours_l7_cejsd(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):

        super(ResNet18_ours_l7_cejsd, self).__init__() 
        self.device = gpu
        self.features_2 =  nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1                  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, num_classes[6]),)                        # num of class 
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[5]),)                        # num of order
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[4]),)                        # num of family
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)                        # num of genus
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)                        # num of family
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)                        # num of genus
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)                        # num of species
  
    def forward(self, x, targets):
        x = self.features_2(x)   
        x = self.max(x)

        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [kingdom_targets, phylum_targets, class_targets, \
        order_targets, family_targets, genus_targets] = get_target_l7(targets)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        kingdom_out = self.classifier_1(x)
        acc_kingdom, _ = accuracy(kingdom_out, kingdom_targets)
        acc_kingdom = acc_kingdom[0].item()
        # ce_loss_kingdom = criterion(kingdom_out, kingdom_targets)   # 3
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        phylum_out = self.classifier_2(x)
        acc_phylum, _ = accuracy(phylum_out, phylum_targets)
        acc_phylum = acc_phylum[0].item()
        # ce_loss_phylum = criterion(phylum_out, phylum_targets)      # 4
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        class_out = self.classifier_3(x)
        acc_class, _ = accuracy(class_out, class_targets)
        acc_class = acc_class[0].item()
        # ce_loss_class = criterion(class_out, class_targets)         # 9
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        order_out = self.classifier_4(x)
        acc_order, _ = accuracy(order_out, order_targets)
        acc_order = acc_order[0].item()        
        # ce_loss_order = criterion(order_out, order_targets)         # 34
        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        family_out = self.classifier_5(x)
        acc_family, _ = accuracy(family_out, family_targets)
        acc_family = acc_family[0].item()
        # ce_loss_family = criterion(family_out, family_targets)      # 57
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        genus_out = self.classifier_6(x)
        acc_genus, _ = accuracy(genus_out, genus_targets)
        acc_genus = acc_genus[0].item()
        # ce_loss_genus = criterion(genus_out, genus_targets)         # 72
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        species_out = self.classifier_7(x)                      
        ce_loss_species = criterion(species_out, targets)           # 1010
        acc_species, _ = accuracy(species_out, targets)
        acc_species = acc_species[0].item()
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        kingdom_out = F.softmax(kingdom_out, dim=1)
        phylum_out = F.softmax(phylum_out, dim=1)
        class_out = F.softmax(class_out, dim=1)
        order_out = F.softmax(order_out, dim=1)
        family_out = F.softmax(family_out, dim=1)
        genus_out = F.softmax(genus_out, dim=1)
        species_out = F.softmax(species_out, dim=1)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_6_7 = torch.tensor(0).to(self.device)                         # species and genus.
        p_hat_species = torch.zeros_like(genus_out).to(self.device)
        for genus in range(len(genus_to_species)):
            p_hat_species[:, genus] = torch.sum(species_out[:, genus_to_species[genus]], dim=1)
        jsd_level_6_7 = simloss(p_hat_species, genus_out)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_5_6 = torch.tensor(0).to(self.device)                         # genus and family.
        p_hat_genus = torch.zeros_like(family_out).to(self.device)
        for family in range(len(family_to_genus)):
            p_hat_genus[:, family] = torch.sum(genus_out[:, family_to_genus[family]], dim=1)
        jsd_level_5_6 = simloss(p_hat_genus, family_out)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)                         # family and order.
        p_hat_family = torch.zeros_like(order_out).to(self.device)
        for order in range(len(order_to_family)):
            p_hat_family[:, order] = torch.sum(family_out[:, order_to_family[order]], dim=1)
        jsd_level_4_5 = simloss(p_hat_family, order_out)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)                       # order and class. 
        p_hat_order = torch.zeros_like(class_out).to(self.device)
        for class_ in range(len(class_to_order)):
            p_hat_order[:, class_] = torch.sum(order_out[:, class_to_order[class_]], dim=1)
        jsd_level_3_4 = simloss(p_hat_order, class_out)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)                       # class and phylum.
        p_hat_class = torch.zeros_like(phylum_out).to(self.device)
        for phylum in range(len(phylum_to_class)):
            p_hat_class[:, phylum] = torch.sum(class_out[:, phylum_to_class[phylum]], dim=1)
        jsd_level_2_3 = simloss(p_hat_class, phylum_out)
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)                         # phylum and kingdom.
        p_hat_phylum = torch.zeros_like(kingdom_out).to(self.device)
        for kingdom in range(len(kingdom_to_phylum)):
            p_hat_phylum[:, kingdom] = torch.sum(phylum_out[:, kingdom_to_phylum[kingdom]], dim=1)
        jsd_level_1_2 = simloss(p_hat_phylum, kingdom_out)
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss = ce_loss_species
        self.acc_list = [acc_species, acc_genus, acc_family, acc_order, acc_class, acc_phylum, acc_kingdom]
        self.ce_loss_list = [ce_loss_species.item()]
        self.jsd_loss_list = [jsd_level_6_7.item(), jsd_level_5_6.item(),
                              jsd_level_4_5.item(), jsd_level_3_4.item(),
                              jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.loss = ce_loss + jsd_loss
        return species_out

# no weight_utils.norm
class ResNet18_ours_l7_cejsd_wtconst(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_ours_l7_cejsd_wtconst, self).__init__()
        self.device = gpu
        self.num_classes = num_classes
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, num_classes[6]),)                        # num of class 
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[5]),)                        # num of order
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[4]),)                        # num of family
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)                        # num of genus
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)                        # num of family
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)                        # num of genus
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)                        # num of species

    def forward(self, x, targets):
        x = self.features_2(x)
        x = self.max(x)

        x = x.view(x.size(0), -1)
        x = self.features_1(x)  # N * 512
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [kingdom_targets, phylum_targets, class_targets, \
         order_targets, family_targets, genus_targets] = get_target_l7(targets)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        classifier_6_weight = F.normalize(self.classifier_6[0].weight, p=2, dim=1)
        classifier_7_weight = F.normalize(self.classifier_7[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, kingdom_targets)
        acc_l1 = acc_l1[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, phylum_targets)
        acc_l2 = acc_l2[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, class_targets)
        acc_l3 = acc_l3[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, order_targets)
        acc_l4 = acc_l4[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        acc_l5, _ = accuracy(l5_out, family_targets)
        acc_l5 = acc_l5[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = torch.mm(x, classifier_6_weight.t()) + self.classifier_6[0].bias
        acc_l6, _ = accuracy(l6_out, genus_targets)
        acc_l6 = acc_l6[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = torch.mm(x, classifier_7_weight.t()) + self.classifier_7[0].bias
        ce_loss_l7 = criterion(l7_out, targets)  # 1010
        acc_l7, _ = accuracy(l7_out, targets)
        acc_l7 = acc_l7[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        l6_out = F.softmax(l6_out, dim=1)
        l7_out = F.softmax(l7_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        classifier_6_weight = F.normalize(classifier_6_weight, p=2, dim=1)
        classifier_7_weight = F.normalize(classifier_7_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_6_7 = torch.tensor(0).to(self.device)  # species and genus.
        cl7_weight_hat = torch.zeros_like(classifier_6_weight).to(self.device)
        for l6 in range(len(genus_to_species)):
            cl7_weight_hat[l6, :] = torch.sum(classifier_7_weight[genus_to_species[l6], :], dim=0)
        cl7_weight_hat = F.normalize(cl7_weight_hat, p=2, dim=1)
        l2_weight_6_7 = torch.mean(cosine_sim(classifier_6_weight, cl7_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_5_6 = torch.tensor(0).to(self.device)  # genus and family.
        cl6_weight_hat = torch.zeros_like(classifier_5_weight).to(self.device)
        for l5 in range(len(family_to_genus)):
            cl6_weight_hat[l5, :] = torch.sum(classifier_6_weight[family_to_genus[l5], :], dim=0)
        cl6_weight_hat = F.normalize(cl6_weight_hat, p=2, dim=1)
        l2_weight_5_6 = torch.mean(cosine_sim(classifier_5_weight, cl6_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # family and order.
        cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
        for l4 in range(len(order_to_family)):
            cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[order_to_family[l4], :], dim=0)
        cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
        l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # order and class.
        cl4_weight_hat = torch.zeros_like(classifier_3_weight).to(self.device)
        for l3 in range(len(class_to_order)):
            cl4_weight_hat[l3, :] = torch.sum(classifier_4_weight[class_to_order[l3], :], dim=0)
        cl4_weight_hat = F.normalize(cl4_weight_hat, p=2, dim=1)
        l2_weight_3_4 = torch.mean(cosine_sim(classifier_3_weight, cl4_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_2_3 = torch.tensor(0).to(self.device)  # class and phylum.
        cl3_weight_hat = torch.zeros_like(classifier_2_weight).to(self.device)
        for l2 in range(len(phylum_to_class)):
            cl3_weight_hat[l2, :] = torch.sum(classifier_3_weight[phylum_to_class[l2], :], dim=0)
        cl3_weight_hat = F.normalize(cl3_weight_hat, p=2, dim=1)
        l2_weight_2_3 = torch.mean(cosine_sim(classifier_2_weight, cl3_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_1_2 = torch.tensor(0).to(self.device)  # phylum and kingdom.
        cl2_weight_hat = torch.zeros_like(classifier_1_weight).to(self.device)
        for l1 in range(len(kingdom_to_phylum)):
            cl2_weight_hat[l1, :] = torch.sum(classifier_2_weight[kingdom_to_phylum[l1], :], dim=0)
        cl2_weight_hat = F.normalize(cl2_weight_hat, p=2, dim=1)
        l2_weight_1_2 = torch.mean(cosine_sim(classifier_1_weight, cl2_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_6_7 = torch.tensor(0).to(self.device)
        p_hat_l7 = torch.zeros_like(l6_out).to(self.device)
        for l6 in range(len(genus_to_species)):
            p_hat_l7[:, l6] = torch.sum(l7_out[:, genus_to_species[l6]], dim=1)
        jsd_level_6_7 = simloss(p_hat_l7, l6_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_5_6 = torch.tensor(0).to(self.device)
        p_hat_l6 = torch.zeros_like(l5_out).to(self.device)
        for l5 in range(len(family_to_genus)):
            p_hat_l6[:, l5] = torch.sum(l6_out[:, family_to_genus[l5]], dim=1)
        jsd_level_5_6 = simloss(p_hat_l6, l5_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
        for l4 in range(len(order_to_family)):
            p_hat_l5[:, l4] = torch.sum(l5_out[:, order_to_family[l4]], dim=1)
        jsd_level_4_5 = simloss(p_hat_l5, l4_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
        for l3 in range(len(class_to_order)):
            p_hat_l4[:, l3] = torch.sum(l4_out[:, class_to_order[l3]], dim=1)
        jsd_level_3_4 = simloss(p_hat_l4, l3_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
        for l2 in range(len(phylum_to_class)):
            p_hat_l3[:, l2] = torch.sum(l3_out[:, phylum_to_class[l2]], dim=1)
        jsd_level_2_3 = simloss(p_hat_l3, l2_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
        for l1 in range(len(kingdom_to_phylum)):
            p_hat_l2[:, l1] = torch.sum(l2_out[:, kingdom_to_phylum[l1]], dim=1)
        jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss = ce_loss_l7
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5 + l2_weight_5_6 + l2_weight_6_7
        self.ce_loss_list = [ce_loss_l7.item()]
        self.jsd_loss_list = [jsd_level_6_7.item(), jsd_level_5_6.item(),
                              jsd_level_4_5.item(), jsd_level_3_4.item(),
                              jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_6_7.item(), l2_weight_5_6.item(),
                               l2_weight_4_5.item(), l2_weight_3_4.item(),
                               l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]
        self.loss = ce_loss + jsd_loss - l2_weight_loss
        return l7_out


class ResNet18_ours_l7_cejsd_wtconst_dissim(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_ours_l7_cejsd_wtconst_dissim, self).__init__()
        self.device = gpu
        self.num_classes = num_classes
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, num_classes[6]),)                        # num of class 
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, num_classes[5]),)                        # num of order
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, num_classes[4]),)                        # num of family
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, num_classes[3]),)                        # num of genus
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, num_classes[2]),)                        # num of family
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, num_classes[1]),)                        # num of genus
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, num_classes[0]),)                        # num of species

    def forward(self, x, targets, is_inference):
        x = self.features_2(x)
        x = self.max(x)

        x = x.view(x.size(0), -1)
        x = self.features_1(x)  # N * 512   
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [kingdom_targets, phylum_targets, class_targets, \
         order_targets, family_targets, genus_targets] = get_target_l7(targets)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        classifier_6_weight = F.normalize(self.classifier_6[0].weight, p=2, dim=1)
        classifier_7_weight = F.normalize(self.classifier_7[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, kingdom_targets)
        acc_l1 = acc_l1[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, phylum_targets)
        acc_l2 = acc_l2[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, class_targets)
        acc_l3 = acc_l3[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, order_targets)
        acc_l4 = acc_l4[0].item()
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        acc_l5, _ = accuracy(l5_out, family_targets)
        acc_l5 = acc_l5[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = torch.mm(x, classifier_6_weight.t()) + self.classifier_6[0].bias
        acc_l6, _ = accuracy(l6_out, genus_targets)
        acc_l6 = acc_l6[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = torch.mm(x, classifier_7_weight.t()) + self.classifier_7[0].bias
        ce_loss_l7 = criterion(l7_out, targets)  # 1010
        acc_l7, _ = accuracy(l7_out, targets)
        acc_l7 = acc_l7[0].item()
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        l6_out = F.softmax(l6_out, dim=1)
        l7_out = F.softmax(l7_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        classifier_6_weight = F.normalize(classifier_6_weight, p=2, dim=1)
        classifier_7_weight = F.normalize(classifier_7_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------        
        dissim_loss_l6 = torch.tensor(0).to(self.device)
        dissim_loss_l5 = torch.tensor(0).to(self.device)
        dissim_loss_l4 = torch.tensor(0).to(self.device)
        dissim_loss_l3 = torch.tensor(0).to(self.device)
        dissim_loss_l2 = torch.tensor(0).to(self.device)
        dissim_loss_l1 = torch.tensor(0).to(self.device)

        l2_weight_6_7 = torch.tensor(0).to(self.device)  # species and genus.
        l2_weight_5_6 = torch.tensor(0).to(self.device)  # genus and family.
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # family and order.
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # order and class.
        l2_weight_2_3 = torch.tensor(0).to(self.device)  # class and phylum.
        l2_weight_1_2 = torch.tensor(0).to(self.device)  # phylum and kingdom.

        jsd_level_6_7 = torch.tensor(0).to(self.device)
        jsd_level_5_6 = torch.tensor(0).to(self.device)
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if is_inference == False:
            dissimilar_pairs_l6 = create_dissimilar_pairs(genus_targets, l6_out, self.device)
            dissim_loss_l6 = dissimiloss(dissimilar_pairs_l6[:, 0, :], dissimilar_pairs_l6[:, 1, :], self.device, margin=3.0)

            dissimilar_pairs_l5 = create_dissimilar_pairs(family_targets, l5_out, self.device)
            dissim_loss_l5 = dissimiloss(dissimilar_pairs_l5[:, 0, :], dissimilar_pairs_l5[:, 1, :], self.device, margin=3.0)

            dissimilar_pairs_l4 = create_dissimilar_pairs(order_targets, l4_out, self.device)
            dissim_loss_l4 = dissimiloss(dissimilar_pairs_l4[:, 0, :], dissimilar_pairs_l4[:, 1, :], self.device, margin=3.0)

            dissimilar_pairs_l3 = create_dissimilar_pairs(class_targets, l3_out, self.device)
            dissim_loss_l3 = dissimiloss(dissimilar_pairs_l3[:, 0, :], dissimilar_pairs_l3[:, 1, :], self.device, margin=3.0)

            dissimilar_pairs_l2 = create_dissimilar_pairs(phylum_targets, l2_out, self.device)
            dissim_loss_l2 = dissimiloss(dissimilar_pairs_l2[:, 0, :], dissimilar_pairs_l2[:, 1, :], self.device, margin=3.0)

            cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl7_weight_hat = torch.zeros_like(classifier_6_weight).to(self.device)
            for l6 in range(len(genus_to_species)):
                cl7_weight_hat[l6, :] = torch.sum(classifier_7_weight[genus_to_species[l6], :], dim=0)
            cl7_weight_hat = F.normalize(cl7_weight_hat, p=2, dim=1)
            l2_weight_6_7 = torch.mean(cosine_sim(classifier_6_weight, cl7_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl6_weight_hat = torch.zeros_like(classifier_5_weight).to(self.device)
            for l5 in range(len(family_to_genus)):
                cl6_weight_hat[l5, :] = torch.sum(classifier_6_weight[family_to_genus[l5], :], dim=0)
            cl6_weight_hat = F.normalize(cl6_weight_hat, p=2, dim=1)
            l2_weight_5_6 = torch.mean(cosine_sim(classifier_5_weight, cl6_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
            for l4 in range(len(order_to_family)):
                cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[order_to_family[l4], :], dim=0)
            cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
            l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl4_weight_hat = torch.zeros_like(classifier_3_weight).to(self.device)
            for l3 in range(len(class_to_order)):
                cl4_weight_hat[l3, :] = torch.sum(classifier_4_weight[class_to_order[l3], :], dim=0)
            cl4_weight_hat = F.normalize(cl4_weight_hat, p=2, dim=1)
            l2_weight_3_4 = torch.mean(cosine_sim(classifier_3_weight, cl4_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl3_weight_hat = torch.zeros_like(classifier_2_weight).to(self.device)
            for l2 in range(len(phylum_to_class)):
                cl3_weight_hat[l2, :] = torch.sum(classifier_3_weight[phylum_to_class[l2], :], dim=0)
            cl3_weight_hat = F.normalize(cl3_weight_hat, p=2, dim=1)
            l2_weight_2_3 = torch.mean(cosine_sim(classifier_2_weight, cl3_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl2_weight_hat = torch.zeros_like(classifier_1_weight).to(self.device)
            for l1 in range(len(kingdom_to_phylum)):
                cl2_weight_hat[l1, :] = torch.sum(classifier_2_weight[kingdom_to_phylum[l1], :], dim=0)
            cl2_weight_hat = F.normalize(cl2_weight_hat, p=2, dim=1)
            l2_weight_1_2 = torch.mean(cosine_sim(classifier_1_weight, cl2_weight_hat))
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l7 = torch.zeros_like(l6_out).to(self.device)
            for l6 in range(len(genus_to_species)):
                p_hat_l7[:, l6] = torch.sum(l7_out[:, genus_to_species[l6]], dim=1)
            jsd_level_6_7 = simloss(p_hat_l7, l6_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l6 = torch.zeros_like(l5_out).to(self.device)
            for l5 in range(len(family_to_genus)):
                p_hat_l6[:, l5] = torch.sum(l6_out[:, family_to_genus[l5]], dim=1)
            jsd_level_5_6 = simloss(p_hat_l6, l5_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
            for l4 in range(len(order_to_family)):
                p_hat_l5[:, l4] = torch.sum(l5_out[:, order_to_family[l4]], dim=1)
            jsd_level_4_5 = simloss(p_hat_l5, l4_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
            for l3 in range(len(class_to_order)):
                p_hat_l4[:, l3] = torch.sum(l4_out[:, class_to_order[l3]], dim=1)
            jsd_level_3_4 = simloss(p_hat_l4, l3_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
            for l2 in range(len(phylum_to_class)):
                p_hat_l3[:, l2] = torch.sum(l3_out[:, phylum_to_class[l2]], dim=1)
            jsd_level_2_3 = simloss(p_hat_l3, l2_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
            for l1 in range(len(kingdom_to_phylum)):
                p_hat_l2[:, l1] = torch.sum(l2_out[:, kingdom_to_phylum[l1]], dim=1)
            jsd_level_1_2 = simloss(p_hat_l2, l1_out)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss = ce_loss_l7  # + ce_loss_l11 + ce_loss_l10 + ce_loss_l9 + ce_loss_l8 + ce_loss_l7 + ce_loss_l6 + ce_loss_l5 + ce_loss_l4 + ce_loss_l3 + ce_loss_l2 + ce_loss_l1
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5 + l2_weight_5_6 + l2_weight_6_7
        dissim_loss = dissim_loss_l6 + dissim_loss_l5 + dissim_loss_l4 + dissim_loss_l3 + dissim_loss_l2 + dissim_loss_l1
        self.dissimiloss_list = [dissim_loss_l6.item(), dissim_loss_l5.item(), dissim_loss_l4.item(), dissim_loss_l3.item(), 
                                dissim_loss_l2.item(), dissim_loss_l1.item()]
        self.ce_loss_list = [ce_loss_l7.item()]
        self.jsd_loss_list = [jsd_level_6_7.item(), jsd_level_5_6.item(),
                              jsd_level_4_5.item(), jsd_level_3_4.item(),
                              jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_6_7.item(), l2_weight_5_6.item(),
                               l2_weight_4_5.item(), l2_weight_3_4.item(),
                               l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]
        self.loss = ce_loss + jsd_loss - l2_weight_loss + dissim_loss
        return l7_out

class ResNet18_ours_l12_cejsd(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_ours_l12_cejsd, self).__init__()
        self.num_classes = num_classes
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[11]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[10]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[9]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[8]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[7]),)
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[6]),)
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[5]),)
        self.classifier_8 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[4]),)
        self.classifier_9 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[3]),)
        self.classifier_10 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[2]),)
        self.classifier_11 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[1]),)
        self.classifier_12 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[0]),)                         # finest level

        # self.layer_outputs = {}
        self.device = gpu

    def forward(self, x, targets, is_inference):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [l1_targets, l2_targets, l3_targets, l4_targets, l5_targets, l6_targets, l7_targets, \
        l8_targets, l9_targets, l10_targets, l11_targets] = get_target_l12(targets)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
        acc_l5, _ = accuracy(l5_out, l5_targets)
        acc_l5 = acc_l5[0].item()
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = self.classifier_6(x)
        acc_l6, _ = accuracy(l6_out, l6_targets)
        acc_l6 = acc_l6[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = self.classifier_7(x)
        acc_l7, _ = accuracy(l7_out, l7_targets)
        acc_l7 = acc_l7[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l8_out = self.classifier_8(x)
        acc_l8, _ = accuracy(l8_out, l8_targets)
        acc_l8 = acc_l8[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l9_out = self.classifier_9(x)
        acc_l9, _ = accuracy(l9_out, l9_targets)
        acc_l9 = acc_l9[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l10_out = self.classifier_10(x)
        acc_l10, _ = accuracy(l10_out, l10_targets)
        acc_l10 = acc_l10[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l11_out = self.classifier_11(x)
        acc_l11, _ = accuracy(l11_out, l11_targets)
        acc_l11 = acc_l11[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l12_out = self.classifier_12(x)
        ce_loss_l12 = criterion(l12_out, targets)           # 608
        acc_l12, _ = accuracy(l12_out, targets)
        acc_l12 = acc_l12[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        l6_out = F.softmax(l6_out, dim=1)
        l7_out = F.softmax(l7_out, dim=1)
        l8_out = F.softmax(l8_out, dim=1)
        l9_out = F.softmax(l9_out, dim=1)
        l10_out = F.softmax(l10_out, dim=1)
        l11_out = F.softmax(l11_out, dim=1)
        l12_out = F.softmax(l12_out, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_11_12 = torch.tensor(0).to(self.device)  # l11 and l12
        jsd_level_10_11 = torch.tensor(0).to(self.device)  # l10 and l11
        jsd_level_9_10 = torch.tensor(0).to(self.device) # l9 and l10
        jsd_level_8_9 = torch.tensor(0).to(self.device)
        jsd_level_7_8 = torch.tensor(0).to(self.device)
        jsd_level_6_7 = torch.tensor(0).to(self.device)
        jsd_level_5_6 = torch.tensor(0).to(self.device)
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if is_inference == False:
            p_hat_l12 = torch.zeros_like(l11_out).to(self.device)
            for l11 in range(len(l11_to_l12)):
                p_hat_l12[:, l11] = torch.sum(l12_out[:, l11_to_l12[l11]], dim=1)
            jsd_level_11_12 = simloss(p_hat_l12, l11_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l11 = torch.zeros_like(l10_out).to(self.device)
            for l10 in range(len(l10_to_l11)):
                p_hat_l11[:, l10] = torch.sum(l11_out[:, l10_to_l11[l10]], dim=1)
            jsd_level_10_11 = simloss(p_hat_l11, l10_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l10 = torch.zeros_like(l9_out).to(self.device)
            for l9 in range(len(l9_to_l10)):
                p_hat_l10[:, l9] = torch.sum(l10_out[:, l9_to_l10[l9]], dim=1)
            jsd_level_9_10 = simloss(p_hat_l10, l9_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l9 = torch.zeros_like(l8_out).to(self.device)
            for l8 in range(len(l8_to_l9)):
                p_hat_l9[:, l8] = torch.sum(l9_out[:, l8_to_l9[l8]], dim=1)
            jsd_level_8_9 = simloss(p_hat_l9, l8_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l8 = torch.zeros_like(l7_out).to(self.device)
            for l7 in range(len(l7_to_l8)):
                p_hat_l8[:, l7] = torch.sum(l8_out[:, l7_to_l8[l7]], dim=1)
            jsd_level_7_8 = simloss(p_hat_l8, l7_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l7 = torch.zeros_like(l6_out).to(self.device)
            for l6 in range(len(l6_to_l7)):
                p_hat_l7[:, l6] = torch.sum(l7_out[:, l6_to_l7[l6]], dim=1)
            jsd_level_6_7 = simloss(p_hat_l7, l6_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l6 = torch.zeros_like(l5_out).to(self.device)
            for l5 in range(len(l5_to_l6)):
                p_hat_l6[:, l5] = torch.sum(l6_out[:, l5_to_l6[l5]], dim=1)
            jsd_level_5_6 = simloss(p_hat_l6, l5_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
            for l4 in range(len(l4_to_l5)):
                p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
            jsd_level_4_5 = simloss(p_hat_l5, l4_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
            for l3 in range(len(l3_to_l4)):
                p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
            jsd_level_3_4 = simloss(p_hat_l4, l3_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
            for l2 in range(len(l2_to_l3)):
                p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
            jsd_level_2_3 = simloss(p_hat_l3, l2_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
            for l1 in range(len(l1_to_l2)):
                p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
            jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_11_12 + jsd_level_10_11 + jsd_level_9_10 + jsd_level_8_9 + jsd_level_7_8 + jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss =  ce_loss_l12 #+ ce_loss_l11 + ce_loss_l10 + ce_loss_l9 + ce_loss_l8 + ce_loss_l7 + ce_loss_l6 + ce_loss_l5 + ce_loss_l4 + ce_loss_l3 + ce_loss_l2 + ce_loss_l1
        self.ce_loss_list = [ce_loss_l12.item()]
        self.jsd_loss_list = [jsd_level_11_12.item(), jsd_level_10_11.item(), jsd_level_9_10.item(), jsd_level_8_9.item(),
                 jsd_level_7_8.item(), jsd_level_6_7.item(), jsd_level_5_6.item(), jsd_level_4_5.item(), jsd_level_3_4.item(),
                  jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.acc_list = [acc_l12, acc_l11, acc_l10, acc_l9, acc_l8, acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]          
        self.loss = ce_loss + jsd_loss
        return l12_out


class ResNet18_ours_l12_cejsd_wtconst(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_ours_l12_cejsd_wtconst, self).__init__()
        self.num_classes = num_classes
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[11]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[10]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[9]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[8]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[7]),)
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[6]),)
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[5]),)
        self.classifier_8 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[4]),)
        self.classifier_9 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[3]),)
        self.classifier_10 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[2]),)
        self.classifier_11 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[1]),)
        self.classifier_12 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[0]),)                         # finest level

        # self.layer_outputs = {}
        self.device = gpu

    def forward(self, x, targets):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [l1_targets, l2_targets, l3_targets, l4_targets, l5_targets, l6_targets, l7_targets, \
        l8_targets, l9_targets, l10_targets, l11_targets] = get_target_l12(targets)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        classifier_6_weight = F.normalize(self.classifier_6[0].weight, p=2, dim=1)
        classifier_7_weight = F.normalize(self.classifier_7[0].weight, p=2, dim=1)
        classifier_8_weight = F.normalize(self.classifier_8[0].weight, p=2, dim=1)
        classifier_9_weight = F.normalize(self.classifier_9[0].weight, p=2, dim=1)
        classifier_10_weight = F.normalize(self.classifier_10[0].weight, p=2, dim=1)
        classifier_11_weight = F.normalize(self.classifier_11[0].weight, p=2, dim=1)
        classifier_12_weight = F.normalize(self.classifier_12[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        acc_l5, _ = accuracy(l5_out, l5_targets)
        acc_l5 = acc_l5[0].item()
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = torch.mm(x, classifier_6_weight.t()) + self.classifier_6[0].bias
        acc_l6, _ = accuracy(l6_out, l6_targets)
        acc_l6 = acc_l6[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = torch.mm(x, classifier_7_weight.t()) + self.classifier_7[0].bias
        acc_l7, _ = accuracy(l7_out, l7_targets)
        acc_l7 = acc_l7[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l8_out = torch.mm(x, classifier_8_weight.t()) + self.classifier_8[0].bias
        acc_l8, _ = accuracy(l8_out, l8_targets)
        acc_l8 = acc_l8[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l9_out = torch.mm(x, classifier_9_weight.t()) + self.classifier_9[0].bias
        acc_l9, _ = accuracy(l9_out, l9_targets)
        acc_l9 = acc_l9[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l10_out = torch.mm(x, classifier_10_weight.t()) + self.classifier_10[0].bias
        acc_l10, _ = accuracy(l10_out, l10_targets)
        acc_l10 = acc_l10[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l11_out = torch.mm(x, classifier_11_weight.t()) + self.classifier_11[0].bias
        acc_l11, _ = accuracy(l11_out, l11_targets)
        acc_l11 = acc_l11[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l12_out = torch.mm(x, classifier_12_weight.t()) + self.classifier_12[0].bias
        ce_loss_l12 = criterion(l12_out, targets)           # 608
        acc_l12, _ = accuracy(l12_out, targets)
        acc_l12 = acc_l12[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        l6_out = F.softmax(l6_out, dim=1)
        l7_out = F.softmax(l7_out, dim=1)
        l8_out = F.softmax(l8_out, dim=1)
        l9_out = F.softmax(l9_out, dim=1)
        l10_out = F.softmax(l10_out, dim=1)
        l11_out = F.softmax(l11_out, dim=1)
        l12_out = F.softmax(l12_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        classifier_6_weight = F.normalize(classifier_6_weight, p=2, dim=1)
        classifier_7_weight = F.normalize(classifier_7_weight, p=2, dim=1)
        classifier_8_weight = F.normalize(classifier_8_weight, p=2, dim=1)
        classifier_9_weight = F.normalize(classifier_9_weight, p=2, dim=1)
        classifier_10_weight = F.normalize(classifier_10_weight, p=2, dim=1)
        classifier_11_weight = F.normalize(classifier_11_weight, p=2, dim=1)
        classifier_12_weight = F.normalize(classifier_12_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_11_12 = torch.tensor(0).to(self.device)  # 11 and 12.
        cl12_weight_hat = torch.zeros_like(classifier_11_weight).to(self.device)
        for l11 in range(len(l11_to_l12)):
            cl12_weight_hat[l11, :] = torch.sum(classifier_12_weight[l11_to_l12[l11], :], dim=0)
        cl12_weight_hat = F.normalize(cl12_weight_hat, p=2, dim=1)
        l2_weight_11_12 = torch.mean(cosine_sim(classifier_11_weight, cl12_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        l2_weight_10_11 = torch.tensor(0).to(self.device)  # 10 and 11.
        cl11_weight_hat = torch.zeros_like(classifier_10_weight).to(self.device)
        for l10 in range(len(l10_to_l11)):
            cl11_weight_hat[l10, :] = torch.sum(classifier_11_weight[l10_to_l11[l10], :], dim=0)
        cl11_weight_hat = F.normalize(cl11_weight_hat, p=2, dim=1)
        l2_weight_10_11 = torch.mean(cosine_sim(classifier_10_weight, cl11_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_9_10 = torch.tensor(0).to(self.device)  # 9 and 10.
        cl10_weight_hat = torch.zeros_like(classifier_9_weight).to(self.device)
        for l9 in range(len(l9_to_l10)):
            cl10_weight_hat[l9, :] = torch.sum(classifier_10_weight[l9_to_l10[l9], :], dim=0)
        cl10_weight_hat = F.normalize(cl10_weight_hat, p=2, dim=1)
        l2_weight_9_10 = torch.mean(cosine_sim(classifier_9_weight, cl10_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_8_9 = torch.tensor(0).to(self.device)  # 8 and 9.
        cl9_weight_hat = torch.zeros_like(classifier_8_weight).to(self.device)
        for l8 in range(len(l8_to_l9)):
            cl9_weight_hat[l8, :] = torch.sum(classifier_9_weight[l8_to_l9[l8], :], dim=0)
        cl9_weight_hat = F.normalize(cl9_weight_hat, p=2, dim=1)
        l2_weight_8_9 = torch.mean(cosine_sim(classifier_8_weight, cl9_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_7_8 = torch.tensor(0).to(self.device)  # 7 and 8.
        cl8_weight_hat = torch.zeros_like(classifier_7_weight).to(self.device)
        for l7 in range(len(l7_to_l8)):
            cl8_weight_hat[l7, :] = torch.sum(classifier_8_weight[l7_to_l8[l7], :], dim=0)
        cl8_weight_hat = F.normalize(cl8_weight_hat, p=2, dim=1)
        l2_weight_7_8 = torch.mean(cosine_sim(classifier_7_weight, cl8_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_6_7 = torch.tensor(0).to(self.device)  # species and genus.
        cl7_weight_hat = torch.zeros_like(classifier_6_weight).to(self.device)
        for l6 in range(len(l6_to_l7)):
            cl7_weight_hat[l6, :] = torch.sum(classifier_7_weight[l6_to_l7[l6], :], dim=0)
        cl7_weight_hat = F.normalize(cl7_weight_hat, p=2, dim=1)
        l2_weight_6_7 = torch.mean(cosine_sim(classifier_6_weight, cl7_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_5_6 = torch.tensor(0).to(self.device)  # genus and family.
        cl6_weight_hat = torch.zeros_like(classifier_5_weight).to(self.device)
        for l5 in range(len(l5_to_l6)):
            cl6_weight_hat[l5, :] = torch.sum(classifier_6_weight[l5_to_l6[l5], :], dim=0)
        cl6_weight_hat = F.normalize(cl6_weight_hat, p=2, dim=1)
        l2_weight_5_6 = torch.mean(cosine_sim(classifier_5_weight, cl6_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # family and order.
        cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
        for l4 in range(len(l4_to_l5)):
            cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[l4_to_l5[l4], :], dim=0)
        cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
        l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # order and class.
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
        jsd_level_11_12 = torch.tensor(0).to(self.device)  # l11 and l12
        p_hat_l12 = torch.zeros_like(l11_out).to(self.device)
        for l11 in range(len(l11_to_l12)):
            p_hat_l12[:, l11] = torch.sum(l12_out[:, l11_to_l12[l11]], dim=1)
        jsd_level_11_12 = simloss(p_hat_l12, l11_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_10_11 = torch.tensor(0).to(self.device)  # l10 and l11
        p_hat_l11 = torch.zeros_like(l10_out).to(self.device)
        for l10 in range(len(l10_to_l11)):
            p_hat_l11[:, l10] = torch.sum(l11_out[:, l10_to_l11[l10]], dim=1)
        jsd_level_10_11 = simloss(p_hat_l11, l10_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_9_10 = torch.tensor(0).to(self.device)  # l9 and l10
        p_hat_l10 = torch.zeros_like(l9_out).to(self.device)
        for l9 in range(len(l9_to_l10)):
            p_hat_l10[:, l9] = torch.sum(l10_out[:, l9_to_l10[l9]], dim=1)
        jsd_level_9_10 = simloss(p_hat_l10, l9_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_8_9 = torch.tensor(0).to(self.device)
        p_hat_l9 = torch.zeros_like(l8_out).to(self.device)
        for l8 in range(len(l8_to_l9)):
            p_hat_l9[:, l8] = torch.sum(l9_out[:, l8_to_l9[l8]], dim=1)
        jsd_level_8_9 = simloss(p_hat_l9, l8_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_7_8 = torch.tensor(0).to(self.device)
        p_hat_l8 = torch.zeros_like(l7_out).to(self.device)
        for l7 in range(len(l7_to_l8)):
            p_hat_l8[:, l7] = torch.sum(l8_out[:, l7_to_l8[l7]], dim=1)
        jsd_level_7_8 = simloss(p_hat_l8, l7_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_6_7 = torch.tensor(0).to(self.device)
        p_hat_l7 = torch.zeros_like(l6_out).to(self.device)
        for l6 in range(len(l6_to_l7)):
            p_hat_l7[:, l6] = torch.sum(l7_out[:, l6_to_l7[l6]], dim=1)
        jsd_level_6_7 = simloss(p_hat_l7, l6_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_5_6 = torch.tensor(0).to(self.device)
        p_hat_l6 = torch.zeros_like(l5_out).to(self.device)
        for l5 in range(len(l5_to_l6)):
            p_hat_l6[:, l5] = torch.sum(l6_out[:, l5_to_l6[l5]], dim=1)
        jsd_level_5_6 = simloss(p_hat_l6, l5_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
        for l4 in range(len(l4_to_l5)):
            p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
        jsd_level_4_5 = simloss(p_hat_l5, l4_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
        for l3 in range(len(l3_to_l4)):
            p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
        jsd_level_3_4 = simloss(p_hat_l4, l3_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
        for l2 in range(len(l2_to_l3)):
            p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
        jsd_level_2_3 = simloss(p_hat_l3, l2_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_level_1_2 = torch.tensor(0).to(self.device)
        p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
        for l1 in range(len(l1_to_l2)):
            p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
        jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_11_12 + jsd_level_10_11 + jsd_level_9_10 + jsd_level_8_9 + jsd_level_7_8 + jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss =  ce_loss_l12 #+ ce_loss_l11 + ce_loss_l10 + ce_loss_l9 + ce_loss_l8 + ce_loss_l7 + ce_loss_l6 + ce_loss_l5 + ce_loss_l4 + ce_loss_l3 + ce_loss_l2 + ce_loss_l1
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5 + l2_weight_5_6 + l2_weight_6_7 + l2_weight_7_8 + l2_weight_8_9 + l2_weight_9_10 + l2_weight_10_11 + l2_weight_11_12
        self.ce_loss_list = [ce_loss_l12.item()]
        self.jsd_loss_list = [jsd_level_11_12.item(), jsd_level_10_11.item(), jsd_level_9_10.item(), jsd_level_8_9.item(),
                 jsd_level_7_8.item(), jsd_level_6_7.item(), jsd_level_5_6.item(), jsd_level_4_5.item(), jsd_level_3_4.item(),
                  jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_11_12.item(), l2_weight_10_11.item(), l2_weight_9_10.item(), l2_weight_8_9.item(), l2_weight_7_8.item(), l2_weight_6_7.item(),
                          l2_weight_5_6.item(), l2_weight_4_5.item(), l2_weight_3_4.item(), l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l12, acc_l11, acc_l10, acc_l9, acc_l8, acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]
        # self.acc_list = [acc_l12.item(), acc_l11.item(), acc_l10.item(), acc_l9.item(), acc_l8.item(), acc_l7.item(), acc_l6.item(),
                # acc_l5.item(), acc_l4.item(), acc_l3.item(), acc_l2.item(), acc_l1.item()]
        self.loss = ce_loss + jsd_loss - l2_weight_loss
        return l12_out


class ResNet18_ours_l12_cejsd_wtconst_dissim(nn.Module):
    def __init__(self, model, feature_size, num_classes, gpu):
        super(ResNet18_ours_l12_cejsd_wtconst_dissim, self).__init__()
        self.num_classes = num_classes
        self.features_2 = nn.Sequential(*list(model.children())[:-2])
        self.max = nn.MaxPool2d(kernel_size=7, stride=7)
        self.num_ftrs = 512 * 1 * 1  # Used for resnet18
        # self.num_ftrs = 2048 * 1 * 1                  # Used for resnet50
        self.features_1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
        )
        self.classifier_1 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[11]),)
        self.classifier_2 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[10]),)
        self.classifier_3 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[9]),)
        self.classifier_4 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[8]),)
        self.classifier_5 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[7]),)
        self.classifier_6 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[6]),)
        self.classifier_7 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[5]),)
        self.classifier_8 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[4]),)
        self.classifier_9 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[3]),)
        self.classifier_10 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[2]),)
        self.classifier_11 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[1]),)
        self.classifier_12 = nn.Sequential(
            nn.Linear(feature_size, self.num_classes[0]),)                         # finest level

        # self.layer_outputs = {}
        self.device = gpu

    def forward(self, x, targets, is_inference):
        x = self.features_2(x)
        x = self.max(x)
        x = x.view(x.size(0), -1)
        x = self.features_1(x) # N * 512
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        [l1_targets, l2_targets, l3_targets, l4_targets, l5_targets, l6_targets, l7_targets, \
        l8_targets, l9_targets, l10_targets, l11_targets] = get_target_l12(targets)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        classifier_1_weight = F.normalize(self.classifier_1[0].weight, p=2, dim=1)
        classifier_2_weight = F.normalize(self.classifier_2[0].weight, p=2, dim=1)
        classifier_3_weight = F.normalize(self.classifier_3[0].weight, p=2, dim=1)
        classifier_4_weight = F.normalize(self.classifier_4[0].weight, p=2, dim=1)
        classifier_5_weight = F.normalize(self.classifier_5[0].weight, p=2, dim=1)
        classifier_6_weight = F.normalize(self.classifier_6[0].weight, p=2, dim=1)
        classifier_7_weight = F.normalize(self.classifier_7[0].weight, p=2, dim=1)
        classifier_8_weight = F.normalize(self.classifier_8[0].weight, p=2, dim=1)
        classifier_9_weight = F.normalize(self.classifier_9[0].weight, p=2, dim=1)
        classifier_10_weight = F.normalize(self.classifier_10[0].weight, p=2, dim=1)
        classifier_11_weight = F.normalize(self.classifier_11[0].weight, p=2, dim=1)
        classifier_12_weight = F.normalize(self.classifier_12[0].weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l1_out = torch.mm(x, classifier_1_weight.t()) + self.classifier_1[0].bias
        acc_l1, _ = accuracy(l1_out, l1_targets)
        acc_l1 = acc_l1[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l2_out = torch.mm(x, classifier_2_weight.t()) + self.classifier_2[0].bias
        acc_l2, _ = accuracy(l2_out, l2_targets)
        acc_l2 = acc_l2[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l3_out = torch.mm(x, classifier_3_weight.t()) + self.classifier_3[0].bias
        acc_l3, _ = accuracy(l3_out, l3_targets)
        acc_l3 = acc_l3[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l4_out = torch.mm(x, classifier_4_weight.t()) + self.classifier_4[0].bias
        acc_l4, _ = accuracy(l4_out, l4_targets)
        acc_l4 = acc_l4[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l5_out = torch.mm(x, classifier_5_weight.t()) + self.classifier_5[0].bias
        acc_l5, _ = accuracy(l5_out, l5_targets)
        acc_l5 = acc_l5[0].item()
        #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l6_out = torch.mm(x, classifier_6_weight.t()) + self.classifier_6[0].bias
        acc_l6, _ = accuracy(l6_out, l6_targets)
        acc_l6 = acc_l6[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l7_out = torch.mm(x, classifier_7_weight.t()) + self.classifier_7[0].bias
        acc_l7, _ = accuracy(l7_out, l7_targets)
        acc_l7 = acc_l7[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l8_out = torch.mm(x, classifier_8_weight.t()) + self.classifier_8[0].bias
        acc_l8, _ = accuracy(l8_out, l8_targets)
        acc_l8 = acc_l8[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l9_out = torch.mm(x, classifier_9_weight.t()) + self.classifier_9[0].bias
        acc_l9, _ = accuracy(l9_out, l9_targets)
        acc_l9 = acc_l9[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l10_out = torch.mm(x, classifier_10_weight.t()) + self.classifier_10[0].bias
        acc_l10, _ = accuracy(l10_out, l10_targets)
        acc_l10 = acc_l10[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l11_out = torch.mm(x, classifier_11_weight.t()) + self.classifier_11[0].bias
        acc_l11, _ = accuracy(l11_out, l11_targets)
        acc_l11 = acc_l11[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        l12_out = torch.mm(x, classifier_12_weight.t()) + self.classifier_12[0].bias
        ce_loss_l12 = criterion(l12_out, targets)           # 608
        acc_l12, _ = accuracy(l12_out, targets)
        acc_l12 = acc_l12[0].item()
        #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Softmax Normalized
        l1_out = F.softmax(l1_out, dim=1)
        l2_out = F.softmax(l2_out, dim=1)
        l3_out = F.softmax(l3_out, dim=1)
        l4_out = F.softmax(l4_out, dim=1)
        l5_out = F.softmax(l5_out, dim=1)
        l6_out = F.softmax(l6_out, dim=1)
        l7_out = F.softmax(l7_out, dim=1)
        l8_out = F.softmax(l8_out, dim=1)
        l9_out = F.softmax(l9_out, dim=1)
        l10_out = F.softmax(l10_out, dim=1)
        l11_out = F.softmax(l11_out, dim=1)
        l12_out = F.softmax(l12_out, dim=1)
        classifier_1_weight = F.normalize(classifier_1_weight, p=2, dim=1)
        classifier_2_weight = F.normalize(classifier_2_weight, p=2, dim=1)
        classifier_3_weight = F.normalize(classifier_3_weight, p=2, dim=1)
        classifier_4_weight = F.normalize(classifier_4_weight, p=2, dim=1)
        classifier_5_weight = F.normalize(classifier_5_weight, p=2, dim=1)
        classifier_6_weight = F.normalize(classifier_6_weight, p=2, dim=1)
        classifier_7_weight = F.normalize(classifier_7_weight, p=2, dim=1)
        classifier_8_weight = F.normalize(classifier_8_weight, p=2, dim=1)
        classifier_9_weight = F.normalize(classifier_9_weight, p=2, dim=1)
        classifier_10_weight = F.normalize(classifier_10_weight, p=2, dim=1)
        classifier_11_weight = F.normalize(classifier_11_weight, p=2, dim=1)
        classifier_12_weight = F.normalize(classifier_12_weight, p=2, dim=1)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        dissim_loss_l11 = torch.tensor(0).to(self.device)
        dissim_loss_l10 = torch.tensor(0).to(self.device)
        dissim_loss_l9 = torch.tensor(0).to(self.device)
        dissim_loss_l8 = torch.tensor(0).to(self.device)
        dissim_loss_l7 = torch.tensor(0).to(self.device)
        dissim_loss_l6 = torch.tensor(0).to(self.device)
        dissim_loss_l5 = torch.tensor(0).to(self.device)
        dissim_loss_l4 = torch.tensor(0).to(self.device)
        dissim_loss_l3 = torch.tensor(0).to(self.device)
        dissim_loss_l2 = torch.tensor(0).to(self.device)
        dissim_loss_l1 = torch.tensor(0).to(self.device)

        jsd_level_11_12 = torch.tensor(0).to(self.device)  # l11 and l12
        jsd_level_10_11 = torch.tensor(0).to(self.device)  # l10 and l11
        jsd_level_9_10 = torch.tensor(0).to(self.device)  # l9 and l10
        jsd_level_8_9 = torch.tensor(0).to(self.device)
        jsd_level_7_8 = torch.tensor(0).to(self.device)
        jsd_level_6_7 = torch.tensor(0).to(self.device)
        jsd_level_5_6 = torch.tensor(0).to(self.device)
        jsd_level_4_5 = torch.tensor(0).to(self.device)
        jsd_level_3_4 = torch.tensor(0).to(self.device)
        jsd_level_2_3 = torch.tensor(0).to(self.device)
        jsd_level_1_2 = torch.tensor(0).to(self.device)

        l2_weight_11_12 = torch.tensor(0).to(self.device)  # 11 and 12.
        l2_weight_10_11 = torch.tensor(0).to(self.device)  # 10 and 11.
        l2_weight_9_10 = torch.tensor(0).to(self.device)  # 9 and 10.
        l2_weight_8_9 = torch.tensor(0).to(self.device)  # 8 and 9.
        l2_weight_7_8 = torch.tensor(0).to(self.device)  # 7 and 8.
        l2_weight_6_7 = torch.tensor(0).to(self.device)  # species and genus.
        l2_weight_5_6 = torch.tensor(0).to(self.device)  # genus and family.
        l2_weight_4_5 = torch.tensor(0).to(self.device)  # family and order.
        l2_weight_3_4 = torch.tensor(0).to(self.device)  # order and class.
        l2_weight_2_3 = torch.tensor(0).to(self.device)  # class and phylum.          
        l2_weight_1_2 = torch.tensor(0).to(self.device)  # phylum and kingdom
        if is_inference == False:
            # added in sec expt.
            dissimilar_pairs_l11 = create_dissimilar_pairs(l11_targets, l11_out, self.device)
            dissim_loss_l11 = dissimiloss(dissimilar_pairs_l11[:, 0, :], dissimilar_pairs_l11[:, 1, :], self.device, margin=3.0)
            
            dissimilar_pairs_l10 = create_dissimilar_pairs(l10_targets, l10_out, self.device)
            dissim_loss_l10 = dissimiloss(dissimilar_pairs_l10[:, 0, :], dissimilar_pairs_l10[:, 1, :], self.device, margin=3.0)
        
            dissimilar_pairs_l9 = create_dissimilar_pairs(l9_targets, l9_out, self.device)
            dissim_loss_l9 = dissimiloss(dissimilar_pairs_l9[:, 0, :], dissimilar_pairs_l9[:, 1, :], self.device, margin=3.0)
        
            dissimilar_pairs_l8 = create_dissimilar_pairs(l8_targets, l8_out, self.device)
            dissim_loss_l8 = dissimiloss(dissimilar_pairs_l8[:, 0, :], dissimilar_pairs_l8[:, 1, :], self.device, margin=3.0)
        
            dissimilar_pairs_l7 = create_dissimilar_pairs(l7_targets, l7_out, self.device)
            dissim_loss_l7 = dissimiloss(dissimilar_pairs_l7[:, 0, :], dissimilar_pairs_l7[:, 1, :], self.device, margin=3.0)
        
            dissimilar_pairs_l6 = create_dissimilar_pairs(l6_targets, l6_out, self.device)
            dissim_loss_l6 = dissimiloss(dissimilar_pairs_l6[:, 0, :], dissimilar_pairs_l6[:, 1, :], self.device, margin=3.0)
        
            dissimilar_pairs_l5 = create_dissimilar_pairs(l5_targets, l5_out, self.device)
            dissim_loss_l5 = dissimiloss(dissimilar_pairs_l5[:, 0, :], dissimilar_pairs_l5[:, 1, :], self.device, margin=3.0)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-8)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl12_weight_hat = torch.zeros_like(classifier_11_weight).to(self.device)
            for l11 in range(len(l11_to_l12)):
                cl12_weight_hat[l11, :] = torch.sum(classifier_12_weight[l11_to_l12[l11], :], dim=0)
            cl12_weight_hat = F.normalize(cl12_weight_hat, p=2, dim=1)
            l2_weight_11_12 = torch.mean(cosine_sim(classifier_11_weight, cl12_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl11_weight_hat = torch.zeros_like(classifier_10_weight).to(self.device)
            for l10 in range(len(l10_to_l11)):
                cl11_weight_hat[l10, :] = torch.sum(classifier_11_weight[l10_to_l11[l10], :], dim=0)
            cl11_weight_hat = F.normalize(cl11_weight_hat, p=2, dim=1)
            l2_weight_10_11 = torch.mean(cosine_sim(classifier_10_weight, cl11_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl10_weight_hat = torch.zeros_like(classifier_9_weight).to(self.device)
            for l9 in range(len(l9_to_l10)):
                cl10_weight_hat[l9, :] = torch.sum(classifier_10_weight[l9_to_l10[l9], :], dim=0)
            cl10_weight_hat = F.normalize(cl10_weight_hat, p=2, dim=1)
            l2_weight_9_10 = torch.mean(cosine_sim(classifier_9_weight, cl10_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl9_weight_hat = torch.zeros_like(classifier_8_weight).to(self.device)
            for l8 in range(len(l8_to_l9)):
                cl9_weight_hat[l8, :] = torch.sum(classifier_9_weight[l8_to_l9[l8], :], dim=0)
            cl9_weight_hat = F.normalize(cl9_weight_hat, p=2, dim=1)
            l2_weight_8_9 = torch.mean(cosine_sim(classifier_8_weight, cl9_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl8_weight_hat = torch.zeros_like(classifier_7_weight).to(self.device)
            for l7 in range(len(l7_to_l8)):
                cl8_weight_hat[l7, :] = torch.sum(classifier_8_weight[l7_to_l8[l7], :], dim=0)
            cl8_weight_hat = F.normalize(cl8_weight_hat, p=2, dim=1)
            l2_weight_7_8 = torch.mean(cosine_sim(classifier_7_weight, cl8_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl7_weight_hat = torch.zeros_like(classifier_6_weight).to(self.device)
            for l6 in range(len(l6_to_l7)):
                cl7_weight_hat[l6, :] = torch.sum(classifier_7_weight[l6_to_l7[l6], :], dim=0)
            cl7_weight_hat = F.normalize(cl7_weight_hat, p=2, dim=1)
            l2_weight_6_7 = torch.mean(cosine_sim(classifier_6_weight, cl7_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl6_weight_hat = torch.zeros_like(classifier_5_weight).to(self.device)
            for l5 in range(len(l5_to_l6)):
                cl6_weight_hat[l5, :] = torch.sum(classifier_6_weight[l5_to_l6[l5], :], dim=0)
            cl6_weight_hat = F.normalize(cl6_weight_hat, p=2, dim=1)
            l2_weight_5_6 = torch.mean(cosine_sim(classifier_5_weight, cl6_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl5_weight_hat = torch.zeros_like(classifier_4_weight).to(self.device)
            for l4 in range(len(l4_to_l5)):
                cl5_weight_hat[l4, :] = torch.sum(classifier_5_weight[l4_to_l5[l4], :], dim=0)
            cl5_weight_hat = F.normalize(cl5_weight_hat, p=2, dim=1)
            l2_weight_4_5 = torch.mean(cosine_sim(classifier_4_weight, cl5_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl4_weight_hat = torch.zeros_like(classifier_3_weight).to(self.device)
            for l3 in range(len(l3_to_l4)):
                cl4_weight_hat[l3, :] = torch.sum(classifier_4_weight[l3_to_l4[l3], :], dim=0)
            cl4_weight_hat = F.normalize(cl4_weight_hat, p=2, dim=1)
            l2_weight_3_4 = torch.mean(cosine_sim(classifier_3_weight, cl4_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl3_weight_hat = torch.zeros_like(classifier_2_weight).to(self.device)
            for l2 in range(len(l2_to_l3)):
                cl3_weight_hat[l2, :] = torch.sum(classifier_3_weight[l2_to_l3[l2], :], dim=0)
            cl3_weight_hat = F.normalize(cl3_weight_hat, p=2, dim=1)
            l2_weight_2_3 = torch.mean(cosine_sim(classifier_2_weight, cl3_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            cl2_weight_hat = torch.zeros_like(classifier_1_weight).to(self.device)
            for l1 in range(len(l1_to_l2)):
                cl2_weight_hat[l1, :] = torch.sum(classifier_2_weight[l1_to_l2[l1], :], dim=0)
            cl2_weight_hat = F.normalize(cl2_weight_hat, p=2, dim=1)
            l2_weight_1_2 = torch.mean(cosine_sim(classifier_1_weight, cl2_weight_hat))
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l12 = torch.zeros_like(l11_out).to(self.device)
            for l11 in range(len(l11_to_l12)):
                p_hat_l12[:, l11] = torch.sum(l12_out[:, l11_to_l12[l11]], dim=1)
            jsd_level_11_12 = simloss(p_hat_l12, l11_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l11 = torch.zeros_like(l10_out).to(self.device)
            for l10 in range(len(l10_to_l11)):
                p_hat_l11[:, l10] = torch.sum(l11_out[:, l10_to_l11[l10]], dim=1)
            jsd_level_10_11 = simloss(p_hat_l11, l10_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l10 = torch.zeros_like(l9_out).to(self.device)
            for l9 in range(len(l9_to_l10)):
                p_hat_l10[:, l9] = torch.sum(l10_out[:, l9_to_l10[l9]], dim=1)
            jsd_level_9_10 = simloss(p_hat_l10, l9_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l9 = torch.zeros_like(l8_out).to(self.device)
            for l8 in range(len(l8_to_l9)):
                p_hat_l9[:, l8] = torch.sum(l9_out[:, l8_to_l9[l8]], dim=1)
            jsd_level_8_9 = simloss(p_hat_l9, l8_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l8 = torch.zeros_like(l7_out).to(self.device)
            for l7 in range(len(l7_to_l8)):
                p_hat_l8[:, l7] = torch.sum(l8_out[:, l7_to_l8[l7]], dim=1)
            jsd_level_7_8 = simloss(p_hat_l8, l7_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l7 = torch.zeros_like(l6_out).to(self.device)
            for l6 in range(len(l6_to_l7)):
                p_hat_l7[:, l6] = torch.sum(l7_out[:, l6_to_l7[l6]], dim=1)
            jsd_level_6_7 = simloss(p_hat_l7, l6_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l6 = torch.zeros_like(l5_out).to(self.device)
            for l5 in range(len(l5_to_l6)):
                p_hat_l6[:, l5] = torch.sum(l6_out[:, l5_to_l6[l5]], dim=1)
            jsd_level_5_6 = simloss(p_hat_l6, l5_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l5 = torch.zeros_like(l4_out).to(self.device)
            for l4 in range(len(l4_to_l5)):
                p_hat_l5[:, l4] = torch.sum(l5_out[:, l4_to_l5[l4]], dim=1)
            jsd_level_4_5 = simloss(p_hat_l5, l4_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l4 = torch.zeros_like(l3_out).to(self.device)
            for l3 in range(len(l3_to_l4)):
                p_hat_l4[:, l3] = torch.sum(l4_out[:, l3_to_l4[l3]], dim=1)
            jsd_level_3_4 = simloss(p_hat_l4, l3_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l3 = torch.zeros_like(l2_out).to(self.device)
            for l2 in range(len(l2_to_l3)):
                p_hat_l3[:, l2] = torch.sum(l3_out[:, l2_to_l3[l2]], dim=1)
            jsd_level_2_3 = simloss(p_hat_l3, l2_out)
            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            p_hat_l2 = torch.zeros_like(l1_out).to(self.device)
            for l1 in range(len(l1_to_l2)):
                p_hat_l2[:, l1] = torch.sum(l2_out[:, l1_to_l2[l1]], dim=1)
            jsd_level_1_2 = simloss(p_hat_l2, l1_out)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        jsd_loss = jsd_level_11_12 + jsd_level_10_11 + jsd_level_9_10 + jsd_level_8_9 + jsd_level_7_8 + jsd_level_6_7 + jsd_level_5_6 + jsd_level_4_5 + jsd_level_3_4 + jsd_level_2_3 + jsd_level_1_2
        ce_loss =  ce_loss_l12 #+ ce_loss_l11 + ce_loss_l10 + ce_loss_l9 + ce_loss_l8 + ce_loss_l7 + ce_loss_l6 + ce_loss_l5 + ce_loss_l4 + ce_loss_l3 + ce_loss_l2 + ce_loss_l1
        l2_weight_loss = l2_weight_1_2 + l2_weight_2_3 + l2_weight_3_4 + l2_weight_4_5 + l2_weight_5_6 + l2_weight_6_7 + l2_weight_7_8 + l2_weight_8_9 + l2_weight_9_10 + l2_weight_10_11 + l2_weight_11_12
        dissim_loss = dissim_loss_l11 + dissim_loss_l10 + dissim_loss_l9 + dissim_loss_l8 + dissim_loss_l7 + dissim_loss_l6 + dissim_loss_l5 + dissim_loss_l4 + dissim_loss_l3 + dissim_loss_l2 + dissim_loss_l1
        self.dissimiloss_list = [dissim_loss_l11.item(), dissim_loss_l10.item(), dissim_loss_l9.item(), dissim_loss_l8.item(), dissim_loss_l7.item(), dissim_loss_l6.item(), 
        dissim_loss_l5.item(), dissim_loss_l4.item(), dissim_loss_l3.item(), dissim_loss_l2.item(), dissim_loss_l1.item()]
        self.ce_loss_list = [ce_loss_l12.item()]
        self.jsd_loss_list = [jsd_level_11_12.item(), jsd_level_10_11.item(), jsd_level_9_10.item(), jsd_level_8_9.item(),
                 jsd_level_7_8.item(), jsd_level_6_7.item(), jsd_level_5_6.item(), jsd_level_4_5.item(), jsd_level_3_4.item(),
                  jsd_level_2_3.item(), jsd_level_1_2.item()]
        self.l2_weight_list = [l2_weight_11_12.item(), l2_weight_10_11.item(), l2_weight_9_10.item(), l2_weight_8_9.item(), l2_weight_7_8.item(), l2_weight_6_7.item(),
                          l2_weight_5_6.item(), l2_weight_4_5.item(), l2_weight_3_4.item(), l2_weight_2_3.item(), l2_weight_1_2.item()]
        self.acc_list = [acc_l12, acc_l11, acc_l10, acc_l9, acc_l8, acc_l7, acc_l6, acc_l5, acc_l4, acc_l3, acc_l2, acc_l1]
        self.loss = ce_loss + jsd_loss - l2_weight_loss + dissim_loss
        return l12_out

