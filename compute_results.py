import argparse
import os
import json
import shutil
from datetime import datetime

import numpy as np
from distutils.util import strtobool as boolean
from pprint import PrettyPrinter

import wandb

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models

from MBM.better_mistakes.util.rand import make_deterministic
from MBM.better_mistakes.util.folders import get_expm_folder
from MBM.better_mistakes.util.label_embeddings import create_embedding_layer
from MBM.better_mistakes.util.devise_and_bd import generate_sorted_embedding_tensor
from MBM.better_mistakes.util.config import load_config
from MBM.better_mistakes.data.softmax_cascade import SoftmaxCascade
from MBM.better_mistakes.model.init import init_model_on_gpu
from MBM.better_mistakes.model.run_xent import run
from MBM.better_mistakes.model.run_nn import run_nn
from MBM.better_mistakes.model.labels import make_all_soft_labels
from MBM.better_mistakes.model.losses import HierarchicalCrossEntropyLoss, CosineLoss, RankingLoss, CosinePlusXentLoss, YOLOLoss
from MBM.better_mistakes.trees import load_hierarchy, get_weighting, load_distances, get_classes
from CRM.src import compute_results
from util import data_loader, logger
from util.data_loader import is_sorted
import copy

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

TASKS = ["training", "testing"]
CUSTOM_MODELS = ["custom_resnet18", "wide_resnet"]
MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.extend(CUSTOM_MODELS)
LOSS_NAMES = ["cross-entropy", "soft-labels", "hierarchical-cross-entropy", "cosine-distance", "ranking-loss", "cosine-plus-xent", "yolo-v2",
"flamingo-l3", "flamingo-l5", "flamingo-l7","flamingo-l12", "ours-l3", "ours-l5", "ours-l7", "ours-l12", "ours-flamingo-l7", "vanilla-single",
"ours-l12-cejsd", "ours-l12-cejsd-wtconst", "ours-l7-cejsd", "ours-l7-cejsd-wtconst"]
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD", "custom_sgd"]
DATASET_NAMES = ["tiered-imagenet-84", "inaturalist19-84", "tiered-imagenet-224", "inaturalist19-224", "cifar-100"]

# config = None

def cosine_anneal_schedule(t, nb_epoch):
    cos_inner = np.pi * (t % nb_epoch)  # t - 1 is used when t has 1-based indexing.
    cos_inner /= nb_epoch
    cos_out = np.cos(cos_inner) + 1
    return float( 0.1 / 2 * cos_out)


def main_worker(gpus_per_node, opts):
    # Worker setup
    if opts.gpu is not None:
        print("Use GPU: {} for training".format(opts.gpu))
    # if opts.if_sweep:
    #     # logger.init(project='cifar-100-sweep', entity='hierarchical-classification', config=opts)
    #     config = wandb.config
    #     opts.lr_backbone = config.lr_backbone
    #     opts.lr_classifier = opts.classifier

    # Enables the cudnn auto-tuner to find the best algorithm to use for your hardware
    cudnn.benchmark = True

    # pretty printer for cmd line options
    pp = PrettyPrinter(indent=4)

    # Setup data loaders --------------------------------------------------------------------------------------------------------------------------------------
    train_dataset, val_dataset, train_loader, val_loader = data_loader.train_data_loader(opts)
    test_dataset, test_loader = data_loader.test_data_loader(opts)

    # Load hierarchy and classes ------------------------------------------------------------------------------------------------------------------------------
    distances = load_distances(opts.data, 'ilsvrc', opts.data_dir)
    hierarchy = load_hierarchy(opts.data, opts.data_dir)

    if opts.loss == "yolo-v2":
        classes, _ = get_classes(hierarchy, output_all_nodes=True)
    else:
        if opts.data == "cifar-100":
            classes = train_dataset.class_to_idx
            classes = ["L3_" + str(classes[i]) for i in classes]
        else:
            classes = train_dataset.classes

    # Adjust the number of epochs to the size of the dataset
    num_batches = len(train_loader)
    if opts.epochs == None:
        opts.epochs = int(round(opts.num_training_steps / num_batches))

    opts.num_classes = len(classes)
    print("num_classes: ", opts.num_classes)

    # shuffle hierarchy nodes
    if opts.shuffle_classes:
        np.random.shuffle(classes)

    # Model, loss, optimizer ----------------------------------------------------------------------------------------------------------------------------------

    # more setup for devise and b+d
    if opts.devise:
        assert not opts.barzdenzler
        assert opts.loss == "cosine-distance" or opts.loss == "ranking-loss"
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    if opts.barzdenzler:
        assert not opts.devise
        assert opts.loss in ["cosine-distance", "ranking-loss", "cosine-plus-xent"]
        embeddings_mat, sorted_keys = generate_sorted_embedding_tensor(opts)
        embeddings_mat = embeddings_mat / np.linalg.norm(embeddings_mat, axis=1, keepdims=True)
        emb_layer, _, opts.embedding_size = create_embedding_layer(embeddings_mat)
        assert is_sorted(sorted_keys)

    # setup model
    model = init_model_on_gpu(gpus_per_node, opts)
    # setup optimizer
    optimizer = _select_optimizer(model, opts)

    # load from checkpoint if existing
    steps = _load_checkpoint(opts, model, optimizer)

    # setup loss
    if opts.loss == "cross-entropy":
        loss_function = nn.CrossEntropyLoss().cuda(opts.gpu)
    elif opts.loss == "soft-labels":
        loss_function = nn.KLDivLoss().cuda(opts.gpu)
    elif opts.loss == "hierarchical-cross-entropy":
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        loss_function = HierarchicalCrossEntropyLoss(hierarchy, classes, weights).cuda(opts.gpu)
    elif opts.loss == "yolo-v2":
        cascade = SoftmaxCascade(hierarchy, classes).cuda(opts.gpu)
        num_leaf_classes = len(hierarchy.treepositions("leaves"))
        weights = get_weighting(hierarchy, "exponential", value=opts.alpha)
        loss_function = YOLOLoss(hierarchy, classes, weights).cuda(opts.gpu)

        def yolo2_corrector(output):
            return cascade.final_probabilities(output)[:, :num_leaf_classes]

    elif opts.loss == "cosine-distance":
        loss_function = CosineLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss == "ranking-loss":
        loss_function = RankingLoss(emb_layer, opts.batch_size, opts.devise_single_negative, margin=0.1).cuda(opts.gpu)
    elif opts.loss == "cosine-plus-xent":
        loss_function = CosinePlusXentLoss(emb_layer).cuda(opts.gpu)
    elif opts.loss in LOSS_NAMES:
        loss_function = None
    else:
        raise RuntimeError("Unkown loss {}".format(opts.loss))

    # for yolo, we need to decode the output of the classifier as it outputs the conditional probabilities
    corrector = yolo2_corrector if opts.loss == "yolo-v2" else lambda x: x

    # create the solft labels
    soft_labels = make_all_soft_labels(distances, classes, opts.beta)

    # Training/evaluation -------------------------------------------------------------------------------------------------------------------------------------

    requires_grad_to_set = True
    # best_accuracy = 0
    # opts.start_epoch = 0
    opts.epochs = opts.start_epoch + opts.epochs
    crm_opts = copy.copy(opts)
    for epoch in range(opts.start_epoch, opts.epochs):
        # do we validate at this epoch?

        if opts.data == "inaturalist19-224":
            if opts.loss == "cross-entropy": #or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy":
                if opts.arch == "custom_resnet18":
                    optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                    optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
                # elif opts.arch == "custom_resnet18_default":
                #     optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                #     optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
            elif (opts.loss == "ours-l7" or opts.loss == "flamingo-l7" or opts.loss == "ours-flamingo-l7" \
                or opts.loss == "ours-l7-cejsd" or opts.loss == "ours-l7-cejsd-wtconst"):
                optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[5]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[6]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[7]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[8]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
            elif opts.loss == "ours-l5" or opts.loss == "flamingo-l5":
                optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[5]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[6]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) /10
            elif opts.loss == "ours-l3" or opts.loss == "flamingo-l3":
                optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
        if opts.data == "tiered-imagenet-224":
            if opts.loss == "cross-entropy" and opts.optimizer == "custom_sgd":
                optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
            elif (opts.loss == "flamingo-l12" and opts.optimizer == "custom_sgd") or \
                    (opts.loss == "ours-l12" and opts.optimizer == "custom_sgd"):
                optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[4]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[5]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[6]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[7]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[8]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[9]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[10]['lr'] = cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[11]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[12]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                optimizer.param_groups[13]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
            else:
                print("Other losses are not integrated yet for TieredImagenet.")
        if opts.data == "cifar-100":
            if opts.optimizer == "custom_sgd" :
                if opts.loss == "cross-entropy": #or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy":
                    optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                    optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
                    # optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)
                    # optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10
                if opts.loss == "flamingo-l3" or opts.loss == "ours-l3":
                    optimizer.param_groups[0]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)       # level 1
                    optimizer.param_groups[1]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)       # level 2
                    optimizer.param_groups[2]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs)       # level 3
                    optimizer.param_groups[3]['lr'] =  cosine_anneal_schedule(epoch, opts.epochs) / 10  # backbone

        # TRAINING
        if opts.devise or opts.barzdenzler:
            # two-stage training for devise and b+d
            if requires_grad_to_set and steps > opts.train_backbone_after:
                for param in model.parameters():
                    param.requires_grad = True
                requires_grad_to_set = False
            summary_train, steps = run_nn(
                train_loader, model, loss_function, distances, classes, opts, epoch, steps, emb_layer, embeddings_mat, optimizer, is_inference=False,
            )
        else:
            summary_train, steps = run(
                train_loader, model, loss_function, distances, soft_labels, classes, opts, epoch, steps, optimizer, is_inference=False, corrector=corrector,
            )

        logger.log(opts.out_folder, summary_train, epoch, is_training=True, save_log=True)

        # print summary of the epoch and save checkpoint
        model.eval()
        state = {"epoch": epoch + 1, "steps": steps, "arch": opts.arch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        _save_checkpoint(state, opts.out_folder, epoch+1)
        model.train()
        print("\nSummary for epoch %04d (for train set):" % epoch)
        pp.pprint(summary_train)

        # VALIDATION MBM
        # if opts.devise or opts.barzdenzler:
        #     summary_val, steps = run_nn(
        #         val_loader, model, loss_function, distances, classes, opts, epoch, steps, emb_layer, embeddings_mat, is_inference=True,
        #     )
        # else:
        #     summary_val, steps = run(
        #         val_loader, model, loss_function, distances, soft_labels, classes, opts, epoch, steps, is_inference=True, corrector=corrector,
        #     )
        # logger.log(opts.out_folder, summary_val, epoch, is_validation=True, save_log=True)
        # _save_best_checkpoint(state, epoch, opts.out_folder)
        # print("\nSummary for epoch %04d (for val set):" % epoch)
        # pp.pprint(summary_val)
        # print("\n\n")
        #
        # # TESTING MBM
        # if opts.devise or opts.barzdenzler:
        #     summary_test, steps = run_nn(
        #         test_loader, model, loss_function, distances, classes, opts, 0, 0, emb_layer, embeddings_mat, is_inference=True,
        #     )
        # else:
        #     summary_test, steps = run(
        #         test_loader, model, loss_function, distances, soft_labels, classes, opts, 0, 0, is_inference=True, corrector=corrector,
        #     )
        # logger.log(opts.out_folder, summary_test, epoch, is_testing=True, save_log=True)
        # print("\nSummary for epoch %04d (for test set):" % epoch)
        # pp.pprint(summary_test)
        # print("\n\n")

        # VAL AND TEST CRM
        crm_opts.out_folder = os.path.join(opts.out_folder, f"{epoch}")
        logger._print("CRM Test Results >>>>>>>>>>", os.path.join(crm_opts.out_folder, "logs.txt"))
        compute_results.main(crm_opts, crm_opts.out_folder)

def _load_checkpoint(opts, model, optimizer):
    if os.path.isfile(os.path.join(opts.out_folder, "best.checkpoint.pth.tar")):
        print("=> loading checkpoint '{}'".format(opts.out_folder))
        checkpoint = torch.load(os.path.join(opts.out_folder, "best.checkpoint.pth.tar"))
        opts.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # steps = checkpoint["steps"]
        steps = 0
        print("=> loaded checkpoint '{}' (epoch {})".format(opts.out_folder, checkpoint["epoch"]))
    elif opts.pretrained_folder is not None:
        if os.path.exists(opts.pretrained_folder):
            print("=> loading pretrained checkpoint '{}'".format(opts.pretrained_folder))
            if os.path.isdir(opts.pretrained_folder):
                checkpoint = torch.load(os.path.join(opts.pretrained_folder, "best.checkpoint.pth.tar"))
            else:
                checkpoint = torch.load(opts.pretrained_folder)
            if opts.devise or opts.barzdenzler:
                model_dict = model.state_dict()
                pretrained_dict = checkpoint["state_dict"]
                # filter out FC layer
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ["fc.1.weight", "fc.1.bias"]}
                # overwrite entries in the existing state dict
                model_dict.update(pretrained_dict)
                # load the new state dict
                model.load_state_dict(pretrained_dict, strict=False)
            else:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            steps = 0
            print("=> loaded pretrained checkpoint '{}' (epoch {})".format(opts.pretrained_folder, checkpoint["epoch"]))
        else:
            raise FileNotFoundError("Can not find {}".format(opts.pretrained_folder))
    else:
        steps = 0
        print("=> no checkpoint found at '{}'".format(opts.out_folder))

    return steps

def _save_checkpoint(state, out_folder, epoch):
    filename = os.path.join(out_folder, f"checkpoint.pth.tar")
    torch.save(state, filename)


def _save_best_checkpoint(state, epoch, out_folder):
    filename = os.path.join(out_folder, f"{epoch}", "best.checkpoint.pth.tar")
    if not os.path.exists(os.path.join(out_folder, f"{epoch}")):
        os.makedirs(os.path.join(out_folder, f"{epoch}"))
    torch.save(state, filename)


def _select_optimizer(model, opts):
    if opts.optimizer == "adagrad":
        return torch.optim.Adagrad(model.parameters(), opts.lr, weight_decay=opts.weight_decay)
    elif opts.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=False)
    elif opts.optimizer == "adam_amsgrad":
        if opts.devise or opts.barzdenzler:
            return torch.optim.Adam(
                [
                    {"params": model.conv1.parameters()},
                    {"params": model.layer1.parameters()},
                    {"params": model.layer2.parameters()},
                    {"params": model.layer3.parameters()},
                    {"params": model.layer4.parameters()},
                    {"params": model.fc.parameters(), "lr": opts.lr_fc, "weight_decay": opts.weight_decay_fc},
                ],
                lr=opts.lr,
                weight_decay=opts.weight_decay,
                amsgrad=True,
            )
        else:
            return torch.optim.Adam(model.parameters(), opts.lr, weight_decay=opts.weight_decay, amsgrad=True, )
    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0)
    elif opts.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), opts.lr, weight_decay=opts.weight_decay, momentum=0, nesterov=False, )
    elif opts.optimizer == "custom_sgd":
        if opts.data == "cifar-100": #or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy":
            if opts.loss == "cross-entropy": #or opts.loss == "soft-labels":
                return torch.optim.SGD([
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "soft-labels":
                return torch.optim.SGD([
                    {'params': model.classifier_3.parameters(), 'lr': 0.0005},          # volcanic-sweep-10 parameters.
                    {'params': model.features_2.parameters(), 'lr': 0.05},
                ], momentum=0.9, weight_decay=5e-4)
                # return torch.optim.SGD([
                #     {'params': model.classifier_3.parameters(), 'lr': opts.lr_classifier},
                #     {'params': model.features_2.parameters(), 'lr': opts.lr_backbone},
                # ], momentum=0.9, weight_decay=5e-4)
                # return torch.optim.SGD([
                #     {'params': model.classifier_3.parameters(), 'lr': 0.0001},        #
                #     {'params': model.features_2.parameters(), 'lr': 0.1},
                # ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "flamingo-l3" or opts.loss == "ours-l3":
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)

        if opts.data == "inaturalist19-224":
            if opts.loss == "cross-entropy":
                if opts.arch == "custom_resnet18":
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': 0.1},
                        {'params': model.features_1.parameters(), 'lr': 0.1},
                        {'params': model.features_2.parameters(), 'lr': 0.01},
                    ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "soft-labels": #or opts.loss == "hierarchical-cross-entropy":
                if opts.arch == "custom_resnet18":
                    return torch.optim.SGD([
                        {'params': model.classifier_3.parameters(), 'lr': opts.lr_classifier},
                        {'params': model.features_1.parameters(), 'lr': opts.lr_backbone},
                        {'params': model.features_2.parameters(), 'lr': opts.lr_backbone},
                    ], momentum=0.9, weight_decay=5e-4)
                # elif opts.arch == "custom_resnet18_default":
                #     return torch.optim.SGD([
                #         {'params': model.classifier_3.parameters(), 'lr': 0.1},
                #         {'params': model.features_2.parameters(), 'lr': 0.01},
                #     ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "flamingo-l7" or opts.loss == "ours-l7" or opts.loss == "ours-flamingo-l7" or opts.loss == "vanilla-single":
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "flamingo-l5" or opts.loss == "ours-l5":
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "flamingo-l3" or opts.loss == "ours-l3":
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ],
                    momentum=0.9, weight_decay=5e-4)
        if opts.data == "tiered-imagenet-224":
            if opts.loss == "cross-entropy" or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy":
                return torch.optim.SGD([
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
            if opts.loss == "flamingo-l12" or opts.loss == "ours-l12":             # Flamingo and ours doesn't work yet.
                return torch.optim.SGD([
                    {'params': model.classifier_1.parameters(), 'lr': 0.1},
                    {'params': model.classifier_2.parameters(), 'lr': 0.1},
                    {'params': model.classifier_3.parameters(), 'lr': 0.1},
                    {'params': model.classifier_4.parameters(), 'lr': 0.1},
                    {'params': model.classifier_5.parameters(), 'lr': 0.1},
                    {'params': model.classifier_6.parameters(), 'lr': 0.1},
                    {'params': model.classifier_7.parameters(), 'lr': 0.1},
                    {'params': model.classifier_8.parameters(), 'lr': 0.1},
                    {'params': model.classifier_9.parameters(), 'lr': 0.1},
                    {'params': model.classifier_10.parameters(), 'lr': 0.1},
                    {'params': model.classifier_11.parameters(), 'lr': 0.1},
                    {'params': model.classifier_12.parameters(), 'lr': 0.1},
                    {'params': model.features_1.parameters(), 'lr': 0.1},
                    {'params': model.features_2.parameters(), 'lr': 0.01},
                ], momentum=0.9, weight_decay=5e-4)
    else:
        raise ValueError("Unknown optimizer", opts.loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES,
                        help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES,
                        help="loss type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None,
                        help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--epochs", default=None, type=int, help="number of epochs")
    parser.add_argument("--num_training_steps", default=200000, type=int,
                        help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    parser.add_argument("--shuffle_classes", default=False, type=boolean, help="Shuffle classes in the hierarchy")
    parser.add_argument("--beta", default=0, type=float,
                        help="Softness parameter: the higher, the closer to one-hot encoding")
    parser.add_argument("--alpha", type=float, default=0, help="Decay parameter for hierarchical cross entropy.")
    # Devise/B&D ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--devise", type=boolean, default=False, help="Use DeViSe label embeddings")
    parser.add_argument("--devise_single_negative", type=boolean, default=False,
                        help="Use one negative per samples instead of all")
    parser.add_argument("--barzdenzler", type=boolean, default=False, help="Use Barz&Denzler label embeddings")
    parser.add_argument("--train_backbone_after", default=float("inf"), type=float,
                        help="Start training backbone too after this many steps")
    parser.add_argument("--use_2fc", default=False, type=boolean, help="Use two FC layers for Devise")
    parser.add_argument("--fc_inner_dim", default=1024, type=int, help="If use_2fc is True, their inner dimension.")
    parser.add_argument("--lr_fc", default=1e-3, type=float, help="learning rate for FC layers")
    parser.add_argument("--weight_decay_fc", default=0.0, type=float, help="weight decay of FC layers")
    parser.add_argument("--use_fc_batchnorm", default=False, type=boolean, help="Batchnorm layer in network head")
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--random", default=False, type=boolean, help="Use random hierarchy?")
    parser.add_argument("--data", default="tiered-imagenet-224",
                        help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int,
                        help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="data_paths.yml")
    parser.add_argument("--data-path", default=None,
                        help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default=None, help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str,
                        help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=5, type=int,
                        help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    parser.add_argument("--start", default="testing", choices=TASKS, help="name of the task | ".join(TASKS))
    ## CRM ----------------------------------------------------------------------------------
    parser.add_argument("--rerank",default=1,type=int,help='whether to use CRM or not')
    parser.add_argument("--checkpoint_path",default=None,type=str,help='path to the best checkpoint file')

    opts = parser.parse_args()

    if opts.random:
        # opts.output = os.path.join("random_hierarchy", opts.output)
        opts.data_dir = os.path.join(opts.data_dir, "random_hierarchy")

    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if not os.path.exists(os.path.join(opts.out_folder, "json/train")):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder, "json/train"))
        os.makedirs(os.path.join(opts.out_folder, "json/val"))
        os.makedirs(os.path.join(opts.out_folder, "json/test"))
        # os.makedirs(os.path.join(opts.out_folder, "model_snapshots"))

    # set if we want to output soft labels or one hot
    opts.soft_labels = opts.beta != 0

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(opts))
    # with open(os.path.join(opts.out_folder, "opts.json"), "w") as fp:
    #     json.dump(vars(opts), fp)

    # setup data path from config file if needed
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    # setup random number generation
    if opts.seed is not None:
        make_deterministic(opts.seed)

    gpus_per_node = torch.cuda.device_count()

    if opts.data == "cifar-100":
        project = "cifar-100"
    elif opts.data == "inaturalist19-224":
        project = "iNaturalist19"
    elif opts.data == "tiered-imagenet-224":
        project = "TieredImagenet"
    entity = "hierarchical-classification"
    # project = "mbm"
    # entity = "depanshu-sani"
    if opts.loss == "soft-labels":
        run_name = "soft-labels (β=%.1f)" % opts.beta
    elif opts.loss == "cross-entropy":
        run_name = "cross-entropy"
    elif opts.loss == "hierarchical-cross-entropy":
        run_name = "hxe (α=%f)" % opts.alpha
    else:
        run_name = opts.loss
    if opts.random:
        run_name += "(random_hier)"
    run_name += " (evaluate)"
    logger.init(project=project, entity=entity, config=opts, run_name=run_name)
    main_worker(gpus_per_node, opts)
