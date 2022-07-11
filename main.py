import argparse
import os
import json

from distutils.util import strtobool as boolean
from pprint import PrettyPrinter

import wandb

import torch.utils.data.distributed
import torchvision.models as models

from MBM.better_mistakes.util.rand import make_deterministic
from MBM.better_mistakes.util.folders import get_expm_folder
from MBM.better_mistakes.util.config import load_config

from MBM.scripts import start_training, start_testing
from util import logger

TASKS = ["training", "testing"]
CUSTOM_MODELS = ["custom_resnet18", "wide_resnet"]
MODEL_NAMES = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))
MODEL_NAMES.extend(CUSTOM_MODELS)
# l5 refers to loss of level-5 for CIFAR-100, l7 refers to loss of level-7 for iNaturalist-19, l12 refers to loss of level-12 for tiered-imageneget-224.
LOSS_NAMES = ["cross-entropy", "soft-labels", "hierarchical-cross-entropy", "cosine-distance", "ranking-loss", "cosine-plus-xent", "yolo-v2",
              "flamingo-l5", "flamingo-l7", "flamingo-l12",                                                         # Cheng et al's [7]
              "ours-l5-cejsd", "ours-l7-cejsd", "ours-l12-cejsd",                                                   # HAF with only Soft Hierarchical Consistency Loss (Section 3.2)
              "ours-l5-cejsd-wtconst", "ours-l7-cejsd-wtconst", "ours-l12-cejsd-wtconst",                           # HAF with Soft Hierarchical Consistency Loss (Section 3.2) + Geometric Consistency Loss (Section 3.4)
              "ours-l5-cejsd-wtconst-dissim", "ours-l7-cejsd-wtconst-dissim", "ours-l12-cejsd-wtconst-dissim"]      # HAF with all three losses (Section 3.2, 3.3, 3.4)
OPTIMIZER_NAMES = ["adagrad", "adam", "adam_amsgrad", "rmsprop", "SGD", "custom_sgd"]
DATASET_NAMES = ["tiered-imagenet-84", "inaturalist19-84", "tiered-imagenet-224", "inaturalist19-224", "cifar-100"] 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="training", choices=TASKS, help="name of the task | ".join(TASKS))
    parser.add_argument("--arch", default="resnet18", choices=MODEL_NAMES, help="model architecture: | ".join(MODEL_NAMES))
    parser.add_argument("--loss", default="cross-entropy", choices=LOSS_NAMES, help="loss type: | ".join(LOSS_NAMES))
    parser.add_argument("--optimizer", default="adam_amsgrad", choices=OPTIMIZER_NAMES, help="optimizer type: | ".join(OPTIMIZER_NAMES))
    parser.add_argument("--lr", default=1e-5, type=float, help="initial learning rate of optimizer")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay of optimizer")
    parser.add_argument("--pretrained", type=boolean, default=True, help="start from ilsvrc12/imagenet model weights")
    parser.add_argument("--pretrained_folder", type=str, default=None, help="folder or file from which to load the network weights")
    parser.add_argument("--dropout", default=0.0, type=float, help="Prob of dropout for network FC layer")
    parser.add_argument("--data_augmentation", type=boolean, default=True, help="Train with basic data augmentation")
    parser.add_argument("--epochs", default=None, type=int, help="number of epochs")
    parser.add_argument("--num_training_steps", default=200000, type=int, help="number of total steps to train for (num_batches*num_epochs)")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=256, type=int, help="total batch size")
    parser.add_argument("--shuffle_classes", default=False, type=boolean, help="Shuffle classes in the hierarchy")
    parser.add_argument("--beta", default=0, type=float, help="Softness parameter: the higher, the closer to one-hot encoding")
    parser.add_argument("--alpha", type=float, default=0, help="Decay parameter for hierarchical cross entropy.")
    # Devise/B&D ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--devise", type=boolean, default=False, help="Use DeViSe label embeddings")
    parser.add_argument("--devise_single_negative", type=boolean, default=False, help="Use one negative per samples instead of all")
    parser.add_argument("--barzdenzler", type=boolean, default=False, help="Use Barz&Denzler label embeddings")
    parser.add_argument("--train_backbone_after", default=float("inf"), type=float, help="Start training backbone too after this many steps")
    parser.add_argument("--use_2fc", default=False, type=boolean, help="Use two FC layers for Devise")
    parser.add_argument("--fc_inner_dim", default=1024, type=int, help="If use_2fc is True, their inner dimension.")
    parser.add_argument("--lr_fc", default=1e-3, type=float, help="learning rate for FC layers")
    parser.add_argument("--weight_decay_fc", default=0.0, type=float, help="weight decay of FC layers")
    parser.add_argument("--use_fc_batchnorm", default=False, type=boolean, help="Batchnorm layer in network head")
    # Data/paths ----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--data", default="inaturalist19-224", help="id of the dataset to use: | ".join(DATASET_NAMES))
    parser.add_argument("--target_size", default=224, type=int, help="Size of image input to the network (target resize after data augmentation)")
    parser.add_argument("--data-paths-config", help="Path to data paths yaml file", default="data_paths.yml")
    parser.add_argument("--data-path", default=None, help="explicit location of the data folder, if None use config file.")
    parser.add_argument("--data_dir", default="data/", help="Folder containing the supplementary data")
    parser.add_argument("--output", default="out/", help="path to the model folder")
    parser.add_argument("--expm_id", default="", type=str, help="Name log folder as: out/<scriptname>/<date>_<expm_id>. If empty, expm_id=time")
    # Log/val -------------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--log_freq", default=100, type=int, help="Log every log_freq batches")
    parser.add_argument("--val_freq", default=1, type=int, help="Validate every val_freq epochs (except the first 10 and last 10)")
    # Execution -----------------------------------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--workers", default=2, type=int, help="number of data loading workers")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training. ")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use.")
    ## CRM ----------------------------------------------------------------------------------
    parser.add_argument("--rerank",default=0,type=int,help='whether to use CRM or not')
    parser.add_argument("--checkpoint_path",default=None,type=str,help='path to the best checkpoint file')

    opts = parser.parse_args()

    # setting the path of level-5 distances and pickle file.
    if opts.data == "cifar-100":
        opts.data_dir = os.path.join(opts.data_dir, "cifar-l5/original/")
        print(opts.data_dir)

    # setup output folder
    opts.out_folder = opts.output if opts.output else get_expm_folder(__file__, "out", opts.expm_id)
    if not os.path.exists(opts.out_folder):
        print("Making experiment folder and subfolders under: ", opts.out_folder)
        os.makedirs(os.path.join(opts.out_folder))

    # set if we want to output soft labels or one hot
    opts.soft_labels = opts.beta != 0

    # print options as dictionary and save to output
    PrettyPrinter(indent=4).pprint(vars(opts))
    if opts.start == "training":
        # Create opts.json file
        with open(os.path.join(opts.out_folder, "opts.json"), "w") as fp:
            json.dump(vars(opts), fp)

    # setup data path from config file if needed
    if opts.data_path is None:
        opts.data_paths = load_config(opts.data_paths_config)
        opts.data_path = opts.data_paths[opts.data]

    # setup random number generation
    if opts.seed is not None:
        make_deterministic(opts.seed)

    # OUR HXE
    if opts.weights:
        opts.weights = tuple(opts.weights)

    gpus_per_node = torch.cuda.device_count()

    if opts.start == "training":
        # Setup wandb logging parameters
        if opts.data == "cifar-100":
            project = "cifar-100"
        elif opts.data == "inaturalist19-224":
            project = "iNaturalist19"
        elif opts.data == "tiered-imagenet-224":
            project = "TieredImagenet"
        entity = "hierarchical-classification"
        if opts.loss == "soft-labels":
            run_name = "soft-labels (β=%.1f)"%opts.beta
        elif opts.loss == "cross-entropy":
            run_name = "cross-entropy"
        elif opts.loss == "hierarchical-cross-entropy":
            run_name = "hxe (α=%f)"%opts.alpha
        else:
            run_name = opts.loss
        logger.init(project=project, entity=entity, config=opts, run_name=run_name)

        # Start training
        start_training.main_worker(gpus_per_node, opts)
    else:
        logger._print("MBM Test Results >>>>>>>>>>", os.path.join(opts.out_folder, "logs.txt"))
        start_testing.main(opts)
