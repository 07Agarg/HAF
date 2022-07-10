import torch.cuda
import torch.nn
from torchvision import models
from util.arch import wideresnet, custom_wideresnet, custom_resnet18


def init_model_on_gpu(gpus_per_node, opts, distances=None):
    arch_dict = models.__dict__
    pretrained = False if not hasattr(opts, "pretrained") else opts.pretrained
    distributed = False if not hasattr(opts, "distributed") else opts.distributed
    print("=> using model '{}', pretrained={}".format(opts.arch, pretrained))
    feature_dim = 512
    if opts.arch == "resnet18":
        feature_dim = 512
    elif opts.arch == "resnet50":
        feature_dim = 2048
    else:
        ValueError("Unknown architecture ", opts.arch)

    try:
        model = arch_dict[opts.arch](pretrained=pretrained)
    except:
        pass

    if opts.arch == "wide_resnet":
        model = wideresnet.WideResNet(num_classes=opts.num_classes)
        if opts.loss == "cross-entropy" or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy" or opts.loss == "yolo-v2":
            model = custom_wideresnet.WideResNet(model, feature_size=512, num_classes=opts.num_classes)
        elif opts.loss == "flamingo-l5":
            model = custom_wideresnet.WideResNet_flamingo_l5(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l5-cejsd-wtconst-dissim":
            model = custom_wideresnet.WideResNet_ours_l5_cejsd_wtconst_dissim(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l5-cejsd-wtconst":
            model = custom_wideresnet.WideResNet_ours_l5_cejsd_wtconst(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l5-cejsd":
            model = custom_wideresnet.WideResNet_ours_l5_cejsd(model, feature_size=512, num_classes=[100, 20, 8, 4, 2], gpu=opts.gpu)
        elif opts.barzdenzler:
            model = custom_wideresnet.WideResNet(model, feature_size=512, num_classes=opts.num_classes)

    elif opts.arch == "custom_resnet18":
        model = models.resnet18(pretrained=True)
        if opts.loss == "cross-entropy" or opts.loss == "soft-labels" or opts.loss == "hierarchical-cross-entropy" or opts.loss == "yolo-v2":
            model = custom_resnet18.ResNet18(model, feature_size=600, num_classes=opts.num_classes)
        elif opts.loss == "ours-l7-cejsd":
            model = custom_resnet18.ResNet18_ours_l7_cejsd(model, feature_size=600, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "ours-l7-cejsd-wtconst":
            model = custom_resnet18.ResNet18_ours_l7_cejsd_wtconst(model, feature_size=600, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "ours-l7-cejsd-wtconst-dissim":
            model = custom_resnet18.ResNet18_ours_l7_cejsd_wtconst_dissim(model, feature_size=600, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "flamingo-l7":
            model = custom_resnet18.ResNet18_flamingo_l7(model, feature_size=600, num_classes=[1010, 72, 57, 34, 9, 4, 3], gpu=opts.gpu)
        elif opts.loss == "flamingo-l12":
            model = custom_resnet18.ResNet18_flamingo_l12(model, feature_size=600, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l12-cejsd":
            model = custom_resnet18.ResNet18_ours_l12_cejsd(model, feature_size=600, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l12-cejsd-wtconst":
            model = custom_resnet18.ResNet18_ours_l12_cejsd_wtconst(model, feature_size=600, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
        elif opts.loss == "ours-l12-cejsd-wtconst-dissim":
            model = custom_resnet18.ResNet18_ours_l12_cejsd_wtconst_dissim(model, feature_size=600, num_classes=[608, 607, 584, 510, 422, 270, 159, 86, 35, 21, 5, 2], gpu=opts.gpu)
    else:
        model.fc = torch.nn.Sequential(torch.nn.Dropout(opts.dropout), torch.nn.Linear(in_features=feature_dim, out_features=opts.num_classes, bias=True))

    if opts.devise or opts.barzdenzler:
        if opts.pretrained or opts.pretrained_folder:
            for param in model.parameters():
                if opts.train_backbone_after == 0:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        if opts.use_2fc:
            if opts.use_fc_batchnorm:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(opts.fc_inner_dim),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
            else:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.fc_inner_dim, bias=True),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=opts.fc_inner_dim, out_features=opts.embedding_size, bias=True),
                )
            if "l12" in opts.loss:
                model.classifier_12 = layer
            elif "l7" in opts.loss:
                model.classifier_7 = layer
            elif "l5" in opts.loss:
                model.classifier_5 = layer
            else:
                model.fc = layer
        else:
            if opts.use_fc_batchnorm:
                layer = torch.nn.Sequential(
                    torch.nn.BatchNorm1d(feature_dim),
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True)
                )
            else:
                layer = torch.nn.Sequential(
                    torch.nn.Linear(in_features=feature_dim, out_features=opts.embedding_size, bias=True))
            if "l12" in opts.loss:
                model.classifier_12 = layer
            elif "l7" in opts.loss:
                model.classifier_7 = layer
            elif "l5" in opts.loss:
                model.classifier_5 = layer
            else:
                model.fc = layer

    if distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opts.gpu is not None:
            torch.cuda.set_device(opts.gpu)
            model.cuda(opts.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            opts.batch_size = int(opts.batch_size / gpus_per_node)
            opts.workers = int(opts.workers / gpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[opts.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif opts.gpu is not None:
        torch.cuda.set_device(opts.gpu)
        model = model.cuda(opts.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()
    return model
