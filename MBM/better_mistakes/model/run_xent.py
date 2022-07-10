import time
import numpy as np
import os.path
import torch
from conditional import conditional
import tensorboardX
from MBM.better_mistakes.model.performance import accuracy
from MBM.better_mistakes.model.labels import make_batch_onehot_labels, make_batch_soft_labels
import wandb

from util.loss_function.jsd import *

# topK_to_consider = (1, 5, 10, 20, 100)
topK_to_consider = (1, )

# lists of ids for loggings performance measures
accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]
level_accuracy = None
level_ce_loss = None
level_jsd_loss = None
level_weight_loss = None
level_dissim_loss = None


def run(loader, model, loss_function, distances, all_soft_labels, classes, opts, epoch, prev_steps, optimizer=None, is_inference=True, corrector=lambda x: x):
    """
    Runs training or inference routine for standard classification with soft-labels style losses
    """
    global topK_to_consider, accuracy_ids, dist_avg_ids, dist_top_ids, dist_avg_mistakes_ids, hprec_ids, hmAP_ids, level_accuracy, level_ce_loss, level_jsd_loss, level_weight_loss, level_dissim_loss

    if opts.start == "training":
        topK_to_consider = (1,)
    else:
        topK_to_consider = (1, 5, 10, 20, 100)

    # lists of ids for loggings performance measures
    accuracy_ids = ["accuracy_top/%02d" % i for i in topK_to_consider]
    dist_avg_ids = ["_avg/%02d" % i for i in topK_to_consider]
    dist_top_ids = ["_top/%02d" % i for i in topK_to_consider]
    dist_avg_mistakes_ids = ["_mistakes/avg%02d" % i for i in topK_to_consider]
    hprec_ids = ["_precision/%02d" % i for i in topK_to_consider]
    hmAP_ids = ["_mAP/%02d" % i for i in topK_to_consider]

    max_dist = max(distances.distances.values())
    # for each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    best_hier_similarities = _make_best_hier_similarities(classes, distances, max_dist)

    # Using different logging frequencies for training and validation
    log_freq = 1 if is_inference else opts.log_freq

    # strings useful for logging
    descriptor = "VAL" if is_inference else "TRAIN"
    loss_id = "loss/" + opts.loss
    dist_id = "ilsvrc_dist"

    # Initialise TensorBoard
    # with_tb = opts.out_folder is not None
    #
    # if with_tb:
    #     tb_writer = tensorboardX.SummaryWriter(os.path.join(opts.out_folder, "tb", descriptor.lower()))

    # Initialise accumulators to store the several measures of performance (accumulate as sum)
    num_logged = 0
    loss_accum = 0.0
    time_accum = 0.0
    norm_mistakes_accum = 0.0
    flat_accuracy_accums = np.zeros(len(topK_to_consider), dtype=np.float)
    flat_level_accuracy_accums = None
    hdist_accums = np.zeros(len(topK_to_consider))
    hdist_top_accums = np.zeros(len(topK_to_consider))
    hdist_mistakes_accums = np.zeros(len(topK_to_consider))
    hprecision_accums = np.zeros(len(topK_to_consider))
    hmAP_accums = np.zeros(len(topK_to_consider))

    # Affects the behaviour of components such as batch-norm
    if is_inference:
        model.eval()
    else:
        model.train()

    with conditional(is_inference, torch.no_grad()):
        time_load0 = time.time()
        for batch_idx, (embeddings, target) in enumerate(loader):

            this_load_time = time.time() - time_load0
            this_rest0 = time.time()

            assert embeddings.size(0) == opts.batch_size, "Batch size should be constant (data loader should have drop_last=True)"
            if opts.gpu is not None:
                embeddings = embeddings.cuda(opts.gpu, non_blocking=True)
            target = target.cuda(opts.gpu, non_blocking=True)

            # get model's prediction
            if (opts.arch == "custom_resnet18" or opts.arch == "wide_resnet") and opts.loss != "yolo-v2":
                if opts.loss == "ours-l12-cejsd-wtconst-dissim" or opts.loss == "ours-l7-cejsd-wtconst-dissim":
                    output = model(embeddings, target, is_inference)
                else:
                    output = model(embeddings, target)
            else:
                output = model(embeddings)

            # for soft-labels we need to add a log_softmax and get the soft labels
            if opts.arch == "custom_resnet18":
                if opts.loss == "flamingo-l12":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(12, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(11, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l12-cejsd":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(11, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l12-cejsd-wtconst":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                        except:
                            pass
                elif opts.loss == "ours-l7-cejsd":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(6, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l7-cejsd-wtconst":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                        except:
                            pass
                elif opts.loss == "ours-l7-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(6, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(6, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l12-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(12, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(11, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(11, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            # flat_level_jsd_loss_accums += model.jsd_loss_list
                            # flat_level_wt_loss_accums += model.l2_weight_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "flamingo-l7":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(7, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(7, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(6, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "soft-labels":
                    output = torch.nn.functional.log_softmax(output, dim=1)
                    if opts.soft_labels:
                        target_distribution = make_batch_soft_labels(all_soft_labels, target, opts.num_classes,
                                                                     opts.batch_size, opts.gpu)
                    else:
                        target_distribution = make_batch_onehot_labels(target, opts.num_classes, opts.batch_size,
                                                                       opts.gpu)
                    loss = loss_function(output, target_distribution)
                else:
                    loss = loss_function(output, target)
            elif opts.arch == "wide_resnet":
                if opts.loss == "flamingo-l5":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(5, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l5-cejsd":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = ["jsd_loss@level-%d" % i for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums += model.jsd_loss_list
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                        except:
                            pass
                elif opts.loss == "ours-l5-cejsd-wtconst":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                        except:
                            pass
                elif opts.loss == "ours-l5-cejsd-wtconst-dissim":
                    loss = model.loss
                    if flat_level_accuracy_accums is None:
                        level_accuracy = ["accuracy@level-%d" % i for i in range(5, 0, -1)]
                        level_ce_loss = ["ce_loss@level-%d" % i for i in range(1, 0, -1)]
                        flat_level_accuracy_accums = model.acc_list
                        flat_level_ce_loss_accums = model.ce_loss_list
                        flat_level_wt_loss_accums = model.l2_weight_list
                        flat_level_dissim_loss_accums = model.dissimiloss_list
                        try:
                            flat_level_jsd_loss_accums = model.jsd_loss_list
                            level_jsd_loss = [f"jsd_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_weight_loss = [f"l2_weight_loss@level-{i+1}-{i}" for i in range(4, 0, -1)]
                            level_dissim_loss = [f"dissim_loss@level-{i}" for i in range(4, 0, -1)]
                        except:
                            pass
                    else:
                        flat_level_accuracy_accums = [sum(accums) for accums in zip(flat_level_accuracy_accums, model.acc_list)]
                        flat_level_ce_loss_accums = [sum(accums) for accums in zip(flat_level_ce_loss_accums, model.ce_loss_list)]
                        try:
                            flat_level_jsd_loss_accums = [sum(accums) for accums in zip(flat_level_jsd_loss_accums, model.jsd_loss_list)]
                            flat_level_wt_loss_accums = [sum(accums) for accums in zip(flat_level_wt_loss_accums, model.l2_weight_list)]
                            flat_level_dissim_loss_accums = [sum(accums) for accums in zip(flat_level_dissim_loss_accums, model.dissimiloss_list)]
                        except:
                            pass
                elif opts.loss == "soft-labels":
                    output = torch.nn.functional.log_softmax(output, dim=1)
                    if opts.soft_labels:
                        target_distribution = make_batch_soft_labels(all_soft_labels, target, opts.num_classes,
                                                                     opts.batch_size, opts.gpu)
                    else:
                        target_distribution = make_batch_onehot_labels(target, opts.num_classes, opts.batch_size,
                                                                       opts.gpu)
                    loss = loss_function(output, target_distribution)
                else:
                    loss = loss_function(output, target)

            elif opts.loss == "soft-labels":
                output = torch.nn.functional.log_softmax(output, dim=1)
                if opts.soft_labels:
                    target_distribution = make_batch_soft_labels(all_soft_labels, target, opts.num_classes, opts.batch_size, opts.gpu)
                else:
                    target_distribution = make_batch_onehot_labels(target, opts.num_classes, opts.batch_size, opts.gpu)
                loss = loss_function(output, target_distribution)
            else:
                loss = loss_function(output, target)

            if not is_inference:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # start/reset timers
            this_rest_time = time.time() - this_rest0
            time_accum += this_load_time + this_rest_time
            time_load0 = time.time()

            # only update total number of batch visited for training
            tot_steps = prev_steps if is_inference else prev_steps + batch_idx

            # correct output of the classifier (for yolo-v2)
            output = corrector(output)

            num_logged += 1
            # compute flat topN accuracy for N \in {topN_to_consider}
            topK_accuracies, topK_predicted_classes = accuracy(output, target, ks=topK_to_consider)
            loss_accum += loss.item()

            if opts.start == "testing":
                topK_hdist = np.empty([opts.batch_size, topK_to_consider[-1]])

                for i in range(opts.batch_size):
                    for j in range(max(topK_to_consider)):
                        class_idx_ground_truth = target[i]
                        class_idx_predicted = topK_predicted_classes[i][j]
                        topK_hdist[i, j] = distances[(classes[class_idx_predicted], classes[class_idx_ground_truth])]
                # select samples which returned the incorrect class (have distance!=0 in the top1 position)
                mistakes_ids = np.where(topK_hdist[:, 0] != 0)[0]
                norm_mistakes_accum += len(mistakes_ids)
                topK_hdist_mistakes = topK_hdist[mistakes_ids, :]
                # obtain similarities from distances
                topK_hsimilarity = 1 - topK_hdist / max_dist
                # all the average precisions @k \in [1:max_k]
                topK_AP = [np.sum(topK_hsimilarity[:, :k]) / np.sum(best_hier_similarities[:, :k]) for k in range(1, max(topK_to_consider) + 1)]

            for i in range(len(topK_to_consider)):
                flat_accuracy_accums[i] += topK_accuracies[i].item()
                if opts.start == "testing":
                    hdist_accums[i] += np.mean(topK_hdist[:, : topK_to_consider[i]])
                    hdist_top_accums[i] += np.mean([np.min(topK_hdist[b, : topK_to_consider[i]]) for b in range(opts.batch_size)])
                    hdist_mistakes_accums[i] += np.sum(topK_hdist_mistakes[:, : topK_to_consider[i]])
                    hprecision_accums[i] += topK_AP[topK_to_consider[i] - 1]
                    hmAP_accums[i] += np.mean(topK_AP[: topK_to_consider[i]])

                # if it is time to log, compute all measures, store in summary and pass to tensorboard.
            if batch_idx % log_freq == 0:
            # Get measures
                print(
                    "**%8s [Epoch %03d/%03d, Batch %05d/%05d]\t"
                    "Time: %2.1f ms | \t"
                    "Loss: %2.3f (%1.3f)\t"
                    % (descriptor, epoch, opts.epochs, batch_idx, len(loader), time_accum / (batch_idx + 1) * 1000, loss.item(), loss_accum / num_logged)
                )

                # if not is_inference:
                #     # update TensorBoard with the current snapshot of the epoch's summary
                #     summary = _generate_summary(
                #         loss_accum,
                #         flat_accuracy_accums,
                #         hdist_accums,
                #         hdist_top_accums,
                #         hdist_mistakes_accums,
                #         hprecision_accums,
                #         hmAP_accums,
                #         num_logged,
                #         norm_mistakes_accum,
                #         loss_id,
                #         dist_id,
                #     )
                #     if with_tb:
                #         _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id, epoch)

        # update TensorBoard with the total summary of the epoch
        if level_accuracy:
            if level_jsd_loss:
                if level_weight_loss:
                    if level_dissim_loss:
                        summary = _generate_summary(
                            loss_accum,
                            flat_accuracy_accums,
                            hdist_accums,
                            hdist_top_accums,
                            hdist_mistakes_accums,
                            hprecision_accums,
                            hmAP_accums,
                            num_logged,
                            norm_mistakes_accum,
                            loss_id,
                            dist_id,
                            flat_level_accuracy_accums,
                            flat_level_ce_loss_accums,
                            flat_level_jsd_loss_accums,
                            flat_level_wt_loss_accums, 
                            flat_level_dissim_loss_accums
                        )
                    else:
                        summary = _generate_summary(
                            loss_accum,
                            flat_accuracy_accums,
                            hdist_accums,
                            hdist_top_accums,
                            hdist_mistakes_accums,
                            hprecision_accums,
                            hmAP_accums,
                            num_logged,
                            norm_mistakes_accum,
                            loss_id,
                            dist_id,
                            flat_level_accuracy_accums,
                            flat_level_ce_loss_accums,
                            flat_level_jsd_loss_accums,
                            flat_level_wt_loss_accums
                        )
                else:
                    summary = _generate_summary(
                        loss_accum,
                        flat_accuracy_accums,
                        hdist_accums,
                        hdist_top_accums,
                        hdist_mistakes_accums,
                        hprecision_accums,
                        hmAP_accums,
                        num_logged,
                        norm_mistakes_accum,
                        loss_id,
                        dist_id,
                        flat_level_accuracy_accums,
                        flat_level_ce_loss_accums,
                        flat_level_jsd_loss_accums
                    )
            else:
                summary = _generate_summary(
                    loss_accum,
                    flat_accuracy_accums,
                    hdist_accums,
                    hdist_top_accums,
                    hdist_mistakes_accums,
                    hprecision_accums,
                    hmAP_accums,
                    num_logged,
                    norm_mistakes_accum,
                    loss_id,
                    dist_id,
                    flat_level_accuracy_accums,
                    flat_level_ce_loss_accums,
                )
        else:
            summary = _generate_summary(
                loss_accum,
                flat_accuracy_accums,
                hdist_accums,
                hdist_top_accums,
                hdist_mistakes_accums,
                hprecision_accums,
                hmAP_accums,
                num_logged,
                norm_mistakes_accum,
                loss_id,
                dist_id,
            )
        # if with_tb:
        #     _update_tb_from_summary(summary, tb_writer, tot_steps, loss_id, dist_id, epoch)

    # if with_tb:
        # wandb.log({"Train Accuracy": summary[accuracy_ids[0]]*100}, step=epoch)
        # wandb.log({"Train loss": summary[loss_id]}, step=epoch)
        # tb_writer.close()

    return summary, tot_steps


def _make_best_hier_similarities(classes, distances, max_dist):
    """
    For each class, create the optimal set of retrievals (used to calculate hierarchical precision @k)
    """
    distance_matrix = np.zeros([len(classes), len(classes)])
    best_hier_similarities = np.zeros([len(classes), len(classes)])

    for i in range(len(classes)):
        for j in range(len(classes)):
            distance_matrix[i, j] = distances[(classes[i], classes[j])]

    for i in range(len(classes)):
        best_hier_similarities[i, :] = 1 - np.sort(distance_matrix[i, :]) / max_dist

    return best_hier_similarities


def _generate_summary(
        loss_accum,
        flat_accuracy_accums,
        hdist_accums,
        hdist_top_accums,
        hdist_mistakes_accums,
        hprecision_accums,
        hmAP_accums,
        num_logged,
        norm_mistakes_accum,
        loss_id,
        dist_id,
        flat_level_accuracy_accums=None,
        flat_level_ce_loss_accums=None,
        flat_level_jsd_loss_accums=None,
        flat_level_weight_loss_accums=None,
        flat_level_dissim_loss_accums=None
):
    """
    Generate dictionary with epoch's summary
    """
    summary = dict()
    summary[loss_id] = loss_accum / num_logged
    # -------------------------------------------------------------------------------------------------
    summary.update({accuracy_ids[i]: flat_accuracy_accums[i]*100 / num_logged for i in range(len(topK_to_consider))})
    if flat_level_accuracy_accums:
        summary.update({level_accuracy[i]: flat_level_accuracy_accums[i]*100 / num_logged for i in range(len(level_accuracy))})
        summary.update({level_ce_loss[i]: flat_level_ce_loss_accums[i] / num_logged for i in range(len(level_ce_loss))})
        if flat_level_jsd_loss_accums:
            summary.update(
                {level_jsd_loss[i]: flat_level_jsd_loss_accums[i]/ num_logged for i in range(len(level_jsd_loss))})
            summary.update(
                {"jsd_loss": sum(flat_level_jsd_loss_accums)})
            if flat_level_weight_loss_accums:
                summary.update(
                    {level_weight_loss[i]: flat_level_weight_loss_accums[i]/ num_logged for i in range(len(level_weight_loss))})
                summary.update(
                    {"weight_loss": sum(flat_level_weight_loss_accums)})
                if flat_level_dissim_loss_accums:
                    summary.update(
                        {level_dissim_loss[i]: flat_level_dissim_loss_accums[i]/ num_logged for i in range(len(level_dissim_loss))})
                    summary.update(
                        {"dissim_loss": sum(flat_level_dissim_loss_accums)})
    if len(topK_to_consider) > 1:
        summary.update({dist_id + dist_avg_ids[i]: hdist_accums[i] / num_logged for i in range(len(topK_to_consider))})
        summary.update({dist_id + dist_top_ids[i]: hdist_top_accums[i] / num_logged for i in range(len(topK_to_consider))})
        summary.update(
            {dist_id + dist_avg_mistakes_ids[i]: hdist_mistakes_accums[i] / (norm_mistakes_accum * topK_to_consider[i]) for i in range(len(topK_to_consider))}
        )
        summary.update({dist_id + hprec_ids[i]: hprecision_accums[i] / num_logged for i in range(len(topK_to_consider))})
        summary.update({dist_id + hmAP_ids[i]: hmAP_accums[i] / num_logged for i in range(len(topK_to_consider))})
    return summary


def _update_tb_from_summary(summary, writer, steps, loss_id, dist_id, epoch=None):
    """
    Update tensorboard from the summary for the epoch
    """
    writer.add_scalar(loss_id, summary[loss_id], steps)

    for i in range(len(topK_to_consider)):
        writer.add_scalar(accuracy_ids[i], summary[accuracy_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + dist_avg_ids[i], summary[dist_id + dist_avg_ids[i]], steps)
        writer.add_scalar(dist_id + dist_top_ids[i], summary[dist_id + dist_top_ids[i]], steps)
        writer.add_scalar(dist_id + dist_avg_mistakes_ids[i], summary[dist_id + dist_avg_mistakes_ids[i]], steps)
        writer.add_scalar(dist_id + hprec_ids[i], summary[dist_id + hprec_ids[i]] * 100, steps)
        writer.add_scalar(dist_id + hmAP_ids[i], summary[dist_id + hmAP_ids[i]] * 100, steps)
