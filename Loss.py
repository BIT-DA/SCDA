# -*- coding: utf-8 -*

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb
from grl import WarmStartGradientReverseLayer

def MI(outputs_target):
    batch_size = outputs_target.size(0)
    softmax_outs_t = nn.Softmax(dim=1)(outputs_target)
    avg_softmax_outs_t = torch.sum(softmax_outs_t, dim=0) / float(batch_size)
    log_avg_softmax_outs_t = torch.log(avg_softmax_outs_t + 1e-8)
    item1 = -torch.sum(avg_softmax_outs_t * log_avg_softmax_outs_t)
    item2 = -torch.sum(softmax_outs_t * torch.log(softmax_outs_t + 1e-8)) / float(batch_size)
    return item1, item2

def JS_Divergence_With_Temperature(p, q, temp_factor, get_softmax=True):
    KLDivLoss = nn.KLDivLoss(reduction='sum')
    if get_softmax:
        p_softmax_output = F.softmax(p / temp_factor)
        q_softmax_output = F.softmax(q / temp_factor)
    log_mean_softmax_output = ((p_softmax_output + q_softmax_output) / 2).log()
    return (KLDivLoss(log_mean_softmax_output, p_softmax_output) + KLDivLoss(log_mean_softmax_output, q_softmax_output)) / 2

def get_inter_pair_loss(labels_source, labels_target, outputs_source, outputs_target, temp_factor, threshold):
    loss = 0.0
    count = 0
    batch_size = len(labels_source)
    softmax_outs_target = nn.Softmax(dim=1)(outputs_target)
    for i in range(batch_size):
        for j in range(batch_size):
            if softmax_outs_target[j][labels_target[j]] < threshold:  # Threshold selection
                continue
            elif labels_source[i] == labels_target[j]:
                count += 1
                loss += JS_Divergence_With_Temperature(outputs_source[i], outputs_target[j], temp_factor)
    if count == 0:
        return loss
    else:
        return loss / count

def get_intra_pair_loss(labels, outputs, temp_factor):
    loss = 0.0
    count = 0
    batch_size = labels.size(0)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i] == labels[j]:
                count += 1
                loss += JS_Divergence_With_Temperature(outputs[i], outputs[j], temp_factor)
    if count == 0:
        return loss
    else:
        return loss / count

def get_PDD_loss(outs, labels, temp=1.0, threshold=0.8):
    batch_size = outs.size(0) // 2
    batch_source = outs[: batch_size]
    batch_target = outs[batch_size:]
    labels_s = labels[:batch_size]
    labels_t = labels[batch_size:]
    loss_pdd_s_t = get_inter_pair_loss(labels_s, labels_t, batch_source, batch_target, temp, threshold)
    loss_pdd_s_s = get_intra_pair_loss(labels_s, batch_source, temp)
    return (temp ** 2) * (loss_pdd_s_t + loss_pdd_s_s)


class AdversarialLoss_PDD(nn.Module):
    def __init__(self, classifier: nn.Module):
        super(AdversarialLoss_PDD, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.classifier = classifier

    def forward(self, f, labels_s, args):
        f_grl = self.grl(f)
        y = self.classifier(f_grl)
        y_s, y_t = y.chunk(2, dim=0)

        pseudo_label_t = y_t.argmax(1)
        labels = torch.cat((labels_s, pseudo_label_t), dim=0)
        loss_pdd = get_PDD_loss(y, labels, temp=args.temp, threshold=args.threshold)

        return loss_pdd


