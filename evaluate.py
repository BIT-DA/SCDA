# -*- coding: utf-8 -*

import random
import time
import warnings
import sys
import argparse
import copy
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
import os.path as osp
import gc

from network import ImageClassifier
import backbone as BackboneNetwork
from utils import ContinuousDataloader
from transforms import ResizeImage
from lr_scheduler import LrScheduler
from data_list import ImageList
from Loss import *


def main(args: argparse.Namespace):
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    val_tranform = transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    test_dataset = ImageList(open(args.t_test_path).readlines(), transform=val_tranform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=64)
    print("transfer model path: {}".format(args.weight_path))
    print("test dataset path: {}".format(args.t_test_path))

    # load model
    print("backbone '{}'".format(args.arch))
    backbone = BackboneNetwork.__dict__[args.arch](pretrained=True)
    if args.dset == "office":
        num_classes = 31
    elif args.dset == "office-home":
        num_classes = 65
    elif args.dset == "domainnet":
        num_classes = 345
    checkpoint = torch.load(args.weight_path)
    classifier = ImageClassifier(backbone, num_classes).cuda()
    classifier.load_state_dict(checkpoint)

    # evaluate on test set
    acc = test(test_loader, classifier)
    print("test_acc = {:.3f}".format(acc))

def test(val_loader, model):
    # switch to evaluate mode
    model.eval()
    start_test = True
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()
            # get logit outputs
            output, _ = model(images)
            if start_test:
                all_output = output.float()
                all_label = target.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, output.float()), 0)
                all_label = torch.cat((all_label, target.float()), 0)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        accuracy = accuracy * 100.0
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Semantic Concentration for Domain Adaptation')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'resnet101'])
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--dset', type=str, default='office', choices=['office', 'office-home', 'domainnet'], help="The dataset used")
    parser.add_argument('--t_test_path', type=str, default='data/list/office/webcam_31.txt', help="The target test dataset path list")
    parser.add_argument('--weight_path', type=str, default='checkpoint/amazon-webcam_model.pth.tar', help="The path of model weight")
    parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 32)')

    args = parser.parse_args()
    main(args)
