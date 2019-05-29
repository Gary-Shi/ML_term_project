from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import torchvision
import torchvision.transforms as transforms

from config import config


def get_CIFAR10():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    trainset = torchvision.datasets.CIFAR10(root=config.DATASET.ROOT, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=config.DATASET.ROOT, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.TRAIN.BATCH_SIZE * config.GPU_NUM,
                                              shuffle=True,
                                              num_workers=config.WORKERS)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.TRAIN.BATCH_SIZE * config.GPU_NUM,
                                             shuffle=False,
                                             num_workers=config.WORKERS)

    return trainset, testset, trainloader, testloader

def create_dataset():
    """
    :return: trainset, testset, trainloader, testloader
    """
    return eval('get_' + config.DATASET.DATASET)()