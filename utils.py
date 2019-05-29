from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path
import cv2
from PIL import Image
import json

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from config import config


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def create_optimizer(config, model):
    """
    create an SGD or ADAM optimizer

    :param config: global configs
    :param model: the model to be trained
    :return: an SGD or ADAM optimizer
    """
    optimizer = None

    if config.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.TRAIN.LR
        )

    return optimizer

def create_logger(phase='train'):
    """
    create a logger for experiment record
    To use a logger to publish message m, just run logger.info(m)

    :param cfg: global config
    :param phase: train or val
    :return: a logger
    """
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(config.DATASET.DATASET, time_str, phase)
    final_log_file = Path(config.OUTPUT_DIR) / log_file
    log_format = '%(asctime)-15s: %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=log_format)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

def compute_acc(score, target):
    """
    compute average accuracy
    :param score: classification score
    :param target: target label
    :return: avg_acc
    """
    batch_size = target.shape[0]
    avg_acc = (score.argmax(dim = 1) == target).sum().item() / batch_size
    return avg_acc

def compute_loss(score, target, model):

    loss = F.cross_entropy(score, target)

    # l1 regularization
    if config.TRAIN.IF_L1REG:
        loss += config.TRAIN.L1REG * model.module.l1_reg()

    # conjugate regularization
    if config.TRAIN.IF_CONJREG:
        loss += config.TRAIN.CONJREG * model.module.conj_reg()

    return loss

def FGSM(model, input, fakelabel = None):
    """
    :return: adversarial perturbation
    """
    
    # choose a target class
    if fakelabel == None:
        fakelabel = np.random.randint(config.DATASET.CLASSNUM, size=(input.shape[0]))
    
    with torch.enable_grad():
        input = input.cuda()
        input.requires_grad = True
        score = model(input)

        loss = F.cross_entropy(score, torch.cuda.LongTensor(fakelabel))
        grad = torch.autograd.grad(loss, input)[0]

        pert = -torch.sign(grad) * config.ADV.LINF_NORM

        return pert.detach().cpu()

def get_pert(model, input, target_label = None):
    """
    compute perturbation
    """
    return eval(config.ADV.TYPE)(model, input, target_label)