from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import csv

import numpy as np
from easydict import EasyDict as edict

# global configuration

config = edict()

config.MODE = 'train'
config.GPUS = '0, 5, 6, 7, 8'
config.GPU_NUM = 5  # number of gpus in config.GPUS
config.WORKERS = 4

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# data-related configs

config.DATASET = edict()

config.DATASET.DATASET = 'CIFAR10'
config.DATASET.ROOT = '/m/shibf/dataset/cifar10'
config.DATASET.CLASSNUM = 10
config.DATASET.IMAGESIZE = [32, 32]

# model-related configs

config.MODEL = edict()

config.MODEL.TYPE = 'ConvNet'  # FCNet / ConvNet
config.MODEL.INPUT_DIM = 3 * 32 * 32

# adversary-related configs

config.ADV = edict()
config.ADV.TYPE = 'FGSM'
config.ADV.LINF_NORM = 3e-2

# training configs

config.TRAIN = edict()

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 120

config.TRAIN.BATCH_SIZE = 32  # batch_size per gpu

config.TRAIN.IF_L1REG = False  # l1 regularization
config.TRAIN.IF_CONJREG = False  # conjugate regularization
config.TRAIN.IF_SPECREG = True
config.TRAIN.IF_ACTIVATION_SPECREG = True  # spectral reg with activation considered
config.TRAIN.L1REG = 5e-5
config.TRAIN.CONJREG = 1
config.TRAIN.SPECREG = 0.5


config.TRAIN.LR = 0.001
config.TRAIN.LR_DECAY_RATE = 0.5
config.TRAIN.LR_MILESTONES = [30, 60, 90]  # at which epoch lr decays

config.TRAIN.OPTIMIZER = 'sgd'  # sgd / adam
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False

config.TRAIN.PRINT_EVERY = 1

# testing configs

config.TEST = edict()

config.TEST.STATE_DICT = 'checkpoint_0.7049.pth'
config.TEST.TEST_EVERY = 5
config.TEST.PRINT_EVERY = 1

def extra():
    config.OUTPUT_DIR = os.path.join('experiments/', config.DATASET.DATASET, config.MODE)
    config.TEST.STATE_DICT = os.path.join(config.OUTPUT_DIR, config.TEST.STATE_DICT)
    
    if config.DATASET.DATASET == 'CIFAR10':
        config.DATASET.CLASSNUM = 10
        config.DATASET.IMAGESIZE = [32, 32]
    
    if config.MODEL.TYPE == 'FCNet':
        config.MODEL.INPUT_DIM = 3 * 32 * 32
    elif config.MODEL.TYPE == 'ConvNet':
        config.MODEL.INPUT_DIM = 16 * 4 * 4

