from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from config import config


class BasicFCBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicFCBlock, self).__init__()

        self.Linear = nn.Linear(input_dim, output_dim)
        self.ReLU = nn.ReLU()
        self.init_params()

    def init_params(self):
        pass

    def forward(self, x):
        x = self.Linear(x)
        x = self.ReLU(x)
        return x

    
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicConvBlock, self).__init__()
        
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.ReLU = nn.ReLU()
        self.init_params()
    
    def init_params(self):
        pass
    
    def forward(self, x):
        x = self.Conv(x)
        x = self.ReLU(x)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


# pre-defined input/output dimensions for each layer
in_dim = [config.MODEL.INPUT_DIM, config.MODEL.INPUT_DIM, config.MODEL.INPUT_DIM, config.MODEL.INPUT_DIM]
out_dim = [config.MODEL.INPUT_DIM, config.MODEL.INPUT_DIM, config.MODEL.INPUT_DIM, config.DATASET.CLASSNUM]

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()

        self.flatten = Flatten()
        self.block1 = BasicFCBlock(in_dim[0], out_dim[0])
        self.block2 = BasicFCBlock(in_dim[1], out_dim[1])
        self.block3 = BasicFCBlock(in_dim[2], out_dim[2])
        self.linear = nn.Linear(in_dim[3], out_dim[3])

    def forward(self, x):
        x = self.flatten(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.linear(x)

        return x
    
    def inspec_forward(self, x):
        interm = []
        x = self.flatten(x)
        x = self.block1(x)
        interm.append(x.clone())
        x = self.block2(x)
        interm.append(x.clone())
        x = self.block3(x)
        interm.append(x.clone())
        x = self.linear(x)
        interm.append(x.clone())
        
        return x, interm

    def l1_reg(self):
        return 0

#     def conj_reg(self):
#         loss = 0
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 loss += (list(module.parameters())[0].mm(list(module.parameters())[0].t())).triu(diagonal=1).abs().mean()
#         return loss

    def conj_reg(self):
        loss = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                norm2 = list(module.parameters())[0].norm(p=2, dim=1)
                loss += (list(module.parameters())[0].mm(list(module.parameters())[0].t()) 
                         / norm2.view(1, -1) / norm2.view(-1, 1)).triu(diagonal=1).abs().mean()
        return loss
    
    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

        
in_channels = [3, 64, 32]
out_channels = [64, 32, 16]
        
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.block1 = BasicConvBlock(in_channels[0], out_channels[0])
        self.block2 = BasicConvBlock(in_channels[1], out_channels[1])
        self.block3 = BasicConvBlock(in_channels[2], out_channels[2])
        self.flatten = Flatten()
        self.linear = nn.Linear(config.MODEL.INPUT_DIM, config.DATASET.CLASSNUM)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x

    def l1_reg(self):
        return 0

    def conj_reg(self):
        return 0
    
    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)



def create_model():

    return eval(config.MODEL.TYPE)()