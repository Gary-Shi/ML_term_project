from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import config
from utils import power_iteration_init
from utils import conv_to_fc


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
        
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
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
        self.init_params()

        self.v = []  # v for power iteration
        self.init_v()

        self.activation = [None for i in range(3)]  # activation map for each ReLU
      
    def init_params(self):
        pass
    
    def init_v(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _, v_temp = power_iteration_init(list(module.parameters())[0].data, 'Linear')
                self.v.append(v_temp)
    
    def forward(self, x):
        activation = [None for i in range(4)]
        x = self.flatten(x)
        x = self.block1(x)
        activation[0] = (x > 1e-7).type(torch.float).mean(dim=0).view(-1).detach().cpu()
        x = self.block2(x)
        activation[1] = (x > 1e-7).type(torch.float).mean(dim=0).view(-1).detach().cpu()
        x = self.block3(x)
        activation[2] = (x > 1e-7).type(torch.float).mean(dim=0).view(-1).detach().cpu()
        x = self.linear(x)

        return x, activation
    
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
        loss = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                loss += (list(module.parameters())[0]).norm(p=1)

        return loss

    def conj_reg(self):
        loss = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                loss += (list(module.parameters())[0].mm(list(module.parameters())[0].t())).triu(diagonal=1).abs().mean()
        return loss

#     def conj_reg(self):
#         loss = 0
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 norm2 = list(module.parameters())[0].norm(p=2, dim=1)
#                 loss += (list(module.parameters())[0].mm(list(module.parameters())[0].t()) 
#                          / norm2.view(1, -1) / norm2.view(-1, 1)).triu(diagonal=1).abs().mean()
#         return loss

    def spec_reg(self):
        i = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                W = list(module.parameters())[0].data.detach().cpu()
                v_temp = self.v[i].clone()
                u = W.mm(v_temp)
                u = u / u[:, 0].norm()
                v_temp = W.t().mm(u)
                
                # update W
                if config.TRAIN.IF_ACTIVATION_SPECREG and self.activation[i] is not None:
                    list(module.parameters())[0].data -= \
                        (self.activation[i].view(-1, 1) *
                         (config.TRAIN.LR * config.TRAIN.SPECREG / u[:, 0].norm() * (u.mm(v_temp.t())))).cuda()
                else:
                    list(module.parameters())[0].data -= \
                        (config.TRAIN.LR * config.TRAIN.SPECREG / u[:, 0].norm() * (u.mm(v_temp.t()))).cuda()
                
                # update v
                self.v[i] = v_temp
                
                i += 1
    
    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)

        
in_channels = [3, 64, 32]
out_channels = [64, 32, 16]
kernel_size = [3, 3, 3]
padding = [1, 1, 1]
        
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.input_size = []
        for i in range(3):
            self.input_size.append([int(config.DATASET.IMAGESIZE[0] / (2 ** i)), int(config.DATASET.IMAGESIZE[1] / (2 ** i))])
        
        self.block1 = BasicConvBlock(in_channels[0], out_channels[0])
        self.max_pl1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = BasicConvBlock(in_channels[1], out_channels[1])
        self.max_pl2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = BasicConvBlock(in_channels[2], out_channels[2])
        self.max_pl3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = Flatten()
        self.linear = nn.Linear(config.MODEL.INPUT_DIM, config.DATASET.CLASSNUM)
        
        self.init_params()
        self.v = []  # v for power iteration
        self.init_v()
    
    def init_params(self):
        pass
    
    def init_v(self):
        i = 0
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                W = list(module.parameters())[0].data.detach().cpu()
                _, v_temp = power_iteration_init(W, 'Conv', input_size = self.input_size[i], stride=1, padding=1)
                i += 1
                self.v.append(v_temp)
            elif isinstance(module, nn.Linear):
                W = list(module.parameters())[0].data.detach().cpu()
                _, v_temp = power_iteration_init(W, 'Linear')
                self.v.append(v_temp)
                    
    
    def forward(self, x):
        x = self.block1(x)
        x = self.max_pl1(x)
        x = self.block2(x)
        x = self.max_pl2(x)
        x = self.block3(x)
        x = self.max_pl3(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x, None
    
    def inspec_forward(self, x):
        interm = []
        x = self.block1(x)
        interm.append(x.clone())
        x = self.block2(x)
        interm.append(x.clone())
        x = self.block3(x)
        interm.append(x.clone())
        x = self.flatten(x)
        x = self.linear(x)
        interm.append(x.clone())
        
        return x, interm

    def l1_reg(self):
        return 0

    def conj_reg(self):
        return 0
    
    def spec_reg(self):
        i = 0  # which layer
        j = 0  # which conv layer
        for module in self.modules():
            # linear layer
            if isinstance(module, nn.Linear):
                W = list(module.parameters())[0].data.detach().cpu()
                
                v_temp = self.v[i].clone()
                u = W.mm(v_temp)
                u = u / u[:, 0].norm()
                v_temp = W.t().mm(u)
                
                # update W
                list(module.parameters())[0].data -= (config.TRAIN.LR * config.TRAIN.SPECREG / u[:, 0].norm() * (u.mm(v_temp.t()))).reshape(list(module.parameters())[0].data.shape).cuda()
                
                # update v
                self.v[i] = v_temp
                
                i += 1
            # conv layer
            elif isinstance(module, nn.Conv2d):
                W = list(module.parameters())[0].data.detach().cpu()
                W_prime = torch.flip(W.transpose(0, 1), dims=[2, 3])
                
                v_temp = self.v[i].clone()
                
                u = F.conv2d(torch.stack([v_temp]), W, stride=1, padding=padding[j])
                u = u / u.view(-1).norm()
                v_temp = F.conv2d(u, W_prime, stride=1, padding=padding[j])[0]
                
                # update W using Monte Carlo
                which_output_channel = np.array(range(out_channels[j]))
                which_row = np.random.randint(int((kernel_size[j] - 1) / 2), int(self.input_size[j][0] - (kernel_size[j] - 1) / 2 - 1))
                which_column = np.random.randint(int((kernel_size[j] - 1) / 2), int(self.input_size[j][0] - (kernel_size[j] - 1) / 2 - 1))
                update = u.view(-1)[self.input_size[j][0] ** 2 * which_output_channel + self.input_size[j][0] * which_row + which_column].view(-1, 1) * v_temp.view(1, -1)
                update = update.view(out_channels[j], in_channels[j], self.input_size[j][0], self.input_size[j][1])
                update = update[:, :, which_row - ((kernel_size[j] - 1) >> 1) : which_row + ((kernel_size[j] - 1) >> 1) + 1, 
                                which_column - ((kernel_size[j] - 1) >> 1) : which_column + ((kernel_size[j] - 1) >> 1) + 1]

                #print(update[0, 0])
                list(module.parameters())[0].data -= (config.TRAIN.LR * config.TRAIN.SPECREG / u.view(-1).norm() * update).cuda()
                
                # update v
                self.v[i] = v_temp
                
                i += 1
                j += 1
    
    def my_load_state_dict(self, state_dict_old, strict=True):
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            #if 'time_conv' in key or 'clf_head' in key:
            state_dict[key.replace('module.', '')] = state_dict_old[key]

        self.load_state_dict(state_dict, strict=strict)



def create_model():

    return eval(config.MODEL.TYPE)()