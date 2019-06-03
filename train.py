#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from config import config
from config import extra
from function import train
from function import valid
from dataset import create_dataset
from models import create_model
from utils import create_optimizer
from utils import create_logger


# In[2]:


config.MODE = 'train'
extra()

# create a logger
logger = create_logger('train')

# logging configurations
logger.info(pprint.pformat(config))

# cudnn related setting
cudnn.benchmark = config.CUDNN.BENCHMARK
torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
torch.backends.cudnn.enabled = config.CUDNN.ENABLED


# In[3]:


# create a model
os.environ["CUDA_VISIBLE_DEVICES"] = config.GPUS
gpus = [int(i) for i in config.GPUS.split(',')]
gpus = range(gpus.__len__())
model = create_model()

model = model.cuda(gpus[0])
model = torch.nn.DataParallel(model, device_ids=gpus)


# In[4]:


# create an optimizer
optimizer = create_optimizer(config, model)

# create a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
optimizer, config.TRAIN.LR_MILESTONES,
config.TRAIN.LR_DECAY_RATE
)

# get dataset
train_dataset, test_dataset, train_loader, test_loader = create_dataset()


# In[ ]:


#training and validating
best_perf = 0
for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
    lr_scheduler.step()

    # train for one epoch
    train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    if (epoch + 1) % config.TEST.TEST_EVERY == 0:
        perf_indicator = valid(test_loader, model)

        if perf_indicator > best_perf:
            logger.info("=> saving checkpoint into {}".format(os.path.join(config.OUTPUT_DIR, 'checkpoint_{}.pth'.format(perf_indicator))))
            best_perf = perf_indicator
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, 'checkpoint_{}.pth'.format(perf_indicator)))

# save the final model
logger.info("=> saving final model into {}".format(
    os.path.join(config.OUTPUT_DIR, 'model_{}.pth'.format(perf_indicator))
))
torch.save(model.state_dict(),
           os.path.join(config.OUTPUT_DIR, 'model_{}.pth'.format(perf_indicator)))


# In[ ]:




