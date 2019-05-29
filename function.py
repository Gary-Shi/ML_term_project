from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import os

import numpy as np
import torch

from utils import AverageMeter
from utils import compute_acc
from utils import compute_loss
from utils import get_pert

from config import config

logger = logging.getLogger(__name__)

def train(train_loader, model, optimizer, epoch):
    """
    :param config: global configs
    :param train_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param optimizer: SGD or ADAM
    :param epoch: current epoch
    :return: None
    """

    # build recorders
    batch_time = AverageMeter()
    data_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()


    # switch to train mode
    model.train()

    end = time.time()
    for i, (x, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        total_batch_size = target.shape[0]
        target = target.cuda()

        # classification
        clf_score = model(x.cuda())

        # update acc
        avg_acc = compute_acc(clf_score, target)
        acc.update(avg_acc, total_batch_size)

        # compute loss
        loss = compute_loss(clf_score, target, model)

        # back prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update total clf_loss
        clf_losses.update(loss.item(), total_batch_size)

        # update time record
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % config.TRAIN.PRINT_EVERY == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=total_batch_size / batch_time.val,
                data_time=data_time, loss=clf_losses, acc=acc)
            logger.info(msg)

def valid(val_loader, model):
    """
    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: None
    """

    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()

    # switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (x, target) in enumerate(val_loader):

            total_batch_size = target.shape[0]
            target = target.cuda()

            clf_score = model(x.cuda())

            # update acc
            avg_acc = compute_acc(clf_score, target)
            acc.update(avg_acc, total_batch_size)

            # compute loss
            loss = compute_loss(clf_score, target, model)

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TEST.PRINT_EVERY == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=clf_losses, acc=acc)
                logger.info(msg)


    return acc.avg

def adv_valid(val_loader, model):
    """
    :param config: global configs
    :param val_loader: data loader
    :param model: model to be trained
    :param criterion: loss module
    :param epoch: current epoch
    :return: None
    """

    # build recorders
    batch_time = AverageMeter()
    clf_losses = AverageMeter()
    acc = AverageMeter()

    # switch to val mode
    model.eval()

    with torch.no_grad():

        end = time.time()
        for i, (x, target) in enumerate(val_loader):
            
            x += get_pert(model, x)
            #x += torch.rand(x.shape) * config.ADV.LINF_NORM

            total_batch_size = target.shape[0]
            target = target.cuda()

            clf_score = model(x.cuda())

            # update acc
            avg_acc = compute_acc(clf_score, target)
            acc.update(avg_acc, total_batch_size)

            # compute loss
            loss = compute_loss(clf_score, target, model)

            # update total clf_loss
            clf_losses.update(loss.item(), total_batch_size)

            # update time record
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.TEST.PRINT_EVERY == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=clf_losses, acc=acc)
                logger.info(msg)


    return acc.avg
