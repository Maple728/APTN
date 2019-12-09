#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/27 22:38
@desc:
"""
from abc import abstractmethod


class LRScheduler(object):

    @abstractmethod
    def update_lr(self, **kwargs):
        pass

    @abstractmethod
    def get_lr(self):
        pass

    @staticmethod
    def generate_scheduler_by_name(scheduler_name, **kwargs):
        for subclass in LRScheduler.__subclasses__():
            if subclass.__name__ == scheduler_name:
                return subclass(**kwargs)

        return ConstantLRScheduler(**kwargs)


class ConstantLRScheduler(LRScheduler):

    def __init__(self, lr, **kwargs):
        self.lr = lr

    def update_lr(self, **kwargs):
        pass

    def get_lr(self):
        return self.lr


class PlateauLRScheduler(LRScheduler):

    def __init__(self, lr, epoch=0, patience=20, min_lr=0.00001, lr_decay_rate=0.4, **kwargs):
        self.lr = lr
        self.patience = patience
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay_rate

        self.pre_best_epoch_num = epoch
        self.pre_best_loss = float('inf')

        if self.lr <= self.min_lr:
            self.lr = self.min_lr
            self.is_min_lr = True
        else:
            self.is_min_lr = False

    def update_lr(self, loss, epoch_num, **kwargs):
        if loss < self.pre_best_loss:
            # update best info
            self.pre_best_loss = loss
            self.pre_best_epoch_num = epoch_num
        else:
            epochs = epoch_num - self.pre_best_epoch_num

            if epochs % self.patience == 0:
                # reduce lr
                self.lr = self.lr * self.lr_decay_rate
                if self.lr < self.min_lr:
                    self.lr = 0.0
        return self.get_lr()

    def get_lr(self):
        return self.lr
