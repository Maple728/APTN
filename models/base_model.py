#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/21 21:49
@desc:
"""
from abc import abstractmethod


class BaseModel(object):

    @abstractmethod
    def train(self, sess, batch_data, **kwargs):
        pass

    @abstractmethod
    def predict(self, sess, batch_data, **kwargs):
        pass

    @staticmethod
    def generate_model_from_config(model_config):
        model_name = model_config.get('name')

        for subclass in BaseModel.__subclasses__():
            if subclass.__name__ == model_name:
                return subclass(model_config)

        raise RuntimeError('No model named ' + model_name)
