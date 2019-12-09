#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/5 15:48
@desc:
"""

import os
import datetime

import numpy as np
import tensorflow as tf
import yaml

from lib.utils import get_num_trainable_params, make_config_string, create_folder
from training.lr_scheduler import LRScheduler
from models import *


class ModelRunner(object):
    def __init__(self, config):
        self._config = config
        self._train_config = config['train']

        # the folders for model and tensor board
        self._model_folder = None
        self._tfb_folder = None

        # build model
        self._model = BaseModel.generate_model_from_config(config['model'])
        self._model_saver = tf.train.Saver()

    @property
    def model(self):
        return self._model

    def _load_train_parameters(self):
        train_config = self._train_config
        # assign parameters
        epoch_num = train_config.get('epoch')
        max_epoch = train_config.get('max_epoch')

        # get lr scheduler
        lr_scheduler = LRScheduler.generate_scheduler_by_name(train_config.get('lr_scheduler'), **train_config)
        model_path = train_config.get('model_path')

        return epoch_num, max_epoch, lr_scheduler, model_path

    def _create_folders(self):
        # create model and tensorflow board folder
        time = datetime.datetime.now()
        timestamp = datetime.datetime.strftime(time, '%m%d%H%M%S')
        model_foldername = make_config_string(self._config['model']) + '-' + timestamp

        self._model_folder = create_folder(self._config['base_dir'], model_foldername, 'models')
        self._tfb_folder = create_folder(self._config['base_dir'], model_foldername, 'tfbs')

    def _save_model_with_config(self, sess):
        train_config = self._train_config
        # update model path in train config
        train_config['model_path'] = os.path.join(self._model_folder, 'model-' + str(train_config['epoch']))

        # save model
        self._model_saver.save(sess, train_config['model_path'])
        # save config to yaml file
        config_path = os.path.join(self._model_folder, 'config-' + str(train_config['epoch']) + '.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f)

    def _update_train_config(self, lr, epoch):
        train_config = self._train_config
        train_config['lr'] = lr
        train_config['epoch'] = epoch

    def _run_epoch(self, sess, data_provider, lr, is_train):
        model = self._model
        if is_train:
            run_func = model.train
        else:
            run_func = model.predict
        loss_list = []
        pred_list = []
        real_list = []
        for batch_data in data_provider.iterate_batch_data(True):
            loss, pred, real = run_func(sess, batch_data, lr=lr)

            loss_list.append(loss)
            pred_list.append(pred)
            real_list.append(real)

        # shape -> [n_items, horizon, D]
        epoch_preds = np.concatenate(pred_list, axis=0)
        epoch_reals = np.concatenate(real_list, axis=0)

        epoch_avg_loss = np.mean(loss_list)
        # inverse scaling data
        epoch_preds = data_provider.data_source.scaler.inverse_scaling(epoch_preds)
        epoch_reals = data_provider.data_source.scaler.inverse_scaling(epoch_reals)

        return epoch_avg_loss, epoch_preds, epoch_reals

    def train_model(self, sess,
                    train_data_provider, valid_data_provider, test_data_provider=None):

        epoch_num, max_epoch, lr_scheduler, model_path = self._load_train_parameters()

        # new model or existed model
        if model_path is not None:
            # restore model if existed
            self._model_saver.restore(sess, model_path)
            epoch_num += 1
            print('Restore model from', model_path)
            # set model folder
            self._model_folder = os.path.dirname(os.path.dirname(model_path))
        else:
            # create and set folders for new model
            self._create_folders()
            # initialize variables
            sess.run([tf.global_variables_initializer()])

        print('----------Trainable parameter count:', get_num_trainable_params(), 'in model', self._model_folder)
        print('Training start ...')
        best_valid_loss = float('inf')
        lr = lr_scheduler.get_lr()
        while lr > 0 and epoch_num <= max_epoch:
            print('epoch:', epoch_num, 'lr:', lr)
            # train
            self._run_epoch(sess, train_data_provider,
                            lr, is_train=True)

            # valid
            loss, _, _ = self._run_epoch(sess, valid_data_provider,
                                         lr, is_train=False)

            # update after train and valid
            # update lr
            lr = lr_scheduler.update_lr(loss=loss, epoch_num=epoch_num)
            # update train_config
            self._update_train_config(lr, epoch_num)

            if loss < best_valid_loss:
                best_valid_loss = loss

                # save best model
                self._save_model_with_config(sess)

            # test
            loss, preds, labels = self._run_epoch(sess, test_data_provider,
                                                  lr, is_train=False)
            metrics = test_data_provider.data_source.get_metrics(preds, labels)
            str_metrics = str(metrics)
            print('Epoch', epoch_num, 'Test Loss:', loss, str_metrics)

            epoch_num += 1
        print('Training Finished!')

    def evaluate_model(self, sess, data_provider):
        self.restore_model(sess)
        loss, preds, labels = self._run_epoch(sess, data_provider,
                                              lr=0, is_train=False)
        metrics = data_provider.data_source.get_metrics(preds, labels)
        return preds, labels, metrics

    def restore_model(self, sess):
        train_config = self._train_config
        model_path = train_config['model_path']
        self._model_saver.restore(sess, model_path)
