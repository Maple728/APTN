#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 15:43
@desc:
"""
import os
from functools import reduce
from operator import mul

from lib.metrics import *


def tensordot(tensor_a, tensor_b):
    """ Tensor dot function. The last dimension of tensor_a and the first dimension of tensor_b must be the same.
    :param tensor_a:
    :param tensor_b:
    :return: the result of tensor_a tensor dot tensor_b.
    """
    last_idx_a = len(tensor_a.get_shape().as_list()) - 1
    return tf.tensordot(tensor_a, tensor_b, [[last_idx_a], [0]])


def get_tf_loss_function(loss_name):
    return eval(loss_name + '_tf')


def get_metric_functions(metric_name_list):
    metric_functions = []
    for metric_name in metric_name_list:
        metric_functions.append(eval(metric_name + '_np'))
    return metric_functions


def get_num_trainable_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def set_random_seed(seed=9899):
    # set random seed for numpy and tensorflow
    np.random.seed(seed)
    tf.set_random_seed(seed)


def make_config_string(config):
    key_len = 4
    str_config = ''
    for k, v in config.items():
        str_config += k[:key_len] + '-' + str(v) + '_'
    return str_config[:-1]


def window_rolling(origin_data, window_size):
    """Rolling data over 0-dim.
    :param origin_data: ndarray of [n_records, ...]
    :param window_size: window_size
    :return: [n_records - window_size + 1, window_size, ...]
    """
    n_records = len(origin_data)
    if n_records < window_size:
        return None

    data = origin_data[:, None]
    all_data = []
    for i in range(window_size):
        all_data.append(data[i: (n_records - window_size + i + 1)])

    # shape -> [n_records - window_size + 1, window_size, ...]
    rolling_data = np.hstack(all_data)

    return rolling_data


def yield2batch_data(arrs, batch_size, keep_remainder=True):
    """Iterate the array of arrs over 0-dim to get batch data.
    :param arrs: a list of [n_items, ...]
    :param batch_size:
    :param keep_remainder: Discard the remainder if False, otherwise keep it.
    :return:
    """
    if arrs is None or len(arrs) == 0:
        return

    idx = 0
    n_items = len(arrs[0])
    while idx < n_items:
        if idx + batch_size > n_items and keep_remainder is False:
            return
        next_idx = min(idx + batch_size, n_items)
        yield [arr[idx: next_idx] for arr in arrs]

        # update idx
        idx = next_idx


def create_folder(*args):
    """Create path if the folder doesn't exist.
    :param args:
    :return: The folder's path depends on operating system.
    """
    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
