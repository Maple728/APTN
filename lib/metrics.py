#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np
import tensorflow as tf


# ----------------- loss function for tensorflow --------------------
def mse_tf(preds, labels):
    return tf.losses.mean_squared_error(labels, preds)


def mae_tf(preds, labels):
    return tf.losses.absolute_difference(labels, preds)


# ---------------- metric functions for numpy -----------------------
def rmse_np(preds, labels):
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    return rmse


def mae_np(preds, labels):
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    mae = np.mean(np.abs(preds - labels))
    return mae


def mape_np(preds, labels, threshold=20):
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    # zero mask
    mask = labels > threshold
    preds = preds[mask]
    labels = labels[mask]

    mape = np.mean(np.abs(preds - labels) / labels)
    return mape


def horizon_rmse_np(preds, labels):
    rmses = []
    horizon = preds.shape[1]
    for i in range(horizon):
        rmses.append(rmse_np(preds[:, i], labels[:, i]))
    return rmses


def horizon_mae_np(preds, labels):
    maes = []
    horizon = preds.shape[1]
    for i in range(horizon):
        maes.append(mae_np(preds[:, i], labels[:, i]))
    return maes


def MdAE_np(preds, labels):
    """
    Median Absolute Error
    :param preds:
    :param labels:
    :return:
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])
    return np.median(np.abs(preds - labels))


def mase_np(preds, labels, benchmark_mae):
    """
    Mean Absolute Scaled Error
    :param preds:
    :param labels:
    :param benchmark_mae:
    :return:
    """
    preds = np.reshape(preds, [-1])
    labels = np.reshape(labels, [-1])

    mase = np.mean(np.abs(preds - labels) / benchmark_mae)
    return mase
