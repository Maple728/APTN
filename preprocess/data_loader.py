#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 16:42
@desc:
"""

import numpy as np
from preprocess.scaler import StandZeroMaxScaler, MinMaxScaler
from preprocess.data_source import DataSource
from lib.utils import get_metric_functions


def get_metrics_func(metric_names):
    metric_functions = get_metric_functions(metric_names)

    def metrics(preds, labels):
        res = dict()
        for metric_name, metric_func in zip(metric_names, metric_functions):
            res[metric_name] = metric_func(preds, labels)
        return res

    return metrics


class DataLoader(object):

    def __init__(self, data_name, data_filename, metrics, cache_dir,
                 T_skip, n, T,
                 **kwargs):
        self._data_name = data_name
        self._data_filename = data_filename
        self._post_len = T_skip * n + T
        self._cache_dir = cache_dir
        self._metrics = get_metrics_func(metrics)

    def get_three_datasource(self):
        # [length, num_of_vertices (307), 3 (traffic flow, occupancy, speed)]
        # shape -> [length, 307]
        records = np.load(self._data_filename)['data'][:, :, 0]

        # split train, valid and test data set
        post_len = self._post_len
        # for same with ASTGCN
        lens = len(records)
        train_split_idx = int(lens * 0.6)
        valid_split_idx = int(lens * 0.8)
        train_records = records[:train_split_idx]
        valid_records = records[train_split_idx - post_len: valid_split_idx]
        test_records = records[valid_split_idx - post_len:]

        # scaling target series
        tgt_scaler = MinMaxScaler()
        # shape -> [length, x_dim]
        train_tgts = tgt_scaler.fit_scaling(train_records)
        valid_tgts = tgt_scaler.scaling(valid_records)
        test_tgts = tgt_scaler.scaling(test_records)

        def get_retrieve_data_callback(data):
            def func():
                yield data

            return func

        train_ds = DataSource(self._data_name + '_train',
                              metric_callback=self._metrics,
                              retrieve_data_callback=get_retrieve_data_callback([train_tgts, train_tgts]),
                              scaler=tgt_scaler, cache_dir=self._cache_dir)
        valid_ds = DataSource(self._data_name + '_valid',
                              metric_callback=self._metrics,
                              retrieve_data_callback=get_retrieve_data_callback([valid_tgts, valid_tgts]),
                              scaler=tgt_scaler, cache_dir=self._cache_dir)
        test_ds = DataSource(self._data_name + '_test',
                             metric_callback=self._metrics,
                             retrieve_data_callback=get_retrieve_data_callback([test_tgts, test_tgts]),
                             scaler=tgt_scaler, cache_dir=self._cache_dir)
        return train_ds, valid_ds, test_ds
