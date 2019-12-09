#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 14:58
@desc:
"""
import os
import numpy as np

from lib.utils import create_folder


class DataSource(object):
    def __init__(self, data_name,
                 retrieve_data_callback,
                 metric_callback,
                 scaler=None,
                 cache_dir=None):
        """
        :param data_name:
        :param retrieve_data_callback:
        :param cache_dir:
        """
        self._data_name = data_name

        self._metric_callback = metric_callback
        self._retrieve_data_callback = retrieve_data_callback

        self._scaler = scaler

        # create cache dir
        self.is_cached = False
        if cache_dir is not None and cache_dir != 'None':
            self._use_cache = True
            self.cache_path = create_folder(cache_dir, self._data_name)
        else:
            self._use_cache = False

    @property
    def scaler(self):
        return self._scaler

    def get_metrics(self, preds, labels):
        return self._metric_callback(preds, labels)

    def load_partition_data(self):
        """Iterate data from callback function or disk cache. The data is an array containing records, whose first dimension
        is the number of records.
        :return: [feat_arr, target_arr]
        """
        if self.is_cached:
            for filename in os.listdir(self.cache_path):
                npzfile = np.load(os.path.join(self.cache_path, filename))
                yield (npzfile['feat'], npzfile['target'])
        else:
            for i, record_data in enumerate(self._retrieve_data_callback()):
                if self._use_cache:
                    # cache data into disk
                    np.savez(os.path.join(self.cache_path, str(i) + '.npz'),
                             feat=record_data[0],
                             target=record_data[1])
                yield record_data
            if self._use_cache:
                self.is_cached = True
