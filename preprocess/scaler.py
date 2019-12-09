#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2019/11/6 16:55
@desc:
"""

import numpy as np
from abc import abstractmethod


class Scaler(object):

    @abstractmethod
    def fit(self, records):
        pass

    @abstractmethod
    def fit_scaling(self, records):
        pass

    @abstractmethod
    def scaling(self, records):
        pass

    @abstractmethod
    def inverse_scaling(self, scaled_records):
        pass


class StandZeroMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._max_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._max_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return records / (self._max_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._max_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return scaled_records * (self._max_val + self._epsilon)


class MinMaxScaler(Scaler):
    def __init__(self, epsilon=1e-8):
        self._min_val = None
        self._max_val = None
        self._epsilon = epsilon

    def fit(self, records):
        if self._min_val is not None:
            raise RuntimeError('Try to fit a fitted scaler!')
        self._max_val = np.max(records, axis=0)
        self._min_val = np.min(records, axis=0)

    def fit_scaling(self, records):
        self.fit(records)
        return self.scaling(records)

    def scaling(self, records):
        if self._min_val is None:
            raise RuntimeError('Try to scaling records using a uninitialized scaler!')
        return (records - self._min_val) / (self._max_val - self._min_val + self._epsilon)

    def inverse_scaling(self, scaled_records):
        if self._min_val is None:
            raise RuntimeError('Try to inverse_scaling records using a uninitialized scaler!')
        return (scaled_records * (self._max_val - self._min_val + self._epsilon)) + self._min_val


