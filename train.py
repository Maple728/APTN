#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: peter.s
@project: APTN
@time: 2018/10/27 16:32
@desc:
"""
import argparse
import yaml
import tensorflow as tf

from training.model_runner import ModelRunner
from preprocess.data_loader import DataLoader
from preprocess.data_provider import DataProvider


def main(args):
    config_filename = args.config_filename
    with open(config_filename) as config_file:
        config = yaml.load(config_file)
        data_config = config['data']
        model_config = config['model']

        # load data
        # get data source
        data_loader = DataLoader(**data_config, **model_config)
        train_ds, valid_ds, test_ds = data_loader.get_three_datasource()
        # get three data provider for model input
        train_dp = DataProvider(train_ds, **data_config, **model_config)
        valid_dp = DataProvider(valid_ds, **data_config, **model_config)
        test_dp = DataProvider(test_ds, **data_config, **model_config)

        with tf.Session() as sess:
            model_runner = ModelRunner(config)
            model_runner.train_model(sess, train_dp, valid_dp, test_dp)
            preds, labels, metrics = model_runner.evaluate_model(sess, test_dp)
            print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', type=str, required=True,
                        help='Configuration filename for training or restoring the model.')
    args = parser.parse_args()
    main(args)
