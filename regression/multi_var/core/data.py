# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : data.py
#   Author      : JiangPan
#   Created date: 2019-09-04 08:42
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import numpy as np
import core.config as cfg


class Data:
    def __init__(self):
        self.data_num = cfg.DATA_NUM
        self.data_dims = cfg.DATA_DIMS
        self.data_mu = cfg.DATA_MU
        self.data_sigma = cfg.DATA_SIGMA

        self.noise_mu = cfg.NOISE_MU
        self.noise_sigma = cfg.NOISE_SIGMA
        self.weights = cfg.WEIGHTS
        self.seed = cfg.SEED

    def generate_data(self):
        np.random.seed(self.seed)
        x = np.random.normal(self.data_mu, self.data_sigma, (self.data_dims, self.data_num))
        tmp = np.ones([1, self.data_num])
        x_stack = np.vstack([tmp, x])
        noise = np.random.uniform(0, 1, self.data_num)
        y = np.transpose(self.weights).dot(x_stack) + noise
        return x_stack, y


if __name__ == '__main__':
    Data().generate_data()
