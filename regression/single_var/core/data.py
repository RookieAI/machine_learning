# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VSCODE
#   File name   : data.py
#   Author      : JIANGPAN
#   Created date: 2019-08-28 09:34:23
#   Description :
#
# ================================================================
import random

import numpy as np
import matplotlib.pyplot as plt

import core.config as cfg


class Data:
    def __init__(self):
        self.num_data = cfg.NUM_DATA
        self.x_max = cfg.X_MAX
        self.x_min = cfg.X_MIN
        self.noise_mu = cfg.NOISE_MU
        self.noise_sigma = cfg.NOISE_SIGMA
        self.set_weights = cfg.SET_WEIGHTS

    def generate_data(self):
        x = [random.uniform(self.x_min, self.x_max)
             for i in range(self.num_data)]
        noise = [random.gauss(self.noise_mu, self.noise_sigma)
                 for j in range(self.num_data)]
        y = [(self.set_weights[0] + x[i] * self.set_weights[1] + noise[i])
             for i in range(self.num_data)]
        return np.array([x, y])


if __name__ == '__main__':
    a = Data().generate_data()
    plt.scatter(a[0, :], a[1, :])
    plt.show()
