# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : model.py
#   Author      : JiangPan
#   Created date: 2019-09-04 10:47
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import numpy as np

import core.config as cfg
from core.data import Data


class LinearModel:
    def __init__(self):
        self.data, self.label = Data().generate_data()
        print((self.label ** 2).sum() / 1000)
        self.data_num = cfg.DATA_NUM

    def calc_cost(self, weights):
        tmp = np.transpose(weights).dot(self.data)
        cost = ((tmp-self.label) ** 2).sum() / (2*self.data_num)
        return cost

    def calc_grad(self, weights):
        grad = []
        for j in range(self.data.shape[0]):
            tmp = np.transpose(weights).dot(self.data) - self.label
            grad.append((tmp * self.data[j, :]).sum() / self.data_num)
        return np.array(grad)


if __name__ == '__main__':
    LinearModel()
