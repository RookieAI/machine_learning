# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : yolov3.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 10:47:03
#   Description :
#
# ================================================================
import core.config as cfg
from core.data import Data


class LinearModel:
    def __init__(self):
        self.data = Data().generate_data()
        self.num_data = cfg.NUM_DATA

    def calc_cost(self, weights):
        cost = 0
        for i in range(self.num_data):
            cost += (weights[0] + weights[1]*self.data[0, i] - self.data[1, i])**2
            cost /= 2*self.num_data
        return cost

    def calc_grad(self, weights):
        delta_w = [0, 0]
        for i in range(self.num_data):
            delta_w[0] += weights[0] + weights[1]*self.data[0, i] - self.data[1, i]
            delta_w[1] += (weights[0] + weights[1]*self.data[0, i] - self.data[1, i])*self.data[0, i]
        delta_w[0] /= self.num_data
        delta_w[1] /= self.num_data
        return delta_w



