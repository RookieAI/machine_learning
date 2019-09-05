# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : JiangPan
#   Created date: 2019-09-05 09:36
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import matplotlib.pyplot as plt

import core.config as cfg
from core.data import Data
from core.model import LinearModel


class LinearModelTrain:
    def __init__(self):
        self.data, self.label = Data().generate_data()
        self.model = LinearModel()

        self.learning_rate = cfg.LEARNING_RATE
        self.max_step = cfg.MAX_STEP
        self.stop_condition = cfg.STOP_CONDITION
        self.global_step = cfg.GLOBAL_STEP

        self.cost = []
        self.weights = cfg.INIT_WEIGHTS

    def train(self):
        self.cost.append(self.model.calc_cost(self.weights))
        while True:
            self.global_step += 1
            grad = self.model.calc_grad(self.weights)
            self.weights -= self.learning_rate*grad
            new_cost = self.model.calc_cost(self.weights)
            self.cost.append(new_cost)
            if abs(self.cost[-1] - self.cost[-2]) < self.stop_condition or \
                    self.global_step > self.max_step:
                break
            print('steps {}, cost {}, weights {}'.format(
                self.global_step, self.cost[-1], self.weights))

    def visualize_cost(self):
        x = range(len(self.cost))
        plt.figure(1)
        plt.plot(x, self.cost, linewidth=2, color='blue')
        plt.title('cost-step')
        plt.xlabel('step')
        plt.ylabel('cost')
        plt.grid()
        plt.show()


if __name__ == '__main__':
    train = LinearModelTrain()
    train.train()
    train.visualize_cost()

