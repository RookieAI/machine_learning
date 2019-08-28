# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VSCODE
#   File name   : train.py
#   Author      : JIANGPAN
#   Created date: 2019-08-28 09:59:18
#   Description :
#
# ================================================================
import matplotlib.pyplot as plt

from core.data import Data
from core.model import LinearModel
import core.config as cfg


class LinearModelTrain:
    def __init__(self):
        self.data = Data().generate_data()
        self.model = LinearModel()

        self.num_data = cfg.NUM_DATA
        self.learning_rate = cfg.LEARNING_RATE
        self.max_step = cfg.MAX_STEP
        self.weights = cfg.INIT_W
        self.cost = []
        self.stop_condition = cfg.STOP_CONDITION

        self.global_step = 0

    def train(self):
        self.cost.append(self.model.calc_cost(self.weights))
        while True:
            self.global_step += 1
            grad = self.model.calc_grad(self.weights)
            for i in range(2):
                self.weights[i] = self.weights[i] - self.learning_rate * grad[i]
            new_cost = self.model.calc_cost(self.weights)
            self.cost.append(new_cost)
            if abs(self.cost[-1] - self.cost[-2]) < self.stop_condition or \
                    self.global_step > self.max_step:
                break
            print('steps {}, cost {}, weights {} {}'
                  .format(self.global_step, self.cost[-1],
                          self.weights[0], self.weights[1]))

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

