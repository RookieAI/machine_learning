# !/usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : config.py
#   Author      : JiangPan
#   Created date: 2019-09-04 08:12
#   Description :
#   Contact     : jsyxjp@163.com
#
# ================================================================
import numpy as np

# Data Config
DATA_NUM = 500
DATA_DIMS = 5
DATA_MU = 0
DATA_SIGMA = 1
NOISE_MU = 0
NOISE_SIGMA = 1
WEIGHTS = np.array([1, 1, 2, 3, 2, 1])
SEED = 1234567890

# Model Config
LEARNING_RATE = 0.03
MAX_STEP = 1000
INIT_WEIGHTS = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
STOP_CONDITION = 0.000001
GLOBAL_STEP = 0

