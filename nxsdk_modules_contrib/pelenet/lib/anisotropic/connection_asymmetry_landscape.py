# -*- coding: utf-8 -*-
#
# connectivity_assymetry_landscape.py
#
# Code is taken from: https://github.com/babsey/spatio-temporal-activity-sequence/tree/6d4ab597c98c01a2a9aa037834a0115faee62587

import numpy as np
import local_values_landscape as lvl

__all__ = [
    'move',
    'landscapes',
]


def move(nrow):
    return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])

landscapes = lvl.landscapes
