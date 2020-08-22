# -*- coding: utf-8 -*-
#
# connectivity_assymetry_landscape.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import local_values_landscape as lvl

__all__ = [
    'move',
    'landscapes',
]


def move(nrow):
    return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])

landscapes = lvl.landscapes
