# -*- coding: utf-8 -*-
#
# connectivity_landscape.py
#
# Code is taken from: https://github.com/babsey/spatio-temporal-activity-sequence/tree/6d4ab597c98c01a2a9aa037834a0115faee62587

import numpy as np
import noise

__all__ = [
    'move',
    'landscapes',
]


def move(nrow):
    return np.array([1, nrow + 1, nrow, nrow - 1, -1, -nrow - 1, -nrow, -nrow + 1])


def homogeneous(nrow, specs={}):
    dir_idx = specs.get('phi', 4)

    npop = np.power(nrow, 2)
    landscape = np.ones(npop, dtype=int) * dir_idx
    return landscape


def random(nrow, specs={}):
    seed = specs.get('seed', 0)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.randint(8, size=npop)
    return landscape


def tiled(nrow, specs={}):
    seed = specs.get('seed', 0)
    tile_size = specs.get('tile_size', 10)

    np.random.seed(seed)
    ncol_dir = nrow / tile_size
    didx = np.random.randint(0, 8, size=[ncol_dir, ncol_dir])
    landscape = np.repeat(np.repeat(didx, tile_size, 0), tile_size, 1)
    return landscape.ravel()


def Perlin(nrow, specs={}):
    size = specs.get('size', 5)
    base = specs.get('base', 0)
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=base)
          for j in y] for i in x]
    m = n - np.min(n)
    landscape = np.array(np.round(m * 7), dtype=int)
    return landscape.ravel()


def Perlin_uniform(nrow, specs={}):
    size = specs.get('size', 5)
    base = specs.get('base', 100)
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size, base=base)
          for j in y] for i in x]
    m = np.concatenate(n)
    a = np.argsort(m)
    b = np.power(nrow, 2) // 8
    for i in range(8):
        m[a[i * b:(i + 1) * b]] = i
    landscape = m.astype(int)
    return landscape


landscapes = {
    'homogeneous': homogeneous,
    'random': random,
    'tiled': tiled,
    'Perlin': Perlin,
    'Perlin_uniform': Perlin_uniform,
}
