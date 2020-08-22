# -*- coding: utf-8 -*-
#
# local_values_landscape.py
#
# Copyright 2017 Sebastian Spreizer
# The MIT License

import numpy as np
import noise

__all__ = [
    'landscapes',
]


def homogeneous(nrow, specs={}):
    phi = specs.get('phi', .5)

    npop = np.power(nrow, 2)
    landscape = np.ones(npop) * phi
    return landscape


def random(nrow, specs={}):
    seed = specs.get('seed', 0)

    np.random.seed(seed)
    npop = np.power(nrow, 2)
    landscape = np.random.uniform(size=npop)
    return landscape


def tiled(nrow, specs={}):
    seed = specs.get('seed', 0)
    tile_size = specs.get('tile_size', 10)

    np.random.seed(seed)
    nrow_tile = nrow / tile_size
    phi = np.random.uniform(size=[nrow_tile, nrow_tile])
    landscape = np.repeat(np.repeat(phi, tile_size, 0), tile_size, 1)
    return landscape.ravel()


def Perlin(nrow, specs={}):
    size = specs.get('size', 5)
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size)
          for j in y] for i in x]
    landscape = n - np.min(n)
    landscape /= landscape.max()
    return landscape.ravel()


def Perlin_uniform(nrow, specs={}, nbins=100):
    size = specs.get('size', 5)
    assert(size > 0)

    x = y = np.linspace(0, size, nrow)
    n = [[noise.pnoise2(i, j, repeatx=size, repeaty=size)
          for j in y] for i in x]
    m = np.concatenate(n)
    a = np.argsort(m)
    b = np.power(nrow, 2) // nbins
    for i, j in enumerate(np.linspace(0, 1, nbins)):
        m[a[i * b:(i + 1) * b]] = j
    landscape = m
    return landscape


landscapes = {
    'homogeneous': homogeneous,
    'random': random,
    'tiled': tiled,
    'Perlin': Perlin,
    'Perlin_uniform': Perlin_uniform,
}
