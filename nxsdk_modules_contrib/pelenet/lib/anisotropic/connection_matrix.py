# -*- coding: utf-8 -*-
#
# connection_matrix.py
#
# Code is taken from: https://github.com/babsey/spatio-temporal-activity-sequence/tree/6d4ab597c98c01a2a9aa037834a0115faee62587


import numpy as np

import lcrn_network as lcrn
import connection_asymmetry_landscape as cal


def I_networks(landscape, nrow, ncol, ncon, kappa, theta, seed=0, **kwargs):
    np.random.seed(seed)

    npop = nrow * ncol
    landscape_mode = landscape['mode']

    if landscape_mode != 'symmetric':
        move = cal.move(nrow)
        ll = cal.landscapes[landscape_mode](
            nrow, landscape.get('specs', {}))
        ll = np.round(ll * 7).astype(int)

    conmat = []
    for ii in range(npop):
        targets, delay = lcrn.lcrn_gamma_targets(
            ii, nrow, ncol, nrow, ncol, ncon, kappa, theta)
        if landscape_mode != 'symmetric':           # asymmetry
            targets = (targets + move[ll[ii] % len(move)]) % npop
        targets = targets[targets != ii]            # no selfconnections
        hist_targets = np.histogram(targets, bins=range(npop + 1))[0]
        conmat.append(hist_targets)

    return np.array(conmat)


def EI_networks(landscape, nrowE, ncolE, nrowI, ncolI, p, stdE, stdI, seed=0, **kwargs):
    np.random.seed(seed)

    npopE = nrowE * ncolE
    npopI = nrowI * ncolI
    landscape_mode = landscape['mode']

    if landscape_mode != 'symmetric':
        move = cal.move(nrowE)
        ll = cal.landscapes[landscape_mode](
            nrowE, landscape.get('specs', {}))
        ll = np.round(ll * 7).astype(int)

    conmatEE, conmatEI = [], []
    for idx in range(npopE):
        # E-> E
        source = idx, nrowE, ncolE, nrowE, ncolE, int(p * npopE), stdE
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        if landscape_mode != 'symmetric':           # asymmetry
            targets = (targets + move[ll[idx] % len(move)]) % npopE
        targets = targets[targets != idx]           # no selfconnections
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatEE.append(hist_targets)

        # E-> I
        source = idx, nrowE, ncolE, nrowI, ncolI, int(p * npopI), stdI
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatEI.append(hist_targets)

    conmatIE, conmatII = [], []
    for idx in range(npopI):

        # I-> E
        source = idx, nrowI, ncolI, nrowE, ncolE, int(p * npopE), stdE
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        hist_targets = np.histogram(targets, bins=range(npopE + 1))[0]
        conmatIE.append(hist_targets)

        # I-> I
        source = idx, nrowI, ncolI, nrowI, ncolI, int(p * npopI), stdI
        targets, delay = lcrn.lcrn_gauss_targets(*source)
        targets = targets[targets != idx]           # no selfconnections
        hist_targets = np.histogram(targets, bins=range(npopI + 1))[0]
        conmatII.append(hist_targets)

    return np.array(conmatEE), np.array(conmatEI), np.array(conmatIE), np.array(conmatII)
