import numpy as np
import matplotlib.pyplot as pl
import matplotlib.colors as mcolors


def virno():

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = pl.cm.viridis(np.linspace(0., 1, 128))
    colors2 = pl.cm.inferno_r(np.linspace(0., 1, 128))

    # combine them and build a new colormap
    colors = np.vstack((colors1[5:][::-1], colors2[12:99][::-1]))
    virno = mcolors.LinearSegmentedColormap.from_list('virno', colors)
    return virno


if __name__ == '__main__':
    data = np.random.rand(10,10) * 2 - 1

    pl.figure()
    pl.pcolor(data, cmap=virno())
    pl.colorbar()
    pl.show()
