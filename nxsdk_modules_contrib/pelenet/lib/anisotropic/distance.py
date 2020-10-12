import numpy as np
import scipy.spatial as spa


__all__ = ['distance_matrix']

def distance_matrix(n_row, n_col):
    """
    Return a distance matrix.

    Parameters
    ----------
    n_row, n_col : int
        Number of rows and columns.

    Returns
    -------
    distance : array_like
        A 1-dimensional array of distance matrix
    """

    n_pop = int(n_row * n_col)
    center = int(n_row/2*(n_col+1))

    pop_idx = np.arange(n_pop)
    pop_idx_col = np.remainder(pop_idx, n_col)
    pop_idx_row = pop_idx // n_row

    pos = np.vstack((pop_idx_col,pop_idx_row)).T
    distance = spa.distance.cdist([pos[center]], pos)[0]

    return distance

