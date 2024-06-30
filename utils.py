import numpy as np


def handle_PBC(
    ng: int,
    g: int,
    n: int,
    R: int,
    coords_j: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Adjust atom coordinates w.r.t the periodic boundary
    conditions in the supplied direction.

    Parameters
    ----------
    `ng` : `int`
        number of grid cells in a given direction
    `g` : `int`
        index of grid cell in a given direction
    `n` : `int`
        index of neighboring grid cell in a given direction
    `R` : `int`
        R vector component in a given direction
    `coords_j` : `np.ndarray`
        coordinates of atom j
    `v` : `np.ndarray`
        lattice vector in a given direction

    Returns
    -------
    `tuple[np.ndarray, int]`
        adjusted coordinates of atom j and R vector component
    """
    if g == 0 and g + n < g and (g + n) % ng == ng - 1:
        R -= 1
        coords_j = coords_j - v
    elif g == ng - 1 and g + n > g and (g + n) % ng == 0:
        R += 1
        coords_j = coords_j + v
    return coords_j, R
