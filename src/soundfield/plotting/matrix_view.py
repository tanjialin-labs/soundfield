"""Coordinate-aware matrix display with cell-value annotations."""

import numpy as np
from ..contracts import real_array


def plot_annotated_matrix(values, *, x_coordinates=None, y_coordinates=None, value_format='.3g', axes=None):
    """Return a figure; rows map to y coordinates and columns to x coordinates."""
    matrix = real_array(values, 'values', 2)
    if min(matrix.shape) == 0:
        raise ValueError('matrix must have nonempty axes')
    x = np.arange(matrix.shape[1]) if x_coordinates is None else real_array(x_coordinates, 'x_coordinates', 1)
    y = np.arange(matrix.shape[0]) if y_coordinates is None else real_array(y_coordinates, 'y_coordinates', 1)
    if x.shape != (matrix.shape[1],) or y.shape != (matrix.shape[0],) or np.any(np.diff(x)<=0) or np.any(np.diff(y)<=0):
        raise ValueError('coordinates must increase and match the matrix dimensions')
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as exception:
        raise ImportError('Matrix figures require soundfield[plotting]') from exception
    if axes is None:
        figure, axes = pyplot.subplots(constrained_layout=True)
    else:
        figure = axes.figure
    artist = axes.pcolormesh(x, y, matrix, shading='nearest')
    for row, coordinate_y in enumerate(y):
        for column, coordinate_x in enumerate(x):
            axes.text(coordinate_x, coordinate_y, format(matrix[row, column], value_format), ha='center', va='center', fontsize=6)
    figure.colorbar(artist, ax=axes)
    return figure
