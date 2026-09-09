"""The upstream cubic polar transform with reusable geometry, not a different resampler."""

from functools import lru_cache

import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.spatial import Delaunay


@lru_cache(maxsize=4)
def geometry(rows, cols, tip, r_max, angle):
    center_x, center_y = tip
    x = np.linspace(-center_x, cols - center_x - 1, cols)
    y = np.linspace(-center_y, rows - center_y - 1, rows)
    x, y = np.meshgrid(x, y)
    points = np.column_stack((x.ravel(), y.ravel()))
    radians = np.radians(-90)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points = (rotation @ points.T).T
    r = np.linspace(0, r_max, rows)
    theta = np.linspace(-angle, angle, cols)
    r, theta = np.meshgrid(r, theta)
    query = np.column_stack(((r * np.cos(theta)).ravel(), (r * np.sin(theta)).ravel()))
    return points, query, Delaunay(points)


def cached_cartesian_to_polar_matrix(
    cartesian_matrix, tip=(61, 7), r_max=107, angle=0.79, interpolation="nearest"
):
    rows, cols = cartesian_matrix.shape
    points, query, triangulation = geometry(rows, cols, tuple(tip), r_max, angle)
    if interpolation == "cubic":
        # Same class and default gradient tolerance used by scipy.griddata(method='cubic').
        # Pixel values/gradients are still computed afresh for every frame.
        values = CloughTocher2DInterpolator(triangulation, cartesian_matrix.ravel(), fill_value=0)(
            query
        )
    else:
        values = griddata(
            points, cartesian_matrix.ravel(), query, method=interpolation, fill_value=0
        )
    return np.rot90(values.reshape(cols, rows), k=-1)
