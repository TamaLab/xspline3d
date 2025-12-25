# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Wenyang Zhao, Osamu Miyashita, Florence Tama, RIKEN


import numpy as np
import numba as nb
from scipy.interpolate import BSpline


__all__ = [
    "check_points",
    "check_bounds",
    "bspline_band",
    "scale_segments",
    "evaluate",
]


def check_points(points):
    """
    Validate the input array of query points.

    Parameters
    ----------
    points : array-like
        Input query point coordinates to be validated.
        Expected shape is (num_points, 3).

    Returns
    -------
    msg : str or None
        Error message if validation fails, otherwise None.
    points : np.ndarray or None
        Validated array of shape (num_points, 3), or None on failure.
    """

    try:
        points = np.asarray(points, dtype=np.float64)
    except Exception:
        msg = f"Invalid points: could not convert to a float64 array."
        return msg, None
    if points.ndim != 2 or points.shape[1] != 3:
        msg = (
            "Invalid points: expected a 2D array with shape (N, 3), "
            f"got shape {points.shape}."
        )
        return msg, None
    if points.shape[0] == 0:
        msg = f"Invalid points: expected at least one point, got empty."
        return msg, None
    if not np.isfinite(points).all():
        msg = "Invalid points: must not contain infs or nans."
        return msg, None
    return None, points


def check_bounds(points, bounds):
    """
    Check whether points lie within the given domain bounds.

    Parameters
    ----------
    points : np.ndarray, shape (num_points, 3)
        Query point coordinates.
    bounds : tuple of tuples
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) defining 3D bounding box.

    Returns
    -------
    flag_all_in : bool
        True if all points lie inside the bounds.
    mask_in : np.ndarray, dtype=bool
        1D boolean mask indicating which points lie inside the bounds.
    index_in : np.ndarray
        Indices of points that lie inside the bounds.
    """

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    mask_in_x = (xmin <= points[:,0]) & (points[:,0] <= xmax)
    mask_in_y = (ymin <= points[:,1]) & (points[:,1] <= ymax)
    mask_in_z = (zmin <= points[:,2]) & (points[:,2] <= zmax)
    mask_in = mask_in_x & mask_in_y & mask_in_z
    flag_all_in = mask_in.all()
    num_in = np.count_nonzero(mask_in)
    return flag_all_in, mask_in, num_in


def bspline_band(p, t, k):
    """
    Compute the B-spline basis function indices and values at given points.

    Parameters
    ----------
    p : np.ndarray
        1D array of coordinates along a single axis of the query points.
    t : np.ndarray
        1D knot vector along the same axis.
    k : int
        B-spline order.

    Returns
    -------
    index : np.ndarray, shape (num_points, k+1), dtype=int
        Indices of the B-spline basis functions that have nonzero values
        at each point, i.e., the basis functions required to evaluate the
        spline at those points.
    bvals : np.ndarray, shape (num_points, k+1)
        Corresponding nonzero values of the B-spline basis functions at
        each point.
    """

    num_points = p.shape[0]
    if num_points == 0:
        index = np.zeros((num_points, k+1), dtype=np.float64)
        bvals = np.zeros((num_points, k+1), dtype=np.float64)
    else:
        # Compressed Sparse Row (CSR) matrix
        csr = BSpline.design_matrix(p, t, k).tocsr()
        index = np.array(csr.indices, dtype=np.int64).reshape(
            (num_points, k+1))
        bvals = np.array(csr.data, dtype=np.float64).reshape(
            (num_points, k+1))
    return index, bvals


def scale_segments(n1_all, N2):
    """
    Segment the global size N2 of a second axis across MPI ranks,
    proportionally to local sizes n1_all along a first axis.

    Parameters
    ----------
    n1_all : np.ndarray, dtype=int
        1D array of local sizes along the 1st axis for all ranks.
    N2 : int
        Global size along the second axis to be segmented.

    Returns
    -------
    n2_all : np.ndarray, dtype=int
        1D array of local sizes along the 2nd axis for all ranks.
    """

    N1 = np.sum(n1_all)

    # Compute ideal (floating-point) proportions
    ideal_n2_all = N2 * (n1_all / N1)

    # Take integer floor of each portion
    n2_all = np.floor(ideal_n2_all).astype(np.int64)

    # Distribute the remaining elements based on largest fractional parts
    remainder = N2 - np.sum(n2_all)
    frac_parts = ideal_n2_all - n2_all
    indices = np.argsort(frac_parts)[::-1]
    for i in range(remainder):
        n2_all[indices[i]] += 1

    return n2_all


@nb.njit(parallel=True, fastmath=True)
def evaluate(
    C,
    bx_vals,
    by_vals,
    bz_vals,
    index_x,
    index_y,
    index_z,
    num,
    d,
    values,
):
    """
    Evaluate at multiple query points using the constructed coefficient
    tensor and the precomputed values of B-spline basis functions.

    Parameters
    ----------
    C : np.ndarray
        3D B-spline coefficient tensor.
    bx_vals, by_vals, bz_vals : np.ndarray, shape (num, d)
        Nonzero values of B-spline basis functions at each point along the
        X, Y, and Z axes.
    index_x, index_y, index_z : np.ndarray, shape (num, d), dtype=int
        Indices of the B-spline basis functions at each point along the X,
        Y, and Z axes.
    num : int
        Number of query points.
    d : int
        Number of required B-spline basis functions per dimension,
        d = B-spline order + 1.
    values : np.ndarray
        1D output array to store the evaluated values, modified in place.

    Returns
    -------
    None
        The evaluated results are stored in-place in the `values` array.
    """

    for i in nb.prange(0, num):
        value = 0.0
        for ix in range(0, d):
            for iy in range(0, d):
                for iz in range(0, d):
                    value += (
                        C[index_x[i,ix], index_y[i,iy], index_z[i,iz]] *
                        bx_vals[i, ix] *
                        by_vals[i, iy] *
                        bz_vals[i, iz]
                    )
        values[i] = value

