# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Wenyang Zhao, Osamu Miyashita, Florence Tama, RIKEN


from threadpoolctl import threadpool_limits
from numba import set_num_threads

import numpy as np
from scipy.interpolate import make_interp_spline

from .utils import (
    check_points,
    check_bounds,
    bspline_band,
    evaluate,
)


__all__ = ["SingleProcessInterpolator"]


class SingleProcessInterpolator:
    """
    Single-process interpolator in a non-decomposed 3D volume.

    Parameters
    ----------
    coordinates : array-like
        A sequence of three 1D arrays (cx, cy, cz),
        representing the grid coordinates of the data points.
    data : array-like
        3D non-decomposed data values.
    method : str, optional
        B-spline order.
        Supported options are:
        - "slinear"  : k = 1
        - "cubic"   : k = 3 [default]
        - "quintic" : k = 5
    bounds_error : bool, optional
        If True, raises an error when query points are outside the domain
        bounds defined by coordinates. Default is True.
    fill_value : float, optional
        Value to use for query points outside the domain bounds.
        Default is numpy.nan.
    num_threads : int, optional
        Number of threads to use for parallel computations. Default is 1.
    """

    def __init__(
        self,
        coordinates,
        data,
        method="cubic",
        bounds_error=True,
        fill_value=np.nan,
        num_threads=1,
    ):

        # Validate and preprocess input parameters
        cx, cy, cz, data, k = self._check_inputs(
            coordinates, data, method
        )

        # Construct the spline representation
        tx, ty, tz, C = self._construct(
            cx, cy, cz, data, k, num_threads
        )

        # Extract domain bounds
        xmin, xmax = cx[0], cx[-1]
        ymin, ymax = cy[0], cy[-1]
        zmin, zmax = cz[0], cz[-1]
        bounds = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

        # Store parameters and internal state
        self.k = k
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.C = C
        self.bounds = bounds
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.num_threads = num_threads

    def _check_inputs(self, coordinates, data, method):
        """
        Validate and preprocess input parameters for the interpolator.

        See the class-level docstring for a full description of parameters.

        Returns
        -------
        cx, cy, cz : np.ndarray
            Validated 1D arrays of grid coordinates
        data : np.ndarray
            Validated 3D non-decomposed data values.
        k : int
            B-spline order.
        """

        msg = []

        # Validate interpolation method
        dict_method = {"slinear": 1, "cubic": 3, "quintic": 5}
        if method not in dict_method:
            msg.append(
                f"Unknown method {method} "
                "(Available: slinear, cubic, quintic)."
            )
            k = None
            valid_method = False
        else:
            k = dict_method[method]
            valid_method = True

        # Validate coordinates format and properties
        try:
            cx, cy, cz = coordinates
            cx = np.asarray(cx, dtype=np.float64)
            cy = np.asarray(cy, dtype=np.float64)
            cz = np.asarray(cz, dtype=np.float64)
        except (ValueError, TypeError):
            msg.append(
                "Coordinates must be a sequence of three 1D arrays: "
                "(cx, cy, cz)."
            )
            valid_coord = False
            cx, cy, cz = None, None, None
        else:
            # Check each coordinate array is 1D
            valid_coord = True
            for coord in (cx, cy, cz):
                if coord.ndim != 1:
                    valid_coord = False
                    break

            if not valid_coord:
                msg.append(
                    "Each coordinate array must be "
                    "1-dimensional."
                )
            else:
                # Check coordinates are finite and strictly increasing
                for coord, name in zip((cx, cy, cz), ("X", "Y", "Z")):
                    if not np.isfinite(coord).all():
                        msg.append(
                            f"{name} coordinates must not contain "
                            "NaNs or infinite values."
                        )
                    if np.any(np.diff(coord) <= 0):
                        msg.append(
                            f"{name} coordinates must be "
                            "strictly increasing."
                        )

        # Check minimum coordinate size with respect to spline order
        if valid_method and valid_coord:
            for coord, name in zip((cx, cy, cz), ("X", "Y", "Z")):
                if coord.shape[0] < k + 1:
                    msg.append(
                        f"{name} size too small: "
                        f"expected at least {k + 1}, "
                        f"got {coord.shape[0]}."
                    )

        # Validate data array
        try:
            data = np.asarray(data, dtype=np.float64)
            valid_data = True
            if data.ndim != 3:
                valid_data = False
        except (ValueError, TypeError):
            valid_data = False

        if not valid_data:
            msg.append("Invalid data: a 3D array is expected.")
        else:
            if not np.isfinite(data).all():
                msg.append(
                    "data must not contain "
                    "NaNs or infinite values."
                )

        # Check consistency between coordinates and data shapes
        if valid_coord and valid_data:
            NX, NY, NZ = cx.shape[0], cy.shape[0], cz.shape[0]
            if data.shape[0] != NX:
                msg.append(
                    "X size mismatch: "
                    f"expected {NX} from coordinates, "
                    f"got {data.shape[0]} from data."
                )
            if data.shape[1] != NY:
                msg.append(
                    "Y size mismatch: "
                    f"expected {NY} from coordinates, "
                    f"got {data.shape[1]} from data."
                )
            if data.shape[2] != NZ:
                msg.append(
                    "Z size mismatch: "
                    f"expected {NZ} from coordinates, "
                    f"got {data.shape[2]} from data."
                )

        # Raise an exception if any errors were found
        if msg:
            raise ValueError("\n".join(msg))

        return cx, cy, cz, data, k

    def _construct(self, cx, cy, cz, data, k, num_threads):
        """
        This method constructs the tensor-product B-spline representation
        of the input data by performing sequential 1D B-spline fitting
        along the X, Y, and Z axes.

        Parameters
        ----------
        cx, cy, cz : np.ndarray
            1D arrays of grid coordinates.
        data : np.ndarray
            3D non-decomposed data values.
        k : int
            B-spline order.
        num_threads : int
            Number of threads to use for B-spline fitting.

        Returns
        -------
        tx, ty, tz : np.ndarray
            1D knot vectors along each axis.
        C : np.ndarray
            Final 3D B-spline coefficient tensor.
        """

        with threadpool_limits(limits=num_threads):
            # B-spline fitting along X-axis
            spl = make_interp_spline(
                cx, data, axis=0, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            tx = spl.t
            C = np.transpose(spl.c, (0, 1, 2))

            # B-spline fitting along Y-axis
            spl = make_interp_spline(
                cy, C, axis=1, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            ty = spl.t
            C = np.transpose(spl.c, (1, 0, 2))

            # B-spline fitting along Z-axis
            spl = make_interp_spline(
                cz, C, axis=2, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            tz = spl.t
            C = np.transpose(spl.c, (1, 2, 0))

        return tx, ty, tz, C


    def __call__(self, points):
        """
        Evaluate the B-spline interpolator at the given query points.

        Parameters
        ----------
        points : array-like
            Input query point coordinates.
            Expected shape is (num_points, 3).

        Returns
        -------
        values : np.ndarray, shape (num_points,)
            Evaluated values at the input points. Points outside the domain
            bounds are set to `fill_value` if `bounds_error` is False.
        """

        # Unpack attributes
        k = self.k
        tx = self.tx
        ty = self.ty
        tz = self.tz
        C = self.C
        bounds = self.bounds
        bounds_error = self.bounds_error
        fill_value = self.fill_value
        num_threads = self.num_threads

        # Validate input points
        msg, points = check_points(points)
        if msg:
            raise ValueError(msg)

        # Check for out-of-bound points
        flag_all_in, mask_in, num_in = check_bounds(points, bounds)
        if bounds_error and not flag_all_in:
            raise ValueError("One or more query points are out of bounds.")

        values_in = np.zeros((num_in), dtype=np.float64)
        if num_in != 0:
            # Compute B-spline basis function indices and values
            with threadpool_limits(limits=num_threads):
                px, py, pz = points[mask_in].T
                index_x, bx_vals = bspline_band(px, tx, k)
                index_y, by_vals = bspline_band(py, ty, k)
                index_z, bz_vals = bspline_band(pz, tz, k)

            # Tensor product using Numba JIT-parallelized function
            set_num_threads(num_threads)
            evaluate(
                C, bx_vals, by_vals, bz_vals,
                index_x, index_y, index_z,
                num_in, k+1, values_in
            )

        # Assign computed values into in-bound points
        values = np.full(points.shape[0], fill_value, dtype=np.float64)
        values[mask_in] = values_in

        return values

