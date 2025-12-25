# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Wenyang Zhao, Osamu Miyashita, Florence Tama, RIKEN


from mpi4py import MPI
from threadpoolctl import threadpool_limits
from numba import set_num_threads

import numpy as np
from scipy.interpolate import make_interp_spline

from .utils import (
    check_points,
    check_bounds,
    bspline_band,
    evaluate,
    scale_segments,
)


__all__ = ["PencilDecompInterpolator"]


class PencilDecompInterpolator:
    """
    MPI-based multiprocessing interpolator in a 3D volume that is
    decomposed along the Y and Z-axes.

    Parameters
    ----------
    comm : MPI communicator
        Communication handle for parallel processing.
    coordinates : array-like
        A sequence of three 1D arrays (cx, cy, cz),
        representing the grid coordinates of the data points.
    data_yz : array-like
        Local 3D data values decomposed along the Y and Z-axes.
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
        comm,
        coordinates,
        data_yz,
        method="cubic",
        bounds_error=True,
        fill_value=np.nan,
        num_threads=1,
    ):

        # Validate and preprocess input parameters
        cx, cy, cz, data_yz, k, ny_all, nz_all = self._check_inputs(
            comm, coordinates, data_yz, method
        )

        # Construct the spline representation
        tx, ty, tz, C, nx_all, ny_all = self._construct(
            comm, cx, cy, cz, data_yz, k, ny_all, nz_all, num_threads
        )

        # Extract domain bounds
        xmin, xmax = cx[0], cx[-1]
        ymin, ymax = cy[0], cy[-1]
        zmin, zmax = cz[0], cz[-1]
        bounds = ((xmin, xmax), (ymin, ymax), (zmin, zmax))

        # Store parameters and internal state
        self.comm = comm
        self.k = k
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.C = C
        self.nx_all = nx_all
        self.ny_all = ny_all
        self.bounds = bounds
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.num_threads = num_threads

    def _check_inputs(self, comm, coordinates, data_yz, method):
        """
        Validate and preprocess input parameters for the interpolator.

        See the class-level docstring for a full description of parameters.

        Returns
        -------
        cx, cy, cz : np.ndarray
            Validated 1D arrays of grid coordinates.
        data_yz : np.ndarray
            Validated local 3D data values decomposed along the Y and Z-axes.
        k : int
            B-spline order.
        ny_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Y-axis.
        nz_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Z-axis.

        Notes
        -----
        The 3D volume is decomposed into ny_all.size × nz_all.size pencils
        along the Y and Z axes. `ny_all` and `nz_all` are Inferred from the
        shape of `data_yz` provided by all ranks.
        """

        rank = comm.Get_rank()
        size = comm.Get_size()
        msg = []

        if rank == 0:
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
        else:
            # For non-root ranks, initialize to None / False placeholders
            valid_method, k, valid_coord = None, None, None
            cx, cy, cz = None, None, None

        # Broadcast validation results and values from root to all ranks
        valid_method = comm.bcast(valid_method, root=0)
        if valid_method:
            k = comm.bcast(k, root=0)
        valid_coord = comm.bcast(valid_coord, root=0)
        if valid_coord:
            cx = comm.bcast(cx, root=0)
            cy = comm.bcast(cy, root=0)
            cz = comm.bcast(cz, root=0)

        # Validate data_yz array
        try:
            data_yz = np.asarray(data_yz, dtype=np.float64)
            valid_data_yz = True
            if data_yz.ndim != 3:
                valid_data_yz = False
        except (ValueError, TypeError):
            valid_data_yz = False

        if not valid_data_yz:
            msg.append("Invalid data_yz: a 3D array is expected.")
        else:
            if not np.isfinite(data_yz).all():
                msg.append(
                    "data_yz must not contain "
                    "NaNs or infinite values."
                )

        # Size consistency check
        if valid_coord and valid_data_yz:
            NX, NY, NZ = cx.shape[0], cy.shape[0], cz.shape[0]

            if data_yz.shape[0] != NX:
                msg.append(
                    "X size mismatch: "
                    f"expected {NX} from coordinates, "
                    f"got {data_yz.shape[0]} from data_yz."
                )

            ny = data_yz.shape[1]
            nz = data_yz.shape[2]
            ny_all = np.asarray(comm.allgather(ny), dtype=np.int64)
            nz_all = np.asarray(comm.allgather(nz), dtype=np.int64)

            if rank == 0:
                size_z = np.sum(ny_all) / NY
                size_y = np.sum(nz_all) / NZ
                if (
                    not size_y.is_integer()
                    or not size_z.is_integer()
                    or int(size_z * size_y) != size
                ):
                    msg.append(
                        "Y or Z size mismatch: "
                        "inconsistent layout across ranks."
                    )
                else:
                    size_y = int(size_y)
                    size_z = int(size_z)

                    ny_all = np.reshape(ny_all, (size_y, size_z))
                    nz_all = np.reshape(nz_all, (size_y, size_z))
                    valid_ny = np.all(ny_all == ny_all[:, [0]])
                    valid_nz = np.all(nz_all == nz_all[[0], :])
                    if not valid_ny:
                        msg.append(
                            "Y size mismatch: "
                            "inconsistent layout across ranks."
                        )
                    if not valid_nz:
                        msg.append(
                            "Z size mismatch: "
                            "inconsistent layout across ranks."
                        )
                    if valid_ny and valid_nz:
                        ny_all = ny_all[:,0]
                        nz_all = nz_all[0,:]

            ny_all = comm.bcast(ny_all, root=0)
            nz_all = comm.bcast(nz_all, root=0)

        # Gather error messages from all ranks
        msg_all = comm.gather("\n".join(msg), root=0)
        if rank == 0:
            if any(msg_all):
                for i, m in enumerate(msg_all):
                    if m:
                        print(f"[Rank {i}]\n{m}\n\n")
                comm.Abort(1)

        comm.Barrier()
        return cx, cy, cz, data_yz, k, ny_all, nz_all

    def _construct(
            self,
            comm,
            cx,
            cy,
            cz,
            data_yz,
            k,
            ny_all,
            nz_all,
            num_threads,
        ):
        """
        This method constructs the tensor-product B-spline representation
        of the input data by performing sequential 1D B-spline fitting
        along the X, Y, and Z axes.

        Initially, both `data_yz` and the intermidiate coefficient tensor
        `C` are decomposed along the Y and Z-axes. After B-spline fitting
        along X, the coefficient tensor `C` is redistributed to be decomposed
        along X and Z, which enables B-spline fitting along Y. Subsequently,
        it is redistributed to be decomposed along X and Y, enabling B-spline
        fitting along Z.

        Parameters
        ----------
        comm : MPI communicator
            Used for inter-process redistribution of the coefficient tensor.
        cx, cy, cz : np.ndarray
            1D arrays of grid coordinates.
        data_yz : np.ndarray
            Local 3D data values decomposed along the Y and Z-axes.
        k : int
            B-spline order.
        ny_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Y-axis,
            before redistribution.
        nz_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Z-axis,
            before redistribution.
        num_threads : int
            Number of threads to use for B-spline fitting.

        Returns
        -------
        tx, ty, tz : np.ndarray
            1D knot vectors along each axis.
        C : np.ndarray
            Final 3D B-spline coefficient tensor,
            decomposed along the X and Y axes.
        nx_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the X-axis,
            after redistribution.
        ny_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Y-axis,
            after redistribution.
        """

        with threadpool_limits(limits=num_threads):
            # B-spline fitting along X-axis
            spl = make_interp_spline(
                cx, data_yz, axis=0, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            tx = spl.t
            C = np.transpose(spl.c, (0, 1, 2))

            # Redistribute coefficients: from Y, Z-decomposed
            # to X, Z-decomposed
            nx_all = scale_segments(ny_all, cx.shape[0])
            C = self._redistribute_yz_to_xz(comm, C, nx_all, ny_all, nz_all)

            # B-spline fitting along Y-axis
            spl = make_interp_spline(
                cy, C, axis=1, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            ty = spl.t
            C = np.transpose(spl.c, (1, 0, 2))

            # Redistribute coefficients: from X,Z-decomposed
            # to X, Y-decomposed
            ny_all = scale_segments(nz_all, cy.shape[0])
            C = self._redistribute_xz_to_xy(comm, C, nx_all, ny_all, nz_all)

            # B-spline fitting along Z-axis
            spl = make_interp_spline(
                cz, C, axis=2, k=k,
                bc_type="not-a-knot", check_finite=False
            )
            tz = spl.t
            C = np.transpose(spl.c, (1, 2, 0))

        return tx, ty, tz, C, nx_all, ny_all

    def _redistribute_yz_to_xz(self, comm, array_yz, nx_all, ny_all, nz_all):
        """
        Redistribute a 3D array from Y,Z-decomposed to X,Z-decomposed.

        Parameters
        ----------
        comm : MPI communicator
            Communication handle for inter-process redistribution.
        array_yz : np.ndarray
            Local 3D data values decomposed along the Y and Z-axes,
            before redistribution.
        nx_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the X-axis,
            after redistribution.
        ny_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Y-axis,
            before redistribution.
        nz_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Z-axis,
            remains unchanged before and after redistribution.

        Returns
        -------
        array_xz : np.ndarray
            Local 3D data values decomposed along the X and Z axes,
            after redistribution.
        """

        rank = comm.Get_rank()

        # Number of segments along each axis
        size_x = nx_all.shape[0]
        size_y = ny_all.shape[0]
        size_z = nz_all.shape[0]

        # X, Y, and Z rank indices, before or after redistribution
        id_y = rank // size_z
        id_z = rank % size_z
        id_x = id_y

        # Local and global sizes
        nx = nx_all[id_x]
        ny = ny_all[id_y]
        nz = nz_all[id_z]
        NX = np.sum(nx_all)
        NY = np.sum(ny_all)

        # Split communicator by id_z
        comm_z = comm.Split(color=id_z, key=id_y)

        # Compute starting indices for each segment
        x_starts = np.cumsum(np.insert(nx_all, 0, 0)[:-1])
        y_starts = np.cumsum(np.insert(ny_all, 0, 0)[:-1])

        # ---------------------
        # Prepare send buffer
        # ---------------------
        # Each rank sends flattened blocks (nx_i, ny, nz)
        # to X-rank i in the same Z-group
        sendcounts = [nx_all[i] * ny * nz for i in range(0, size_x)]
        sdispls = np.cumsum([0] + sendcounts[:-1])
        sendbuf = np.empty((NX * ny * nz), dtype=np.float64)

        for i in range(0, size_x):
            x_start = x_starts[i]
            x_end = x_start + nx_all[i]
            sendbuf[sdispls[i] : sdispls[i] + sendcounts[i]] = (
                array_yz[x_start : x_end, :, :].ravel()
            )

        # ---------------------
        # Prepare receive buffer
        # ---------------------
        # Each rank receives flattened blocks (nx, ny_i, nz)
        # from Y-rank i in the same Z-group
        recvcounts = [nx * ny_all[i] * nz for i in range(0, size_y)]
        rdispls = np.cumsum([0] + recvcounts[:-1])
        recvbuf = np.empty((nx * NY * nz), dtype=np.float64)

        # ---------------------
        # Perform all-to-all communication within each Z-group
        # ---------------------
        comm_z.Alltoallv([sendbuf, sendcounts, sdispls, MPI.DOUBLE],
                         [recvbuf, recvcounts, rdispls, MPI.DOUBLE])

        # ---------------------
        # Reconstruct local array
        # ---------------------
        array_xz = np.empty((nx, NY, nz), dtype=np.float64)
        for i in range(0, size_y):
            y_start = y_starts[i]
            y_end = y_starts[i] + ny_all[i]
            array_xz[:, y_start : y_end, :] = np.reshape(
                recvbuf[rdispls[i] : rdispls[i] + recvcounts[i]],
                (nx, ny_all[i], nz)
            )

        return array_xz

    def _redistribute_xz_to_xy(self, comm, array_xz, nx_all, ny_all, nz_all):
        """
        Redistribute a 3D array from X,Z-decomposed to X,Y-decomposed.

        Parameters
        ----------
        comm : MPI communicator
            Communication handle for inter-process redistribution.
        array_xz : np.ndarray
            Local 3D data values decomposed along the X and Z-axes,
            before redistribution.
        nx_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the X-axis,
            remains unchanged before and after redistribution.
        ny_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Y-axis,
            after redistribution.
        nz_all : np.ndarray, dtype=int
            1D array specifying the sizes of segments along the Z-axis,
            before redistribution.

        Returns
        -------
        array_xz : np.ndarray
            Local 3D data values decomposed along the X and Y axes,
            after redistribution.
        """

        rank = comm.Get_rank()

        # Number of segments along Y and Z axes
        size_y = ny_all.shape[0]
        size_z = nz_all.shape[0]

        # X, Y, and Z rank indices, before or after redistribution
        id_x = rank // size_z
        id_z = rank % size_z
        id_y = id_z

        # Local and global sizes
        nx = nx_all[id_x]
        ny = ny_all[id_y]
        nz = nz_all[id_z]
        NY = np.sum(ny_all)
        NZ = np.sum(nz_all)

        # Split communicator by id_x
        comm_x = comm.Split(color=id_x, key=id_z)

        # Compute starting indices for each segment
        y_starts = np.cumsum(np.insert(ny_all, 0, 0)[:-1])
        z_starts = np.cumsum(np.insert(nz_all, 0, 0)[:-1])

        # ---------------------
        # Prepare send buffer
        # ---------------------
        # Each rank sends flattened blocks (nx, ny_i, nz)
        # to Y-rank i in the same X-group
        sendcounts = [nx * ny_all[i] * nz for i in range(0, size_y)]
        sdispls = np.cumsum([0] + sendcounts[:-1])
        sendbuf = np.empty((nx * NY * nz), dtype=np.float64)

        for i in range(0, size_y):
            y_start = y_starts[i]
            y_end = y_start + ny_all[i]
            sendbuf[sdispls[i] : sdispls[i] + sendcounts[i]] = (
                array_xz[:, y_start : y_end, :].ravel()
            )

        # ---------------------
        # Prepare receive buffer
        # ---------------------
        # Each rank receives flattened blocks (nx, ny, nz_i)
        # from Z-rank i in the same X-group
        recvcounts = [nx * ny * nz_all[i] for i in range(0, size_z)]
        rdispls = np.cumsum([0] + recvcounts[:-1])
        recvbuf = np.empty((nx * ny * NZ), dtype=np.float64)

        # ---------------------
        # Perform all-to-all communication within each X-group
        # ---------------------
        comm_x.Alltoallv([sendbuf, sendcounts, sdispls, MPI.DOUBLE],
                         [recvbuf, recvcounts, rdispls, MPI.DOUBLE])

        # ---------------------
        # Reconstruct local array
        # ---------------------
        array_xy = np.empty((nx, ny, NZ), dtype=np.float64)
        for i in range(0, size_z):
            z_start = z_starts[i]
            z_end = z_start + nz_all[i]
            array_xy[:, :, z_start : z_end] = np.reshape(
                recvbuf[rdispls[i] : rdispls[i] + recvcounts[i]],
                (nx, ny, nz_all[i])
            )

        return array_xy

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
        values : np.ndarray of shape (num_points,) or None
            On rank 0: Evaluated values at the input points. Points outside
            the domain bounds are set to `fill_value` if `bounds_error` is
            False.
            On other ranks: None.
        """

        # Unpack attributes
        comm = self.comm
        rank = comm.Get_rank()

        k = self.k
        tx = self.tx
        ty = self.ty
        tz = self.tz
        C = self.C
        nx_all = self.nx_all
        ny_all = self.ny_all
        bounds = self.bounds
        bounds_error = self.bounds_error
        fill_value = self.fill_value
        num_threads = self.num_threads

        # Local XY-segment information
        size_y = ny_all.shape[0]
        id_x = rank // size_y
        id_y = rank % size_y
        x_start = np.cumsum(np.insert(nx_all, 0, 0)[:-1])[id_x]
        nx = nx_all[id_x]
        x_end = x_start + nx - 1
        y_start = np.cumsum(np.insert(ny_all, 0, 0)[:-1])[id_y]
        ny = ny_all[id_y]
        y_end = y_start + ny - 1

        # Validate and assign points (only on root)
        if rank == 0:
            # Validate input points
            msg, points = check_points(points)
            if msg:
                print (msg)
                comm.Abort(1)

            # Check for out-of-bound points
            flag_all_in, mask_in, num_in = check_bounds(points, bounds)
            if bounds_error and not flag_all_in:
                print ("One or more query points are out of bounds.")
                comm.Abort(1)
        else:
            num_in = None

        # Broadcast the in-bound points to all ranks
        num_in = comm.bcast(num_in, root=0)
        if rank == 0:
            points_in = np.ascontiguousarray(
                points[mask_in], dtype=np.float64
            )
        else:
            points_in = np.empty((num_in, 3), dtype=np.float64)
        comm.Bcast([points_in, MPI.DOUBLE], root=0)

        # Extract the points that need to be compute locally
        points_local, mask_local, num_local = self._select_local_points(
            points_in, k, tx, ty, x_start, x_end, y_start, y_end
        )

        # Calculate local results
        values_local = np.zeros((num_local), dtype=np.float64)
        if nx != 0 and ny != 0 and num_local != 0:
            # Compute the B-spline basis function indices and values
            with threadpool_limits(limits=num_threads):
                px, py, pz = points_local.T
                index_x, bx_vals = bspline_band(px, tx, k)
                index_y, by_vals = bspline_band(py, ty, k)
                index_z, bz_vals = bspline_band(pz, tz, k)

                # For local segment, adjust X indices and values
                index_x -= x_start
                bx_vals[(index_x < 0) | (index_x > nx - 1)] = 0.0
                np.clip(index_x, 0, nx - 1, out=index_x)

                # For local segment, adjust Y indices and values
                index_y -= y_start
                by_vals[(index_y < 0) | (index_y > ny - 1)] = 0.0
                np.clip(index_y, 0, ny - 1, out=index_y)

            # Tensor product using Numba JIT-parallelized function
            set_num_threads(num_threads)
            evaluate(
                C, bx_vals, by_vals, bz_vals,
                index_x, index_y, index_z,
                num_local, k+1, values_local
            )

        # Merge local results to root
        values_in = np.zeros((num_in), dtype=np.float64)
        values_in[mask_local] = values_local
        if rank == 0:
            values_in_sum = np.empty((num_in), dtype=np.float64)
        else:
            values_in_sum = None
        comm.Reduce(values_in, values_in_sum, op=MPI.SUM, root=0)

        if rank == 0:
            values = np.full(points.shape[0], fill_value, dtype=np.float64)
            values[mask_in] = values_in_sum
        else:
            values = None

        return values

    def _select_local_points(
        self, points, k, tx, ty, x_start, x_end, y_start, y_end
    ):
        """
        Select the subset of query points assigned to this rank based on
        their (x, y) location relative to the local XY-segment.

        Each query point requires B-spline basis functions over index ranges
        [i-k, i] in X and [j-k, j] in Y, where i satisfies tx[i] <= x
        < tx[i+1], and j satisfies ty[j] <= y < ty[j+1]. A point is assigned
        to this rank if, in both X and Y directions, any of its required basis
        function indices intersect with the local XY-segment, defined by
        [x_start, x-end] and [y_start, y_end].

        Parameters
        ----------
        points : np.ndarray, shape (num_points, 3)
            Query point coordinates. Only the X and Y components are used.
        k : int
            B-spline order.
        tx : np.ndarray
            1D knot vector along the X-axis.
        ty : np.ndarray
            1D knot vector along the Y-axis.
        x_start : int
            Starting X index (inclusive) of the local segment.
        x_end : int
            Ending X index (inclusive) of the local segment.
        y_start : int
            Starting Y index (inclusive) of the local segment.
        y_end : int
            Ending Y index (inclusive) of the local segment.

        Returns
        -------
        points_local : np.ndarray, shape (num_local, 3)
            Subset of input points assigned to this rank.
        mask_local : np.ndarray, shape (num_points,), dtype=bool
            Boolean mask indicating which input points assigned to this rank.
        num_local : int
            Number of points assigned to this rank.
        """

        Ny = ty.shape[0] - k - 1

        # For each point, find the indices of the X- and Y-axis B-spline
        # basis functions required
        Nx = tx.shape[0] - k - 1
        index_x_max = np.searchsorted(tx, points[:,0], side="right") - 1
        np.clip(index_x_max, k, Nx - 1, out=index_x_max)
        index_x_min = index_x_max - k

        Ny = ty.shape[0] - k - 1
        index_y_max = np.searchsorted(ty, points[:,1], side="right") - 1
        np.clip(index_y_max, k, Ny - 1, out=index_y_max)
        index_y_min = index_y_max - k

        mask_x = ((index_x_max >= x_start) & (index_x_min <= x_end))
        mask_y = ((index_y_max >= y_start) & (index_y_min <= y_end))
        mask_local = (mask_x & mask_y)
        points_local = np.ascontiguousarray(points[mask_local])
        num_local = np.count_nonzero(mask_local)

        return points_local, mask_local, num_local
