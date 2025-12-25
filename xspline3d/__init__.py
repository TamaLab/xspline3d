# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025, Wenyang Zhao, Osamu Miyashita, Florence Tama, RIKEN


"""
interp_spline_3d: MPI-based spline interpolation in distributed 3D volumes

Contributors:
    - Wenyang Zhao <wenyang.zhao@riken.jp>
    - Osamu Miyashita <osamu.miyashita@riken.jp>
    - Florence Tama <florence.tama@riken.jp>

Affiliation:
    Computational Structural Biology Research Team
    RIKEN Center for Computational Science
"""


__version__ = "0.1.0"

from .interp_single import SingleProcessInterpolator
from .interp_mpi_slab import SlabDecompInterpolator
from .interp_mpi_pencil import PencilDecompInterpolator

__all__ = [
    "SingleProcessInterpolator",
    "SlabDecompInterpolator",
    "PencilDecompInterpolator",
]

