"""
Example: 3D B-spline interpolation in volumes decomposed along the Z-axis.
"""

import numpy as np
from mpi4py import MPI
from xspline3d import SlabDecompInterpolator

# Initialize MPI communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define the grid size along each axis
Nx, Ny, Nz = 60, 64, 68

# Define the Z segment size for each rank (example: uneven decomposition)
nz_all = [5, 13, 21, 29]
assert sum(nz_all) == Nz
assert len(nz_all) == size
nz = nz_all[rank]

# Define coordinate arrays for the regular grid
# Required only on rank 0; ignored on other ranks
if rank == 0:
    cx = np.linspace(0, 1, Nx, endpoint=True)
    cy = np.linspace(0, 2, Ny, endpoint=True)
    cz = np.linspace(0, 3, Nz, endpoint=True)
else:
    cx, cy, cz = None, None, None

# Create synthetic 3D scalar field data for this rank
data_z = np.random.rand(Nx, Ny, nz)

# Generate random query points within the domain bounds
# Required only on rank 0; ignored on other ranks
if rank == 0:
    num_points = 100
    points = np.random.uniform(
        low=[0, 0, 0], high=[1, 2, 3], size=(num_points, 3)
    )
else:
    points = None

# --- Phase 1: Construct the MPI interpolator ---
interp = SlabDecompInterpolator(
    comm,
    (cx, cy, cz),
    data_z,
    method="cubic",
    num_threads=2,
)

# --- Phase 2: Evaluate at query points ---
# Output is collected on rank 0; None is returned on other ranks
values = interp(points)

# Print the interpolated values
if rank == 0:
    print ()
    print ("3D B-spline interpolation with Z decomposition:")
    print(f"{'Index':>8} | {'Value':>12}")
    for i, val in enumerate(values):
        print(f"{i:8d} | {val:12.6f}")
