"""
Example: 3D B-spline interpolation on a single process.
"""

import numpy as np
from xspline3d import SingleProcessInterpolator

# Define the grid size along each axis
Nx, Ny, Nz = 60, 64, 68

# Define coordinate arrays for the regular grid
cx = np.linspace(0, 1, Nx, endpoint=True)
cy = np.linspace(0, 2, Ny, endpoint=True)
cz = np.linspace(0, 3, Nz, endpoint=True)

# Create synthetic 3D scalar field data on the grid
data = np.random.rand(Nx, Ny, Nz)

# Generate random query points within the domain bounds
num_points = 100
points = np.random.uniform(
    low=[0, 0, 0], high=[1, 2, 3], size=(num_points, 3)
)

# --- Phase 1: Construct the interpolator ---
interp = SingleProcessInterpolator(
    (cx, cy, cz),
    data,
    method="cubic",
    num_threads=2,
)

# --- Phase 2: Evaluate at query points ---
values = interp(points)

# Print the interpolated values
print ()
print ("3D B-spline interpolation (single process):")
print(f"{'Index':>8} | {'Value':>12}")
for i, val in enumerate(values):
    print(f"{i:8d} | {val:12.6f}")
