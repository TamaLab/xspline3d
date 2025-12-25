# Examples

This directory contains example scripts demonstrating how to use the
`xspline3d` library for 3D B-spline interpolation in various
parallelization setups.

## Running the examples

```bash
# Single-process example (no MPI)
python example_single_process.py

# MPI example with decomposition along the Z-axis
mpiexec -n 4 python example_decomposed_z.py

# MPI example with decomposition along the Y and Z axes
mpiexec -n 6 python example_decomposed_yz.py
```
