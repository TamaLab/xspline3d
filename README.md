# xspline3d

MPI-based spline interpolation in distributed 3D volumes.

---

## Overview

`xspline3d` is a Python library for high-efficiency spline
interpolation in 3D volumes. It supports MPI-based memory-distributed
parallel processing and allows large-scale 3D volumes to be decomposed along
the Z-axis (slab decomposition) or both the Y and Z axes (pencil
decomposition). This makes it well-suited for memory-distributed scientific
computing tasks on HPC systems.

---

## Features

- Spline interpolation in segmented 3D volumes
- Support for both slab (Z-axis) and pencil (Y-Z axes) domain decompositions
- MPI-based parallelism for distributed-memory environments
- High efficiency in both interpolator construction and evaluation
- Pure Python interface backed by high-performance libraries (NumPy, SciPy,
Numba, etc.)

---

## Installation

Install directly from GitHub via pip:

```bash
pip install git+https://github.com/TamaLab/xspline3d.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/TamaLab/xspline3d.git
cd xspline3d
pip install .
```

---

## Requirements

- Python >= 3.11
- numba >= 0.61
- numpy >= 2.2
- scipy >= 1.15
- threadpoolctl >= 3
- mpi4py

**Note on mpi4py:** For proper functionality and performance, `mpi4py` must
be built against and consistent with your system's MPI installation (e.g.,
OpenMPI, MPICH). Please ensure your system's MPI module is loaded and the
`MPICC` environment variable is set correctly before installing `mpi4py`.
Refer to the [environment.yaml](environment.yaml) file and its installation
notes for detailed instructions on how to set up `mpi4py` by building it from
source.

---

## Usage

Example scripts demonstrating how to use the interpolators are provided in
the [examples/](examples/) directory.

---

## License

This project is licensed under the BSD 3-Clause License. See the
[LICENSE](LICENSE) file for details.

---

## Citation

If you use `xspline3d` in your research, please cite the following paper:

> Wenyang Zhao, Osamu Miyashita, and Florence Tama. 2026.
> xspline3d: A Python Library for MPI-Based Spline Interpolation Enforcing
> Global Continuity in Distributed 3D Volumes.
> In *SCA/HPCAsia 2026 Workshops: Supercomputing Asia and International
> Conference on High Performance Computing in Asia Pacific Region Workshops
> (SCA/HPCAsiaWS 2026), January 26–29, 2026, Osaka, Japan*.
> ACM, New York, NY, USA, 7 pages.
> https://doi.org/10.1145/3784828.3786260

An extended manuscript is also available in the [paper/](paper/) directory.

---

## Authors and Affiliations

- Wenyang Zhao <wenyang.zhao@riken.jp>
- Osamu Miyashita <osamu.miyashita@riken.jp>
- Florence Tama <florence.tama@riken.jp>

Computational Structural Biology Research Team

RIKEN Center for Computational Science
