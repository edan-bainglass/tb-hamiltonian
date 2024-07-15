# Code to compute a TB Hamiltonian

An evolving package to compute tight-binding Hamiltonians.

## MPI

Band structure

1. `multiprocessing` package for core-only parallelization
2. `mpi4py` package for MPI parallelization

If using `mpi4py`, install the `mpi` optional dependencies with `pip install .[mpi]`.

If `pip` fails to build the package, you could try this [stackoverflow discussion](https://stackoverflow.com/questions/74427664/error-could-not-build-wheels-for-mpi4py-which-is-required-to-install-pyproject).

If this does not work, you can install `mpi4py` using `conda` (or `mamba`).
