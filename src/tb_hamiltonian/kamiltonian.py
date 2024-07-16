from __future__ import annotations

import typing as t

import numpy as np
from numpy.linalg import eigvalsh as numpy_solver
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh as scipy_sparse_solver

if t.TYPE_CHECKING:
    from tb_hamiltonian.hamiltonian import TBHamiltonian


class TBKamiltonian:
    """Class to represent the k-space Hamiltonian of a tight-binding model."""

    def __init__(self, H: TBHamiltonian, k: np.ndarray):
        """`TBKamiltonian` constructor.

        Parameters
        ----------
        `H` : `TBHamiltonian`
            The tight-binding Hamiltonian.
        `k` : `np.ndarray`
            The k-point in the Brillouin zone.
        """
        self.H = H
        self.k = k
        self.matrix = lil_matrix((H.natoms, H.natoms), dtype=complex)

    def build(self, consider_atomic_positions=False):
        """Build the k-space Hamiltonian.

        Parameters
        ----------
        `consider_atomic_positions` : `bool`, optional
            Whether to consider the atomic positions when building the Hamiltonian.
            Default is `False`.
        """

        for ri, Hr in enumerate(self.H):
            exp_k_R = np.exp(2j * np.pi * self.k.dot(self.H.R[ri]))

            if consider_atomic_positions:
                scaled_positions = self.H.structure.get_scaled_positions()
                for i, j in zip(*Hr.nonzero()):
                    Δij = scaled_positions[j] - scaled_positions[i]
                    exp_k_D = np.exp(2j * np.pi * self.k.dot(Δij))
                    self.matrix[i, j] += Hr[i, j] * exp_k_R * exp_k_D
            else:
                self.matrix += Hr * exp_k_R

    def get_eigenvalues(
        self,
        use_sparse_solver=False,
        sparse_solver_params: dict | None = None,
    ) -> np.ndarray:
        """Get the eigenvalues of the k-space Hamiltonian.

        If `use_sparse_solver` is `True`, the eigenvalues are computed using
        `scipy.sparse.linalg.eigsh`. Otherwise, the eigenvalues are computed
        using `numpy.linalg.eigvalsh`. Note that the `scipy` solver may yield
        spurious eigenvalues (spikes) in the band structure, particularly near
        high-symmetry k-points. See the `scipy` documentation for more details.

        Parameters
        ----------
        `use_sparse_solver` : `bool`, optional
            Whether to use a sparse solver to compute the eigenvalues.
            Default is `False`.
        `sparse_solver_params` : `dict`, optional
            Additional parameters to pass to the `scipy` solver.

        Returns
        -------
        `np.ndarray`
            The eigenvalues of the k-space Hamiltonian.
        """
        if use_sparse_solver:
            eigenvalues = scipy_sparse_solver(
                self.matrix,
                return_eigenvectors=False,
                **sparse_solver_params or {},
            )
        else:
            eigenvalues = numpy_solver(self.matrix.toarray()).real
        return np.sort(eigenvalues.real)
