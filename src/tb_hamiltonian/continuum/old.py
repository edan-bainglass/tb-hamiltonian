from __future__ import annotations

import typing as t

import numpy as np


class GrapheneContinuumModel:
    """Continuum model for a BLG system.

    Logic adapted from Julia code developed by Dr. Gonçalo Santos Catarina
    https://www.empa.ch/web/s205/goncalo-catarina
    """

    # Pauli matrices
    s0 = np.array([[1, 0], [0, 1]], dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)

    def __init__(
        self,
        nearest_neighbor_hopping=2.7,
        bond_length=1.42,
        interlayer_hopping=0.4,
        superlattice_potential_periodicity=10,
        superlattice_potential_amplitude=0,
        gate_bias=0,
        layer_potential_ratio=0,
        nearest_neighbor_order=1,
    ):
        """`GrapheneContinuumModel` constructor.

        Parameters
        ----------
        `nearest_neighbor_hopping` : `float`, optional
            Nearest neighbor hopping energy in eV.
        `bond_length` : `float`, optional
            Carbon-carbon bond length in Angstrom.
        `interlayer_hopping` : `float`, optional
            Interlayer hopping energy in eV.
        `superlattice_potential_periodicity` : `int`, optional
            Periodicity of the superlattice potential in Angstrom.
        `superlattice_potential_amplitude` : `float`, optional
            Amplitude of the superlattice potential in eV.
        `gate_bias` : `float`, optional
            Bias potential in eV.
        `layer_potential_ratio` : `float`, optional
            Ratio of the potential in the top layer to the bottom layer.
        `nearest_neighbor_order` : `int`, optional
            Number of nearest neighbor vectors to consider.
        """
        self.nearest_neighbor_hopping = nearest_neighbor_hopping
        self.bond_length = bond_length
        self.interlayer_hopping = interlayer_hopping
        self.hbar_vF = 3 * nearest_neighbor_hopping * bond_length / 2

        self.Qn = create_Qn(superlattice_potential_periodicity)

        self.Q_list, ind_nondiag = create_Q_vector_list(
            self.Qn,
            nearest_neighbor_order,
        )

        self.H_bias = self.compute_H_bias(
            gate_bias,
            self.Q_list,
        )

        self.H_superlattice = self.compute_H_superlattice_potential(
            superlattice_potential_amplitude,
            layer_potential_ratio,
            ind_nondiag,
        )

    def M_BLG_Dirac(self, k: np.ndarray, Kpoint: str) -> np.ndarray:
        """Compute the BLG Hamiltonian in the Dirac basis.

        Parameters
        ----------
        `k` : `np.ndarray`
            Momentum vector.
        `Kpoint` : `str`
            K-point in the Brillouin zone about which to expand.

        Returns
        -------
        `np.ndarray`
            BLG Hamiltonian in the Dirac basis.
        """

        assert Kpoint in {"K", "Kp"}, "Incorrect Kpoint. Must be 'K' or 'Kp'."

        kx, ky = k

        intralayer_term = self.hbar_vF * (
            kx * np.kron(self.s0, self.s1) + ky * np.kron(self.s0, self.s2)
        )

        if Kpoint == "Kp":
            intralayer_term *= -1

        interlayer_term = (self.interlayer_hopping / 2) * (
            np.kron(self.s1, self.s1) - np.kron(self.s2, self.s2)
        )

        return intralayer_term + interlayer_term

    def M_bias(self, V0: float) -> np.ndarray:
        """Compute the bias Hamiltonian.

        Parameters
        ----------
        `V0` : `float`
            Gate bias potential.

        Returns
        -------
        `np.ndarray`
            Bias Hamiltonian.
        """
        return V0 * np.kron(self.s3, self.s0)

    def M_superlattice_potential(self, VSL: float, alpha: float) -> np.ndarray:
        """Compute the superlattice potential Hamiltonian.

        Parameters
        ----------
        `VSL` : `float`
            Amplitude of the superlattice potential.
        `alpha` : `float`
            Ratio of the potential in the top layer to the bottom layer.

        Returns
        -------
        `np.ndarray`
            Superlattice potential Hamiltonian.
        """
        return (VSL / 2) * (
            np.kron(self.s0 + self.s3, self.s0)
            + (alpha * np.kron(self.s0 - self.s3, self.s0))
        )

    def compute_H_BLG_Dirac(
        self,
        k: np.ndarray,
        Q_list: list[np.ndarray],
        Kpoint: str,
    ) -> np.ndarray:
        """Compute the superlattice BLG Hamiltonian in the Dirac basis.

        Parameters
        ----------
        `k` : `np.ndarray`
            Momentum vector.
        `Q_list` : `list[np.ndarray]`
            List of Q vectors.
        `Kpoint` : `str`
            K-point in the Brillouin zone about which to expand.

        Returns
        -------
        `np.ndarray`
            Superlattice BLG Hamiltonian in the Dirac basis.
        """
        dim = 4 * len(Q_list)
        H_BLG = np.zeros((dim, dim), dtype=complex)

        for i in range(len(Q_list)):
            i_start = i * 4
            i_end = i_start + 4

            M_BLG = self.M_BLG_Dirac(k + Q_list[i], Kpoint)

            H_BLG[i_start:i_end, i_start:i_end] += M_BLG

        return H_BLG

    def compute_H_bias(self, V0: float, Q_list: list[np.ndarray]) -> np.ndarray:
        """Compute the superlattice bias Hamiltonian.

        parameters
        ----------
        `V0` : `float`
            Gate bias potential.
        `Q_list` : `list[np.ndarray]`
            List of Q vectors.

        returns
        -------
        `np.ndarray`
            Superlattice Hamiltonian under a gate bias.
        """
        M_bias = self.M_bias(V0)

        dim = 4 * len(Q_list)
        H_bias = np.zeros((dim, dim), dtype=complex)

        for i in range(len(Q_list)):
            i_start = i * 4
            i_end = i_start + 4

            H_bias[i_start:i_end, i_start:i_end] += M_bias

        return H_bias

    def compute_H_superlattice_potential(
        self,
        VSL: float,
        alpha: float,
        nondiagonal_indices: list[int],
    ) -> np.ndarray:
        """Compute the superlattice Hamiltonian in a superlattice potential.

        Parameters
        ----------
        `VSL` : `float`
            Amplitude of the superlattice potential.
        `alpha` : `float`
            Ratio of the potential in the top layer to the bottom layer.
        `nondiagonal_indices` : `list[int]`
            List of indices of non-diagonal elements. # TODO - check this

        Returns
        -------
        `np.ndarray`
            Superlattice Hamiltonian in a superlattice potential.
        """
        M_SL = self.M_superlattice_potential(VSL, alpha)
        dim = 4 * len(nondiagonal_indices)
        H_SL = np.zeros((dim, dim), dtype=complex)

        for row in range(len(nondiagonal_indices)):
            row_start = row * 4
            row_end = row_start + 4

            for col in nondiagonal_indices[row]:
                col_start = col * 4
                col_end = col_start + 4

                H_SL[row_start:row_end, col_start:col_end] += M_SL

        return H_SL

    def H_BLG_K(self, k: np.ndarray) -> np.ndarray:
        return self.compute_H_BLG_Dirac(k, self.Q_list, "K")

    def H_BLG_Kp(self, k: np.ndarray) -> np.ndarray:
        return self.compute_H_BLG_Dirac(k, self.Q_list, "Kp")

    def H_total_K(self, k: np.ndarray) -> np.ndarray:
        return self.H_BLG_K(k) + self.H_bias + self.H_superlattice

    def H_total_Kp(self, k: np.ndarray) -> np.ndarray:
        return self.H_BLG_Kp(k) + self.H_bias + self.H_superlattice


def compute_eigenstuff(
    H_calculator: t.Callable[[np.ndarray], np.ndarray],
    kpath: np.ndarray,
    nbands: int = 0,
    use_mpi: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Compute the eigenvalues and eigenvectors along a k-path.

    Parameters
    ----------
    `H_calculator` : `Callable[[np.ndarray], np.ndarray]`
        Function that computes the Hamiltonian for a given k-point.
    `kpath` : `np.ndarray`
        List of k-points along the path.
    `nbands` : `int`, optional
        Number of bands to consider.
    `use_mpi` : `bool`, optional
        Whether to use MPI for parallel computation.

    Returns
    -------
    `tuple[np.ndarray, np.ndarray]`, optional
        Eigenvalues and eigenvectors along the k-path.
    """
    nbands = nbands or H_calculator(np.array([0, 0])).shape[0]

    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        kpath_split = np.array_split(kpath, size)
        local_kpath = comm.scatter(kpath_split, root=0)
        total_kpoints = len(kpath)

        all_eigenvalues = np.zeros((total_kpoints, nbands)) if rank == 0 else None
        all_eigenvectors = (
            np.zeros((total_kpoints, nbands, nbands), dtype=complex)
            if rank == 0
            else None
        )

    else:
        rank = 0
        local_kpath = kpath

    eigenvalues = np.zeros((len(local_kpath), nbands))
    eigenvectors = np.zeros((len(local_kpath), nbands, nbands), dtype=complex)

    for i, k in enumerate(local_kpath):
        H = H_calculator(k)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigenvalues[i] = eigvals
        eigenvectors[i] = eigvecs

    if use_mpi:
        comm.Gather(eigenvalues, all_eigenvalues, root=0)
        comm.Gather(eigenvectors, all_eigenvectors, root=0)
        eigenvalues = all_eigenvalues if rank == 0 else None
        eigenvectors = all_eigenvectors if rank == 0 else None

    return (eigenvalues, eigenvectors) if rank == 0 else (None, None)


def interpolate_path(
    points: list[np.ndarray],
    total_points: int = 100,
) -> tuple[np.ndarray, list[int]]:
    """Interpolate a path between high-symmetry points.

    Parameters
    ----------
    `points` : `list[np.ndarray]`
        List of high-symmetry points.
    `total_points` : `int`, optional
        Number of points to interpolate.

    Returns
    -------
    `np.ndarray`
        List of interpolated k-points.
    """
    distances = [distance(points[i], points[i + 1]) for i in range(len(points) - 1)]
    total_distance = sum(distances)

    path = []
    k_point_indices = [0]
    current_index = 0

    for i in range(len(points) - 1):
        segment_distance = distances[i]
        segment_points = round(total_points * (segment_distance / total_distance))

        for p in np.linspace(0, 1, segment_points):
            path.append(points[i] * (1 - p) + points[i + 1] * p)
            current_index += 1

        k_point_indices.append(current_index - 1)

    return np.array(path), k_point_indices


def create_Qn(L):
    """Create the Q vectors for the superlattice potential."""
    Q = 2 * np.pi / L
    return [
        Q * np.array([np.cos(2 * np.pi * n / 6), np.sin(2 * np.pi * n / 6)])
        for n in range(1, 7)
    ]


def create_Q_vector_list(Qn, NNorder):
    """Create the list of Q vectors for the superlattice potential."""
    list_Q = [np.array([0.0, 0.0])]  # [[0, 0], [Q1], [Q2], [Q3], [Q4], [Q5], [Q6]]
    list_len = [0, 1]  # [0, 1], [0, 1, 7]...

    ind_nondiag = []  # [[1, 2, 3, 4, 5, 6], [2, 0, 7]]

    # add Q vectors and indices of diagonal
    for it_NNorder in range(1, NNorder + 1):  # NNorder = 1, so 1 to 2, i.e. 1
        start_idx = list_len[it_NNorder - 1]  # For it_NNorder = 1, start_idx = 0
        end_idx = list_len[it_NNorder]  # For it_NNorder = 1, end_idx = 1

        for it_Q in range(start_idx, end_idx):
            Qs_it = [list_Q[it_Q] + Q for Q in Qn]

            ind_it = []  # [1, 2, 3, 4, 5, 6]
            for Q in Qs_it:
                append_if_not_present(list_Q, Q)

                ind = find_first(list_Q, Q)
                ind_it.append(ind)
            ind_nondiag.append(ind_it)

        list_len.append(len(list_Q))

    # add indices of non-diagonal elements for each Q vector
    # these represent <Qi|H|Qj> of the same order
    for it_Q in range(list_len[-2], list_len[-1]):  # 1 to 7, 7 to 19
        Qs_it = [
            list_Q[it_Q] + Q for Q in Qn
        ]  # when it_Q = 1, Qs_it = [Q1, Q2, Q3, Q4, Q5, Q6]

        ind_it = []  # [2, 0, 6]...
        for Q in Qs_it:
            ind = find_first(list_Q, Q)
            if ind is not None:  # ignores Q vectors outside of NNorder
                ind_it.append(ind)
        ind_nondiag.append(ind_it)

    return list_Q, ind_nondiag


def append_if_not_present(list_of_vectors, new_vector, atol=1e-8):
    for vec in list_of_vectors:
        if np.allclose(vec, new_vector, atol=atol):
            return list_of_vectors  # If already present, return the original list
    list_of_vectors.append(new_vector)
    return list_of_vectors


def find_first(list_of_vectors, target_vector, atol=1e-8):
    return next(
        (
            i
            for i, vec in enumerate(list_of_vectors)
            if np.allclose(vec, target_vector, atol=atol)
        ),
        None,
    )


def distance(p1, p2):
    return np.linalg.norm(p2 - p1)
