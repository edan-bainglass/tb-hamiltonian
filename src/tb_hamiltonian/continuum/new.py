from __future__ import annotations

import typing as t

import numpy as np


class BLGContinuumModel:
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
        superlattice_potential_amplitude=0.0,
        gate_bias=0.0,
        layer_potential_ratio=0.0,
        concentric_order=1,
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
        `concentric_order` : `int`, optional
            Used in computing `Q_concentric`.
        """
        self.t1 = nearest_neighbor_hopping
        self.dcc = bond_length
        self.tp = interlayer_hopping
        self.L = superlattice_potential_periodicity

        # Real space lattice vectors for Graphene unit cell
        self.a1G = np.sqrt(3) * np.array([0.5, 0.5 * np.sqrt(3)]) * bond_length
        self.a2G = np.sqrt(3) * np.array([-0.5, 0.5 * np.sqrt(3)]) * bond_length

        # Reciprocal space lattice vectors for Graphene unit cell
        self.b1G = 4 * np.pi / (3 * bond_length) * np.array([0.5 * np.sqrt(3), 0.5])
        self.b2G = 4 * np.pi / (3 * bond_length) * np.array([-0.5 * np.sqrt(3), 0.5])

        # Reciprocal space lattice vectors for the supercell
        self.b1 = self.b1G / np.floor(
            2 * superlattice_potential_periodicity / 3 / bond_length
        )
        self.b2 = self.b2G / np.floor(
            2 * superlattice_potential_periodicity / 3 / bond_length
        )

        self.KG = (self.b1G - self.b2G) / 3

        self.Qn = self.compute_Qn()

        self.Q_vectors = self.Q_concentric(concentric_order)

        self.VSL = superlattice_potential_amplitude
        self.V0 = gate_bias
        self.alpha = layer_potential_ratio

    def compute_Qn(self):
        θ = np.pi / 3
        R60 = np.array([[np.cos(θ), -np.sin(θ)], [np.sin(θ), np.cos(θ)]])

        # Initialize rotated_vectors with the original vector
        rotated_vectors = [self.b1]
        current_v = self.b1

        # Apply the 60-degree rotation 5 times and collect results
        for _ in range(5):
            current_v = R60 @ current_v  # Rotate by 60 degrees
            rotated_vectors.append(current_v)

        return rotated_vectors

    def Q_concentric(self, concentric_order):
        """Generate concentric vectors in G defined by basis vectors b1 and b2."""
        vectors = []

        for m_sum in range(concentric_order + 1):
            new_vectors = 0

            for m1 in range(-m_sum, m_sum + 1):
                m2 = m_sum - abs(m1)

                Q = m1 * self.b1 + m2 * self.b2
                if not _is_in_GG_lattice(Q, vectors, self.b1G, self.b2G):
                    vectors.append(Q)
                    new_vectors += 1

                if m2 != 0:
                    Q = m1 * self.b1 - m2 * self.b2
                    if not _is_in_GG_lattice(Q, vectors, self.b1G, self.b2G):
                        vectors.append(Q)
                        new_vectors += 1

            if new_vectors == 0:
                break

        return vectors

    def f(self, k):
        return 1 + np.exp(1j * np.dot(k, self.a1G)) + np.exp(1j * np.dot(k, self.a2G))

    def M_V0(self) -> np.ndarray:
        """Compute the bias Hamiltonian.

        Returns
        -------
        `np.ndarray`
            Bias Hamiltonian.
        """
        return self.V0 * np.kron(self.s3, self.s0)

    def H_V0(self, k: np.ndarray) -> np.ndarray:
        """Compute the superlattice bias Hamiltonian.

        parameters
        ----------
        `k` : `np.ndarray`
            Momentum vector.

        returns
        -------
        `np.ndarray`
            Superlattice Hamiltonian under a gate bias.
        """
        f_k = self.f(k)
        tp = self.tp

        v1 = -self.t1 * f_k
        v2 = -self.t1 * np.conj(f_k)

        H_bias = np.array(
            [
                [0, v1, 0, tp],
                [v2, 0, 0, 0],
                [0, 0, 0, v1],
                [tp, 0, v2, 0],
            ]
        )

        return H_bias + self.M_V0()

    def M_VSL(self) -> np.ndarray:
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
        return (self.VSL / 2) * (
            np.kron(self.s0 + self.s3, self.s0)
            + (self.alpha * np.kron(self.s0 - self.s3, self.s0))
        )

    def H_folded(self, k: np.ndarray) -> np.ndarray:
        """Compute the superlattice BLG Hamiltonian.

        Parameters
        ----------
        `k` : `np.ndarray`
            Momentum vector.

        Returns
        -------
        `np.ndarray`
            Superlattice BLG Hamiltonian in the Dirac basis.
        """
        dim = 4 * len(self.Q_vectors)
        H_folded = np.zeros((dim, dim), dtype=complex)

        for i in range(len(self.Q_vectors)):
            i_start = i * 4
            i_end = i_start + 4
            H_folded[i_start:i_end, i_start:i_end] += self.H_V0(k + self.Q_vectors[i])

            for Q in self.Qn:
                idx = _find_first(self.Q_vectors, Q + self.Q_vectors[i])

                if idx is not None:
                    j_start = idx * 4
                    j_end = j_start + 4
                    H_folded[i_start:i_end, j_start:j_end] += self.M_VSL()

        return H_folded

    def decompose_and_keep_remainder(self, K):
        # Create a matrix B using b1 and b2 as columns
        B = np.column_stack((self.b1, self.b2))

        # Solve for m1 and m2 (not necessarily integers)
        coeffs = np.linalg.solve(B, K)

        # Round m1 and m2 to the nearest integers
        m1 = int(round(coeffs[0]))
        m2 = int(round(coeffs[1]))

        # Compute the remainder by subtracting the integer combination of b1 and b2 from P
        rest = K - (m1 * self.b1 + m2 * self.b2)

        # Return only the remainder
        return rest

    def H2D_KG(self, k: np.ndarray) -> np.ndarray:
        """Compute the Hamiltonian around K."""
        return self.H_folded(k + self.KG)

    def H2D_KpG(self, k: np.ndarray) -> np.ndarray:
        """Compute the Hamiltonian around Kp."""
        return self.H_folded(k - self.KG)


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

        current_index += segment_points

        k_point_indices.append(current_index - 1)

    return np.array(path), k_point_indices


def distance(p1, p2):
    return np.linalg.norm(p2 - p1)


def _find_first(vectors, target_vector, atol=1e-6):
    return next(
        (
            i
            for i, vector in enumerate(vectors)
            if np.allclose(vector, target_vector, atol=atol)
        ),
        None,
    )


def _is_in_GG_lattice(Q, Q_vectors, b1G, b2G, tol=1e-6):
    """Check if Q can be expressed as an integer combination
    of b1G and b2G plus any of the Q vectors.
    """
    B = np.column_stack((b1G, b2G))

    for Q_vector in Q_vectors:
        coefficients, *_ = np.linalg.lstsq(B, Q - Q_vector, rcond=None)

        if np.allclose(coefficients, np.round(coefficients), atol=tol):
            return True

    return False
