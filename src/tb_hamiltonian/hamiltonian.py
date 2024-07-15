from __future__ import annotations

import importlib.util
import itertools
import typing as t
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from matplotlib.patches import Rectangle
from scipy import sparse

from tb_hamiltonian.kamiltonian import TBKamiltonian

from .potentials import PotentialFactory, PotentialFunction


class TBHamiltonian:
    """Class to represent the tight-binding Hamiltonian of a system."""

    def __init__(
        self,
        structure: Atoms,
        nearest_neighbor: int,
        distances: list[float],
        hopping_parameters: list[float],
        interlayer_coupling: float,
    ):
        """`TBHamiltonian` constructor.

        Parameters
        ----------
        `structure` : `Atoms`
            Atomic structure of the system.
        `nearest_neighbor` : `int`
            1: nearest neighbor, 2: next-nearest neighbor, etc.
        `distances` : `list[float]`
            List of distances corresponding to the hopping parameters.
        `hopping_parameters` : `list[float]`
            List of hopping parameters corresponding to the distances.
        `interlayer_coupling` : `float`
            Interlayer coupling parameter.
        """
        self.structure = structure
        self.threshold = distances[nearest_neighbor]
        self.distances = distances
        self.hopping_parameters = hopping_parameters
        self.interlayer_coupling = interlayer_coupling
        self.natoms = len(structure)
        self.R = [np.array([i, j, 0]) for i in range(-1, 2) for j in range(-1, 2)]
        self.matrix = [sparse.lil_matrix((self.natoms, self.natoms)) for _ in range(len(self.R))]
        self._interaction_count_dict = OrderedDict.fromkeys(range(1, self.natoms + 1), 0)

        self.R_index_map: dict[tuple[int, int], int] = {
            (int(Rx), int(Ry)): i for i, (Rx, Ry, _) in enumerate(self.R)
        }

    def build(self):
        """Build the Hamiltonian matrix."""
        self.grid = self._get_search_grid()
        self.ngx, self.ngy = len(self.grid[0]), len(self.grid)
        for gy in range(self.ngy):
            for gx in range(self.ngx):
                self._compute_coupling_for_grid_cell(gy, gx)

    def update_onsite_terms(
        self,
        onsite_term: float = 0.0,
        potential: PotentialFunction = PotentialFactory("null"),
        alpha: t.Sequence[float] | None = None,
    ) -> None:
        """Update the on-site terms of the Hamiltonian.

        Parameters
        ----------
        `onsite_term` : `float`, optional
            On-site term to be added to the Hamiltonian.
        `potential` : `PotentialFunction`, optional
            Potential function to be applied to the on-site term.
        `alpha` : `Sequence[float]`, optional
            Screening parameters for the potential function.

        Raises
        ------
        `ValueError`
            If the length of `alpha` is not equal to the number of layers.
        """
        layer_heights = np.unique(self.structure.positions[:, 2])
        if not isinstance(alpha, t.Sequence) or len(alpha) != len(layer_heights):
            raise ValueError(
                f"`alpha` must be a sequence of length equal to the number of layers ({len(layer_heights)})"
            )
        scaled_positions = self.structure.get_scaled_positions()
        for i in range(self.natoms):
            self[4][i, i] = onsite_term
            for h, height in enumerate(layer_heights):
                if np.isclose(self.structure.positions[i][2], height):
                    break
            frac_i = scaled_positions[i]
            self[4][i, i] += alpha[h] * potential(frac_i)

    def onsite_count(self) -> int:
        """Return the number of non-zero on-site terms in the Hamiltonian."""
        return np.count_nonzero(self[4].diagonal())

    def interaction_counts(self):
        """Return the number of interactions for each atom."""
        for atom, count in self._interaction_count_dict.items():
            print(f"Atom {atom} interacts with {count} other atoms.")

    def get_kamiltonian(
        self,
        k: np.ndarray,
        consider_atomic_positions=False,
    ) -> TBKamiltonian:
        """Compute the k-space Hamiltonian.

        Parameters
        ----------
        `k` : `np.ndarray`
            k-vector in the reciprocal space.
        `consider_atomic_positions` : `bool`, optional
            Whether to consider the atomic positions when building the Hamiltonian.

        Returns
        -------
        `TBKamiltonian`
            k-space Hamiltonian.
        """
        H_k = TBKamiltonian(self, k)
        H_k.build(consider_atomic_positions)
        return H_k

    def get_band_structure(
        self,
        k_points: np.ndarray,
        use_sparse_solver=False,
        sparse_solver_params: dict | None = None,
        use_mpi=False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the band structure.

        Parameters
        ----------
        `k_points` : `np.ndarray`
            List of k-points.
        `use_sparse_solver` : `bool`, optional
            Whether to use the sparse solver.
        `sparse_solver_params` : `dict`, optional
            Parameters for the sparse solver.
        `use_mpi` : `bool`, optional
            Whether to use the MPI parallelization.

        Returns
        -------
        `tuple[np.ndarray, np.ndarray]`
            Distances and band structure.
        """
        if sparse_solver_params is None:
            sparse_solver_params = {
                "k": 10,
                "sigma": 1e-3,
                "maxiter": 1000,
                "tol": 1e-6,
            }

        if use_mpi:
            if importlib.util.find_spec("mpi4py") is None:
                print("mpi4py is not installed. ", end="")
            else:
                print("Using MPI parallelization")
                return self._get_band_structure_mpi(
                    k_points,
                    use_sparse_solver,
                    sparse_solver_params,
                )
        print("Using multiprocessing parallelization")
        return self._get_band_structure_multiprocessing(
            k_points,
            use_sparse_solver,
            sparse_solver_params,
        )

    def write_to_file(
        self,
        path: Path = Path("output/example"),
        filename: str = "TG_hr.dat",
    ) -> None:
        """Write the Hamiltonian to a file.

        Parameters
        ----------
        `path` : `Path`, optional
            Path to the directory where the Hamiltonian will be written.
        `filename` : `str`, optional
            Name of the file where the Hamiltonian will be written.
        """
        non_zero_elements = sum(H.count_nonzero() for H in self)
        system_label = self.structure.info.get("label", self.structure.symbols)

        with (path / filename).open("w") as file:
            # write banner
            for line in (
                f"! Tight binding model for {system_label} system, theta=0\n",
                f"{non_zero_elements:5d}  ! Number of non-zeros lines of HmnR\n",
                f"{self.natoms:5d}  ! Number of Wannier functions\n",
                f"{len(self.R):5d}  ! Number of R points\n",
                "    1" * len(self.R) + "\n",
            ):
                file.write(line)

            # write H
            for i, r in enumerate(self.R):
                Hcoo = self[i].tocoo()
                for ai, aj, v in zip(Hcoo.row, Hcoo.col, Hcoo.data):
                    Rx, Ry, Rz = r
                    file.write(f"{Rx:5d}{Ry:5d}{Rz:5d}{ai + 1:8d}{aj + 1:8d}{v:13.6f}{0:13.6f}\n")

    def plot_matrix(self, R_index=4, start=0, end=-1, step=1):
        """Plot the Hamiltonian matrix.

        Parameters
        ----------
        `R_index` : `int`, optional
            Index of the Hamiltonian matrix.
        `start` : `int`, optional
            Start index of the plot.
        `end` : `int`, optional
            End index of the plot.
        `step` : `int`, optional
            Step size for the ticks.
        """
        Hr = self[R_index].toarray()
        end = self.natoms if end < start else end
        plt.imshow(Hr[start:end, start:end], cmap="inferno", interpolation="nearest")
        plt.xticks(
            np.arange(0, end - start, step),
            [str(i) for i in np.arange(start, end, step) + 1],
        )
        plt.yticks(
            np.arange(0, end - start, step),
            [str(i) for i in np.arange(start, end, step) + 1],
        )
        plt.colorbar()
        plt.show()

    def plot_grid(self, show_ticks=False, show_labels=False):
        """Plot the search grid.

        Parameters
        ----------
        `show_ticks` : `bool`, optional
            Whether to show the grid ticks.
        `show_labels` : `bool`, optional
            Whether to show the atom labels.
        """
        if not self.grid:
            return "Grid not yet generated. Run the `build` method first."

        _, ax = plt.subplots()
        ax.set_aspect("equal")

        gxs = self.structure.cell.lengths()[0] / self.ngx
        gys = self.structure.cell.lengths()[1] / self.ngy

        for gy in range(self.ngy):
            for gx in range(self.ngx):
                ax.add_patch(
                    Rectangle(
                        (gx * gxs, gy * gys),
                        gxs,
                        gys,
                        fill=None,
                        edgecolor="black",
                        linewidth=0.5,
                    )
                )

        ax.set_xlim(0, self.structure.cell.lengths()[0])
        ax.set_ylim(0, self.structure.cell.lengths()[1])

        if show_ticks:
            ax.set_xticks(np.arange(0, self.structure.cell.lengths()[0], gxs))
            ax.set_yticks(np.arange(0, self.structure.cell.lengths()[1], gys))
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        inter_layer_height = 0.5 * self.structure.cell.lengths()[2]

        for atom in self.structure:
            x, y, z = atom.position
            if z < inter_layer_height:
                ax.plot(x, y, "o", color="blue")
                if show_labels:
                    ax.text(x, y - 0.1, atom.index, fontsize=12, ha="center", va="center")
            else:
                ax.plot(x + 0.05, y, "o", color="red")
                if show_labels:
                    ax.text(x, y + 0.1, atom.index, fontsize=12, ha="center", va="center")

        plt.show()

    def plot_potential(self):
        """Plot the potential over the atoms."""
        scaled = self.structure.get_scaled_positions()
        x, y = scaled[:, 0], scaled[:, 1]
        V = np.zeros(self.natoms)
        for ai in range(self.natoms):
            V[ai] = self[4][ai, ai]
        plt.scatter(x, y, c=V, cmap="rainbow", s=20)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("On-site potential")
        plt.show()

    def plot_bands(
        self,
        high_sym_points: dict[str, t.Sequence[float]],
        k_path: str,
        points_per_segment: int,
        use_sparse_solver=False,
        sparse_solver_params: dict | None = None,
        use_mpi=False,
    ):
        """Plot the band structure.

        Parameters
        ----------
        `high_sym_points` : `dict[str, t.Sequence]`
            Dictionary of high-symmetry points.
        `k_path` : `str`
            Path of the k-points.
        `points_per_segment` : `int`
            Number of points per segment.
        `use_sparse_solver` : `bool`, optional
            Whether to use the sparse solver.
        `sparse_solver_params` : `dict`, optional
            Parameters for the sparse solver.
        `use_mpi` : `bool`, optional
            Whether to use the MPI parallelization.
        """
        path = k_path.split()
        segments = np.array([high_sym_points[k] for k in path])
        k_points = self._get_k_points(segments, points_per_segment)
        distances, bands = self.get_band_structure(
            k_points,
            use_sparse_solver,
            sparse_solver_params,
            use_mpi,
        )

        for band in bands.T:
            plt.plot(distances, band)

        tick_positions = np.cumsum(
            np.linalg.norm(np.diff(np.array(segments), axis=0, prepend=0), axis=1)
        )

        for x in tick_positions[1:-1]:
            plt.axvline(x, c="k", ls="--", lw=0.5)

        plt.xlim(distances[0], distances[-1])
        plt.xticks(tick_positions, path)
        plt.ylim(np.min(bands), np.max(bands))
        plt.ylabel("Energy (eV)")
        plt.show()

    def _get_search_grid(self) -> list[list[list[int]]]:
        """Create a search grid for a given structure.

        The search grid is a 2D list of lists, where each list contains the indices
        of atoms in a grid cell. The grid cells are defined by the search distance.
        Grid cell sizes are extended to perfectly fit the cell.

        Raises
        ------
        `ValueError`
            If the search distance is zero.

        Returns
        -------
        `list[list[list[int]]]`
            The search grid with atom indices assigned by their coordinates.
        """
        if self.threshold == 0:
            raise ValueError("search distance is zero; use distances index > 0")

        a, b, _ = self.structure.cell.lengths()

        # calculate the number of grid squares in each dimension
        ngx = int(np.ceil(a / self.threshold)) - 1
        ngy = int(np.ceil(b / self.threshold)) - 1
        ngx = 1 if ngx == 0 else ngx
        ngy = 1 if ngy == 0 else ngy

        # calculate adjusted grid cell size in each dimension to perfectly fit the cell
        gxs = a / ngx  # grid cell size in x
        gys = b / ngy  # grid cell size in y

        # create the grid
        grid: list[list[list[int]]] = [[[] for _ in range(ngx)] for _ in range(ngy)]

        # assign atoms to grid cells
        for ai, atom in enumerate(self.structure):
            x, y, _ = atom.position
            gx = int(x / gxs)
            gy = int(y / gys)
            gx = ngx - 1 if gx == ngx else gx
            gy = ngy - 1 if gy == ngy else gy
            grid[gy][gx].append(ai)

        return grid

    def _compute_coupling_for_grid_cell(self, gy: int, gx: int):
        """Compute coupling parameters for a grid cell.

        Parameters
        ----------
        `gy` : `int`
            Index of the grid cell in the y direction.
        `gx` : `int`
            Index of the grid cell in the x direction.
        """
        for ny, nx in itertools.product(range(-1, 2), range(-1, 2)):
            lgy = (gy + ny) % self.ngy
            lgx = (gx + nx) % self.ngx
            self._compute_coupling_with_nearest_grid_cells(gy, gx, ny, nx, lgy, lgx)

    def _compute_coupling_with_nearest_grid_cells(
        self,
        gy: int,
        gx: int,
        ny: int,
        nx: int,
        lgy: int,
        lgx: int,
    ):
        """Compute coupling between atoms in a grid cell and its nearest neighbors.

        Parameters
        ----------
        `gy` : `int`
            Index of the grid cell in the y direction.
        `gx` : `int`
            Index of the grid cell in the x direction.
        `ny` : `int`
            Neighboring grid cell shift in the y direction.
        `nx` : `int`
            Neighboring grid cell shift in the x direction.
        `lgy` : `int`
            Index of the neighboring grid cell in the y direction,
            cycled w.r.t cell boundaries.
        `lgx` : `int`
            Index of the neighboring grid cell in the x direction,
            cycled w.r.t cell boundaries.
        """
        for ai in self.grid[gy][gx]:
            for aj in self.grid[lgy][lgx]:
                if ai < aj:
                    self._compute_coupling_with_nearest_neighbors(ai, aj, gy, gx, ny, nx)

    def _compute_coupling_with_nearest_neighbors(
        self,
        ai: int,
        aj: int,
        gy: int,
        gx: int,
        ny: int,
        nx: int,
    ):
        """Compute the hopping parameter between two atoms.

        Parameters
        ----------
        `ai` : `int`
            Index of atom i.
        `aj` : `int`
            Index of atom j.
        `gy` : `int`
            Index of the grid cell in the y direction.
        `gx` : `int`
            Index of the grid cell in the x direction.
        `ny` : `int`
            Neighboring grid cell shift in the y direction.
        `nx` : `int`
            Neighboring grid cell shift in the x direction.
        """
        coords_i = self.structure.positions[ai]
        coords_j, Rx, Ry = self._apply_boundary_conditions(aj, gy, gx, ny, nx)

        displacement = np.round(coords_i - coords_j, 3)
        distance = np.round(np.abs(np.linalg.norm(displacement)), 3)

        if distance <= self.threshold:
            hp = self._get_hopping_parameter(distance)
            Ri = self.R_index_map[(Rx, Ry)]
            self[Ri][ai, aj] = hp
            Ri = self.R_index_map[(-Rx, -Ry)]
            self[Ri][aj, ai] = hp
            self._interaction_count_dict[ai + 1] += 1
            self._interaction_count_dict[aj + 1] += 1

        if displacement[0] == 0 and displacement[1] == 0 and displacement[2] != 0:
            Ri = self.R_index_map[(0, 0)]
            self[Ri][ai, aj] = self[Ri][aj, ai] = self.interlayer_coupling
            self._interaction_count_dict[ai + 1] += 1
            self._interaction_count_dict[aj + 1] += 1

    def _apply_boundary_conditions(
        self,
        aj: int,
        gy: int,
        gx: int,
        ny: int,
        nx: int,
    ) -> tuple[np.ndarray, int, int]:
        """Apply periodic boundary conditions to the atom coordinates.

        Parameters
        ----------
        `aj` : `int`
            Index of atom j.
        `gy` : `int`
            Index of the grid cell in the y direction.
        `gx` : `int`
            Index of the grid cell in the x direction.
        `ny` : `int`
            Neighboring grid cell shift in the y direction.
        `nx` : `int`
            Neighboring grid cell shift in the x direction.

        Returns
        -------
        `tuple[np.ndarray, int, int]`
            Adjusted coordinates and nearest neighbor shifts.
        """
        coords_j = self.structure.positions[aj]

        Rx, Ry = 0, 0

        # handle boundary conditions in the x
        coords_j, Rx = self._handle_PBC(
            self.ngx,
            gx,
            nx,
            Rx,
            coords_j,
            self.structure.cell.array[0],
        )

        # handle boundary conditions in the y
        coords_j, Ry = self._handle_PBC(
            self.ngy,
            gy,
            ny,
            Ry,
            coords_j,
            self.structure.cell.array[1],
        )

        return coords_j, Rx, Ry

    def _handle_PBC(
        self,
        ng: int,
        g: int,
        n: int,
        R: int,
        coords_j: np.ndarray,
        v: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Adjust atom coordinates w.r.t the periodic boundary
        conditions in the supplied direction.

        Parameters
        ----------
        `ng` : `int`
            Number of grid cells in the handled direction.
        `g` : `int`
            Index of the grid cell.
        `n` : `int`
            Neighboring grid cell shift in the handled direction.
        `R` : `int`
            R vector component in the handled direction.
        `coords_j` : `np.ndarray`
            Coordinates of atom j.
        `v` : `np.ndarray`
            Lattice vector component.

        Returns
        -------
        `tuple[np.ndarray, int]`
            Adjusted coordinates of atom j and R vector component.
        """
        if n != 0:
            if g == 0 and g + n < g and (g + n) % ng == ng - 1:
                R -= 1
                coords_j = coords_j - v
            elif g == ng - 1 and g + n > g and (g + n) % ng == 0:
                R += 1
                coords_j = coords_j + v
        return coords_j, R

    def _get_hopping_parameter(self, distance: float) -> float:
        """Get the hopping parameter corresponding to the nearest neighbor distance.

        Parameters
        ----------
        `distance` : `float`
            Distance between two atoms.

        Returns
        -------
        `float`
            Hopping parameter corresponding to the distance.
        """
        return next(
            (self.hopping_parameters[i] for i, d in enumerate(self.distances) if distance == d),
            0.0,
        )

    def _get_k_points(
        self,
        segments: np.ndarray,
        points_per_segment: int = 5,
    ) -> np.ndarray:
        """Generate k-points for the band structure calculation.

        Parameters
        ----------
        `segments` : `np.ndarray`
            List of high-symmetry points.
        `points_per_segment` : `int`, optional
            Number of points per segment.

        Returns
        -------
        `np.ndarray`
            List of k-points.
        """
        k_points = []
        for i in range(len(segments) - 1):
            start, end = segments[i], segments[(i + 1) % len(segments)]
            kx = np.linspace(start[0], end[0], points_per_segment, endpoint=True)
            ky = np.linspace(start[1], end[1], points_per_segment, endpoint=True)
            kz = np.linspace(start[2], end[2], points_per_segment, endpoint=True)
            points = np.array(list(zip(kx, ky, kz)))
            k_points.append(points)
        return np.concatenate(k_points)

    def _get_band_structure_mpi(
        self,
        k_points: np.ndarray,
        use_sparse_solver: bool,
        sparse_solver_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the band structure using the MPI parallelization.

        Parameters
        ----------
        `k_points` : `np.ndarray`
            List of k-points.
        `use_sparse_solver` : `bool`
            Whether to use the sparse solver.
        `sparse_solver_params` : `dict`
            Parameters for the sparse solver.

        Returns
        -------
        `tuple[np.ndarray, np.ndarray]`
            Distances and band structure.
        """
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        local_k_points = np.array_split(k_points, size)[rank]

        local_band_structure = []
        for k_point in local_k_points:
            H_k = self.get_kamiltonian(k_point)
            eigenvalues = H_k.get_eigenvalues(use_sparse_solver, sparse_solver_params)
            local_band_structure.append(eigenvalues)

        all_band_structure = comm.gather(local_band_structure, root=0)

        if rank == 0:
            band_structure = [item for sublist in all_band_structure for item in sublist]  # type: ignore
            distances = np.cumsum(np.linalg.norm(np.diff(k_points, axis=0, prepend=0), axis=1))
            return distances, np.array(band_structure)

        return np.array([]), np.array([])

    def _get_band_structure_multiprocessing(
        self,
        k_points: np.ndarray,
        use_sparse_solver: bool,
        sparse_solver_params: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the band structure using the multiprocessing parallelization.

        Parameters
        ----------
        `k_points` : `np.ndarray`
            List of k-points.
        `use_sparse_solver` : `bool`
            Whether to use the sparse solver.
        `sparse_solver_params` : `dict`
            Parameters for the sparse solver.

        Returns
        -------
        `tuple[np.ndarray, np.ndarray]`
            Distances and band structure.
        """
        from multiprocessing import Pool, cpu_count

        pool = Pool(cpu_count())
        results = pool.map(self.get_kamiltonian, k_points)
        pool.close()
        pool.join()

        band_structure = []
        for H_k in results:
            eigenvalues = H_k.get_eigenvalues(use_sparse_solver, sparse_solver_params)
            band_structure.append(eigenvalues)

        distances = np.cumsum(np.linalg.norm(np.diff(k_points, axis=0, prepend=0), axis=1))

        return distances, np.array(band_structure)

    def __getitem__(self, i: int) -> sparse.lil_matrix:
        return self.matrix[i]

    def __iter__(self) -> t.Iterator[sparse.lil_matrix]:
        return iter(self.matrix)

    def __str__(self) -> str:
        return (
            f"TightBindingHamiltonian({self.structure.info.get('label', self.structure.symbols)})"
        )
