from __future__ import annotations

from curses import endwin
import typing as t
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from matplotlib.patches import Rectangle
from scipy.sparse import lil_matrix

from potentials import PotentialFactory, PotentialFunction


def get_structure(
    unit_cell_filepath: Path = Path("."),
    unit_cell_file_format: str = "vasp",
    repetitions: t.Tuple[int, int, int] | None = None,
    lengths: t.Tuple[float, float, float] | None = None,
    structure_filepath: Path = Path("."),
    structure_file_format: str = "vasp",
) -> Atoms:
    """Create a sorted structure object.

    Create a structure object from a unit cell, optionally extended by
    repetitions provided or derived from provided lengths. Alternatively,
    create a structure directly from a provided file. In either case,
    atoms are sorted by x, y, z, in order.

    Parameters
    ----------
    `unit_cell_filepath` : `Path`, optional
        Path to the unit cell file.
    `unit_cell_file_format` : `str`, optional
        File format of the unit cell file.
    `repetitions` : `tuple[int, int, int]`, optional
        Number of repetitions in the x, y, z directions.
    `lengths` : `tuple[float, float, float]`, optional
        Lengths of the supercell in the x, y, z directions.
    `structure_filepath` : `Path`, optional
        Path to the structure file.
    `structure_file_format` : `str`, optional
        File format of the structure file.

    Returns
    -------
    `ase.Atoms`
        Structure as `ase.Atoms` object with atoms sorted
        by x, y, and z, in order.
    """
    if unit_cell_filepath:
        unit_cell = read(unit_cell_filepath, format=unit_cell_file_format)
        if repetitions is not None:
            nx, ny, nz = repetitions
            return unit_cell.repeat((nx, ny, nz))  # type: ignore
        elif lengths is not None:
            nx, ny, nz = lengths // unit_cell.cell.lengths()  # type: ignore
            return unit_cell.repeat([int(i) for i in (nx, ny, nz)])  # type: ignore
        else:
            return sort_atoms(unit_cell)  # type: ignore
    elif structure_filepath:
        try:
            structure = read(structure_filepath, format=structure_file_format)
            return sort_atoms(structure)  # type: ignore
        except Exception as err:
            raise err
    else:
        raise ValueError("Either `unit_cell` or `structure_filepath` must be provided.")


def sort_atoms(atoms):
    """Sort atoms by their positions in the x, y, z directions.

    Parameters
    ----------
    `atoms` : `ase.Atoms`
        Atoms object to be sorted.

    Returns
    -------
    `ase.Atoms`
        Sorted Atoms object.
    """
    return atoms[
        np.lexsort(
            (
                atoms.positions[:, 0],
                atoms.positions[:, 1],
                atoms.positions[:, 2],
            )
        )
    ]


class TightBindingHamiltonian:
    """docstring"""

    def __init__(
        self,
        structure: Atoms,
        nearest_neighbor: int,
        distances: list[float],
        hopping_parameters: list[float],
        interlayer_coupling: float,
    ):
        """docstring"""
        self.structure = structure
        self.threshold = distances[nearest_neighbor]
        self.distances = distances
        self.hopping_parameters = hopping_parameters
        self.interlayer_coupling = interlayer_coupling
        self.natoms = len(structure)
        self.R = [np.array([i, j, 0]) for i in range(-1, 2) for j in range(-1, 2)]
        self.matrix = [lil_matrix((self.natoms, self.natoms)) for _ in range(len(self.R))]
        self.grid = self._get_search_grid()
        self.ngx, self.ngy = len(self.grid[0]), len(self.grid)
        self._interaction_count_dict = OrderedDict.fromkeys(range(1, self.natoms + 1), 0)

        self.R_index_map: dict[tuple[int, int], int] = {
            (int(Rx), int(Ry)): i for i, (Rx, Ry, _) in enumerate(self.R)
        }

    def build(self):
        for gy in range(self.ngy):
            for gx in range(self.ngx):
                # consider only first nearest neighboring grid cells
                for ny in range(-1, 2):
                    for nx in range(-1, 2):
                        # loop over atoms in the current grid cell
                        for ai in self.grid[gy][gx]:
                            # define local grid indices
                            lgy = (gy + ny) % self.ngy  # wrap around cell
                            lgx = (gx + nx) % self.ngx  # wrap around cell

                            # loop over atoms in the neighboring grid cell
                            for aj in self.grid[lgy][lgx]:
                                # H is symmetric, so we compute
                                # only the upper triangular matrix
                                if ai < aj:
                                    coords_i = self.structure.positions[ai]
                                    coords_j = self.structure.positions[aj]

                                    # used later to derive the R index of H
                                    Rx, Ry = 0, 0

                                    if nx != 0:
                                        # handle boundary conditions in the x
                                        coords_j, Rx = self._handle_PBC(
                                            self.ngx,
                                            gx,
                                            nx,
                                            Rx,
                                            coords_j,
                                            self.structure.cell.array[0],
                                        )

                                    if ny != 0:
                                        # handle boundary conditions in the y
                                        coords_j, Ry = self._handle_PBC(
                                            self.ngy,
                                            gy,
                                            ny,
                                            Ry,
                                            coords_j,
                                            self.structure.cell.array[1],
                                        )

                                    v = np.round(coords_i - coords_j, 3)  # displacement vector

                                    # update H with hopping parameters
                                    # if ai-aj distance is within threshold
                                    d = np.round(np.abs(np.linalg.norm(v)), 3)
                                    if d <= self.threshold:
                                        # pick hopping parameter based on distance
                                        for i, distance in enumerate(self.distances):
                                            if d == distance:
                                                hp = self.hopping_parameters[i]
                                                break

                                        Ri = self.R_index_map[(Rx, Ry)]
                                        self[Ri][ai, aj] = hp

                                        Ri = self.R_index_map[(-Rx, -Ry)]
                                        self[Ri][aj, ai] = hp

                                        self._interaction_count_dict[ai + 1] += 1
                                        self._interaction_count_dict[aj + 1] += 1

                                    # update H with interlayer coupling
                                    if v[0] == 0 and v[1] == 0 and v[2] != 0:
                                        Ri = self.R_index_map[(0, 0)]
                                        self[Ri][ai, aj] = self[Ri][aj, ai] = (
                                            self.interlayer_coupling
                                        )
                                        self._interaction_count_dict[ai + 1] += 1
                                        self._interaction_count_dict[aj + 1] += 1

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

    def write_to_file(
        self,
        path: Path = Path("output/example"),
        filename: str = "TG_Hr.dat",
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

    def plot_matrix(self, R_index: int = 4, start=0, end=-1, step=1, figsize=(5, 5)):
        """Plot the Hamiltonian matrix."""
        Hr = self[R_index].toarray()
        end = self.natoms if end < start else end
        plt.figure(figsize=figsize)
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

    def plot_grid(self):
        """Plot the search grid."""

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
        ax.set_xticks(np.arange(0, self.structure.cell.lengths()[0], gxs))
        ax.set_yticks(np.arange(0, self.structure.cell.lengths()[1], gys))
        ax.grid(True)

        inter_layer_height = 0.5 * self.structure.cell.lengths()[2]

        for atom in self.structure:
            x, y, z = atom.position
            if z < inter_layer_height:
                ax.plot(x, y, "o", color="blue")
                ax.text(x, y - 0.1, atom.index, fontsize=12, ha="center", va="center")
            else:
                ax.plot(x + 0.05, y, "o", color="red")
                ax.text(x, y + 0.1, atom.index, fontsize=12, ha="center", va="center")

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
            grid[gy][gx].append(ai)

        return grid

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
            number of grid cells in a given direction
        `g` : `int`
            index of grid cell in a given direction
        `n` : `int`
            index of neighboring grid cell in a given direction
        `R` : `int`
            R vector component in a given direction
        `coords_j` : `np.ndarray`
            coordinates of atom j
        `v` : `np.ndarray`
            lattice vector in a given direction

        Returns
        -------
        `tuple[np.ndarray, int]`
            adjusted coordinates of atom j and R vector component
        """
        if g == 0 and g + n < g and (g + n) % ng == ng - 1:
            R -= 1
            coords_j = coords_j - v
        elif g == ng - 1 and g + n > g and (g + n) % ng == 0:
            R += 1
            coords_j = coords_j + v
        return coords_j, R

    def __getitem__(self, key: int) -> lil_matrix:
        return self.matrix[key]

    def __iter__(self):
        return iter(self.matrix)

    def __str__(self) -> str:
        return (
            f"TightBindingHamiltonian({self.structure.info.get('label', self.structure.symbols)})"
        )
