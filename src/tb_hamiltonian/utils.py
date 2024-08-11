from __future__ import annotations

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read


def get_structure(
    initial_structure: Atoms,
    repetitions: t.Tuple[int, int, int] | None = None,
    lengths: t.Tuple[float, float, float] | None = None,
    structure_filepath: Path | None = None,
    structure_file_format: str = "vasp",
) -> Atoms:
    """Create a sorted structure object.

    Create a structure object from a unit cell, optionally extended by
    repetitions provided or derived from provided lengths. Alternatively,
    create a structure directly from a provided file. In either case,
    atoms are sorted by x, y, z, in order.

    Parameters
    ----------
    `initial_structure` : `ase.Atoms`
        Initial structure object.
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
    if initial_structure:
        if repetitions is not None:
            nx, ny, nz = repetitions
            structure = initial_structure.repeat((nx, ny, nz))  # type: ignore
            return sort_atoms(structure)  # type: ignore
        elif lengths is not None:
            nx, ny, nz = lengths // initial_structure.cell.lengths()  # type: ignore
            structure = initial_structure.repeat([int(i) or 1 for i in (nx, ny, nz)])  # type: ignore
            return sort_atoms(structure)  # type: ignore
        else:
            return sort_atoms(initial_structure)  # type: ignore
    elif structure_filepath:
        try:
            structure = read(structure_filepath, format=structure_file_format)
            return sort_atoms(structure)  # type: ignore
        except Exception as err:
            raise err
    else:
        raise ValueError(
            "Either `unit_cell_filepath` or `structure_filepath` must be provided."
        )


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


def zipped(params: dict):
    from itertools import product

    for onsite_term, potential_type, alpha in product(
        params.get("onsite_term", [0.0]) or [0.0],
        params.get("potential_type", ["null"]) or ["null"],
        params.get("alpha", [[1.0]]) or [[1.0]],
    ):
        base_params = {
            "onsite_term": onsite_term,
            "potential_type": potential_type,
            "alpha": alpha,
        }
        if potential_type == "null":
            yield base_params
        elif potential_type in ("kronig-penney", "sine"):
            for amplitude in params["potential_params"].get("amplitude", []):
                yield {
                    **base_params,
                    "amplitude": amplitude,
                }
        elif potential_type in ("triangular", "rectangular"):
            for amplitude, width, height in product(
                params["potential_params"].get("amplitude", []),
                params["potential_params"].get("width", [0.5]) or [0.5],
                params["potential_params"].get("height", [0.0]) or [0.0],
            ):
                yield {
                    **base_params,
                    "amplitude": amplitude,
                    "width": width,
                    "height": height or 2 * width,
                }


def generate_k_points(
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


class BandStructure:
    """Band structure utility class.

    Attributes
    ----------
    `high_sym_points` : `dict[str, t.Sequence[float]]`
        High symmetry points in the Brillouin zone.
    `path` : `str | t.Sequence[str]`
        K-point path through the Brillouin zone.
    `distances` : `np.ndarray`
        K-point distances.
    `eigenvalues` : `np.ndarray`
        Eigenvalues of the band structure.
    """

    def __init__(
        self,
        high_sym_points: dict[str, t.Sequence[float]],
        path: str | t.Sequence[str],
        distances: np.ndarray,
        eigenvalues: np.ndarray,
    ):
        """`BandStructure` constructor."""
        self.path = path.split() if isinstance(path, str) else path
        self.high_sym_points = high_sym_points
        self.distances = distances
        self.eigenvalues = eigenvalues

    def plot(
        self,
        title="Band Structure",
        mode: t.Literal["line", "scatter"] = "line",
        plot_params: dict | None = None,
        fig_params: dict | None = None,
    ) -> plt.Axes:
        """Plot the band structure.

        Parameters
        ----------
        `mode` : `str`, optional
            Plotting mode. Choose from 'line' or 'scatter'.
        `plot_params` : `dict`, optional
            Plotting parameters.
        `fig_params` : `dict`, optional
            Figure parameters.

        Returns
        -------
        `plt.Axes`
            Band structure plot axes object.
        """
        segments = np.array([self.high_sym_points[k] for k in self.path])

        if fig_params is None:
            fig_params = {
                "figsize": (8, 6),
                "ylim": (np.min(self.eigenvalues), np.max(self.eigenvalues)),
            }

        if "figsize" not in fig_params:
            fig_params["figsize"] = (8, 6)

        if plot_params is None:
            plot_params = {}

        _, ax = plt.subplots(figsize=fig_params.pop("figsize"))

        for eigen_col in self.eigenvalues.T:
            if mode == "line":
                ax.plot(self.distances, eigen_col, **plot_params)
            elif mode == "scatter":
                ax.scatter(self.distances, eigen_col, **plot_params)
            else:
                raise ValueError("Invalid mode. Choose 'line' or 'scatter'.")

        tick_positions = np.cumsum(
            np.linalg.norm(np.diff(np.array(segments), axis=0, prepend=0), axis=1)
        )

        for x in tick_positions[1:-1]:
            ax.axvline(x, c="k", ls="--", lw=0.5)

        ax.set(
            title=title,
            xlim=(self.distances[0], self.distances[-1]),
            xticks=tick_positions,
            xticklabels=self.path,
            ylabel="Energy (eV)",
            **fig_params,
        )

        return ax

    def __repr__(self):
        return f"""BandStructure({' -> '.join(self.path)})"""
