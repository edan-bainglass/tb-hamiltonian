from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read


def get_structure(
    unit_cell_filepath: Path | None = None,
    unit_cell_file_format: str = "vasp",
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
            structure = unit_cell.repeat((nx, ny, nz))  # type: ignore
            return sort_atoms(structure)  # type: ignore
        elif lengths is not None:
            nx, ny, nz = lengths // unit_cell.cell.lengths()  # type: ignore
            structure = unit_cell.repeat([int(i) or 1 for i in (nx, ny, nz)])  # type: ignore
            return sort_atoms(structure)  # type: ignore
        else:
            return sort_atoms(unit_cell)  # type: ignore
    elif structure_filepath:
        try:
            structure = read(structure_filepath, format=structure_file_format)
            return sort_atoms(structure)  # type: ignore
        except Exception as err:
            raise err
    else:
        raise ValueError("Either `unit_cell_filepath` or `structure_filepath` must be provided.")


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
