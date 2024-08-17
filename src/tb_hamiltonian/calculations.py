from __future__ import annotations

from aiida_workgraph import task
from ase import Atoms

from tb_hamiltonian.hamiltonian import TBHamiltonian
from tb_hamiltonian.utils import BandStructure


@task.pythonjob(
    outputs=[
        {"name": "structure"},
        {"name": "H"},
    ]
)
def build_hamiltonian(
    initial_structure: Atoms,
    repetitions: list = None,
    lengths: list = None,
    structure_label: str = "",
    distances: list = None,
    nearest_neighbor: int = 1,
    hopping_parameters: list = None,
    interlayer_coupling: float = 0.0,
    use_mpi: bool = False,
) -> dict:
    from tb_hamiltonian.hamiltonian import TBHamiltonian
    from tb_hamiltonian.utils import get_structure

    structure = get_structure(
        initial_structure,
        repetitions=repetitions,
        lengths=lengths,
    )
    structure.info["label"] = structure_label

    filename = "POSCAR"
    structure.write(filename, format="vasp")

    H = TBHamiltonian(
        structure=structure,
        nearest_neighbor=nearest_neighbor,
        distances=distances or [0.0],
        hopping_parameters=hopping_parameters or [0.0],
        interlayer_coupling=interlayer_coupling,
    )
    H.build()
    H.write_to_file(use_mpi=use_mpi)
    return {
        "structure": structure,
        "H": H,
    }


@task.pythonjob()
def apply_onsite_term(
    H: TBHamiltonian,
    potential_type: str,
    potential_params: dict,
    onsite_term: float = 0.0,
    alpha: list = None,
    use_mpi: bool = False,
) -> TBHamiltonian:
    from tb_hamiltonian.potentials import PotentialFactory

    onsite_H = H.copy()
    potential = PotentialFactory(potential_type)
    potential.params = potential_params
    onsite_H.update_onsite_terms(onsite_term, potential, alpha)
    onsite_H.write_to_file(use_mpi=use_mpi)
    return onsite_H


@task.pythonjob()
def get_band_structure(
    H: TBHamiltonian,
    band_params: dict,
    use_mpi: bool = False,
) -> BandStructure:
    return H.get_band_structure(
        **band_params,
        use_mpi=use_mpi,
    )
