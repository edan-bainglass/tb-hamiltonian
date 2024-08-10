from aiida_workgraph import task
from ase import Atoms

from tb_hamiltonian.hamiltonian import TBHamiltonian


@task.pythonjob()
def define_structure(
    initial_structure: Atoms,
    repetitions: list = None,
    structure_label: str = "",
) -> Atoms:
    from tb_hamiltonian.utils import get_structure

    structure = get_structure(
        initial_structure,
        repetitions=repetitions or [1, 1, 1],
    )
    structure.info["label"] = structure_label
    structure.write("POSCAR", format="vasp")
    return structure


@task.pythonjob()
def build_hamiltonian(
    structure: Atoms,
    distances: list = None,
    nearest_neighbor: int = 1,
    hopping_parameters: list = None,
    interlayer_coupling: float = 0.0,
    use_mpi: bool = False,
) -> TBHamiltonian:
    from tb_hamiltonian.hamiltonian import TBHamiltonian

    H = TBHamiltonian(
        structure=structure,
        nearest_neighbor=nearest_neighbor,
        distances=distances or [0.0],
        hopping_parameters=hopping_parameters or [0.0],
        interlayer_coupling=interlayer_coupling,
    )
    H.build()
    H.write_to_file(use_mpi=use_mpi)
    return H


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


@task.pythonjob(
    outputs=[
        {"name": "distances"},
        {"name": "eigenvalues"},
        {"name": "fig"},
    ],
)
def get_band_structure(
    H: TBHamiltonian,
    band_params: dict,
) -> dict:
    import numpy as np
    from PIL import Image

    H.plot_bands(**band_params)
    distances = np.load("distances.npy")
    eigenvalues = np.load("eigenvalues.npy")
    fig = Image.open(band_params.get("fig_filename", "bands.png"))

    return {
        "distances": distances,
        "eigenvalues": eigenvalues,
        "fig": fig,
    }
