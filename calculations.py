import pickle
from pathlib import Path

from aiida import orm
from aiida_workgraph import task
from tb_hamiltonian.hamiltonian import TBHamiltonian
from tb_hamiltonian.potentials import PotentialFactory
from tb_hamiltonian.utils import get_structure


@task.calcfunction()
def define_structure(
    paths: dict,
    repetitions: list = None,
    structure_label: str = "",
) -> orm.StructureData:
    structure = get_structure(
        Path(paths["input_path"]) / "POSCAR",
        repetitions=repetitions or [1, 1, 1],
    )
    structure.info["label"] = structure_label
    structure.write(Path(paths["output_path"]).parent / "POSCAR", format="vasp")
    return orm.StructureData(ase=structure)


@task.calcfunction()
def build_hamiltonian(
    structure: orm.StructureData,
    workdir: str,
    nearest_neighbor: int = 1,
    distances: list = None,
    hopping_parameters: list = None,
    interlater_coupling: float = 0.0,
    use_mpi: bool = False,
) -> orm.SinglefileData:
    H = TBHamiltonian(
        structure=structure.get_ase(),
        nearest_neighbor=nearest_neighbor.value,
        distances=distances.get_list() or [0.0],
        hopping_parameters=hopping_parameters.get_list() or [0.0],
        interlayer_coupling=interlater_coupling.value,
    )
    H.build()
    H.write_to_file(Path(workdir.value), use_mpi=use_mpi.value)
    path = Path(workdir.value) / "H.pkl"
    with path.open(mode="wb") as file:
        pickle.dump(H, file)
    return orm.SinglefileData(path.absolute())


@task.calcfunction(
    outputs=[
        {"name": "H_file"},
        {"name": "workdir"},
    ]
)
def apply_onsite_term(
    H_file: orm.SinglefileData,
    potential_type: str,
    potential_params: dict,
    workdir: str,
    onsite_term: float = 0.0,
    alpha: list = None,
    use_mpi: bool = False,
) -> dict:
    with H_file.open(mode="rb") as file:
        H: TBHamiltonian = pickle.load(file)
    H_onsite = H.copy()
    potential = PotentialFactory(potential_type.value)
    potential.params = potential_params
    H_onsite.update_onsite_terms(onsite_term.value, potential, alpha.get_list() or [1.0])
    path: Path = Path(workdir.value) / potential_type.value
    if potential_type != "null":
        path /= f"amplitude_{potential_params['amplitude']}"
        if "width" in potential_params:
            path /= f"width_{potential_params['width']}"
        if "height" in potential_params:
            path /= f"height_{potential_params['height']}"
    path.mkdir(parents=True, exist_ok=True)
    H_onsite.write_to_file(path, use_mpi=use_mpi)
    pickle_path = path / "H.pkl"
    with pickle_path.open(mode="wb") as file:
        pickle.dump(H_onsite, file)
    return {
        "H_file": orm.SinglefileData(pickle_path.absolute()),
        "workdir": orm.Str(path.absolute().as_posix()),
    }


@task.calcfunction()
def get_band_structure(
    H_file: orm.SinglefileData,
    workdir: str,
    band_params: dict,
) -> orm.SinglefileData:
    with H_file.open(mode="rb") as file:
        H: TBHamiltonian = pickle.load(file)
    workdir_path = Path(workdir.value)
    filepath: Path = workdir_path / band_params.get("fig_filename", "bands.png")
    H.plot_bands(**{"workdir": workdir_path, **band_params.get_dict()})
    return orm.SinglefileData(filepath.absolute())
