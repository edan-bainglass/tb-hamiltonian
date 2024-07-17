from pathlib import Path

from aiida_workgraph import WorkGraph, task

import calculations


@task.graph_builder()
def compute_bands(
    paths: dict,
    repetitions: list,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    potential_type: str,
    potential_params: dict,
    onsite_term: float,
    alpha: list,
    band_params: dict,
) -> WorkGraph:
    wg = WorkGraph("BLG_bands")
    wg.add_task(
        calculations.define_structure,
        name="define_structure",
        paths=paths,
        repetitions=repetitions,
    )
    wg.add_task(
        calculations.build_hamiltonian,
        name="build_hamiltonian",
        structure=wg.tasks["define_structure"].outputs["result"],
        workdir=paths["output_path"],
        distances=distances,
        nearest_neighbor=nearest_neighbor,
        hopping_parameters=hopping_parameters,
        interlater_coupling=interlayer_coupling,
    )
    wg.add_task(
        calculations.apply_onsite_term,
        name="apply_onsite_term",
        H_file=wg.tasks["build_hamiltonian"].outputs["result"],
        potential_type=potential_type,
        potential_params=potential_params,
        workdir=paths["output_path"],
        onsite_term=onsite_term,
        alpha=alpha,
    )
    wg.add_task(
        calculations.get_band_structure,
        name="get_band_structure",
        H_file=wg.tasks["apply_onsite_term"].outputs["H_file"],
        workdir=wg.tasks["apply_onsite_term"].outputs["workdir"],
        band_params=band_params,
    )
    return wg


@task.graph_builder()
def sweep_cell_sizes(
    input_path: Path,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    potential_type: str,
    potential_params: dict,
    onsite_term: float,
    alpha: list,
    band_params: dict,
    sizes: list,
) -> WorkGraph:
    wg = WorkGraph("sweep_cell_size")
    for n in sizes:
        workdir = input_path / f"rep_{n}x{n}/nn_{nearest_neighbor}"
        workdir.mkdir(parents=True, exist_ok=True)
        wg.add_task(
            compute_bands(
                {
                    "input_path": input_path.as_posix(),
                    "output_path": workdir.as_posix(),
                },
                [n, n, 1],
                distances,
                nearest_neighbor,
                hopping_parameters,
                interlayer_coupling,
                potential_type,
                potential_params,
                onsite_term,
                alpha,
                band_params,
            )
        )
    return wg
