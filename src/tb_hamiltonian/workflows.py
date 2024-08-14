from __future__ import annotations

import typing as t

from aiida_workgraph import WorkGraph, task
from ase import Atoms

from tb_hamiltonian import calculations
from tb_hamiltonian.utils import zipped


@task.graph_builder()
def compute_bands(
    structure_label: str,
    initial_structure: Atoms,
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
    metadata: dict | None = None,
    suffix: str = "",
) -> WorkGraph:
    suffix = f"_{suffix}" if suffix else ""
    wg = WorkGraph(f"Bands{suffix}")
    task_metadata: dict = metadata.get("define_structure", {})
    wg.add_task(
        calculations.define_structure,
        name=f"define_structure{suffix}",
        initial_structure=initial_structure,
        repetitions=repetitions,
        structure_label=structure_label,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("build_hamiltonian", {})
    wg.add_task(
        calculations.build_hamiltonian,
        name=f"build_hamiltonian{suffix}",
        structure=wg.tasks[f"define_structure{suffix}"].outputs["result"],
        distances=distances,
        nearest_neighbor=nearest_neighbor,
        hopping_parameters=hopping_parameters,
        interlayer_coupling=interlayer_coupling,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("apply_onsite_term", {})
    wg.add_task(
        calculations.apply_onsite_term,
        name=f"apply_onsite_term{suffix}",
        H=wg.tasks[f"build_hamiltonian{suffix}"].outputs["result"],
        potential_type=potential_type,
        potential_params=potential_params,
        onsite_term=onsite_term,
        alpha=alpha,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("get_band_structure", {})
    wg.add_task(
        calculations.get_band_structure,
        name=f"get_band_structure{suffix}",
        H=wg.tasks[f"apply_onsite_term{suffix}"].outputs["result"],
        band_params=band_params,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    return wg


@task.graph_builder()
def sweep_cell_sizes(
    structure_label: str,
    initial_structure: Atoms,
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
    metadata: dict | None = None,
) -> WorkGraph:
    wg = WorkGraph("BandsCellSizeSweep")
    for size in sizes:
        if isinstance(size, t.Sequence):
            nx, ny = size
        elif isinstance(size, int):
            nx = ny = size
        else:
            raise ValueError(f"Invalid size: {size}")
        suffix = f"{nx}x{ny}"
        wg.add_task(
            compute_bands(
                structure_label,
                initial_structure,
                [nx, ny, 1],
                distances,
                nearest_neighbor,
                hopping_parameters,
                interlayer_coupling,
                potential_type,
                potential_params,
                onsite_term,
                alpha,
                band_params,
                metadata,
                suffix=suffix,
            ),
            name=f"Bands_{suffix}",
        )
    return wg


@task.graph_builder()
def sweep_onsite_parameters(
    structure_label: str,
    initial_structure: Atoms,
    repetitions: list,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    sweep_params: dict,
    band_params: dict,
    metadata: dict | None = None,
) -> WorkGraph:
    wg = WorkGraph("BandsOnsiteParameterSweep")

    task_metadata: dict = metadata.get("define_structure", {})
    wg.add_task(
        calculations.define_structure,
        name="define_structure",
        initial_structure=initial_structure,
        repetitions=repetitions,
        structure_label=structure_label,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )

    task_metadata: dict = metadata.get("build_hamiltonian", {})
    wg.add_task(
        calculations.build_hamiltonian,
        name="build_hamiltonian",
        structure=wg.tasks["define_structure"].outputs["result"],
        distances=distances,
        nearest_neighbor=nearest_neighbor,
        hopping_parameters=hopping_parameters,
        interlayer_coupling=interlayer_coupling,
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )

    for params in zipped(sweep_params):
        task_params = {
            "H": wg.tasks["build_hamiltonian"].outputs["result"],
            "potential_type": params.get("potential_type", "null"),
            "potential_params": {},
        }

        label_parts = []

        if "onsite_term" in params and params["onsite_term"]:
            task_params["onsite_term"] = params["onsite_term"]
            label_parts.append(f"onsite_{params['onsite_term']}")

        label_parts.append(f"{task_params['potential_type']}")

        if "amplitude" in params:
            task_params["potential_params"]["amplitude"] = params["amplitude"]
            label_parts.append(f"amplitude_{params['amplitude']}")
        if "width" in params:
            task_params["potential_params"]["width"] = params["width"]
            label_parts.append(f"width_{params['width']}")
        if "height" in params:
            task_params["potential_params"]["height"] = params["height"]
            label_parts.append(f"height_{params['height']}")
        if "alpha" in params:
            task_params["alpha"] = params["alpha"]
            if len(alpha := params["alpha"]) > 1:
                label_parts.append(f"alpha_{'_'.join(list(map(str, alpha[1:])))}")

        label = "_".join(label_parts).replace(".", "_").replace("-", "_")

        task_metadata: dict = metadata.get("apply_onsite_term", {})
        wg.add_task(
            calculations.apply_onsite_term,
            name=f"apply_onsite_term_{label}",
            **task_params,
            computer=task_metadata.get("computer", "localhost"),
            metadata=task_metadata.get("metadata", {}),
        )

        task_metadata: dict = metadata.get("get_band_structure", {})
        wg.add_task(
            calculations.get_band_structure,
            name=f"get_band_structure_{label}",
            H=wg.tasks[f"apply_onsite_term_{label}"].outputs["result"],
            band_params=band_params,
            computer=task_metadata.get("computer", "localhost"),
            metadata=task_metadata.get("metadata", {}),
        )

    return wg
