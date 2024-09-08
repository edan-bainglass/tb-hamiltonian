from __future__ import annotations

import typing as t

from aiida_workgraph import WorkGraph, task
from ase import Atoms

from tb_hamiltonian import calculations
from tb_hamiltonian.utils import zipped


@task.graph_builder()
def compute_bands(
    *,
    structure_label: str = "",
    initial_structure: Atoms,
    repetitions: list | None = None,
    lengths: list | None = None,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    potential_type: str = "null",
    potential_params: dict,
    onsite_term: float,
    alpha: list,
    band_params: dict,
    metadata: dict | None = None,
    suffix: str = "",
) -> WorkGraph:
    suffix = f"_{suffix}" if suffix else ""
    wg = WorkGraph(f"Bands{suffix}")
    task_metadata: dict = metadata.get("build_hamiltonian", {})
    wg.add_task(
        calculations.build_hamiltonian,
        name=f"build_hamiltonian{suffix}",
        initial_structure=initial_structure,
        repetitions=repetitions,
        lengths=lengths,
        structure_label=structure_label,
        distances=distances,
        nearest_neighbor=nearest_neighbor,
        hopping_parameters=hopping_parameters,
        interlayer_coupling=interlayer_coupling,
        use_mpi=task_metadata.get("options", {}).get("withmpi", False),
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("apply_onsite_term", {})
    wg.add_task(
        calculations.apply_onsite_term,
        name=f"apply_onsite_term{suffix}",
        H=wg.tasks[f"build_hamiltonian{suffix}"].outputs["H"],
        potential_type=potential_type,
        potential_params=potential_params,
        onsite_term=onsite_term,
        alpha=alpha,
        use_mpi=task_metadata.get("options", {}).get("withmpi", False),
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("get_band_structure", {})
    wg.add_task(
        calculations.get_band_structure,
        name=f"get_band_structure{suffix}",
        H=wg.tasks[f"apply_onsite_term{suffix}"].outputs["result"],
        band_params=band_params,
        use_mpi=task_metadata.get("options", {}).get("withmpi", False),
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )
    return wg


@task.graph_builder()
def sweep_cell_sizes(
    *,
    structure_label: str = "",
    initial_structure: Atoms,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    potential_type: str = "null",
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
                structure_label=structure_label,
                initial_structure=initial_structure,
                repetitions=[nx, ny, 1],
                distances=distances,
                nearest_neighbor=nearest_neighbor,
                hopping_parameters=hopping_parameters,
                interlayer_coupling=interlayer_coupling,
                potential_type=potential_type,
                potential_params=potential_params,
                onsite_term=onsite_term,
                alpha=alpha,
                band_params=band_params,
                metadata=metadata,
                suffix=suffix,
            ),
            name=f"Bands_{suffix}",
        )
    return wg


@task.graph_builder()
def sweep_onsite_parameters(
    *,
    structure_label: str = "",
    initial_structure: Atoms,
    repetitions: list | None = None,
    lengths: list | None = None,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    sweep_params: dict,
    band_params: dict,
    metadata: dict | None = None,
) -> WorkGraph:
    wg = WorkGraph("BandsOnsiteParameterSweep")

    task_metadata: dict = metadata.get("build_hamiltonian", {})
    wg.add_task(
        calculations.build_hamiltonian,
        name="build_hamiltonian",
        initial_structure=initial_structure,
        repetitions=repetitions,
        lengths=lengths,
        structure_label=structure_label,
        distances=distances,
        nearest_neighbor=nearest_neighbor,
        hopping_parameters=hopping_parameters,
        interlayer_coupling=interlayer_coupling,
        use_mpi=task_metadata.get("options", {}).get("withmpi", False),
        computer=task_metadata.get("computer", "localhost"),
        metadata=task_metadata.get("metadata", {}),
    )

    for params in zipped(sweep_params):
        task_params = {
            "H": wg.tasks["build_hamiltonian"].outputs["H"],
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
            use_mpi=task_metadata.get("options", {}).get("withmpi", False),
            computer=task_metadata.get("computer", "localhost"),
            metadata=task_metadata.get("metadata", {}),
        )

        task_metadata: dict = metadata.get("get_band_structure", {})
        wg.add_task(
            calculations.get_band_structure,
            name=f"get_band_structure_{label}",
            H=wg.tasks[f"apply_onsite_term_{label}"].outputs["result"],
            band_params=band_params,
            use_mpi=task_metadata.get("options", {}).get("withmpi", False),
            computer=task_metadata.get("computer", "localhost"),
            metadata=task_metadata.get("metadata", {}),
        )

    return wg


@task.graph_builder()
def compute_continuum_bands(
    *,
    nearest_neighbor_hopping: float = 2.7,
    bond_length: float = 1.42,
    interlayer_hopping: float = 0.4,
    superlattice_potential_periodicity: int = 10,
    superlattice_potential_amplitude: float = 0.0,
    gate_bias: float = 0.0,
    layer_potential_ratio: float = 1.0,
    nearest_neighbor_order: int = 1,
    high_sym_points: dict,
    path: str,
    total_points: int = 100,
    use_mpi: bool = False,
    metadata: dict | None = None,
) -> WorkGraph:
    wg = WorkGraph("ContinuumBands")
    task_metadata: dict = metadata.get("get_continuum_model", {})
    wg.add_task(
        calculations.get_continuum_model,
        name="get_continuum_model",
        nearest_neighbor_hopping=nearest_neighbor_hopping,
        bond_length=bond_length,
        interlayer_hopping=interlayer_hopping,
        superlattice_potential_periodicity=superlattice_potential_periodicity,
        superlattice_potential_amplitude=superlattice_potential_amplitude,
        gate_bias=gate_bias,
        layer_potential_ratio=layer_potential_ratio,
        nearest_neighbor_order=nearest_neighbor_order,
        metadata=task_metadata.get("metadata", {}),
    )
    task_metadata: dict = metadata.get("get_continuum_band_structure", {})
    wg.add_task(
        calculations.get_continuum_band_structure,
        name="get_continuum_band_structure",
        model=wg.tasks["get_continuum_model"].outputs["result"],
        high_sym_points=high_sym_points,
        path=path,
        total_points=total_points,
        use_mpi=use_mpi,
        metadata=task_metadata.get("metadata", {}),
    )
    return wg
