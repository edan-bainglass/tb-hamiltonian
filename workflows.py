from pathlib import Path

from aiida_workgraph import WorkGraph, task

import calculations


@task.graph_builder()
def compute_bands(
    structure_label: str,
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
        structure_label=structure_label,
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
    structure_label: str,
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
                structure_label,
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


@task.graph_builder()
def sweep_onsite_parameters(
    structure_label: str,
    paths: dict,
    repetitions: list,
    distances: list,
    nearest_neighbor: int,
    hopping_parameters: list,
    interlayer_coupling: float,
    sweep_params: dict,
    band_params: dict,
) -> WorkGraph:
    wg = WorkGraph("sweep_onsite_parameters")

    wg.add_task(
        calculations.define_structure,
        name="define_structure",
        paths=paths,
        repetitions=repetitions,
        structure_label=structure_label,
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

    for params in zipped(sweep_params):
        task_params = {
            "H_file": wg.tasks["build_hamiltonian"].outputs["result"],
            "workdir": paths["output_path"],
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

        wg.add_task(
            calculations.apply_onsite_term,
            name=f"apply_onsite_term_{label}",
            **task_params,
        )

        wg.add_task(
            calculations.get_band_structure,
            name=f"get_band_structure_{label}",
            H_file=wg.tasks[f"apply_onsite_term_{label}"].outputs["H_file"],
            workdir=wg.tasks[f"apply_onsite_term_{label}"].outputs["workdir"],
            band_params=band_params,
        )

    return wg


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
