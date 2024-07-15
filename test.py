import sys
from pathlib import Path

import numpy as np
from tb_hamiltonian import TBHamiltonian
from tb_hamiltonian.potentials import PotentialFactory
from tb_hamiltonian.utils import get_structure

sys.tracebacklimit = None

nn = 1  # number of nearest neighbors | don't use 0!

workdir = Path("examples/BLG/AB/rectangular")

# lengths
lx = 50  # length in x direction (Å)
ly = lx / np.sqrt(3)  # length in y direction (Å) keeping the b/a ratio
lz = 10  # length in z direction (Å)
basepath = workdir / f"len_{lx}x{int(ly)}/nn_{nn}"

# or, repetitions
nx = 1  # number of repetitions in x direction
ny = 1  # number of repetitions in y direction
nz = 1  # number of repetitions in z direction
basepath = workdir / f"rep_{nx}x{ny}/nn_{nn}"

basepath.mkdir(parents=True, exist_ok=True)

# Define structure

structure = get_structure(
    unit_cell_filepath=workdir / "POSCAR",  # local unit cell file
    # lengths=(lx, ly, lz),
    repetitions=(nx, ny, nz),
)

structure.info["label"] = "BLG"  # will show up at top of Hamiltonian output file

structure.write(basepath / "POSCAR", format="vasp")

# Compute H

H = TBHamiltonian(
    structure=structure,
    nearest_neighbor=nn,
    distances=(0.0, 1.425, 2.468, 2.850),
    hopping_parameters=(0.0, -2.7, 0.0, -0.27),
    interlayer_coupling=0.33,
)

H.build()

# Apply onsite term

potential = PotentialFactory("null")
potential.params = {
    "amplitude": 1.0,
    "width": 0.5,
}

H.update_onsite_terms(
    onsite_term=0.0,
    potential=potential,
    alpha=(1.0, 0.5),
)

path = (
    basepath / f"{potential.name}"
    # / f"amplitude_{potential.params['amplitude']}"
    # / f"width_{potential.params['width']}"
)
path.mkdir(parents=True, exist_ok=True)

# Write H to file

H.write_to_file(path)

# Plotting

H.plot_bands(
    high_sym_points={
        "Γ": (0.00000, 0.00000, 0.00000),
        "P": (0.00000, 0.33333, 0.00000),
        "X": (0.00000, 0.50000, 0.00000),
        "W": (0.50000, 0.50000, 0.00000),
        "Y": (0.50000, 0.00000, 0.00000),
    },
    k_path="Γ P X W Y Γ W",
    points_per_segment=20,
    use_sparse_solver=False,
    sparse_solver_params={"k": H.natoms - 2, "sigma": 1e-8},
    use_mpi=False,
    savefig_path=path / "bands.png",
)
