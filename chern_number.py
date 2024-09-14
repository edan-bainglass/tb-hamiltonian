#!/usr/bin/env python3

import sys
import numpy as np

from tb_hamiltonian.continuum import GrapheneContinuumModel, compute_eigenstuff

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def compute_berry_curvature(kpoints, dkx, dky, bnd_idx, H_calculator):
    _, psi = compute_eigenstuff(H_calculator, kpoints)
    _, psi_dkx = compute_eigenstuff(H_calculator, [[k[0] + dkx, k[1]] for k in kpoints])
    _, psi_dky = compute_eigenstuff(H_calculator, [[k[0], k[1] + dky] for k in kpoints])

    psi = psi[:, :, bnd_idx]
    dpsi_dkx = (psi - psi_dkx[:, :, bnd_idx]) / dkx
    dpsi_dky = (psi - psi_dky[:, :, bnd_idx]) / dky

    berry_flux = np.imag(
        np.einsum("ij,ij->i", np.conj(dpsi_dkx), dpsi_dky)
        - np.einsum("ij,ij->i", np.conj(dpsi_dky), dpsi_dkx)
    )
    return np.sum(berry_flux)


def compute_berry_curvature_log(kpoints, dkx, dky, bnd_idx, H_calculator):
    _, psi = compute_eigenstuff(H_calculator, kpoints)
    _, psi_right = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1]] for k in kpoints]
    )
    _, psi_up = compute_eigenstuff(H_calculator, [[k[0], k[1] + dky] for k in kpoints])
    _, psi_diag = compute_eigenstuff(
        H_calculator, [[k[0] + dkx, k[1] + dky] for k in kpoints]
    )

    psi = psi[:, :, bnd_idx]
    psi_right = psi_right[:, :, bnd_idx]
    psi_up = psi_up[:, :, bnd_idx]
    psi_diag = psi_diag[:, :, bnd_idx]

    Ux = np.einsum("ij,ij->i", np.conj(psi), psi_right)
    Uy = np.einsum("ij,ij->i", np.conj(psi), psi_up)
    Ux_dagger = np.einsum("ij,ij->i", np.conj(psi_up), psi_diag)
    Uy_dagger = np.einsum("ij,ij->i", np.conj(psi_right), psi_diag)

    berry_flux_log = np.imag(np.log(Ux * Uy * np.conj(Ux_dagger) * np.conj(Uy_dagger)))

    return np.sum(berry_flux_log)


def compute_chern_number(kpoints, dkx, dky, bnd_idx, H_calculator):
    # for k in kpoints:
    #     kx, ky = k
    #     berry_flux += compute_berry_curvature(kx, ky, dkx, dky, bnd_idx, H_calculator)
    #     # berry_flux_log += compute_berry_curvature_log(
    #     #     kx, ky, dkx, dky, bnd_idx, H_calculator
    #     # )
    # Normalization by BZ area
    berry_flux = compute_berry_curvature(kpoints, dkx, dky, bnd_idx, H_calculator)
    berry_flux_log = compute_berry_curvature_log(
        kpoints, dkx, dky, bnd_idx, H_calculator
    )

    chern_number = berry_flux * dkx * dky / (2 * np.pi)
    chern_number_log = berry_flux_log * dkx * dky / (2 * np.pi)
    return chern_number, chern_number_log


if __name__ == "__main__":
    inputs = dict(
        bond_length=1.425,
        interlayer_hopping=0.22,
        superlattice_potential_periodicity=500,
        superlattice_potential_amplitude=0.020,
        gate_bias=0.024,
        layer_potential_ratio=0.3,
        nearest_neighbor_order=1,
    )

    model = GrapheneContinuumModel(**inputs)

    # Constants for Brillouin zone grid
    nkpts = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    kx_vals = np.linspace(-np.pi, np.pi, nkpts)
    ky_vals = np.linspace(-np.pi, np.pi, nkpts)

    dkx = kx_vals[1] - kx_vals[0]
    dky = ky_vals[1] - ky_vals[0]

    H_calculator = model.H_total_K

    bnd_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    kpoints = [[kx, ky] for kx in kx_vals for ky in ky_vals]

    # Split the kx_vals array across ranks
    kpoints_split = np.array_split(kpoints, size)

    # Each rank gets its own subset of kx_vals and ky_vals
    kpoints_local = kpoints_split[rank]

    # Compute the Chern number for this subset of kx and ky values
    chern_local, chern_log_local = compute_chern_number(
        kpoints_local, dkx, dky, bnd_idx, H_calculator
    )

    # Sum the local results to rank 0
    chern_total = comm.reduce(chern_local, op=MPI.SUM, root=0)
    chern_log_total = comm.reduce(chern_log_local, op=MPI.SUM, root=0)

    if rank == 0:
        print(f"Chern number: {chern_total}")
        print(f"Chern number log: {chern_log_total}")
