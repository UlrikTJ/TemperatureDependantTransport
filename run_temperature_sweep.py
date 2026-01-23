
import numpy as np
import matplotlib.pyplot as plt
from eskild_function import *

# For handling structures and visualizing structures
from ase import Atoms
from ase.build import graphene_nanoribbon
from ase.io import read, write

# For MD
from ase.calculators.tersoff import Tersoff
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.neighborlist import neighbor_list

import os
from tqdm import tqdm
from joblib import Parallel, delayed

# --- Hamiltonian & Helper Functions ---

# Define Hamiltonian with larger cutoff to avoid artificial bond breaking
def hamiltonian(xyz):
    bond = 1.43877067
    Vpppi = -2.7
    # Cutoff adjusted to be physically robust:
    # Includes 1st neighbors (~1.42 A) and thermal stretching.
    # Excludes 2nd neighbors (~2.46 A).
    cut = bond + 0.8
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where((dist < cut) & (dist > 0.1), Vpppi * (bond / dist)**2, 0.0)
    return H

def resistance(Transmission):
    # Resistance in units of 1/G0
    T_safe = np.array(Transmission)
    T_safe[T_safe < 1e-12] = 1e-12
    return 1.0 / T_safe

# --- Structure Definition ---
kind = "armchair"     # "armchair" or "zigzag"
n = 5                 # width parameter
length = 1            # periodic repetitions along z
vacuum = 15.0         # vacuum in non-periodic directions (Å)
bond = 1.43877067     

ribbon = graphene_nanoribbon(n=n,
                             m=length,
                             type=kind,
                             C_C=bond,
                             vacuum=vacuum)
ribbon.pbc = True
ribbon = sort_atoms(ribbon) # Function from eskild_function
structure = ribbon

# --- Hamiltonian Setup (Pristine) ---
# Create a transport structure with enough length to define leads
Ntransport = 5 
pristine_structure = structure.repeat((1, 1, Ntransport))
pos = pristine_structure.positions

# Sort atoms (Standard sorting for this project)
x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
if len(x_sort_args) > 10:
    x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
pos_sorted = pos[x_sort_args]

# Parameters for splitting
n_lead_atoms = 20 

# Extract Matrices
H_full = hamiltonian(pos_sorted) 

idx_L = np.arange(n_lead_atoms)
idx_R = np.arange(len(pos_sorted)-n_lead_atoms, len(pos_sorted))
idx_D = np.arange(n_lead_atoms, len(pos_sorted)-n_lead_atoms)

H_L = H_full[np.ix_(idx_L, idx_L)]      
H_D = H_full[np.ix_(idx_D, idx_D)]      
H_R = H_full[np.ix_(idx_R, idx_R)]      
t_L = H_full[np.ix_(idx_D[:n_lead_atoms], idx_L)]      
t_R = H_full[np.ix_(idx_D[-n_lead_atoms:], idx_R)]   

# --- Simulation Parameters ---
ntiles_list = np.linspace(5, 50, 45).astype(int)
md_nsteps = 1000
md_dump = 1
eta = 0.0001
E_fermi_list = [0.0]  # Only E=0
temps_list = [0, 50, 100, 150, 200, 250, 300]

calc = Tersoff.from_lammps("C.tersoff")
timestep = 1.0

# Pre-compute Self-Energies
print("Pre-computing self-energies...")
self_energies = {}
for E in E_fermi_list:
    z_fermi = E + 1j * eta
    sL, _, _, _ = self_energy_decimation(z_fermi, H_L, t_L, iterations=100)
    sR, _, _, _ = self_energy_decimation(z_fermi, H_R, t_R, iterations=100)
    self_energies[E] = (sL, sR)

def calculate_T_at_energy(H_device, sig_L, sig_R, energy):
    size = H_device.shape[0]
    I = np.eye(size)
    z = energy + 1j * eta
    
    device_zero = np.zeros((size, size), dtype=complex)
    gray_left = device_zero.copy()
    n_lead = sig_L.shape[0]
    gray_left[:n_lead, :n_lead] = sig_L
    
    gray_right = device_zero.copy()
    n_lead_R = sig_R.shape[0]
    gray_right[-n_lead_R:, -n_lead_R:] = sig_R
    
    H_eff = H_device + gray_left + gray_right
    gamma_L = 1j * (gray_left - np.conjugate(np.transpose(gray_left)))
    gamma_R = 1j * (gray_right - np.conjugate(np.transpose(gray_right)))
    
    G_eff = np.linalg.inv(z*I - H_eff)
    T = np.trace(gamma_L @ G_eff @ gamma_R @ np.conjugate(np.transpose(G_eff)))
    return np.real(T)

def run_simulation_for_length(ntile, current_temp):
    # 1. Create Structure
    xyz = structure.positions
    lattice = structure.cell[:]
    tiledir = 2 
    for n in range(1, ntile):
        xyz = np.concatenate((xyz, structure.positions + lattice[tiledir, :]*n))
    
    tilemat = np.eye(3, dtype=int)
    tilemat[tiledir, tiledir] = ntile
    lattice_long = tilemat @ lattice
    natoms = len(xyz)
    current_md_structure = Atoms(natoms*["C"], positions=xyz, cell=lattice_long, pbc=True)
    
    # Constraints
    natoms_elec = len(structure)
    fixed_uc = 2
    leftinds = list(range(0, natoms_elec*fixed_uc))
    rightinds = list(range(natoms - natoms_elec*fixed_uc, natoms))
    cutoff = 1.5
    bulk_nneighbors = 3
    i_list, j_list = neighbor_list("ij", current_md_structure, cutoff)
    counts = np.bincount(i_list, minlength=len(current_md_structure))
    edgeinds = list(np.where(counts < bulk_nneighbors)[0])
    allinds = np.unique(leftinds + rightinds + edgeinds)
    current_md_structure.set_constraint(FixAtoms(mask=allinds))
    current_md_structure.calc = calc

    # 2. Run MD
    os.makedirs("MD_files", exist_ok=True)
    # Include Temperature in filename
    temp_xyz_file = f"MD_files/md_T{int(current_temp)}_L{ntile}_N{md_nsteps}.xyz"
    
    if os.path.exists(temp_xyz_file) and os.path.getsize(temp_xyz_file) > 100:
        pass
    else:
        MaxwellBoltzmannDistribution(current_md_structure, temperature_K=current_temp)
        dyn = Langevin(current_md_structure, timestep*units.fs, temperature_K=current_temp, friction=0.01/units.fs, logfile=None)
        dyn.attach(lambda: write(temp_xyz_file, current_md_structure, append=True), interval=md_dump)
        dyn.run(md_nsteps)
    
    # 3. Analyze
    try:
        traj = read(temp_xyz_file, index=":")
        if len(traj) == 0: return {E: (np.nan, np.nan) for E in E_fermi_list}
    except Exception:
        return {E: (np.nan, np.nan) for E in E_fermi_list}
        
    start_analysis_frame = int(len(traj)*0.25)
    traj_analysis = traj[start_analysis_frame:]
    if len(traj_analysis) == 0: return {E: (np.nan, np.nan) for E in E_fermi_list}

    trans_samples_dict = {E: [] for E in E_fermi_list}
    
    for frame in traj_analysis:
        pos = frame.positions
        x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
        if len(x_sort_args) > 10: x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
        pos_sorted = pos[x_sort_args]
        
        H_full_frame = hamiltonian(pos_sorted)
        n_lead_atoms = 20
        idx_D = np.arange(n_lead_atoms, len(pos_sorted)-n_lead_atoms)
        H_D_frame = H_full_frame[np.ix_(idx_D, idx_D)]
        
        for E in E_fermi_list:
            sig_L, sig_R = self_energies[E]
            T = calculate_T_at_energy(H_D_frame, sig_L, sig_R, E)
            trans_samples_dict[E].append(T)

    R_quantum = 12906.0
    final_results = {}
    for E in E_fermi_list:
        trans_samples = trans_samples_dict[E]
        avg_transmission = np.mean(trans_samples)
        if avg_transmission < 1e-12: res_from_avg_T = np.nan
        else: res_from_avg_T = (1.0 / avg_transmission) * R_quantum
        R_samples = resistance(trans_samples) * R_quantum
        res_avg_instant = np.mean(R_samples)
        final_results[E] = (res_from_avg_T, res_avg_instant)
    
    return final_results

# --- Main Execution ---
if __name__ == "__main__":
    output_dir = "Defected_tranmission_data/Resistance_Scaling"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting Temperature Sweep for Temps: {temps_list}")

    for current_temp_K in temps_list:
        print(f"\n--- Processing Temperature T = {current_temp_K} K ---")
        
        results_db = {E: {} for E in E_fermi_list}

        # Load existing
        for E in E_fermi_list:
            filename = f"Resistance_vs_Length_T{int(current_temp_K)}_E{E}_Nsteps{md_nsteps}_Kind{kind}_W{n}.csv"
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                try:
                    existing_data = np.loadtxt(filepath, delimiter=",", comments="#")
                    if existing_data.size > 0:
                        if existing_data.ndim == 1: existing_data = existing_data.reshape(1, -1)
                        for row in existing_data:
                            results_db[E][int(row[0])] = (row[1], row[2])
                    print(f"Loaded {len(results_db[E])} points for E={E}")
                except Exception: pass

        # Identify missing
        ntiles_to_run = set()
        for t in ntiles_list:
            for E in E_fermi_list:
                if t not in results_db[E]:
                    ntiles_to_run.add(t)
                    break
        ntiles_to_run = sorted(list(ntiles_to_run))

        if ntiles_to_run:
            print(f"Calculating for lengths: {ntiles_to_run}")
            # IMPORTANT: Pass current_temp_K to the worker function
            new_results_list = Parallel(n_jobs=-1)(
                delayed(run_simulation_for_length)(ntile, current_temp_K) for ntile in tqdm(ntiles_to_run)
            )
            # Update DB
            for t, res_dict in zip(ntiles_to_run, new_results_list):
                for E, val in res_dict.items():
                    results_db[E][t] = val
            
            # Save results immediately
            for i, E in enumerate(E_fermi_list):
                filename = f"Resistance_vs_Length_T{int(current_temp_K)}_E{E}_Nsteps{md_nsteps}_Kind{kind}_W{n}.csv"
                filepath = os.path.join(output_dir, filename)
                lengths = sorted(results_db[E].keys())
                data_save = []
                for L in lengths:
                    val = results_db[E][L]
                    data_save.append([L, val[0], val[1]])
                np.savetxt(filepath, data_save, delimiter=",", header=f"Length,R_Method1,R_Method2\nParams: T={current_temp_K}, E={E}")
                print(f"Saved T={current_temp_K} E={E} to {filepath}")
        else:
            print(f"All data already calculated for T={current_temp_K}.")
