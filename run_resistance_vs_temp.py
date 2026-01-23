
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

def hamiltonian(xyz):
    bond = 1.43877067
    Vpppi = -2.7
    cut = bond + 0.8
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where((dist < cut) & (dist > 0.1), Vpppi * (bond / dist)**2, 0.0)
    return H

def resistance(Transmission):
    T_safe = np.array(Transmission)
    T_safe[T_safe < 1e-12] = 1e-12
    return 1.0 / T_safe

# --- Structure Definition ---
kind = "armchair"     
n = 5                 
length = 1            
vacuum = 15.0         
bond = 1.43877067     

ribbon = graphene_nanoribbon(n=n,
                             m=length,
                             type=kind,
                             C_C=bond,
                             vacuum=vacuum)
ribbon.pbc = True
ribbon = sort_atoms(ribbon) 
structure = ribbon

# --- Hamiltonian Setup (Pristine) ---
Ntransport = 5 
pristine_structure = structure.repeat((1, 1, Ntransport))
pos = pristine_structure.positions

x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
if len(x_sort_args) > 10:
    x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
pos_sorted = pos[x_sort_args]

n_lead_atoms = 20 

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
ntile_fixed = 10  # FIXED tile length
md_nsteps = 1000
md_dump = 1
eta = 0.0001
E_fermi_val = 0.0 # FIXED Fermi energy
temps_list = np.arange(0, 301, 1) # Sweep 0 to 300 K

calc = Tersoff.from_lammps("C.tersoff")
timestep = 1.0

# Pre-compute Self-Energies (Only for E=0)
print(f"Pre-computing self-energies for E={E_fermi_val}...")
z_fermi = E_fermi_val + 1j * eta
sL, _, _, _ = self_energy_decimation(z_fermi, H_L, t_L, iterations=100)
sR, _, _, _ = self_energy_decimation(z_fermi, H_R, t_R, iterations=100)
self_energy_tuple = (sL, sR)

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

def run_simulation_temp(current_temp):
    # 1. Create Structure (Length = 10 tiles)
    # Reconstruct the structure for 10 tiles just like in loop
    xyz = structure.positions
    lattice = structure.cell[:]
    tiledir = 2 
    for n in range(1, ntile_fixed):
        xyz = np.concatenate((xyz, structure.positions + lattice[tiledir, :]*n))
    
    tilemat = np.eye(3, dtype=int)
    tilemat[tiledir, tiledir] = ntile_fixed
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
    # NOTE: Since L is fixed, we vary T
    temp_xyz_file = f"MD_files/md_T{int(current_temp)}_L{ntile_fixed}_N{md_nsteps}.xyz"
    
    # Check if MD already done
    if not (os.path.exists(temp_xyz_file) and os.path.getsize(temp_xyz_file) > 100):
        MaxwellBoltzmannDistribution(current_md_structure, temperature_K=current_temp)
        dyn = Langevin(current_md_structure, timestep*units.fs, temperature_K=current_temp, friction=0.01/units.fs, logfile=None)
        dyn.attach(lambda: write(temp_xyz_file, current_md_structure, append=True), interval=md_dump)
        dyn.run(md_nsteps)
    
    # 3. Analyze
    try:
        traj = read(temp_xyz_file, index=":")
        if len(traj) == 0: return (current_temp, np.nan, np.nan)
    except Exception:
        return (current_temp, np.nan, np.nan)
        
    start_analysis_frame = int(len(traj)*0.25)
    traj_analysis = traj[start_analysis_frame:]
    if len(traj_analysis) == 0: return (current_temp, np.nan, np.nan)

    trans_samples = []
    
    for frame in traj_analysis:
        pos = frame.positions
        x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
        if len(x_sort_args) > 10: x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
        pos_sorted = pos[x_sort_args]
        
        H_full_frame = hamiltonian(pos_sorted)
        # Use same lead size
        idx_D = np.arange(n_lead_atoms, len(pos_sorted)-n_lead_atoms)
        H_D_frame = H_full_frame[np.ix_(idx_D, idx_D)]
        
        sig_L, sig_R = self_energy_tuple
        T = calculate_T_at_energy(H_D_frame, sig_L, sig_R, E_fermi_val)
        trans_samples.append(T)

    R_quantum = 12906.0
    avg_transmission = np.mean(trans_samples)
    if avg_transmission < 1e-12: res_from_avg_T = np.nan
    else: res_from_avg_T = (1.0 / avg_transmission) * R_quantum
    
    R_samples = resistance(trans_samples) * R_quantum
    res_avg_instant = np.mean(R_samples)
    
    return (current_temp, res_from_avg_T, res_avg_instant)

# --- Main Execution ---
if __name__ == "__main__":
    output_dir = "Defected_tranmission_data/Resistance_Scaling"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "Resistance_vs_Temp_FixedL10_E0.csv")
    
    print(f"Starting Temperature Sweep for Temps 0..300 with L={ntile_fixed}, E={E_fermi_val}")
    
    # Load existing to skip
    existing_temps = set()
    if os.path.exists(output_file):
        try:
            data = np.loadtxt(output_file, delimiter=",", skiprows=1) # skip header
            if data.ndim == 1 and data.size > 0:
                existing_temps.add(int(data[0]))
            elif data.ndim > 1:
                for row in data:
                    existing_temps.add(int(row[0]))
            print(f"Loaded {len(existing_temps)} existing data points.")
        except Exception:
            pass
            
    temps_to_run = [t for t in temps_list if t not in existing_temps]
    
    if temps_to_run:
        print(f"Running for {len(temps_to_run)} temperatures...")
        results_list = Parallel(n_jobs=-1)(
            delayed(run_simulation_temp)(t) for t in tqdm(temps_to_run)
        )
        
        # Merge with existing file (append)
        mode = 'a' if os.path.exists(output_file) else 'w'
        with open(output_file, mode) as f:
            if mode == 'w':
                 f.write(f"Temperature_K,R_Method1,R_Method2\n")
            
            for (t, r1, r2) in results_list:
                f.write(f"{t},{r1},{r2}\n")
                print(f"Completed T={t} K, R1={r1:.2e}, R2={r2:.2e}")
    else:
        print("All temperatures already calculated.")
