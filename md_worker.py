import numpy as np
import os
from ase import Atoms
from ase.build import graphene_nanoribbon
from ase.io import read, write
from ase.calculators.tersoff import Tersoff
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.neighborlist import neighbor_list

# Constants (Global)
vacuum = 15.0         
bond = 1.43877067     
tersoff_file = "C.tersoff"

def create_base_ribbon(kind="armchair", width=5):
    ribbon = graphene_nanoribbon(n=width, m=1, type=kind, C_C=bond, vacuum=vacuum)
    ribbon.pbc = True
    return ribbon

def generate_md_structure_and_run(temp_K, ntile_length, filename, nsteps=1000, dump_interval=10, kind="armchair", width=5):
    """
    Generates the structure, sets up constraints, runs MD, and saves trajectory to filename.
    """
    # 1. Create Base Structure (Unit Cell)
    base_ribbon = create_base_ribbon(kind, width)
    natom_uc = len(base_ribbon)
    
    # 2. Extend to ntile_length
    xyz = base_ribbon.positions
    cell = base_ribbon.cell[:]
    tiledir = 2 
    
    xyz_list = [xyz + cell[tiledir, :]*n for n in range(ntile_length)]
    xyz_long = np.concatenate(xyz_list)
    
    tilemat = np.eye(3, dtype=int)
    tilemat[tiledir, tiledir] = ntile_length
    cell_long = tilemat @ cell
    
    md_structure = Atoms(len(xyz_long)*["C"], positions=xyz_long, cell=cell_long, pbc=True)
    
    # 3. Setup Calculator
    if not os.path.exists(tersoff_file):
        # Fallback for checking if file exists relative to where script is run
        if not os.path.exists(tersoff_file):
             print(f"Warning: {tersoff_file} not found in {os.getcwd()}")
        
    calc = Tersoff.from_lammps(tersoff_file)
    md_structure.calc = calc
        
    # 4. Constraints
    # Fixed L and R electrodes (2 unit cells each) and edges
    fixed_uc = 2
    natoms = len(md_structure)
    
    leftinds = list(range(0, natom_uc*fixed_uc))
    rightinds = list(range(natoms - natom_uc*fixed_uc, natoms))
    
    # Edges
    cutoff = 1.5
    bulk_nneighbors = 3
    i_list, j_list = neighbor_list("ij", md_structure, cutoff)
    counts = np.bincount(i_list, minlength=len(md_structure))
    edgeinds = list(np.where(counts < bulk_nneighbors)[0])
    
    allinds_fix = np.unique(leftinds + rightinds + edgeinds)
    md_structure.set_constraint(FixAtoms(mask=allinds_fix))
    
    # 5. Run MD
    log_file = filename.replace(".xyz", ".log")
    if os.path.exists(filename): 
        print(f"file {filename} already exists") # os.remove(filename)
        return filename
    if os.path.exists(log_file): os.remove(log_file)
    
    MaxwellBoltzmannDistribution(md_structure, temperature_K=temp_K)
    dyn = Langevin(md_structure, 1.0*units.fs, temperature_K=temp_K, friction=0.01/units.fs, logfile=log_file)
    
    # Write frames
    dyn.attach(lambda: write(filename, md_structure, append=True), interval=dump_interval)
    
    # print(f"Starting MD: T={temp_K}K, L={ntile_length} -> {filename}")
    dyn.run(nsteps)
    # print(f"MD Finished for {filename}")
    
    return filename

def worker_wrapper(args):
    """
    Wrapper to unpack arguments for multiprocessing map
    args: (temp_K, ntile_length, filename, nsteps, dump_interval)
    """
    return generate_md_structure_and_run(*args)
