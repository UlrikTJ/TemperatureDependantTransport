# Function for drawing atoms with atom indices and color depending MD constraint
# General
import numpy as np
import matplotlib.pyplot as plt

# For handling structures and visualizing structures
from ase import Atoms
from ase.build import graphene_nanoribbon
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.io import read, write

# For MD
from ase.calculators.tersoff import Tersoff
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase import units
from ase.neighborlist import neighbor_list

def draw_atoms(atoms, radius=200, drawfixed=True):
    fig, ax = plt.subplots()

    # For drawing constraints
    if len(atoms.constraints) > 0:
        constraints = atoms.constraints[0].index
    else:
        constraints = []

    for ai, atom in enumerate(atoms):
        color = "k"
        if drawfixed and ai in constraints:
            color = "tab:red"
        ax.scatter(atom.position[2], atom.position[0], color=color, alpha=0.5, s=radius)
        ax.annotate(ai, (atom.position[2], atom.position[0]), ha="center", va="center", color="white")
    ax.set_xlim([0, atoms.cell[2,2]])
    ax.set_ylim([0, atoms.cell[0,0]])
    ax.set_xlabel("z, Å")
    ax.set_ylabel("x, Å")
    ax.axis("equal")
    return ax

# Function for loading and plotting temperature along MD trajectory
def plot_mdlog(fname="md.log"):
    log = np.loadtxt(fname, skiprows=1)
    fig, ax = plt.subplots()
    ax.plot(log[:,0], log[:, 4], color="k", alpha=0.5)
    ax.set_xlabel("Time, ps")
    ax.set_ylabel("Temperature, K")
    return ax

# Function for sorting if needed
def sort_atoms(atoms):
    pos = atoms.get_positions()
    sorted_indices = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))
    return atoms[sorted_indices]

def self_energy_decimation(energy, h0, V, iterations=100, tol=1e-9, eta=0.0001, print_check=False):
    """
    Calculates surface Green's function and Self Energy using the 
    Sancho-Rubio Decimation method (matches the image logic).
    
    Converges exponentially fast (doubles chain length each step).
    """
    size = h0.shape[0]
    I = np.eye(size)
    
    # t corresponds to hopping V in your notation
    t = V 
    t_dag = np.conjugate(np.transpose(V))
    
    # eps is the surface energy term we are updating
    eps_s = h0.copy()
    # eps is the bulk energy term
    eps = h0.copy()
    
    for _ in range(iterations):
        # Green's function for the bulk element being removed
        g = np.linalg.inv(energy * I - eps)
        
        # Update surface energy (connecting to the rest of the chain)
        eps_s = eps_s + t @ g @ t_dag
        
        # Update bulk energy and hopping parameters for the next iteration
        # This effectively calculates h^(1), V^(1) shown in your image
        eps_new = eps + t @ g @ t_dag + t_dag @ g @ t
        t_new = t @ g @ t 
        t_dag_new = t_dag @ g @ t_dag
        
        
        # Check for convergence (if hopping goes to zero, we are disconnected)
        if np.max(np.abs(t_new)) < tol:
            if print_check==True:
                print(f"Converged after {_+1} iterations.")
                break
            elif print_check==False:
                break

        eps = eps_new
        t = t_new
        t_dag = t_dag_new
            

        

    # The surface Green's function is the inverse of the effective surface Hamiltonian
    Gs = np.linalg.inv((energy + 1j * eta) * I - eps_s)
    Gb = np.linalg.inv((energy + 1j * eta) * I - eps)
    
    # Calculate Sigma from Gs: Sigma = zI - h0 - Gs^-1
    # Alternatively, specifically for the surface term: Sigma = t_initial @ G_bulk_surface @ t_dag_initial
    # But often directly calculating Sigma via the relation to Gs is easier:
    Sigma_s = eps_s - h0
    Sigma_b = eps - h0
    return Sigma_s, Sigma_b, Gs, Gb

#V^dagger
def V_dagger(V):
    return np.conjugate(np.transpose(V))

#Plot the real and imaginary parts of the green's function as a function of energy
def greens_function_vs_energy(H_onsite, V, energy_range, eta=0.01, print_check=False):
    real_parts = []
    imag_parts = []
    real_bulk_parts = []
    imag_bulk_parts = []
    
    size = H_onsite.shape[0]
    I = np.eye(size)

    for E in energy_range:
        z = E + 1j * eta  # Small imaginary part for causality/broadening
        
        # Use the self-energy from decimation to get the semi-infinite result
        Sigma, no,  Gs, Gb = self_energy_decimation(z, H_onsite, V,iterations=100, print_check=print_check)
        
        
        real_parts.append(np.real(Gs[0,0]))
        imag_parts.append(np.imag(Gs[0,0]))
        real_bulk_parts.append(np.real(Gb[0,0]))
        imag_bulk_parts.append(np.imag(Gb[0,0]))
    
    return real_parts, imag_parts , real_bulk_parts, imag_bulk_parts


#Plot LDOS
def LDOS(H_onsite, V, energy_range, eta=0.01):
    ldos_values = []
    ldos_bulk_values = []
    size = H_onsite.shape[0]
    I = np.eye(size)

    for E in energy_range:
        z = E + 1j * eta
        # Calculate surface Green's function for semi-infinite chain using decimation
        Sigma_s, Sigma_b, Gs, Gb = self_energy_decimation(z, H_onsite, V, iterations=100)
        
        G_trace = np.trace(Gs, axis1=0, axis2=1)
        ldos = -1/np.pi * np.imag(G_trace)
        ldos_values.append(ldos)

        Gb_trace = np.trace(Gb, axis1=0, axis2=1)
        ldos_bulk = -1/np.pi * np.imag(Gb_trace)
        ldos_bulk_values.append(ldos_bulk)
    return ldos_values, ldos_bulk_values


#Make a function
def transmission_vs_energy(H_device, hamil_R, V_R, hamil_L, V_L, energy_range, eta=0.0001):
    transmission_values = []
    size = H_device.shape[0]
    I = np.eye(size)

    for E in energy_range:
        z = E + 1j * eta
        
        # Get Surface Green's Functions for semi-infinite leads
        # Right Lead: Extends to +x. We need surface GF at left end. Decimation uses hopping *into* bulk (+x)
        # lead_hopping_into_right should be hopping from cell n to n+1
        sig_s_R, _, g_surface_R, _ = self_energy_decimation(z, hamil_R, V_R, iterations=100)
        
        # Left Lead: Extends to -x. We need surface GF at right end. Decimation uses hopping *into* bulk (-x)
        # lead_hopping_into_left should be hopping from cell n to n-1
        sig_s_L, _, g_surface_L, _ = self_energy_decimation(z, hamil_L, V_L, iterations=100)
        
        
        device_zero = np.zeros((size, size), dtype = complex)
        #Place the self energies in the top left and bottom right corners
        gray_left = device_zero.copy()
        small_size = g_surface_L.shape[0]
        gray_left[:small_size, :small_size] = sig_s_L
        gray_right = device_zero.copy()
        small_size = g_surface_R.shape[0]
        gray_right[-small_size:, -small_size:] = sig_s_R

        #Add the self energies to the device hamiltonian
        H_D = H_device + gray_left + gray_right
        
        #Gamma matricies: gamma_L = i * (Sigma_L - Sigma_L_dag)
        big_size = H_D.shape[0]
        gamma_L = 1j * (gray_left - V_dagger(gray_left))
        gamma_R = 1j * (gray_right - V_dagger(gray_right))
              
        #Greens function G_D
        G_D = np.linalg.inv(z*I - H_D)

        #Transmission T = Trace(Gamma_L @ G_D @ Gamma_R @ G_D_dag)
        T_value = np.real(np.trace(gamma_L @ G_D @ gamma_R @ V_dagger(G_D)))
        transmission_values.append(np.real(T_value))
    
    return transmission_values

def hamiltonian(xyz):
    bond = 1.43877067
    Vpppi = -2.7
    cut = bond + 0.3
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.where((dist < cut) & (dist > 0.1), Vpppi * (bond / dist)**2, 0.0)
    return H

# Calculate band structure
def calculate_bands(structure, n_kpoints=300):
    """
    Calculate band structure for a periodic system along z-direction
    """
    # Get lattice constant in transport direction (z)
    lattice_z = structure.cell[2, 2]
    
    # Create k-points along the Brillouin zone
    k_points = np.linspace(-np.pi/lattice_z, np.pi/lattice_z, n_kpoints)
    
    # Get positions
    pos = structure.positions
    n_atoms = len(pos)
    
    # Neighbor parameters
    a = 1.42  # C-C bond length
    bond_min = 0.1
    bond_max = a + 0.1
    
    # Storage for bands
    bands = []
    
    for k in k_points:
        # Build H(k) using Bloch's theorem
        H_k = np.zeros((n_atoms, n_atoms), dtype=complex)
        
        for i in range(n_atoms):
            for j in range(n_atoms):
                # Check neighbors in 0, +1, and -1 unit cells
                for m in [-1, 0, 1]:
                    # Vector from atom i to atom j in image m
                    # R_vector is the lattice vector shift: [0, 0, m * lattice_z]
                    shift = np.array([0, 0, m * lattice_z])
                    diff_vec = pos[j] + shift - pos[i]
                    dist = np.linalg.norm(diff_vec)
                    
                    if bond_min < dist < bond_max:
                        # Found a neighbor!
                        # The hopping term is t = -1
                        # The Bloch phase factor depends on the lattice vector R = m * lattice_z
                        # H(k) = sum_R H(R) * exp(i * k * R)
                        
                        phase = np.exp(1j * k * (m * lattice_z))
                        H_k[i, j] += -1.0 * phase

        # Diagonalize to get eigenvalues (energies)
        eigenvalues = np.linalg.eigvalsh(H_k)
        bands.append(eigenvalues)
    
    return k_points, np.array(bands)


#Pre calculate surface energys for leads
def precompute_self_energies(energy_range, hamil_lead, V_lead, eta=0.0001):
    sig_s_list = []
    g_surface_list = []
    size = hamil_lead.shape[0]
    I = np.eye(size)

    for E in energy_range:
        z = E + 1j * eta
        
        # Get Surface Green's Functions for semi-infinite leads
        sig_s, _, g_surface, _ = self_energy_decimation(z, hamil_lead, V_lead, iterations=100)
        sig_s_list.append(sig_s)
        g_surface_list.append(g_surface)
    
    return np.array(sig_s_list), np.array(g_surface_list)


def calculate_transmission_one_energy(E, i, eta, H_device, sig_s_L_list, sig_s_R_list, V_dagger_func):
    size = H_device.shape[0]
    I = np.eye(size)
    z = E + 1j * eta    
    
    # Retrieve the self-energies for the current energy point
    sig_s_L = sig_s_L_list[i]
    sig_s_R = sig_s_R_list[i]

    device_zero = np.zeros((size, size), dtype = complex)
    #Place the self energies in the top left and bottom right corners
    gray_left = device_zero.copy()
    small_size = sig_s_L.shape[0]
    gray_left[:small_size, :small_size] = sig_s_L
    gray_right = device_zero.copy()
    small_size = sig_s_R.shape[0]
    gray_right[-small_size:, -small_size:] = sig_s_R

    #Add the self energies to the device hamiltonian
    H_D = H_device + gray_left + gray_right
    
    #Gamma matricies: gamma_L = i * (Sigma_L - Sigma_L_dag)
    gamma_L = 1j * (gray_left - V_dagger_func(gray_left))
    gamma_R = 1j * (gray_right - V_dagger_func(gray_right))
            
    #Greens function G_D
    G_D = np.linalg.inv(z*I - H_D)

    #Transmission T = Trace(Gamma_L @ G_D @ Gamma_R @ G_D_dag)
    T_value = np.real(np.trace(gamma_L @ G_D @ gamma_R @ V_dagger_func(G_D)))
    return np.real(T_value)

def transmission_vs_energy_one(H_device, energy_range, sig_s_R_list, sig_s_L_list, g_surface_R_list, g_surface_L_list, eta=0.0001):
    # This sequential version is kept for reference or if no parallel wrapper is used
    transmission_values = []
    size = H_device.shape[0]
    I = np.eye(size)

    for i, E in enumerate(energy_range):
        T_val = calculate_transmission_one_energy(E, i, eta, H_device, sig_s_L_list, sig_s_R_list, V_dagger)
        transmission_values.append(T_val)
    
    return transmission_values

def process_frame(frame, energy_range, sig_L_list, sig_R_list, eta):
    molecule = frame
    
    # Sort numbering along x direction
    pos = molecule.positions
    x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
    # Manually swap atoms 9 and 10 as requested in previous cells
    x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
    
    # Re-order positions
    pos = pos[x_sort_args]
        
    n_lead_atoms = 20
    idx_L = np.arange(n_lead_atoms) # Since we sorted pos, indices are now just 0..N-1
    idx_R = np.arange(len(pos)-n_lead_atoms, len(pos))
    idx_D = np.arange(n_lead_atoms, len(pos)-n_lead_atoms)
    
    # Important: H_full must be calculated with the sorted positions
    H_full = hamiltonian(pos) 
    
    # Extract blocks
    # H_L = H_full[np.ix_(idx_L, idx_L)]      
    H_D = H_full[np.ix_(idx_D, idx_D)]   
    # H_R = H_full[np.ix_(idx_R, idx_R)]      
    # t_L = H_full[np.ix_(idx_D[:n_lead_atoms], idx_L)]      
    # t_R = H_full[np.ix_(idx_D[-n_lead_atoms:], idx_R)]
    
    # For the transmission calculation we need to loop over energies
    # We can use the sequential function inside the parallel loop over frames
    # Or parallelize the energy loop for each frame. 
    # Usually parallelizing the outer loop (frames) is more efficient if many frames.
    
    trans_val = transmission_vs_energy_one(H_D, energy_range, sig_R_list, sig_L_list, None, None, eta=eta)
    return trans_val

# Define a new function that handles the defect
def process_frame_defected(frame, energy_range, sig_L_list, sig_R_list, eta, defect_atom_indices=[47], defect_energies=[-2.0]):
    molecule = frame
    
    # Sort numbering along x direction
    pos = molecule.positions
    x_sort_args = np.lexsort((pos[:, 0], pos[:, 2]))
    # Manually swap atoms 9 and 10 as requested
    x_sort_args[[9, 10]] = x_sort_args[[10, 9]]
    
    # Sort positions for Hamiltonian calculation
    pos_sorted = pos[x_sort_args]
        
    n_lead_atoms = 20
    # Create indices for the sorted array
    idx_D = np.arange(n_lead_atoms, len(pos_sorted)-n_lead_atoms)
    
    # Calculate Hamiltonian for the sorted positions
    H_full = hamiltonian(pos_sorted) 
    
    # Apply the defect 
    # We need to find where 'defect_atom_indices' ended up in the sorted array
    # sorted_idx is the index in H_full corresponding to original atom 'defect_atom_index'
    
    # Vectorized lookup for indices
    # We want to find i such that x_sort_args[i] is in defect_atom_indices
    # sorter = np.argsort(x_sort_args)
    # sorted_indices = sorter[np.searchsorted(x_sort_args, defect_atom_indices, sorter=sorter)]
    
    # A cleaner way given x_sort_args is a permutation: 
    # If we want to find where original index `k` went, we can invert the permutation.
    inverse_permutation = np.empty_like(x_sort_args)
    inverse_permutation[x_sort_args] = np.arange(len(x_sort_args))
    
    sorted_defect_indices = inverse_permutation[defect_atom_indices]
    
    # Add the on-site energies using advanced indexing (no for loops)
    H_full[sorted_defect_indices, sorted_defect_indices] = defect_energies
    
    # Extract the device part
    H_D = H_full[np.ix_(idx_D, idx_D)]   
    
    # Calculate transmission
    trans_val = transmission_vs_energy_one(H_D, energy_range, sig_R_list, sig_L_list, None, None, eta=eta)
    return trans_val