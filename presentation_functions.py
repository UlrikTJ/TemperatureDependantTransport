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



# Distance-dependent tight binding hamiltonian for a set of atomic coordinates in units of Vpppi
def hamdd(xyz, bond=1.43877067, Vpppi=-2.7):
    cut = bond + 0.3 # look up to this distance
    N = len(xyz)
    hamdd = np.zeros([N,N])
    dist = np.linalg.norm(xyz[None, :, :] - xyz[:, None, :], axis=2)
    for i in np.arange(N):
        for j in np.arange(N):
            if (i != j) & (dist[i,j] < cut):
                hamdd[i,j] = Vpppi*(bond/dist[i,j])**2 # distance dependence
    return hamdd

# The hopping/hamiltonian between the set of atoms
def hamhop(xyz1,xyz2):
    n1 = len(xyz1)
    n2 = len(xyz2)
    xyz = np.concatenate((xyz1,xyz2)) ## combine coordinates to total
    htmp = hamdd(xyz)
    return htmp[0:n1,n1:n1+n2]

# Fourier transform to k-space
def get_ham_k(atoms, k_scaled, hamhop, Rmax=1, hermitize=True):
    # H(k) = sum_R H(R) exp(i 2π k·R)
    xyz = atoms.get_positions()
    cell = atoms.cell.array
    pbc  = atoms.get_pbc()
    n    = len(xyz)

    # Integer translations R (only periodic directions)
    ranges = [range(-Rmax, Rmax+1) if p else [0] for p in pbc]
    R = np.array(np.meshgrid(*ranges, indexing="ij")).reshape(3, -1).T  # (nR,3)

    # Cartesian translations
    T = R @ cell  # (nR,3)
    
    # Handle R=0 (self-interaction) separately to avoid division by zero in hamhop
    HR_list = []
    for Ti in T:
        if np.linalg.norm(Ti) < 1e-5:
            HR_list.append(hamdd(xyz))
        else:
            HR_list.append(hamhop(xyz, xyz + Ti))
            
    HR = np.stack(HR_list, axis=0)  # (nR,n,n)

    # Phase factors and contraction
    phases = np.exp(1j * 2*np.pi * (R @ k_scaled))              # (nR,)
    Hk = np.tensordot(phases, HR, axes=(0, 0))                  # (n,n)
    if hermitize:
        Hk = 0.5 * (Hk + Hk.conj().T)
    return Hk


# Function for sorting if needed
def sort_atoms(atoms):
    pos = atoms.get_positions()
    sorted_indices = np.lexsort((pos[:, 0], pos[:, 1], pos[:, 2]))
    return atoms[sorted_indices]

# Function for drawing atoms with atom indices and color depending MD constraint
def draw_atoms(atoms, radius=200, drawfixed=True):
    fig, ax = plt.subplots()

    # For showing which atoms are fixed during MD (constraint)
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

# Plotting band structure along transport direction (Z)
def plot_bandstructure(structure, kpoints=300):
    k_z = np.linspace(-0.5, 0.5, kpoints)
    energies = []
    
    for k in k_z:
        # Construct k-vector for Z-direction transport (index 2)
        k_scaled = np.array([0, 0, k])
        
        # Calculate Hamiltonian in k-space
        Hk = get_ham_k(structure, k_scaled, hamhop)
        
        # Diagonalize
        eigvals = np.linalg.eigvalsh(Hk)
        energies.append(eigvals)
        
    energies = np.array(energies)
    
    # Plotting
    plt.figure(figsize=(6, 5))
    plt.plot(k_z, energies, 'k-', alpha=0.6)
    plt.title("Bandstructure")
    plt.xlabel(r"$k_z$ ($2\pi/L_z$)")
    plt.ylabel("Energy [eV]")
    plt.xlim(-0.5, 0.5)
    plt.grid(True, alpha=0.3)
    plt.show()

# Splitting Hamiltonian into lead and device blocks
def SplitHam(H, nL, nR):
    no = H.shape[0]
    nC = no - 2*nL - 2*nR
    nD = nL + nC + nR
    if nC < 1:
        print("Setup error: central region size =", nC)
        print("Use [L | L | C | R | R] setup")
        return
    
    # Boundaries
    b0 = 0
    b1 = nL
    b2 = 2*nL
    b3 = 2*nL + nC
    b4 = 2*nL + nC + nR
    b5 = no

    HL = H[b0:b1, b0:b1]    # Left lead onsite (L1)
    VL = H[b1:b2, b0:b1]    # L2 → L1 hopping

    VCL = H[b2:b3, b1:b2]   # C → L2
    VLC = VCL.T.conj()      # L2 → C

    HC = H[b2:b3, b2:b3]    # Central region
    VCR = H[b2:b3, b3:b4]   # C → R1
    VRC = VCR.T.conj()      # R1 → C

    VR = H[b3:b4, b4:b5]    # R1 → R2 hopping

    HR = H[b4:b5, b4:b5]    # Right lead onsite (R2)

    HD = H[b1:b4, b1:b4]    # Device block
    VLD = H[b1:b4, b0:b1]   # L1 → device
    VRD = H[b1:b4, b4:b5]   # device → R2

    # Return in left → right order
    return HL,VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD

# Plotting the Hamiltonian structure with block divisions
def plot_ham(nL, nR, Hbig):
    nC = Hbig.shape[0] - 2*nL - 2*nR
    b0 = 0
    b1 = nL
    b2 = 2*nL
    b3 = 2*nL + nC
    b4 = 2*nL + nC + nR
    b5 = Hbig.shape[0]

    plt.figure(figsize=(8,8))
    plt.spy(Hbig,extent=(0, Hbig.shape[0], Hbig.shape[0],0))
    # Draw block boundaries
    for b in [b1, b2, b3, b4]:
        plt.axhline(b, color='k', linewidth=1)
        plt.axvline(b, color='k', linewidth=1)

    # Add text labels
    plt.text((b0+b1)/2, (b0+b1)/2, "L1", ha='center', va='center', color='blue')

    plt.text((b1+b2)/2, (b1+b2)/2, "L2", ha='center', va='center', color='green')
    plt.text((b2+b3)/2, (b2+b3)/2, "C", ha='center', va='center', color='green')
    plt.text((b3+b4)/2, (b3+b4)/2, "R1", ha='center', va='center', color='green')
    plt.text((b4+b5)/2, (b4+b5)/2, "R2", ha='center', va='center', color='blue')

    plt.show()

# Surface Green's function 
def get_surface_greens_function(h_unit, v_unit, z, max_iter=100,tol=1e-10):
    h = np.array(h_unit, dtype=complex)
    v = np.array(v_unit, dtype=complex)
    v_dag = v.T.conj()
    dim = h.shape[0]
    I = np.eye(dim)
    
    eps_s, eps = h.copy(), h.copy()
    alpha, beta = v.copy(), v_dag.copy()
       
    for _ in range(max_iter):
        zI_eps = z * I - eps
        # Using solve for better numerical stability than direct inv
        g_alpha = np.linalg.solve(zI_eps, alpha)
        g_beta = np.linalg.solve(zI_eps, beta)
           
        alpha_next = alpha @ g_alpha
        beta_next = beta @ g_beta
        eps_next = eps + alpha @ g_beta + beta @ g_alpha
        eps_s_next = eps_s + alpha @ g_beta
           
        if np.linalg.norm(alpha_next, ord=np.inf) < tol:
            eps_s = eps_s_next
            break
        alpha, beta, eps, eps_s = alpha_next, beta_next, eps_next, eps_s_next

    g_s=np.linalg.inv(z * I - eps_s)
    g_b=np.linalg.inv(z * I - eps)
    sigma_s=eps_s-h.copy()
    sigma_b=eps-h.copy()

    return g_s,g_b,sigma_s,sigma_b

# Calculate transmission
def calculate_transmission(atoms, energy, nL, nR):
    positions = atoms.get_positions()
    Hbig = hamdd(positions)
    HL, VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD = SplitHam(Hbig, nL, nR)
    dim = HC.shape[0]
    I = np.eye(dim, dtype=complex)

    transmission = np.zeros_like(energy)
    for i, e in enumerate(energy):
        z = e + 1j*1e-8
        gl_s, gl_b, sigmal_s, sigmal_b = get_surface_greens_function(HL, VL, z)
        gr_s, gr_b, sigmar_s, sigmar_b = get_surface_greens_function(HR, VR, z)

        # left and right lead, eq 134
        sigma_L = VLC.T.conj() @ gl_s @ VLC
        sigma_R = VRC.T.conj() @ gr_s @ VRC

        # gammas, eq 136
        gamma_L = 1j * (sigma_L - sigma_L.T.conj())
        gamma_R = 1j * (sigma_R - sigma_R.T.conj())

        g_C = np.linalg.inv(z * I - HC - sigma_L - sigma_R)
        t_matrix = gamma_R @ g_C @ gamma_L @ g_C.T.conj()

        transmission[i] = np.trace(t_matrix).real
    return transmission

# Loading and plotting temperature along MD trajectory
def plot_mdlog(fname):
    log = np.loadtxt(fname, skiprows=1)
    fig, ax = plt.subplots()
    ax.plot(log[:,0], log[:, 4], color="k", alpha=0.5)
    ax.set_xlabel("Time, ps")
    ax.set_ylabel("Temperature, K")
    return ax

# Precalculate lead Green's functions for faster transmission calculations
def precalculate_leads_greens_functions(atoms, energy, nL, nR):
    positions = atoms.get_positions()
    Hbig = hamdd(positions)
    HL, VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD = SplitHam(Hbig, nL, nR)
    
    n_energy = len(energy)
    sz = HL.shape[0]
    gl_s_list = np.zeros((n_energy, sz, sz), dtype=complex)
    gr_s_list = np.zeros((n_energy, sz, sz), dtype=complex)
    
    for i, e in enumerate(energy):
        z = e + 1j*1e-8
        gl_s, gl_b, sigmal_s, sigmal_b = get_surface_greens_function(HL, VL, z)
        gr_s, gr_b, sigmar_s, sigmar_b = get_surface_greens_function(HR, VR, z)
        gl_s_list[i] = gl_s
        gr_s_list[i] = gr_s
        
    return gl_s_list, gr_s_list

def calculate_transmission_fast(atoms, energy, gl_s_list, gr_s_list, nL, nR):
    positions = atoms.get_positions()
    Hbig = hamdd(positions)
    HL, VL, HR, VR, HC, VLC, VRC, HD, VLD, VRD = SplitHam(Hbig, nL, nR)
    dim = HC.shape[0]
    I = np.eye(dim, dtype=complex)

    transmission = np.zeros_like(energy)
    for i, e in enumerate(energy):
        z = e + 1j*1e-8
        
        gl_s = gl_s_list[i]
        gr_s = gr_s_list[i]

        # left and right lead, eq 134
        sigma_L = VLC.T.conj() @ gl_s @ VLC
        sigma_R = VRC.T.conj() @ gr_s @ VRC

        # gammas, eq 136
        gamma_L = 1j * (sigma_L - sigma_L.T.conj())
        gamma_R = 1j * (sigma_R - sigma_R.T.conj())

        g_C = np.linalg.inv(z * I - HC - sigma_L - sigma_R)
        t_matrix = gamma_R @ g_C @ gamma_L @ g_C.T.conj()

        transmission[i] = np.trace(t_matrix).real
    return transmission