
import numpy as np
from ase import Atoms

def test_ase_deletion():
    try:
        # Create dummy atoms
        atoms = Atoms('C'*10, positions=np.zeros((10, 3)))
        print(f"Original length: {len(atoms)}")
        
        # indices to delete: 0, 1
        indices = np.array([0, 1])
        del atoms[indices]
        print(f"Length after deleting [0,1]: {len(atoms)}") # Should be 8
        
        # positions are now indices 2..9 of original
        
        # Test negative indices deletion
        # delete last 2
        neg_indices = np.array([-1, -2])
        del atoms[neg_indices]
        print(f"Length after deleting [-1,-2]: {len(atoms)}") # Should be 6 if batch, ?? if sequential
        
    except Exception as e:
        print(f"Error: {e}")

test_ase_deletion()
