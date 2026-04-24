#%%
# Comprehensive RDKit-based analysis of transition states
# With visualization and modular feature detection

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen
import numpy as np
from scipy.spatial.distance import cdist
import pickle
import py3Dmol

#========== LOAD DATA ==========
with open('/Users/sdruber/Documents/GitHub/research/Hackathon/DATA/Halo8/train.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data['reactant']['num_atoms'])} transition states")

#========== MOLECULAR CONSTRUCTION ==========
# Typical single bond lengths (Angstroms)
BOND_LENGTHS = {
    (1, 1): 0.74,   (1, 6): 1.09,  (1, 7): 1.01,  (1, 8): 0.96,  (1, 9): 0.92,   (1, 16): 1.34,  (1, 17): 1.27,
    (6, 6): 1.54,   (6, 7): 1.47,  (6, 8): 1.43,  (6, 9): 1.35,  (6, 16): 1.82,  (6, 17): 1.77,
    (7, 7): 1.45,   (7, 8): 1.40,  (7, 9): 1.37,  (7, 16): 1.60,
    (8, 8): 1.48,   (8, 9): 1.41,  (8, 16): 1.60,
    (9, 9): 1.40,   (16, 16): 2.05
}

def get_bond_cutoff(z1, z2):
    """Get bond distance cutoff for two atomic numbers"""
    key = tuple(sorted([z1, z2]))
    # Default to 1.6 if not in table, use 20% tolerance
    return BOND_LENGTHS.get(key, 1.6) * 1.2

def create_mol_from_data(positions, charges):
    """Create RDKit molecule from atomic positions and charges"""
    mol = Chem.RWMol()
    
    # Add atoms
    for charge in charges:
        atom = Chem.Atom(int(charge))
        mol.AddAtom(atom)
    
    # Add bonds based on realistic distances
    distances = cdist(positions, positions)
    n_atoms = len(positions)
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            z_i, z_j = int(charges[i]), int(charges[j])
            cutoff = get_bond_cutoff(z_i, z_j)
            
            if distances[i, j] < cutoff:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
    
    mol = mol.GetMol()
    
    # Calculate implicit valence and sanitize
    try:
        Chem.SanitizeMol(mol)
    except:
        # If sanitization fails, just calculate valence
        for atom in mol.GetAtoms():
            Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
    
    # Assign 3D coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    
    return mol

#========== VISUALIZATION ==========
def xyz_string_from_mol(mol):
    """Convert RDKit mol to XYZ format for py3Dmol"""
    lines = []
    conf = mol.GetConformer()
    lines.append(str(mol.GetNumAtoms()))
    lines.append("Transition State")
    
    for atom, pos in zip(mol.GetAtoms(), conf.GetPositions()):
        symbol = atom.GetSymbol()
        lines.append(f"{symbol}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}")
    
    return "\n".join(lines)

def visualize_molecule(mol, width=600, height=400):
    """Display molecule using py3Dmol - stick and sphere representation"""
    xyz_str = xyz_string_from_mol(mol)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_str, "xyz")
    view.setStyle({}, {"stick": {}, "sphere": {"scale": 0.3}})
    view.zoomTo()
    return view.show()


def analyze_and_visualize(mol_index, include_features=True):
    """Comprehensive analysis and visualization of a molecule"""
    print(f"\n{'='*60}")
    print(f"ANALYZING AND VISUALIZING MOLECULE {mol_index}")
    print(f"{'='*60}")
    
    positions = data['reactant']['positions'][mol_index]
    charges = data['reactant']['charges'][mol_index]
    
    # Create molecule
    mol = create_mol_from_data(positions, charges)
    
    # Display basic info
    print(f"\nMolecular Properties:")
    print(f"  Total Atoms: {mol.GetNumAtoms()}")
    print(f"  Total Bonds: {mol.GetNumBonds()}")
    print(f"  Molecular Weight: {Chem.Descriptors.MolWt(mol):.2f} g/mol")
    print(f"  LogP (Lipophilicity): {Chem.Descriptors.MolLogP(mol):.2f}")
    
    # Visualize
    print(f"\n3D Visualization (Stick & Sphere style):")
    visualize_molecule(mol, width=700, height=600)
    
    return mol

#========== INTERACTIVE VISUALIZATION DEMO ==========
def demo_visualizations():
    """Run through visualization examples"""
    # Get first few molecules
    for i in range(min(3, len(data['reactant']['num_atoms']))):
        mol = analyze_and_visualize(i, include_features=True)
        print("\n" + "-"*60)

# %%
#========== EXAMPLE: VISUALIZE SPECIFIC MOLECULES ==========
print("\n=== VISUALIZING MOLECULES WITH PY3DMOL ===\n")

# Visualize molecule 0
print("Molecule 0:")
mol_0 = analyze_and_visualize(0)

# %%
# Visualize molecule 1
print("\nMolecule 1:")
mol_1_positions = data['reactant']['positions'][1]
mol_1_charges = data['reactant']['charges'][1]
mol_1 = create_mol_from_data(mol_1_positions, mol_1_charges)
visualize_molecule(mol_1, width=600, height=500)

# %%
# Visualize molecule 2
print("\nMolecule 2:")
mol_2_positions = data['reactant']['positions'][2]
mol_2_charges = data['reactant']['charges'][2]
mol_2 = create_mol_from_data(mol_2_positions, mol_2_charges)
visualize_molecule(mol_2, width=600, height=500)

# %%
print("\n" + "="*60)
print("VISUALIZATION COMPLETE!")
print("="*60)
