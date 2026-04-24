#%%
# Molecular Structure Classification based on RDKit similarity
# Classifies molecules by comparing structural features and fingerprints

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#========== LOAD DATA ==========
with open('/Users/sdruber/Documents/GitHub/research/Hackathon/DATA/Halo8/train.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data['reactant']['num_atoms'])} transition states")

#========== BOND LENGTHS & MOL CREATION (imported from RDKit_Analysis) ==========
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
    return BOND_LENGTHS.get(key, 1.6) * 1.2

def create_mol_from_data(positions, charges):
    """Create RDKit molecule from atomic positions and charges"""
    mol = Chem.RWMol()
    
    for charge in charges:
        atom = Chem.Atom(int(charge))
        mol.AddAtom(atom)
    
    distances = cdist(positions, positions)
    n_atoms = len(positions)
    
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            z_i, z_j = int(charges[i]), int(charges[j])
            cutoff = get_bond_cutoff(z_i, z_j)
            
            if distances[i, j] < cutoff:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)
    
    mol = mol.GetMol()
    
    try:
        Chem.SanitizeMol(mol)
    except:
        for atom in mol.GetAtoms():
            Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
    
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, pos)
    mol.AddConformer(conf)
    
    return mol

#========== SIMILARITY METRICS ==========
def compute_tanimoto_similarity(mol1, mol2, radius=2, nBits=2048):
    """Compute Tanimoto similarity between two molecules using Morgan fingerprints"""
    gen = AllChem.MorganGenerator(radius=radius, nBits=nBits)
    fp1 = np.array(gen.GetFingerprint(mol1))
    fp2 = np.array(gen.GetFingerprint(mol2))
    
    # Manual Tanimoto calculation
    intersection = np.sum(fp1 & fp2)
    union = np.sum(fp1 | fp2)
    similarity = intersection / (union + 1e-10)
    
    return similarity

def compute_descriptor_similarity(mol1, mol2):
    """Compute similarity based on molecular descriptors"""
    descriptors = [
        'MolWt', 'LogP', 'NumHBD', 'NumHBA', 'NumRotatableBonds',
        'NumAromaticRings', 'TPSA', 'NumHeavyAtoms'
    ]
    
    desc1 = np.array([getattr(Descriptors, d)(mol1) for d in descriptors])
    desc2 = np.array([getattr(Descriptors, d)(mol2) for d in descriptors])
    
    # Normalize and compute euclidean distance
    scaler = StandardScaler()
    desc1_norm = scaler.fit_transform(desc1.reshape(1, -1))[0]
    desc2_norm = scaler.fit_transform(desc2.reshape(1, -1))[0]
    
    distance = np.linalg.norm(desc1_norm - desc2_norm)
    similarity = 1 / (1 + distance)  # Convert distance to similarity
    
    return similarity

def compute_structural_features(mol):
    """Extract structural features for classification"""
    ring_info = mol.GetRingInfo()
    features = {
        'num_atoms': mol.GetNumAtoms(),
        'num_bonds': mol.GetNumBonds(),
        'num_rings': ring_info.NumRings(),
        'num_aromatic_rings': sum(1 for ring in ring_info.AtomRings() 
                                  if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring)),
        'num_double_bonds': sum(1 for bond in mol.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE),
        'num_heteroatoms': sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ['C', 'H']),
        'mol_weight': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'tpsa': Descriptors.TPSA(mol),
        'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
    }
    return features

#========== FINGERPRINT GENERATION ==========
def generate_morgan_fingerprints(molecules, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a list of molecules"""
    fingerprints = []
    for mol in molecules:
        try:
            gen = AllChem.GetMorganGenerator(radius=radius, nBits=nBits)
            fp = gen.GetFingerprint(mol)
            fingerprints.append(np.array(fp, dtype=np.float32))
        except Exception as e:
            print(f"Error generating fingerprint: {e}")
            fingerprints.append(np.zeros(nBits, dtype=np.float32))
    
    fp_array = np.array(fingerprints)
    # Convert to float for better clustering
    return fp_array.astype(np.float32)

def generate_feature_vectors(molecules):
    """Generate feature vectors from structural properties"""
    feature_list = []
    for mol in molecules:
        features = compute_structural_features(mol)
        feature_list.append(list(features.values()))
    
    return np.array(feature_list)

#========== CLASSIFICATION ALGORITHMS ==========
def classify_kmeans(fingerprints, n_clusters=5):
    """Classify molecules using K-Means clustering"""
    print(f"\n{'='*60}")
    print("K-MEANS CLUSTERING")
    print(f"{'='*60}")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(fingerprints)
    
    # Check if we actually have multiple clusters
    unique_labels = len(set(labels))
    print(f"Number of clusters requested: {n_clusters}")
    print(f"Number of clusters found: {unique_labels}")
    
    # Only calculate silhouette if we have at least 2 clusters
    if unique_labels > 1:
        try:
            silhouette = silhouette_score(fingerprints, labels)
            davies_bouldin = davies_bouldin_score(fingerprints, labels)
            print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
            print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
        except Exception as e:
            print(f"Could not calculate silhouette scores: {e}")
    else:
        print(f"WARNING: All molecules assigned to a single cluster!")
    
    for i in range(unique_labels):
        cluster_size = sum(labels == i)
        print(f"  Cluster {i}: {cluster_size} molecules")
    
    return labels, kmeans

def classify_dbscan(fingerprints, eps=0.5, min_samples=5):
    """Classify molecules using DBSCAN clustering"""
    print(f"\n{'='*60}")
    print("DBSCAN CLUSTERING")
    print(f"{'='*60}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(fingerprints)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        # Filter out noise points for silhouette calculation
        mask = labels != -1
        if sum(mask) > 0:
            silhouette = silhouette_score(fingerprints[mask], labels[mask])
            print(f"Silhouette Score: {silhouette:.4f}")
    
    for i in range(n_clusters):
        cluster_size = sum(labels == i)
        print(f"  Cluster {i}: {cluster_size} molecules")
    
    if n_noise > 0:
        print(f"  Noise points: {n_noise}")
    
    return labels, dbscan

def classify_hierarchical(fingerprints, n_clusters=5, linkage='ward'):
    """Classify molecules using Hierarchical clustering"""
    print(f"\n{'='*60}")
    print("HIERARCHICAL CLUSTERING")
    print(f"{'='*60}")
    
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hierarchical.fit_predict(fingerprints)
    
    unique_labels = len(set(labels))
    print(f"Linkage method: {linkage}")
    print(f"Number of clusters: {n_clusters}")
    
    # Only calculate silhouette if we have at least 2 clusters
    if unique_labels > 1:
        try:
            silhouette = silhouette_score(fingerprints, labels)
            davies_bouldin = davies_bouldin_score(fingerprints, labels)
            print(f"Silhouette Score: {silhouette:.4f}")
            print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")
        except Exception as e:
            print(f"Could not calculate silhouette scores: {e}")
    else:
        print(f"WARNING: All molecules assigned to a single cluster!")
    
    for i in range(unique_labels):
        cluster_size = sum(labels == i)
        print(f"  Cluster {i}: {cluster_size} molecules")
    
    return labels, hierarchical

#========== CLASSIFICATION PIPELINE ==========
def classify_molecules(n_molecules=None, n_clusters=5):
    """Full pipeline for classifying molecules"""
    if n_molecules is None:
        n_molecules = min(100, len(data['reactant']['num_atoms']))
    
    print(f"\n{'='*70}")
    print(f"MOLECULAR STRUCTURE CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Processing {n_molecules} molecules...")
    
    # Create molecules
    molecules = []
    for i in range(n_molecules):
        if i % 50 == 0:
            print(f"  Creating molecule {i}/{n_molecules}")
        try:
            positions = data['reactant']['positions'][i]
            charges = data['reactant']['charges'][i]
            mol = create_mol_from_data(positions, charges)
            molecules.append(mol)
        except Exception as e:
            print(f"  Error creating molecule {i}: {e}")
            continue
    
    print(f"\nSuccessfully created {len(molecules)} molecules")
    
    # Generate fingerprints
    print("\nGenerating Morgan fingerprints...")
    fingerprints = generate_morgan_fingerprints(molecules, radius=2, nBits=2048)
    print(f"Fingerprints shape: {fingerprints.shape}")
    
    # Also generate structural feature vectors for better clustering
    print("\nGenerating structural features...")
    feature_vectors = generate_feature_vectors(molecules)
    print(f"Feature vectors shape: {feature_vectors.shape}")
    
    # Normalize feature vectors
    scaler = StandardScaler()
    feature_vectors_normalized = scaler.fit_transform(feature_vectors)
    
    # Use a combination of fingerprints and features for better clustering
    combined_data = np.hstack([fingerprints.astype(np.float32), feature_vectors_normalized.astype(np.float32)])
    print(f"Combined data shape: {combined_data.shape}")
    
    # Run classifications
    kmeans_labels, kmeans_model = classify_kmeans(combined_data, n_clusters=n_clusters)
    hierarchical_labels, hierarchical_model = classify_hierarchical(combined_data, n_clusters=n_clusters)
    dbscan_labels, dbscan_model = classify_dbscan(combined_data, eps=5.0, min_samples=5)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'molecule_index': range(len(molecules)),
        'kmeans_cluster': kmeans_labels,
        'hierarchical_cluster': hierarchical_labels,
        'dbscan_cluster': dbscan_labels,
    })
    
    return molecules, fingerprints, results_df, {
        'kmeans': kmeans_model,
        'hierarchical': hierarchical_model,
        'dbscan': dbscan_model
    }

#========== VISUALIZATION FUNCTIONS ==========
def plot_cluster_distribution(results_df, method='kmeans'):
    """Plot distribution of clusters"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, method in enumerate(['kmeans_cluster', 'hierarchical_cluster', 'dbscan_cluster']):
        cluster_counts = results_df[method].value_counts().sort_index()
        axes[idx].bar(cluster_counts.index, cluster_counts.values, color='steelblue')
        axes[idx].set_xlabel('Cluster')
        axes[idx].set_ylabel('Number of Molecules')
        axes[idx].set_title(f'{method.replace("_", " ").title()} Distribution')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/sdruber/Documents/GitHub/research/Hackathon/cluster_distribution.png', dpi=100)
    plt.show()
    print("Cluster distribution plot saved!")

def save_results(results_df, molecules, fingerprints, models, output_dir='/Users/sdruber/Documents/GitHub/research/Hackathon/'):
    """Save classification results to file"""
    # Save dataframe
    results_df.to_csv(f'{output_dir}classification_results.csv', index=False)
    print(f"Results saved to classification_results.csv")
    
    # Save models and data
    classification_data = {
        'results_df': results_df,
        'fingerprints': fingerprints,
        'models': models,
    }
    
    with open(f'{output_dir}classification_models.pkl', 'wb') as f:
        pickle.dump(classification_data, f)
    print(f"Models and data saved to classification_models.pkl")

# %%
#========== EXAMPLE: RUN CLASSIFICATION ==========
print("\n" + "="*70)
print("STARTING MOLECULAR CLASSIFICATION")
print("="*70)

molecules, fingerprints, results_df, models = classify_molecules(n_molecules=100, n_clusters=5)

# %%
# Display results
print("\n" + "="*70)
print("CLASSIFICATION RESULTS SUMMARY")
print("="*70)
print(results_df.head(20))

# %%
# Save results
save_results(results_df, molecules, fingerprints, models)

# %%
# Visualize cluster distributions
plot_cluster_distribution(results_df)

# %%
print("\n" + "="*70)
print("CLASSIFICATION COMPLETE!")
print("="*70)
