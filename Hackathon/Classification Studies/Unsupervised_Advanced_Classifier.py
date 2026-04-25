#%%
# Advanced Unsupervised Learning for Molecular Structure Classification
# Tests Isolation Forest, HDBSCAN, Spectral Clustering, Gaussian Mixture Models

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
from scipy.spatial.distance import cdist
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("WARNING: HDBSCAN not installed. Install with: pip install hdbscan")

#========== LOAD DATA ==========
with open('/Users/sdruber/Documents/GitHub/research/Hackathon/DATA/Halo8/train.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Loaded {len(data['reactant']['num_atoms'])} transition states")

#========== BOND LENGTHS & MOL CREATION ==========
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

def generate_morgan_fingerprints(molecules, radius=2, nBits=2048):
    """Generate Morgan fingerprints for a list of molecules"""
    fingerprints = []
    for mol in molecules:
        try:
            gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
            fp = gen.GetFingerprint(mol)
            fingerprints.append(np.array(fp, dtype=np.float32))
        except Exception as e:
            fingerprints.append(np.zeros(nBits, dtype=np.float32))
    
    return np.array(fingerprints).astype(np.float32)

def generate_feature_vectors(molecules):
    """Generate feature vectors from structural properties"""
    feature_list = []
    for mol in molecules:
        features = compute_structural_features(mol)
        feature_list.append(list(features.values()))
    
    return np.array(feature_list)

#========== UNSUPERVISED CLASSIFICATION MODELS ==========

def classify_isolation_forest(data, contamination=0.1):
    """Classify using Isolation Forest (anomaly detection)"""
    print(f"\n{'='*70}")
    print("ISOLATION FOREST")
    print(f"{'='*70}")
    
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
    labels = model.fit_predict(data)
    
    # Convert -1 (anomaly) to cluster labels
    n_normal = (labels == 1).sum()
    n_anomaly = (labels == -1).sum()
    
    print(f"Normal samples: {n_normal}")
    print(f"Anomalous samples: {n_anomaly} ({100*n_anomaly/len(labels):.1f}%)")
    
    return labels, model

def classify_spectral_clustering(data, n_clusters=5, affinity='nearest_neighbors'):
    """Classify using Spectral Clustering"""
    print(f"\n{'='*70}")
    print("SPECTRAL CLUSTERING")
    print(f"{'='*70}")
    
    model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, 
                               random_state=42, n_init=10)
    labels = model.fit_predict(data)
    
    # Compute metrics
    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.4f} (higher is better)")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f} (lower is better)")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.2f} (higher is better)")
    
    for i in range(n_clusters):
        cluster_size = sum(labels == i)
        print(f"  Cluster {i}: {cluster_size} molecules")
    
    return labels, model

def classify_gaussian_mixture(data, n_components=5):
    """Classify using Gaussian Mixture Model"""
    print(f"\n{'='*70}")
    print("GAUSSIAN MIXTURE MODEL (GMM)")
    print(f"{'='*70}")
    
    model = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    labels = model.fit_predict(data)
    
    # Compute BIC and AIC
    bic = model.bic(data)
    aic = model.aic(data)
    
    # Soft assignments (probability of each point belonging to each cluster)
    soft_labels = model.predict_proba(data)
    max_prob = soft_labels.max(axis=1)
    mean_uncertainty = 1 - max_prob.mean()
    
    print(f"Number of components: {n_components}")
    print(f"BIC: {bic:.2f} (lower is better)")
    print(f"AIC: {aic:.2f} (lower is better)")
    print(f"Mean assignment uncertainty: {mean_uncertainty:.4f} (lower is better)")
    print(f"Log-Likelihood: {model.score(data):.2f}")
    
    for i in range(n_components):
        cluster_size = sum(labels == i)
        print(f"  Component {i}: {cluster_size} molecules")
    
    return labels, model

def classify_hdbscan(data, min_cluster_size=5, min_samples=5):
    """Classify using HDBSCAN (hierarchical density-based)"""
    if not HAS_HDBSCAN:
        print("HDBSCAN not available - skipping")
        return None, None
    
    print(f"\n{'='*70}")
    print("HDBSCAN")
    print(f"{'='*70}")
    
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = model.fit_predict(data)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    if n_clusters > 1:
        mask = labels != -1
        if sum(mask) > 0:
            silhouette = silhouette_score(data[mask], labels[mask])
            print(f"Silhouette Score: {silhouette:.4f}")
    
    for i in range(n_clusters):
        cluster_size = sum(labels == i)
        print(f"  Cluster {i}: {cluster_size} molecules")
    
    if n_noise > 0:
        print(f"  Noise points: {n_noise}")
    
    return labels, model

#========== COMPARISON FUNCTION ==========

def compare_all_models(data, n_molecules=100, n_clusters=5):
    """Compare all unsupervised learning models"""
    print(f"\n{'='*70}")
    print(f"COMPARING UNSUPERVISED LEARNING MODELS")
    print(f"Data shape: {data.shape}")
    print(f"{'='*70}")
    
    results = {}
    
    # 1. Spectral Clustering
    spec_labels, spec_model = classify_spectral_clustering(data, n_clusters=n_clusters)
    results['spectral'] = {'labels': spec_labels, 'model': spec_model}
    
    # 2. Gaussian Mixture Model
    gmm_labels, gmm_model = classify_gaussian_mixture(data, n_components=n_clusters)
    results['gmm'] = {'labels': gmm_labels, 'model': gmm_model}
    
    # 3. HDBSCAN (if available)
    if HAS_HDBSCAN:
        hdb_labels, hdb_model = classify_hdbscan(data, min_cluster_size=5)
        results['hdbscan'] = {'labels': hdb_labels, 'model': hdb_model}
    
    # 4. Isolation Forest (anomaly detection - different paradigm)
    iso_labels, iso_model = classify_isolation_forest(data, contamination=0.1)
    results['isolation_forest'] = {'labels': iso_labels, 'model': iso_model}
    
    return results

#========== MODEL SELECTION GUIDE ==========

def print_model_guide():
    """Print guide for choosing unsupervised model"""
    print(f"\n{'='*70}")
    print("MODEL SELECTION GUIDE FOR TRANSITION STATES")
    print(f"{'='*70}")
    print("""
SPECTRAL CLUSTERING:
  ✓ Best for: Well-separated, non-convex clusters
  ✓ Works well with molecular fingerprints
  ✓ Good for discovering reaction mechanisms
  ✗ Requires pre-specifying n_clusters
  
GAUSSIAN MIXTURE MODEL (GMM):
  ✓ Best for: Soft assignments (probabilities)
  ✓ Can estimate n_components from data
  ✓ Good for mixed mechanisms
  ✗ Assumes Gaussian distributions
  
HDBSCAN:
  ✓ Best for: Unknown, varying cluster sizes
  ✓ Doesn't require n_clusters (finds automatically)
  ✓ Robust to outliers/noise
  ✗ Slower than Spectral Clustering
  ✓ Great for reaction discovery!
  
ISOLATION FOREST:
  ✓ Best for: Anomaly detection
  ✓ Identifies unusual/rare reaction pathways
  ✗ Not traditional clustering
  ✓ Good for quality control

RECOMMENDATION FOR YOUR DATA:
  → Start with HDBSCAN (discovers natural structure)
  → Then try Spectral Clustering (for fixed n_clusters)
  → Use Isolation Forest to find rare reaction types
""")

#========== MAIN PIPELINE ==========

def run_unsupervised_classification(n_molecules=100, n_clusters=5):
    """Full unsupervised classification pipeline"""
    print(f"\n{'='*70}")
    print("UNSUPERVISED MOLECULAR CLASSIFICATION PIPELINE")
    print(f"{'='*70}")
    
    # Load and process molecules
    print(f"\nProcessing {n_molecules} molecules...")
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
            continue
    
    print(f"Successfully created {len(molecules)} molecules")
    
    # Generate fingerprints
    print("\nGenerating Morgan fingerprints...")
    fingerprints = generate_morgan_fingerprints(molecules, radius=2, nBits=2048)
    print(f"Fingerprints shape: {fingerprints.shape}")
    
    # Generate features
    print("Generating structural features...")
    features = generate_feature_vectors(molecules)
    print(f"Features shape: {features.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Combine fingerprints + features
    combined_data = np.hstack([fingerprints, features_normalized])
    print(f"Combined data shape: {combined_data.shape}")
    
    # Print model selection guide
    print_model_guide()
    
    # Compare all models
    results = compare_all_models(combined_data, n_molecules=n_molecules, n_clusters=n_clusters)
    
    # Save results
    results_summary = pd.DataFrame({
        'molecule_index': range(len(molecules)),
        'spectral_cluster': results['spectral']['labels'],
        'gmm_cluster': results['gmm']['labels'],
    })
    
    if HAS_HDBSCAN:
        results_summary['hdbscan_cluster'] = results['hdbscan']['labels']
    
    results_summary['isolation_forest_anomaly'] = results['isolation_forest']['labels']
    
    return molecules, fingerprints, features, results, results_summary

# %%
#========== EXAMPLE: RUN UNSUPERVISED CLASSIFICATION ==========
print("\n" + "="*70)
print("STARTING UNSUPERVISED CLASSIFICATION")
print("="*70)

molecules, fingerprints, features, results, results_summary = run_unsupervised_classification(
    n_molecules=100, 
    n_clusters=5
)

# %%
# Display results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(results_summary.head(20))

# %%
# Save results
results_summary.to_csv('/Users/sdruber/Documents/GitHub/research/Hackathon/unsupervised_classification_results.csv', index=False)
print("\nResults saved to unsupervised_classification_results.csv")

# Save models
with open('/Users/sdruber/Documents/GitHub/research/Hackathon/unsupervised_models.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Models saved to unsupervised_models.pkl")

# %%
print("\n" + "="*70)
print("UNSUPERVISED CLASSIFICATION COMPLETE!")
print("="*70)
