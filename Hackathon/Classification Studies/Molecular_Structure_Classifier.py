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
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
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
            gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nBits)
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

def analyze_cluster_properties(molecules, results_df, method='kmeans_cluster'):
    """Analyze structural properties of each cluster to understand their physical meaning"""
    print(f"\n{'='*70}")
    print(f"CLUSTER INTERPRETATION - Analyzing {method}")
    print(f"{'='*70}")
    
    # Compute features for all molecules
    all_features = []
    for mol in molecules:
        features = compute_structural_features(mol)
        all_features.append(features)
    
    features_df = pd.DataFrame(all_features)
    
    # Combine with cluster assignments
    analysis_df = pd.concat([results_df[[method]], features_df], axis=1)
    
    # For each cluster, compute statistics
    cluster_stats = {}
    unique_clusters = sorted(analysis_df[method].unique())
    
    for cluster_id in unique_clusters:
        cluster_data = analysis_df[analysis_df[method] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'n_molecules': len(cluster_data),
            'percentage': f"{100*len(cluster_data)/len(analysis_df):.1f}%"
        }
        
        # Compute mean and std for each feature
        for col in features_df.columns:
            stats[f'{col}_mean'] = cluster_data[col].mean()
            stats[f'{col}_std'] = cluster_data[col].std()
        
        cluster_stats[cluster_id] = stats
        
        # Print summary
        print(f"\n--- CLUSTER {cluster_id} ({stats['n_molecules']} molecules, {stats['percentage']}) ---")
        print(f"  Atoms:           {stats['num_atoms_mean']:.1f} ± {stats['num_atoms_std']:.1f}")
        print(f"  Bonds:           {stats['num_bonds_mean']:.1f} ± {stats['num_bonds_std']:.1f}")
        print(f"  Rings:           {stats['num_rings_mean']:.1f} ± {stats['num_rings_std']:.1f}")
        print(f"  Aromatic Rings:  {stats['num_aromatic_rings_mean']:.1f} ± {stats['num_aromatic_rings_std']:.1f}")
        print(f"  Double Bonds:    {stats['num_double_bonds_mean']:.1f} ± {stats['num_double_bonds_std']:.1f}")
        print(f"  Heteroatoms:     {stats['num_heteroatoms_mean']:.1f} ± {stats['num_heteroatoms_std']:.1f}")
        print(f"  Mol Weight:      {stats['mol_weight_mean']:.1f} ± {stats['mol_weight_std']:.1f}")
        print(f"  LogP:            {stats['logp_mean']:.2f} ± {stats['logp_std']:.2f}")
        print(f"  TPSA:            {stats['tpsa_mean']:.1f} ± {stats['tpsa_std']:.1f}")
        print(f"  Rotatable Bonds: {stats['num_rotatable_bonds_mean']:.1f} ± {stats['num_rotatable_bonds_std']:.1f}")
    
    return cluster_stats, analysis_df

def visualize_cluster_properties(analysis_df, method='kmeans_cluster', output_dir='/Users/sdruber/Documents/GitHub/research/Hackathon/'):
    """Create box plots of key properties for each cluster"""
    properties = ['num_atoms', 'num_bonds', 'num_rings', 'mol_weight', 'logp', 'tpsa', 'num_heteroatoms']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, prop in enumerate(properties):
        ax = axes[idx]
        clusters = sorted(analysis_df[method].unique())
        data_by_cluster = [analysis_df[analysis_df[method] == c][prop].values for c in clusters]
        
        bp = ax.boxplot(data_by_cluster, labels=clusters)
        ax.set_xlabel('Cluster')
        ax.set_ylabel(prop.replace('_', ' ').title())
        ax.grid(axis='y', alpha=0.3)
    
    # Hide the last unused subplot
    axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}cluster_properties_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    print(f"Cluster properties plot saved to cluster_properties_comparison.png")

def identify_bond_patterns(molecules, results_df, data, method='kmeans_cluster'):
    """Identify which atom pairs are forming/breaking bonds in transition states"""
    print(f"\n{'='*70}")
    print("BOND FORMATION/BREAKING PATTERNS")
    print(f"{'='*70}")
    
    unique_clusters = sorted(results_df[method].unique())
    
    for cluster_id in unique_clusters:
        cluster_indices = results_df[results_df[method] == cluster_id]['molecule_index'].values
        
        print(f"\n--- CLUSTER {cluster_id} ({len(cluster_indices)} molecules) ---")
        print("Analyzing bond patterns...")
        
        bond_patterns = {}
        
        for mol_idx in cluster_indices[:5]:  # Analyze first 5 molecules in each cluster
            mol = molecules[mol_idx]
            
            # Analyze bonds in the transition state
            for bond in mol.GetBonds():
                begin_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                end_atom = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                
                atom_pair = f"{begin_atom.GetSymbol()}-{end_atom.GetSymbol()}"
                bond_type = str(bond.GetBondType())
                
                key = f"{atom_pair} ({bond_type})"
                bond_patterns[key] = bond_patterns.get(key, 0) + 1
        
        # Sort and display
        sorted_patterns = sorted(bond_patterns.items(), key=lambda x: x[1], reverse=True)
        print("  Most common bonds in transition state:")
        for bond_type, count in sorted_patterns[:8]:
            print(f"    {bond_type}: {count} occurrences")

def track_geometric_changes(molecules, results_df, data, method='kmeans_cluster'):
    """Track bond distances and angles at transition state"""
    print(f"\n{'='*70}")
    print("GEOMETRIC ANALYSIS - BOND DISTANCES & ANGLES")
    print(f"{'='*70}")
    
    unique_clusters = sorted(results_df[method].unique())
    
    for cluster_id in unique_clusters:
        cluster_indices = results_df[results_df[method] == cluster_id]['molecule_index'].values
        
        print(f"\n--- CLUSTER {cluster_id} ({len(cluster_indices)} molecules) ---")
        
        bond_distances = []
        bond_angles = []
        
        for mol_idx in cluster_indices[:5]:
            mol = molecules[mol_idx]
            conf = mol.GetConformer()
            
            # Get all bond distances
            for bond in mol.GetBonds():
                pos1 = conf.GetAtomPosition(bond.GetBeginAtomIdx())
                pos2 = conf.GetAtomPosition(bond.GetEndAtomIdx())
                distance = pos1.Distance(pos2)
                bond_distances.append(distance)
            
            # Get angles (3 consecutive atoms)
            for atom in mol.GetAtoms():
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
                if len(neighbors) >= 2:
                    for i in range(len(neighbors)):
                        for j in range(i+1, len(neighbors)):
                            idx1, idx2, idx3 = neighbors[i], atom.GetIdx(), neighbors[j]
                            pos1 = conf.GetAtomPosition(idx1)
                            pos2 = conf.GetAtomPosition(idx2)
                            pos3 = conf.GetAtomPosition(idx3)
                            
                            # Calculate angle
                            v1 = (pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z)
                            v2 = (pos3.x - pos2.x, pos3.y - pos2.y, pos3.z - pos2.z)
                            
                            dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                            mag1 = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
                            mag2 = (v2[0]**2 + v2[1]**2 + v2[2]**2)**0.5
                            
                            if mag1 > 0 and mag2 > 0:
                                cos_angle = dot / (mag1 * mag2)
                                angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
                                angle_deg = np.degrees(angle_rad)
                                bond_angles.append(angle_deg)
        
        if bond_distances:
            print(f"  Bond distances: {np.mean(bond_distances):.3f} ± {np.std(bond_distances):.3f} Å")
            print(f"    Range: {np.min(bond_distances):.3f} - {np.max(bond_distances):.3f} Å")
        
        if bond_angles:
            print(f"  Bond angles: {np.mean(bond_angles):.1f} ± {np.std(bond_angles):.1f}°")
            print(f"    Range: {np.min(bond_angles):.1f} - {np.max(bond_angles):.1f}°")

def classify_reaction_types(molecules, results_df, data, method='kmeans_cluster'):
    """Classify transition states by likely reaction mechanism"""
    print(f"\n{'='*70}")
    print("REACTION TYPE CLASSIFICATION")
    print(f"{'='*70}")
    
    unique_clusters = sorted(results_df[method].unique())
    
    reaction_type_descriptions = {
        'high_heteroatoms': "SN2-like mechanism (nucleophilic substitution with polar TS)",
        'high_double_bonds': "Elimination or C=C forming reaction",
        'high_rings': "Cycloaddition or ring-opening mechanism",
        'high_aromatic': "Aromatic rearrangement or electrophilic aromatic substitution",
        'high_rotation': "Conformational/rotational transition state",
        'balanced': "Mixed mechanism or complex multi-step reaction"
    }
    
    for cluster_id in unique_clusters:
        cluster_indices = results_df[results_df[method] == cluster_id]['molecule_index'].values
        
        # Compute average features
        heteroatom_counts = []
        double_bond_counts = []
        ring_counts = []
        aromatic_ring_counts = []
        rotatable_bond_counts = []
        
        for mol_idx in cluster_indices:
            mol = molecules[mol_idx]
            features = compute_structural_features(mol)
            heteroatom_counts.append(features['num_heteroatoms'])
            double_bond_counts.append(features['num_double_bonds'])
            ring_counts.append(features['num_rings'])
            aromatic_ring_counts.append(features['num_aromatic_rings'])
            rotatable_bond_counts.append(features['num_rotatable_bonds'])
        
        avg_heteroatoms = np.mean(heteroatom_counts)
        avg_double_bonds = np.mean(double_bond_counts)
        avg_rings = np.mean(ring_counts)
        avg_aromatic = np.mean(aromatic_ring_counts)
        avg_rotation = np.mean(rotatable_bond_counts)
        
        # Classify based on feature dominance
        features_scores = {
            'high_heteroatoms': avg_heteroatoms,
            'high_double_bonds': avg_double_bonds,
            'high_rings': avg_rings,
            'high_aromatic': avg_aromatic,
            'high_rotation': avg_rotation,
        }
        
        dominant_type = max(features_scores, key=features_scores.get)
        dominant_score = features_scores[dominant_type]
        
        # Check if scores are balanced
        normalized_scores = {k: v/max(features_scores.values()) for k, v in features_scores.items()}
        if max(normalized_scores.values()) < 1.3:  # Close to balanced
            reaction_class = "balanced"
        else:
            reaction_class = dominant_type
        
        print(f"\n--- CLUSTER {cluster_id} ({len(cluster_indices)} molecules) ---")
        print(f"Predicted Reaction Type: {reaction_type_descriptions.get(reaction_class, 'Unknown')}")
        print(f"\nFeature breakdown:")
        print(f"  Heteroatoms (SN2):     {avg_heteroatoms:.2f}")
        print(f"  Double Bonds (E1/E2):  {avg_double_bonds:.2f}")
        print(f"  Rings (Cycloaddition): {avg_rings:.2f}")
        print(f"  Aromatic (Ar-Sub):     {avg_aromatic:.2f}")
        print(f"  Rot. Bonds (Conf):     {avg_rotation:.2f}")


    """Identify which features best distinguish between clusters"""
    print(f"\n{'='*70}")
    print("KEY DISTINGUISHING FEATURES")
    print(f"{'='*70}")
    
    feature_cols = [col for col in analysis_df.columns if col != method and not col.startswith('molecule')]
    
    # For each feature, compute variance between clusters
    feature_importance = {}
    for feat in feature_cols:
        cluster_means = analysis_df.groupby(method)[feat].mean()
        cluster_stds = analysis_df.groupby(method)[feat].std()
        
        # Between-cluster variance / within-cluster variance ratio (simplified F-statistic)
        between_var = cluster_means.var()
        within_var = cluster_stds.mean()
        
        if within_var > 0:
            f_ratio = between_var / within_var
        else:
            f_ratio = 0
        
        feature_importance[feat] = f_ratio
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMost distinguishing features (higher = better separates clusters):")
    for feat, importance in sorted_features[:10]:
        print(f"  {feat:30s}: {importance:.2f}")
    
    return feature_importance

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
# Analyze what each cluster means physically
cluster_stats, analysis_df = analyze_cluster_properties(molecules, results_df, method='kmeans_cluster')

# %%
# Identify which features distinguish clusters
feature_importance = identify_distinguishing_features(analysis_df, method='kmeans_cluster')

# %%
# Visualize cluster properties
visualize_cluster_properties(analysis_df, method='kmeans_cluster')

# %%
# TRANSITION STATE SPECIFIC ANALYSIS
# Identify bond formation/breaking patterns
identify_bond_patterns(molecules, results_df, data, method='kmeans_cluster')

# %%
# Track geometric changes at transition state
track_geometric_changes(molecules, results_df, data, method='kmeans_cluster')

# %%
# Classify reaction types based on structural features
classify_reaction_types(molecules, results_df, data, method='kmeans_cluster')

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
