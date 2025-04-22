# KIBA Dataset Project Handoff Document

## Project Overview
This project involves predicting binding affinities (KIBA scores) between small molecules and protein kinases using the KIBA (Kinase Inhibitor BioActivity) dataset. The goal is to build a model that can accurately predict how strongly a given drug-like molecule will bind to a protein target, which is critical for drug discovery applications.

## Dataset Description
- **Source**: Downloaded from Kaggle (blk1804/kiba-drug-binding-dataset)
- **Size**: 118,000+ rows
- **Key columns**:
  - `smiles`: SMILES string representation of molecules
  - `target_sequence`: Amino acid sequences of protein targets
  - `interaction`: KIBA score (binding affinity metric)

## Data Statistics
The KIBA score distribution shows:
- mean: 11.72
- std: 0.84
- min: 0.00
- 25%: 11.20
- 50%: 11.50 (median)
- 75%: 11.92
- max: 17.20

## Technical Implementation

### Molecular Representations
We chose MACCS keys (MACCSkeys.GenMACCSKeys) over Morgan/ECFP fingerprints for molecular representation because:
1. **Stability**: The Morgan fingerprint API triggered deprecation warnings
2. **Memory Efficiency**: MACCS keys produce a fixed 167-bit fingerprint per molecule
3. **Interpretability**: MACCS keys represent specific chemical substructures
4. **Performance**: MACCS keys work well on CPU with limited memory

Implementation:
```python
def smiles_to_maccs_fp(smiles_list):
    fingerprints = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = MACCSkeys.GenMACCSKeys(mol)
            arr = np.zeros((167,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr)
        else:
            fingerprints.append(np.zeros(167, dtype=np.int8))
    
    return np.array(fingerprints)
```

### Protein Sequence Representations
For protein sequences, we used HashingVectorizer with n-gram features because:
1. **Memory Constraints**: Our initial conjoint triad method produced 8000-dimensional vectors
2. **Scalability**: Feature hashing maintains constant memory regardless of vocabulary size
3. **Performance**: Character-level n-grams capture local sequence patterns efficiently

Implementation:
```python
def protein_feature_hashing(protein_sequences, n_features=1000, batch_size=1000):
    from sklearn.feature_extraction.text import HashingVectorizer
    
    vectorizer = HashingVectorizer(n_features=n_features, analyzer='char', ngram_range=(3, 3))
    
    all_encodings = []
    for i in range(0, len(protein_sequences), batch_size):
        batch = protein_sequences[i:i+batch_size]
        encodings = vectorizer.transform(batch).toarray()
        all_encodings.append(encodings)
    
    return np.vstack(all_encodings)
```

### Batch Processing
We implemented batch processing to handle the large dataset with limited memory:
```python
def process_smiles_in_batches(smiles_list, batch_size=1000):
    all_fingerprints = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        batch_fps = smiles_to_maccs_fp(batch)
        all_fingerprints.append(batch_fps)
    
    return np.vstack(all_fingerprints)
```

### Model Selection and Performance
We chose RandomForestRegressor due to:
1. **Interpretability**: Provides feature importance metrics
2. **CPU Efficiency**: Works well without GPU acceleration
3. **Robustness**: Handles heterogeneous features well

Current model performance:
- **RMSE**: 0.52
- **R2**: 0.60

## Feature Importance Visualization
We implemented a function to visualize the most important molecular features:
```python
def plot_top_features(model, n_features=20, figsize=(12, 8)):
    # Implementation details for visualizing feature importance
```

## Future Work Recommendations

### Model Improvements
1. **Target Transformation**:
   - The KIBA score distribution appears slightly right-skewed
   - Try log-transformation: `y_log = np.log1p(df['interaction'])`
   - Standard scaling may also help: `from sklearn.preprocessing import StandardScaler`

2. **Feature Engineering**:
   - Combine MACCS keys with selected RDKit 2D descriptors
   - Implement additional protein featurization methods like CTD (Composition/Transition/Distribution)
   - Explore domain-specific features like pharmacophore fingerprints

3. **Advanced Models**:
   - Gradient boosting with LightGBM (CPU-friendly and often outperforms RandomForest)
   - Simple neural networks using Keras or PyTorch
   - Implement hyperparameter tuning via GridSearchCV or Bayesian optimization

4. **Ensemble Methods**:
   - Create multiple models with different featurization techniques
   - Implement simple averaging or weighted averaging
   - Consider stacking with a meta-learner

### Technical Optimizations
1. **Memory Efficiency**:
   - Implement memory-mapped arrays for large intermediate results:
     ```python
     fp_memmap = np.memmap('fingerprints.dat', dtype=np.int8, mode='w+', shape=(n_samples, 167))
     ```
   - Explore incremental learning (partial_fit for compatible models)
   - Consider dimensionality reduction techniques like PCA or truncated SVD

2. **Feature Selection**:
   - Remove low-importance features using the existing importance metrics
   - Consider statistical feature selection (e.g., mutual information)

3. **Cross-Validation Strategy**:
   - Implement stratified K-fold to ensure representative splits
   - Consider target-based splitting to avoid protein target leakage

### Advanced Topics
1. **Transfer Learning**:
   - Pre-train on larger chemical datasets before fine-tuning on KIBA
   - Explore pre-trained protein language models like ESM-2 or ProtTrans

2. **Interpretability Enhancements**:
   - Map MACCS keys to their chemical meanings for better feature interpretation
   - Implement SHAP values for more robust feature importance analysis

3. **Deployment Optimization**:
   - Convert the model pipeline to ONNX format for faster inference
   - Implement a simple REST API using FastAPI or Flask

## Conclusion
The current implementation achieves reasonable performance (R² = 0.60) with memory-efficient and interpretable feature representations. The batched processing approach allows handling the full dataset on CPU hardware. Future improvements should focus on feature engineering, model selection, and memory optimization strategies.