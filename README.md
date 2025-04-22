# KIBA-BME2025
Project for the data analysis lab
KIBA Dataset Analysis
KIBA Dataset Description
Overview:
The KIBA dataset is a widely used benchmark in drug–target binding affinity research. It integrates bioactivity measurements from multiple sources into a single, unified score—the KIBA score—designed to reflect the interaction strength between small molecule drugs and protein targets.
Key Components:
•	Drug Representation:
Each drug is represented by its chemical structure in the form of a SMILES (Simplified Molecular Input Line Entry System) string. This standardized format allows for computational models to process and learn chemical features directly from the structure.
•	Target Representation:
Protein targets are provided as amino acid sequences. These sequences capture the primary structure of the protein, which is critical for predicting interaction sites and binding affinities.
•	Binding Affinity Scores (KIBA Score):
The KIBA score is a composite measure derived from integrating various experimental binding affinity measurements. It serves as the target variable for predictive modeling. The original dataset provides these scores in a unified format, and many benchmark studies (such as DeepDTA) use them directly without additional normalization.
Dataset Properties:
•	Size and Coverage:
The dataset contains tens of thousands to over a hundred thousand drug–target interaction entries, depending on the version used. Each entry links a specific drug to a target protein along with its corresponding binding affinity score.
•	Standardization:
The scores are provided in a consistent manner, meaning that the dataset creators have preprocessed the raw experimental measurements to yield a unified scale. This consistency is one of the reasons the KIBA dataset has become a standard benchmark in the field.
•	Evaluation Metrics:
Researchers typically assess model performance on the KIBA dataset using:
o	Mean Squared Error (MSE): Measures the average squared difference between predicted and actual binding affinities.
o	Concordance Index (CI): Evaluates the ranking capability of the model, i.e., how well the predicted ordering of interactions matches the actual ordering.
o	R-squared (R²): Indicates the proportion of variance in the binding affinity that is predictable from the input features.

Project Overview and Objectives
Our project aims to push the boundaries of drug–target binding affinity prediction by exploring a diverse range of modeling approaches on the KIBA dataset. We plan to experiment with models that span the spectrum from low-memory, CPU-friendly architectures to advanced methods that leverage large language models (LLMs) for encoding drugs and proteins.
Key Goals:
1.	Model Diversity:
o	Low-Memory, Local CPU Models:
We will implement and evaluate streamlined models optimized for environments with limited computational resources. These models are designed to be efficient, ensuring that they can be run on local machines without the need for high-end GPUs or extensive memory.
o	LLM-Based Encodings:
In parallel, we will explore the use of large language models to generate rich, context-aware embeddings for both drug SMILES strings and protein sequences. The goal is to assess whether the semantic richness captured by LLMs can enhance the prediction of binding affinities compared to traditional tokenization and embedding methods.
2.	Benchmarking and Comparisons:
We are committed to rigorous evaluation and fair comparisons. This involves:
o	Correct Train/Test Splits:
Ensuring that our data is split correctly to avoid data leakage and maintain consistency with existing studies. We will pay special attention to standardized protocols (e.g., cross-validation or specific train/test splits as reported in benchmark studies) to guarantee that our results are directly comparable to the literature.
o	Evaluation Metrics:
We will measure model performance using key metrics such as Mean Squared Error (MSE), Concordance Index (CI), and R-squared (R²). This will allow us to gauge both the prediction accuracy and the ranking capabilities of our models.
o	Direct Benchmark Comparisons:
By aligning our preprocessing, splitting protocols, and evaluation strategies with those used in seminal works (e.g., DeepDTA and subsequent studies), we aim to provide a clear and fair comparison of our models' performance against established benchmarks.
Simple local CPU based implementation:

Data Processing
1.	Data Loading: Dataset downloaded from Kaggle using kagglehub
2.	Column Preparation: Renamed columns for clarity
o	compound_iso_smiles → smiles
o	Ki, Kd and IC50 (KIBA Score) → interaction
Feature Engineering
1.	Compound Representation:
o	MACCS molecular fingerprints (167-bit binary vectors)
o	Implemented batch processing for efficiency
2.	Protein Representation:
o	Feature hashing of protein sequences
o	Character-level 3-grams
o	1000-dimensional feature vectors
Model Implementation
•	Algorithm: Random Forest Regressor
•	Hyperparameters:
o	100 estimators
o	Parallel processing (all cores)
o	Random state 42 for reproducibility
•	Training: 80% of data used for model training
•	Testing: 20% of data used for evaluation
Performance Metrics
•	Mean Squared Error (MSE): 0.273
•	R-squared (R²): 0.604
•	Concordance Index: 0.823

