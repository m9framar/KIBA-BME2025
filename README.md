# KIBA-BME2025

## Project Overview

KIBA-BME2025 is a drug-target binding affinity prediction project using the KIBA (Kinase Inhibitor BioActivity) dataset. The project implements both traditional machine learning methods (Random Forest) and deep learning approaches using PyTorch and transformer-based embeddings for molecular and protein representations. This implementation is optimized for high-performance computing (HPC) environments with GPU acceleration through Singularity containers and Slurm workload management.

## Directory Structure

```
KIBA-BME2025/
├── src/
│   └── main.py                # Main GPU-accelerated implementation
├── data/
│   └── KIBA.csv               # Dataset file (placed manually)
├── results/                   # Output directory (created during execution)
│   ├── final_metrics.json     # Performance metrics 
│   ├── predictions_vs_actual.png  # Scatter plot of predictions
│   └── training_loss.png      # Learning curves visualization
├── docs/
│   ├── engineering_decisions.md   # Technical design documentation
│   ├── slurm_guide.md            # Guide for Slurm job management
│   └── komondor_usage_log.md     # Log of HPC usage and issues
├── KIBA.def                   # Singularity container definition
├── requirements.txt           # Python package dependencies
├── run_main_singularity.sbatch # Slurm batch script
└── README.md                  # This documentation
```

## Prerequisites

- Access to the Komondor HPC cluster or similar system
- Singularity/Apptainer installed and accessible
- Slurm workload manager
- Access to GPU nodes

## Setup Instructions

### Step 1: Prepare the Directory Structure

Create the necessary directories in your home, project, and scratch spaces:

```bash
# Create project structure in home directory
mkdir -p ~/KIBA-BME2025/{src,docs}

# Create data directory in project space (if you have permissions)
mkdir -p /project/your_project_id/KIBA-BME2025/data
mkdir -p /project/your_project_id/KIBA-BME2025/results

# Create cache directory in scratch space
mkdir -p /scratch/your_project_id/cache/{huggingface,matplotlib}
```

### Step 2: Obtain and Place Data

Place the KIBA.csv dataset file in your data directory:
```bash
# Copy KIBA.csv to your data directory
cp /path/to/your/KIBA.csv /project/your_project_id/KIBA-BME2025/data/
```

### Step 3: Build the Singularity Image

**⚠️ IMPORTANT:** Build the Singularity image on the login node, NOT within a Slurm job. As documented in the usage log, building within jobs caused multiple permission issues.

```bash
# Load singularity module
module purge
module load singularity

# Navigate to your base home directory (important!)
cd ~

# Build the image using fakeroot
singularity build --fakeroot KIBA.sif /path/to/your/KIBA.def
```

The image must be built from your base home directory (`~`) to avoid permission issues documented in the Komondor usage log.

## Running the Job

### Step 1: Configure the Slurm Script

Edit the run_main_singularity.sbatch file to update paths:

```bash
# Update these paths to match your environment
PROJECT_DIR="/home/your_username/KIBA-BME2025"  # Location of code
IMAGE_PATH="/home/your_username/KIBA.sif"       # Location of built image
HOST_DATA_DIR="/project/your_project_id/KIBA-BME2025/data"
HOST_RESULTS_DIR="/project/your_project_id/KIBA-BME2025/results"
HOST_CACHE_DIR="/scratch/your_project_id/cache"
```

### Step 2: Submit the Job

```bash
# Navigate to your project directory
cd ~/KIBA-BME2025

# Submit the job
sbatch run_main_singularity.sbatch
```

### Step 3: Monitor the Job

```bash
# Check job status
squeue -u $USER

# Check job output in real-time
tail -f kiba_mlp_*.log
```

## Understanding the Results

After the job completes, check the results directory for:

1. **final_metrics.json** - Contains performance metrics:
   - MSE (Mean Squared Error)
   - RMSE (Root Mean Squared Error)
   - R² (Coefficient of Determination)
   - CI (Concordance Index)
   - Training and validation metrics per epoch

2. **Visualizations**:
   - `training_loss.png` - Training and validation loss curves
   - predictions_vs_actual.png - Scatter plot of predicted vs. actual values

## Common Issues and Solutions

### Permission Issues

If you encounter permission errors when accessing directories:

```bash
# Check your group membership
groups

# Ensure you're a member of the project group
# Contact admin if needed to be added to the group
```

### GPU Detection Failures

If the code doesn't detect the GPU:

```bash
# Check GPU visibility inside the job
srun --pty nvidia-smi

# Make sure you're using the --nv flag with singularity exec
```

### Data Loading Issues

If the code can't find or load the KIBA.csv file:

```bash
# Verify the file exists in the expected location
ls -l /project/your_project_id/KIBA-BME2025/data/KIBA.csv

# Check permissions
chmod 644 /project/your_project_id/KIBA-BME2025/data/KIBA.csv
```

### Container Build Issues

The komondor_usage_log.md documents multiple issues with building containers in various locations. If you encounter similar issues:

- Build in your base home directory (`~`), not in subdirectories
- Use the `--fakeroot` flag
- Do not build within Slurm jobs
- Make sure you have sufficient memory (at least 16GB) for building

## Customizing the Run

To modify hyperparameters or model settings, you can pass arguments to the container:

```bash
# In run_main_singularity.sbatch:
singularity exec \
  --nv \
  --bind ${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}:ro \
  --bind ${HOST_RESULTS_DIR}:${CONTAINER_RESULTS_DIR}:rw \
  --bind ${HOST_CACHE_DIR}:${CONTAINER_CACHE_DIR}:rw \
  ${IMAGE_PATH} python /app/src/main.py \
    --smiles_model "alternative/model" \
    --protein_model "different/protein/model" \
    --batch_size 16 \
    --epochs 10
```

## Additional Resources

For more detailed information, refer to:

- engineering_decisions.md - Technical documentation on implementation choices
- slurm_guide.md - Comprehensive guide on using Slurm with Singularity
- komondor_usage_log.md - Chronological log of issues encountered and their resolutions


