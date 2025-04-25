#!/bin/bash
#SBATCH --job-name=kiba_main_gpu   # Job name
#SBATCH --output=/project/nr_haml2025/KIBA-BME2025/slurm_logs/kiba_main_gpu.%j.out # Log to project space
#SBATCH --error=/project/nr_haml2025/KIBA-BME2025/slurm_logs/kiba_main_gpu.%j.err  # Log to project space
#SBATCH --time=08:00:00             # Wall clock time limit (hh:mm:ss)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8           # CPU cores for your task (adjust if needed)
#SBATCH --mem-per-cpu=4000M         # Memory per CPU core (adjust if needed)
#SBATCH --partition=gpu             # Specify the GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU

# --- Job Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load necessary modules
module purge
module load singularity # Load the Singularity module

# --- Define Paths (Using Canonical Paths) ---
# Location of the pre-built Singularity image
IMAGE_PATH="/project/nr_haml2025/KIBA-BME2025/KIBA.sif"

# Data, Results, and Cache directories on the HOST filesystem
HOST_DATA_DIR="/project/nr_haml2025/KIBA-BME2025/data"       # Persistent data
HOST_RESULTS_DIR="/project/nr_haml2025/KIBA-BME2025/results"   # Persistent results
HOST_CACHE_DIR="/scratch/nr_haml2025/cache/huggingface" # Cache in scratch for speed
HOST_LOG_DIR="/project/nr_haml2025/KIBA-BME2025/slurm_logs" # Slurm logs

# Corresponding paths INSIDE the container
CONTAINER_APP_DIR="/app"
CONTAINER_DATA_DIR="/data"
CONTAINER_RESULTS_DIR="/results"
CONTAINER_CACHE_DIR="/root/.cache/huggingface" # Default cache location for root user inside container

# --- Prepare Host Directories ---
echo "Creating host directories (if they don't exist)..."
mkdir -p "$HOST_DATA_DIR"
mkdir -p "$HOST_RESULTS_DIR"
mkdir -p "$HOST_CACHE_DIR"
mkdir -p "$HOST_LOG_DIR"

# --- Build Singularity Image (REMOVED) ---
# Build step is now done manually on the login node.
# Ensure the image exists at $IMAGE_PATH before submitting the job.
echo "Checking for existing Singularity image: $IMAGE_PATH"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "ERROR: Singularity image not found at $IMAGE_PATH!"
    echo "Please build the image manually on the login node first."
    exit 1
fi
echo "Using existing Singularity image."

# --- Execute Script inside Container ---
echo "Running main.py inside Singularity container..."

# Check GPU availability on the node before launching container
echo "Node GPU Info:"
nvidia-smi || echo "nvidia-smi command failed or no GPU detected by Slurm."

singularity exec \
    --nv \# Enable Nvidia GPU support inside the container
    --bind "$HOST_DATA_DIR:$CONTAINER_DATA_DIR:ro" \# Mount host data dir (read-only)
    --bind "$HOST_RESULTS_DIR:$CONTAINER_RESULTS_DIR" \# Mount host results dir (writable)
    --bind "$HOST_CACHE_DIR:$CONTAINER_CACHE_DIR" \# Mount host cache dir for Hugging Face models
    "$IMAGE_PATH" \
    python "$CONTAINER_APP_DIR/src/main.py" \
        --output_dir "$CONTAINER_RESULTS_DIR" \
        # Add any other command-line arguments for main.py below
        # e.g., --epochs 10 --batch_size 64

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Singularity execution failed with exit code $EXIT_CODE."
else
    echo "Singularity execution completed successfully."
fi

# --- Job Completion ---
echo "Job finished at $(date)"
exit $EXIT_CODE
