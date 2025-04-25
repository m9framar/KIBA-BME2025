#!/bin/bash
#SBATCH --job-name=kiba_main_gpu   # Job name
#SBATCH --output=kiba_main_gpu.%j.out # Standard output file (%j expands to jobID)
#SBATCH --error=kiba_main_gpu.%j.err  # Standard error file (%j expands to jobID)
#SBATCH --time=08:00:00             # Wall clock time limit (hh:mm:ss)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Number of tasks (processes) per node
#SBATCH --cpus-per-task=8           # Increase CPU cores to get more memory (8 * 4G = 32G)
#SBATCH --mem-per-cpu=4000M         # Keep memory per CPU at max for GPU partition
#SBATCH --partition=gpu             # Specify the GPU partition
#SBATCH --gres=gpu:1                # Request 1 GPU

# --- Job Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted from: $SLURM_SUBMIT_DIR"

# Load necessary modules
module purge
module load singularity # Load the Singularity module

# --- Define Paths ---
# It's best practice to use absolute paths or paths relative to $SLURM_SUBMIT_DIR
# However, $SLURM_SUBMIT_DIR resolves differently on compute nodes (/project/...) causing permission issues.
# Define PROJECT_DIR using the known, writable home path instead for image output.
PROJECT_DIR="/home/nr_hafm/nr_haml2025/KIBA-BME2025" # <<< CONFIRMED home path
IMAGE_NAME="KIBA.sif"
# Define DEFINITION_FILE using the explicit home path as well
DEFINITION_FILE="$PROJECT_DIR/KIBA.def"
IMAGE_PATH="$PROJECT_DIR/$IMAGE_NAME" # Image will be built in your writable home project dir

# Data, Results, and Cache directories on the HOST filesystem
# Use $SLURM_SUBMIT_DIR for finding input data/results relative to submission location
HOST_DATA_DIR="$SLURM_SUBMIT_DIR/data"
HOST_RESULTS_DIR="$SLURM_SUBMIT_DIR/results"
HOST_CACHE_DIR="$HOME/.cache/huggingface" # Standard Hugging Face cache location

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

# --- Build Singularity Image (if needed) ---
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Singularity image $IMAGE_NAME not found at $IMAGE_PATH. Building..."
    # Use absolute paths for both output image and input definition file, add --fix-perms
    singularity build --fakeroot --fix-perms "$IMAGE_PATH" "$DEFINITION_FILE"
    if [ $? -ne 0 ]; then
        echo "Singularity image build failed! Exiting."
        exit 1
    fi
    echo "Singularity image built successfully: $IMAGE_PATH"
else
    echo "Using existing Singularity image: $IMAGE_PATH"
fi

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
