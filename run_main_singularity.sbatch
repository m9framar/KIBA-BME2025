#!/bin/bash
#SBATCH --job-name=kiba_mlp_train       # Job name
#SBATCH --partition=gpu                 # Partition name (e.g., gpu)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --nodes=1                       # Run all processes on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=8               # Number of CPU cores per task (increased for build)
#SBATCH --mem-per-cpu=4000M             # Memory per CPU core (increased for build)
#SBATCH --time=02:00:00                 # Time limit hrs:min:sec
#SBATCH --output=kiba_mlp_%j.log        # Standard output and error log (%j expands to job ID)

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
echo "Working Directory: $(pwd)"
echo "GPU(s) assigned: $CUDA_VISIBLE_DEVICES"

# --- Configuration ---
# Project directory location on the login node (where code and definition file reside)
# CHANGED: Pointing to the new location in the home directory
PROJECT_DIR="/home/nr_hafm/KIBA-BME2025"

# Path where the Singularity image WILL BE STORED (must be writable by user)
# CHANGED: Image will now reside within the project directory in /home
IMAGE_PATH="${PROJECT_DIR}/KIBA.sif"

# Path to the Singularity definition file (needed only for manual build)
# CHANGED: Definition file is now in the project directory in /home
DEFINITION_FILE="${PROJECT_DIR}/KIBA.def"

# Host directory containing the input data (mounted read-only)
# UNCHANGED: Keeping data in /project
HOST_DATA_DIR="/project/nr_haml2025/KIBA-BME2025/data"

# Host directory for saving results (mounted read-write)
# UNCHANGED: Keeping results in /project
HOST_RESULTS_DIR="/project/nr_haml2025/KIBA-BME2025/results"

# Host directory for caching models/tokenizers (mounted read-write)
# UNCHANGED: Keeping cache in /scratch
HOST_CACHE_DIR="/scratch/nr_haml2025/cache"

# Container paths (these generally don't need changing)
CONTAINER_DATA_DIR="/data"
CONTAINER_RESULTS_DIR="/results"
CONTAINER_CACHE_DIR="/cache"
CONTAINER_APP_DIR="/app" # Matches %files section in KIBA.def

# --- Environment Setup ---
echo "Loading Singularity module..."
module purge
module load singularity

# --- Create Host Directories if they don't exist ---
# Important: Run this on the login node *before* submitting, or ensure Slurm has write access
# mkdir -p "$HOST_DATA_DIR" # Data should already exist
mkdir -p "$HOST_RESULTS_DIR"
mkdir -p "$HOST_CACHE_DIR"
echo "Ensured host directories exist (or will be created by user):"
echo "  Data: $HOST_DATA_DIR"
echo "  Results: $HOST_RESULTS_DIR"
echo "  Cache: $HOST_CACHE_DIR"

# --- Build Image (Manual Step Recommended) ---
# Building inside the Slurm job is complex due to permissions.
# It's recommended to build the image MANUALLY on the LOGIN NODE beforehand:
#
# cd /home/nr_hafm/KIBA-BME2025  # Go to the project directory
# module purge && module load singularity
# singularity build --fakeroot KIBA.sif KIBA.def
#
# echo "Checking if Singularity image exists: $IMAGE_PATH"
# if [ ! -f "$IMAGE_PATH" ]; then
#     echo "ERROR: Singularity image not found at $IMAGE_PATH."
#     echo "Please build it manually on the login node first."
#     exit 1
# fi

# --- Run Container ---
echo "Running application in Singularity container..."
singularity exec \
    --nv \
    --bind "$HOST_DATA_DIR:$CONTAINER_DATA_DIR:ro" \
    --bind "$HOST_RESULTS_DIR:$CONTAINER_RESULTS_DIR:rw" \
    --bind "$HOST_CACHE_DIR:$CONTAINER_CACHE_DIR:rw" \
    --env HF_HOME="$CONTAINER_CACHE_DIR/huggingface" \
    --env MPLCONFIGDIR="$CONTAINER_CACHE_DIR/matplotlib" \
    "$IMAGE_PATH" \
    python "$CONTAINER_APP_DIR/src/main.py" --output_dir "$CONTAINER_RESULTS_DIR"

EXIT_CODE=$?
echo "Singularity execution finished with exit code: $EXIT_CODE"
echo "Date: $(date)"

exit $EXIT_CODE
