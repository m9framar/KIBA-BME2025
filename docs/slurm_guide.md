# Slurm Job Management on Komondor HPC with Singularity

This guide provides essential information for running jobs, particularly containerized applications using Singularity, on the Komondor HPC system using the Slurm workload manager.

**Table of Contents**

1.  [What is Slurm?](#what-is-slurm)
2.  [Basic Slurm Commands](#basic-slurm-commands)
3.  [What is Singularity?](#what-is-singularity)
4.  [Using Singularity on Komondor](#using-singularity-on-komondor)
    *   [Building a Singularity Image](#building-a-singularity-image)
    *   [Running a Singularity Container with Slurm](#running-a-singularity-container-with-slurm)
5.  [Slurm Job Script Structure (with Singularity)](#slurm-job-script-structure-with-singularity)
6.  [Requesting Resources Effectively](#requesting-resources-effectively)
    *   [Partitions](#partitions)
    *   [CPU Cores (`--cpus-per-task`)](#cpu-cores---cpus-per-task)
    *   [Memory (`--mem` or `--mem-per-cpu`)](#memory---mem-or---mem-per-cpu)
    *   [Wall Time (`--time`)](#wall-time---time)
    *   [GPUs (`--gres=gpu`)](#gpus---gresgpu)
7.  [Monitoring Jobs](#monitoring-jobs)
    *   [`squeue`](#squeue)
    *   [`sacct`](#sacct)
    *   [`scontrol show job`](#scontrol-show-job)
    *   [Checking Output Files](#checking-output-files)
8.  [Interactive Jobs (`srun`)](#interactive-jobs-srun)
9.  [Job Arrays (`--array`)](#job-arrays---array)
10. [Managing Dependencies within Singularity](#managing-dependencies-within-singularity)
11. [Data, Results, and Cache Management](#data-results-and-cache-management)
12. [File System Best Practices (`$HOME` vs. `$SCRATCH`)](#file-system-best-practices-home-vs-scratch)
13. [Common Errors and Troubleshooting](#common-errors-and-troubleshooting)
    *   [Slurm Errors](#slurm-errors)
    *   [Singularity Errors](#singularity-errors)
14. [Getting Help](#getting-help)

## What is Slurm?

Slurm (Simple Linux Utility for Resource Management) is an open-source workload manager designed for Linux clusters. It allocates resources (compute nodes), provides a framework for starting, executing, and monitoring work (jobs), and arbitrates resource contention by managing a queue of pending work.

## Basic Slurm Commands

| Command | Description |
|---|---|
| `sbatch <script.sh>` | Submit a batch job script to the queue. This is the most common way to run non-interactive jobs. |
| `srun <command>` | Run a command on allocated compute resources. Often used for interactive jobs or specific parallel tasks within a batch job. |
| `squeue` | View the status of all jobs in the queue. |
| `squeue -u <username>` | View only your jobs. Highly recommended! |
| `squeue -p <partition_name>` | View jobs in a specific partition. |
| `scancel <job_id>` | Cancel a specific pending or running job. |
| `scancel -u <username>` | Cancel all jobs belonging to a user. Use with caution! |
| `sinfo` | View information about Slurm nodes and partitions (queues). |
| `sinfo -p <partition_name>` | Show detailed info for a specific partition. |
| `sacct` | View accounting information about past jobs (useful for checking resource usage). |
| `scontrol show job <job_id>` | Show detailed configuration and status information for a specific job. |

## What is Singularity?

Singularity is a container platform optimized for High-Performance Computing (HPC) environments. It allows you to package applications, libraries, and dependencies into a single file (a container image, typically `.sif`), ensuring reproducibility and simplifying deployment across different systems. Unlike Docker, Singularity is designed with security (running containers as the user, not root by default) and HPC integration (e.g., GPU support, network filesystems) in mind.

## Using Singularity on Komondor

Komondor provides the `singularity` module. Always load it before using Singularity commands:

```bash
module load singularity
```

### Building a Singularity Image

You define a container using a definition file (e.g., `MyProject.def`). You build the image (e.g., `MyProject.sif`) using the `singularity build` command.

```bash
# Load the singularity module first
module load singularity

# Build the image (this might take time and requires specific permissions)
# Use --fakeroot if building as a non-root user (common on HPC)
singularity build --fakeroot <image_name>.sif <definition_file>.def
```

*   **Definition File (`.def`):** Specifies the base OS/image, software installation steps (`%post`), files to copy (`%files`), environment variables (`%environment`), and the default run command (`%runscript`).
*   **`--fakeroot`:** Essential for building images as a regular user on HPC systems. It simulates root privileges using user namespaces.
*   **Build Location:** It's often best to build images on login nodes or dedicated build nodes if available, rather than within a compute job, unless the build is part of an automated workflow.

For GPU-accelerated applications, ensure your definition file uses a base image with the necessary CUDA toolkit (e.g., `nvidia/cuda`, `pytorch/pytorch` from Docker Hub) and that your application inside the container can access the GPU.

### Running a Singularity Container with Slurm

You execute commands within your built container using `singularity exec` or run its defined default command using `singularity run`.

*   `singularity exec [options] <image.sif> <command>`: Executes a specific `<command>` inside the container.
*   `singularity run [options] <image.sif> [args]`: Executes the `%runscript` defined in the image's definition file, passing optional `[args]` to it.

**Key Singularity Options for Slurm Jobs:**

*   `--nv`: **Crucial for GPU jobs.** Enables NVIDIA GPU support inside the container by mapping the host's GPU drivers and libraries.
*   `--bind <host_path>:<container_path>` or `--bind <host_path>:<container_path>:ro`: Mounts directories from the host system into the container. This is essential for accessing data, saving results, and sharing caches. `:ro` makes the mount read-only inside the container.
*   `--pwd /path/inside/container`: Sets the working directory inside the container.
*   `--env VAR=value`: Set environment variables inside the container.

## Slurm Job Script Structure (with Singularity)

```bash
#!/bin/bash
#SBATCH --job-name=singularity_job # Descriptive job name
#SBATCH --output=slurm_out/%x.%j.out # Standard output (%x=jobname, %j=jobID)
#SBATCH --error=slurm_out/%x.%j.err  # Standard error
#SBATCH --time=04:00:00            # Max wall time (hh:mm:ss)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks-per-node=1        # Tasks per node (usually 1 unless using MPI/multi-process)
#SBATCH --cpus-per-task=4          # CPU cores for your task
#SBATCH --mem-per-cpu=4000M        # Memory per CPU core (check partition limits!)
#SBATCH --partition=gpu            # Specify partition (e.g., gpu)
#SBATCH --gres=gpu:1               # Request 1 GPU (adjust type if needed, e.g., gpu:A100:1)

# --- Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID, Job Name: $SLURM_JOB_NAME"
echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Allocated CPUs: $SLURM_CPUS_PER_TASK, Allocated Memory: $SLURM_MEM_PER_CPU MB per CPU"
echo "Allocated GPUs: $SLURM_GPUS_ON_NODE"

# Create directory for Slurm output files if it doesn't exist
mkdir -p slurm_out

# Load necessary modules
module purge
module load singularity # Load singularity module
# module load cuda/XXX # Often NOT needed if using --nv and container has CUDA toolkit

# Define paths (use absolute paths or $SLURM_SUBMIT_DIR)
PROJECT_DIR="$SLURM_SUBMIT_DIR"
IMAGE_NAME="MyProject.sif"
IMAGE_PATH="$PROJECT_DIR/$IMAGE_NAME"
HOST_DATA_DIR="$PROJECT_DIR/data"       # Host data location
HOST_RESULTS_DIR="$PROJECT_DIR/results"   # Host results location
HOST_CACHE_DIR="$HOME/.cache/my_app_cache" # Host cache location

# Create results/cache directories on the host if they don't exist
mkdir -p "$HOST_RESULTS_DIR"
mkdir -p "$HOST_CACHE_DIR"

# --- Build Singularity Image (Optional - can be done outside job) ---
# cd "$PROJECT_DIR" # Ensure correct build context
# if [ ! -f "$IMAGE_PATH" ]; then
#     echo "Building Singularity image..."
#     singularity build --fakeroot "$IMAGE_NAME" MyProject.def
#     # Add error checking
# fi

# --- Execute Script inside Container ---
echo "Running script inside Singularity container..."

# Check GPU availability on the node
echo "Node GPU Info:"
nvidia-smi || echo "nvidia-smi command failed or no GPU detected by Slurm."

singularity exec \
    --nv \# Enable GPU access
    --bind "$HOST_DATA_DIR:/data:ro" \# Mount data read-only
    --bind "$HOST_RESULTS_DIR:/results" \# Mount results writable
    --bind "$HOST_CACHE_DIR:/root/.cache/my_app_cache" \# Mount cache
    "$IMAGE_PATH" \
    python /app/main_script.py --input /data --output /results --config /app/config.yaml
    # Add any other arguments for your script here

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Singularity execution failed with exit code $EXIT_CODE."
else
    echo "Singularity execution completed successfully."
fi

# --- Job Completion ---
echo "Job finished at $(date)"
exit $EXIT_CODE
```

## Requesting Resources Effectively

Requesting appropriate resources is crucial for job scheduling and efficient cluster usage. Over-requesting can lead to longer queue times, while under-requesting can cause jobs to fail.

### Partitions
Partitions (queues) group nodes with specific characteristics (hardware, time limits). Use `sinfo` to see available partitions. Choose the one matching your needs (e.g., `gpu` for GPU nodes).

### CPU Cores (`--cpus-per-task`)
Request the number of CPU cores your application can effectively use. For single-threaded applications, request 1. For multi-threaded applications (like some NumPy/SciPy operations or PyTorch DataLoader workers), request more (e.g., 4, 8). Don't request more cores than available on a node or more than your code can utilize.

### Memory (`--mem` or `--mem-per-cpu`)
*   `--mem=<size>[M|G]`: Total memory for the job *per node*. Example: `--mem=16G`.
*   `--mem-per-cpu=<size>[M|G]`: Memory *per requested CPU core*. Example: `--mem-per-cpu=4000M`.
*   **Recommendation:** Use `--mem-per-cpu` as it scales with `--cpus-per-task` and often aligns better with cluster policies (like Komondor's GPU partition limit of 4000M/core).
*   Check partition limits using `sinfo -p <partition_name> -o "%N %c %m %G %l"` (shows Nodes, CPUs, Memory(MB), GPUs, TimeLimit).
*   Estimate your job's memory usage by testing on smaller datasets or checking `sacct` results from previous runs.

### Wall Time (`--time`)
Format: `DD-HH:MM:SS`. Estimate how long your job will run and add a buffer (e.g., 20-30%). Jobs exceeding their requested time will be killed. Check partition time limits with `sinfo`.

### GPUs (`--gres=gpu`)
*   `--gres=gpu:<count>`: Request a specific number of generic GPUs. Example: `--gres=gpu:1`.
*   `--gres=gpu:<type>:<count>`: Request GPUs of a specific type (if available and configured). Example: `--gres=gpu:A100:2`. Check available GPU types with `sinfo` or cluster documentation.
*   Only request GPUs if your code (and container) is set up to use them.

## Monitoring Jobs

### `squeue`
*   `squeue -u $USER`: Shows your jobs.
*   **Key Columns:**
    *   `JOBID`: Unique job identifier.
    *   `PARTITION`: Queue the job is in.
    *   `NAME`: Job name (`#SBATCH --job-name`).
    *   `USER`: Your username.
    *   `ST`: Job Status (see below).
    *   `TIME`: Time the job has been running.
    *   `NODES`: Number of nodes allocated.
    *   `NODELIST(REASON)`: Nodes allocated or reason for being pending.
*   **Common Status Codes (`ST`):**
    *   `PD` (Pending): Waiting for resources.
    *   `R` (Running): Executing on compute node(s).
    *   `CG` (Completing): Finishing up, cleaning resources.
    *   `F` (Failed): Job terminated with non-zero exit code.
    *   `CA` (Cancelled): Job cancelled by user or admin.
    *   `TO` (Timeout): Job killed after exceeding wall time limit.
    *   `NF` (Node Fail): Job terminated due to node failure.

### `sacct`
Used to view information about *completed* jobs. Useful for checking resource usage.
*   `sacct -j <job_id>`: Show info for a specific job.
*   `sacct -j <job_id> --format=JobID,JobName,Partition,AllocCPUS,ReqMem,MaxRSS,Elapsed,State`: Show specific fields.
    *   `ReqMem`: Requested memory.
    *   `MaxRSS`: Maximum Resident Set Size (peak memory usage). Compare this to `ReqMem` to see if you requested appropriate memory.
    *   `Elapsed`: Actual run time.
    *   `State`: Final job state.
*   `sacct`: Shows recent jobs.

### `scontrol show job`
`scontrol show job <job_id>`: Provides very detailed information about a job's configuration, requested/allocated resources, state, and more. Useful for debugging scheduling issues.

### Checking Output Files
Regularly check the `.out` and `.err` files specified by `#SBATCH --output` and `#SBATCH --error`. These contain your application's standard output and errors, crucial for debugging.

## Interactive Jobs (`srun`)

Useful for debugging, testing code snippets, or interactive data exploration on compute nodes.

```bash
# Request an interactive session on a GPU node
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem-per-cpu=4000M --time=01:00:00 --pty bash -i
```

*   `--pty bash -i`: Requests a pseudo-terminal (`pty`) and runs an interactive bash shell (`bash -i`).

Once you have the interactive shell on the compute node:

```bash
# Load modules
module load singularity

# Navigate to your project directory
cd $SLURM_SUBMIT_DIR

# Run commands directly or start an interactive shell inside your container
singularity shell --nv --bind <mounts> MyProject.sif

# Now you are inside the container
python
>>> import torch
>>> print(torch.cuda.is_available())
exit()

# Exit the container shell
exit

# Exit the srun session
exit
```

## Job Arrays (`--array`)

Efficiently submit and manage large numbers of similar jobs (e.g., processing multiple input files, hyperparameter sweeps).

```bash
#!/bin/bash
#SBATCH --job-name=my_array_job
#SBATCH --output=slurm_out/%x_%A_%a.out # %A=array job ID, %a=array task ID
#SBATCH --error=slurm_out/%x_%A_%a.err
#SBATCH --array=1-100%10 # Run tasks 1 to 100, max 10 running concurrently
# ... other SBATCH directives (time, mem, cpu, partition) ...

# Load modules
module load singularity

# Get the task ID for this specific job instance
TASK_ID=$SLURM_ARRAY_TASK_ID
echo "Running array task $TASK_ID"

# Define input based on task ID (example: process file_1.dat, file_2.dat, ...)
INPUT_FILE="input_data/file_${TASK_ID}.dat"
OUTPUT_DIR="results/task_${TASK_ID}"
mkdir -p "$OUTPUT_DIR"

# Execute command using the task ID
singularity exec \
    --nv \
    --bind ... \
    MyProject.sif \
    python /app/process.py --input "$INPUT_FILE" --output "$OUTPUT_DIR"

echo "Finished array task $TASK_ID"
```

*   `--array=1-100`: Creates 100 tasks with IDs 1, 2, ..., 100.
*   `%10`: Limits the number of concurrently running tasks from this array to 10.
*   `$SLURM_ARRAY_TASK_ID`: Environment variable holding the unique ID for each task instance.

## Managing Dependencies within Singularity

Dependencies are defined in the Singularity definition file (`%post` section). Common methods include:

*   **System Packages:** Use the base OS package manager (e.g., `apt-get update && apt-get install -y ...` for Debian/Ubuntu bases).
*   **Python Packages (pip):** Install pip and then use `pip install -r requirements.txt` or `pip install package1 package2`.
*   **Python Packages (Conda):** Install Miniconda in the container, create/activate a Conda environment, and install packages using `conda install`. This is often preferred for complex dependencies like RDKit.
*   **Compiling from Source:** Use standard `wget`, `tar`, `./configure`, `make`, `make install` steps.

Ensure any necessary files (like `requirements.txt`) are copied into the container using the `%files` section.

## Data, Results, and Cache Management

Containers are typically ephemeral. Use bind mounts (`--bind`) to connect container paths to persistent host paths:

*   **Data:** Place input data on the host filesystem (ideally `$SCRATCH` if large) and mount it into the container, often read-only (`:ro`).
*   **Results:** Create a results directory on the host (can be `$HOME` or `$SCRATCH`) and mount it read-write into the container where your application saves output.
*   **Caches:** Mount host cache directories (e.g., `$HOME/.cache/huggingface`, `$HOME/.cache/pip`) to the expected locations inside the container (e.g., `/root/.cache/huggingface`). This prevents re-downloading large models or packages on every run.

## File System Best Practices (`$HOME` vs. `$SCRATCH`)

Most HPC systems have different file systems with distinct characteristics:

*   **Home Directory (`$HOME`, e.g., `/haml/<user>`):**
    *   Persistent, backed up.
    *   Smaller storage quota.
    *   Slower I/O performance.
    *   **Use for:** Source code, scripts, Singularity definition files, final Singularity images (`.sif`), important configuration files, small essential results.
*   **Scratch Directory (`$SCRATCH`, e.g., `/haml_scratch/<user>`):**
    *   Not persistent, **NOT backed up**, may have purge policies (old files deleted).
    *   Larger storage quota.
    *   Faster I/O performance.
    *   **Use for:** Large input datasets, intermediate files, large output results, software installations (if building outside containers), container build caches.

**Workflow Recommendation:** Keep code/scripts in `$HOME`. Place large data in `$SCRATCH`. Run jobs from `$HOME` but configure scripts/containers to read data from `$SCRATCH` and write results to `$SCRATCH`. Copy essential final results back to `$HOME` if needed.

## Common Errors and Troubleshooting

### Slurm Errors
*   **Job Pending (`PD`) for a long time:** Cluster might be busy, or you requested resources that are hard to satisfy (e.g., many nodes, long wall time, specific hardware). Check `squeue -u $USER` for the reason.
*   **`ReqNodeNotAvail` / `Resources` / `PartitionNodeLimit`:** Resources you requested are not available or exceed partition limits.
*   **`InvalidAccount` / `InvalidQOS`:** Issues with your user account or project allocation.
*   **Job Fails Immediately:** Check `.err` file. Often due to incorrect paths, missing files, module load errors, or permission issues.
*   **Job Killed (`TIMEOUT`):** Exceeded requested wall time (`--time`). Request more time.
*   **Job Killed (`OUT_OF_MEMORY`):** Exceeded requested memory. Request more memory (`--mem-per-cpu` or `--mem`). Check `sacct` for peak usage (`MaxRSS`).

### Singularity Errors
*   **Build Error (`FATAL: ... required to build ... as non-root user`):** Add `--fakeroot` to `singularity build` command.
*   **Build Error (Package not found):** Check base image OS, repository configuration (`apt update`), package names, network connectivity during build.
*   **Runtime Error (`command not found`):** Check `$PATH` inside the container (`singularity exec <image> echo $PATH`). Ensure the command is installed and in the path.
*   **Runtime Error (File not found):** Check `--bind` mounts. Ensure host paths exist and container paths match what the application expects.
*   **GPU Issues (`CUDA error`, `no CUDA-capable device is detected`):**
    *   Did you request a GPU node (`--partition=gpu --gres=gpu:1`)?
    *   Did you use the `--nv` flag with `singularity exec/run`?
    *   Does the container's CUDA toolkit version roughly match the host driver? (`--nv` helps bridge minor differences).
    *   Is your application correctly configured to use the GPU (e.g., `tensor.to('cuda')` in PyTorch)?

## Getting Help

When encountering issues:
1.  Carefully read Slurm output/error files (`.out`, `.err`).
2.  Check application-specific log files.
3.  Verify resource requests (`#SBATCH` directives).
4.  Simplify your job script or container definition to isolate the problem.
5.  Consult the official Komondor documentation.
6.  Contact the Komondor HPC support team (hpc-support@bme.hu), providing:
    *   Your username.
    *   The Job ID(s) involved.
    *   The location of your job script and definition file.
    *   Relevant output/error messages.
    *   A clear description of the problem and what you were trying to achieve.