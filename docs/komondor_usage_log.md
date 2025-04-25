# Komondor HPC Usage Log - KIBA Project (April 2025)

This document logs the steps and issues encountered while setting up and running the KIBA drug-target interaction prediction project on the Komondor HPC cluster.

**Goal:** Run a Python-based machine learning workflow (using PyTorch, Transformers, RDKit) on GPU nodes using Slurm and Singularity for environment management.

**Initial Setup:**
*   Project code (`src/main.py`)
*   Dependencies (`requirements.txt`)
*   Target environment: Komondor GPU partition.

**Approach:**
1.  Containerize the application environment using Singularity (`KIBA.def`).
2.  Create a Slurm batch script (`run_main_singularity.sbatch`) to:
    *   Request GPU resources.
    *   Build the Singularity image if it doesn't exist.
    *   Run `main.py` inside the container with appropriate data/cache mounts.

**Timeline & Issues:**

1.  **File Transfer (Local -> Komondor):**
    *   **Issue:** Initial `rsync` failed due to incorrect target path (`/nr_haml2025`).
    *   **Resolution:** Verified correct home directory path (`pwd` on Komondor) and used it in `rsync`.
    *   **Issue:** Transfer very slow due to inclusion of local `.venv` directory.
    *   **Resolution:** Excluded `.venv` from `rsync` (`--exclude='.venv'`). Transfer successful.
    *   **Action:** Placed project in `$HOME`, created `$HOME/KIBA-BME2025/data` and placed `KIBA.csv` there.

2.  **First Slurm Job Submission (`sbatch run_main_singularity.sbatch`):**
    *   **Issue:** Job failed during `singularity build` step.
    *   **Error Log:** `FATAL: --remote, --fakeroot, or the proot command are required to build this source as a non-root user`.
    *   **Resolution:** Added `--fakeroot` flag to the `singularity build` command in the Slurm script.

3.  **Second Slurm Job Submission (with `--fakeroot`):**
    *   **Issue:** Job failed during `singularity build` step (while `conda install rdkit`).
    *   **Error Log:** `Killed`, `FATAL: While performing build: while running engine: exit status 137`, `slurmstepd: error: Detected 1 oom-kill event(s) ...`.
    *   **Diagnosis:** Out-Of-Memory error. Initial resource request was 4 CPUs, 4000M/CPU (~16GB total).
    *   **Resolution:** Increased requested resources in Slurm script to 8 CPUs, 4000M/CPU (~32GB total) (`--cpus-per-task=8`).

4.  **Third Slurm Job Submission (with `--fakeroot`, 32GB memory):**
    *   **Issue:** Job failed during the final stage of `singularity build`.
    *   **Error Log:** `FATAL: While performing build: while creating SIF: while creating container: open /project/nr_haml2025/KIBA-BME2025/KIBA.sif: permission denied`.
    *   **Diagnosis:** Although user has write permission in the target directory (`/home/nr_hafm/nr_haml2025/KIBA-BME2025`), Singularity (possibly due to `--fakeroot` or Slurm environment interaction) attempted to write the final image to an incorrect path (`/project/...`). Also noted `rdkit-pypi` was still being installed via `requirements.txt`.
    *   **Resolution:** 
        1. Manually removed `rdkit-pypi` from `requirements.txt` (as it's not needed by `main.py`).
        2. Modified `singularity build` command in Slurm script to explicitly specify the full output path (`singularity build --fakeroot "$IMAGE_PATH" ...`).

5.  **Fourth Slurm Job Submission (no rdkit, explicit build path):**
    *   **Issue:** Job failed again during the final stage of `singularity build`.
    *   **Error Log:** `FATAL: While performing build: while creating SIF: while creating container: open /project/nr_haml2025/KIBA-BME2025/KIBA.sif: permission denied`.
    *   **Diagnosis:** Discovered `$SLURM_SUBMIT_DIR` resolves to `/project/nr_haml2025/...` on compute nodes, not the user's home directory (`/home/nr_hafm/nr_haml2025/...`). The script was trying to write the image to this non-writable `/project/` path.
    *   **Resolution:** Modified `run_main_singularity.sbatch` to define `IMAGE_PATH` using the user's explicit, known-writable home directory path, instead of `$SLURM_SUBMIT_DIR`.

6.  **Fifth Slurm Job Submission (explicit home dir build path):**
    *   **Issue:** Job failed again during `singularity build`.
    *   **Error Log:** `FATAL: Unable to build from /project/nr_haml2025/KIBA-BME2025/KIBA.def: ... permission denied`.
    *   **Diagnosis:** The script was still trying to read the input `KIBA.def` file using the `$SLURM_SUBMIT_DIR` path (`/project/...`), which is inaccessible from the compute node.
    *   **Resolution:** Modified `run_main_singularity.sbatch` again to define `DEFINITION_FILE` using the explicit home directory path (`$PROJECT_DIR`) and removed the `cd` command before the build.

7.  **Sixth Slurm Job Submission (explicit home dir for build input & output):**
    *   **Issue:** Job failed again during `singularity build`.
    *   **Error Log:** `FATAL: Unable to build from /home/nr_hafm/nr_haml2025/KIBA-BME2025/KIBA.def: ... permission denied`.
    *   **Diagnosis:** Process running under Slurm/fakeroot on compute node cannot read the definition file from the user's home directory, despite correct path and expected permissions.
    *   **Resolution:** Added `--fix-perms` flag to the `singularity build` command, as seen in documentation examples, hoping it influences build environment permissions.

8.  **Seventh Slurm Job Submission (with --fix-perms):**
    *   **Status:** Ready to submit (as of 2025-04-25).
    *   **Expectation:** Uncertain. If `--fix-perms` resolves the access issue, build may succeed. If not, filesystem access from compute nodes needs further investigation.
