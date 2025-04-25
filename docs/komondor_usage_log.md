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
    *   **Issue:** Job failed again during `singularity build`.
    *   **Error Log:** `FATAL: Unable to build from /home/nr_hafm/nr_haml2025/KIBA-BME2025/KIBA.def: ... permission denied`.
    *   **Diagnosis:** Process running under Slurm/fakeroot on compute node cannot read the definition file from the user's home directory. Previous errors suggest it *can* read from `$SLURM_SUBMIT_DIR` (/project/...) but cannot write there, and cannot read from `/home/...`.
    *   **Resolution:** Modified `run_main_singularity.sbatch` to use `$SLURM_SUBMIT_DIR` for the input `DEFINITION_FILE` path (readable) and the explicit home directory path for the output `IMAGE_PATH` (writable).

9.  **Eighth Slurm Job Submission (Mixed Paths: Input from /project, Output to /home):**
    *   **Issue:** Job failed again during `singularity build`.
    *   **Error Log:** `FATAL: Unable to build from /project/nr_hafm/nr_haml2025/KIBA-BME2025/KIBA.def: ... permission denied`.
    *   **Diagnosis:** Confirmed compute node cannot read from `/project/...` path either. Also tested manual build on login node (`singularity build --fakeroot /home/.../KIBA.sif /home/.../KIBA.def`) which also failed with `FATAL: Unable to build from /home/.../KIBA.def: ... permission denied`.
    *   **Action:** Checked directory permissions (`ls -ld`) and added `chmod o+x /home/nr_hafm/`.
    *   **Result:** Manual build still failed with the same permission denied error reading `/home/.../KIBA.def`.

10. **Impasse - Permission Denied Reading/Writing During Build:**
    *   **Status:** Unable to build Singularity image either via Slurm or manually on login node due to persistent "permission denied" errors.
    *   **Details:**
        *   Build fails writing to `/project/...` even when run manually.
        *   Build fails writing to `/scratch/...` even when run manually.
        *   Build fails reading from `/home/...` when paths are set explicitly.
        *   Standard `touch` command confirms user write access to `/project/...`.
        *   `chmod o+x /home/nr_hafm/` did not resolve read errors from home.
    *   **Dummy File Test:**
        *   `dummy.def` created for minimal testing.
        *   Build output to `/home/...`: Failed at the *end* (write error).
        *   Build output to `/project/...`: Failed at the *start* (read error from `/home/...`).
        *   Build output to `/scratch/...`: Failed at the *start* (read error from `/home/...`).
    *   **Successful Build Location Found:**
        *   Manual build of `dummy.sif` succeeded *only* when run from and outputting to `/home/nr_hafm`.
        *   Manual build of `KIBA.sif` succeeded *only* after copying source files (`KIBA.def`, `src/`, `requirements.txt`) to `/home/nr_hafm` and running the build there, outputting `/home/nr_hafm/KIBA.sif`.
    *   **Diagnosis:** `singularity build --fakeroot` on login node cannot access subdirectories (`/home/nr_hafm/nr_haml2025/...`, `/project/...`, `/scratch/...`) for reading inputs or writing outputs, but *can* operate within the base `/home/nr_hafm` directory.
    *   **Resolution:** Build image manually in `/home/nr_hafm`. Update Slurm script to use `/home/nr_hafm/KIBA.sif` while still mounting data/results/cache from `/project` and `/scratch`.

11. **Ninth Slurm Job Submission (Using Image from /home/nr_hafm):**
    *   **Status:** Ready to submit (as of 2025-04-25).
    *   **Expectation:** Job should now run successfully, using the pre-built image.

12. **Tenth Slurm Job Submission (After Rebuilding Image with Updated kagglehub):**
    *   **Issue:** Job failed during Python execution inside the container.
    *   **Error Log:** `kagglehub.exceptions.KaggleApiHTTPError: 404 Client Error. Dataset not found` for `blk1804/kiba-drug-binding-dataset`.
    *   **Diagnosis:** The script was hardcoded to download the dataset via `kagglehub`, which failed because the dataset ID was incorrect/missing. It wasn't designed to load the manually placed `KIBA.csv`.
    *   **Resolution:** Modified `src/main.py`'s `load_kiba_data` function to directly load `/data/KIBA.csv` using `pandas.read_csv`, removing the `kagglehub` dependency for this step. Instructed user to manually rebuild the image with the updated script.

13. **Eleventh Slurm Job Submission (After Rebuilding Image with pd.read_csv):**
    *   **Status:** Ready to submit after manual rebuild (as of 2025-04-25).
    *   **Expectation:** Job should successfully load data from the mounted `/data/KIBA.csv` file.
