# Fish Brain Dynamics — CNMF Pipeline

Automated calcium imaging analysis for zebrafish brain recordings. The pipeline uses CaImAn's CNMF (Constrained Nonnegative Matrix Factorization) to detect and extract single-neuron activity traces, with Bayesian hyperparameter optimization and post-hoc quality filtering to suppress false positives.

---

## Goals

- Reliably detect active neurons across multiple zebrafish brain imaging datasets acquired with different rigs and acquisition schemes.
- Automatically tune CNMF parameters per dataset without manual grid search.
- Validate parameter generalizability across time windows, z-planes, and behavioral tasks.
- Suppress non-neuronal detections (noise blobs, out-of-brain pixels) using geometry- and signal-based quality filters.

---

## Quick Setup

CaImAn must be installed via conda-forge. The project was developed and run on Linux.

```bash
# Create and activate the caiman environment
conda create -n caiman -c conda-forge caiman
conda activate caiman

# Install Bayesian optimizer (not bundled with caiman)
pip install scikit-optimize
```

All other dependencies (h5py, scikit-image, tifffile, scipy, matplotlib, pandas) are included with the conda-forge caiman build. The `requirements.txt` in this repo is a full pip freeze of the development environment and is provided for reference only — do not use it to install from scratch.

---

## Running the Pipeline

```bash
# Time-split validation — tune on first half of frames, test on second half
python p4_universal.py --mode time-split \
    --data-dir "/path/to/data_dir" \
    --run-name my_run \
    --resolution 512 --n-calls 10

# Cross-task generalization — tune on Task 1, test on Task 3
python p4_universal.py --mode file-plane-split \
    --tune-dir "/path/to/task1" \
    --test-dir  "/path/to/task3" \
    --run-name task1_to_task3 \
    --resolution 512 --n-calls 10

# Smoke test (2 trials, fast)
python p4_universal.py --mode time-split \
    --data-dir <DIR> --run-name smoke --resolution 512 \
    --n-calls 2 --n-initial 2
```

**Monitor long runs** by piping stdout through `monitor.py`, which timestamps key CNMF events and appends them to `logs/`:

```bash
python p4_universal.py ... 2>&1 | python monitor.py --filename my_run_logs.txt
```

All outputs (plots, traces, CSVs, `summary.json`) are written to `results/<run-name>/`.

---

## Validation Modes

| Mode | Description |
|---|---|
| `time-split` | Tune on first 50% of frames, test on second 50%. Same file and z-plane. |
| `plane-split` | Tune on one z-plane, test on every other z-plane in the same recording. |
| `file-plane-split` | Tune on File A at a given z-plane, test on File B at the same z-plane. |
| `file-split` | Tune on File A, test on File B across all z-planes. |

---

## Completed Runs

| Run folder | Dataset | Mode | Resolution | Neurons (kept / raw) | Notes |
|---|---|---|---|---|---|
| `7iii25_Task4` | Task 4 | time-split | 2048×2048 | 2509 / — | Early full-res run |
| `7iii25_Task5` | Task 5 | time-split | 2048×2048 | 482 / — | |
| `7iii25_Task6` | Task 6 | time-split | 2048×2048 | 168 / — | |
| `Task4_timesplit_z3` | Task 4 z=3 | time-split | — | — | Single-plane z3 |
| `13iii26_task1_timesplit` | 13iii26 Task 1 | time-split | 512×512 | 66 / 266 (full movie) | multi-cam format, 50 frames |
| `13iii26_task1_to_task3` | 13iii26 Task 1→3 | file-plane-split | 512×512 | — | Cross-task generalization |
| `13iii26_task3_timesplit` | 13iii26 Task 3 | time-split | 512×512 | — | |
| `13iii26_task3_to_task1` | 13iii26 Task 3→1 | file-plane-split | 512×512 | — | Reverse cross-task |
| `20iv26_142407_time_split_full` | 20iv26 #142407 | time-split | full (2048×2048) | — | single-movie format |
| `20iv26_144159_time_split_full` | 20iv26 #144159 | time-split | full (2048×2048) | — | |
| `20iv26_144321_time_split_full` | 20iv26 #144321 | time-split | full (2048×2048) | 2 / 15 (test half) | 1000 frames, heavy max-area filtering |

`results/all_runs.csv` aggregates headline metrics across every run.

---

## Changes from the Original Pipeline (p3 → p4)

### Input format auto-detection
p3 assumed a fixed file layout. p4 auto-detects five formats:

| Format | Description |
|---|---|
| `multi-tp` | Many `tp-*.lux.h5` files, each `(Z, H, W)` |
| `multi-cam` | Many `Cam_long_*.lux*.h5` files, each `(1, H, W)` — 13iii26 style |
| `single-movie` | One large `Cam_long_*.lux*.h5`, shape `(T, H, W)` — 20iv26 style |
| `interleaved` | One `*.lux*.h5` with z-planes packed into the T axis; `n_planes` read from HDF5 metadata |
| `legacy` | Any `*.h5` with a `Data` key |

Format can also be overridden via `--format`.

### Brain mask preprocessing
An Otsu-threshold brain mask is computed from the mean image, cleaned with morphological opening/closing, and applied before CNMF to zero out dark periphery pixels. This prevents CNMF from initialising components in regions the biology team flagged as outside the imaging plane. Disabled with `--no-mask`.

### Quality filters inside Bayesian tuning
p3 ran quality filters only post-hoc. p4 applies them **inside every Bayesian trial**: the composite score that the optimizer maximises is computed on the **filtered** neuron count, not the raw count. This means the optimizer is rewarded for finding real neurons rather than accumulating noise blobs.

Filters applied per component:
- **Circularity** `4π·area / perimeter²` ≥ `--min-circularity` (default 0.5)
- **Max area** ≤ `--max-area-factor × π × gSig²` (default 4×)
- **In-mask** centroid must fall inside the brain mask

### Composite scoring
The Bayesian objective balances:
```
score = 1.0 × (1 − recon_error)
      + 0.5 × spatial_compactness
      − 0.3 × log(1 + trace_sparsity)
      + 1.0 × stability          # cross-half footprint overlap
      + 0.001 × log(1 + n_kept)  # small bonus for real neuron count
```

### Configurable resolution
`--resolution {full, 1024, 512}` — search space bounds for `gSig`, `rf`, and motion-correction parameters scale automatically with resolution.

### CPU pinning
`--pin-cpus 0-31` binds the process to specific cores via `os.sched_setaffinity` (Linux only), useful on shared HPC nodes. `--n-workers N` sets the CNMF worker count explicitly. When neither flag is provided, workers default to `os.cpu_count() - 1`.

### Stripe removal
Per-column temporal median subtraction removes light-sheet illumination stripes before CNMF. Disabled with `--no-stripe`.

### Monitoring script
`monitor.py` is a stdin-pipe logger that timestamps CNMF lifecycle events (`fit_file starting`, `fit_file done`, time-split boundaries) and appends them to `logs/` without blocking the main run.
