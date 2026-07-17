# Zebrafish Calcium Imaging Pipeline & Parameter Calibration Engine

An automated, robust pipeline for whole-brain and multi-plane calcium imaging analysis in larval zebrafish (*Danio rerio*), powered by **CaImAn** and custom spatial-temporal preprocessing routines. 

This repository provides an end-to-end framework to process calcium imaging datasets, perform automated hyper-parameter tuning, enforce strict spatial/signal quality control, and validate parameter transferability across experimental conditions.

## Project Overview & Key Objectives

In large-scale larval zebrafish calcium imaging, variance in optical setups (light-sheet/SPIM, two-photon, spinning disk), acquisition framerates, signal-to-noise ratios (SNR), and optical artifacts makes standard fixed-parameter CNMF (Constrained Non-negative Matrix Factorization) fragile and prone to false detections. 

This project addresses these challenges through four core objectives:

1. **Reliable Cross-Rig Neuronal Detection**: Detect active neurons consistently across multiple zebrafish brain imaging datasets acquired with different optical rigs, magnification factors, and acquisition schemes.
2. **Automated Parameter Tuning**: Automatically optimize CNMF parameters per dataset on representative calibration windows, eliminating tedious manual grid searches and subjective parameter tweaking.
3. **Validation of Parameter Generalizability**: Test and validate parameter robustness and transferability across different recording time windows, optical z-planes, and behavioral tasks.
4. **Noise & Non-Neuronal Artifact Suppression**: Suppress out-of-brain pixels, excitation light stripe artifacts, and non-neuronal spatial noise blobs using geometry- and signal-based quality filters.

## System Architecture & Workflow

The pipeline is organized into modular processing stages, accessible via main scripts: `calibrate_cnmf.py` (for automated calibration) and `new_pipeline.py` (for production execution).

```
                    Raw TIFF / Movie
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 1. Preprocessing & Artifact Mitigation                 │
│    ├─ Spatial Downsampling (downsample)                │
│    ├─ Illumination Stripe Removal (stripe_remove)      │
│    └─ Brain Mask Generation & Cropping (apply_mask)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Automated Parameter Tuning (calibrate_cnmf.py)       │ 
│    ├─ Calculate PNR & Spatial Correlation Maps          │
│    ├─ Estimate Noise Floor & Expected Radius (gSig)     │
│    └─ Export Optimal Hyper-parameters (best_params.json)│
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 3. Rigid / Non-Rigid Motion Correction (McMap)         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 4. CNMF Factorization & Trace Extraction               │
│    ├─ Spatial Initialization (GreedyROI/Greedymult)    │
│    └─ Temporal Deconvolution & AR Modeling (p=1/p=2)   │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ 5. Quality Control & Component Filtering               │
│    ├─ Spatial Geometry & Compactness Filters           │
│    ├─ SNR & Trace Correlation Cutoffs (rval_thr)       │
│    └─ Out-of-Brain Spatial Masking Validation          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
Structured Results (HDF5 / NPY / Diagnostic PNG Plots)
```

## Core Components & Features

### 1. Preprocessing (`preprocess_movie`)
- **Stripe Removal**: Eliminates characteristic illumination lines and laser scanning artifacts from light-sheet imaging using FFT and row/column spatial median filtering.
- **Anatomical Brain Masking**: Uses Otsu thresholding, Gaussian smoothing, and morphological contour detection to restrict signal extraction strictly to brain tissue, eliminating background noise.
- **Memory Optimization**: Memory-maps raw datasets to allow smooth processing of multi-gigabyte files under restricted RAM footprints.

### 2. Automated Calibration Engine (`calibrate_cnmf.py`)
- Automatically evaluates Peak-to-Noise Ratio (PNR) and spatial correlation across calibration frames (`preprocess_calib`).
- Infers dataset-specific parameters (such as `gSig`, `min_corr`, `min_pnr`, `decay_time`, and frame rates `fr`).
- Pre-computes motion correction maps (`precompute_mc`) to prevent dual mmap conflict errors during re-fitting routines.

### 3. Dual-Stage Quality Filtering
- **Geometry-Based Filters**: Rejects components based on non-somatic aspect ratios, irregular spatial extent, or spatial overlap with out-of-brain masks.
- **Signal-Based Filters**: Evaluates spatial footprint correlation (`rval_thr`), transient baseline stability, and signal-to-noise ratio (`min_SNR`).


## Installation & Environment Setup

### Prerequisites
- Python 3.9 or 3.10
- Conda / Mamba environment
- CUDA / OpenMP enabled for multi-core CaImAn processing

### Step-by-Step Setup
```bash
# 1. Clone the repository
git clone [https://github.com/your-org/zebrafish-caiman-pipeline.git](https://github.com/your-org/zebrafish-caiman-pipeline.git)
cd zebrafish-caiman-pipeline

# 2. Create and activate conda environment
conda create -n fish_caiman python=3.10 -y
conda activate fish_caiman

# 3. Install CaImAn dependencies
conda install -c conda-forge caiman

# 4. Install additional Python requirements
pip install numpy scipy matplotlib scikit-image h5py openpyxl

```

## Usage

### Step 1: Automated Calibration (`calibrate_cnmf.py`)

Run parameter calibration on a representative dataset slice or trial to automatically generate optimal parameter settings:

```bash
python calibrate_cnmf.py \
    --input_file /data/zebrafish_plane01.tif \
    --output_dir ./calibration_output \
    --save_plots

```

*Outputs:* `best_params.json`, `preprocess_calib.png`, `brain_mask_calib.png`.

### Step 2: Production Execution (`new_pipeline.py`)

Execute the production pipeline using pre-tuned parameters across complete experiments:

```bash
python orig_pipeline_refactored.py \
    --input_file /data/zebrafish_full_session.tif \
    --params_file best_params.json \
    --output_dir ./production_output

```

*Outputs:* `preprocess_movie.png`, `brain_mask_movie.png`, extracted traces, and HDF5 component files.


**Monitor long runs** by piping stdout through `monitor.py`, which timestamps key CNMF events and appends them to `logs/`:

```bash
python p4_universal.py ... 2>&1 | python monitor.py --filename my_run_logs.txt
```

All outputs (plots, traces, CSVs, `summary.json`) are written to `results/<run-name>/`.


`results/all_runs.csv` aggregates headline metrics across every run.


## Configuration Parameters

Primary parameters are defined in `BASE_PARAMS` and overridden via `best_params.json`:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `fr` | float | `30` / `5` | Acquisition frame rate (Hz) |
| `decay_time` | float | `0.75` | Calcium indicator decay half-life (seconds) |
| `gSig` | tuple | `(3, 3)` | Expected half-size of target neurons (pixels) |
| `min_corr` | float | `0.85` | Minimum spatial correlation threshold |
| `min_pnr` | float | `10.0` | Minimum Peak-to-Noise Ratio cutoff |
| `p` | int | `1` / `2` | Order of autoregressive AR model |
| `rval_thr` | float | `0.85` | Spatial profile correlation threshold |
| `border_nan` | str | `"copy"` | Boundary handling mode for motion correction |

## Validation Modes

| Mode | Description |
|---|---|
| `time-split` | Tune on first 50% of frames, test on second 50%. Same file and z-plane. |
| `plane-split` | Tune on one z-plane, test on every other z-plane in the same recording. |
| `file-plane-split` | Tune on File A at a given z-plane, test on File B at the same z-plane. |
| `file-split` | Tune on File A, test on File B across all z-planes. |

### Input format auto-detection
p3 assumed a fixed file layout. p4 auto-detects five formats:

| Format | Description |
|---|---|
| `multi-tp` | Many `tp-*.lux.h5` files, each `(Z, H, W)` |
| `multi-cam` | Many `Cam_long_*.lux*.h5` files, each `(1, H, W)` — 13iii26 style |
| `single-movie` | One large `Cam_long_*.lux*.h5`, shape `(T, H, W)` — 20iv26 style |
| `interleaved` | One `*.lux*.h5` with z-planes packed into the T axis; `n_planes` read from HDF5 metadata |
| `legacy` | Any `*.h5` with a `Data` key |

### Configurable resolution
`--resolution {full, 1024, 512}` — search space bounds for `gSig`, `rf`, and motion-correction parameters scale automatically with resolution.

### Stripe removal
Per-column temporal median subtraction removes light-sheet illumination stripes before CNMF. Disabled with `--no-stripe`.

### Monitoring script
`monitor.py` is a stdin-pipe logger that timestamps CNMF lifecycle events (`fit_file starting`, `fit_file done`, time-split boundaries) and appends them to `logs/` without blocking the main run.

## License & Acknowledgments

This project builds upon the open-source **CaImAn** framework (Flatiron Institute) adapted specifically for whole-brain larval zebrafish dynamics. Distributed under the MIT License.
