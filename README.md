# Zebrafish Brain CNMF Pipeline

A robust calcium imaging analysis pipeline built on **CaImAn** for automated neuron detection and fluorescence trace extraction from zebrafish whole-brain imaging datasets.

This repository provides:

- Automated preprocessing of HDF5 imaging movies
- Bayesian optimization of CNMF hyperparameters
- Universal data loading for multiple microscope formats
- Motion correction using NoRMCorre
- Automatic brain masking
- CNMF-based neuron extraction
- Diagnostic visualization and trace export


## Features

### Universal Dataset Support

The pipeline automatically detects and loads several imaging formats, including:

- Single HDF5 movies
- Multi-timepoint acquisitions
- Multi-camera recordings
- Interleaved multi-plane recordings
- Legacy datasets

---

### Preprocessing

The preprocessing pipeline includes:

- Spatial downsampling (Full / 1024 / 512 resolution)
- Stripe artifact removal
- Automatic brain masking using Otsu thresholding
- Optional motion correction (NoRMCorre)

---

### Bayesian Parameter Optimization

`calibrate.py` performs Bayesian optimization of CNMF parameters including:

- gSig
- rf
- overlap fraction
- occupancy fraction
- background rank (nb)
- decay time

Instead of optimizing stride directly, stride and K are derived from the optimized parameters, resulting in a more physically meaningful search space.

Each candidate is evaluated using a composite objective that considers:

- Extraction quality
- Reconstruction fidelity
- Spatial/temporal redundancy
- Patch seam artifacts
- Background leakage
- Calcium decay consistency
- Neuron yield

---

### CNMF Pipeline

After calibration, the optimized parameters can be used by the production pipeline to:

- Load imaging data
- Apply preprocessing
- Perform CNMF
- Refit detected neurons
- Evaluate components
- Export traces
- Save visualization figures


## Repository Structure

```
.
├── calibrate.py          # Bayesian parameter calibration
├── new_pipeline.py       # Main CNMF pipeline
├── monitor.py            # Runtime log monitor
├── test.py
├── logs/
└── results/
```


## Installation

Clone the repository:

```bash
git clone https://github.com/soluchi07/zebrafish_brain_cnmf.git
cd zebrafish_brain_cnmf
```

Create a Python environment:

```bash
conda create -n zebrafish caiman python=3.11
conda activate zebrafish
```

Major dependencies include:

- CaImAn
- NumPy
- SciPy
- scikit-image
- scikit-optimize
- OpenCV
- h5py
- pandas
- matplotlib
- tifffile

These dependencies are bundled with CaImAn.


## Calibrating Parameters

Run Bayesian optimization:

```bash
python calibrate.py \
    --run-name calibration_run \
    --data-dir /path/to/data \
    --resolution 512 \
    --n-calls 20
```

Example options:

| Option | Description | Default Value |
|---------|-------------|-----------|
| `--mask` | Enable automatic brain masking | Mask OFF |
| `--no-mc` | Disable motion correction | Motion Correction ON |
| `--no-stripe` | Disable stripe removal | Stripe-removal ON |
| `--resolution` | full, 1024, or 512 | 512 |
| `--max-frames` | Limit frames loaded | None |
| `--n-workers` | Number of CPU workers | None |


## Running the Pipeline

Once parameters have been calibrated:

```bash
python new_pipeline.py \
    --run-name experiment_01 \
    --data-dir /path/to/data \
    --best-params-path results/calibration_run/calibration_summary.json
```

Additional options include:

```text
--mode
--mask
--resolution
--z-index
--n-planes
--no-mc
--save-traces
```


## Outputs

Each `calibrate.py` run creates a directory under `results/` containing:

```
results/
└── experiment_name/
    ├── pipeline_results.hdf5
    ├── neuron_traces.npy
    ├── neuron_traces.csv
    ├── optimized_contours.png
    ├── optimized_traces.png
    ├── calibration_summary.json
    ├── tune_calib_log.csv
    └── convergence_calib.png
```


## Logging

`monitor.py` can be used to capture pipeline output and monitor long-running jobs.

Example:

```bash
python new_pipeline.py ... | python monitor.py --filename run_log
```

Logs are written to the `logs/` directory.


## Typical Workflow

```text
Raw HDF5 Movies
        │
        ▼
Preprocessing
        │
        ▼
Motion Correction
        │
        ▼
Bayesian Calibration
        │
        ▼
Optimal CNMF Parameters
        │
        ▼
CNMF Extraction
        │
        ▼
Refinement & Component Evaluation
        │
        ▼
Neuron Traces + Spatial Footprints
```


## Project Goals

This project aims to provide a reproducible, automated workflow for calcium imaging analysis that:

- minimizes manual parameter tuning,
- supports diverse imaging formats,
- improves neuron detection quality through Bayesian optimization, and
- generates publication-ready outputs for downstream neuroscience analysis.


## License

This project is licensed under the **BSD 3-Clause License**.

> **Note on Dependencies:** This software is released under the BSD 3-Clause License. However, please note that core dependencies such as **CaImAn** are distributed under the GNU General Public License v3.0 (GPLv3). Users combining or redistributing this pipeline alongside GPL-licensed dependencies should ensure compliance with their respective licensing terms.


## Acknowledgements

This project builds upon the excellent work of the **CaImAn** developers for calcium imaging analysis and the **NoRMCorre** motion correction framework and my colleague [@Mikito-Coder](https://github.com/Mikito-Coder).
