#!/usr/bin/env python3
"""
p4_universal.py  —  Phase 4: Universal CNMF Pipeline (fixed-parameter, refit-validated)

Major upgrades over p3:
  * Auto-detects 5 input formats:
      multi-tp     : many tp-*.lux.h5 files, each (Z, H, W)
      multi-cam    : many Cam_long_*.lux.h5 files, each (1, H, W)   (13iii26 style)
      single-movie : one big Cam_long_*.lux.h5 file, shape (T, H, W) (20iv26 style)
      interleaved  : one *.lux*.h5 file, shape (T*Z, H, W); planes
                     stride the time axis (e.g. 7-plane → data[z::7]).
                     n_planes auto-detected from metadata["stack"]["n"].
      legacy       : one .h5 file with shape (T, H, W)
  * CNMF hyperparameters are fixed, not searched: `best_params.json` (produced
    by the separate calibrate_cnmf.py calibration script) is loaded and merged
    on top of the resolution-aware defaults in BASE_PARAMS. There is no
    Bayesian tuning step in this script anymore — every mode just applies
    BASE_PARAMS to each split it processes.
  * No brain-mask preprocessing — the full frame is used as-is.
  * No post-hoc geometric quality filtering (circularity / max-area / in-mask).
    Instead, every CNMF run gets a second validation pass via
    cnmf_model.refit(), on top of CaImAn's own evaluate_components /
    select_components, before the run is scored.
  * Configurable resolution: --resolution {full, 1024, 512}
  * 4 validation modes (time-split, plane-split, file-plane-split, file-split)
    with graceful skipping when data doesn't support a mode (e.g. Z=1). These
    still hold out a reference split to compute footprint stability, but no
    longer search for parameters — they use whatever is in BASE_PARAMS.

Usage examples:

  source ~/Documents/fishBrain_Kiitan/Fish_Brain_Dynamics/venv/bin/activate

  # 13iii26 task1, time-split, 512x512 (fast)
  python p4_universal.py --mode time-split \\
      --data-dir "/path/to/13iii26/Xgcamp_..._task1stack0_..._channel_2_obj_bottom" \\
      --run-name 13iii26_task1_timesplit --resolution 512

  # Cross-task generalization (file-plane-split reference Task1 -> test Task3)
  python p4_universal.py --mode file-plane-split \\
      --tune-dir "/path/to/...task1..." \\
      --test-dir "/path/to/...task3..." \\
      --run-name 13iii26_task1_to_task3 --resolution 512

  # Force a specific format
  python p4_universal.py --mode time-split --data-dir <DIR> \\
      --run-name forced --format multi-cam --resolution 512

  # Use a best_params.json that isn't at the script's root directory
  python p4_universal.py --mode time-split --data-dir <DIR> \\
      --run-name custom_params --best-params-path /path/to/best_params.json
"""

from __future__ import annotations

import os
import cv2
import argparse
import glob
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"

try:
    cv2.setNumThreads(0)
except:
    pass



# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal CNMF pipeline using fixed, pre-calibrated parameters, "
                     "with refit-based validation and 4 cross-validation modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=["time-split", "plane-split", "file-plane-split", "file-split"])
    p.add_argument("--run-name", required=True, help="Output folder name under results/")

    # Data sources
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Folder for time-split, plane-split")
    p.add_argument("--tune-dir", type=Path, default=None,
                   help="Reference folder for file-plane-split, file-split")
    p.add_argument("--test-dir", type=Path, default=None,
                   help="Test folder for file-plane-split, file-split")

    # Z-plane selection
    p.add_argument("--z-index", type=int, default=None,
                   help="Z-plane index for time-split, file-plane-split (default: middle)")
    p.add_argument("--tune-z", type=int, default=None,
                   help="Reference Z-plane for plane-split (default: middle)")
    p.add_argument("--n-planes", type=int, default=None,
                   help="Number of Z-planes interleaved in a single-movie file "
                        "(e.g. 7 when 700-rep x 7-plane = 4900 total frames). "
                        "Strides the T axis: keeps frames z_index, z_index+n_planes, ... "
                        "Pair with --z-index to pick a specific plane (default: middle).")

    # Resolution / preprocessing
    p.add_argument("--resolution", choices=["full", "1024", "512"], default="512",
                   help="Spatial resolution (default 512)")
    p.add_argument("--no-stripe", action="store_true",
                   help="Disable column-median stripe removal")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap loaded frames at N (useful for huge single-movie files)")

    # Format override
    p.add_argument("--format", dest="format_override",
                   choices=["multi-tp", "multi-cam", "single-movie", "interleaved", "legacy"],
                   default=None, help="Override auto-detect format")

    # Fixed CNMF parameters
    p.add_argument("--best-params-path", type=Path, default=None,
                   help="Path to best_params.json produced by calibrate_cnmf.py "
                        "(default: best_params.json in the script's root directory)")
    p.add_argument("--n-workers", type=int, default=None,
                   help="CPU workers for CNMF patch processing (default: --pin-cpus count, else cpu_count - 1)")
    p.add_argument("--pin-cpus", type=str, default=None,
                   help="CPU cores to pin to, e.g. '0-31' or '0-15,32-47' (Linux only)")

    # Quality filter thresholds
    p.add_argument("--min-snr-trace", type=float, default=1.5,
                   help="Reject components with trace SNR below this (used by CaImAn's "
                        "own evaluate_components/select_components, and by refit)")

    # dF/F
    p.add_argument("--no-bleach-correct", action="store_true",
                   help="Skip double-exponential photobleaching correction (default: ON for T>=100)")
    p.add_argument("--dff-percentile", type=float, default=8.0,
                   help="Percentile used for F0 baseline in dF/F (default: 8)")
    # Temp files
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep intermediate .tif files in _work/ after memmap creation (default: clean up)")

    return p.parse_args()


def parse_cpu_spec(spec: str) -> set:
    """Parse '0-31', '0,2,4', or '0-15,32-47' into a set of core indices."""
    cores: set = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            cores.update(range(int(lo), int(hi) + 1))
        else:
            cores.add(int(part))
    return cores


ARGS = parse_args()

# Validate args per mode
if ARGS.mode in ("time-split", "plane-split"):
    if ARGS.data_dir is None:
        print(f"ERROR: --data-dir required for mode {ARGS.mode}", file=sys.stderr)
        sys.exit(1)
elif ARGS.mode in ("file-plane-split", "file-split"):
    if ARGS.tune_dir is None or ARGS.test_dir is None:
        print(f"ERROR: --tune-dir and --test-dir required for mode {ARGS.mode}", file=sys.stderr)
        sys.exit(1)

_pinned_cores: set = set()
if ARGS.pin_cpus:
    _pinned_cores = parse_cpu_spec(ARGS.pin_cpus)
    try:
        os.sched_setaffinity(0, _pinned_cores)
    except AttributeError:
        print("WARNING: os.sched_setaffinity not available on this OS — --pin-cpus ignored.")
        _pinned_cores = set()
    except PermissionError:
        print("WARNING: Permission denied for sched_setaffinity — --pin-cpus ignored.")
        _pinned_cores = set()

if ARGS.n_workers is not None:
    N_WORKERS = ARGS.n_workers
elif _pinned_cores:
    N_WORKERS = len(_pinned_cores)
else:
    N_WORKERS = os.cpu_count() - 1

OUTPUT_DIR = RESULTS_ROOT / ARGS.run_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = OUTPUT_DIR / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMPORTS
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import tifffile
import skimage.transform
from skimage.morphology import convex_hull_image
from scipy.optimize import linear_sum_assignment, curve_fit

import caiman as cm
import caiman.mmapping
import caiman.base.movies
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from caiman.source_extraction.cnmf import params as params_module
from caiman.motion_correction import MotionCorrect

if not hasattr(cm, "load"):
    cm.load = caiman.base.movies.load
if not hasattr(cm, "movie"):
    cm.movie = caiman.base.movies.movie
if not hasattr(cm, "paths"):
    import caiman.paths


# =============================================================================
# FIXED CNMF PARAMETERS (loaded from best_params.json)
# =============================================================================

def load_best_params(path: Path) -> dict:
    """
    Load calibrated CNMF params produced by calibrate_cnmf.py.

    Accepts either:
      - a flat dict of CNMF params, e.g. {"gSig": 3, "gSig_filt": 2, ...}
      - a full calibration_summary.json with the params nested under
        the "best_params" key.
    Returns {} (falling back to BASE_PARAMS defaults only) if the file is
    missing or unreadable.
    """
    if not path.is_file():
        print(f"WARNING: best-params file not found at {path} — using base CNMF defaults only.")
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
    except Exception as exc:
        print(f"WARNING: failed to read best-params file at {path} ({exc}) — using base CNMF defaults only.")
        return {}

    best = raw.get("best_params", raw) if isinstance(raw, dict) else {}
    print(f"Loaded best params from {path}:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return best


BEST_PARAMS_PATH = ARGS.best_params_path or (SCRIPT_DIR / "best_params.json")


print("=" * 70)
print(f"p4_universal.py  |  mode={ARGS.mode}  |  run={ARGS.run_name}")
print("=" * 70)
print(f"Output dir : {OUTPUT_DIR}")
print(f"Resolution : {ARGS.resolution}")
print(f"Best params: {BEST_PARAMS_PATH}")
_cpu_pin_str = f"{ARGS.pin_cpus}  ({len(_pinned_cores)} cores)" if _pinned_cores else "unpinned"
print(f"CPU pin    : {_cpu_pin_str}  workers={N_WORKERS}")
if ARGS.n_planes:
    _default_z = ARGS.z_index if ARGS.z_index is not None else ARGS.n_planes // 2
    print(f"Z-planes   : {ARGS.n_planes} interleaved  (extracting z={_default_z})")


# =============================================================================
# FORMAT DETECTION
# =============================================================================

def read_n_planes(filepath: str) -> int:
    """Read n_planes from metadata["metaData"]["stack"]["n"] in a .lux*.h5 file; fall back to 1."""
    try:
        with h5py.File(filepath, "r") as fh:
            if "metadata" not in fh:
                return 1
            raw = fh["metadata"][()]
            meta = json.loads(raw)
            return int(meta["metaData"]["stack"]["n"])
    except Exception as e:
        print(f"  WARNING: read_n_planes failed ({e}); defaulting to 1 (treating as single-plane)")
        return 1


def detect_format(folder: Path) -> tuple[str, list[str], tuple]:
    """
    Return (format_name, file_list, sample_shape).

    multi-tp     : many tp-*.lux.h5 files, each (Z, H, W) with Z>1
    multi-cam    : many Cam_long_*.lux*.h5 files, each (1, H, W)
    single-movie : one Cam_long_*.lux*.h5 file with 1 plane (T, H, W)
    interleaved  : one *.lux*.h5 file with Z>1 planes packed into T axis
    legacy       : one .h5 file with (T, H, W)
    """
    folder = Path(folder)

    tp_files = sorted(
        glob.glob(str(folder / "tp-*_ch-*_st-*_obj-*_cam-*.lux.h5")),
        key=lambda p: int(re.search(r"tp-0-(\d+)", p).group(1)),
    )
    if tp_files:
        with h5py.File(tp_files[0], "r") as fh:
            shape = tuple(fh["Data"].shape)
        return "multi-tp", tp_files, shape

    # Broaden glob to catch both *.lux.h5 and *.lux-NNN.h5 naming
    cam_files = sorted(
        glob.glob(str(folder / "Cam_long_*.lux*.h5")),
        key=lambda p: int(re.search(r"Cam_long_(\d+)", p).group(1)),
    )
    if cam_files:
        with h5py.File(cam_files[0], "r") as fh:
            shape = tuple(fh["Data"].shape)
        if len(cam_files) > 1:
            if shape[0] == 1:
                return "multi-cam", cam_files, shape
            return "single-movie", cam_files, shape
        # Single file — check metadata for plane count
        n_z = read_n_planes(cam_files[0])
        if n_z > 1:
            return "interleaved", cam_files, shape
        return "single-movie", cam_files, shape

    # Catch-all for any other *.lux*.h5 files (non-Cam_long naming)
    lux_files = sorted(glob.glob(str(folder / "*.lux*.h5")))
    if lux_files:
        with h5py.File(lux_files[0], "r") as fh:
            if "Data" in fh:
                shape = tuple(fh["Data"].shape)
                n_z = read_n_planes(lux_files[0])
                if n_z > 1:
                    return "interleaved", lux_files, shape
                return "single-movie", lux_files, shape

    h5_files = sorted(glob.glob(str(folder / "*.h5")))
    if h5_files:
        with h5py.File(h5_files[0], "r") as fh:
            keys = list(fh.keys())
            if "Data" in keys:
                shape = tuple(fh["Data"].shape)
                return "legacy", h5_files, shape

    raise FileNotFoundError(f"No recognizable .lux*.h5 / .h5 files in {folder}")


def discover(folder: Path, override: Optional[str] = None,
             n_planes_override: Optional[int] = None
             ) -> tuple[str, list[str], tuple, Optional[int]]:
    """Detect or override format. Print result. Returns (fmt, files, shape, n_planes_detected).

    n_planes_detected is set only for interleaved format when detected from
    metadata; otherwise None. Callers should use a locally-scoped variable
    rather than relying on global ARGS.n_planes being populated.
    """
    fmt, files, shape = detect_format(folder)
    if override and override != fmt:
        print(f"  Format override: detected={fmt} -> using={override}")
        fmt = override
    detected_n_planes: Optional[int] = None
    if fmt == "interleaved":
        if n_planes_override is not None:
            detected_n_planes = n_planes_override
        else:
            detected_n_planes = read_n_planes(files[0])
            print(f"  Auto-detected n_planes={detected_n_planes} from metadata")
    print(f"  Folder : {folder}")
    print(f"  Format : {fmt}  ({len(files)} file(s), sample shape={shape})")
    return fmt, files, shape, detected_n_planes



# =============================================================================
# UNIVERSAL LOADER
# =============================================================================

def load_movie(folder: Path, fmt: str, files: list[str], shape: tuple,
               z_index: Optional[int] = None,
               max_frames: Optional[int] = None,
               n_planes: Optional[int] = None) -> np.ndarray:
    """
    Build (T, H, W) float32 movie regardless of source format.

    For multi-tp: pick z_index plane from each timepoint file.
    For multi-cam: each file is one timepoint, use Data[0].
    For interleaved: stride the T axis by n_planes to extract one Z plane.
    For single-movie/legacy: read whole Data array (or chunks if huge).
    """
    if fmt == "interleaved":
        T_full, H, W = shape
        nz = n_planes or 1
        if z_index is None:
            z_index = nz // 2
        if z_index >= nz:
            raise ValueError(f"z_index {z_index} >= n_planes {nz}")
        indices = list(range(z_index, T_full, nz))
        if max_frames:
            indices = indices[:max_frames]
        print(f"  Interleaved: z={z_index}/{nz} planes -> {len(indices)} frames")
        with h5py.File(files[0], "r") as fh:
            data = fh["Data"][indices].astype(np.float32)
        return data

    if fmt == "multi-tp":
        Z, H, W = shape
        if z_index is None:
            z_index = Z // 2
        if z_index >= Z:
            raise ValueError(f"z_index {z_index} >= Z={Z}")
        T = len(files)
        if max_frames:
            T = min(T, max_frames)
            files = files[:T]
        print(f"  Loading {T} timepoints @ z={z_index}...")
        data = np.zeros((T, H, W), dtype=np.float32)
        for i, fp in enumerate(files):
            with h5py.File(fp, "r") as fh:
                data[i] = fh["Data"][z_index].astype(np.float32)
        return data

    if fmt == "multi-cam":
        _, H, W = shape
        T = len(files)
        if max_frames:
            T = min(T, max_frames)
            files = files[:T]
        print(f"  Loading {T} single-plane files...")
        data = np.zeros((T, H, W), dtype=np.float32)
        for i, fp in enumerate(files):
            with h5py.File(fp, "r") as fh:
                data[i] = fh["Data"][0].astype(np.float32)
        return data

    if fmt in ("single-movie", "legacy"):
        T_full, H, W = shape
        with h5py.File(files[0], "r") as fh:
            if n_planes and n_planes > 1:
                if z_index is None:
                    z_index = n_planes // 2
                if z_index >= n_planes:
                    raise ValueError(f"z_index {z_index} >= n_planes {n_planes}")
                indices = list(range(z_index, T_full, n_planes))
                if max_frames:
                    indices = indices[:max_frames]
                T = len(indices)
                print(f"  Striding {T_full} frames by n_planes={n_planes} (z={z_index}) -> {T} time-points...")
                data = fh["Data"][indices].astype(np.float32)
            else:
                T = T_full if max_frames is None else min(T_full, max_frames)
                print(f"  Loading {T}/{T_full} frames from single file...")
                data = fh["Data"][:T].astype(np.float32)
        return data

    raise ValueError(f"Unknown format: {fmt}")


def load_plane_multi_tp(files: list[str], z_index: int) -> np.ndarray:
    """Load specific z-plane across all multi-tp files. Used by plane-split."""
    with h5py.File(files[0], "r") as fh:
        Z, H, W = fh["Data"].shape
    if z_index >= Z:
        raise ValueError(f"z_index {z_index} >= Z={Z}")
    T = len(files)
    data = np.zeros((T, H, W), dtype=np.float32)
    for i, fp in enumerate(files):
        with h5py.File(fp, "r") as fh:
            data[i] = fh["Data"][z_index].astype(np.float32)
    return data


def load_plane_interleaved(filepath: str, z_index: int, n_planes: int) -> np.ndarray:
    """Load one Z plane from an interleaved single file by striding the T axis."""
    with h5py.File(filepath, "r") as fh:
        T_full, H, W = fh["Data"].shape
        if z_index >= n_planes:
            raise ValueError(f"z_index {z_index} >= n_planes {n_planes}")
        indices = list(range(z_index, T_full, n_planes))
        return fh["Data"][indices].astype(np.float32)


# =============================================================================
# PREPROCESSING  (brain mask removed — full frame used as-is)
# =============================================================================

def downsample(data: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize each frame with anti-aliasing."""
    T = data.shape[0]
    out = np.zeros((T, target_h, target_w), dtype=np.float32)
    for t in range(T):
        out[t] = skimage.transform.resize(
            data[t], (target_h, target_w), anti_aliasing=True,
        )
    return out


def stripe_remove(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract per-column temporal+row median (light-sheet stripe artifact)."""
    col_median = np.median(data, axis=(0, 2), keepdims=True)
    cleaned = np.clip(data - col_median, 0, None).astype(np.float32)
    return cleaned, col_median


def compute_dff(traces: np.ndarray) -> np.ndarray:
    """
    Compute dF/F with optional double-exponential photobleaching correction.

    For recordings >= 100 frames (and --no-bleach-correct not set), fits
    a * exp(-t/tau1) + b * exp(-t/tau2) per neuron and subtracts the trend
    before computing F0 via constant percentile baseline.

    Returns (N, T) dF/F array.
    """
    N, T = traces.shape
    dff = np.zeros_like(traces, dtype=np.float32)
    t = np.arange(T, dtype=np.float64)
    bleach_correct = (not ARGS.no_bleach_correct) and (T >= 100)

    def _double_exp(t, a, tau1, b, tau2):
        return a * np.exp(-t / tau1) + b * np.exp(-t / tau2)

    for i in range(N):
        F = traces[i].astype(np.float64)

        if bleach_correct:
            try:
                F_range = float(F.max() - F.min())
                if F_range < 1e-6:
                    raise ValueError("flat trace, cannot fit bleach")
                p0 = [F_range * 0.5, T * 0.3, F_range * 0.3, T * 0.1]
                bounds = ([0, 1, 0, 1], [np.inf, T * 10, np.inf, T * 10])
                popt, _ = curve_fit(_double_exp, t, F - F.min(), p0=p0,
                                    bounds=bounds, maxfev=5000)
                trend = _double_exp(t, *popt)
                # Sanity checks on the fitted trend
                if trend[-1] > trend[0] + 0.2 * F_range:
                    raise ValueError(f"bleach trend is increasing (end={trend[-1]:.2f} > start={trend[0]:.2f})")
                residual = F - trend
                if residual.min() < -0.5 * F_range:
                    raise ValueError(f"bleach trend overshoots (residual min={residual.min():.2f} < {-0.5*F_range:.2f})")
                F = residual  # subtract bleach, keep residual + offset
            except Exception as exc:
                print(f"  WARNING: bleach fit rejected for neuron {i} ({exc}); using raw trace")
                pass  # fit failed or rejected; use raw F

        F0 = np.percentile(F, ARGS.dff_percentile)
        if F0 < 1e-6:
            print(f"  WARNING: neuron {i} F0={F0:.4f} <= 0 after bleach correction; clamping to 1e-6")
            F0 = 1e-6
        dff[i] = (F - F0) / F0

    return dff


def preprocess_movie(data: np.ndarray, label: str = "") -> tuple[np.ndarray, dict]:
    """
    Preprocessing pipeline: resolution + optional stripe removal.
    No brain mask is applied — the full frame is used as-is.
    Returns (preprocessed_movie, metadata_dict).
    """
    info = {"original_shape": tuple(data.shape)}
    print(f"\n[preprocess {label}]  input shape={data.shape}")

    # Resolution
    if ARGS.resolution == "512":
        target = (512, 512)
    elif ARGS.resolution == "1024":
        target = (1024, 1024)
    else:
        target = data.shape[1:]

    if data.shape[1:] != target:
        print(f"  Downsampling -> {target}")
        data = downsample(data, *target)
        info["downsampled_to"] = target

    # Stripe removal
    if not ARGS.no_stripe:
        data, col_median = stripe_remove(data)
        info["stripe_removed"] = True
        # Save stripe plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].imshow(data[data.shape[0] // 2], cmap="gray")
        axes[0].set_title(f"Frame {data.shape[0]//2} after stripe removal")
        axes[0].axis("off")
        axes[1].imshow(col_median.reshape(1, -1), cmap="gray", aspect="auto")
        axes[1].set_title("Removed stripe pattern")
        axes[1].set_xlabel("Column index")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"preprocess_{label or 'main'}.png"), dpi=100)
        plt.close(fig)
    else:
        info["stripe_removed"] = False

    info["final_shape"] = tuple(data.shape)
    return data, info


# =============================================================================
# CNMF CONFIG (resolution-aware base params + calibrated best params)
# =============================================================================

def get_base_params() -> dict:
    if ARGS.resolution == "512":
        mc = dict(max_shifts=(3, 3), strides=(48, 48),
                  overlaps=(24, 24), max_deviation_rigid=2, border_nan="copy")
        ssub = 1
    elif ARGS.resolution == "1024":
        mc = dict(max_shifts=(6, 6), strides=(96, 96),
                  overlaps=(48, 48), max_deviation_rigid=3, border_nan="copy")
        ssub = 1
    else:
        mc = dict(max_shifts=(12, 12), strides=(192, 192),
                  overlaps=(96, 96), max_deviation_rigid=3, border_nan="copy")
        ssub = 2  # subsample 2x to cut init time ~16x on large FOV

    return {
        "fr": 5,  # TODO: verify actual frame rate from acquisition metadata/timestamps
        "decay_time": 0.25,  # GCaMP8m off-kinetics; revisit once measured from real transients
        "method_init": "corr_pnr",
        "K": None,
        "nb": 0,
        "nb_patch": 0,
        "center_psf": True,
        "ring_size_factor": 1.4,
        "merge_thr": 0.85,
        "use_cnn": False,
        "min_SNR": ARGS.min_snr_trace,
        "rval_thr": 0.85,
        "del_duplicates": True,
        "ssub": ssub,
        "tsub": 1,
        "only_init": False,
        "pw_rigid": True,
        **mc,
    }


# Base (resolution-aware) params, with the calibrated best_params.json values
# merged on top — this is the single source of CNMF params used everywhere
# below; there is no per-run search or override beyond this.
BASE_PARAMS = get_base_params()
BASE_PARAMS.update(load_best_params(BEST_PARAMS_PATH))


# =============================================================================
# CNMF + REFIT VALIDATION
# =============================================================================

def array_to_memmap(array: np.ndarray, basename: Path) -> str:
    tif = str(basename) + ".tif"
    tifffile.imwrite(tif, array.astype(np.float32))
    mmap_path = caiman.mmapping.save_memmap(
        [tif], base_name=str(basename), order="C", border_to_0=0,
    )
    if not ARGS.keep_temp:
        try:
            os.remove(tif)
        except OSError:
            pass  # best-effort cleanup
    return mmap_path

def _prep_params(params_override: dict, fname_mmap: str) -> "params_module.CNMFParams":
    """Stage: build CNMFParams from overrides. Normalizes gSig/gSig_filt to int tuples."""
    p = {**BASE_PARAMS, **params_override, "fnames": [fname_mmap]}
    for key in ("gSig", "gSig_filt"):
        val = p.get(key)
        if val is None:
            continue
        if isinstance(val, tuple):
            p[key] = (int(val[0]), int(val[1]))
        else:
            p[key] = (int(val), int(val))

    if "gSiz" not in params_override:
        g = p["gSig"]
        p["gSiz"] = (4 * int(g[0]) + 1, 4 * int(g[1]) + 1)

    return params_module.CNMFParams(params_dict=p)

def _setup_cluster():
    """Stage: start multiprocessing cluster. Falls back to single-threaded on failure."""
    try:
        _, cluster, n_processes = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=N_WORKERS, single_thread=False
        )
        return cluster, n_processes
    except Exception as exc:
        print(f"  [STAGE:cluster_setup] failed, running single-threaded: {exc}", flush=True)
        return None, 1

def _motion_correct(fname_mmap: str, opts, cluster) -> str:
    """Stage: motion correction. Raises on failure — caller decides how to handle."""
    print("  [STAGE:motion_correction] starting", flush=True)
    mc = MotionCorrect([fname_mmap], dview=cluster, **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)
    fname_to_use = mc.mmap_file[0] if isinstance(mc.mmap_file, list) else mc.mmap_file
    print(f"  [STAGE:motion_correction] done -> {fname_to_use}", flush=True)
    return fname_to_use

def _fit_cnmf(fname_to_use: str, opts, n_processes: int, cluster):
    """Stage: core CNMF fit. Raises on failure — this is the critical path."""
    opts.change_params({'fnames': [fname_to_use]})
    
    images = cm.load(fname_to_use)
    cnmf_obj = cnmf_module.CNMF(n_processes=n_processes, params=opts, dview=cluster)
    
    print('[STAGE:fit] starting', flush=True)
    cnmf_obj.fit(images)
    
    n_components = cnmf_obj.estimates.A.shape[1] if cnmf_obj.estimates.A is not None else 0
    print(f'[STAGE:fit] done -> {n_components} components', flush=True)
    
    return cnmf_obj

def _reload_images(fname_to_use: str):
    """Stage: reload the mmap actually fit, for evaluate/refit. Returns None on failure
    (non-fatal — evaluate/refit are simply skipped downstream)."""
    try:
        Yr, dims, T_loc = caiman.mmapping.load_memmap(fname_to_use)
        return np.reshape(Yr.T, [T_loc] + list(dims), order="F")
    except Exception as exc:
        print(f"  [STAGE:reload_images] failed, evaluate/refit will be skipped: {exc}", flush=True)
        return None

def _evaluate_and_select(cnmf_obj, images, cluster):
    """Stage: CaImAn's own component evaluation (SNR/spatial/CNN). Non-fatal on failure —
    keeps whatever components existed before this stage."""
    try:
        print("  [STAGE:evaluate_components] starting", flush=True)
        cnmf_obj.estimates.evaluate_components(imgs=images, params=cnmf_obj.params, dview=cluster)
        cnmf_obj.estimates.select_components(use_object=True)
        print(f"  [STAGE:evaluate_components] done -> {cnmf_obj.estimates.A.shape[1]} components remain", flush=True)
    except Exception as exc:
        print(f"  [STAGE:evaluate_components] failed, keeping unfiltered components: {exc}", flush=True)


def _refit(cnmf_obj, images, cluster):
    """Stage: second-pass validation refit. Non-fatal on failure — keeps pre-refit estimates."""
    try:
        print("  [STAGE:refit] starting — second-pass validation of accepted neurons", flush=True)
        refit_obj = cnmf_obj.refit(images, dview=cluster)
        print(f"  [STAGE:refit] done -> {refit_obj.estimates.A.shape[1]} neurons remain", flush=True)
        return refit_obj
    except Exception as exc:
        print(f"  [STAGE:refit] failed, keeping pre-refit estimates: {exc}", flush=True)
        return cnmf_obj


def run_cnmf(params_override: dict, fname_mmap: str,
             do_mc: bool = True, do_filter_caiman: bool = True,
             do_refit: bool = True):
    """
    Run CNMF, then validate found neurons in two passes:
      1. CaImAn's own evaluate_components / select_components (SNR, spatial
         consistency, CNN if enabled).
      2. cnmf_model.refit() — re-runs the full CNMF fit seeded with the
         accepted spatial/temporal components, re-estimating traces and
         re-deriving background against the actual data. This acts as an
         additional, independent validation pass before we score the run.

    Returns (cnmf_obj, elapsed_seconds, fname_used), where fname_used is the
    path of the mmap that was actually fit (the motion-corrected one, when
    do_mc is True). Callers must reload Yr from this path when scoring —
    not from the pre-motion-correction mmap that was originally passed in —
    otherwise recon_error is computed against misaligned data and comes out
    close to 1 regardless of fit quality.

    On failure, the exception is caught and printed with a [STAGE:...] tag
    identifying exactly which step raised it, and (None, elapsed, None) is
    returned. Only cluster_setup, evaluate_components, and refit are
    non-fatal by design (they degrade gracefully); motion_correction and
    fit_file are fatal because downstream stages depend on their output.
    """
    opts = _prep_params(params_override, fname_mmap)
    cluster, n_processes = _setup_cluster()

    t0 = time.time()
    try:
        fname_to_use = fname_mmap
        if do_mc:
            fname_to_use = _motion_correct(fname_mmap, opts, cluster)
        else:
            print("  [STAGE:motion_correction] skipped — using precomputed mmap", flush=True)
        
        cm.stop_server(dview=cluster)
        cluster, n_processes = _setup_cluster()
        
        cnmf_obj = _fit_cnmf(fname_to_use, opts, n_processes, cluster)

        images = None
        if cnmf_obj.estimates.A.shape[1] > 0:
            images = _reload_images(fname_to_use)

        if do_filter_caiman and images is not None:
            _evaluate_and_select(cnmf_obj, images, cluster)

        if do_refit and images is not None and cnmf_obj.estimates.A.shape[1] > 0:
            cnmf_obj = _refit(cnmf_obj, images, cluster)

        return cnmf_obj, time.time() - t0, fname_to_use

    except Exception as exc:
        # Only _motion_correct and _fit_cnmf raise past this point (fatal stages).
        # The printed message already carries a [STAGE:...] tag from inside those
        # functions where the exception originated; if not, tag it here.
        print(f"    CNMF failed: {exc}", flush=True)
        return None, time.time() - t0, None
    finally:
        if cluster is not None:
            try:
                caiman.stop_server(dview=cluster)
            except Exception:
                pass

def load_yr_for_scoring(fname_used: Optional[str], fallback_Yr):
    """
    Reload Yr from the exact mmap a CNMF run was fit on, so scoring never
    compares AC against a different (e.g. pre-motion-correction) movie than
    the one the model was actually fit to.

    Falls back to `fallback_Yr` if fname_used is missing or fails to load;
    this is safe because score_run() only reaches the point of using Yr
    when cnmf_obj is non-None with accepted components — if run_cnmf itself
    failed (fname_used is None), score_run returns its sentinel without
    touching Yr at all.
    """
    if fname_used is None:
        return fallback_Yr
    try:
        Yr_used, _, _ = caiman.mmapping.load_memmap(fname_used)
        return Yr_used
    except Exception as exc:
        print(f"  [WARNING: could not reload registered mmap '{fname_used}' for scoring "
              f"({exc}); falling back to pre-registration Yr]", flush=True)
        return fallback_Yr


def score_run(cnmf_obj, Yr, dims: tuple[int, int], stability: float = 0.0) -> dict:
    """
    Composite score over all neurons CaImAn accepted (post evaluate_components/
    select_components/refit — no additional geometric quality filtering here).

    composite = 1.0*(1 - recon_error)
              + 0.5*spatial_compactness
              - 0.3*log(1 + trace_sparsity)
              + 1.0*stability
              + 0.001*log(1 + n_neurons)   # small bonus for finding more neurons
    """
    sentinel = {
        "n_neurons": 0, "recon_error": 1.0,
        "spatial_compactness": 0.0, "trace_sparsity": float("inf"),
        "stability": stability, "composite_score": -float("inf"),
    }
    if cnmf_obj is None or cnmf_obj.estimates.A.shape[1] == 0:
        return sentinel

    A = cnmf_obj.estimates.A
    C = cnmf_obj.estimates.C
    n = A.shape[1]

    # Reconstruction error via Frobenius norm identity - avoids materialising
    # the (pixels x frames) dense reconstruction matrix (~11 GB at full res).
    #   ||Y - A@C||^2_F = ||Y||^2_F - 2*trace(C^T * A^T * Y) + ||A@C||^2_F
    # All intermediates are (n x T) or (n x n) - at most a few MB.
    Yr_norm_sq = float(np.linalg.norm(Yr, "fro") ** 2)
    AtYr = A.T @ Yr                     # (n x T) dense
    AtA = A.T @ A                       # (n x n) sparse
    recon_norm_sq = Yr_norm_sq - 2.0 * float(np.sum(C * AtYr)) + float(np.sum(C * (AtA @ C)))

    b = getattr(cnmf_obj.estimates, "b", None)
    f_bg = getattr(cnmf_obj.estimates, "f", None)
    if b is not None and f_bg is not None and b.shape[1] > 0:
        # Background term   ||b*f_bg||^2_F = sum(f_bg * (b^T * b * f_bg))
        bt_plus = b.T @ b
        bg_norm_sq = float(np.sum(f_bg * (bt_plus @ f_bg)))
        # Cross-term  2*trace(C^T * A^T * b * f_bg) = 2*sum( (A^T*b) * (C*f_bg^T) )
        Atb = A.T @ b                    # (n x nb) dense
        CfbgT = C @ f_bg.T               # (n x nb) dense
        cross = 2.0 * float(np.sum(Atb * CfbgT))
        recon_norm_sq += bg_norm_sq + cross

    recon_norm_sq = max(recon_norm_sq, 0.0)
    recon_error = float(np.sqrt(recon_norm_sq / max(Yr_norm_sq, 1e-9)))

    H_val, W_val = dims
    comps = []
    for col in range(n):
        fp = np.asarray(A[:, col].todense()).flatten().reshape(H_val, W_val)
        binary = fp > (fp.max() * 0.2)
        if binary.sum() < 5:
            continue
        try:
            hull = convex_hull_image(binary)
            comps.append(float(binary.sum()) / float(hull.sum()))
        except Exception:
            pass
    spatial_compactness = float(np.mean(comps)) if comps else 0.0

    l1 = np.sum(np.abs(C), axis=1)
    l2 = np.linalg.norm(C, axis=1)
    trace_sparsity = float(np.mean(l1 / (l2 + 1e-9)))

    composite = (
        1.0 * (1.0 - recon_error)
        + 0.5 * spatial_compactness
        - 0.3 * np.log1p(trace_sparsity)
        + 1.0 * stability
        + 0.001 * np.log1p(n)
    )

    return {
        "n_neurons": n,
        "recon_error": recon_error,
        "spatial_compactness": spatial_compactness,
        "trace_sparsity": trace_sparsity,
        "stability": stability,
        "composite_score": float(composite),
    }


def compute_stability(A1, A2, threshold: float = 0.5) -> float:
    if A1 is None or A2 is None or A1.shape[1] == 0 or A2.shape[1] == 0:
        return 0.0
    n1 = np.asarray(np.sqrt(A1.power(2).sum(axis=0))).flatten() + 1e-9
    n2 = np.asarray(np.sqrt(A2.power(2).sum(axis=0))).flatten() + 1e-9
    corr = np.asarray((A1.multiply(1.0 / n1).T @ A2.multiply(1.0 / n2)).todense())
    ri, ci = linear_sum_assignment(-corr)
    return float(np.mean(corr[ri, ci] >= threshold))


# =============================================================================
# TEST / VALIDATION PHASE
# =============================================================================

def test_cnmf(params: dict, mmap_path: str, data: np.ndarray,
              dims: tuple[int, int], label: str,
              tune_A=None) -> dict:
    """Run CNMF (+ refit validation) on a split, save plots/traces, and score the run.

    If tune_A is given (footprints from a reference/held-out split), the
    footprint-matching stability between this run and the reference is
    folded into the composite score.
    """
    print(f"\n{'='*60}\nTEST: {label}\n{'='*60}")

    Yr_unregistered, _, _ = caiman.mmapping.load_memmap(mmap_path)
    cnmf_obj, rt, fname_used = run_cnmf(params, mmap_path)
    # Score against the mmap CNMF actually fit on (post motion-correction),
    # not the pre-registration Yr_unregistered loaded above — otherwise a
    # pixel-shift mismatch between the fitted data and the scored data
    # drives recon_error to ~1 independent of how good the fit actually is.
    Yr = load_yr_for_scoring(fname_used, Yr_unregistered)

    stability = 0.0
    if tune_A is not None and cnmf_obj is not None and cnmf_obj.estimates.A.shape[1] > 0:
        stability = compute_stability(tune_A, cnmf_obj.estimates.A)

    metrics = score_run(cnmf_obj, Yr, dims, stability=stability)
    metrics["label"] = label
    metrics["runtime_s"] = round(rt, 1)

    n = metrics["n_neurons"]
    print(f"  Neurons: {n}  composite: {metrics['composite_score']:+.4f}  stability: {stability:.3f}  t={rt:.0f}s")

    if cnmf_obj is not None and n > 0:
        safe = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        H_val, W_val = dims

        # Contour plot
        mean_frame = data.mean(axis=0)
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(mean_frame, cmap="gray")
        ax.set_title(f"{label}: {n} neurons", fontsize=10)
        ax.axis("off")
        for i in range(n):
            fp = np.asarray(cnmf_obj.estimates.A[:, i].todense()).flatten().reshape(H_val, W_val)
            if fp.max() == 0:
                continue
            ax.contour(fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.4, alpha=0.85)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"contours_{safe}.png"), dpi=150)
        plt.close(fig)

        # Raw traces and dF/F
        traces = cnmf_obj.estimates.C
        dff = compute_dff(traces)
        bleach_applied = (not ARGS.no_bleach_correct) and (traces.shape[1] >= 100)

        n_plot = min(8, traces.shape[0])
        T = data.shape[0]
        t_axis = np.arange(T)
        fig, axes_t = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
        if n_plot == 1:
            axes_t = [axes_t]
        for i, ax in enumerate(axes_t):
            ax.plot(t_axis, dff[i], lw=1)
            ax.set_ylabel(f"N{i}", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes_t[-1].set_xlabel("Frame")
        bleach_tag = " + bleach corr" if bleach_applied else ""
        plt.suptitle(f"dF/F{bleach_tag} — {label}")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"traces_{safe}.png"), dpi=120)
        plt.close(fig)

        np.save(str(OUTPUT_DIR / f"traces_{safe}.npy"), traces)
        np.save(str(OUTPUT_DIR / f"dff_{safe}.npy"), dff)
        print(f"  dF/F: percentile={ARGS.dff_percentile}  bleach_correct={bleach_applied}")

    return metrics


# =============================================================================
# OUTPUT
# =============================================================================

def save_summary(mode: str, params: dict, test_results: dict,
                 fmt_info: dict, extra: dict | None = None):
    summary = {
        "mode": mode,
        "run_name": ARGS.run_name,
        "resolution": ARGS.resolution,
        "stripe_removal": not ARGS.no_stripe,
        "min_snr_trace": ARGS.min_snr_trace,
        "best_params_path": str(BEST_PARAMS_PATH),
        "params": params,
        "format_info": fmt_info,
        "tests": {},
    }
    for label, m in test_results.items():
        if isinstance(m, dict):
            summary["tests"][label] = {
                k: v for k, v in m.items()
                if not isinstance(v, (np.ndarray,))
            }
    if extra:
        summary.update(extra)

    with open(str(OUTPUT_DIR / "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved summary.json -> {OUTPUT_DIR}/")

    # Append master CSV
    master = RESULTS_ROOT / "all_runs.csv"
    # Use the held-out test result as the canonical headline row.
    # Priority order: "test_file" > "test_half" > first non-tune key > first key.
    # This ensures cross-validation results (not the full-movie re-run) are
    # recorded in the aggregate table.
    prefs = ["test_file", "test_half"]
    headline_key = None
    for k in prefs:
        if k in test_results:
            headline_key = k
            break
    if headline_key is None:
        # Fallback: exclude keys containing "tune" and pick the first remaining
        for k, v in test_results.items():
            if "tune" not in k and isinstance(v, dict):
                headline_key = k
                break
    if headline_key is None:
        headline_key = next(iter(test_results.keys()), None)
    headline_test = test_results.get(headline_key)
    if isinstance(headline_test, dict):
        row = {
            "run_name": ARGS.run_name,
            "mode": mode,
            "resolution": ARGS.resolution,
            "test_key": headline_key,
            "n_neurons": headline_test.get("n_neurons", 0),
            "composite_score": headline_test.get("composite_score", float("nan")),
            "stability": headline_test.get("stability", 0.0),
            "gSig": params.get("gSig"),
            "min_corr": params.get("min_corr"),
            "min_pnr": params.get("min_pnr"),
            "rf": params.get("rf"),
            "p": params.get("p"),
        }
        df_row = pd.DataFrame([row])
        if master.is_file():
            df_row.to_csv(master, mode="a", header=False, index=False)
        else:
            df_row.to_csv(master, index=False)
        print(f"Appended row to {master}")


# =============================================================================
# MODES
# =============================================================================

def mode_time_split():
    """Run CNMF with fixed params on the first half of frames (reference, for
    stability comparison) and test on the second half + full movie. Same file/Z."""
    print(f"\n--- TIME-SPLIT mode ---")
    fmt, files, sample_shape, _nplanes = discover(ARGS.data_dir, ARGS.format_override)

    z_index = ARGS.z_index
    if fmt == "multi-tp":
        Z = sample_shape[0]
        z_index = Z // 2 if z_index is None else z_index
    elif fmt == "interleaved" and z_index is None:
        z_index = (_nplanes if _nplanes is not None else ARGS.n_planes) // 2
    elif ARGS.n_planes and z_index is None:
        z_index = ARGS.n_planes // 2

    raw = load_movie(ARGS.data_dir, fmt, files, sample_shape,
                     z_index=z_index, max_frames=ARGS.max_frames,
                     n_planes=_nplanes if _nplanes is not None else ARGS.n_planes)
    data, prep_info = preprocess_movie(raw, label="movie")

    T_full = data.shape[0]
    if T_full < 4:
        print(f"ERROR: only {T_full} frames; time-split needs >=4")
        sys.exit(1)
    mid = T_full // 2
    print(f"\nTime split: reference frames 0-{mid-1} ({mid}), test frames {mid}-{T_full-1} ({T_full-mid})")

    dims = data.shape[1:]
    tune_mmap = array_to_memmap(data[:mid], WORK_DIR / "tune_half")
    test_mmap = array_to_memmap(data[mid:], WORK_DIR / "test_half")

    print("\nRunning CNMF with configured params on reference half (for stability)...")
    cnmf_tune, _, _ = run_cnmf(BASE_PARAMS, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune is not None else None

    test_metrics = test_cnmf(
        BASE_PARAMS, test_mmap, data[mid:], dims,
        label=f"test_half (frames {mid}-{T_full-1})", tune_A=tune_A,
    )
    full_mmap = array_to_memmap(data, WORK_DIR / "full_movie")
    full_metrics = test_cnmf(
        BASE_PARAMS, full_mmap, data, dims,
        label="full_movie", tune_A=tune_A,
    )

    fmt_info = {"format": fmt, "n_files": len(files), "sample_shape": list(sample_shape),
                **prep_info}
    save_summary("time-split", BASE_PARAMS,
                 {"test_half": test_metrics, "full_movie": full_metrics},
                 fmt_info,
                 extra={"z_index": z_index, "n_planes": _nplanes if _nplanes is not None else ARGS.n_planes,
                        "T_total": T_full, "data_dir": str(ARGS.data_dir)})


def mode_plane_split():
    """Run CNMF with fixed params on one Z-plane (reference, for stability) and
    test on every other Z. Supports multi-tp and interleaved formats."""
    print(f"\n--- PLANE-SPLIT mode ---")
    fmt, files, sample_shape, _nplanes = discover(ARGS.data_dir, ARGS.format_override)

    if fmt == "interleaved":
        Z = _nplanes if _nplanes is not None else ARGS.n_planes
        def _load_plane(z):
            return load_plane_interleaved(files[0], z, Z)
    elif fmt == "multi-tp":
        Z = sample_shape[0]
        def _load_plane(z):
            return load_plane_multi_tp(files, z)
    else:
        print(f"ERROR: plane-split requires multi-tp or interleaved format (got {fmt}).")
        print("Use time-split for single-plane data, or file-plane-split for cross-file.")
        sys.exit(2)

    if Z is None or Z < 2:
        print(f"ERROR: plane-split needs Z>=2 (got {Z})")
        sys.exit(2)

    tune_z = ARGS.tune_z if ARGS.tune_z is not None else Z // 2
    if tune_z >= Z:
        print(f"ERROR: --tune-z {tune_z} >= Z={Z}")
        sys.exit(1)
    print(f"\nZ={Z}; reference z={tune_z}, test z=0..{Z-1}\\{{tune_z}}")

    tune_raw = _load_plane(tune_z)
    tune_data, prep_info = preprocess_movie(tune_raw, label=f"tune_z{tune_z}")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / f"tune_z{tune_z}")

    print(f"\nRunning CNMF with configured params on z={tune_z} (reference for stability)...")
    cnmf_tune, _, _ = run_cnmf(BASE_PARAMS, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune is not None else None

    all_metrics = {}
    for z in range(Z):
        print(f"\n--- z={z} ---")
        test_raw = _load_plane(z)
        test_data, _ = preprocess_movie(test_raw, label=f"test_z{z}")
        test_mmap = array_to_memmap(test_data, WORK_DIR / f"test_z{z}")
        m = test_cnmf(
            BASE_PARAMS, test_mmap, test_data, test_data.shape[1:],
            label=f"z{z}" + (" (tune)" if z == tune_z else ""),
            tune_A=tune_A if z != tune_z else None,
        )
        all_metrics[f"z{z}"] = m

    rows = [
        {"z_plane": k, "n_neurons": v["n_neurons"],
         "composite": v["composite_score"], "stability_vs_tune": v.get("stability", 0.0),
         "recon_error": v["recon_error"]}
        for k, v in all_metrics.items()
    ]
    pd.DataFrame(rows).to_csv(str(OUTPUT_DIR / "plane_split_summary.csv"), index=False)
    print("\n[plane-split summary]")
    print(pd.DataFrame(rows).to_string(index=False))

    fmt_info = {"format": fmt, "n_files": len(files), "sample_shape": list(sample_shape),
                **prep_info}
    save_summary("plane-split", BASE_PARAMS, all_metrics, fmt_info,
                 extra={"tune_z": tune_z, "Z": Z, "data_dir": str(ARGS.data_dir)})


def mode_file_plane_split():
    """Run CNMF with fixed params on file A z (reference, for stability), test on file B same z."""
    print(f"\n--- FILE-PLANE-SPLIT mode ---")
    print("\nReference dataset:")
    fmt_t, tune_files, shape_t, _nplanes_t = discover(ARGS.tune_dir, ARGS.format_override)
    print("\nTest dataset:")
    fmt_te, test_files, shape_te, _nplanes_te = discover(ARGS.test_dir, ARGS.format_override)

    z_index = ARGS.z_index
    if fmt_t == "multi-tp" and z_index is None:
        z_index = shape_t[0] // 2
    elif fmt_t == "interleaved" and z_index is None:
        z_index = (_nplanes_t if _nplanes_t is not None else ARGS.n_planes) // 2
    elif ARGS.n_planes and z_index is None:
        z_index = ARGS.n_planes // 2
    if z_index is None:
        z_index = 0
    print(f"\nUsing z={z_index}")

    print("\n[Loading reference]")
    tune_raw = load_movie(ARGS.tune_dir, fmt_t, tune_files, shape_t,
                          z_index=z_index, max_frames=ARGS.max_frames,
                          n_planes=_nplanes_t if _nplanes_t is not None else ARGS.n_planes)
    tune_data, prep_info_tune = preprocess_movie(tune_raw, label="tune")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / "tune")

    print("\nRunning CNMF with configured params on reference file (for stability)...")
    cnmf_tune, _, _ = run_cnmf(BASE_PARAMS, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune is not None else None

    print("\n[Loading test]")
    test_raw = load_movie(ARGS.test_dir, fmt_te, test_files, shape_te,
                          z_index=z_index, max_frames=ARGS.max_frames,
                          n_planes=_nplanes_te if _nplanes_te is not None else ARGS.n_planes)
    test_data, prep_info_test = preprocess_movie(test_raw, label="test")
    test_mmap = array_to_memmap(test_data, WORK_DIR / "test")

    test_metrics = test_cnmf(
        BASE_PARAMS, test_mmap, test_data, test_data.shape[1:],
        label="test_file", tune_A=tune_A,
    )

    fmt_info = {"tune": {"format": fmt_t, "n_files": len(tune_files),
                          "sample_shape": list(shape_t), **prep_info_tune},
                "test": {"format": fmt_te, "n_files": len(test_files),
                          "sample_shape": list(shape_te), **prep_info_test}}
    save_summary("file-plane-split", BASE_PARAMS, {"test_file": test_metrics},
                 fmt_info,
                 extra={"z_index": z_index,
                        "tune_dir": str(ARGS.tune_dir),
                        "test_dir": str(ARGS.test_dir)})


def mode_file_split():
    """Run CNMF with fixed params on file A (reference, one Z), test on file B at
    every Z if multi-tp/interleaved; else a single test."""
    print(f"\n--- FILE-SPLIT mode ---")
    print("\nReference dataset:")
    fmt_t, tune_files, shape_t, _nplanes_t = discover(ARGS.tune_dir, ARGS.format_override)
    print("\nTest dataset:")
    fmt_te, test_files, shape_te, _nplanes_te = discover(ARGS.test_dir, ARGS.format_override)

    if fmt_t == "multi-tp":
        tune_z = shape_t[0] // 2
    elif fmt_t == "interleaved":
        tune_z = (_nplanes_t if _nplanes_t is not None else ARGS.n_planes) // 2
    else:
        tune_z = 0
    print(f"\nUsing reference-file z={tune_z}")

    print("\n[Loading reference]")
    tune_raw = load_movie(ARGS.tune_dir, fmt_t, tune_files, shape_t,
                          z_index=tune_z, max_frames=ARGS.max_frames,
                          n_planes=_nplanes_t if _nplanes_t is not None else ARGS.n_planes)
    tune_data, prep_info_tune = preprocess_movie(tune_raw, label=f"tune_z{tune_z}")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / "tune")

    print("\nRunning CNMF with configured params on reference file (for stability)...")
    cnmf_tune, _, _ = run_cnmf(BASE_PARAMS, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune is not None else None

    if fmt_te == "multi-tp":
        Z_te = shape_te[0]
        z_iter = range(Z_te)
    elif fmt_te == "interleaved":
        Z_te = _nplanes_te if _nplanes_te is not None else ARGS.n_planes
        z_iter = range(Z_te)
    else:
        Z_te = 1
        _z0 = ARGS.z_index if ARGS.z_index is not None else ((_nplanes_te if _nplanes_te is not None else ARGS.n_planes) // 2 if (_nplanes_te is not None or ARGS.n_planes) else 0)
        z_iter = [_z0]

    all_metrics = {}
    for z in z_iter:
        label = f"test_z{z}" if Z_te > 1 else "test_file"
        print(f"\n--- {label} ---")
        if fmt_te == "multi-tp":
            test_raw = load_plane_multi_tp(test_files, z)
        elif fmt_te == "interleaved":
            test_raw = load_plane_interleaved(test_files[0], z, Z_te)
        else:
            test_raw = load_movie(ARGS.test_dir, fmt_te, test_files, shape_te,
                                  z_index=z, max_frames=ARGS.max_frames,
                                  n_planes=_nplanes_te if _nplanes_te is not None else ARGS.n_planes)
        test_data, _ = preprocess_movie(test_raw, label=label)
        test_mmap = array_to_memmap(test_data, WORK_DIR / label)
        m = test_cnmf(
            BASE_PARAMS, test_mmap, test_data, test_data.shape[1:],
            label=label, tune_A=tune_A,
        )
        all_metrics[label] = m

    if Z_te > 1:
        rows = [
            {"z_plane": k, "n_neurons": v["n_neurons"],
             "composite": v["composite_score"],
             "stability_vs_tune": v.get("stability", 0.0),
             "recon_error": v["recon_error"]}
            for k, v in all_metrics.items()
        ]
        pd.DataFrame(rows).to_csv(str(OUTPUT_DIR / "file_split_summary.csv"), index=False)
        print("\n[file-split summary]")
        print(pd.DataFrame(rows).to_string(index=False))

    fmt_info = {"tune": {"format": fmt_t, "n_files": len(tune_files),
                          "sample_shape": list(shape_t), **prep_info_tune},
                "test": {"format": fmt_te, "n_files": len(test_files),
                          "sample_shape": list(shape_te)}}
    save_summary("file-split", BASE_PARAMS, all_metrics, fmt_info,
                 extra={"tune_dir": str(ARGS.tune_dir),
                        "test_dir": str(ARGS.test_dir),
                        "tune_z": tune_z, "Z_test": Z_te})


# =============================================================================
# MAIN
# =============================================================================

MODES = {
    "time-split": mode_time_split,
    "plane-split": mode_plane_split,
    "file-plane-split": mode_file_plane_split,
    "file-split": mode_file_split,
}

t_start = time.time()
MODES[ARGS.mode]()
elapsed_min = (time.time() - t_start) / 60.0

print(f"\n{'='*70}")
print(f"DONE  |  {ARGS.mode}  |  {ARGS.run_name}  |  {elapsed_min:.1f} min")
print(f"{'='*70}")