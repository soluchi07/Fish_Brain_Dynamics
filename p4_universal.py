#!/usr/bin/env python3
"""
p4_universal.py  —  Phase 4: Universal CNMF Pipeline with Quality Filters

Major upgrades over p3:
  * Auto-detects 4 input formats:
      multi-tp     : many tp-*.lux.h5 files, each (Z, H, W)
      multi-cam    : many Cam_long_*.lux.h5 files, each (1, H, W)   (13iii26 style)
      single-movie : one big Cam_long_*.lux.h5 file, shape (T, H, W) (20iv26 style)
      legacy       : one .h5 file with shape (T, H, W)
  * Optional brain mask preprocessing (default ON) — kills false positives
    in dark periphery (addresses bio team "shouldn't capture above the planes")
  * Quality filters (circularity, max-area, in-mask, SNR) applied INSIDE every
    Bayesian trial — the composite score uses FILTERED neuron count, so the
    optimizer is rewarded for finding REAL neurons, not piles of noise blobs
  * Configurable resolution: --resolution {full, 1024, 512}
  * 4 validation modes (time-split, plane-split, file-plane-split, file-split)
    with graceful skipping when data doesn't support a mode (e.g. Z=1)

Usage examples:

  source ~/Documents/fishBrain_Kiitan/Fish_Brain_Dynamics/venv/bin/activate

  # 13iii26 task1, time-split, 512x512 (fast)
  python p4_universal.py --mode time-split \\
      --data-dir "/path/to/13iii26/Xgcamp_..._task1stack0_..._channel_2_obj_bottom" \\
      --run-name 13iii26_task1_timesplit \\
      --resolution 512 --n-calls 10

  # Cross-task generalization (file-plane-split tune Task1 -> test Task3)
  python p4_universal.py --mode file-plane-split \\
      --tune-dir "/path/to/...task1..." \\
      --test-dir "/path/to/...task3..." \\
      --run-name 13iii26_task1_to_task3 \\
      --resolution 512 --n-calls 10

  # Smoke test (2 trials)
  python p4_universal.py --mode time-split \\
      --data-dir <DIR> --run-name smoke --resolution 512 \\
      --n-calls 2 --n-initial 2

  # Disable brain mask
  python p4_universal.py --mode time-split --data-dir <DIR> \\
      --run-name no_mask --no-mask --resolution 512 --n-calls 10

  # Force a specific format
  python p4_universal.py --mode time-split --data-dir <DIR> \\
      --run-name forced --format multi-cam --resolution 512 --n-calls 10
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results"


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Universal CNMF pipeline with quality filters and 4 validation modes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=["time-split", "plane-split", "file-plane-split", "file-split"])
    p.add_argument("--run-name", required=True, help="Output folder name under results/")

    # Data sources
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Folder for time-split, plane-split")
    p.add_argument("--tune-dir", type=Path, default=None,
                   help="Tune folder for file-plane-split, file-split")
    p.add_argument("--test-dir", type=Path, default=None,
                   help="Test folder for file-plane-split, file-split")

    # Z-plane selection
    p.add_argument("--z-index", type=int, default=None,
                   help="Z-plane index for time-split, file-plane-split (default: middle)")
    p.add_argument("--tune-z", type=int, default=None,
                   help="Z-plane to tune on for plane-split (default: middle)")

    # Resolution / preprocessing
    p.add_argument("--resolution", choices=["full", "1024", "512"], default="512",
                   help="Spatial resolution (default 512)")
    p.add_argument("--no-mask", action="store_true",
                   help="Disable brain mask (default: mask ON)")
    p.add_argument("--no-stripe", action="store_true",
                   help="Disable column-median stripe removal")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap loaded frames at N (useful for huge single-movie files)")

    # Format override
    p.add_argument("--format", dest="format_override",
                   choices=["multi-tp", "multi-cam", "single-movie", "legacy"],
                   default=None, help="Override auto-detect format")

    # Bayesian search
    p.add_argument("--n-calls", type=int, default=10)
    p.add_argument("--n-initial", type=int, default=5)

    # Quality filter thresholds
    p.add_argument("--min-circularity", type=float, default=0.5,
                   help="Reject footprints with circularity below this (0-1)")
    p.add_argument("--max-area-factor", type=float, default=4.0,
                   help="Reject footprints with area > factor * pi * gSig^2")
    p.add_argument("--min-snr-trace", type=float, default=1.5,
                   help="Reject components with trace SNR below this")
    p.add_argument("--no-quality-filters", action="store_true",
                   help="Disable post-hoc quality filters (debug only)")

    return p.parse_args()


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
from skimage.morphology import convex_hull_image, binary_closing, binary_opening, disk, remove_small_objects
from skimage.filters import threshold_otsu
from scipy.optimize import linear_sum_assignment
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence

import caiman as cm
import caiman
import caiman.mmapping
import caiman.base.movies
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from caiman.source_extraction.cnmf import params as params_module

if not hasattr(cm, "load"):
    cm.load = caiman.base.movies.load
if not hasattr(cm, "movie"):
    cm.movie = caiman.base.movies.movie
if not hasattr(cm, "paths"):
    import caiman.paths


print("=" * 70)
print(f"p4_universal.py  |  mode={ARGS.mode}  |  run={ARGS.run_name}")
print("=" * 70)
print(f"Output dir : {OUTPUT_DIR}")
print(f"Resolution : {ARGS.resolution}")
print(f"Brain mask : {'OFF' if ARGS.no_mask else 'ON'}")
print(f"Quality    : {'OFF' if ARGS.no_quality_filters else 'ON'}")
print(f"Trials     : n_calls={ARGS.n_calls}  n_initial={ARGS.n_initial}")


# =============================================================================
# FORMAT DETECTION
# =============================================================================

def detect_format(folder: Path) -> tuple[str, list[str], tuple]:
    """
    Return (format_name, file_list, sample_shape).

    multi-tp     : many tp-*.lux.h5 files, each (Z, H, W) with Z>1
    multi-cam    : many Cam_long_*.lux.h5 files, each (1, H, W)
    single-movie : one big .lux.h5 file with (T, H, W)
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

    cam_files = sorted(
        glob.glob(str(folder / "Cam_long_*.lux.h5")),
        key=lambda p: int(re.search(r"Cam_long_(\d+)", p).group(1)),
    )
    if cam_files:
        with h5py.File(cam_files[0], "r") as fh:
            shape = tuple(fh["Data"].shape)
        if len(cam_files) == 1:
            return "single-movie", cam_files, shape
        if shape[0] == 1:
            return "multi-cam", cam_files, shape
        return "single-movie", cam_files, shape

    h5_files = sorted(glob.glob(str(folder / "*.h5")))
    if h5_files:
        with h5py.File(h5_files[0], "r") as fh:
            keys = list(fh.keys())
            if "Data" in keys:
                shape = tuple(fh["Data"].shape)
                return "legacy", h5_files, shape

    raise FileNotFoundError(f"No recognizable .lux.h5 / .h5 files in {folder}")


def discover(folder: Path, override: Optional[str] = None
             ) -> tuple[str, list[str], tuple]:
    """Detect or override format. Print result."""
    fmt, files, shape = detect_format(folder)
    if override and override != fmt:
        print(f"  Format override: detected={fmt} -> using={override}")
        fmt = override
    print(f"  Folder : {folder}")
    print(f"  Format : {fmt}  ({len(files)} file(s), sample shape={shape})")
    return fmt, files, shape


# =============================================================================
# UNIVERSAL LOADER
# =============================================================================

def load_movie(folder: Path, fmt: str, files: list[str], shape: tuple,
               z_index: Optional[int] = None,
               max_frames: Optional[int] = None) -> np.ndarray:
    """
    Build (T, H, W) float32 movie regardless of source format.

    For multi-tp: pick z_index plane from each timepoint file.
    For multi-cam: each file is one timepoint, use Data[0].
    For single-movie/legacy: read whole Data array (or chunks if huge).
    """
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
        T = T_full if max_frames is None else min(T_full, max_frames)
        print(f"  Loading {T}/{T_full} frames from single file...")
        with h5py.File(files[0], "r") as fh:
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


# =============================================================================
# PREPROCESSING
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
    col_median = np.median(data, axis=(0, 1), keepdims=True)
    cleaned = np.clip(data - col_median, 0, None).astype(np.float32)
    return cleaned, col_median


def make_brain_mask(data: np.ndarray, label: str = "") -> np.ndarray:
    """
    Build a binary mask of brain pixels using Otsu on the mean image.
    Cleans up with morphological opening/closing and keeps the largest blob.
    """
    mean_img = data.mean(axis=0)
    try:
        thr = threshold_otsu(mean_img)
    except Exception:
        thr = mean_img.mean() + mean_img.std()

    mask = mean_img > thr
    if mask.sum() < 100:
        print(f"  WARNING: Otsu mask is tiny ({mask.sum()} px). Falling back to no mask.")
        return np.ones_like(mask, dtype=bool)

    h, w = mask.shape
    se_radius = max(3, min(h, w) // 100)
    mask = binary_opening(mask, disk(se_radius))
    mask = binary_closing(mask, disk(se_radius * 2))
    mask = remove_small_objects(mask, min_size=max(200, (h * w) // 5000))

    if mask.sum() < 100:
        print(f"  WARNING: brain mask too small after cleanup. Disabling mask.")
        return np.ones_like(mask, dtype=bool)

    coverage = 100.0 * mask.sum() / mask.size
    print(f"  Brain mask coverage: {coverage:.1f}% of frame")
    return mask


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (data * mask[None, :, :]).astype(np.float32)


def preprocess_movie(data: np.ndarray, label: str = "") -> tuple[np.ndarray, dict]:
    """
    Full preprocessing pipeline.
    Returns (preprocessed_movie, metadata_dict)
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

    # Brain mask
    if not ARGS.no_mask:
        mask = make_brain_mask(data, label=label)
        data = apply_mask(data, mask)
        info["brain_mask_used"] = True
        info["mask_coverage_frac"] = float(mask.sum() / mask.size)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(data.mean(axis=0), cmap="gray")
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.0)
        ax.set_title(f"Brain mask overlay ({label or 'main'})")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"brain_mask_{label or 'main'}.png"), dpi=100)
        plt.close(fig)
    else:
        mask = np.ones(data.shape[1:], dtype=bool)
        info["brain_mask_used"] = False

    info["final_shape"] = tuple(data.shape)
    return data, mask, info


# =============================================================================
# CNMF CONFIG (resolution-aware)
# =============================================================================

def get_search_space() -> list:
    """Bayesian search space scaled to resolution."""
    if ARGS.resolution == "512":
        return [
            Integer(2, 5, name="gSig"),
            Integer(1, 4, name="gSig_filt"),
            Real(0.4, 0.85, name="min_corr"),
            Integer(3, 12, name="min_pnr"),
            Categorical([25, 40, 60, 80], name="rf"),
            Categorical([1, 2], name="p"),
        ]
    if ARGS.resolution == "1024":
        return [
            Integer(4, 10, name="gSig"),
            Integer(2, 8, name="gSig_filt"),
            Real(0.4, 0.85, name="min_corr"),
            Integer(3, 12, name="min_pnr"),
            Categorical([50, 80, 120, 160], name="rf"),
            Categorical([1, 2], name="p"),
        ]
    return [
        Integer(4, 16, name="gSig"),
        Integer(4, 16, name="gSig_filt"),
        Real(0.4, 0.85, name="min_corr"),
        Integer(3, 12, name="min_pnr"),
        Categorical([100, 160, 240, 320], name="rf"),
        Categorical([1, 2], name="p"),
    ]


def get_base_params() -> dict:
    if ARGS.resolution == "512":
        mc = dict(max_shifts=(3, 3), strides=(48, 48),
                  overlaps=(24, 24), max_deviation_rigid=2)
    elif ARGS.resolution == "1024":
        mc = dict(max_shifts=(6, 6), strides=(96, 96),
                  overlaps=(48, 48), max_deviation_rigid=3)
    else:
        mc = dict(max_shifts=(12, 12), strides=(192, 192),
                  overlaps=(96, 96), max_deviation_rigid=3)

    return {
        "fr": 5,
        "decay_time": 1.0,
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
        "ssub": 1,
        "tsub": 1,
        "only_init": False,
        "pw_rigid": True,
        **mc,
    }


SEARCH_SPACE = get_search_space()
PARAM_NAMES = [s.name for s in SEARCH_SPACE]
BASE_PARAMS = get_base_params()


# =============================================================================
# CNMF + QUALITY FILTERS
# =============================================================================

def array_to_memmap(array: np.ndarray, basename: Path) -> str:
    tif = str(basename) + ".tif"
    tifffile.imwrite(tif, array.astype(np.float32))
    return caiman.mmapping.save_memmap(
        [tif], base_name=str(basename), order="C", border_to_0=0,
    )


def run_cnmf(params_override: dict, fname_mmap: str,
             do_mc: bool = True, do_filter_caiman: bool = True):
    """Run CNMF + CaImAn's built-in evaluate_components / select_components."""
    p = {**BASE_PARAMS, **params_override, "fnames": [fname_mmap]}

    for key in ("gSig", "gSig_filt"):
        val = p.get(key)
        if val is None:
            continue
        if isinstance(val, tuple):
            p[key] = (int(val[0]), int(val[1]))
        else:
            p[key] = (int(val), int(val))

    g = p["gSig"]
    p["gSiz"] = (4 * int(g[0]) + 1, 4 * int(g[1]) + 1)

    opts = params_module.CNMFParams(params_dict=p)
    cnmf_obj = cnmf_module.CNMF(n_processes=1, params=opts)

    t0 = time.time()
    try:
        cnmf_obj.fit_file(motion_correct=do_mc)
        if do_filter_caiman and cnmf_obj.estimates.A.shape[1] > 0:
            try:
                Yr, dims, T_loc = caiman.mmapping.load_memmap(fname_mmap)
                images = np.reshape(Yr.T, [T_loc] + list(dims), order="F")
                cnmf_obj.estimates.evaluate_components(
                    imgs=images, params=cnmf_obj.params, dview=None)
                cnmf_obj.estimates.select_components(use_object=True)
            except Exception:
                pass
        return cnmf_obj, time.time() - t0
    except Exception as exc:
        print(f"    CNMF failed: {exc}")
        return None, time.time() - t0


def quality_filter(cnmf_obj, dims: tuple[int, int], mask: np.ndarray,
                   gSig: int) -> tuple[list[int], dict]:
    """
    Post-hoc quality filter. Returns (kept_indices, counts_log).

    Filters applied (in order):
      1. circularity >= ARGS.min_circularity
      2. area <= ARGS.max_area_factor * pi * gSig^2
      3. centroid inside brain mask
    """
    if cnmf_obj is None or cnmf_obj.estimates.A.shape[1] == 0:
        return [], {"input": 0, "circularity": 0, "max_area": 0,
                    "in_mask": 0, "final": 0}

    A = cnmf_obj.estimates.A
    n = A.shape[1]
    H, W = dims
    counts = {"input": n}

    if ARGS.no_quality_filters:
        return list(range(n)), {"input": n, "final": n}

    max_area = ARGS.max_area_factor * np.pi * gSig * gSig

    keep = []
    rej_circ = 0
    rej_area = 0
    rej_mask = 0

    for i in range(n):
        fp = np.asarray(A[:, i].todense()).flatten().reshape(H, W)
        if fp.max() <= 0:
            rej_circ += 1
            continue
        binary = fp > (fp.max() * 0.2)
        area = int(binary.sum())
        if area < 5:
            rej_circ += 1
            continue

        # Circularity = 4*pi*area / perimeter^2
        # Perimeter via 4-neighbor edges
        eroded = np.zeros_like(binary)
        eroded[1:-1, 1:-1] = (
            binary[1:-1, 1:-1] & binary[:-2, 1:-1] & binary[2:, 1:-1] &
            binary[1:-1, :-2] & binary[1:-1, 2:]
        )
        perimeter = max(int((binary & ~eroded).sum()), 1)
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < ARGS.min_circularity:
            rej_circ += 1
            continue

        if area > max_area:
            rej_area += 1
            continue

        ys, xs = np.nonzero(binary)
        cy, cx = int(ys.mean()), int(xs.mean())
        if 0 <= cy < H and 0 <= cx < W and not mask[cy, cx]:
            rej_mask += 1
            continue

        keep.append(i)

    counts["circularity_rejected"] = rej_circ
    counts["max_area_rejected"] = rej_area
    counts["in_mask_rejected"] = rej_mask
    counts["final"] = len(keep)
    return keep, counts


def score_run(cnmf_obj, Yr, dims: tuple[int, int], mask: np.ndarray,
              gSig: int, stability: float = 0.0
              ) -> tuple[dict, list[int], dict]:
    """
    Composite score uses FILTERED neuron count.

    composite = 1.0*(1 - recon_error)
              + 0.5*spatial_compactness
              - 0.3*log(1 + trace_sparsity)
              + 1.0*stability
              + 0.001*log(1 + n_filtered)   # small bonus for finding more real neurons

    Returns (metrics_dict, kept_indices, filter_counts)
    """
    sentinel = {
        "n_neurons_pre": 0, "n_neurons": 0, "recon_error": 1.0,
        "spatial_compactness": 0.0, "trace_sparsity": float("inf"),
        "stability": stability, "composite_score": -float("inf"),
    }
    if cnmf_obj is None:
        return sentinel, [], {"input": 0, "final": 0}

    keep, counts = quality_filter(cnmf_obj, dims, mask, gSig)
    n_pre = cnmf_obj.estimates.A.shape[1]
    n = len(keep)
    if n == 0:
        sentinel["n_neurons_pre"] = n_pre
        return sentinel, [], counts

    A = cnmf_obj.estimates.A[:, keep]
    C = cnmf_obj.estimates.C[keep, :]

    Y_hat = A @ C
    b = getattr(cnmf_obj.estimates, "b", None)
    f_bg = getattr(cnmf_obj.estimates, "f", None)
    if b is not None and f_bg is not None and b.shape[1] > 0:
        Y_hat = Y_hat + b @ f_bg
    recon_error = float(
        np.linalg.norm(Yr - Y_hat, "fro") / (np.linalg.norm(Yr, "fro") + 1e-9))

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
        "n_neurons_pre": n_pre,
        "n_neurons": n,
        "recon_error": recon_error,
        "spatial_compactness": spatial_compactness,
        "trace_sparsity": trace_sparsity,
        "stability": stability,
        "composite_score": float(composite),
    }, keep, counts


def compute_stability(A1, A2, threshold: float = 0.5) -> float:
    if A1 is None or A2 is None or A1.shape[1] == 0 or A2.shape[1] == 0:
        return 0.0
    n1 = np.asarray(np.sqrt(A1.power(2).sum(axis=0))).flatten() + 1e-9
    n2 = np.asarray(np.sqrt(A2.power(2).sum(axis=0))).flatten() + 1e-9
    corr = np.asarray((A1.multiply(1.0 / n1).T @ A2.multiply(1.0 / n2)).todense())
    ri, ci = linear_sum_assignment(-corr)
    return float(np.mean(corr[ri, ci] >= threshold))


# =============================================================================
# BAYESIAN TUNING
# =============================================================================

def bayesian_tune(mmap_path: str, dims: tuple[int, int],
                  mask: np.ndarray, tag: str = "tune"
                  ) -> tuple[dict, pd.DataFrame, list[dict]]:
    """Run Bayesian search. Returns (best_params, trials_df, filter_counts_per_trial)."""
    Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
    trial_log: list[dict] = []
    counts_log: list[dict] = []

    def objective(params):
        tp = dict(zip(PARAM_NAMES, params))
        tp["stride"] = tp["rf"] // 2
        num = len(trial_log) + 1
        print(f"  Trial {num:2d}: {tp} ...", end=" ", flush=True)

        cnmf_obj, rt = run_cnmf(tp, mmap_path)
        metrics, _, counts = score_run(cnmf_obj, Yr, dims, mask, gSig=int(tp["gSig"]))
        metrics.update(tp)
        metrics["runtime_s"] = round(rt, 1)
        trial_log.append(metrics)
        counts["trial"] = num
        counts_log.append(counts)

        n_pre = metrics.get("n_neurons_pre", 0)
        n_post = metrics.get("n_neurons", 0)
        print(f"raw={n_pre:3d} kept={n_post:3d} composite={metrics['composite_score']:+.4f} t={rt:.0f}s")
        score = -metrics["composite_score"]
        if not np.isfinite(score):
            score = 1e4
        return score

    print(f"\n{'='*60}\nBAYESIAN TUNE ({tag})\n{'='*60}")

    opt_result = gp_minimize(
        objective, SEARCH_SPACE,
        n_calls=ARGS.n_calls, n_initial_points=min(ARGS.n_initial, ARGS.n_calls),
        random_state=42, verbose=False,
    )

    df = pd.DataFrame(trial_log)
    df.to_csv(str(OUTPUT_DIR / f"tune_{tag}_log.csv"), index=False)

    df_counts = pd.DataFrame(counts_log)
    df_counts.to_csv(str(OUTPUT_DIR / f"quality_filters_{tag}_log.csv"), index=False)

    df_sorted = df.sort_values("composite_score", ascending=False)
    best = df_sorted.iloc[0]
    best_params = {
        "gSig": int(best["gSig"]),
        "gSig_filt": int(best["gSig_filt"]),
        "min_corr": float(best["min_corr"]),
        "min_pnr": int(best["min_pnr"]),
        "rf": int(best["rf"]),
        "stride": int(best["rf"]) // 2,
        "p": int(best["p"]),
        "merge_thr": 0.85,
    }

    print(f"\nBest params ({tag}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_convergence(opt_result, ax=ax)
    ax.set_title(f"Convergence — {tag}")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"convergence_{tag}.png"), dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, pname in zip(axes.flat, PARAM_NAMES):
        ax.scatter(df[pname], df["composite_score"], alpha=0.6, s=30)
        ax.set_xlabel(pname)
        ax.set_ylabel("Composite (post-filter)")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Param vs composite — {tag}")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"param_vs_score_{tag}.png"), dpi=120)
    plt.close(fig)

    if not df_counts.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        df_counts_plot = df_counts.fillna(0)
        ax.bar(df_counts_plot["trial"], df_counts_plot["input"], color="lightgray", label="input")
        ax.bar(df_counts_plot["trial"], df_counts_plot.get("final", 0), color="seagreen", label="kept")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Components")
        ax.set_title(f"Quality filter survival — {tag}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"quality_filters_{tag}.png"), dpi=120)
        plt.close(fig)

    return best_params, df, counts_log


# =============================================================================
# TEST PHASE
# =============================================================================

def test_cnmf(best_params: dict, mmap_path: str, data: np.ndarray,
              dims: tuple[int, int], mask: np.ndarray, label: str,
              tune_A=None) -> dict:
    """Apply tuned params to test data, run filters, save plots and traces."""
    print(f"\n{'='*60}\nTEST: {label}\n{'='*60}")

    Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
    cnmf_obj, rt = run_cnmf(best_params, mmap_path)

    stability = 0.0
    if tune_A is not None and cnmf_obj is not None:
        # Apply same quality filter to get matched A
        keep_pre, _ = quality_filter(cnmf_obj, dims, mask, int(best_params["gSig"]))
        if keep_pre:
            stability = compute_stability(tune_A, cnmf_obj.estimates.A[:, keep_pre])

    metrics, keep, counts = score_run(
        cnmf_obj, Yr, dims, mask,
        gSig=int(best_params["gSig"]), stability=stability,
    )
    metrics["label"] = label
    metrics["runtime_s"] = round(rt, 1)
    metrics["filter_counts"] = counts

    n_pre = metrics["n_neurons_pre"]
    n = metrics["n_neurons"]
    print(f"  Raw neurons: {n_pre}  kept: {n}  composite: {metrics['composite_score']:+.4f} stability: {stability:.3f}  t={rt:.0f}s")

    if cnmf_obj is not None and n > 0:
        safe = label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        H_val, W_val = dims

        # Contour plot
        mean_frame = data.mean(axis=0)
        fig, ax = plt.subplots(figsize=(11, 11))
        ax.imshow(mean_frame, cmap="gray")
        ax.set_title(f"{label}: {n} kept neurons (raw {n_pre})", fontsize=10)
        ax.axis("off")
        if not ARGS.no_mask:
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.5, alpha=0.5)
        for i in keep:
            fp = np.asarray(cnmf_obj.estimates.A[:, i].todense()).flatten().reshape(H_val, W_val)
            if fp.max() == 0:
                continue
            ax.contour(fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.4, alpha=0.85)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"contours_{safe}.png"), dpi=150)
        plt.close(fig)

        # Sample traces
        traces = cnmf_obj.estimates.C[keep]
        n_plot = min(8, traces.shape[0])
        T = data.shape[0]
        t_axis = np.arange(T)
        fig, axes_t = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
        if n_plot == 1:
            axes_t = [axes_t]
        for i, ax in enumerate(axes_t):
            ax.plot(t_axis, traces[i], lw=1)
            ax.set_ylabel(f"N{i}", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes_t[-1].set_xlabel("Frame")
        plt.suptitle(f"Traces — {label}")
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"traces_{safe}.png"), dpi=120)
        plt.close(fig)

        np.save(str(OUTPUT_DIR / f"traces_{safe}.npy"), traces)

    return metrics


# =============================================================================
# OUTPUT
# =============================================================================

def save_summary(mode: str, best_params: dict, test_results: dict,
                 fmt_info: dict, extra: dict | None = None):
    summary = {
        "mode": mode,
        "run_name": ARGS.run_name,
        "resolution": ARGS.resolution,
        "brain_mask": not ARGS.no_mask,
        "stripe_removal": not ARGS.no_stripe,
        "quality_filters": not ARGS.no_quality_filters,
        "n_calls": ARGS.n_calls,
        "best_params": best_params,
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
    headline_test = next(iter(test_results.values()), None) if test_results else None
    if isinstance(headline_test, dict):
        row = {
            "run_name": ARGS.run_name,
            "mode": mode,
            "resolution": ARGS.resolution,
            "brain_mask": not ARGS.no_mask,
            "n_neurons_kept": headline_test.get("n_neurons", 0),
            "n_neurons_raw": headline_test.get("n_neurons_pre", 0),
            "composite_score": headline_test.get("composite_score", float("nan")),
            "stability": headline_test.get("stability", 0.0),
            "gSig": best_params.get("gSig"),
            "min_corr": best_params.get("min_corr"),
            "min_pnr": best_params.get("min_pnr"),
            "rf": best_params.get("rf"),
            "p": best_params.get("p"),
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
    """Tune on first half of frames, test on second half. Same file/Z."""
    print(f"\n--- TIME-SPLIT mode ---")
    fmt, files, sample_shape = discover(ARGS.data_dir, ARGS.format_override)

    z_index = ARGS.z_index
    if fmt == "multi-tp":
        Z = sample_shape[0]
        z_index = Z // 2 if z_index is None else z_index

    raw = load_movie(ARGS.data_dir, fmt, files, sample_shape,
                     z_index=z_index, max_frames=ARGS.max_frames)
    data, mask, prep_info = preprocess_movie(raw, label="movie")

    T_full = data.shape[0]
    if T_full < 4:
        print(f"ERROR: only {T_full} frames; time-split needs >=4")
        sys.exit(1)
    mid = T_full // 2
    print(f"\nTime split: tune frames 0-{mid-1} ({mid}), test frames {mid}-{T_full-1} ({T_full-mid})")

    dims = data.shape[1:]
    tune_mmap = array_to_memmap(data[:mid], WORK_DIR / "tune_half")
    test_mmap = array_to_memmap(data[mid:], WORK_DIR / "test_half")

    best_params, _, _ = bayesian_tune(tune_mmap, dims, mask, tag="time_split")

    print("\nRe-running best params on tune half...")
    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_keep, _ = quality_filter(cnmf_tune, dims, mask, int(best_params["gSig"])) if cnmf_tune else ([], {})
    tune_A = cnmf_tune.estimates.A[:, tune_keep] if cnmf_tune and tune_keep else None

    test_metrics = test_cnmf(
        best_params, test_mmap, data[mid:], dims, mask,
        label=f"test_half (frames {mid}-{T_full-1})", tune_A=tune_A,
    )
    full_mmap = array_to_memmap(data, WORK_DIR / "full_movie")
    full_metrics = test_cnmf(
        best_params, full_mmap, data, dims, mask,
        label="full_movie", tune_A=tune_A,
    )

    fmt_info = {"format": fmt, "n_files": len(files), "sample_shape": list(sample_shape),
                **prep_info}
    save_summary("time-split", best_params,
                 {"test_half": test_metrics, "full_movie": full_metrics},
                 fmt_info,
                 extra={"z_index": z_index, "T_total": T_full,
                        "data_dir": str(ARGS.data_dir)})


def mode_plane_split():
    """Tune on one Z, test on every other Z. Requires multi-tp format."""
    print(f"\n--- PLANE-SPLIT mode ---")
    fmt, files, sample_shape = discover(ARGS.data_dir, ARGS.format_override)

    if fmt != "multi-tp":
        print(f"\nWARNING: plane-split requires multi-tp format. Detected: {fmt}")
        if fmt in ("multi-cam", "single-movie") and len(sample_shape) >= 1:
            z_dim = sample_shape[0]
            if z_dim == 1:
                print("This dataset has only Z=1. plane-split is impossible.")
                print("Use time-split or file-plane-split instead.")
                sys.exit(2)
        print("Continuing anyway (may fail).")

    Z = sample_shape[0] if fmt == "multi-tp" else 1
    if Z < 2:
        print(f"ERROR: plane-split needs Z>=2 (got {Z})")
        sys.exit(2)

    tune_z = ARGS.tune_z if ARGS.tune_z is not None else Z // 2
    if tune_z >= Z:
        print(f"ERROR: --tune-z {tune_z} >= Z={Z}")
        sys.exit(1)
    dims_native = sample_shape[1:]
    print(f"\nZ={Z}; tune on z={tune_z}, test on z=0..{Z-1}\\{{tune_z}}")

    tune_raw = load_plane_multi_tp(files, tune_z)
    tune_data, tune_mask, prep_info = preprocess_movie(tune_raw, label=f"tune_z{tune_z}")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / f"tune_z{tune_z}")

    best_params, _, _ = bayesian_tune(tune_mmap, dims, tune_mask, tag=f"z{tune_z}")

    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_keep, _ = quality_filter(cnmf_tune, dims, tune_mask, int(best_params["gSig"])) if cnmf_tune else ([], {})
    tune_A = cnmf_tune.estimates.A[:, tune_keep] if cnmf_tune and tune_keep else None

    all_metrics = {}
    for z in range(Z):
        print(f"\n--- z={z} ---")
        test_raw = load_plane_multi_tp(files, z)
        test_data, test_mask, _ = preprocess_movie(test_raw, label=f"test_z{z}")
        test_mmap = array_to_memmap(test_data, WORK_DIR / f"test_z{z}")
        m = test_cnmf(
            best_params, test_mmap, test_data, test_data.shape[1:], test_mask,
            label=f"z{z}" + (" (tune)" if z == tune_z else ""),
            tune_A=tune_A if z != tune_z else None,
        )
        all_metrics[f"z{z}"] = m

    rows = [
        {"z_plane": k, "n_neurons_kept": v["n_neurons"],
         "n_neurons_raw": v["n_neurons_pre"],
         "composite": v["composite_score"], "stability_vs_tune": v.get("stability", 0.0),
         "recon_error": v["recon_error"]}
        for k, v in all_metrics.items()
    ]
    pd.DataFrame(rows).to_csv(str(OUTPUT_DIR / "plane_split_summary.csv"), index=False)
    print("\n[plane-split summary]")
    print(pd.DataFrame(rows).to_string(index=False))

    fmt_info = {"format": fmt, "n_files": len(files), "sample_shape": list(sample_shape),
                **prep_info}
    save_summary("plane-split", best_params, all_metrics, fmt_info,
                 extra={"tune_z": tune_z, "Z": Z, "data_dir": str(ARGS.data_dir)})


def mode_file_plane_split():
    """Tune on file A z, test on file B same z."""
    print(f"\n--- FILE-PLANE-SPLIT mode ---")
    print("\nTune dataset:")
    fmt_t, tune_files, shape_t = discover(ARGS.tune_dir, ARGS.format_override)
    print("\nTest dataset:")
    fmt_te, test_files, shape_te = discover(ARGS.test_dir, ARGS.format_override)

    z_index = ARGS.z_index
    if fmt_t == "multi-tp" and z_index is None:
        z_index = shape_t[0] // 2
    if z_index is None:
        z_index = 0
    print(f"\nUsing z={z_index}")

    print("\n[Loading tune]")
    tune_raw = load_movie(ARGS.tune_dir, fmt_t, tune_files, shape_t,
                          z_index=z_index, max_frames=ARGS.max_frames)
    tune_data, tune_mask, prep_info_tune = preprocess_movie(tune_raw, label="tune")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / "tune")

    best_params, _, _ = bayesian_tune(tune_mmap, dims, tune_mask, tag="tune")

    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_keep, _ = quality_filter(cnmf_tune, dims, tune_mask, int(best_params["gSig"])) if cnmf_tune else ([], {})
    tune_A = cnmf_tune.estimates.A[:, tune_keep] if cnmf_tune and tune_keep else None

    print("\n[Loading test]")
    test_raw = load_movie(ARGS.test_dir, fmt_te, test_files, shape_te,
                          z_index=z_index, max_frames=ARGS.max_frames)
    test_data, test_mask, prep_info_test = preprocess_movie(test_raw, label="test")
    test_mmap = array_to_memmap(test_data, WORK_DIR / "test")

    test_metrics = test_cnmf(
        best_params, test_mmap, test_data, test_data.shape[1:], test_mask,
        label="test_file", tune_A=tune_A,
    )

    fmt_info = {"tune": {"format": fmt_t, "n_files": len(tune_files),
                          "sample_shape": list(shape_t), **prep_info_tune},
                "test": {"format": fmt_te, "n_files": len(test_files),
                          "sample_shape": list(shape_te), **prep_info_test}}
    save_summary("file-plane-split", best_params, {"test_file": test_metrics},
                 fmt_info,
                 extra={"z_index": z_index,
                        "tune_dir": str(ARGS.tune_dir),
                        "test_dir": str(ARGS.test_dir)})


def mode_file_split():
    """Tune on file A (one Z), test on file B at every Z if multi-tp; else single test."""
    print(f"\n--- FILE-SPLIT mode ---")
    print("\nTune dataset:")
    fmt_t, tune_files, shape_t = discover(ARGS.tune_dir, ARGS.format_override)
    print("\nTest dataset:")
    fmt_te, test_files, shape_te = discover(ARGS.test_dir, ARGS.format_override)

    tune_z = (shape_t[0] // 2) if fmt_t == "multi-tp" else 0
    print(f"\nTuning on tune-file z={tune_z}")

    print("\n[Loading tune]")
    tune_raw = load_movie(ARGS.tune_dir, fmt_t, tune_files, shape_t,
                          z_index=tune_z, max_frames=ARGS.max_frames)
    tune_data, tune_mask, prep_info_tune = preprocess_movie(tune_raw, label=f"tune_z{tune_z}")
    dims = tune_data.shape[1:]
    tune_mmap = array_to_memmap(tune_data, WORK_DIR / "tune")

    best_params, _, _ = bayesian_tune(tune_mmap, dims, tune_mask, tag="tune")

    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_keep, _ = quality_filter(cnmf_tune, dims, tune_mask, int(best_params["gSig"])) if cnmf_tune else ([], {})
    tune_A = cnmf_tune.estimates.A[:, tune_keep] if cnmf_tune and tune_keep else None

    if fmt_te == "multi-tp":
        Z_te = shape_te[0]
        z_iter = range(Z_te)
    else:
        Z_te = 1
        z_iter = [0]

    all_metrics = {}
    for z in z_iter:
        label = f"test_z{z}" if Z_te > 1 else "test_file"
        print(f"\n--- {label} ---")
        if fmt_te == "multi-tp":
            test_raw = load_plane_multi_tp(test_files, z)
        else:
            test_raw = load_movie(ARGS.test_dir, fmt_te, test_files, shape_te,
                                  z_index=z, max_frames=ARGS.max_frames)
        test_data, test_mask, _ = preprocess_movie(test_raw, label=label)
        test_mmap = array_to_memmap(test_data, WORK_DIR / label)
        m = test_cnmf(
            best_params, test_mmap, test_data, test_data.shape[1:], test_mask,
            label=label, tune_A=tune_A,
        )
        all_metrics[label] = m

    if Z_te > 1:
        rows = [
            {"z_plane": k, "n_neurons_kept": v["n_neurons"],
             "n_neurons_raw": v["n_neurons_pre"],
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
    save_summary("file-split", best_params, all_metrics, fmt_info,
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
