"""
new_pipeline.py  —  Universal CNMF Pipeline

Pipeline includes:
  1. External Parameters (loading best_params.json)
  2. Interleaved Format Detection
  3. Execution Update (motion correction, .fit() & .refit())
  4. Cluster & Multiprocessing (CaImAn cluster setup with n_workers)
"""

# Must be the first lines of the file, before ALL imports
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"


import argparse
import glob
import json
import re
import sys
import multiprocessing
import time
import warnings
from pathlib import Path
from typing import Optional

import cv2

try:
    cv2.setNumThreads(0)
except:
    pass

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
    p.add_argument(
        "--mode",
        choices=["single-plane", "all-planes"],
        default="single-plane",
        help="Run for a single plane or all planes",
    )
    p.add_argument(
        "--run-name",
        required=True,
        default=None,
        help="Output folder name under results/",
    )

    # Data sources
    p.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        default=None,
        help="Folder for run",
    )

    # Z-plane selection
    p.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="Z-plane index for single-plane mode (default: middle)",
    )

    p.add_argument(
        "--n-planes",
        type=int,
        default=None,
        help="Number of Z-planes in a single-movie file",
    )

    # Resolution / preprocessing
    p.add_argument(
        "--resolution",
        choices=["full", "1024", "512"],
        default="512",
        help="Spatial resolution (default 512)",
    )
    p.add_argument(
        "--mask", action="store_true", help="Disable brain mask (default: mask ON)"
    )
    p.add_argument(
        "--no-stripe", action="store_true", help="Disable column-median stripe removal"
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap loaded frames at N (useful for huge single-movie files)",
    )

    # Format override
    p.add_argument(
        "--format",
        dest="format_override",
        choices=["multi-tp", "multi-cam", "single-movie", "stacked", "legacy"],
        default=None,
        help="Override auto-detect format",
    )

    
    p.add_argument(
        "--min-snr-trace",
        type=float,
        default=1.5,
        help="Reject components with trace SNR below this",
    )

    p.add_argument(
        "--best-params-path",
        type=Path,
        default=None,
        help="Path to best_params.json to load external parameters",
    )

    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="CPU workers for CNMF patch processing (default: cpu_count - 1)",
    )

    return p.parse_args()


ARGS = parse_args()


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
from skimage.morphology import (
    convex_hull_image,
    binary_closing,
    binary_opening,
    disk,
    remove_small_objects,
)
from skimage.filters import threshold_otsu
from scipy.optimize import linear_sum_assignment

import caiman as cm
import caiman
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
# EXTERNAL PARAMETERS 
# =============================================================================


def load_best_params(path: Path) -> dict:
    """Load external parameters to avoid hardcoding defaults."""
    if not path.is_file():
        print(f"WARNING: best-params file not found at {path} — using base defaults.")
        return {}
    try:
        with open(path) as fh:
            raw = json.load(fh)
        best = raw.get("best_params", raw) if isinstance(raw, dict) else {}
        print(f"Loaded best params from {path}: {list(best.keys())}")
        return best
    except Exception as exc:
        print(f"WARNING: failed to read best-params file at {path} ({exc}).")
        return {}


BEST_PARAMS_PATH = ARGS.best_params_path or (SCRIPT_DIR / "best_params.json")
LOADED_BEST_PARAMS = load_best_params(BEST_PARAMS_PATH)


print("=" * 70)
print(f"p4_universal.py  |  mode={ARGS.mode}  |  run={ARGS.run_name}")
print("=" * 70)
print(f"Output dir : {OUTPUT_DIR}")
print(f"Resolution : {ARGS.resolution}")
print(f"Brain mask : {'ON' if ARGS.mask else 'OFF'}")


# =============================================================================
# FORMAT DETECTION
# =============================================================================


def read_n_planes(filepath: str) -> int:
    """Read n_planes from metadata in a .lux*.h5 file; fall back to 1."""
    try:
        with h5py.File(filepath, "r") as fh:
            if "metadata" not in fh:
                return 1
            raw = fh["metadata"][()]
            meta = json.loads(raw)
            return int(meta["metaData"]["stack"]["n"])
    except Exception as e:
        print(
            f"  WARNING: read_n_planes failed ({e}); defaulting to 1 (treating as single-plane)"
        )
        return 1


def detect_format(folder: Path) -> tuple[str, list[str], tuple]:
    """
    Return (format_name, file_list, sample_shape). Includes 'interleaved' detection.
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
        # Check interleaved based on n_planes
        if read_n_planes(cam_files[0]) > 1:
            return "stacked", cam_files, shape
        return "single-movie", cam_files, shape

    lux_files = sorted(glob.glob(str(folder / "*.lux*.h5")))
    if lux_files:
        with h5py.File(lux_files[0], "r") as fh:
            if "Data" in fh:
                shape = tuple(fh["Data"].shape)
                if read_n_planes(lux_files[0]) > 1:
                    return "stacked", lux_files, shape
                return "single-movie", lux_files, shape

    h5_files = sorted(glob.glob(str(folder / "*.h5")))
    if h5_files:
        with h5py.File(h5_files[0], "r") as fh:
            keys = list(fh.keys())
            if "Data" in keys:
                shape = tuple(fh["Data"].shape)
                return "legacy", h5_files, shape

    raise FileNotFoundError(f"No recognizable .lux.h5 / .h5 files in {folder}")


def discover(
    folder: Path,
    override: Optional[str] = None,
    n_planes_override: Optional[int] = None,
) -> tuple[str, list[str], tuple, Optional[int]]:
    """Detect or override format. Print result."""
    fmt, files, shape = detect_format(folder)
    if override and override != fmt:
        print(f"  Format override: detected={fmt} -> using={override}")
        fmt = override

    detected_n_planes = None
    if fmt == "stacked":
        detected_n_planes = (
            n_planes_override if n_planes_override else read_n_planes(files[0])
        )
        print(f"  Auto-detected n_planes={detected_n_planes}")

    print(f"  Folder : {folder}")
    print(f"  Format : {fmt}  ({len(files)} file(s), sample shape={shape})")
    return fmt, files, shape, detected_n_planes


# =============================================================================
# UNIVERSAL LOADER
# =============================================================================


def load_movie(
    folder: Path,
    fmt: str,
    files: list[str],
    shape: tuple,
    z_index: Optional[int] = None,
    max_frames: Optional[int] = None,
    n_planes: Optional[int] = None,
) -> np.ndarray:
    """Build (T, H, W) float32 movie regardless of source format."""
    if fmt == "stacked":
        T_full, H, W = shape
        nz = n_planes or 1
        if z_index is None:
            z_index = nz // 2
        if z_index >= nz:
            raise ValueError(f"z_index {z_index} >= n_planes {nz}")
        frames_per_plane = T_full // nz
        
        if max_frames:
            frames_per_plane = min(frames_per_plane, max_frames)
        print(f"  Stacked: z={z_index}/{nz} planes -> {frames_per_plane} frames")
        with h5py.File(files[0], "r") as fh:
            data = fh["Data"][:].astype(np.float32)
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
        T = T_full if max_frames is None else min(T_full, max_frames)
        print(f"  Loading {T}/{T_full} frames from single file...")
        with h5py.File(files[0], "r") as fh:
            data = fh["Data"][:T].astype(np.float32)
        return data

    raise ValueError(f"Unknown format: {fmt}")


# =============================================================================
# PREPROCESSING
# =============================================================================

def stripe_remove(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    col_median = np.median(data, axis=(0, 1), keepdims=True)
    cleaned = np.clip(data - col_median, 0, None).astype(np.float32)
    return cleaned, col_median

def make_brain_mask(data: np.ndarray, label: str = "") -> np.ndarray:
    """
    Build a binary mask of brain pixels using Otsu on the mean image.
    Cleans up with morphological opening/closing and keeps the largest blob.
    """
    ref_img = np.percentile(data, 20, axis=0)  # low percentile ≈ baseline, motion-robust
    
    try:
        thr = threshold_otsu(ref_img)
    except Exception:
        pass
        
    mask = ref_img > thr
    
    if mask.sum() < 100:
        print(
            f"  WARNING: Otsu mask is tiny ({mask.sum()} px). Falling back to no mask."
        )
        return np.ones_like(mask, dtype=bool)

    from scipy.ndimage import binary_fill_holes
    mask = binary_fill_holes(mask)

    # 4. Tie structuring element to pixel size
    um_per_px = 0.208
    se_radius = max(3, int(10 / um_per_px))  # ~10 µm in pixel units

    mask = binary_opening(mask, disk(se_radius))
    mask = binary_closing(mask, disk(se_radius * 2))
    mask = binary_fill_holes(mask)  # again after closing
    mask = remove_small_objects(mask, min_size=int(500 / um_per_px**2))
    
    if mask.sum() < 100:
        print(f"  WARNING: brain mask too small after cleanup. Disabling mask.")
        return np.ones_like(mask, dtype=bool)

    coverage = 100.0 * mask.sum() / mask.size
    print(f"  Brain mask coverage: {coverage:.1f}% of frame")
    return mask


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (data * mask[None, :, :]).astype(np.float32)


def preprocess_movie(
    data: np.ndarray, label: str = ""
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Full preprocessing pipeline. Returns (preprocessed_movie, mask, metadata_dict)."""
    info = {"original_shape": tuple(data.shape)}
    print(f"\n[preprocess {label}]  input shape={data.shape}")

    # Downsample if necessary
    res = ARGS.resolution
    if data.shape[1] != res:
        ratio = res / data.shape[1]

        print(f"  Downsampling -> {(res, res)}")
        data = data.resize(fx=ratio, fy=ratio, fz=1.0)
        info["downsampled_to"] = (res, res)

    if not ARGS.no_stripe:
        data, col_median = stripe_remove(data)
        
        data = cm.movie(data)

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

    if not ARGS.no_mask:
        mask = make_brain_mask(data, label=label)
        data = cm.movie(apply_mask(data, mask))
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


def get_base_params(external_params: dict = None) -> dict:
    """ Merges external best_params JSON with hardcoded ones."""
    external = external_params or {}

    if ARGS.resolution == "512":
        mc = dict(
            max_shifts=(3, 3),
            strides=(48, 48),
            overlaps=(24, 24),
            max_deviation_rigid=2,
        )
    elif ARGS.resolution == "1024":
        mc = dict(
            max_shifts=(6, 6),
            strides=(96, 96),
            overlaps=(48, 48),
            max_deviation_rigid=3,
        )
    else:
        mc = dict(
            max_shifts=(12, 12),
            strides=(192, 192),
            overlaps=(96, 96),
            max_deviation_rigid=3,
        )

    base = {
        "fr": external.get("fr", 5),
        "decay_time": external.get("decay_time", 1.0),
        "method_init": external.get("method_init", "corr_pnr"),
        "K": external.get("K", None),
        "nb": external.get("nb", 0),
        "nb_patch": external.get("nb_patch", 0),
        "center_psf": external.get("center_psf", True),
        "ring_size_factor": external.get("ring_size_factor", 1.4),
        "merge_thr": external.get("merge_thr", 0.85),
        "use_cnn": external.get("use_cnn", False),
        "min_SNR": ARGS.min_snr_trace,
        "rval_thr": external.get("rval_thr", 0.85),
        "del_duplicates": external.get("del_duplicates", True),
        "ssub": external.get("ssub", 1),
        "tsub": external.get("tsub", 1),
        "only_init": external.get("only_init", False),
        "pw_rigid": external.get("pw_rigid", True),
        "border_nan": "copy",
        **mc,
        **external,
    }

    print(
        "  Base CNMF params:", {k: v for k, v in base.items() if k not in ("fnames",)}
    )
    return base


BASE_PARAMS = get_base_params(LOADED_BEST_PARAMS)


# =============================================================================
# CNMF + QUALITY FILTERS [UPGRADE 3 & 4]
# =============================================================================


def array_to_memmap(array: np.ndarray, basename: Path) -> str:
    tif = str(basename) + ".tif"
    tifffile.imwrite(tif, array.astype(np.float32))
    return caiman.mmapping.save_memmap(
        [tif],
        base_name=str(basename),
        order="C",
        border_to_0=0,
    )


def _setup_cluster(nworkers=N_WORKERS):
    try:
        _, cluster, n_processes = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=nworkers, single_thread=False
        )
        return cluster, n_processes
    except Exception as exc:
        print(f"  [STAGE:cluster_setup] failed: {exc}", flush=True)
        return None, 1


def _stop_cluster(cluster):
    # Sends the termination signal to the workers
    cm.stop_server(dview=cluster)

    # Synchronously blocks the script until every single worker is dead
    for process in multiprocessing.active_children():
        process.join()


def _prep_params(params_override: dict, fname_mmap: str) -> "params_module.CNMFParams":
    """Stage: build CNMFParams from overrides. Normalizes gSig/gSig_filt to int tuples."""
    print("  [STAGE:prepping parameters] starting", flush=True)
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

    print("  [STAGE:prepping parameters] done", flush=True)

    return params_module.CNMFParams(params_dict=p)


def _motion_correct(fname_mmap: str, opts, cluster) -> str:
    """Stage: motion correction. Raises on failure — caller decides how to handle."""
    print("  [STAGE:motion_correction] starting", flush=True)
    mc = MotionCorrect([fname_mmap], dview=cluster, **opts.get_group("motion"))
    mc.motion_correct(save_movie=True)
    fname_to_use = cm.save_memmap(
        mc.mmap_file,
        base_name="memmap_",
        order="C",
        border_to_0=0,  # exclude borders, if that was done
        dview=cluster,
    )
    print(f"  [STAGE:motion_correction] done -> {fname_to_use}", flush=True)
    return fname_to_use


def _fit_cnmf(fname_to_use: str, opts, n_processes: int, cluster):
    """Stage: core CNMF fit. Raises on failure — this is the critical path."""
    print(f"[LOG] run_cnmf: Initializing CNMF object...")
    opts.change_params({"fnames": [fname_to_use]})
    cnmf_obj = cnmf_module.CNMF(n_processes=n_processes, params=opts, dview=cluster)

    print(f"[LOG] run_cnmf: Loading memmap...")
    Yr, dims, num_frames = cm.load_memmap(fname_to_use)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")

    print("[STAGE:fit] starting", flush=True)
    cnmf_obj.fit(images)
    print(f"[LOG] run_cnmf: .fit() completed")

    n_components = (
        cnmf_obj.estimates.A.shape[1] if cnmf_obj.estimates.A is not None else 0
    )
    print(f"[STAGE:fit] done -> {n_components} components", flush=True)

    return cnmf_obj


def _reload_images(fname_to_use: str):
    """Stage: reload the mmap actually fit, for evaluate/refit. Returns None on failure
    (non-fatal — evaluate/refit are simply skipped downstream)."""
    print(f"[STAGE:reloading images] starting...", flush=True)
    try:
        Yr, dims, T_loc = caiman.mmapping.load_memmap(fname_to_use)
        return np.reshape(Yr.T, [T_loc] + list(dims), order="F")
    except Exception as exc:
        print(
            f"  [STAGE:reload_images] failed, evaluate/refit will be skipped: {exc}",
            flush=True,
        )
        return None


def _evaluate_and_select(cnmf_obj, images, cluster):
    """Stage: CaImAn's own component evaluation (SNR/spatial/CNN). Non-fatal on failure —
    keeps whatever components existed before this stage."""
    try:
        print("  [STAGE:evaluate_components] starting", flush=True)
        cnmf_obj.estimates.evaluate_components(
            imgs=images, params=cnmf_obj.params, dview=cluster
        )
        cnmf_obj.estimates.select_components(use_object=True)
        print(
            f"  [STAGE:evaluate_components] done -> {cnmf_obj.estimates.A.shape[1]} components remain",
            flush=True,
        )
    except Exception as exc:
        print(
            f"  [STAGE:evaluate_components] failed, keeping unfiltered components: {exc}",
            flush=True,
        )


def _refit(cnmf_obj, images, cluster):
    """Stage: second-pass validation refit. Non-fatal on failure — keeps pre-refit estimates."""
    try:
        print(
            "  [STAGE:refit] starting — second-pass validation of accepted neurons",
            flush=True,
        )
        refit_obj = cnmf_obj.refit(images, dview=cluster)
        print(
            f"  [STAGE:refit] done -> {refit_obj.estimates.A.shape[1]} neurons remain",
            flush=True,
        )
        return refit_obj
    except Exception as exc:
        print(f"  [STAGE:refit] failed, keeping pre-refit estimates: {exc}", flush=True)
        return cnmf_obj


def run_cnmf(
    params_override: dict,
    fname_mmap: str,
    do_mc: bool = True,
    do_filter_caiman: bool = True,
    do_refit: bool = True,
):
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
    cluster = None

    t0 = time.time()
    try:
        fname_to_use = fname_mmap
        if do_mc:
            try:
                cluster, n_processes = _setup_cluster(10)
                fname_to_use = _motion_correct(fname_mmap, opts, cluster)
            finally:
                if cluster is not None:
                    _stop_cluster(cluster)
                    cluster = None  # Reset so the master finally block ignores it

        else:
            print(
                "  [STAGE:motion_correction] skipped — using precomputed mmap",
                flush=True,
            )

        cluster, n_processes = _setup_cluster()
        cnmf_obj = _fit_cnmf(fname_to_use, opts, n_processes, cluster)

        images = None
        if cnmf_obj.estimates.A.shape[1] > 0:
            images = _reload_images(fname_to_use)

        if do_refit and images is not None and cnmf_obj.estimates.A.shape[1] > 0:
            cnmf_obj = _refit(cnmf_obj, images, cluster)

        if do_filter_caiman and images is not None:
            _evaluate_and_select(cnmf_obj, images, cluster)

        _stop_cluster(cluster)
        cluster = None
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
                _stop_cluster(cluster)
            except Exception:
                pass




def compute_stability(A1, A2, threshold: float = 0.5) -> float:
    if A1 is None or A2 is None or A1.shape[1] == 0 or A2.shape[1] == 0:
        return 0.0
    n1 = np.asarray(np.sqrt(A1.power(2).sum(axis=0))).flatten() + 1e-9
    n2 = np.asarray(np.sqrt(A2.power(2).sum(axis=0))).flatten() + 1e-9
    corr = np.asarray((A1.multiply(1.0 / n1).T @ A2.multiply(1.0 / n2)).todense())
    ri, ci = linear_sum_assignment(-corr)
    return float(np.mean(corr[ri, ci] >= threshold))


# =============================================================================
# TEST PHASE
# =============================================================================


# def test_cnmf(
#     best_params: dict,
#     mmap_path: str,
#     data: np.ndarray,
#     dims: tuple[int, int],
#     mask: np.ndarray,
#     label: str,
#     tune_A=None,
# ) -> dict:
#     print(f"\n{'='*60}\nTEST: {label}\n{'='*60}")

#     Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
#     cnmf_obj, rt = run_cnmf(best_params, mmap_path)

#     stability = 0.0
#     if tune_A is not None and cnmf_obj is not None:
#         keep_pre, _ = quality_filter(cnmf_obj, dims, mask, int(best_params["gSig"]))
#         if keep_pre:
#             stability = compute_stability(tune_A, cnmf_obj.estimates.A[:, keep_pre])

#     metrics, keep, counts = score_run(
#         cnmf_obj,
#         Yr,
#         dims,
#         mask,
#         gSig=int(best_params["gSig"]),
#         stability=stability,
#     )
#     metrics["label"] = label
#     metrics["runtime_s"] = round(rt, 1)
#     metrics["filter_counts"] = counts

#     n_pre = metrics["n_neurons_pre"]
#     n = metrics["n_neurons"]
#     print(
#         f"  Raw neurons: {n_pre}  kept: {n}  composite: {metrics['composite_score']:+.4f} stability: {stability:.3f}  t={rt:.0f}s"
#     )

#     if cnmf_obj is not None and n > 0:
#         safe = (
#             label.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
#         )
#         H_val, W_val = dims

#         # Contour plot
#         mean_frame = data.mean(axis=0)
#         fig, ax = plt.subplots(figsize=(11, 11))
#         ax.imshow(mean_frame, cmap="gray")
#         ax.set_title(f"{label}: {n} kept neurons (raw {n_pre})", fontsize=10)
#         ax.axis("off")
#         if not ARGS.no_mask:
#             ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.5, alpha=0.5)
#         for i in keep:
#             fp = (
#                 np.asarray(cnmf_obj.estimates.A[:, i].todense())
#                 .flatten()
#                 .reshape(H_val, W_val)
#             )
#             if fp.max() == 0:
#                 continue
#             ax.contour(
#                 fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.4, alpha=0.85
#             )
#         plt.tight_layout()
#         plt.savefig(str(OUTPUT_DIR / f"contours_{safe}.png"), dpi=150)
#         plt.close(fig)

#         # Sample traces
#         traces = cnmf_obj.estimates.C[keep]
#         n_plot = min(8, traces.shape[0])
#         T = data.shape[0]
#         t_axis = np.arange(T)
#         fig, axes_t = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
#         if n_plot == 1:
#             axes_t = [axes_t]
#         for i, ax in enumerate(axes_t):
#             ax.plot(t_axis, traces[i], lw=1)
#             ax.set_ylabel(f"N{i}", fontsize=8)
#             ax.grid(True, alpha=0.3)
#         axes_t[-1].set_xlabel("Frame")
#         plt.suptitle(f"Traces — {label}")
#         plt.tight_layout()
#         plt.savefig(str(OUTPUT_DIR / f"traces_{safe}.png"), dpi=120)
#         plt.close(fig)

#         np.save(str(OUTPUT_DIR / f"traces_{safe}.npy"), traces)

#     return metrics


# =============================================================================
# OUTPUT
# =============================================================================


def save_summary(
    mode: str,
    best_params: dict,
    test_results: dict,
    fmt_info: dict,
    extra: dict | None = None,
):
    summary = {
        "mode": mode,
        "run_name": ARGS.run_name,
        "resolution": ARGS.resolution,
        "brain_mask": not ARGS.no_mask,
        "stripe_removal": not ARGS.no_stripe,
        "quality_filters": not ARGS.no_quality_filters,
        "best_params": best_params,
        "format_info": fmt_info,
        "tests": {},
    }
    for label, m in test_results.items():
        if isinstance(m, dict):
            summary["tests"][label] = {
                k: v for k, v in m.items() if not isinstance(v, (np.ndarray,))
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


# def mode_time_split():
#     print(f"\n--- TIME-SPLIT mode ---")
#     fmt, files, sample_shape, det_n_planes = discover(
#         ARGS.data_dir, ARGS.format_override, ARGS.n_planes
#     )

#     z_index = ARGS.z_index
#     if fmt == "multi-tp":
#         Z = sample_shape[0]
#         z_index = Z // 2 if z_index is None else z_index

#     # Data Loading Stage
#     print(f"[STAGE] Loading movie data (Format: {fmt})...")
#     raw = load_movie(
#         ARGS.data_dir,
#         fmt,
#         files,
#         sample_shape,
#         z_index=z_index,
#         max_frames=ARGS.max_frames,
#         n_planes=det_n_planes,
#     )

#     # Preprocessing Stage
#     print(f"[STAGE] Running preprocessing (masking, stripe removal, downsampling)...")
#     data, mask, prep_info = preprocess_movie(raw, label="movie")

#     T_full = data.shape[0]
#     if T_full < 4:
#         print(f"ERROR: only {T_full} frames; time-split needs >=4")
#         sys.exit(1)

#     mid = T_full // 2

#     print(
#         f"[INFO] Time split established: tune on frames 0-{mid-1}, total test set: {T_full} frames."
#     )

#     dims = data.shape[1:]

#     # Memory Mapping Stage
#     print(f"[STAGE] Creating memory maps for processing...")
#     tune_mmap = array_to_memmap(data[:mid], WORK_DIR / "tune_half")

#     print("\nRe-running best params on tune half...")

#     # Tuning Stage
#     print(f"[STAGE] Running initial CNMF on tuning half for stability metrics...")
#     cnmf_tune, _ = run_cnmf(BASE_PARAMS, tune_mmap)

#     gSig_val = BASE_PARAMS["gSig"]
#     gSig_int = (
#         int(gSig_val[0]) if isinstance(gSig_val, (tuple, list)) else int(gSig_val)
#     )

#     tune_keep, _ = (
#         quality_filter(cnmf_tune, dims, mask, gSig_int) if cnmf_tune else ([], {})
#     )
#     tune_A = cnmf_tune.estimates.A[:, tune_keep] if cnmf_tune and tune_keep else None

#     # Full Pipeline Execution Stage
#     print(f"[STAGE] Running final CNMF pipeline on full dataset...")
#     full_mmap = array_to_memmap(data, WORK_DIR / "full_movie")
#     full_metrics = test_cnmf(
#         BASE_PARAMS,
#         full_mmap,
#         data,
#         dims,
#         mask,
#         label="full_movie",
#         tune_A=tune_A,
#     )

#     # Reporting Stage
#     print(f"[STAGE] Finalizing run and saving summary...")
#     fmt_info = {
#         "format": fmt,
#         "n_files": len(files),
#         "sample_shape": list(sample_shape),
#         **prep_info,
#     }

#     save_summary(
#         "time-split",
#         BASE_PARAMS,
#         {"full_movie": full_metrics},
#         fmt_info,
#         extra={"z_index": z_index, "T_total": T_full, "data_dir": str(ARGS.data_dir)},
#     )

#     print(f"[STAGE] Time-split mode completed successfully.")


def mode_single_plane():
    print(f"\n--- SINGLE-PLANE mode ---") 
    fmt, files, sample_shape, det_n_planes = discover(
        ARGS.data_dir, ARGS.format_override, ARGS.n_planes
    )
    T, _, _ = sample_shape
    # frames_per_plane = T // det_n_planes
    n_frames = T if ARGS.max_frames is None else (ARGS.max_frames * det_n_planes)
    
    # Data Loading Stage
    print(f"[STAGE] Loading movie data (Format: {fmt})...")
    movie_orig = cm.load(files[0], var_name_hdf5='Data', subindices=slice(0, n_frames, 7)) # TODO assumes interleaved format. edit to include discover and other formats as well
    
    # Preprocessing Stage
    print(f"[STAGE] Running preprocessing (masking, stripe removal, downsampling)...")
    data, mask, prep_info = preprocess_movie(movie_orig, label="movie")
    
    T_full = data.shape[0]

    print(
        f"[INFO] Single plane established: Run with {T_full} frames."
    )

    dims = data.shape[1:]

    # Memory Mapping Stage
    print(f"[STAGE] Creating memory maps for processing...")
    processed_mmap = array_to_memmap(data, WORK_DIR / "full_movie")

    z = det_n_planes // 2 if ARGS.z_index is None else ARGS.z_index
    
    print(f"[STAGE] Running CNMF on plane {z}")
    cnmf_obj, _, fname = run_cnmf(BASE_PARAMS, processed_mmap)

    # Full Pipeline Execution Stage
    print(f"[STAGE] Getting traces for full dataset...")
    
    # TODO fill out the rest of this pipeline with traces


    # Reporting Stage
    print(f"[STAGE] Finalizing run and saving summary...")
    # fmt_info = {
    #     "format": fmt,
    #     "n_files": len(files),
    #     "sample_shape": list(sample_shape),
    #     **prep_info,
    # }

    # save_summary(
    #     "time-split",
    #     BASE_PARAMS,
    #     {"full_movie": full_metrics},
    #     fmt_info,
    #     extra={"z_index": z_index, "T_total": T_full, "data_dir": str(ARGS.data_dir)},
    # )

    print(f"[STAGE] Single-Plane mode completed successfully.")
    

def mode_all_planes():
    print(f"\n--- ALL-PLANES mode ---") # TODO assumes interleaved format. edit to include discover and other formats as well
    fmt, files, sample_shape, det_n_planes = discover(
        ARGS.data_dir, ARGS.format_override, ARGS.n_planes
    )


MODES = {
    "single-plane": mode_single_plane,
    "all-planes": mode_all_planes,
}

if __name__ == "__main__":
    t_start = time.time()
    MODES[ARGS.mode]()
    elapsed_min = (time.time() - t_start) / 60.0

    print(f"\n{'='*70}")
    print(f"DONE  |  {ARGS.mode}  |  {ARGS.run_name}  |  {elapsed_min:.1f} min")
    print(f"{'='*70}")
