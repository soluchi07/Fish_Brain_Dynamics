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
from scipy.ndimage import binary_fill_holes
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
        description="Universal CaImAn CNMF pipeline (motion correction optional, preprocessing "
        "with downsampling/striping-removal/brain-masking, patch-wise CNMF, "
        "refit, component evaluation and DF/F0).",
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
        help="Path to the raw movie (e.g. an HDF5 file/folder)",
    )
    p.add_argument(
        "--var-name-hdf5",
        default="Data",
        help="Variable name to load from the HDF5 file",
    )
    p.add_argument(
        "--save-path",
        default="pipeline_results.hdf5",
        help="Filename (within --output-dir) for the saved CNMF results",
    )
    p.add_argument(
        "--save-traces", action="store_true", help="Also save estimates.C to a CSV file"
    )  # TODO save by default

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
        "--mask", action="store_true", help="Enable brain mask (default: mask OFF)"
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

    p.add_argument(
        "--no-mc",
        action="store_true",
        help="Don't run NoRMCorre piecewise-rigid motion correction before CNMF "
        "(disabled by default, matching the source notebook)",
    )

    # Format override
    p.add_argument(
        "--format",
        dest="format_override",
        choices=["multi-tp", "multi-cam", "single-movie", "interleaved", "legacy"],
        default=None,
        help="Override auto-detect format",
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

    p.add_argument(
        "--no-play",
        action="store_true",
        help="Skip generating/saving diagnostic plots (headless-friendly; movie playback "
        "and interactive Bokeh widgets from the notebook are always skipped in this "
        "CLI script)",
    )
    p.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary files (mmap, intermediate TIFFs) instead of deleting them after CNMF",
    )

    return p.parse_args()


ARGS = parse_args()


OUTPUT_DIR = RESULTS_ROOT / ARGS.run_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORK_DIR = OUTPUT_DIR / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

N_WORKERS = ARGS.n_workers


# =============================================================================
# IMPORTS
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import tifffile
from skimage.morphology import (
    closing,
    opening,
    disk,
    remove_small_objects,
)
from skimage.filters import threshold_otsu
from scipy.optimize import linear_sum_assignment

import caiman as cm
import caiman.mmapping
import caiman.base.movies
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from caiman.source_extraction.cnmf import params as params_module

from caiman.motion_correction import MotionCorrect
from caiman.utils.visualization import plot_contours, get_contours


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
    Return (format_name, file_list, sample_shape). Includes 'Interleaved' detection.
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
            return "Interleaved", cam_files, shape
        return "single-movie", cam_files, shape

    lux_files = sorted(glob.glob(str(folder / "*.lux*.h5")))
    if lux_files:
        with h5py.File(lux_files[0], "r") as fh:
            if "Data" in fh:
                shape = tuple(fh["Data"].shape)
                if read_n_planes(lux_files[0]) > 1:
                    return "Interleaved", lux_files, shape
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
    if fmt == "Interleaved":
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
    if fmt == "Interleaved":
        T_full, H, W = shape
        nz = n_planes or 1
        if z_index is None:
            z_index = nz // 2
        if z_index >= nz:
            raise ValueError(f"z_index {z_index} >= n_planes {nz}")
        frames_per_plane = T_full // nz

        if max_frames:
            frames_per_plane = min(frames_per_plane, max_frames)
        print(f"  Interleaved: z={z_index}/{nz} planes -> {frames_per_plane} frames")
        with h5py.File(files[0], "r") as fh:
            data = fh["Data"][z_index::nz][:frames_per_plane].astype(np.float32)
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

def stripe_remove(data: np.ndarray):
    col_median = np.median(data, axis=(0, 1), keepdims=True)
    return np.clip(data - col_median, 0, None).astype(np.float32), col_median



def make_brain_mask(data: np.ndarray, ratio: float, label: str = "") -> np.ndarray:
    """
    Build a binary mask of brain pixels using Otsu on a low-percentile
    (motion-robust, baseline-like) projection. Cleans up with morphological
    opening/closing and fills holes.
    """
    ref_img = np.percentile(data, 20, axis=0)

    try:
        thr = threshold_otsu(ref_img)
    except Exception:
        print(
            "[STAGE:make_brain_mask] WARNING: Otsu failed. Falling back to mean threshold."
        )
        thr = np.mean(ref_img)

    mask = ref_img > thr

    if mask.sum() < 100:
        print(
            f"[STAGE:make_brain_mask] WARNING: Otsu mask is tiny ({mask.sum()} px). Falling back to no mask."
        )
        return np.ones_like(mask, dtype=bool)

    mask = binary_fill_holes(mask)

    um_per_px = 0.208 / ratio
    se_radius = max(3, int(10 / um_per_px))  # ~10 µm in pixel units

    mask = opening(mask, disk(se_radius))
    mask = closing(mask, disk(se_radius * 2))
    mask = binary_fill_holes(mask)  # again after closing
    mask = remove_small_objects(mask, min_size=int(500 / um_per_px**2))

    if mask.sum() < 100:
        print(f"  WARNING: brain mask too small after cleanup. Disabling mask.")
        return np.ones_like(mask, dtype=bool)

    coverage = 100.0 * mask.sum() / mask.size
    print(
        f"[STAGE:make_brain_mask] INFO: Brain mask coverage: {coverage:.1f}% of frame ({label})"
    )
    return mask


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (data * mask[None, :, :]).astype(np.float32)


def preprocess_movie(data: np.ndarray, label: str = ""):
    info = {"original_shape": tuple(data.shape)}
    data_cm = cm.movie(data)

    target_res = int(ARGS.resolution) if ARGS.resolution != "full" else data.shape[1]
    ratio = target_res / data.shape[1]

    if ratio != 1:
        data_cm = data_cm.resize(fx=ratio, fy=ratio, fz=1.0)
        info["downsampled_to"] = (target_res, target_res)

    if not ARGS.no_stripe:
        data_arr, col_median = stripe_remove(np.array(data_cm))
        data_cm = cm.movie(data_arr)
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

    if ARGS.mask:
        mask = make_brain_mask(data_cm, ratio=ratio, label=label)
        data_cm = cm.movie(apply_mask(data_cm, mask))
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
        mask = np.ones(data_cm.shape[1:], dtype=bool)
        info["brain_mask_used"] = False

    info["final_shape"] = tuple(data_cm.shape)
    return data_cm, mask, info


# =============================================================================
# CNMF CONFIG (resolution-aware)
# =============================================================================


def get_base_params(external_params: dict = None) -> dict:
    """Merges external best_params JSON with hardcoded ones."""
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
        "fr": 5,
        "decay_time": external.get("decay_time", 0.25), #tuned
        "method_init": "greedy_roi", 
        "K": external.get("K", None),  #tuned
        # TODO compute gSiz from gSig here instead of run_cnmf
        "nb": external.get("nb", 1), #tuned
        "nb_patch": external.get("nb_patch", 1), # same as nb
        "merge_thr": 0.9,
        "use_cnn": True,
        "min_cnn_thr": 0.99,  # threshold for CNN based classifier
        "cnn_lowest": 0.1,  # neurons with cnn probability lower than this value are rejected
        "min_SNR": 2.0,
        "rval_thr": 0.85,
        "del_duplicates": True,
        "ssub": 1,
        "tsub": 1,
        "pw_rigid": True,
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
# CNMF EXECUTION
# =============================================================================


def array_to_memmap(array: np.ndarray, basename: Path) -> str:
    """Write an array to a temporary TIFF and memory-map it via Caiman."""
    tif = str(basename) + ".tif"
    tifffile.imwrite(tif, array.astype(np.float32))
    mmap_path = caiman.mmapping.save_memmap(
        [tif],
        base_name=str(basename),
        order="C",
        border_to_0=0,
    )
    if not ARGS.keep_temp and os.path.exists(tif):
        os.remove(tif)
    return mmap_path


def _setup_cluster(nworkers=N_WORKERS):
    try:
        print(f"    [STAGE]:Setting up cluster")
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
    g = int(p["gSig"])
    p["gSig"] = (g, g)

    if "gSiz" not in params_override:
        siz = int(round((2 * g) + 1))
        p["gSiz"] = (siz, siz)

    print("  [STAGE:prepping parameters] done", flush=True)

    return params_module.CNMFParams(params_dict=p)

def _motion_correct(fname_mmap: str) -> str:
    """Run motion correction once on the input mmap and return the MC'd output path.

    Called once before the Bayesian optimization loop so MC is not repeated
    per trial (MC params are constant across trials).

    CaImAn names the MC output by appending to the input filename, so the
    output ends up with two 'order_X' substrings, e.g.:
      ...order_C_frames_350_els__d1_..._order_F_frames_350.mmap
    CaImAn's load_memmap uses re.search and hits 'order_C' (from the input)
    first, but the file is actually F-order — causing scrambled data and SVD
    failure when fit_file(motion_correct=False) loads it. We fix this by
    force-loading the raw MC output as F-order and writing a clean, unambiguously
    named C-order mmap that load_memmap will parse correctly.
    """
    mc_keys = ("max_shifts", "strides", "overlaps", "max_deviation_rigid", "pw_rigid", "border_nan")
    mc_params = {k: BASE_PARAMS[k] for k in mc_keys if k in BASE_PARAMS}
    t0 = time.time()
    print("  [precomputing motion correction — runs once for all trials]", flush=True)
    cluster, n_processes = _setup_cluster(10)
    print(
        f"Successfully initialized multicore processing with a pool of {n_processes} CPU cores"
    )
    try:
        mc = MotionCorrect([fname_mmap], dview=cluster, **mc_params)
        mc.motion_correct(save_movie=True)
        pw = BASE_PARAMS.get("pw_rigid", True)
        mc_raw = mc.fname_tot_els[-1] if pw else mc.fname_tot_rig[-1]

        # Parse the true output dimensions from the filename (last d1/d2/d3/frames match).
        m = re.search(r"d1_(\d+)_d2_(\d+)_d3_(\d+).*frames_(\d+)\.mmap$", mc_raw)
        d1, d2, d3, T = (
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            int(m.group(4)),
        )

        # Force-load as F-order (the true MC output format) and write a clean C-order
        # mmap with an unambiguous single-'order' filename.
        out_path = str(
            Path(mc_raw).parent
            / f"Yr_mc_d1_{d1}_d2_{d2}_d3_{d3}_order_C_frames_{T}.mmap"
        )
        src = np.memmap(
            mc_raw, dtype=np.float32, mode="r", shape=(d1 * d2 * d3, T), order="F"
        )
        dst = np.memmap(
            out_path, dtype=np.float32, mode="w+", shape=(d1 * d2 * d3, T), order="C"
        )
        np.copyto(dst, src)
        dst[~np.isfinite(dst)] = 0  # zero NaN/Inf border pixels left by MC
        del dst  # flush to disk
    finally:
        try:
            _stop_cluster(cluster)
        except Exception:
            pass

    print(f"  [MC done in {time.time()-t0:.1f}s → {Path(out_path).name}]", flush=True)
    return out_path


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
                fname_to_use = _motion_correct(fname_mmap)
            except:
                print("Motion correction failed. Using original memmap", flush=True)
            finally:
                if cluster is not None:
                    _stop_cluster(cluster)
                    cluster = None  # Reset so the master finally block ignores it

        else:
            print(
                "  [STAGE:motion_correction] skipped — using orignal memmap",
                flush=True,
            )

        cluster, n_processes = _setup_cluster()

        cnmf_obj = _fit_cnmf(fname_to_use, opts, n_processes, cluster)

        cnmf_refit = cnmf_obj

        images = None
        if cnmf_obj.estimates.A is not None and cnmf_obj.estimates.A.shape[1] > 0:
            images = _reload_images(fname_to_use)

        
        if (
            images is not None
            and cnmf_obj.estimates.A is not None
            and cnmf_obj.estimates.A.shape[1] > 0
        ):
            cnmf_refit = _refit(cnmf_obj, images, cluster)

        if images is not None:
            _evaluate_and_select(cnmf_refit, images, cluster)

        _stop_cluster(cluster)
        cluster = None
        return cnmf_refit, time.time() - t0, fname_to_use

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


# =============================================================================
# OUTPUT
# =============================================================================

def save_summary(
    mode: str,
    best_params: dict,
    fmt: str,
    prep_info: dict,
    cnmf_obj,
    extra: dict | None = None,
):
    """
    Saves a JSON summary and appends key metrics to a master CSV.
    Updated to capture actual data from the new CNMF run and preprocessing stages.
    """
    # Safely extract the number of neurons kept after evaluation/refit
    n_neurons = 0
    if cnmf_obj and hasattr(cnmf_obj, 'estimates') and cnmf_obj.estimates.A is not None:
        n_neurons = cnmf_obj.estimates.A.shape[1]

    # Build the comprehensive JSON summary
    summary = {
        "mode": mode,
        "run_name": ARGS.run_name,
        "resolution": ARGS.resolution,
        "brain_mask": ARGS.mask,
        "stripe_removal": not ARGS.no_stripe,
        "best_params": best_params,
        "format_info": fmt,
        "preprocessing_info": prep_info,
        "n_neurons_found": n_neurons,
    }
    
    if extra:
        summary.update(extra)

    # Save to JSON in the run's specific output directory
    with open(str(OUTPUT_DIR / "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved summary.json -> {OUTPUT_DIR}/")

    # Build the row for the master CSV file
    master = RESULTS_ROOT / "all_runs.csv"
    row = {
        "run_name": ARGS.run_name,
        "mode": mode,
        "format": fmt,
        "resolution": ARGS.resolution,
        "brain_mask": ARGS.mask,
        "stripe_removal": not ARGS.no_stripe,
        "n_neurons_kept": n_neurons,
        "mask_coverage_frac": prep_info.get("mask_coverage_frac", float("nan")),
        "gSig": best_params.get("gSig"),
        "rf": best_params.get("rf"),
        "p": best_params.get("p"),
    }
    
    df_row = pd.DataFrame([row])
    
    # Append or create the master CSV
    if master.is_file():
        df_row.to_csv(master, mode="a", header=False, index=False)
    else:
        df_row.to_csv(master, index=False)
    print(f"Appended row to {master}")


def plot_contours_via_coordinates(images, cnm, output_dir):
    print("[STAGE: plotting] Extracting contour coordinates...")
    
    # 1. Generate the anatomical background (Mean Image)
    mean_frame = images.mean(axis=0)
    H_val, W_val = mean_frame.shape

    # 2. Filter for accepted components only
    if hasattr(cnm.estimates, "idx_components") and cnm.estimates.idx_components is not None and len(cnm.estimates.idx_components) > 0:
        A_accepted = cnm.estimates.A[:, cnm.estimates.idx_components]
    else:
        A_accepted = cnm.estimates.A
        
    num_accepted = A_accepted.shape[1]

    # 3. Extract the (X, Y) coordinates of the footprint boundaries
    # thr=0.2 (or 0.8) uses CaImAn's cumulative energy math to find the boundary
    coors = get_contours(A_accepted, (H_val, W_val)) #, thr=0.8)

    # 4. Plotting
    fig, ax = plt.subplots(figsize=(11, 11), dpi=150)
    ax.imshow(mean_frame, cmap="gray")
    ax.set_title(f"Optimized Spatial Footprints: {num_accepted} kept neurons", fontsize=10)
    ax.axis("off")

    # 5. Draw the coordinates as clean cyan lines
    for c in coors:
        # c['coordinates'] is an array of shape (N, 2) containing X, Y points
        coords = c['coordinates']
        
        # Plot a line connecting the boundary points
        ax.plot(coords[:, 0], coords[:, 1], color="cyan", linewidth=0.6, alpha=0.85)

    plt.tight_layout()
    contour_path = os.path.join(output_dir, "optimized_contours_coords.png")
    plt.savefig(contour_path)
    plt.close(fig)
    print(f"[STAGE: Save] Saved coordinate-based contour plot to {contour_path}")


def generate_optimized_plots(images, cnm, output_dir=OUTPUT_DIR):
    print("[STAGE: plotting] Generating diagnostic contour and trace plots...")

    # --- 1. Plot Contours via Coordinates ---
    plot_contours_via_coordinates(images, cnm, output_dir)

    
    # --- 2. Extract / Calculate Traces ---
    Cn = cm.local_correlations(images, swap_dim=False)
    # Cn[np.isnan(Cn)] = 0
    if hasattr(cnm.estimates, "F_dff") and cnm.estimates.F_dff is not None:
        print("[STAGE: traces] estimates.F_dff already defined.")
        traces = cnm.estimates.F_dff
    else:
        try:
            print("[STAGE: traces] Calculating estimates.F_dff...")
            cnm.estimates.detrend_df_f(
                quantileMin=8, 
                frames_window=250,
                flag_auto=False,
                use_residuals=False
            )
            # Fetch the newly calculated dF/F traces
            traces = cnm.estimates.F_dff 
        except Exception as e:
            print(f"[STAGE: traces] Detrend failed ({e}). Falling back to estimates.C")
            traces = cnm.estimates.C
            
    # --- 3. Save Results as Data ---
    # Attach correlation image to the object before saving
    cnm.estimates.Cn = Cn  
    save_path = str(os.path.join(output_dir, "pipeline_results.hdf5"))
    cnm.save(save_path)
    print(f"[STAGE: Save] Saved CNMF results to {save_path}")

    # Save full traces as a raw numpy array
    npy_save_path = str(os.path.join(output_dir, "neuron_traces.npy"))
    np.save(npy_save_path, traces)
    print(f"[STAGE: Save] Saved raw traces as numpy array to {npy_save_path}")

    # Calculate frame times to build a proper DataFrame
    num_frames = images.shape[0]
    frame_rate = cnm.params.data.get("fr", 30) # Defaults to 30fps if 'fr' is missing
    frame_times = np.linspace(0, num_frames / frame_rate, num_frames)

    # Save full traces as a CSV with a time index
    data_to_save = np.vstack((frame_times, traces)).T
    save_df = pd.DataFrame(data_to_save)
    save_df.rename(columns={0: "time"}, inplace=True)
    c_save_path = os.path.join(output_dir, "neuron_traces.csv")
    save_df.to_csv(c_save_path, index=False)
    print(f"[STAGE: Save] Saved traces CSV to {c_save_path}")

    # --- 4. Plot Sample Traces ---
    num_comps = traces.shape[0]
    if num_comps > 0:
        fig_traces, ax_traces = plt.subplots(figsize=(13, 5), dpi=100)
        components_to_plot = min(15, num_comps)
        offset = 0

        for i in range(components_to_plot):
            trace = traces[i, :]
            # Normalize trace between 0 and 1 for clean stacked plotting
            trace_norm = (trace - np.min(trace)) / (np.max(trace) - np.min(trace) + 1e-6)
            ax_traces.plot(frame_times, trace_norm + offset, lw=1.2)
            offset += 1.2

        ax_traces.set_title(f"Temporal Traces (Top {components_to_plot} of {num_comps} Components)")
        ax_traces.set_xlabel("Time (s)")
        ax_traces.set_yticks([])

        trace_path = os.path.join(output_dir, "optimized_traces.png")
        fig_traces.tight_layout()
        fig_traces.savefig(trace_path)
        plt.close(fig_traces)
        print(f"[STAGE: Save] Saved sample trace plot to {trace_path}")



# =============================================================================
# MODES
# =============================================================================


def mode_single_plane():
    print(f"\n--- SINGLE-PLANE mode ---")
    fmt, files, sample_shape, det_n_planes = discover(
        ARGS.data_dir, ARGS.format_override, ARGS.n_planes
    )

    T, _, _ = sample_shape
    n_frames = T if ARGS.max_frames is None else (ARGS.max_frames * det_n_planes)
    z = det_n_planes // 2 if ARGS.z_index is None else ARGS.z_index #TODO possibly remove n_planes argument

    # Data Loading Stage
    print(f"[STAGE] Loading movie data (Format: {fmt})...")
    data_raw = load_movie(
        folder=ARGS.data_dir,
        fmt=fmt,
        files=files,
        shape=sample_shape,
        z_index=z,
        max_frames=ARGS.max_frames,
        n_planes=det_n_planes,
    )
    movie_orig = cm.movie(data_raw)

    if not ARGS.no_play:
        max_projection_orig = np.max(movie_orig, axis=0)
        correlation_image_orig = caiman.local_correlations(movie_orig, swap_dim=False)
        correlation_image_orig[np.isnan(correlation_image_orig)] = 0
        import matplotlib.pyplot as plt

        f, (ax_max, ax_corr) = plt.subplots(1, 2, figsize=(6, 3))
        ax_max.imshow(
            max_projection_orig,
            cmap="viridis",
            vmin=np.percentile(np.ravel(max_projection_orig), 50),
            vmax=np.percentile(np.ravel(max_projection_orig), 99.5),
        )
        ax_max.set_title("Max Projection Orig", fontsize=12)
        ax_corr.imshow(
            correlation_image_orig,
            cmap="viridis",
            vmin=np.percentile(np.ravel(correlation_image_orig), 50),
            vmax=np.percentile(np.ravel(correlation_image_orig), 99.5),
        )
        ax_corr.set_title("Correlation Image Orig", fontsize=12)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / "orig_projections.png"), dpi=100)
        plt.close(f)

    # ---- preprocess ---------------------------------------------------
    print(f"[STAGE] Running preprocessing (masking, stripe removal, downsampling)...")
    data, mask, prep_info = preprocess_movie(movie_orig, label="movie")

    T_full = data.shape[0]
    print(f"[INFO] Single plane established: Run with {T_full} frames.")

    dims = data.shape[1:]

    # Memory Mapping Stage
    print(f"[STAGE] Creating memory maps for processing...")
    processed_mmap = array_to_memmap(data, WORK_DIR / "full_movie")

    print(f"[STAGE] Running CNMF on plane {z}")
    cnmf_obj, _, fname = run_cnmf(BASE_PARAMS, processed_mmap, do_mc=not ARGS.no_mc)

    # Reporting Stage
    print(f"[STAGE] Generating traces and contours...")
    mmap_file_path = cnmf_obj.params.data['fnames'][0]
    
    # 2. Load the 2D memory-mapped array (Yr)
    # Yr is flattened to (pixels x frames) to speed up math operations
    Yr, dims, num_frames = cm.load_memmap(mmap_file_path)
    
    # 3. Reshape the 2D array back into a 3D movie: (frames, x, y)
    # The order="F" (Fortran) parameter is required because of how CaImAn flattens the data
    images = np.reshape(Yr.T, [num_frames] + list(dims), order="F")
    generate_optimized_plots(images=images, cnm=cnmf_obj)
    
    print("[STAGE] Saving summary...")
    save_summary(mode="single_plane", best_params=LOADED_BEST_PARAMS, fmt=fmt, prep_info=prep_info, cnmf_obj=cnmf_obj)

    print(f"[STAGE] Single-Plane mode completed successfully.")


def mode_all_planes():
    print(f"\n--- ALL-PLANES mode ---")
    # fmt, files, sample_shape, det_n_planes = discover(
    #     ARGS.data_dir, ARGS.format_override, ARGS.n_planes
    # )
    # TODO implement for multi-plane too
    


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
