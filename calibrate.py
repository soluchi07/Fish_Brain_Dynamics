#!/usr/bin/env python3
"""
calibrate_cnmf.py  —  Standalone Bayesian calibration of CNMF parameters
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLAS_NUM_THREADS"] = "1"

import cv2

try:
    cv2.setNumThreads(0)
except:
    pass

import argparse
import glob
import json
import re
from scipy.ndimage import binary_fill_holes
import sys
import time
import multiprocessing
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


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Bayesian calibration of CNMF parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", required=True, help="Output folder name")
    p.add_argument(
        "--data-dir", type=Path, required=True, help="Folder containing the dataset"
    )
    p.add_argument("--z-index", type=int, default=None, help="Z-plane index to extract (default: middle)")
    p.add_argument(
        "--n-planes", type=int, default=None, help="Number of Z-planes interleaved"
    )
    p.add_argument("--resolution", choices=["full", "1024", "512"], default="512")
    p.add_argument("--no-stripe", action="store_true", help="Disable stripe removal (default: stripe ON)")
    p.add_argument("--no-mc", action="store_true", help="Disable motion correction (default: mc ON)")
    p.add_argument("--mask", action="store_true", help="Enable brain mask (default: mask OFF)")
    p.add_argument("--max-frames", type=int, default=None, help="Cap loaded frames")
    p.add_argument(
        "--format",
        dest="format_override",
        choices=["multi-tp", "multi-cam", "single-movie", "interleaved", "legacy"],
        default=None,
    )
    p.add_argument("--n-calls", type=int, default=20)
    p.add_argument("--n-initial", type=int, default=5)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args()


ARGS = parse_args()

N_WORKERS = ARGS.n_workers
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
from skimage.morphology import (
    closing,
    opening,
    disk,
    remove_small_objects,
)
from skimage.filters import threshold_otsu
import scipy.stats as stats

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence

import caiman as cm
import caiman.mmapping
import caiman.base.movies
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from caiman.source_extraction.cnmf import params as params_module
from caiman.motion_correction import MotionCorrect
from caiman.utils.visualization import plot_contours

if not hasattr(cm, "load"):
    cm.load = caiman.base.movies.load
if not hasattr(cm, "movie"):
    cm.movie = caiman.base.movies.movie


print("=" * 70)
print(f"calibrate_cnmf.py  |  run={ARGS.run_name}")
print("=" * 70)
print(f"Data dir   : {ARGS.data_dir}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Resolution : {ARGS.resolution}")
print(f"Brain mask : {'ON' if ARGS.mask else 'OFF'}")
print(f"Trials     : n_calls={ARGS.n_calls}  n_initial={ARGS.n_initial}")
print(f"Workers={N_WORKERS}")
if ARGS.n_planes:
    _default_z = ARGS.z_index if ARGS.z_index is not None else ARGS.n_planes // 2
    print(f"Z-planes   : {ARGS.n_planes} stacked  (extracting z={_default_z})")
    
    
# =============================================================================
# FORMAT DETECTION & LOADER (Aligned with new_pipeline.py)
# =============================================================================


def read_n_planes(filepath: str) -> int:
    try:
        with h5py.File(filepath, "r") as fh:
            if "metadata" not in fh:
                return 1
            meta = json.loads(fh["metadata"][()])
            return int(meta["metaData"]["stack"]["n"])
    except Exception:
        return 1


def detect_format(folder: Path) -> tuple[str, list[str], tuple]:
    folder = Path(folder)
    tp_files = sorted(
        glob.glob(str(folder / "tp-*_ch-*_st-*_obj-*_cam-*.lux.h5")),
        key=lambda p: int(re.search(r"tp-0-(\d+)", p).group(1)),
    )
    if tp_files:
        with h5py.File(tp_files[0], "r") as fh:
            return "multi-tp", tp_files, tuple(fh["Data"].shape)

    cam_files = sorted(
        glob.glob(str(folder / "Cam_long_*.lux*.h5")),
        key=lambda p: int(re.search(r"Cam_long_(\d+)", p).group(1)),
    )
    if cam_files:
        with h5py.File(cam_files[0], "r") as fh:
            shape = tuple(fh["Data"].shape)
        if len(cam_files) > 1:
            return "multi-cam" if shape[0] == 1 else "single-movie", cam_files, shape
        if read_n_planes(cam_files[0]) > 1:
            return "interleaved", cam_files, shape
        return "single-movie", cam_files, shape

    lux_files = sorted(glob.glob(str(folder / "*.lux*.h5")))
    if lux_files:
        with h5py.File(lux_files[0], "r") as fh:
            if "Data" in fh:
                shape = tuple(fh["Data"].shape)
                if read_n_planes(lux_files[0]) > 1:
                    return "interleaved", lux_files, shape
                return "single-movie", lux_files, shape

    h5_files = sorted(glob.glob(str(folder / "*.h5")))
    if h5_files:
        with h5py.File(h5_files[0], "r") as fh:
            if "Data" in fh:
                return "legacy", h5_files, tuple(fh["Data"].shape)
    raise FileNotFoundError(f"No recognizable files in {folder}")


def discover(
    folder: Path,
    override: Optional[str] = None,
    n_planes_override: Optional[int] = None,
):
    fmt, files, shape = detect_format(folder)
    if override:
        fmt = override
    det_n = n_planes_override or (
        read_n_planes(files[0]) if fmt == "interleaved" else None
    )
    return fmt, files, shape, det_n


def load_movie(
    folder, fmt, files, shape, z_index=None, max_frames=None, n_planes=None
) -> np.ndarray:
    if fmt == "interleaved":
        nz = n_planes or 1
        z = z_index or nz // 2
        with h5py.File(files[0], "r") as fh:
            data = fh["Data"][z::nz][:max_frames].astype(np.float32)
        return data
    elif fmt == "single-movie" or fmt == "legacy":
        with h5py.File(files[0], "r") as fh:
            if n_planes and n_planes > 1:
                z = z_index or n_planes // 2
                data = fh["Data"][z::n_planes][:max_frames].astype(np.float32)
            else:
                data = fh["Data"][:max_frames].astype(np.float32)
        return data
    elif fmt == "multi-tp":
        z = z_index or shape[0] // 2
        T = min(len(files), max_frames) if max_frames else len(files)
        data = np.zeros((T, shape[1], shape[2]), dtype=np.float32)
        for i, fp in enumerate(files[:T]):
            with h5py.File(fp, "r") as fh:
                data[i] = fh["Data"][z].astype(np.float32)
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
# BAYESIAN SEARCH SPACE & BASE PARAMS
# =============================================================================


SEARCH_SPACE = [
    Integer(3, 12, name="gSig"),
    Categorical([15, 25, 40, 60, 80], name="rf"),
    Real(0.2, 0.5, name="overlap_fraction"),
    Integer(3, 4, name="nb"), # changed nb search sace from 1, 2 -> 3, 4 to avoid background leakage
    Categorical([0.1, 0.15, 0.2, 0.25, 0.3], name="decay_time"), # changed decay_time search space from 0.2 to 0.4 -> 0.1 to 0.3 to avoid decay time issues
    Real(0.05, 0.3, name="occupancy_frac"),
]

PARAM_NAMES = [s.name for s in SEARCH_SPACE]


def get_base_params():
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
        
    return {
        "fr": 5,
        "method_init": "greedy_roi",
        "min_cnn_thr": 0.99,  # threshold for CNN based classifier
        "cnn_lowest": 0.1,  # neurons with cnn probability lower than this value are rejected
        "merge_thr": 0.9,
        "use_cnn": True,
        "rval_thr": 0.85,
        "min_SNR": 2.0,
        "ssub": 1,
        "tsub": 1,
        "del_duplicates": True,
        "pw_rigid": True,
        "border_nan": "copy",
        **mc
    }


BASE_PARAMS = get_base_params()


# =============================================================================
# CNMF RUNNER
# =============================================================================


def array_to_memmap(array: np.ndarray, basename: Path):
    tif = str(basename) + ".tif"
    tifffile.imwrite(tif, array.astype(np.float32))
    mmap_path = caiman.mmapping.save_memmap(
        [tif], base_name=str(basename), order="C", border_to_0=0
    )
    if not ARGS.keep_temp and os.path.exists(tif):
        os.remove(tif)
    return mmap_path


def _setup_cluster(nworkers=N_WORKERS):
    try:
        _, cluster, n_processes = cm.cluster.setup_cluster(
            backend="multiprocessing", n_processes=nworkers, single_thread=False
        )
        return cluster, n_processes
    except:
        return None, 1


def _stop_cluster(cluster):
    cm.stop_server(dview=cluster)
    for p in multiprocessing.active_children():
        p.join()


# def _motion_correct(fname_mmap: str, opts, cluster):
#     mc = MotionCorrect([fname_mmap], dview=cluster, **opts.get_group("motion"))
#     mc.motion_correct(save_movie=True)
#     return cm.save_memmap(
#         mc.mmap_file, base_name="memmap_", order="C", border_to_0=0, dview=cluster
#     )

def precompute_mc(fname_mmap: str) -> str:
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


def run_cnmf(params_override: dict, fname_to_use: str):
    opts = _prep_params(params_override, fname_to_use)

    cluster, n_processes = None, 1
    t0 = time.time()
    try:
        cluster, n_processes = _setup_cluster()
        
        cnmf_obj = _fit_cnmf(fname_to_use, opts, n_processes, cluster)
        images = None
        
        if cnmf_obj.estimates.A.shape[1] > 0:
            images = _reload_images(fname_to_use)
        
        if images is not None and cnmf_obj.estimates.A.shape[1] > 0:
            cnmf_obj = _refit(cnmf_obj, images, cluster)
            
        if images is not None:
            _evaluate_and_select(cnmf_obj, images, cluster)
        
        _stop_cluster(cluster)
        cluster = None  # Reset so finally block doesn't stop it twice
        return cnmf_obj, time.time() - t0, fname_to_use

    except Exception as exc:
        print(f"    [STAGE:CNMF] failed: {exc}", flush=True)
        return None, time.time() - t0, None
    finally:
        if cluster is not None:
            try:
                _stop_cluster(cluster)
            except Exception:
                pass

# =============================================================================
# SCORING METHODOLOGY
# =============================================================================

def score_run(cnmf_obj, Yr_mmap_path, dims, mask, params):
    """
    Composite scoring function for a single CNMF trial.

    Metrics
    -------
    Q   — Extraction quality: weighted median of logistic-squashed SNR,
          spatial correlation (rval), and CNN probability.
    F   — Reconstruction fidelity: 1 - ||Y - Y_hat||_F / ||Y||_F.
    D   — Joint spatial-temporal redundancy: fraction of component pairs
          that share both spatial overlap (cosine similarity > 0.5) AND
          near-identical temporal traces (Pearson r > 0.8). Spatial
          correlations are computed fully in sparse arithmetic to avoid
          densifying the full (pixels x N) footprint matrix.
    Psi — Patch-seam artifact index: KS statistic of centroid positions
          modulo the true patch period = (2*rf+1) - stride, not stride
          itself, which is the correct seam spacing.
    B   — Background leakage: lag-1 autocorrelation of a random sample of
          pixel-level residual traces. Elevated when neuropil/background
          is not fully captured by the nb background components.
    Phi — Decay-time diagnostic: mean absolute lag-1 autocorrelation of
          per-component trace residuals (YrA). Elevated when assumed AR
          kinetics do not match the data; uses YrA not pixel residuals.
    N   — Yield relative to patch-geometry capacity (light tie-breaker).
    """
    sentinel = {"composite_score": -float("inf")}
    if (
        cnmf_obj is None
        or cnmf_obj.estimates.A is None
        or cnmf_obj.estimates.A.shape[1] == 0
    ):
        return sentinel

    from scipy import sparse, stats
    from scipy.sparse import linalg as sp_linalg

    A = cnmf_obj.estimates.A          # sparse (pixels × N)
    C = np.asarray(cnmf_obj.estimates.C)   # (N × T)
    num_comps = A.shape[1]
    if num_comps == 0:
        return sentinel

    # ---- load raw movie (non-fatal) ----------------------------------------
    Yr = None
    try:
        import caiman
        Yr, _, num_frames = caiman.mmapping.load_memmap(Yr_mmap_path)
    except Exception as exc:
        print(f"  [score_run] WARNING: could not load mmap ({exc}); F/B/Phi will be 0")

    # =========================================================================
    # Q — Extraction quality (continuous, no accept/reject threshold)
    # =========================================================================
    def squash(x):
        return 1.0 / (1.0 + np.exp(-0.5 * (np.asarray(x, dtype=float) - 2.0)))

    snr  = np.nan_to_num(getattr(cnmf_obj.estimates, "SNR_comp",  np.full(num_comps, 2.0)))
    rval = np.nan_to_num(getattr(cnmf_obj.estimates, "r_values",  np.full(num_comps, 0.5)))
    cnn  = np.nan_to_num(getattr(cnmf_obj.estimates, "cnn_preds", np.full(num_comps, 0.5)))
    rval = np.clip(rval, 0.0, 1.0)
    cnn  = np.clip(cnn,  0.0, 1.0)

    Q = float(np.mean(0.4 * squash(snr) + 0.4 * rval + 0.2 * cnn))

    # =========================================================================
    # D — Joint spatial-temporal redundancy (fully sparse, mask-aware)
    # =========================================================================
    D = 0.0
    if num_comps > 1:
        # Apply brain mask: zero out pixels outside the mask so background
        # pixels do not inflate spatial overlap between components.
        A_masked = A.copy().astype(np.float64)
        if mask is not None:
            flat_mask = mask.flatten(order="F")          # CaImAn uses Fortran order
            out_of_mask = np.where(~flat_mask)[0]
            if len(out_of_mask):
                A_masked = A_masked.tolil()
                A_masked[out_of_mask, :] = 0
                A_masked = A_masked.tocsc()

        # Column norms — stay sparse throughout; only the (N×N) inner
        # product is densified, never the full (pixels×N) matrix.
        col_norms = np.asarray(
            np.sqrt(A_masked.power(2).sum(axis=0))
        ).flatten() + 1e-9
        A_norm = A_masked.multiply(1.0 / col_norms)     # sparse (pixels × N)
        spatial_corr = np.asarray(
            (A_norm.T @ A_norm).todense()               # dense (N × N) only here
        )
        np.fill_diagonal(spatial_corr, 0.0)

        # Temporal correlation
        C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        temp_corr = C_norm @ C_norm.T
        np.fill_diagonal(temp_corr, 0.0)

        redundant_pairs = (spatial_corr > 0.5) & (temp_corr > 0.8)
        D = float(np.sum(redundant_pairs) / (num_comps * 2))

    # =========================================================================
    # Psi — Patch-seam artifact index
    # =========================================================================
    Psi = 0.0
    rf     = int(params.get("rf", 0))
    stride = int(params.get("stride", 0))
    # Seam lines repeat every (patch_width - stride) pixels, NOT every stride
    # pixels.  Using stride itself bins against the wrong grid entirely.
    period = max(1, (2 * rf + 1) - stride)

    if stride > 0 and num_comps >= 5:
        try:
            import caiman as cm
            centroids = cm.base.rois.com(A, *dims)      # (N × 2), [row, col]
            cy = centroids[:, 0] % period
            cx = centroids[:, 1] % period
            stat_y, _ = stats.kstest(cy / period, "uniform")
            stat_x, _ = stats.kstest(cx / period, "uniform")
            Psi = float((stat_y + stat_x) / 2.0)
        except Exception as exc:
            print(f"  [score_run] WARNING: Psi computation failed ({exc}); defaulting to 0")

    # =========================================================================
    # Residual-based metrics: F, B, Phi
    # =========================================================================
    F, B, Phi = 0.0, 0.0, 0.0

    if Yr is not None:
        b = getattr(cnmf_obj.estimates, "b", None)
        f = getattr(cnmf_obj.estimates, "f", None)

        Y_hat = A @ C
        if b is not None and f is not None and b.shape[1] > 0:
            Y_hat = Y_hat + b @ f

        residual = Yr - Y_hat                           # (pixels × T)

        # --- F: Reconstruction fidelity --------------------------------------
        norm_Y   = np.linalg.norm(Yr,      "fro") + 1e-9
        norm_res = np.linalg.norm(residual, "fro")
        F = float(1.0 - norm_res / norm_Y)

        # --- B: Background leakage via pixel-residual lag-1 ACF --------------
        # Elevated when neuropil / background is not fully captured.
        # We sample pixel traces (rows of residual) rather than component traces.
        n_px_sample = min(1000, residual.shape[0])
        px_idx = np.random.choice(residual.shape[0], n_px_sample, replace=False)
        res_sample = residual[px_idx, :]                # (sample × T)
        lag1_px = []
        for i in range(res_sample.shape[0]):
            r = res_sample[i]
            denom = np.dot(r, r) + 1e-9
            lag1_px.append(np.dot(r[:-1], r[1:]) / denom)
        raw_b = float(np.mean(lag1_px))
        B = float(max(0.0, raw_b))

        # --- Phi: Decay-time diagnostic via per-component trace residual -----
        # YrA = raw fluorescence projected onto each footprint, minus the
        # estimated calcium trace C.  Lag-1 ACF on YrA is elevated when the
        # AR model (governed by decay_time) under- or over-captures the
        # transient shape; this is a distinct signal from pixel-level B above.
        YrA = getattr(cnmf_obj.estimates, "YrA", None)
        if YrA is not None and YrA.shape[0] == num_comps:
            acfs = []
            for i in range(num_comps):
                x = np.asarray(YrA[i]).flatten()
                x = x - x.mean()
                denom = np.dot(x, x) + 1e-9
                if len(x) > 1:
                    acfs.append(abs(np.dot(x[:-1], x[1:]) / denom))
            Phi = float(np.mean(acfs)) if acfs else 0.0
        else:
            # YrA unavailable — fall back to component-trace lag-1 ACF as a
            # weaker but still component-level proxy.
            acfs = []
            for i in range(num_comps):
                x = C[i] - C[i].mean()
                denom = np.dot(x, x) + 1e-9
                if len(x) > 1:
                    acfs.append(abs(np.dot(x[:-1], x[1:]) / denom))
            Phi = float(np.mean(acfs)) if acfs else 0.0

    # =========================================================================
    # N — Yield relative to patch-geometry capacity (light tie-breaker)
    # =========================================================================
    # Use the same patch-geometry-derived capacity as in the parameter design:
    # n_patches × K, not a neuron-area formula (which introduced a spurious ×4
    # overcounting when gSig was used as radius rather than half-width).
    K      = int(params.get("K", 1))
    d1, d2 = dims
    patch_w    = max(1, 2 * rf + 1)
    patch_step = max(1, patch_w - stride)
    n_patches  = (
        max(1, int(np.ceil(d1 / patch_step)))
        * max(1, int(np.ceil(d2 / patch_step)))
    )
    capacity = max(1, n_patches * K)
    N = float(np.log1p(num_comps / capacity))

    # =========================================================================
    # Composite
    # =========================================================================
    S = (
          2.0 * Q
        + 1.0 * F
        - 1.5 * D
        - 1.0 * Psi
        - 0.5 * B
        - 1.0 * Phi
        + 0.5 * N
    )

    return {
        "n_neurons":        num_comps,
        "Q":                float(Q),
        "F":                float(F),
        "D":                float(D),
        "Psi":              float(Psi),
        "B":                float(B),
        "Phi":              float(Phi),
        "N":                float(N),
        "composite_score":  float(S),
    }

# =============================================================================
# BAYESIAN TUNING
# =============================================================================


def bayesian_tune(mmap_path, dims, mask, tag="calib"):
    """
    Gaussian-process Bayesian optimisation over the reparametrised search
    space (gSig, rf, overlap_fraction, occupancy_frac, nb, decay_time).
    stride and K are derived inside objective() from the primary parameters
    and are never direct optimisation variables.

    nb_patch is always tied to nb so that background rank is consistent
    between the patch-wise initialisation and the full-FOV refit.

    do_mc defaults to not ARGS.no_mc so --no-mc suppresses motion correction
    inside tuning trials the same way it does in the main pipeline.
    """
    print(f"\n{'='*60}\nBAYESIAN TUNE ({tag})\n{'='*60}")

    # Handle motion correction once upfront before starting trial iterations
    if not ARGS.no_mc:
        working_mmap = precompute_mc(mmap_path)
    else:
        print("  [MOTION CORRECTION] Bypassed via --no-mc flag", flush=True)
        working_mmap = mmap_path

    trial_log = []

    def objective(params_list):
        tp = dict(zip(PARAM_NAMES, params_list))

        # ---- re-derive constrained / integer params -------------------------
        tp["gSig"] = int(tp["gSig"])
        tp["rf"]   = max(int(tp["rf"]), 2 * tp["gSig"])

        # Constraint 1: stride from rf × overlap_fraction
        tp["stride"] = max(2, int(tp["rf"] * 2 * tp["overlap_fraction"]))

        # Constraint 2: K from occupancy_frac, rf, gSig
        tp["K"] = max(
            1,
            round(
                tp["occupancy_frac"]
                * (2 * tp["rf"] + 1) ** 2
                / (2 * tp["gSig"] + 1) ** 2
            ),
        )

        # Tie nb_patch to nb so background rank is consistent across both
        # the patch-wise fit and the full-FOV refit.
        tp["nb_patch"] = tp["nb"]

        num = len(trial_log) + 1
        print(f"  Trial {num:2d}: {tp} ...", end=" ", flush=True)
        
        # do_mc is False because working_mmap is either pre-corrected or --no-mc was used
        cnmf_obj, rt, fname_used = run_cnmf(
            tp,
            working_mmap
        )

        metrics = score_run(cnmf_obj, fname_used, dims, mask, tp)

        metrics.update(tp)
        metrics["runtime_s"] = round(rt, 1)
        trial_log.append(metrics)

        n_post = metrics.get("n_neurons", 0)
        S = metrics["composite_score"]
        print(f"kept={n_post:3d}  composite={S:+.4f}  t={rt:.0f}s")

        # gp_minimize minimises; return large positive on non-finite scores
        return -S if np.isfinite(S) else 1e4

    print(f"\n{'='*60}\nBAYESIAN TUNE ({tag})\n{'='*60}")

    opt_result = gp_minimize(
        objective,
        SEARCH_SPACE,
        n_calls=ARGS.n_calls,
        n_initial_points=min(ARGS.n_initial, ARGS.n_calls),
        random_state=42,
        verbose=False,
    )

    df = pd.DataFrame(trial_log)
    df.to_csv(str(OUTPUT_DIR / f"tune_{tag}_log.csv"), index=False)

    df_sorted = df.sort_values("composite_score", ascending=False)
    best_full = df_sorted.iloc[0].to_dict()

    # Filter the best parameters to return only the requested metrics
    best = {
        "gSig": int(best_full.get("gSig")),
        "rf": int(best_full.get("rf")),
        "nb": int(best_full.get("nb")),
        "nb_patch": int(best_full.get("nb_patch")),
        "decay_time": float(best_full.get("decay_time")),
        "stride": int(best_full.get("stride")),
        "K": int(best_full.get("K"))
    }

    print(f"\nBest params ({tag}):")
    for k, v in best.items():
        print(f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_convergence(opt_result, ax=ax)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"convergence_{tag}.png"), dpi=120)
    plt.close(fig)

    return best, df, opt_result


# =============================================================================
# OPTIMIZED PLOTS
# =============================================================================

def generate_optimized_plots(images, cnm, output_dir=OUTPUT_DIR):
    print("[STAGE: plotting] Generating diagnostic contour and trace plots...")

    # --- 1. Spatial Footprints (Contours) ---
    Cn = cm.local_correlations(images, swap_dim=False)
    Cn[np.isnan(Cn)] = 0

    fig_spatial, ax_spatial = plt.subplots(figsize=(10, 10), dpi=100)
    plot_contours(cnm.estimates.A, Cn, ax=ax_spatial, display_numbers=False, thr=0.9)
    ax_spatial.set_title("Optimized Spatial Footprints (Contours)")
    ax_spatial.axis("off")

    contour_path = os.path.join(output_dir, "optimized_contours.png")
    fig_spatial.savefig(contour_path, bbox_inches="tight")
    plt.close(fig_spatial)
    
    # --- 2. Extract / Calculate Traces ---
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
# MAIN
# =============================================================================


def main():
    t_start = time.time()

    fmt, files, sample_shape, _nplanes = discover(
        ARGS.data_dir, ARGS.format_override, ARGS.n_planes
    )
    z_idx = _nplanes // 2 if ARGS.z_index is None else ARGS.z_index

    raw = load_movie(
        ARGS.data_dir,
        fmt,
        files,
        sample_shape,
        z_index=z_idx,
        max_frames=ARGS.max_frames,
        n_planes=_nplanes,
    )
    data, mask, prep_info = preprocess_movie(raw, label="calib")

    dims = data.shape[1:]
    mmap_path = array_to_memmap(data, WORK_DIR / "calib_movie")

    best_params, trials_df, _ = bayesian_tune(mmap_path, dims, mask, tag="calib")

    # Fit final model with best params
    print(
        "\n[STAGE: Validation] Re-fitting final model with best parameters to save outputs..."
    )
    final_cnmf, _, final_mmap = run_cnmf(best_params, mmap_path)

    if (
        final_cnmf
        and final_cnmf.estimates.A is not None
        and final_cnmf.estimates.A.shape[1] > 0
    ):
        Yr, dims_out, num_frames = caiman.mmapping.load_memmap(final_mmap)
        images = np.reshape(Yr.T, [num_frames] + list(dims_out), order="F")
        generate_optimized_plots(images, final_cnmf, OUTPUT_DIR)

    summary = {
        "run_name": ARGS.run_name,
        "format": fmt,
        "data_dir": str(ARGS.data_dir),
        "n_files": len(files),
        "sample_shape": list(sample_shape),
        "z_index": z_idx,
        "n_planes": _nplanes if _nplanes is not None else ARGS.n_planes,
        "resolution": ARGS.resolution,
        "stripe_removal": not ARGS.no_stripe,
        "n_calls": ARGS.n_calls,
        "n_initial": ARGS.n_initial,
        "best_params": best_params,
        "best_trial_metrics": trials_df.sort_values("composite_score", ascending=False)
        .iloc[0]
        .to_dict(),
    }

    with open(str(OUTPUT_DIR / "calibration_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    elapsed_min = (time.time() - t_start) / 60.0
    
    print(f"\n{'='*70}")
    print(f"DONE  |  calibrate_cnmf  |  {ARGS.run_name}  |  {elapsed_min:.1f} min")
    print(f"Best params saved -> {OUTPUT_DIR}/calibration_summary.json")
    print(f"{'='*70}")


# =========================================================
# Execution Hook
# =========================================================
if __name__ == "__main__":
    main()
