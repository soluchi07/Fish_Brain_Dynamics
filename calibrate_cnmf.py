#!/usr/bin/env python3
"""
calibrate_cnmf.py  —  Standalone Bayesian calibration of CNMF parameters

Extracted from p4_universal.py's `bayesian_tune()` so parameter calibration
can be run on its own, without going through a full time-split / plane-split
/ file-split pipeline. Useful for occasional re-tuning when new data comes in
or the quality-filter thresholds change.

It reuses the same format auto-detection, preprocessing (resolution, stripe
removal), CNMF runner, quality filters, and composite score as
p4_universal.py, so parameters tuned here are directly compatible with it.

Supported input formats (auto-detected):
  multi-tp     : many tp-*.lux.h5 files, each (Z, H, W)
  multi-cam    : many Cam_long_*.lux.h5 files, each (1, H, W)
  single-movie : one big Cam_long_*.lux.h5 file, shape (T, H, W)
  interleaved  : one *.lux*.h5 file, shape (T*Z, H, W), planes strided
  legacy       : one .h5 file with shape (T, H, W)

Usage examples:

  # Basic calibration run at 512x512, 15 Bayesian trials
  python calibrate_cnmf.py --data-dir "/path/to/dataset_folder" \\
      --run-name calib_13iii26_task1 --resolution 512 --n-calls 15

  # Smoke test (2 trials)
  python calibrate_cnmf.py --data-dir <DIR> --run-name smoke \\
      --resolution 512 --n-calls 2 --n-initial 2

  # Calibrate on a single Z-plane extracted from an interleaved stack
  python calibrate_cnmf.py --data-dir <DIR> --run-name calib_z3 \\
      --n-planes 7 --z-index 3 --resolution 512 --n-calls 15
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
except Exception:
    pass


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Standalone Bayesian calibration of CNMF parameters (single dataset, "
                     "no train/test split — for occasional parameter re-tuning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--run-name", required=True, help="Output folder name under results/")
    p.add_argument("--data-dir", type=Path, required=True, help="Folder containing the dataset")

    # Z-plane selection
    p.add_argument("--z-index", type=int, default=None,
                   help="Z-plane index to extract (default: middle)")
    p.add_argument("--n-planes", type=int, default=None,
                   help="Number of Z-planes interleaved in a single-movie file "
                        "(e.g. 7 when 700-rep x 7-plane = 4900 total frames). "
                        "Strides the T axis: keeps frames z_index, z_index+n_planes, ... ")

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

    # Bayesian search
    p.add_argument("--n-calls", type=int, default=15)
    p.add_argument("--n-initial", type=int, default=5)
    p.add_argument("--n-workers", type=int, default=None,
                   help="CPU workers for CNMF patch processing (default: --pin-cpus count, else cpu_count - 1)")
    p.add_argument("--tune-p", type=int, default=1, choices=[1, 2],
                   help="AR order during Bayesian tuning trials (default: 1, fast)")
    p.add_argument("--final-p", type=int, default=1, choices=[1, 2],
                   help="AR order recorded in best_params for downstream final runs (default: 1)")

    # Trace SNR threshold fed to CaImAn's own component evaluation
    p.add_argument("--min-snr-trace", type=float, default=1.5,
                   help="Reject components with trace SNR below this (used by CaImAn's "
                        "own evaluate_components/select_components, and by refit)")

    # Temp files
    p.add_argument("--keep-temp", action="store_true",
                   help="Keep intermediate .tif files in _work/ after memmap creation (default: clean up)")

    return p.parse_args()



ARGS = parse_args()


N_WORKERS = ARGS.n_workers

OUTPUT_DIR = RESULTS_ROOT / ARGS.run_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = OUTPUT_DIR / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMPORTS (heavy deps loaded after CLI/dir setup, same as p4_universal.py)
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import tifffile
import skimage.transform
from skimage.morphology import convex_hull_image, binary_closing, binary_opening, disk, remove_small_objects
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence

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


print("=" * 70)
print(f"calibrate_cnmf.py  |  run={ARGS.run_name}")
print("=" * 70)
print(f"Data dir   : {ARGS.data_dir}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Resolution : {ARGS.resolution}")
print(f"Trials     : n_calls={ARGS.n_calls}  n_initial={ARGS.n_initial}")
print(f"Workers={N_WORKERS}")
if ARGS.n_planes:
    _default_z = ARGS.z_index if ARGS.z_index is not None else ARGS.n_planes // 2
    print(f"Z-planes   : {ARGS.n_planes} interleaved  (extracting z={_default_z})")


# =============================================================================
# FORMAT DETECTION  (identical to p4_universal.py)
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
        n_z = read_n_planes(cam_files[0])
        if n_z > 1:
            return "interleaved", cam_files, shape
        return "single-movie", cam_files, shape

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
    """Detect or override format. Print result. Returns (fmt, files, shape, n_planes_detected)."""
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
# UNIVERSAL LOADER  (identical to p4_universal.py)
# =============================================================================

def load_movie(folder: Path, fmt: str, files: list[str], shape: tuple,
               z_index: Optional[int] = None,
               max_frames: Optional[int] = None,
               n_planes: Optional[int] = None) -> np.ndarray:
    """Build (T, H, W) float32 movie regardless of source format."""
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


# =============================================================================
# PREPROCESSING  (identical to p4_universal.py)
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


def preprocess_movie(data: np.ndarray, label: str = "") -> tuple[np.ndarray, dict]:
    """Full preprocessing pipeline. Returns (preprocessed_movie, metadata_dict)."""
    info = {"original_shape": tuple(data.shape)}
    print(f"\n[preprocess {label}]  input shape={data.shape}")

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

    if not ARGS.no_stripe:
        data, col_median = stripe_remove(data)
        info["stripe_removed"] = True
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
# CNMF CONFIG (resolution-aware, identical to p4_universal.py)
# =============================================================================

def get_search_space() -> list:
    """Bayesian search space scaled to resolution.

    Notes:
    - p is excluded from tuning: always use p=1 during search (fast), p=2
      only affects temporal AR fitting cost, not spatial components.
      Apply p=2 downstream via --final-p if needed.
    - rf=320 removed at full res: creates patches too large to run in
      reasonable time on a full-FOV movie.
    - min_corr/min_pnr floors raised: values of 0.4/3 flood CNMF with
      spurious candidates and dominate runtime without improving neurons found.
    """
    if ARGS.resolution == "512":
        return [
            Integer(2, 5, name="gSig"),
            Integer(1, 4, name="gSig_filt"),
            Real(0.5, 0.85, name="min_corr"),
            Integer(5, 12, name="min_pnr"),
            Categorical([25, 40, 60, 80], name="rf"),
        ]
    if ARGS.resolution == "1024":
        return [
            Integer(4, 10, name="gSig"),
            Integer(2, 8, name="gSig_filt"),
            Real(0.5, 0.85, name="min_corr"),
            Integer(5, 12, name="min_pnr"),
            Categorical([50, 80, 120, 160], name="rf"),
        ]
    return [
        Integer(4, 10, name="gSig"),
        Integer(4, 10, name="gSig_filt"),
        Real(0.5, 0.85, name="min_corr"),
        Integer(5, 12, name="min_pnr"),
        # Floor raised from 100 to 160: rf=100 on a 2048x2048 canvas creates
        # ~1700 overlapping patches, and on longer recordings (more raw
        # corr_pnr candidates before merging) patch-consolidation inside
        # fit_file() can try to densify a huge (pixels x components) matrix
        # and blow past available RAM. Fewer, larger patches avoid this.
        Categorical([160, 200, 240], name="rf"),
    ]


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
        # TODO: this MUST match the "fr" used in p4_universal.py's
        # get_base_params() and the real acquisition frame rate — a mismatch
        # here means params are calibrated against one timescale and deployed
        # against another.
        "fr": 30,
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


SEARCH_SPACE = get_search_space()
PARAM_NAMES = [s.name for s in SEARCH_SPACE]
BASE_PARAMS = get_base_params()


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
            pass
    return mmap_path


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
    mc_keys = ('max_shifts', 'strides', 'overlaps', 'max_deviation_rigid', 'pw_rigid')
    mc_params = {k: BASE_PARAMS[k] for k in mc_keys if k in BASE_PARAMS}
    t0 = time.time()
    print("  [precomputing motion correction — runs once for all trials]", flush=True)
    _, cluster, n_processes = cm.cluster.setup_cluster(
        backend="multiprocessing", n_processes=N_WORKERS, ignore_preexisting=False
    )
    print(f"Successfully initialized multicore processing with a pool of {n_processes} CPU cores")
    try:
        mc = MotionCorrect([fname_mmap], dview=cluster, **mc_params)
        mc.motion_correct(save_movie=True)
        pw = BASE_PARAMS.get('pw_rigid', True)
        mc_raw = mc.fname_tot_els[-1] if pw else mc.fname_tot_rig[-1]

        # Parse the true output dimensions from the filename (last d1/d2/d3/frames match).
        m = re.search(r'd1_(\d+)_d2_(\d+)_d3_(\d+).*frames_(\d+)\.mmap$', mc_raw)
        d1, d2, d3, T = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))

        # Force-load as F-order (the true MC output format) and write a clean C-order
        # mmap with an unambiguous single-'order' filename.
        out_path = str(Path(mc_raw).parent / f'Yr_mc_d1_{d1}_d2_{d2}_d3_{d3}_order_C_frames_{T}.mmap')
        src = np.memmap(mc_raw,   dtype=np.float32, mode='r',  shape=(d1 * d2 * d3, T), order='F')
        dst = np.memmap(out_path, dtype=np.float32, mode='w+', shape=(d1 * d2 * d3, T), order='C')
        np.copyto(dst, src)
        dst[~np.isfinite(dst)] = 0   # zero NaN/Inf border pixels left by MC
        del dst                       # flush to disk
    finally:
        try:
            caiman.stop_server(dview=cluster)
        except Exception:
            pass

    print(f"  [MC done in {time.time()-t0:.1f}s → {Path(out_path).name}]", flush=True)
    return out_path

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
    fname_to_use = cm.save_memmap(mc.mmap_file, 
                                    base_name='memmap_', 
                                    order='C',
                                    border_to_0=0,  # exclude borders, if that was done
                                    dview=cluster)
    print(f"  [STAGE:motion_correction] done -> {fname_to_use}", flush=True)
    return fname_to_use

def _fit_cnmf(fname_to_use: str, opts, n_processes: int, cluster):
    """Stage: core CNMF fit. Raises on failure — this is the critical path."""
    opts.change_params({'fnames': [fname_to_use]})
    
    Yr, dims, num_frames = cm.load_memmap(fname_to_use)
    images = np.reshape(Yr.T, [num_frames] + list(dims), order='F') #reshape frames in standard 3d format (T x X x Y)

    # cm.stop_server(dview=cluster) # restart cluster to clean up memory in preparation for CNMF run.
    # cluster, n_processes = _setup_cluster()

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
        
        # cm.stop_server(dview=cluster)
        # cluster, n_processes = _setup_cluster()
        
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


def score_run(cnmf_obj, Yr, dims: tuple[int, int]) -> dict:
    """
    Composite score over all neurons CaImAn accepted (post evaluate_components/
    select_components/refit — no additional geometric quality filtering here).

    composite = 1.0*(1 - recon_error)
              + 0.5*spatial_compactness
              - 0.3*log(1 + trace_sparsity)
              + 0.001*log(1 + n_neurons)   # small bonus for finding more neurons
    """
    sentinel = {
        "n_neurons": 0, "recon_error": 1.0,
        "spatial_compactness": 0.0, "trace_sparsity": float("inf"),
        "composite_score": -float("inf"),
    }
    if cnmf_obj is None or cnmf_obj.estimates.A.shape[1] == 0:
        return sentinel

    A = cnmf_obj.estimates.A
    C = cnmf_obj.estimates.C
    n = A.shape[1]
    
    # Reconstruction error via Frobenius norm identity - avoids materialising
    # the (pixels x frames) dense reconstruction matrix (~11 GB at full res).
    #   ||Y - A@C||^2_F = ||Y||^2_F - 2*trace(C^T * A^T * Y) + ||A@C||^2_F
    Yr_norm_sq = float(np.linalg.norm(Yr, "fro") ** 2)
    AtYr = A.T @ Yr
    AtA = A.T @ A
    recon_norm_sq = Yr_norm_sq - 2.0 * float(np.sum(C * AtYr)) + float(np.sum(C * (AtA @ C)))

    b = getattr(cnmf_obj.estimates, "b", None)
    f_bg = getattr(cnmf_obj.estimates, "f", None)
    if b is not None and f_bg is not None and b.shape[1] > 0:
        bt_plus = b.T @ b
        bg_norm_sq = float(np.sum(f_bg * (bt_plus @ f_bg)))
        Atb = A.T @ b
        CfbgT = C @ f_bg.T
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
        + 0.001 * np.log1p(n)
    )

    return {
        "n_neurons": n,
        "recon_error": recon_error,
        "spatial_compactness": spatial_compactness,
        "trace_sparsity": trace_sparsity,
        "composite_score": float(composite),
    }


# =============================================================================
# BAYESIAN TUNING  (extracted from p4_universal.py, unchanged in behavior)
# =============================================================================

def bayesian_tune(mmap_path: str, dims: tuple[int, int], tag: str = "tune"
                  ) -> tuple[dict, pd.DataFrame]:
    """Run Bayesian search. Returns (best_params, trials_df)."""
    Yr_unregistered, _, _ = caiman.mmapping.load_memmap(mmap_path)
    trial_log: list[dict] = []

    # MC params are constant across trials — run once and reuse the output mmap
    try:
        mc_mmap = precompute_mc(mmap_path)
        mc_precomputed = True
    except Exception as exc:
        print(f"  [WARNING: MC precompute failed ({exc}), falling back to per-trial MC]", flush=True)
        mc_mmap = mmap_path
        mc_precomputed = False

    def objective(params):
        tp = dict(zip(PARAM_NAMES, params))
        tp["stride"] = tp["rf"] // 2
        tp["p"] = ARGS.tune_p  # p is fixed during tuning to keep trials fast
        num = len(trial_log) + 1
        print(f"  Trial {num:2d}: {tp} ...", end=" ", flush=True)

        cnmf_obj, rt, fname_used = run_cnmf(tp, mc_mmap, do_mc=not mc_precomputed)
        # Score against the mmap CNMF actually fit on (post-MC when do_mc=True,
        # or the precomputed registered mmap when mc_precomputed=True) rather
        # than the pre-registration Yr_unregistered loaded above — otherwise a
        # pixel-shift mismatch between the fitted data and the scored data can
        # blow up recon_error by orders of magnitude independent of how good
        # the trial's parameters actually are.
        Yr_scored = load_yr_for_scoring(fname_used, Yr_unregistered)
        metrics = score_run(cnmf_obj, Yr_scored, dims)
        metrics.update(tp)
        metrics["runtime_s"] = round(rt, 1)
        trial_log.append(metrics)
        
        n = metrics.get("n_neurons", 0)
        print(f"neurons={n:3d} composite={metrics['composite_score']:+.4f} t={rt:.0f}s")
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

    df_sorted = df.sort_values("composite_score", ascending=False)
    best = df_sorted.iloc[0]
    best_params = {
        "gSig": int(best["gSig"]),
        "gSig_filt": int(best["gSig_filt"]),
        "min_corr": float(best["min_corr"]),
        "min_pnr": int(best["min_pnr"]),
        "rf": int(best["rf"]),
        "stride": int(best["rf"]) // 2,
        "p": ARGS.final_p,  # recorded for downstream final runs; upgrade via --final-p 2
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

    return best_params, df


# =============================================================================
# MAIN — load one dataset, preprocess, calibrate, save best params
# =============================================================================

def main():
    t_start = time.time()

    fmt, files, sample_shape, _nplanes = discover(ARGS.data_dir, ARGS.format_override)

    z_index = ARGS.z_index
    if fmt == "multi-tp" and z_index is None:
        z_index = sample_shape[0] // 2
    elif fmt == "interleaved" and z_index is None:
        z_index = (_nplanes if _nplanes is not None else ARGS.n_planes) // 2
    elif ARGS.n_planes and z_index is None:
        z_index = ARGS.n_planes // 2

    raw = load_movie(ARGS.data_dir, fmt, files, sample_shape,
                     z_index=z_index, max_frames=ARGS.max_frames,
                     n_planes=_nplanes if _nplanes is not None else ARGS.n_planes)
    data, prep_info = preprocess_movie(raw, label="calib")

    dims = data.shape[1:]
    mmap_path = array_to_memmap(data, WORK_DIR / "calib_movie")

    best_params, trials_df = bayesian_tune(mmap_path, dims, tag="calib")

    summary = {
        "run_name": ARGS.run_name,
        "data_dir": str(ARGS.data_dir),
        "format": fmt,
        "n_files": len(files),
        "sample_shape": list(sample_shape),
        "z_index": z_index,
        "n_planes": _nplanes if _nplanes is not None else ARGS.n_planes,
        "resolution": ARGS.resolution,
        "stripe_removal": not ARGS.no_stripe,
        "min_snr_trace": ARGS.min_snr_trace,
        "n_calls": ARGS.n_calls,
        "n_initial": ARGS.n_initial,
        "best_params": best_params,
        "best_trial_metrics": trials_df.sort_values(
            "composite_score", ascending=False
        ).iloc[0].to_dict(),
        "preprocessing": {k: v for k, v in prep_info.items()},
    }
    with open(str(OUTPUT_DIR / "calibration_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    elapsed_min = (time.time() - t_start) / 60.0
    print(f"\n{'='*70}")
    print(f"DONE  |  calibrate_cnmf  |  {ARGS.run_name}  |  {elapsed_min:.1f} min")
    print(f"Best params saved -> {OUTPUT_DIR}/calibration_summary.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()