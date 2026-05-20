#!/usr/bin/env python3
"""
p3_cross_validate.py  —  Phase 3: CNMF Cross-Validation Framework

Four validation modes for testing CNMF parameter generalization:

  time-split        Tune & test within same file/z-plane, split by time
  plane-split       Tune on one z-plane, test on all other z-planes (same file)
  file-plane-split  Tune on file A one z-plane, test on file B same z-plane
  file-split        Tune on file A (middle z), test on file B all z-planes

Full 2048×2048 resolution — NO downsampling.
Search space scaled for 2048×2048.
Composite score reformulated to be positive (higher = better).

Usage examples:

  source ~/Documents/fishBrain_Kiitan/Fish_Brain_Dynamics/venv/bin/activate

  # (a) time-split — tune on first half of frames, test on second half
  python p3_cross_validate.py --mode time-split \\
      --data-dir <TASK_DIR> --z-index 3 --run-name Task5_timesplit_z3

  # (b) plane-split — tune on z=3, test on z=0,1,2,4,5,6
  python p3_cross_validate.py --mode plane-split \\
      --data-dir <TASK_DIR> --tune-z 3 --run-name Task5_planesplit

  # (c) file-plane-split — tune on Task4 z=3, test on Task5 z=3
  python p3_cross_validate.py --mode file-plane-split \\
      --tune-dir <TASK4_DIR> --test-dir <TASK5_DIR> \\
      --z-index 3 --run-name T4vsT5_z3

  # (d) file-split — tune on Task4, test on Task5 all z-planes
  python p3_cross_validate.py --mode file-split \\
      --tune-dir <TASK4_DIR> --test-dir <TASK5_DIR> \\
      --run-name T4vsT5_allZ

  # Quick smoke test (2 trials)
  python p3_cross_validate.py --mode time-split \\
      --data-dir <TASK_DIR> --z-index 3 --run-name smoke \\
      --n-calls 2 --n-initial 2
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
        description="CNMF cross-validation: 4 modes for parameter generalization testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", required=True,
                   choices=["time-split", "plane-split", "file-plane-split", "file-split"],
                   help="Validation mode")
    p.add_argument("--run-name", required=True, help="Output folder name under results/")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="H5 folder (required for time-split, plane-split)")
    p.add_argument("--tune-dir", type=Path, default=None,
                   help="Tune-data folder (required for file-plane-split, file-split)")
    p.add_argument("--test-dir", type=Path, default=None,
                   help="Test-data folder (required for file-plane-split, file-split)")
    p.add_argument("--z-index", type=int, default=3,
                   help="Z-plane index (time-split, file-plane-split)")
    p.add_argument("--tune-z", type=int, default=3,
                   help="Z-plane to tune on (plane-split)")
    p.add_argument("--n-calls", type=int, default=20, help="Bayesian trials")
    p.add_argument("--n-initial", type=int, default=8, help="Random exploration points")
    return p.parse_args()


ARGS = parse_args()

# Validate required args per mode
if ARGS.mode in ("time-split", "plane-split"):
    if ARGS.data_dir is None:
        print("ERROR: --data-dir required for mode", ARGS.mode, file=sys.stderr)
        sys.exit(1)
elif ARGS.mode in ("file-plane-split", "file-split"):
    if ARGS.tune_dir is None or ARGS.test_dir is None:
        print("ERROR: --tune-dir and --test-dir required for mode", ARGS.mode, file=sys.stderr)
        sys.exit(1)

OUTPUT_DIR = RESULTS_ROOT / ARGS.run_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR = OUTPUT_DIR / "_work"
WORK_DIR.mkdir(parents=True, exist_ok=True)

N_CALLS = ARGS.n_calls
N_INITIAL = ARGS.n_initial


# =============================================================================
# IMPORTS
# =============================================================================

import h5py
import numpy as np
import pandas as pd
import tifffile
from skimage.morphology import convex_hull_image
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


print(f"Mode       : {ARGS.mode}")
print(f"Run name   : {ARGS.run_name}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Trials     : n_calls={N_CALLS}  n_initial={N_INITIAL}")


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def discover_h5(data_dir: Path) -> tuple[list[str], int, int, int]:
    """Glob and sort tp-*.lux.h5 files. Returns (file_list, Z, H, W)."""
    pattern = str(data_dir / "tp-*_ch-*_st-*_obj-*_cam-*.lux.h5")
    files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"tp-0-(\d+)", p).group(1)),
    )
    if not files:
        print(f"ERROR: No .lux.h5 files matching {pattern}", file=sys.stderr)
        sys.exit(1)
    with h5py.File(files[0], "r") as fh:
        z, h, w = fh["Data"].shape
    print(f"  Found {len(files)} files  (Z={z}, H={h}, W={w})")
    print(f"  First: {Path(files[0]).name}")
    print(f"  Last:  {Path(files[-1]).name}")
    return files, z, h, w


def load_plane(h5_files: list[str], z_index: int) -> np.ndarray:
    """Load one z-plane across all timepoints → (T, H, W) float32."""
    T = len(h5_files)
    with h5py.File(h5_files[0], "r") as fh:
        _, h, w = fh["Data"].shape
    data = np.zeros((T, h, w), dtype=np.float32)
    for i, fp in enumerate(h5_files):
        with h5py.File(fp, "r") as fh:
            data[i] = fh["Data"][z_index].astype(np.float32)
    return data


def preprocess(data: np.ndarray, label: str = "") -> np.ndarray:
    """Column-median stripe removal (in-place). NO downsampling."""
    col_median = np.median(data, axis=(0, 1), keepdims=True)
    data = np.clip(data - col_median, 0, None).astype(np.float32)
    print(f"  Stripe removal done{f' ({label})' if label else ''}: shape={data.shape}")
    return data, col_median


def make_memmap(data: np.ndarray, tag: str) -> tuple[str, tuple[int, int], int]:
    """Save array to tiff → CaImAn memmap. Returns (mmap_path, dims, T)."""
    tif = str(WORK_DIR / f"{tag}.tif")
    mmap_base = str(WORK_DIR / f"{tag}_mmap")
    tifffile.imwrite(tif, data)
    mmap_path = caiman.mmapping.save_memmap(
        [tif], base_name=mmap_base, order="C", border_to_0=0,
    )
    T, H, W = data.shape
    return mmap_path, (H, W), T


def save_preprocess_plot(data: np.ndarray, col_median: np.ndarray, label: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(data[data.shape[0] // 2], cmap="gray")
    axes[0].set_title(f"Frame {data.shape[0]//2} — {label}")
    axes[0].axis("off")
    axes[1].imshow(col_median.reshape(1, -1), cmap="gray", aspect="auto")
    axes[1].set_title("Removed stripe pattern")
    axes[1].set_xlabel("Column index")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"preprocess_{label.replace(' ', '_')}.png"), dpi=100)
    plt.close(fig)


# =============================================================================
# CNMF HELPERS — full 2048×2048 resolution
# =============================================================================

BASE_PARAMS = {
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
    "min_SNR": 1.5,
    "rval_thr": 0.85,
    "del_duplicates": True,
    "ssub": 1,
    "tsub": 1,
    "only_init": False,
    "pw_rigid": True,
    "max_shifts": (12, 12),
    "strides": (192, 192),
    "overlaps": (96, 96),
    "max_deviation_rigid": 3,
}

SEARCH_SPACE = [
    Integer(8, 20, name="gSig"),
    Integer(4, 16, name="gSig_filt"),
    Real(0.4, 0.85, name="min_corr"),
    Integer(3, 12, name="min_pnr"),
    Categorical([100, 160, 240, 320], name="rf"),
    Categorical([1, 2], name="p"),
]
PARAM_NAMES = [s.name for s in SEARCH_SPACE]


def run_cnmf(params_override: dict, fname_mmap: str,
             do_mc: bool = True, do_filter: bool = True):
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
        if do_filter and cnmf_obj.estimates.A.shape[1] > 0:
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


def score_run(cnmf_obj, Yr, dims: tuple[int, int], stability: float = 0.0) -> dict:
    """
    Composite score (positive, higher = better):
      1.0*(1 - recon_error) + 0.5*compactness - 0.3*log(1+sparsity) + 1.0*stability
    """
    sentinel = {
        "n_neurons": 0, "recon_error": 1.0, "spatial_compactness": 0.0,
        "trace_sparsity": float("inf"), "stability": stability,
        "composite_score": -float("inf"),
    }
    if cnmf_obj is None:
        return sentinel

    A = cnmf_obj.estimates.A
    C = cnmf_obj.estimates.C
    n = A.shape[1]
    if n == 0:
        return sentinel

    Y_hat = A @ C
    b = getattr(cnmf_obj.estimates, "b", None)
    f_bg = getattr(cnmf_obj.estimates, "f", None)
    if b is not None and f_bg is not None and b.shape[1] > 0:
        Y_hat = Y_hat + b @ f_bg
    recon_error = float(
        np.linalg.norm(Yr - Y_hat, "fro") / (np.linalg.norm(Yr, "fro") + 1e-9))

    H_val, W_val = dims
    comp_list = []
    for i in range(n):
        fp = np.asarray(A[:, i].todense()).flatten().reshape(H_val, W_val)
        binary = fp > (fp.max() * 0.2)
        if binary.sum() < 5:
            continue
        try:
            hull = convex_hull_image(binary)
            comp_list.append(float(binary.sum()) / float(hull.sum()))
        except Exception:
            pass
    spatial_compactness = float(np.mean(comp_list)) if comp_list else 0.0

    l1 = np.sum(np.abs(C), axis=1)
    l2 = np.linalg.norm(C, axis=1)
    trace_sparsity = float(np.mean(l1 / (l2 + 1e-9)))

    composite = (
        1.0 * (1.0 - recon_error)
        + 0.5 * spatial_compactness
        - 0.3 * np.log1p(trace_sparsity)
        + 1.0 * stability
    )

    return {
        "n_neurons": n, "recon_error": recon_error,
        "spatial_compactness": spatial_compactness,
        "trace_sparsity": trace_sparsity, "stability": stability,
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
# BAYESIAN TUNE PHASE
# =============================================================================

def bayesian_tune(mmap_path: str, dims: tuple[int, int],
                  tag: str = "tune") -> tuple[dict, pd.DataFrame]:
    """Run Bayesian search. Returns (best_params_dict, trials_df)."""
    Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
    trial_log: list[dict] = []

    def objective(params):
        tp = dict(zip(PARAM_NAMES, params))
        tp["stride"] = tp["rf"] // 2
        num = len(trial_log) + 1
        print(f"  Trial {num:2d}: {tp} ...", end=" ", flush=True)

        cnmf_obj, rt = run_cnmf(tp, mmap_path)
        metrics = score_run(cnmf_obj, Yr, dims)
        metrics.update(tp)
        metrics["runtime_s"] = round(rt, 1)
        trial_log.append(metrics)

        print(
            f"neurons={metrics['n_neurons']:3d}  "
            f"composite={metrics['composite_score']:+.4f}  "
            f"t={rt:.0f}s"
        )
        score = -metrics["composite_score"]
        if not np.isfinite(score):
            score = 1e4
        return score

    print(f"\n{'='*60}")
    print(f"BAYESIAN TUNE ({tag})")
    print(f"{'='*60}")

    opt_result = gp_minimize(
        objective, SEARCH_SPACE,
        n_calls=N_CALLS, n_initial_points=min(N_INITIAL, N_CALLS),
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
        "p": int(best["p"]),
        "merge_thr": 0.85,
    }

    print(f"\nBest params ({tag}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Convergence plot
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_convergence(opt_result, ax=ax)
    ax.set_title(f"Convergence — {tag}")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"convergence_{tag}.png"), dpi=120)
    plt.close(fig)

    # Param scatter
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, pname in zip(axes.flat, PARAM_NAMES):
        ax.scatter(df[pname], df["composite_score"], alpha=0.6, s=30)
        ax.set_xlabel(pname)
        ax.set_ylabel("Composite")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Param vs composite — {tag}")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / f"param_vs_score_{tag}.png"), dpi=120)
    plt.close(fig)

    return best_params, df


# =============================================================================
# TEST PHASE — apply params, evaluate, save plots
# =============================================================================

def test_cnmf(best_params: dict, mmap_path: str, data: np.ndarray,
              dims: tuple[int, int], T: int, label: str,
              tune_A=None) -> dict:
    """Run CNMF with fixed params, save contours + traces, return metrics."""
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")

    Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
    cnmf_obj, rt = run_cnmf(best_params, mmap_path)

    stability = 0.0
    if tune_A is not None and cnmf_obj is not None:
        stability = compute_stability(tune_A, cnmf_obj.estimates.A)

    metrics = score_run(cnmf_obj, Yr, dims, stability=stability)
    metrics["label"] = label
    metrics["runtime_s"] = round(rt, 1)

    n = metrics["n_neurons"]
    print(f"  Neurons: {n}  composite: {metrics['composite_score']:+.4f}  "
          f"stability: {stability:.3f}  t={rt:.0f}s")

    if cnmf_obj is not None and n > 0:
        safe_label = label.replace(" ", "_").replace("/", "_")

        # Contour plot
        mean_frame = data.mean(axis=0)
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(mean_frame, cmap="gray")
        ax.set_title(f"{label}: {n} neurons", fontsize=10)
        ax.axis("off")
        H_val, W_val = dims
        for i in range(n):
            fp = np.asarray(cnmf_obj.estimates.A[:, i].todense()).flatten().reshape(H_val, W_val)
            if fp.max() == 0:
                continue
            ax.contour(fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.4, alpha=0.8)
        plt.tight_layout()
        plt.savefig(str(OUTPUT_DIR / f"contours_{safe_label}.png"), dpi=150)
        plt.close(fig)

        # Sample traces
        traces = cnmf_obj.estimates.C
        n_plot = min(8, traces.shape[0])
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
        plt.savefig(str(OUTPUT_DIR / f"traces_{safe_label}.png"), dpi=120)
        plt.close(fig)

        np.save(str(OUTPUT_DIR / f"traces_{safe_label}.npy"), traces)

    return metrics


# =============================================================================
# MODE IMPLEMENTATIONS
# =============================================================================

def mode_time_split():
    """(a) Tune on first half of frames, test on second half. Same file, same z-plane."""
    z = ARGS.z_index
    print(f"\nMODE: time-split  |  z={z}")

    h5_files, Z, H, W = discover_h5(ARGS.data_dir)
    assert z < Z, f"z-index {z} >= Z={Z}"

    print(f"\nLoading z-plane {z}...")
    raw = load_plane(h5_files, z)
    data, col_med = preprocess(raw, f"z{z}")
    save_preprocess_plot(data, col_med, f"z{z}")

    T_full = data.shape[0]
    mid = T_full // 2
    tune_data = data[:mid]
    test_data = data[mid:]
    print(f"  Tune: frames 0–{mid-1} ({mid} frames)")
    print(f"  Test: frames {mid}–{T_full-1} ({T_full - mid} frames)")

    dims = (H, W)

    tune_mmap, _, T_tune = make_memmap(tune_data, "tune_half")
    test_mmap, _, T_test = make_memmap(test_data, "test_half")

    best_params, df_tune = bayesian_tune(tune_mmap, dims, tag="time_split")

    # Run on tune half to get footprints for stability comparison
    print("\nRe-running best params on tune half for stability reference...")
    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune else None

    # Test on second half
    test_metrics = test_cnmf(
        best_params, test_mmap, test_data, dims, T_test,
        label=f"test_half (frames {mid}–{T_full-1})",
        tune_A=tune_A,
    )

    # Also test on full movie
    full_mmap, _, T_all = make_memmap(data, "full_movie")
    full_metrics = test_cnmf(
        best_params, full_mmap, data, dims, T_all,
        label="full_movie",
        tune_A=tune_A,
    )

    save_summary("time-split", best_params,
                 {"tune_half": score_run(cnmf_tune, *_load_yr(tune_mmap), dims) if cnmf_tune else {},
                  "test_half": test_metrics, "full_movie": full_metrics},
                 extra={"z_index": z, "T_total": T_full, "T_tune": mid,
                        "T_test": T_full - mid, "data_dir": str(ARGS.data_dir)})


def mode_plane_split():
    """(b) Tune on one z-plane, test on all other z-planes. Same file."""
    tune_z = ARGS.tune_z
    print(f"\nMODE: plane-split  |  tune_z={tune_z}")

    h5_files, Z, H, W = discover_h5(ARGS.data_dir)
    assert tune_z < Z, f"tune-z {tune_z} >= Z={Z}"
    dims = (H, W)

    # Load and tune on the reference z-plane
    print(f"\nLoading tune z-plane {tune_z}...")
    tune_raw = load_plane(h5_files, tune_z)
    tune_data, tune_cm = preprocess(tune_raw, f"tune_z{tune_z}")
    save_preprocess_plot(tune_data, tune_cm, f"tune_z{tune_z}")

    tune_mmap, _, T_tune = make_memmap(tune_data, f"tune_z{tune_z}")
    best_params, df_tune = bayesian_tune(tune_mmap, dims, tag=f"z{tune_z}")

    # Get tune footprints for stability comparison
    print(f"\nRe-running best params on z={tune_z} for reference footprints...")
    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune else None

    # Test on every other z-plane
    all_metrics = {}
    for z in range(Z):
        print(f"\nLoading test z-plane {z}...")
        test_raw = load_plane(h5_files, z)
        test_data, test_cm = preprocess(test_raw, f"test_z{z}")
        if z != tune_z:
            save_preprocess_plot(test_data, test_cm, f"test_z{z}")
        test_mmap, _, T_test = make_memmap(test_data, f"test_z{z}")

        m = test_cnmf(
            best_params, test_mmap, test_data, dims, T_test,
            label=f"z{z}" + (" (tune)" if z == tune_z else ""),
            tune_A=tune_A if z != tune_z else None,
        )
        all_metrics[f"z{z}"] = m

    # Summary table
    rows = []
    for zname, m in all_metrics.items():
        rows.append({
            "z_plane": zname, "n_neurons": m["n_neurons"],
            "composite": m["composite_score"],
            "stability_vs_tune": m.get("stability", 0.0),
            "recon_error": m["recon_error"],
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(str(OUTPUT_DIR / "plane_split_summary.csv"), index=False)
    print(f"\n{'='*60}")
    print("PLANE-SPLIT SUMMARY")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))

    save_summary("plane-split", best_params, all_metrics,
                 extra={"tune_z": tune_z, "Z": Z,
                        "data_dir": str(ARGS.data_dir)})


def mode_file_plane_split():
    """(c) Tune on file A z-plane, test on file B same z-plane."""
    z = ARGS.z_index
    print(f"\nMODE: file-plane-split  |  z={z}")

    print("\n--- Tune data ---")
    tune_files, Z_t, H, W = discover_h5(ARGS.tune_dir)
    print("\n--- Test data ---")
    test_files, Z_te, _, _ = discover_h5(ARGS.test_dir)
    assert z < Z_t and z < Z_te
    dims = (H, W)

    print(f"\nLoading tune file z={z}...")
    tune_raw = load_plane(tune_files, z)
    tune_data, tune_cm = preprocess(tune_raw, "tune")
    save_preprocess_plot(tune_data, tune_cm, "tune")

    tune_mmap, _, _ = make_memmap(tune_data, "tune")
    best_params, _ = bayesian_tune(tune_mmap, dims, tag="tune")

    print(f"\nRe-running best params on tune data for reference footprints...")
    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune else None

    print(f"\nLoading test file z={z}...")
    test_raw = load_plane(test_files, z)
    test_data, test_cm = preprocess(test_raw, "test")
    save_preprocess_plot(test_data, test_cm, "test")
    test_mmap, _, T_test = make_memmap(test_data, "test")

    test_metrics = test_cnmf(
        best_params, test_mmap, test_data, dims, T_test,
        label="test_file", tune_A=tune_A,
    )

    save_summary("file-plane-split", best_params,
                 {"test_file": test_metrics},
                 extra={"z_index": z, "tune_dir": str(ARGS.tune_dir),
                        "test_dir": str(ARGS.test_dir)})


def mode_file_split():
    """(d) Tune on file A (middle z), test on file B all z-planes."""
    print(f"\nMODE: file-split")

    print("\n--- Tune data ---")
    tune_files, Z_t, H, W = discover_h5(ARGS.tune_dir)
    print("\n--- Test data ---")
    test_files, Z_te, _, _ = discover_h5(ARGS.test_dir)
    dims = (H, W)

    tune_z = Z_t // 2
    print(f"\nTuning on middle z-plane (z={tune_z}) of tune file...")
    tune_raw = load_plane(tune_files, tune_z)
    tune_data, tune_cm = preprocess(tune_raw, f"tune_z{tune_z}")
    save_preprocess_plot(tune_data, tune_cm, f"tune_z{tune_z}")

    tune_mmap, _, _ = make_memmap(tune_data, "tune")
    best_params, _ = bayesian_tune(tune_mmap, dims, tag="tune")

    cnmf_tune, _ = run_cnmf(best_params, tune_mmap)
    tune_A = cnmf_tune.estimates.A if cnmf_tune else None

    all_metrics = {}
    for z in range(Z_te):
        print(f"\nLoading test file z={z}...")
        test_raw = load_plane(test_files, z)
        test_data, test_cm = preprocess(test_raw, f"test_z{z}")
        save_preprocess_plot(test_data, test_cm, f"test_z{z}")
        test_mmap, _, T_test = make_memmap(test_data, f"test_z{z}")

        m = test_cnmf(
            best_params, test_mmap, test_data, dims, T_test,
            label=f"test_z{z}", tune_A=tune_A,
        )
        all_metrics[f"z{z}"] = m

    rows = []
    for zname, m in all_metrics.items():
        rows.append({
            "z_plane": zname, "n_neurons": m["n_neurons"],
            "composite": m["composite_score"],
            "stability_vs_tune": m.get("stability", 0.0),
            "recon_error": m["recon_error"],
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(str(OUTPUT_DIR / "file_split_summary.csv"), index=False)
    print(f"\n{'='*60}")
    print("FILE-SPLIT SUMMARY")
    print(f"{'='*60}")
    print(df_summary.to_string(index=False))

    save_summary("file-split", best_params, all_metrics,
                 extra={"tune_dir": str(ARGS.tune_dir),
                        "test_dir": str(ARGS.test_dir),
                        "tune_z": tune_z, "Z": Z_te})


# =============================================================================
# OUTPUT
# =============================================================================

def _load_yr(mmap_path):
    Yr, _, _ = caiman.mmapping.load_memmap(mmap_path)
    return (Yr,)


def save_summary(mode: str, best_params: dict, test_results: dict,
                 extra: dict | None = None):
    summary = {
        "mode": mode,
        "run_name": ARGS.run_name,
        "best_params": best_params,
        "n_calls": N_CALLS,
        "tests": {},
    }
    for label, m in test_results.items():
        if isinstance(m, dict):
            summary["tests"][label] = {
                k: v for k, v in m.items()
                if k not in ("label",) and not isinstance(v, (np.ndarray,))
            }
    if extra:
        summary.update(extra)

    with open(str(OUTPUT_DIR / "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nSaved summary.json → {OUTPUT_DIR}/")


# =============================================================================
# MAIN DISPATCH
# =============================================================================

MODE_MAP = {
    "time-split": mode_time_split,
    "plane-split": mode_plane_split,
    "file-plane-split": mode_file_plane_split,
    "file-split": mode_file_split,
}

t_start = time.time()
MODE_MAP[ARGS.mode]()
elapsed = time.time() - t_start

print(f"\n{'='*60}")
print(f"DONE  —  {ARGS.mode}  |  {ARGS.run_name}  |  {elapsed/60:.1f} min")
print(f"{'='*60}")
