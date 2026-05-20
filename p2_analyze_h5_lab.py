#!/usr/bin/env python3
"""
p2_analyze_h5_lab.py  —  Lab machine version of the Phase 2 CNMF pipeline.

Designed for the abl-dell machine with multi-file .lux.h5 datasets where each
file is one timepoint with shape (Z, 2048, 2048).

Usage:
  source ~/Documents/fishBrain_Kiitan/Fish_Brain_Dynamics/venv/bin/activate

  # Task 4 (42 timepoints)
  python p2_analyze_h5_lab.py \\
      --data-dir ~/Documents/fishBrain_Kiitan/ZebraFishUMichiganProject/Army\\ Project/7iii25/2025-03-07_090808/20250307-231142_Task_4_Description_for_C \\
      --run-name 7iii25_Task4 \\
      --n-calls 30

  # Quick smoke test (2 trials, skip stability)
  python p2_analyze_h5_lab.py \\
      --data-dir <same path> \\
      --run-name 7iii25_Task4_test \\
      --n-calls 2 --n-initial 2 --skip-stability

Outputs go to:
  ~/Documents/fishBrain_Kiitan/Fish_Brain_Dynamics/results/<run-name>/
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lab CNMF Bayesian tuning for multi-file .lux.h5 data.")
    p.add_argument(
        "--data-dir", type=Path, required=True,
        help="Folder containing tp-*_ch-*_st-*_obj-*_cam-*.lux.h5 files",
    )
    p.add_argument(
        "--run-name", type=str, required=True,
        help="Short name for this run (used as output folder name under results/)",
    )
    p.add_argument("--z-method", choices=["middle", "max"], default="middle",
                   help="How to collapse Z: 'middle' picks middle plane, 'max' does max-projection")
    p.add_argument("--n-calls", type=int, default=30, help="Bayesian optimisation trials")
    p.add_argument("--n-initial", type=int, default=10, help="Random exploration points before GP")
    p.add_argument("--skip-stability", action="store_true", help="Skip split-half + final CNMF")
    return p.parse_args()


ARGS = parse_args()
OUTPUT_DIR = str(RESULTS_ROOT / ARGS.run_name)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
WORK_DIR = Path(OUTPUT_DIR) / "_work"
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
import skimage.transform
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

print(f"Run name   : {ARGS.run_name}")
print(f"Data dir   : {ARGS.data_dir}")
print(f"Z method   : {ARGS.z_method}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Trials     : n_calls={N_CALLS}  n_initial={N_INITIAL}")


# =============================================================================
# LOAD MULTI-FILE .lux.h5  →  (T, H, W)
# Each tp-0-<N>_*.lux.h5 contains Data with shape (Z, 2048, 2048).
# We glob, sort by timepoint number, and pick one Z plane (or max-project).
# =============================================================================

print("\n" + "=" * 60)
print("LOADING DATA")
print("=" * 60)

h5_pattern = str(ARGS.data_dir / "tp-*_ch-*_st-*_obj-*_cam-*.lux.h5")
h5_files = sorted(
    glob.glob(h5_pattern),
    key=lambda p: int(re.search(r"tp-0-(\d+)", p).group(1)),
)

if not h5_files:
    print(f"ERROR: No .lux.h5 files found matching {h5_pattern}", file=sys.stderr)
    sys.exit(1)

print(f"Found {len(h5_files)} timepoint files")
print(f"  First: {Path(h5_files[0]).name}")
print(f"  Last:  {Path(h5_files[-1]).name}")

# Inspect first file for Z, H, W
with h5py.File(h5_files[0], "r") as fh:
    sample_shape = fh["Data"].shape
    sample_dtype = fh["Data"].dtype

Z_FULL, H_ORIG, W_ORIG = sample_shape
T_ORIG = len(h5_files)
print(f"Per file: (Z={Z_FULL}, H={H_ORIG}, W={W_ORIG})  dtype={sample_dtype}")
print(f"Total movie: T={T_ORIG} timepoints")

if ARGS.z_method == "middle":
    Z_IDX = Z_FULL // 2
    print(f"Z method: middle plane (z={Z_IDX})")
else:
    print("Z method: max-projection across all Z")

# Stack into (T, H, W)
data_full = np.zeros((T_ORIG, H_ORIG, W_ORIG), dtype=np.float32)
for i, fpath in enumerate(h5_files):
    with h5py.File(fpath, "r") as fh:
        vol = fh["Data"][:]
        if ARGS.z_method == "middle":
            data_full[i] = vol[Z_IDX].astype(np.float32)
        else:
            data_full[i] = vol.max(axis=0).astype(np.float32)

print(f"Loaded movie: shape={data_full.shape}  dtype={data_full.dtype}")

# ── Downsample 4× to 512×512 ────────────────────────────────────────────────
TARGET_H, TARGET_W = 512, 512
data_512 = np.zeros((T_ORIG, TARGET_H, TARGET_W), dtype=np.float32)
for t in range(T_ORIG):
    data_512[t] = skimage.transform.resize(
        data_full[t], (TARGET_H, TARGET_W), anti_aliasing=True,
    )
del data_full

T, H, W = data_512.shape
DIMS = (H, W)
print(f"Downsampled: shape={data_512.shape}")

# ── Stripe removal ──────────────────────────────────────────────────────────
col_median = np.median(data_512, axis=(0, 1), keepdims=True)
data_512 = np.clip(data_512 - col_median, 0, None).astype(np.float32)
print("Stripe removal: subtracted per-column temporal median")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(data_512[T // 2], cmap="gray")
axes[0].set_title(f"Frame {T // 2} — after stripe removal")
axes[0].axis("off")
axes[1].imshow(col_median.reshape(1, -1), cmap="gray", aspect="auto")
axes[1].set_title("Removed stripe pattern")
axes[1].set_xlabel("Column index")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/preprocessing_check.png", dpi=100)
plt.close(fig)
print("Saved preprocessing_check.png")


# =============================================================================
# TIFF + MEMMAP
# =============================================================================

TIF_PATH = str(WORK_DIR / "movie_512.tif")
MMAP_BASE = str(WORK_DIR / "movie_512_mmap")

print("\nSaving float32 tiff...")
tifffile.imwrite(TIF_PATH, data_512)

print("Converting to CaImAn memmap...")
FNAME_MMAP = caiman.mmapping.save_memmap(
    [TIF_PATH], base_name=MMAP_BASE, order="C", border_to_0=0,
)
print(f"Memmap: {FNAME_MMAP}")

Yr_check, dims_check, T_check = caiman.mmapping.load_memmap(FNAME_MMAP)
assert dims_check == DIMS
assert T_check == T
print(f"Verified: dims={dims_check}  T={T_check}")

with open(f"{OUTPUT_DIR}/memmap_path.json", "w") as fh:
    json.dump({"FNAME_MMAP": FNAME_MMAP, "DIMS": list(DIMS), "T": int(T)}, fh, indent=2)


# =============================================================================
# HELPERS
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
    "max_shifts": (3, 3),
    "strides": (48, 48),
    "overlaps": (24, 24),
    "max_deviation_rigid": 2,
}


def array_to_memmap(array, basename):
    tif_path = str(basename) + ".tif"
    tifffile.imwrite(tif_path, array.astype(np.float32))
    return caiman.mmapping.save_memmap(
        [tif_path], base_name=str(basename), order="C", border_to_0=0,
    )


def run_cnmf(params_override, fname_mmap, do_motion_correction=True, filter_components=True):
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
        cnmf_obj.fit_file(motion_correct=do_motion_correction)
        if filter_components and cnmf_obj.estimates.A.shape[1] > 0:
            try:
                Yr, dims, T_loc = caiman.mmapping.load_memmap(fname_mmap)
                images = np.reshape(Yr.T, [T_loc] + list(dims), order="F")
                cnmf_obj.estimates.evaluate_components(imgs=images, params=cnmf_obj.params, dview=None)
                cnmf_obj.estimates.select_components(use_object=True)
            except Exception:
                pass
        return cnmf_obj, time.time() - t0
    except Exception as exc:
        print(f"    CNMF failed: {exc}")
        return None, time.time() - t0


def score_run(cnmf_obj, Yr, stability=0.0):
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
    recon_error = float(np.linalg.norm(Yr - Y_hat, "fro") / (np.linalg.norm(Yr, "fro") + 1e-9))

    H_val, W_val = DIMS
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

    composite = -1.0 * recon_error + 0.5 * spatial_compactness - 0.3 * np.log1p(trace_sparsity) + 1.0 * stability

    return {
        "n_neurons": n, "recon_error": recon_error,
        "spatial_compactness": spatial_compactness, "trace_sparsity": trace_sparsity,
        "stability": stability, "composite_score": float(composite),
    }


def compute_stability(A1, A2, threshold=0.5):
    if A1 is None or A2 is None or A1.shape[1] == 0 or A2.shape[1] == 0:
        return 0.0
    norms1 = np.asarray(np.sqrt(A1.power(2).sum(axis=0))).flatten() + 1e-9
    norms2 = np.asarray(np.sqrt(A2.power(2).sum(axis=0))).flatten() + 1e-9
    A1n = A1.multiply(1.0 / norms1)
    A2n = A2.multiply(1.0 / norms2)
    corr = np.asarray((A1n.T @ A2n).todense())
    row_ind, col_ind = linear_sum_assignment(-corr)
    return float(np.mean(corr[row_ind, col_ind] >= threshold))


print("Helpers defined.")


# =============================================================================
# BAYESIAN SEARCH
# =============================================================================

print("\n" + "=" * 60)
print("BAYESIAN HYPERPARAMETER SEARCH")
print("=" * 60)

Yr_global, _, _ = caiman.mmapping.load_memmap(FNAME_MMAP)

SEARCH_SPACE = [
    Integer(2, 5, name="gSig"),
    Integer(1, 4, name="gSig_filt"),
    Real(0.4, 0.85, name="min_corr"),
    Integer(3, 12, name="min_pnr"),
    Categorical([25, 40, 60, 80], name="rf"),
    Categorical([1, 2], name="p"),
]
PARAM_NAMES = [s.name for s in SEARCH_SPACE]
trial_log = []


def objective(params):
    trial_params = dict(zip(PARAM_NAMES, params))
    trial_params["stride"] = trial_params["rf"] // 2
    trial_num = len(trial_log) + 1
    print(f"  Trial {trial_num:2d}: {trial_params} ...", end=" ", flush=True)

    cnmf_obj, runtime = run_cnmf(trial_params, FNAME_MMAP)
    metrics = score_run(cnmf_obj, Yr_global)
    metrics.update(trial_params)
    metrics["runtime_s"] = round(runtime, 1)
    trial_log.append(metrics)

    print(f"neurons={metrics['n_neurons']:3d}  composite={metrics['composite_score']:+.4f}  t={runtime:.0f}s")
    score = -metrics["composite_score"]
    if not np.isfinite(score):
        score = 1e4
    return score


opt_result = gp_minimize(
    objective, SEARCH_SPACE,
    n_calls=N_CALLS, n_initial_points=min(N_INITIAL, N_CALLS),
    random_state=42, verbose=False,
)

print(f"\nCompleted {N_CALLS} trials.")
df_trials = pd.DataFrame(trial_log)
df_trials.to_csv(f"{OUTPUT_DIR}/bayesian_search_log.csv", index=False)
print("Saved bayesian_search_log.csv")


# =============================================================================
# RESULTS + PLOTS
# =============================================================================

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

df_sorted = df_trials.sort_values("composite_score", ascending=False)
display_cols = [
    "gSig", "gSig_filt", "min_corr", "min_pnr", "rf", "p",
    "n_neurons", "recon_error", "spatial_compactness",
    "trace_sparsity", "composite_score", "runtime_s",
]
print("\nTop trials:")
print(df_sorted[display_cols].head(min(5, len(df_sorted))).to_string(index=False))

best_row = df_sorted.iloc[0]
best_trial_params = {
    "gSig": int(best_row["gSig"]),
    "gSig_filt": int(best_row["gSig_filt"]),
    "min_corr": float(best_row["min_corr"]),
    "min_pnr": int(best_row["min_pnr"]),
    "rf": int(best_row["rf"]),
    "stride": int(best_row["rf"]) // 2,
    "p": int(best_row["p"]),
    "merge_thr": 0.85,
}

print("\nBest parameters:")
for k, v in best_trial_params.items():
    print(f"  {k}: {v}")

fig, ax = plt.subplots(figsize=(10, 4))
plot_convergence(opt_result, ax=ax)
ax.set_title("Bayesian optimisation convergence")
ax.set_ylabel("Negative composite score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/convergence.png", dpi=120)
plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, param in zip(axes.flat, PARAM_NAMES):
    ax.scatter(df_trials[param], df_trials["composite_score"], alpha=0.6, s=30)
    ax.set_xlabel(param)
    ax.set_ylabel("Composite score")
    ax.grid(True, alpha=0.3)
plt.suptitle("Parameter vs composite score")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/param_vs_score.png", dpi=120)
plt.close(fig)


# =============================================================================
# SPLIT-HALF STABILITY
# =============================================================================

stability = 0.0

if ARGS.skip_stability:
    print("\nSkipping split-half stability (--skip-stability)")
else:
    print("\n" + "=" * 60)
    print("SPLIT-HALF STABILITY")
    print("=" * 60)

    fname_h1 = array_to_memmap(data_512[:T // 2], WORK_DIR / "half1")
    fname_h2 = array_to_memmap(data_512[T // 2:], WORK_DIR / "half2")

    cnmf_h1, _ = run_cnmf(best_trial_params, fname_h1)
    cnmf_h2, _ = run_cnmf(best_trial_params, fname_h2)

    n_h1 = cnmf_h1.estimates.A.shape[1] if cnmf_h1 else 0
    n_h2 = cnmf_h2.estimates.A.shape[1] if cnmf_h2 else 0
    stability = compute_stability(
        cnmf_h1.estimates.A if cnmf_h1 else None,
        cnmf_h2.estimates.A if cnmf_h2 else None,
    )
    print(f"  First half:  {n_h1} neurons")
    print(f"  Second half: {n_h2} neurons")
    print(f"  Stability: {stability:.3f}")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_params_full = {
    **best_trial_params,
    "split_half_stability": round(stability, 3),
    "composite_score": float(best_row["composite_score"]),
    "data_dir": str(ARGS.data_dir),
    "run_name": ARGS.run_name,
    "z_method": ARGS.z_method,
    "T": int(T),
    "Z": int(Z_FULL),
    "H_orig": int(H_ORIG),
    "W_orig": int(W_ORIG),
    "n_files": len(h5_files),
}

with open(f"{OUTPUT_DIR}/best_params.json", "w") as fh:
    json.dump(best_params_full, fh, indent=2)

print("\nBest params:")
for k, v in best_params_full.items():
    print(f"  {k}: {v}")
print(f"\nSaved: {OUTPUT_DIR}/best_params.json")


# =============================================================================
# FINAL CNMF
# =============================================================================

cnmf_final = None

if ARGS.skip_stability:
    print("\nSkipping final CNMF (--skip-stability)")
else:
    print("\n" + "=" * 60)
    print("FINAL CNMF RUN")
    print("=" * 60)

    cnmf_final, final_runtime = run_cnmf(best_trial_params, FNAME_MMAP)

    if cnmf_final is None:
        print("Final CNMF failed.")
    else:
        n_final = cnmf_final.estimates.A.shape[1]
        print(f"Accepted components: {n_final}  ({final_runtime:.0f}s)")

        final_metrics = score_run(cnmf_final, Yr_global, stability=stability)
        print(f"Composite score: {final_metrics['composite_score']:.4f}")

        mean_frame = data_512.mean(axis=0)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(mean_frame, cmap="gray")
        ax.set_title(
            f"Detected neurons ({n_final}) — "
            f"gSig={best_trial_params['gSig']}, "
            f"min_corr={best_trial_params['min_corr']:.2f}, "
            f"min_pnr={best_trial_params['min_pnr']}, "
            f"rf={best_trial_params['rf']}, p={best_trial_params['p']}",
            fontsize=9,
        )
        ax.axis("off")

        H_val, W_val = DIMS
        for i in range(n_final):
            fp = np.asarray(cnmf_final.estimates.A[:, i].todense()).flatten().reshape(H_val, W_val)
            if fp.max() == 0:
                continue
            ax.contour(fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.5, alpha=0.8)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/final_neuron_contours.png", dpi=150)
        plt.close(fig)
        print("Saved final_neuron_contours.png")


# =============================================================================
# SAVE TRACES
# =============================================================================

if cnmf_final is not None and cnmf_final.estimates.C.shape[0] > 0:
    traces = cnmf_final.estimates.C
    footprints_dense = cnmf_final.estimates.A.toarray()

    np.save(f"{OUTPUT_DIR}/final_traces.npy", traces)
    np.save(f"{OUTPUT_DIR}/final_footprints.npy", footprints_dense)
    print(f"Saved final_traces.npy       shape={traces.shape}")
    print(f"Saved final_footprints.npy   shape={footprints_dense.shape}")

    n_plot = min(8, traces.shape[0])
    t_axis = np.arange(T)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t_axis, traces[i], lw=1)
        ax.set_ylabel(f"N{i}", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Frame")
    plt.suptitle(f"Sample traces — {ARGS.run_name}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_traces.png", dpi=120)
    plt.close(fig)
    print("Saved sample_traces.png")

    # Append to master CSV (one row per run across all datasets)
    master_csv = RESULTS_ROOT / "all_runs.csv"
    row = {
        "run_name": ARGS.run_name,
        "data_dir": str(ARGS.data_dir),
        "n_files": len(h5_files),
        "T": int(T),
        "Z": int(Z_FULL),
        "n_neurons": int(traces.shape[0]),
        "composite_score": float(best_row["composite_score"]),
        "split_half_stability": round(stability, 3),
        "gSig": best_trial_params["gSig"],
        "min_corr": best_trial_params["min_corr"],
        "min_pnr": best_trial_params["min_pnr"],
        "rf": best_trial_params["rf"],
        "p": best_trial_params["p"],
    }
    df_row = pd.DataFrame([row])
    if master_csv.is_file():
        df_row.to_csv(master_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(master_csv, index=False)
    print(f"Appended row to {master_csv}")

elif not ARGS.skip_stability:
    print("No traces — final CNMF had no accepted components.")

print(f"\nAll outputs in {OUTPUT_DIR}/")
print("Done.")
