#!/usr/bin/env python3
"""
p2_analyze_h5.py  —  Phase 2: CNMF Hyperparameter Tuning (v2)

Zebrafish light-sheet calcium imaging.
Data: (T, X, Y) HDF5  →  downsampled to (T, 512, 512) for tuning.

Changes from v1 (cnmf_experiments_colab.py):
  1. Multi-objective composite score replaces reconstruction-error-only selection.
  2. gSig_filt tuned jointly with gSig for proper background separation.
  3. Bayesian optimisation (scikit-optimize gp_minimize) replaces sequential grid search.
  4. Component filtering (SNR + spatial correlation) applied inside every trial.
  5. Motion correction enabled (piecewise-rigid).
  6. rf search space scaled to 512×512 resolution.
  7. Background preprocessing: column-median stripe removal before CNMF.

Run in Google Colab. Copy each # %% cell block into a separate notebook cell.
Run cells in order — each depends on variables from the previous one.
"""


# =============================================================================
# %% cell0 — INSTALL AND IMPORTS
# After this cell, restart runtime if Colab prompts you, then run cell1.
# =============================================================================

# Colab cell 0 — run once per session (uncomment in notebook):
# !pip install git+https://github.com/flatironinstitute/CaImAn.git tifffile scikit-image scikit-optimize --quiet

import os
import sys
import time
import json
import warnings
import importlib.util
import subprocess
warnings.filterwarnings("ignore")

# --- Auto-install missing deps (Colab: run this cell first; first run can take 5–15+ min) ---
if importlib.util.find_spec("skopt") is None:
    print("Installing scikit-optimize …")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "scikit-optimize"],
    )

# PyPI package name "caiman" is not the Flatiron library — must use GitHub install.
if importlib.util.find_spec("caiman") is None:
    print(
        "Installing CaImAn from GitHub (several minutes, large download) …\n"
        "If this fails, run in a separate cell:\n"
        "  !pip install git+https://github.com/flatironinstitute/CaImAn.git tifffile scikit-image scikit-optimize -q"
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/flatironinstitute/CaImAn.git",
        ]
    )

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path

import skimage.transform
from skimage.morphology import convex_hull_image
from scipy.optimize import linear_sum_assignment

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.plots import plot_convergence

import caiman as cm
import caiman  # noqa: F401 — must bind name `caiman` for `caiman.mmapping.*` in other Colab cells
from caiman.source_extraction.cnmf import cnmf as cnmf_module
from caiman.source_extraction.cnmf import params as params_module

OUTPUT_DIR = "/content/cnmf_results_v2"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

print(f"CaImAn version : {cm.__version__}")
print(f"Output directory: {OUTPUT_DIR}")


# =============================================================================
# %% cell1 — MOUNT DRIVE, LOAD H5, DOWNSAMPLE, STRIPE REMOVAL
#
# Background preprocessing: the Phase 1 exploration revealed a vertical
# striping artifact.  Subtracting the per-column temporal median removes
# the static stripe pattern before CNMF ever sees the data.
# =============================================================================

from google.colab import drive
drive.mount("/content/drive")

FILENAME = "/content/drive/My Drive/Zebra_fish/10vii25_task18_60frames.h5"

with h5py.File(FILENAME, "r") as fh:
    data_full = fh["Data"][:]           # (60, 2048, 2048) uint16 ≈ 480 MB
    shape_orig = data_full.shape
    dtype_orig = data_full.dtype

T_ORIG, H_ORIG, W_ORIG = shape_orig
print(f"Original  : shape={shape_orig}  dtype={dtype_orig}")

# ── Downsample 4× to 512×512 ────────────────────────────────────────────────
TARGET_H, TARGET_W = 512, 512
data_512 = np.zeros((T_ORIG, TARGET_H, TARGET_W), dtype=np.float32)
for t in range(T_ORIG):
    data_512[t] = skimage.transform.resize(
        data_full[t].astype(np.float32),
        (TARGET_H, TARGET_W),
        anti_aliasing=True,
    )
del data_full

T, H, W = data_512.shape
DIMS = (H, W)
print(f"Downsampled: shape={data_512.shape}  dtype={data_512.dtype}")

# ── Stripe removal (column-median subtraction) ──────────────────────────────
# The median across time and rows for each column captures the static stripe.
col_median = np.median(data_512, axis=(0, 1), keepdims=True)  # shape (1, 1, W)
data_512 = data_512 - col_median
data_512 = np.clip(data_512, 0, None).astype(np.float32)
print("Stripe removal: subtracted per-column temporal median")

# ── Sanity check ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(data_512[T // 2], cmap="gray")
axes[0].set_title(f"Frame {T // 2} — after stripe removal")
axes[0].axis("off")
# 2D array required for imshow — (1, W) = one row of per-column medians (stripe profile)
axes[1].imshow(col_median.reshape(1, -1), cmap="gray", aspect="auto")
axes[1].set_title("Removed stripe pattern (column medians)")
axes[1].set_xlabel("Column index")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/preprocessing_check.png", dpi=100)
plt.show()


# =============================================================================
# %% cell2 — SAVE TIFF AND CONVERT TO CAIMAN MEMMAP
# =============================================================================

import caiman.mmapping
import caiman.base.movies

if not hasattr(cm, "load"):
    cm.load = caiman.base.movies.load
if not hasattr(cm, "movie"):
    cm.movie = caiman.base.movies.movie
if not hasattr(cm, "paths"):
    import caiman.paths

TIF_PATH  = "/content/fish_512_v2.tif"
MMAP_BASE = "/content/fish_512_v2_mmap"

print("Saving float32 tiff...")
tifffile.imwrite(TIF_PATH, data_512)

print("Converting to CaImAn memmap...")
FNAME_MMAP = caiman.mmapping.save_memmap(
    [TIF_PATH],
    base_name=MMAP_BASE,
    order="C",
    border_to_0=0,
)
print(f"Memmap: {FNAME_MMAP}")

Yr_check, dims_check, T_check = caiman.mmapping.load_memmap(FNAME_MMAP)
assert dims_check == DIMS, f"Dimension mismatch: {dims_check} vs {DIMS}"
assert T_check == T, f"Frame count mismatch: {T_check} vs {T}"
print(f"Verified: Yr={Yr_check.shape}  dims={dims_check}  T={T_check}")

# Persist paths so later Colab cells (run out of order) can reload FNAME_MMAP / DIMS / T
_MEMMAP_META = Path(OUTPUT_DIR) / "memmap_path.json"
with open(_MEMMAP_META, "w") as fh:
    json.dump(
        {
            "FNAME_MMAP": FNAME_MMAP,
            "DIMS": list(DIMS),
            "T": int(T),
        },
        fh,
        indent=2,
    )
print(f"Saved memmap metadata: {_MEMMAP_META}")


# =============================================================================
# %% cell3 — HELPER FUNCTIONS
#
# Key changes from v1:
#   • run_cnmf: motion correction ON, handles gSig_filt, runs component
#     filtering inside so metrics reflect quality after false-positive removal.
#   • score_run: returns a composite_score that balances reconstruction error,
#     spatial compactness, trace sparsity, and (optionally) stability.
#   • compute_stability: unchanged (Hungarian matching on spatial footprints).
# =============================================================================

# ── Base CNMF parameters (fixed across all trials) ──────────────────────────
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
    # Motion correction (piecewise-rigid)
    "pw_rigid": True,
    "max_shifts": (3, 3),
    "strides": (48, 48),
    "overlaps": (24, 24),
    "max_deviation_rigid": 2,
}


def array_to_memmap(array, basename):
    """Convert (T, H, W) float32 array to a CaImAn memmap."""
    tif_path = f"{basename}.tif"
    tifffile.imwrite(tif_path, array.astype(np.float32))
    return caiman.mmapping.save_memmap(
        [tif_path], base_name=basename, order="C", border_to_0=0,
    )


def run_cnmf(params_override, fname_mmap, do_motion_correction=True,
             filter_components=True):
    """
    Run CNMF with BASE_PARAMS merged with params_override.
    Returns (cnmf_obj or None, runtime_seconds).

    gSig / gSig_filt can be passed as int → auto-converted to tuple.
    gSiz is derived as (4*gSig+1, 4*gSig+1).
    Component filtering is applied inside so downstream metrics reflect
    quality after false-positive removal.
    """
    p = {**BASE_PARAMS, **params_override, "fnames": [fname_mmap]}

    # skopt / NumPy pass np.int64 — isinstance(x, int) is False; CaImAn needs (px, px) tuples
    for key in ("gSig", "gSig_filt"):
        val = p.get(key)
        if val is None:
            continue
        if isinstance(val, tuple):
            p[key] = (int(val[0]), int(val[1]))
        else:
            v = int(val)
            p[key] = (v, v)

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
                cnmf_obj.estimates.evaluate_components(
                    imgs=images,
                    params=cnmf_obj.params,
                    dview=None,
                )
                cnmf_obj.estimates.select_components(use_object=True)
            except Exception:
                pass  # keep unfiltered if evaluation fails

        return cnmf_obj, time.time() - t0
    except Exception as exc:
        print(f"    CNMF failed: {exc}")
        return None, time.time() - t0


def score_run(cnmf_obj, Yr, stability=0.0):
    """
    Multi-objective scoring.

    Returns dict with individual metrics and a composite_score:
        composite = -1.0*recon_error
                    +0.5*spatial_compactness
                    -0.3*log(1 + trace_sparsity)
                    +1.0*stability

    Higher composite_score is better.
    """
    sentinel = {
        "n_neurons": 0,
        "recon_error": 1.0,
        "spatial_compactness": 0.0,
        "trace_sparsity": float("inf"),
        "stability": stability,
        "composite_score": -float("inf"),
    }
    if cnmf_obj is None:
        return sentinel

    A = cnmf_obj.estimates.A
    C = cnmf_obj.estimates.C
    n_neurons = A.shape[1]
    if n_neurons == 0:
        return sentinel

    # ── Reconstruction error ─────────────────────────────────────────────
    Y_hat = A @ C
    b = getattr(cnmf_obj.estimates, "b", None)
    f_bg = getattr(cnmf_obj.estimates, "f", None)
    if b is not None and f_bg is not None and b.shape[1] > 0:
        Y_hat = Y_hat + b @ f_bg
    recon_error = float(
        np.linalg.norm(Yr - Y_hat, "fro") / (np.linalg.norm(Yr, "fro") + 1e-9)
    )

    # ── Spatial compactness ──────────────────────────────────────────────
    H_val, W_val = DIMS
    compactness_list = []
    for i in range(n_neurons):
        fp = np.asarray(A[:, i].todense()).flatten().reshape(H_val, W_val)
        binary = fp > (fp.max() * 0.2)
        if binary.sum() < 5:
            continue
        try:
            hull = convex_hull_image(binary)
            compactness_list.append(float(binary.sum()) / float(hull.sum()))
        except Exception:
            pass
    spatial_compactness = float(np.mean(compactness_list)) if compactness_list else 0.0

    # ── Trace sparsity ───────────────────────────────────────────────────
    l1 = np.sum(np.abs(C), axis=1)
    l2 = np.linalg.norm(C, axis=1)
    trace_sparsity = float(np.mean(l1 / (l2 + 1e-9)))

    # ── Composite score ──────────────────────────────────────────────────
    composite = (
        -1.0 * recon_error
        + 0.5 * spatial_compactness
        - 0.3 * np.log1p(trace_sparsity)
        + 1.0 * stability
    )

    return {
        "n_neurons": n_neurons,
        "recon_error": recon_error,
        "spatial_compactness": spatial_compactness,
        "trace_sparsity": trace_sparsity,
        "stability": stability,
        "composite_score": float(composite),
    }


def compute_stability(A1, A2, threshold=0.5):
    """
    Fraction of neurons in A1 matched to a neuron in A2
    with spatial correlation >= threshold (Hungarian matching).
    """
    if A1 is None or A2 is None or A1.shape[1] == 0 or A2.shape[1] == 0:
        return 0.0
    norms1 = np.asarray(np.sqrt(A1.power(2).sum(axis=0))).flatten() + 1e-9
    norms2 = np.asarray(np.sqrt(A2.power(2).sum(axis=0))).flatten() + 1e-9
    A1n = A1.multiply(1.0 / norms1)
    A2n = A2.multiply(1.0 / norms2)
    corr = np.asarray((A1n.T @ A2n).todense())
    row_ind, col_ind = linear_sum_assignment(-corr)
    matched = corr[row_ind, col_ind]
    return float(np.mean(matched >= threshold))


print("Helper functions defined.")


# =============================================================================
# %% cell4 — BAYESIAN HYPERPARAMETER SEARCH (joint optimisation)
#
# All key parameters are tuned jointly using Gaussian-Process-based
# Bayesian optimisation (scikit-optimize gp_minimize).
#
# Search space (scaled for 512×512):
#   gSig      : [2, 5]   neuron radius (pixels at 512×512)
#   gSig_filt : [1, 4]   high-pass filter radius for seed detection
#   min_corr  : [0.4, 0.85]
#   min_pnr   : [3, 12]
#   rf        : {25, 40, 60, 80}   patch size
#   p         : {1, 2}             AR model order
#
# n_calls=30 → ~30 CNMF runs.  At ~2-4 min each ≈ 1-2 hours.
# The first n_initial_points=10 are random; the remaining 20 are
# guided by the GP surrogate model.
# =============================================================================

print("=" * 60)
print("BAYESIAN HYPERPARAMETER SEARCH")
print("=" * 60)

import caiman  # Colab: `import caiman as cm` does not define `caiman`; needed for line below

# Colab: if this cell runs in a new session or without cell 2, restore FNAME_MMAP from disk
_MEMMAP_META = Path(OUTPUT_DIR) / "memmap_path.json"
if "FNAME_MMAP" not in globals():
    if _MEMMAP_META.is_file():
        _meta = json.loads(_MEMMAP_META.read_text())
        FNAME_MMAP = _meta["FNAME_MMAP"]
        if "DIMS" not in globals():
            DIMS = tuple(_meta["DIMS"])
        if "T" not in globals():
            T = int(_meta["T"])
        print(f"Restored FNAME_MMAP from {_MEMMAP_META}")
    else:
        raise RuntimeError(
            "FNAME_MMAP is not defined and memmap_path.json was not found. "
            "Run cells in order: 0 (imports) → 1 (load H5) → 2 (tiff + memmap) → 3 (helpers), "
            "then this cell. Or re-run cell 2 in this session."
        )
elif "DIMS" not in globals() and _MEMMAP_META.is_file():
    _meta = json.loads(_MEMMAP_META.read_text())
    DIMS = tuple(_meta["DIMS"])
    T = int(_meta["T"])

Yr_global, _, _ = caiman.mmapping.load_memmap(FNAME_MMAP)

SEARCH_SPACE = [
    Integer(2, 5,         name="gSig"),
    Integer(1, 4,         name="gSig_filt"),
    Real(0.4, 0.85,       name="min_corr"),
    Integer(3, 12,        name="min_pnr"),
    Categorical([25, 40, 60, 80], name="rf"),
    Categorical([1, 2],          name="p"),
]

PARAM_NAMES = [s.name for s in SEARCH_SPACE]

trial_log = []

def objective(params):
    """Objective for gp_minimize (returns value to MINIMISE)."""
    trial_params = dict(zip(PARAM_NAMES, params))
    trial_params["stride"] = trial_params["rf"] // 2
    trial_num = len(trial_log) + 1

    print(f"  Trial {trial_num:2d}: {trial_params} ...", end=" ", flush=True)

    cnmf_obj, runtime = run_cnmf(trial_params, FNAME_MMAP)
    metrics = score_run(cnmf_obj, Yr_global)
    metrics.update(trial_params)
    metrics["runtime_s"] = round(runtime, 1)
    trial_log.append(metrics)

    print(
        f"neurons={metrics['n_neurons']:3d}  "
        f"composite={metrics['composite_score']:+.4f}  "
        f"t={runtime:.0f}s"
    )

    # gp_minimize minimises, so negate composite_score
    return -metrics["composite_score"]


N_CALLS = 30
N_INITIAL = 10

opt_result = gp_minimize(
    objective,
    SEARCH_SPACE,
    n_calls=N_CALLS,
    n_initial_points=N_INITIAL,
    random_state=42,
    verbose=False,
)

print(f"\nCompleted {N_CALLS} trials.")

df_trials = pd.DataFrame(trial_log)
df_trials.to_csv(f"{OUTPUT_DIR}/bayesian_search_log.csv", index=False)
print(f"Saved bayesian_search_log.csv")


# =============================================================================
# %% cell5 — SEARCH RESULTS AND CONVERGENCE
# =============================================================================

print("=" * 60)
print("SEARCH RESULTS")
print("=" * 60)

# Top 5 trials by composite score
df_sorted = df_trials.sort_values("composite_score", ascending=False)
display_cols = [
    "gSig", "gSig_filt", "min_corr", "min_pnr", "rf", "p",
    "n_neurons", "recon_error", "spatial_compactness",
    "trace_sparsity", "composite_score", "runtime_s",
]
print("\nTop 5 trials (by composite score):")
print(df_sorted[display_cols].head(5).to_string(index=False))

# Best parameters
best_idx  = df_sorted.index[0]
best_row  = df_sorted.iloc[0]
best_trial_params = {
    "gSig":      int(best_row["gSig"]),
    "gSig_filt": int(best_row["gSig_filt"]),
    "min_corr":  float(best_row["min_corr"]),
    "min_pnr":   int(best_row["min_pnr"]),
    "rf":        int(best_row["rf"]),
    "stride":    int(best_row["rf"]) // 2,
    "p":         int(best_row["p"]),
    "merge_thr": 0.85,
}

print(f"\nBest parameters:")
for k, v in best_trial_params.items():
    print(f"  {k}: {v}")

# Convergence plot
fig, ax = plt.subplots(figsize=(10, 4))
plot_convergence(opt_result, ax=ax)
ax.set_title("Bayesian optimisation convergence")
ax.set_ylabel("Negative composite score (lower = better)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/convergence.png", dpi=120)
plt.show()

# Parameter importance — scatter matrix of top metrics
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, param in zip(axes.flat, PARAM_NAMES):
    ax.scatter(df_trials[param], df_trials["composite_score"], alpha=0.6, s=30)
    ax.set_xlabel(param)
    ax.set_ylabel("Composite score")
    ax.grid(True, alpha=0.3)
plt.suptitle("Parameter vs composite score (all trials)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/param_vs_score.png", dpi=120)
plt.show()


# =============================================================================
# %% cell6 — SPLIT-HALF STABILITY (standalone robustness metric)
#
# This is NOT tied to any single parameter — it measures whether the best
# parameter set produces neurons that are consistent across time.
# Run CNMF on frames 0-29 and 30-59 separately, then match neurons
# by spatial correlation.  A high stability score means the detected
# neurons are real structure rather than overfits to noise.
# =============================================================================

print("=" * 60)
print("SPLIT-HALF STABILITY")
print("=" * 60)

fname_first  = array_to_memmap(data_512[:T // 2],  "/content/fish_half1_v2")
fname_second = array_to_memmap(data_512[T // 2:],   "/content/fish_half2_v2")

cnmf_h1, _ = run_cnmf(best_trial_params, fname_first,  do_motion_correction=True)
cnmf_h2, _ = run_cnmf(best_trial_params, fname_second, do_motion_correction=True)

n_h1 = cnmf_h1.estimates.A.shape[1] if cnmf_h1 else 0
n_h2 = cnmf_h2.estimates.A.shape[1] if cnmf_h2 else 0
stability = compute_stability(
    cnmf_h1.estimates.A if cnmf_h1 else None,
    cnmf_h2.estimates.A if cnmf_h2 else None,
)

print(f"  First half  (frames 0–{T // 2 - 1}): {n_h1} neurons")
print(f"  Second half (frames {T // 2}–{T - 1}): {n_h2} neurons")
print(f"  Stability score: {stability:.3f}  (fraction matched at spatial corr >= 0.5)")


# =============================================================================
# %% cell7 — SUMMARY AND BEST PARAMS
# =============================================================================

print("=" * 60)
print("SUMMARY")
print("=" * 60)

best_params_full = {
    **best_trial_params,
    "split_half_stability": round(stability, 3),
    "composite_score": float(best_row["composite_score"]),
    "note": (
        "Parameters optimised jointly via Bayesian search (gp_minimize, "
        f"{N_CALLS} trials) on (60, 512, 512) stripe-corrected data. "
        "gSig/gSig_filt are scaled for 512×512 (≈ original / 4). "
        "Component filtering (SNR + spatial corr) applied inside every trial. "
        "Motion correction (piecewise-rigid) enabled. "
        "A full-resolution 2048×2048 run is the ideal next step but may "
        "exceed Colab compute limits."
    ),
}

with open(f"{OUTPUT_DIR}/best_params.json", "w") as fh:
    json.dump(best_params_full, fh, indent=2)

print("\nFinal selected parameters:")
for k, v in best_params_full.items():
    if k == "note":
        continue
    print(f"  {k}: {v}")
print(f"\nSaved: {OUTPUT_DIR}/best_params.json")

# Score breakdown
print(f"\nScore breakdown:")
print(f"  Reconstruction error : {float(best_row['recon_error']):.4f}")
print(f"  Spatial compactness  : {float(best_row['spatial_compactness']):.4f}")
print(f"  Trace sparsity       : {float(best_row['trace_sparsity']):.4f}")
print(f"  Split-half stability : {stability:.4f}")
print(f"  Composite score      : {float(best_row['composite_score']):.4f}")


# =============================================================================
# %% cell8 — FINAL CNMF RUN WITH BEST PARAMETERS
# =============================================================================

print("=" * 60)
print("FINAL CNMF RUN (best params, with motion correction + filtering)")
print("=" * 60)

cnmf_final, final_runtime = run_cnmf(
    best_trial_params, FNAME_MMAP,
    do_motion_correction=True,
    filter_components=True,
)

if cnmf_final is None:
    print("Final CNMF run failed. Review parameters above.")
else:
    n_final = cnmf_final.estimates.A.shape[1]
    print(f"Accepted components: {n_final}  ({final_runtime:.0f}s)")

    final_metrics = score_run(cnmf_final, Yr_global, stability=stability)
    print(f"Composite score: {final_metrics['composite_score']:.4f}")

    # Contour overlay on mean projection
    mean_frame = data_512.mean(axis=0)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(mean_frame, cmap="gray")
    ax.set_title(
        f"Detected neurons ({n_final}) — "
        f"gSig={best_trial_params['gSig']}, "
        f"min_corr={best_trial_params['min_corr']:.2f}, "
        f"min_pnr={best_trial_params['min_pnr']}, "
        f"rf={best_trial_params['rf']}, "
        f"p={best_trial_params['p']}",
        fontsize=9,
    )
    ax.axis("off")

    H_val, W_val = DIMS
    for i in range(n_final):
        fp = np.asarray(
            cnmf_final.estimates.A[:, i].todense()
        ).flatten().reshape(H_val, W_val)
        if fp.max() == 0:
            continue
        ax.contour(fp, levels=[fp.max() * 0.5], colors="cyan", linewidths=0.5, alpha=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/final_neuron_contours.png", dpi=150)
    plt.show()
    print("Saved final_neuron_contours.png")


# =============================================================================
# %% cell9 — SAVE OUTPUTS AND PLOT SAMPLE TRACES
# =============================================================================

if cnmf_final is not None and cnmf_final.estimates.C.shape[0] > 0:
    traces            = cnmf_final.estimates.C            # (n_neurons, T)
    footprints_sparse = cnmf_final.estimates.A            # (d1*d2, n_neurons) sparse
    footprints_dense  = footprints_sparse.toarray()       # dense for .npy

    np.save(f"{OUTPUT_DIR}/final_traces.npy",     traces)
    np.save(f"{OUTPUT_DIR}/final_footprints.npy", footprints_dense)
    print(f"Saved final_traces.npy       shape={traces.shape}")
    print(f"Saved final_footprints.npy   shape={footprints_dense.shape}")

    # Plot up to 8 sample traces
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
    plt.suptitle("Sample neuron calcium traces (best params, filtered)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/sample_traces.png", dpi=120)
    plt.show()
    print("Saved sample_traces.png")

    # ── Summary of all output files ──────────────────────────────────────
    print(f"\nAll outputs in {OUTPUT_DIR}/:")
    print("  bayesian_search_log.csv       — all 30 trial results")
    print("  convergence.png               — Bayesian optimisation convergence")
    print("  param_vs_score.png            — parameter vs composite score")
    print("  best_params.json              — winning parameter set")
    print(f"  final_traces.npy              — shape {traces.shape}")
    print(f"  final_footprints.npy          — shape {footprints_dense.shape}")
    print("  final_neuron_contours.png     — contours on mean frame")
    print("  sample_traces.png             — example calcium traces")
    print("  preprocessing_check.png       — stripe removal before/after")
else:
    print("No outputs saved — final CNMF produced no accepted components.")
    print("Try widening the search space (lower min_corr/min_pnr bounds) and re-running.")
