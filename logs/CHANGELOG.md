# Changelog — p4_universal.py

---

## 2026-06-18 (6)

### Fix concatenation error introduced by `nb_patch=1`

**Problem:**
After the `nb=0 → nb=1` fix in entry (5), every CNMF trial failed with:
```
CNMF failed: all the input array dimensions except for the concatenation axis must
match exactly, but along dimension 1, the array at index 0 has size 350 and the
array at index 1 has size 1
```
Note: the SVD error was gone — `nb=1` fixed that correctly.

**Root cause:**
With `nb_patch=1`, CaImAn adds a low-rank background temporal trace per patch
(shape `(1, T_patch)`). Patches that contain no neurons (the majority at 2048×2048
with only 24.2% brain coverage) initialize their background trace as `(1, 1)` instead
of `(1, T_patch)`. When CaImAn assembles patch results, it tries to concatenate neural
traces of shape `(n, 350)` with background traces of shape `(1, 1)` — the time axis
mismatches, causing the axis-1 shape error.

**Fix:**
- `"nb_patch": 1` → `"nb_patch": 0` in `BASE_PARAMS`

With `nb_patch=0`, patches output no per-patch background component (they rely on
the ring model for local background). The global `nb=1` background component is still
estimated after patch assembly from well-shaped data. Empty patches no longer produce
malformed `(1, 1)` background arrays.

---

## 2026-06-18 (5)

### Fix SVD failure: `nb=0` passes `k=0` to `scipy.sparse.linalg.svds`

**Problem:**
Every CNMF trial failed with `SVD did not converge in Linear Least Squares` regardless
of resolution, masking, or thread count.

**Root cause (confirmed from CaImAn source — `initialization.py` / `compute_W`):**
`BASE_PARAMS` set `"nb": 0` (zero background components). CaImAn's `greedyROI_corr`
passes `nb` directly to `scipy.sparse.linalg.svds` as the `k` argument via `compute_W`:
```python
b_in, s_in, f_in = spr.linalg.svds(B, k=nb)   # k=0 is undefined behaviour
```
`svds(k=0)` is not a valid call — it raises `LinAlgError` or produces LAPACK parameter
warnings (`** On entry to DLASCL`) depending on the scipy/LAPACK version. This caused
100% trial failure at every resolution and parameter combination.

The per-trial CaImAn warning `gnb=0, hence setting keys nb_patch and low_rank_background
in group patch automatically` was the library flagging this exact condition on every trial.

**Fix:**
- `"nb": 0` → `"nb": 1` in `BASE_PARAMS`
- `"nb_patch": 0` → `"nb_patch": 1` in `BASE_PARAMS`

One background component gives `svds` a valid `k=1`. The ring model (`ring_size_factor=1.4`,
`center_psf=True`) still handles local background suppression as before; the single global
background component is a minor addition that does not change neuron detection behaviour.

---

## 2026-06-18 (4)

### Fix SVD convergence failure and DLASCL worker warnings

**Problem:**
All CNMF trials failed with `CNMF failed: SVD did not converge in Linear Least Squares`
across every parameter combination and resolution. At 1024×1024 the error was intermittent
(only test_half); at full (2048×2048) it was universal. Runs also emitted
`** On entry to DLASCL parameter number 4 had an illegal value` twice after DONE.

**Root cause (confirmed via regression tests at full and 1024 resolution, with and without
brain mask):**
`caiman.cluster.setup_cluster()` sets `OMP_NUM_THREADS=1` (and related BLAS env vars) in
the main process to prevent thread oversubscription across cluster workers. These env vars
are read at BLAS library load time, so they cannot be reversed by changing `os.environ`
after the fact. With a single BLAS thread, LAPACK's `dgelsd` SVD driver uses a simpler
iterative algorithm instead of divide-and-conquer. At large spatial scales the iterative
path fails to converge on the borderline ill-conditioned matrices produced by the ring
model background in CNMF-E.

The DLASCL warnings (`** On entry to DLASCL parameter number 4 had an illegal value`) are
a downstream symptom of the same ill-conditioned matrices: they are printed to C-level
stderr by LAPACK workers that were forked before the fix and inherited the live fd 2.
Because worker stderr is buffered and flushed on shutdown, the messages appear after DONE.

**Changes:**

#### threadpoolctl wrap around `fit_file` (SVD fix)
- `from threadpoolctl import threadpool_limits` added to module-level imports (line 75).
- `cnmf_obj.fit_file(motion_correct=do_mc)` wrapped in
  `with threadpool_limits(limits=N_WORKERS):` inside `run_cnmf`.
- `threadpool_limits` calls the BLAS library's own runtime thread-control API
  (e.g. `openblas_set_num_threads`), bypassing the env-var limit set by `setup_cluster`.
  Multi-threaded divide-and-conquer SVD is restored for the duration of each fit.

#### fd-2 redirect before `setup_cluster` (DLASCL guard)
- fd 2 (C-level stderr) is redirected to `/dev/null` immediately before the
  `setup_cluster()` call and restored immediately after.
- On Linux, `setup_cluster` uses `fork()`. Forked worker processes inherit the file
  descriptor table at fork time, so each worker gets fd 2 → `/dev/null`.
- Any LAPACK parameter warnings printed by workers write to `/dev/null`; the main
  process stderr is unaffected.

---

## 2026-06-18 (3)

### CaImAn API fixes — `dview` placement and cluster teardown

**Problem:**
Two API mismatches with the installed CaImAn version caused every `run_cnmf` call
to fail with `CNMF failed: CNMF.fit_file() got an unexpected keyword argument 'dview'`,
and the cluster teardown to crash with
`AttributeError: module 'caiman.cluster' has no attribute 'stop_cluster'`.
All 10 Bayesian trials returned `raw=0 kept=0 composite=-inf`.

**Root cause (confirmed against CaImAn source):**
- `fit_file(motion_correct, indices)` has no `dview` parameter. The CNMF object
  reads `self.dview` internally — dview must be passed to the constructor, not
  `fit_file`.
- `caiman.cluster` exposes `stop_server`, not `stop_cluster`.

**Changes:**
- `CNMF(n_processes=N_WORKERS, params=opts)` →
  `CNMF(n_processes=N_WORKERS, dview=DVIEW, params=opts)` — dview now set on the
  object at construction so `fit_file` picks it up via `self.dview`.
- `cnmf_obj.fit_file(motion_correct=do_mc, dview=DVIEW)` →
  `cnmf_obj.fit_file(motion_correct=do_mc)` — invalid kwarg removed.
- `_cluster.stop_cluster(dview=DVIEW)` →
  `_cluster.stop_server(dview=DVIEW)` — correct function name.

---

## 2026-06-18 (2)

### `--n-workers` wired to cluster pool; NUMA-safe default

**Problem:**
`N_WORKERS` was set to `os.cpu_count() - 1` (95 on the target machine) and passed
only to `CNMF(n_processes=...)`, which CaImAn ignores when a `dview` is provided.
`setup_cluster` was hardcoded to `n_processes=40`. The result: `--n-workers` was a
dead flag that changed nothing, and the header `workers=` line was misleading.

**Changes:**
- Default formula changed from `os.cpu_count() - 1` to
  `(os.cpu_count() // 2 // 10) * 10` — largest multiple of 10 ≤ half the logical
  CPU count. On the 96-core target: `96 // 2 = 48 → 40`. Keeps all workers on one
  NUMA node (48 physical cores per socket) avoiding cross-socket memory traffic.
- `setup_cluster(n_processes=N_WORKERS)` — cluster pool now tracks `N_WORKERS`
  instead of hardcoded 40. This is the change that makes `--n-workers` functional:
  passing `--n-workers N` now controls the actual cluster pool size.
- `--n-workers` help text updated to reflect the new default and scope.
- `--pin-cpus` behaviour unchanged — pinned-core count still takes priority over
  the formula when cores are explicitly pinned.

### `read_n_planes` silent failure now logged

**Problem:**
`read_n_planes` caught all exceptions and returned 1 with no output. A wrong
metadata key path (`meta["stack"]["n"]` instead of `meta["metaData"]["stack"]["n"]`)
caused the 144321 dataset to be silently misdetected as `single-movie`, feeding
7-plane interleaved frames raw into CNMF and producing 1 kept neuron across a
743-minute run.

**Change:**
`except Exception as e` now prints
`WARNING: read_n_planes failed (<reason>); defaulting to 1 (treating as single-plane)`
before returning 1. Future metadata format mismatches will be immediately visible
in the run log instead of silently corrupting format detection.

---

## 2026-06-18

### Persistent CaImAn cluster across Bayesian trials (A1–A4)

**Problem:**
Each call to `run_cnmf` / `MotionCorrect` / `evaluate_components` was implicitly
creating and tearing down a worker pool. On the Linux target (2× Xeon Gold 6240R,
96 logical CPUs) this incurred pool-setup overhead for every Bayesian trial.

**Changes:**
- Added module-level `DVIEW = None` sentinel (line 699) initialised before any mode
  runs, replaced by the real dview once `setup_cluster()` completes.
- `run_motion_correction`: passes `dview=DVIEW` to `MotionCorrect`.
- `run_cnmf`: passes `dview=DVIEW` to both `fit_file` and `evaluate_components`.
- Main dispatch (lines 1476–1486): wraps `MODES[ARGS.mode]()` in a `try/finally` that
  calls `_cluster.setup_cluster(backend="local", n_processes=40)` before the run and
  `_cluster.stop_cluster(dview=DVIEW)` in the `finally` block.
- All mode functions inherit the persistent pool through the global; no signature
  changes needed.

**Expected benefit:**
Eliminates pool setup/teardown on every trial. Estimated 1–3 min saved per full run
on the Linux target (modest due to fork semantics vs. spawn on Windows).

---

### Correctness fixes (C1, C2)

#### C1 — OOB centroid passes mask filter (line 837)

**Problem:**
The original condition `if 0 <= cy < H and 0 <= cx < W and not mask[cy, cx]`
accepted out-of-bounds centroids instead of rejecting them.

**Fix:**
`if not (0 <= cy < H and 0 <= cx < W) or not mask[cy, cx]` — now rejects any
centroid that is out of bounds OR falls outside the brain mask.

#### C2 — dF/F sign inverted when F0 < 0 (lines 514–517)

**Problem:**
Dividing by `abs(F0)` inverted the sign of all transients when the baseline went
negative after photobleach correction.

**Fix:**
Guard changed from `abs(F0) < 1e-6` to `F0 < 1e-6`; denominator changed from
`abs(F0)` to `F0`. A warning is printed when clamping is applied. dF/F transients
are now always positive when fluorescence rises above baseline.

---

### Warning fixes (W1–W5)

#### W1 + W2 — MC mmap mismatch in test phase corrupts `recon_error` (lines 1053–1055, 762–763)

**Problem:**
`test_cnmf` was loading `Yr` from the raw (pre-MC) mmap and then fitting CNMF on
the motion-corrected mmap. The two matrices described different movies, inflating
`recon_error` in all composite scores. Similarly, `evaluate_components` inside
`run_cnmf` was loading `Yr` from the original `fname_mmap` even after `fit_file`
had internally redirected to a corrected path.

**Fix (W1):**
`test_cnmf` now calls `run_motion_correction(mmap_path)` first, loads `Yr` from the
returned `mc_mmap`, and passes `mc_mmap` to `run_cnmf` with `do_mc=False`.

**Fix (W2):**
`evaluate_components` block reads the corrected path via
`cnmf_obj.params.data.get('fnames', [fname_mmap])[0]` so `Yr` always matches the
mmap CNMF actually fitted.

#### W3 — `area < 5` increments wrong counter (lines 804, 813–815, 844)

**Problem:**
Sub-5-pixel components were counted under `circularity_rejected`, making filter
diagnostics misleading.

**Fix:**
Added `rej_small = 0` counter; `area < 5` now increments `rej_small` and records
`counts["small_rejected"]` separately from `counts["circularity_rejected"]`.

#### W4 — `re.search` crash on unexpected filename (lines 287, 297)

**Problem:**
If a `.lux.h5` filename didn't match the expected pattern, `m.group(1)` raised
`AttributeError` (`NoneType` has no attribute `group`).

**Fix:**
Both sort-key lambdas use a walrus-operator guard:
`int(m.group(1)) if (m := re.search(...)) else 0` — falls back to `0` on mismatch.

#### W5 — Misleading log when `do_mc=False` (lines 756–757)

**Problem:**
Log always printed `"MC first, then CNMF init"` even when motion correction was
skipped, making trial logs confusing.

**Fix:**
`_label = "MC + CNMF init" if do_mc else "CNMF init (no MC)"` — log now reflects
whether MC actually runs.

---

### Performance and style fixes (S1–S3)

#### S1 — `_double_exp` redefined on every loop iteration (lines 495–498)

**Problem:**
The `_double_exp` helper was defined inside the `for i in range(N)` loop, creating
a new function object on every iteration.

**Fix:**
Moved the definition to immediately before the loop; one object, N uses.

#### S2 — Dense `Y_hat` materialisation in `score_run` (lines 883–894)

**Problem:**
`Y_hat = A @ C` materialised a dense (pixels × frames) matrix. At full resolution
(2048×2048, 1000 frames) this is ~16 GB, causing OOM on large datasets.

**Fix:**
Replaced with the Frobenius norm identity:

```
‖Yr − AC‖² = ‖Yr‖² − 2·sum(C ⊙ (Aᵀ·Yr)) + sum((AᵀA·C) ⊙ C)
```

Computed via three small intermediates (`AtYr`, `AtA`, `AC_norm_sq`). Peak memory
is now O(n·T) instead of O(pixels·T).

#### S3 — Dead `dims_native` assignment in `mode_plane_split` (~line 1261)

**Problem:**
`dims_native = sample_shape[1:]` was assigned but never referenced.

**Fix:**
Line removed entirely.

---

## 2026-06-17

### Pre-compute motion correction once before Bayesian tuning

**Problem:**
Motion correction (piecewise-rigid, `pw_rigid=True`) was being re-run inside every
Bayesian trial via `run_cnmf(..., do_mc=True)`. With `n_calls=10`, this meant 11 MC
runs per tune dataset (10 trials + 1 post-tune re-run) even though MC output is
identical across trials — only CNMF hyperparameters (`gSig`, `min_corr`, `rf`, etc.)
vary.

**Changes:**
- Added `run_motion_correction(fname_mmap)` function: calls `MotionCorrect` directly
  with `BASE_PARAMS` MC settings (`max_shifts`, `strides`, `overlaps`,
  `max_deviation_rigid`, `pw_rigid`), saves corrected movie, returns corrected memmap
  path.
- Added `from caiman.motion_correction import MotionCorrect` import.
- `bayesian_tune()` objective: changed `run_cnmf(tp, mmap_path)` to
  `run_cnmf(tp, mmap_path, do_mc=False)` — caller now pre-MCs the mmap.
- All four mode functions (`time-split`, `plane-split`, `file-plane-split`,
  `file-split`): call `run_motion_correction(tune_mmap)` once before
  `bayesian_tune()`, then pass `mc_tune_mmap` to `bayesian_tune()` and to the
  post-tune re-run with `do_mc=False`.
- `test_cnmf()` is unchanged — each test dataset is fitted exactly once so MC there
  runs once with no redundancy.

**Expected speedup:**
`(n_calls - 1) + 1 = n_calls` fewer MC runs per tune dataset.
With default `n_calls=10`: 10 fewer MC runs per mode invocation.
Estimated wall-clock reduction: 30–60% depending on movie length and resolution.
