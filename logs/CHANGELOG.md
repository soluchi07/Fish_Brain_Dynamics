2026-06-17 — p4_universal.py: Pre-compute motion correction once before Bayesian tuning

Problem:
  Motion correction (piecewise-rigid, pw_rigid=True) was being re-run inside every
  Bayesian trial via run_cnmf(..., do_mc=True). With n_calls=10, this meant 11 MC
  runs per tune dataset (10 trials + 1 post-tune re-run) even though MC output is
  identical across trials — only CNMF hyperparameters (gSig, min_corr, rf, etc.) vary.

Changes:
  - Added run_motion_correction(fname_mmap) function: calls MotionCorrect directly
    with BASE_PARAMS MC settings (max_shifts, strides, overlaps, max_deviation_rigid,
    pw_rigid), saves corrected movie, returns corrected memmap path.
  - Added `from caiman.motion_correction import MotionCorrect` import.
  - bayesian_tune() objective: changed run_cnmf(tp, mmap_path) to
    run_cnmf(tp, mmap_path, do_mc=False) — caller now pre-MCs the mmap.
  - All four mode functions (time-split, plane-split, file-plane-split, file-split):
    call run_motion_correction(tune_mmap) once before bayesian_tune(), then pass
    mc_tune_mmap to bayesian_tune() and to the post-tune re-run with do_mc=False.
  - test_cnmf() is unchanged — each test dataset is fitted exactly once so MC there
    runs once with no redundancy.

Expected speedup:
  (n_calls - 1) + 1 = n_calls fewer MC runs per tune dataset.
  With default n_calls=10: 10 fewer MC runs per mode invocation.
  Estimated wall-clock reduction: 30-60% depending on movie length and resolution.
