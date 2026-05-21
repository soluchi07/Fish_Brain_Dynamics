<!--
Slide deck for the 13iii26 cross-task validation runs.
Designed to live at the root of the Fish_Brain_Dynamics repo so image paths resolve.

Render options:
  - Copy section-by-section into Google Slides (one slide per "---" block)
  - Or render directly with Marp:
      npm install -g @marp-team/marp-cli
      marp slides_4runs.md -o slides_4runs.pdf
  - Speaker notes live in HTML comments
-->

---
marp: true
theme: default
paginate: true
---

# Cross-Task Validation
## 13iii26 — same fish, two tasks, four runs

**Kiitan Ayandosu**
Phase 4 of the pipeline (`p4_universal.py`)

<!--
Speaker notes:
- This builds on 7iii25 Tasks 4/5/6 work
- Now we're testing whether tuned CNMF parameters GENERALIZE
- Single fish, two task recordings, four cross-validation experiments
-->

---

## What changed since 7iii25

| | **7iii25 (old)** | **13iii26 (new)** |
|---|---|---|
| File format | many `tp-*.lux.h5`, each `(Z=7, 2048, 2048)` | many `Cam_long_*.lux.h5`, each `(1, 2048, 2048)` |
| Z-planes per timepoint | 7 | **1** |
| Tasks per fish | 4, 5, 6 | task1, task3, task6 |
| Frames per task | 42–56 | 50 |
| What we can test | within-task baselines | **cross-task generalization** |

<!--
The data shape changed shape between datasets. Old data had multiple Z-planes per timepoint.
New data has a single plane. So "multi-plane" became "multi-task on the same plane".
-->

---

## We saw four flavors of imaging data this month

- **7iii25** — `multi-tp`: many timepoint files, Z=7 → can do plane-split
- **13iii26** — `multi-cam`: many timepoint files, Z=1 → cross-task only
- **20iv26** — `single-movie`: one big file, T=4900 frames → too big for naive load
- **12iii26 / 17iv26** — corrupt or `NO_DATA_KEY` headers → skipped (~257 GB cleaned)

The pipeline now **auto-detects** which format it is loading and adjusts accordingly.

<!--
This is why we built p4_universal.py — one script handles all formats.
The corrupt files were freed to disk.
-->

---

## Why we settled on 13iii26 for this round

- **Three usable task folders** under one fish (`task1stack0`, `task3stack0`, `task6stack0`)
- All three have the same imaging plane (Z=1) on the same animal
- Recordings are short (50 frames each) but **clean and consistent**
- No file corruption, no missing data keys
- Lets us ask: *do tuned CNMF params transfer between behavioral tasks?*

13iii26 isn't multi-Z — but it gives us **multi-task on the same plane**, which is a different and equally useful generalization axis.

---

## Re-framing "multi-plane" → "multi-task"

Originally the professor's split-by-plane test required Z>1.

Since 13iii26 is Z=1, we adapted by treating each task folder as the equivalent of a "plane":

- **Within-file (time-split)** — tune on first half of frames, test on second
- **Cross-file (file-plane-split)** — tune on task1, test on task3 (and reverse)

Same logic, different generalization axis.

---

## The Pipeline — `p4_universal.py`

```text
.lux.h5 directory  ──→  auto-detect format  ──→  load (T, H, W) movie
    ↓
512×512 downsample   ←── tradeoff: speed for proof of concept
    ↓
column-median stripe removal
    ↓
brain mask via Otsu  ←── kills components in dark periphery
    ↓
CNMF + Bayesian search (8 trials, gSig / gSig_filt / min_corr / min_pnr / rf / p)
    ↓
quality filters: circularity ≥ 0.5, max-area, in-mask centroid, SNR
    ↓
positive composite score = (1 − recon_error) + ½·compactness − 0.3·log(sparsity) + stability
```

<!--
The composite score is now POSITIVE — addresses professor's earlier feedback.
Quality filters are inside every Bayesian trial so the optimizer is rewarded for finding REAL neurons.
-->

---

## Why we deferred Sweet3D / 3D segmentation

Bio team raised three concerns:

> 1. *"Catching multiple neurons — how to stop that"*
> 2. *"Shouldn't capture above the planes"*
> 3. *"Sweet3D rather than CNMF for boundaries — when neurons fire together it's hard to distinguish"*

**Our staged response:**

| Concern | This phase (`p4`) | Future phase (`p5`) |
|---|---|---|
| Over-merged blobs | `max-area` quality filter | Sweet3D segmentation |
| Out-of-brain detections | Otsu brain mask + in-mask centroid filter | Volumetric mask |
| Non-circular footprints | circularity ≥ 0.5 filter | Sweet3D circular priors |

We get most of the benefit now without rewriting the pipeline from scratch.

---

## Preprocessing — same fish, same stripe artifact

![w:900](results/13iii26_task1_timesplit/preprocess_movie.png)

The light-sheet column-stripe pattern is identical to 7iii25 — same hardware, same fix.

---

## Brain mask — task1 vs task3 (same fish, same plane)

<div style="display: flex; gap: 20px;">
  <div style="flex: 1;">
    <h3>Task 1 — 20.7% coverage</h3>
    <img src="results/13iii26_task1_timesplit/brain_mask_movie.png" width="450">
  </div>
  <div style="flex: 1;">
    <h3>Task 3 — 20.2% coverage</h3>
    <img src="results/13iii26_task3_timesplit/brain_mask_movie.png" width="450">
  </div>
</div>

Mask outlines (lime green) match between tasks → confirms the same imaging plane.

---

## Run 1 — Task 1 time-split

![w:700](results/13iii26_task1_timesplit/contours_full_movie.png)

- **66 kept neurons** (266 raw → 75% rejected by quality filters)
- Composite score **+0.28** | tune↔full stability **0.43**
- Best params: `gSig=2, min_corr=0.47, min_pnr=3, rf=80, p=1`

---

## Run 1 — sample traces

![w:700](results/13iii26_task1_timesplit/traces_full_movie.png)

Some clean transients, some flat — typical of permissive `min_pnr=3` thresholds with 50 frames.

---

## Run 2 — Task 3 time-split

![w:700](results/13iii26_task3_timesplit/contours_full_movie.png)

- **51 kept neurons** (236 raw)
- Composite score **+0.35** | tune↔full stability **0.50**
- Best params: `gSig=2, min_corr=0.51, min_pnr=3, rf=80, p=1`

Almost identical params to Task 1 → pipeline is consistent across tasks.

---

## Run 3 — Task 1 → Task 3 (cross-task transfer)

![w:700](results/13iii26_task1_to_task3/contours_test_file.png)

- Tune on task1, **apply best params to task3 unseen**
- **50 kept neurons** in task3 with task1's params
- **Stability = 1.00** — every footprint from task1 found a match in task3

This is the headline result.

---

## Run 4 — Task 3 → Task 1 (reverse direction)

![w:700](results/13iii26_task3_to_task1/contours_test_file.png)

- Tune on task3, apply to task1
- **56 kept neurons** | **stability = 0.96**
- Composite score **+0.82**

The transfer is **bidirectional and near-perfect**.

---

## Bayesian search converged to the same answer in all four runs

| Run | gSig | gSig_filt | min_corr | min_pnr | rf |
|---|---|---|---|---|---|
| task1 time-split | 2 | 3 | 0.47 | 3 | 80 |
| task3 time-split | 2 | 3 | 0.51 | 3 | 80 |
| task1 → task3 | 2 | 3 | 0.52 | 3 | 80 |
| task3 → task1 | 2 | 3 | 0.52 | 3 | 80 |

The optimizer finds the same regime regardless of which task we tune on.

![w:700](results/13iii26_task1_timesplit/convergence_time_split.png)

---

## Quality filter breakdown — what's killing components

![w:900](results/13iii26_task1_timesplit/quality_filters_time_split.png)

| Filter | Avg rejection | What it catches |
|---|---|---|
| circularity (≥ 0.5) | ~6% | Stringy / non-blob shapes |
| **max-area** | **~36%** | **Over-merged neuron blobs** |
| **in-mask** | **~32%** | **Components outside the brain** |

≈75% of CNMF's raw output is filtered away → only ~25% survives.
This is exactly addressing the bio team's "catching multiple neurons" concern.

---

## Stability tells the cleanest story

| Comparison | Stability | Interpretation |
|---|---|---|
| time-split: tune-half ↔ full movie | **0.43 / 0.50** | Half the tune neurons recovered when CNMF sees full movie |
| time-split: tune-half ↔ test-half | 0.0 | 25 frames of test data → too sparse to recover |
| **cross-task: task1 ↔ task3** | **0.96 / 1.00** | Same fish, same plane → spatial footprints transfer perfectly |

Cross-task transfer is **dramatically more stable** than within-task halves —
because the limiting factor is **frame count**, not parameter quality.

---

## Summary — four runs

| Run | Mode | Raw → Kept | Composite | Stability |
|---|---|---|---|---|
| Task 1 | time-split | 266 → 66 | +0.28 | 0.43 |
| Task 3 | time-split | 236 → 51 | +0.35 | 0.50 |
| **Task 1 → Task 3** | file-plane-split | 215 → 50 | **+0.86** | **1.00** |
| **Task 3 → Task 1** | file-plane-split | 234 → 56 | **+0.82** | **0.96** |

**Verdict:** parameters tuned on one task transfer essentially perfectly to another task on the same fish/plane.

---

## What's working

- Pipeline runs end-to-end on **any** of our 4 file formats (auto-detected)
- Composite score is now **positive and interpretable** (was negative before)
- Quality filters reject ~75% of CNMF noise → output is dominated by real-looking neurons
- **Cross-task footprint stability = 0.96–1.00** — the strongest result so far
- Bayesian search converges in ~4 useful trials → cheap to repeat on new datasets

---

## Current limitations

- **Time-split is data-starved** at 50 frames per task (test-half stability = 0)
- **Brain mask too tight** at 20% coverage — rejects ~32% of components, some likely real
- **Max-area filter too aggressive** at default `factor=4` — rejects ~36%, again partly real
- **3 of 8 Bayesian trials wasted** on unproductive `min_pnr ≥ 5` regime
- Running at 512×512 (downsampled from 2048×2048) — gives speed but loses detail

<!--
Honesty slide. These are all addressable in the next phase.
-->

---

## Next steps

1. **Loosen quality filters** (`--max-area-factor 8`, dilate brain mask) → expect ~30% more kept neurons
2. **Tighten Bayesian prior** to `min_pnr ∈ [3, 5]` → no more wasted trials
3. **Re-run at full 2048×2048** on best params from this phase (no info loss)
4. **Phase 5: Sweet3D segmentation** for true single-cell boundaries when neurons co-fire
5. **Push to longer recordings** (200+ frames) so time-split mode actually discriminates

<!--
Roadmap addresses every limitation listed on the previous slide.
-->

---

## Questions?

**Repo:** [github.com/Mikito-Coder/Fish_Brain_Dynamics](https://github.com/Mikito-Coder/Fish_Brain_Dynamics)

All outputs (plots, traces, parameter logs, JSON summaries, master CSV) under
`results/13iii26_task{1,3}_*/`

**Script:** `p4_universal.py` — universal loader, 4 cross-validation modes, quality filters, brain mask, Bayesian tuning.

<!--
End with a pointer so reviewers can dig into the data themselves.
-->
