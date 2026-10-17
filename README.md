# Measurement Resolution of Video Anomaly Detection Benchmarks — code release

Companion code and data for the paper *"Measurement Resolution of Video Anomaly
Detection Benchmarks: Seed Variance, Detectability Bounds, and a Calibrated
Premise Test for Oscillatory Priors"* (submitted to IEEE T-CSVT).

The paper asks two questions and this repository is the instrument for both:

1. **Is the premise true?** Do anomalies in CLIP feature space actually carry
   oscillatory structure in the band a behavioural rhythm would occupy
   (periods of 10–21 frames)?
2. **Could the benchmark have resolved the difference anyway?** How many
   random seeds does a claimed gain of δ AP points actually require?

There is no new detector here. The premise and resolution halves are
measurement code and run on frozen, pre-extracted CLIP features with no GPU.
Three things here are training-side rather than measurement-side, and they are
the only parts that ever needed a GPU: `architecture/`, the two training
harnesses behind the architecture and mechanism tables; `logs/`, the raw
per-seed training logs behind two of those tables; and `batch_rev.sh`, the
driver that produced them. Together they mean the arms can be re-run, not only
re-read.

---

## 1. What is in here

| Directory | Contents |
|---|---|
| `premise_test/` | `spectral.py` — the spectral-shape statistic **and** its injection calibration. This is the single source of truth: every table and figure that quotes a band share or a z score is produced by this file, so the numbers cannot drift apart. `cluster_recheck.py` re-runs the same statistic with the **video** rather than the overlapping window as the analysis unit, and refuses to report anything unless its window-level z first reproduces the published table exactly. |
| `controls/` | The four controls: `backbone.py` (backbone), `jitter_v1.py` and `jitter.py` (encoder jitter — the first gives the noise floor, the second the sign of the shift), `compress.py` (compression), `shuffle_ped2.py` (temporal order), plus the XD and Avenue shuffle controls (`shuffle_xd.py`, `shuffle_avenue.py`) and the multi-backbone feature extractor (`extract_backbones.py`). |
| `injection/` | `avenue_injection.py` — power-matched tone injection on CUHK Avenue, giving the minimum detectable tone. |
| `resolution/` | The five-seed audit of the reference implementation, the epoch-budget measurement, and the resolution half of the paper: `mde_table.py` prints the minimum-detectable-effect / seeds-needed table, `mde_and_sigma_ci.py` reads that table and adds the σ confidence intervals, and `arms_paired.py` recomputes the multi-scale arm table and the paired same-seed comparison table from the four-seed values and **exits non-zero** unless every printed cell comes out. |
| `complementarity/` | `wave_complement.py` — the residual test that asks what the oscillatory (wave) operator adds *after* the trained trunk score has been accounted for, against a plain magnitude control — `residual_test.py`, which persists the result for all four candidate scores, `mechanisms.py`, which measures all four mechanisms the paper says do not help against one shared no-wave control at one seed, and `cluster_residual.py`, which re-does the residual test with the **video** rather than the segment as the resampling unit and reports the design effect. |
| `protocol/` | `r6_evidence.py` — locates every line of the reference implementation that the paper quotes, prints it, and **fails loudly** if the text is not there. Read-only; it never writes to the reference tree. |
| `figures/` | The plotting scripts: `figs_main.py` (Figs. 1, 2, 3, 8), `figs_backbone.py` (Figs. 4, 5), `figs_xd_shuffle.py` (Fig. 6), `figs_avenue.py` (Fig. 7), and `figs_ped2.py`, which renders two Ped2 diagnostics the paper does not print. They emit **vector PDF** (Okabe–Ito palette, no in-figure titles) so the figures are print-ready. |
| `results/` | The captured output of every script above — the auditable evidence. |
| `checklist/` | The ten-item minimum reporting standard from the paper, as a standalone markdown file you can copy into your own paper. |
| `architecture/` | The training code behind every arm and mechanism in the paper. **This is the only third-party code in the repository**: `architecture/*/clip/` is a vendored and modified copy of OpenAI's CLIP, MIT, under `architecture/CLIP-LICENSE`. Everything else in the repository is our own. |
| `seedstats.py` | The single source of truth for the seed statistics and for parsing training logs: sample mean and sd, the paired *t* test, Holm–Bonferroni, and the log readers. Nothing below computes these itself, because the same quantity is quoted in more than one table and the failure mode we are guarding against is two tables computing it by different routes and drifting apart. Standard library only — no numpy, no scipy — so "no third-party code" stays true. |
| `logs/` | The raw per-seed training logs behind the paper's tables, kept verbatim so the numbers can be recomputed rather than trusted: `arms_n8/` (the three multi-scale arms at eight seeds), `decimation_n8/` (the decimation arm that separates the pooling kernel from the branch factors), `mechanism_n4/` (the two mechanism rows re-measured at four seeds), `wavefield_n4/` (the wave field re-measured at four seeds at one batch size), `wavefield_factorial/` (the same operator over batch size, fusion and gain), `wavefield_warm/` (the four warm-start rows against a per-seed `A_0`), and `batch_summary.txt` (one line per run). |
| `wf_batch.sh` | The driver that produced `logs/wavefield_n4/`, with the exact command line for every run. Resumable. **Not needed to reproduce any table in this repository** — it is here so the logs in `logs/` are traceable, not so you can re-run a GPU batch. |
| `wf2_batch.sh` | The driver that produced `logs/wavefield_factorial/` and `logs/wavefield_warm/`. Part A is the factorial; Part B is gated behind `WITH_WARM=1` and trains a converged `A_0` per seed, because only one seed's `A_0` survived on disk. Same caveat as above: shipped for traceability. |
| `check_release_consistency.py` | **Run this first.** Exits non-zero if anything in this repository still claims something the manuscript has stopped saying, or if a table the manuscript quotes has no shipped report and no shipped logs behind it. |

Three of these are newer than the rest and cover the tables added in revision:

| Script | Reproduces |
|---|---|
| `resolution/decimation_arm.py` | Table 12 (`tab:decimate`) and the lower block of Table 9 (`tab:paired`). Reads `logs/`, checks the alignment gate first — that the decimated arm reports the same factors and a different pooling mode, and nothing else — and **exits non-zero** unless every value the paper prints is reproduced. |
| `complementarity/mechanism_rerun.py` | The bottom block of Table 7 (`tab:mechanisms`): the (b) wave resonator bank and the (c) PDE residual, re-measured as same-seed paired differences at four seeds. Prints the n=1 screen value beside the measurement rather than replacing it, and **exits non-zero** unless the paper's numbers come out. |
| `complementarity/wavefield_factorial.py` | The (a) rows of Table 7 that come from `logs/wavefield_factorial/`: five cells over batch size, fusion wiring and field gain, each differenced against **its own** no-wave control at the same batch size. This is the script that resolved the batch-size confound behind the published −0.0754 row, and it prints the per-batch-size means and the shift between them, not just the cells. |
| `complementarity/wavefield_warm.py` | The four warm-start (a) rows of Table 7, from `logs/wavefield_warm/`, paired against a per-seed `A_0`. It first checks that every run was set up as intended — `[Freeze] trunk frozen` present on exactly the frozen rows, `[InitCkpt]` present on all sixteen — and says so in its output, because a warm row that silently ran cold would be indistinguishable from a null. |

All four need no configuration and no data beyond `logs/`:

```bash
python resolution/decimation_arm.py
python complementarity/mechanism_rerun.py
python complementarity/wavefield_factorial.py logs/wavefield_factorial
python complementarity/wavefield_warm.py      logs/wavefield_warm
```

These reproduction scripts carry the caveat they are meant to be read with, in
their output and in their docstrings: `decimation_arm.py` states that the
contrast attributes the difference to the pooling **kernel** and does not
isolate frame averaging from anti-aliasing, and `mechanism_rerun.py` states
that a null is not equivalence. The two wave-field scripts state a second one:
the batch-96 rows are the only rows in Table 7 whose uncorrected `p` falls
below 0.05, and Holm–Bonferroni over the ten re-measured rows takes them to 0.074
and 0.203, so **no** row of that table is significant once the number of rows
tested is accounted for. `wavefield_warm.py` also notes that its four rows ran
at batch 64, which `wavefield_factorial.py` shows is where the effect is
smallest, so they are a lower bound rather than a measurement at the budget that
exposes it. See §2b for what the kernels themselves are.

## 2. What is *not* in here

* **Everything outside the two harnesses in `architecture/`.** The training code
  that produced the arms and mechanisms is now here, but only the parts a number
  in the manuscript depends on. Left out: the `*_wpo.py` visual branch, which is
  a different architecture and backs no table; `_diag_*` / `_smoke_*` /
  `diagnose_*` development scaffolding; and `wave_decode*.py`. Two harnesses are
  shipped rather than one because they share no imports — each imports its
  modules top-level, so each must be run from its own directory — and because
  they are not comparable to each other: `src/` has no multi-scale branch and
  `src_sgl/` has no wave field. That is why the wave-field rows of Table 7 are not
  comparable to the other mechanism rows, and why they are re-measured against
  their **own** no-wave controls inside `src/` instead of against the shared
  control the other three use — mixing the harnesses to force one control would
  have introduced a larger confound than the one being measured. An earlier
  version of this file said those rows "cannot be re-measured"; that was wrong.
  They are re-measured at four seeds over batch size, fusion and gain, and the
  paper's limitations no longer claim otherwise.
* **The reference VadCLIP pipeline** from which the multi-scale branches inherit.
  It is a third-party codebase with its own licence; we reference it, we do not
  redistribute it. Two things here read it:
  `resolution/five_seed_audit.py` reads *training logs* from a run of that
  pipeline, so you need your own run to reproduce that number; and
  `protocol/r6_evidence.py` reads the *source tree* read-only to check the lines
  the paper cites. Point `VADCLIP_SRC` at your checkout — if you have a
  different revision, that script is the check: it asserts the exact text on
  each cited line and exits non-zero rather than letting a stale citation pass.
* **The datasets.** UCSD Ped2, CUHK Avenue, XD-Violence and UCF-Crime are
  distributed by their own authors under their own terms. Two scripts read the
  *released* XD-Violence feature archive rather than anything of ours
  (`controls/shuffle_xd.py` and `premise_test/oscillatory_by_class.py`); point
  `XD_FEATURES_ZIP` at your copy of it.
* **The extracted features.** The scripts read pre-extracted CLIP features from
  disk; see §3. The Avenue features in particular are *not* shipped, so the
  Avenue half of `premise_test/cluster_recheck.py` is skipped unless you point
  `AVENUE_FEAT` at your own extraction.
* **The wave-probe dumps.** `complementarity/` reads two intermediate dumps —
  `aligned_pair.npz` (trunk scores aligned to segment labels) and
  `wave_energy_test.npz` (the wave-probe output). They are derived from
  XD-Violence test features, so regenerate them from your own feature dump and
  point `XD_PAIR_DIR` at the directory holding them.
* **The per-arm XD score dumps.** `complementarity/mechanisms.py` reads
  `scores_xd_<arm>_s234.npz` (one score per segment per video, plus frame
  ground truth) from `XD_SCORE_DIR`. These are outputs of our own training
  runs, so they are not shipped; the shipped `results/mechanisms.txt` is their
  captured report.
* **A few internal helpers.** Some docstrings name working-tree scripts that
  are not part of this release — the Ped2 and Avenue feature extractors, the
  pooled XD diagnostic, the baseline-log auditor, the superseded v1 shuffle and
  the alignment checker for the XD dumps. They are cited as provenance, not as
  entry points; the entry points are the ones listed in §5. Note also that the
  working tree's `ops/` prefix is gone and the files are grouped by role, so a
  docstring may still call a module by its old name (`_ped2_spectral` is
  `premise_test/spectral.py` here, `_ped2_spectral_bb` is
  `controls/backbone.py`, `_ped2_extract_bb` is `controls/extract_backbones.py`).
  The statistic itself is imported, never copied: every consumer carries a
  short shim that puts `premise_test/` on `sys.path`.

## 3. Configuring paths

Every absolute path in the original working copies has been changed to an
environment variable with the author's original value as the fallback, so the
scripts run unchanged for the author and are portable for everyone else.

| Variable | Meaning | Used by |
|---|---|---|
| `PED2_FEAT` | directory of extracted Ped2 patch/global features | `premise_test/spectral.py`, `controls/shuffle_ped2.py` |
| `PED2_RAW` | raw UCSD Ped2 frame directory | `controls/compress.py`, `controls/jitter*.py`, `controls/extract_backbones.py` |
| `XD_FEATURES_ZIP` | XD-Violence released feature archive | `controls/shuffle_xd.py`, `premise_test/oscillatory_by_class.py` |
| `CLIP_CKPT` | CLIP weight download cache | `controls/extract_backbones.py` |
| `BASELINE_LOG` | training log(s) of the reference pipeline | `resolution/five_seed_audit.py` |
| `FIG_OUT` | where figures are written | `figures/figs_ped2.py`, `figures/figs_backbone.py` |
| `XD_PAIR_DIR` | directory holding `aligned_pair.npz` (aligned trunk scores + labels), `wave_energy_test.npz` (the wave-probe dump) and `scores_xd_base_s234.npz` (the trunk dump whose per-video segment counts give the cluster units) | `complementarity/wave_complement.py`, `complementarity/residual_test.py`, `complementarity/cluster_residual.py` |
| `XD_SCORE_DIR` | directory holding `scores_xd_<arm>_s234.npz` | `complementarity/mechanisms.py` |
| `VADCLIP_SRC` | the `src/` directory of a VadCLIP checkout | `protocol/r6_evidence.py` |
| `AVENUE_FEAT` | directory of extracted CUHK Avenue features | `premise_test/cluster_recheck.py` (Avenue half) |
| `CLUSTER_OUT`, `MECH_OUT`, `R6_OUT`, `RESIDUAL_OUT`, `CLUSTER_RESID_OUT` | override where those five reports are written | `premise_test/cluster_recheck.py`, `complementarity/mechanisms.py`, `protocol/r6_evidence.py`, `complementarity/residual_test.py`, `complementarity/cluster_residual.py` |
| `RESULTS_DIR` | where the shipped `.txt` evidence is read from (used as a control input) | `complementarity/cluster_residual.py` |

Paths that were already relative to the repository root (the results
directory) are left alone.

```bash
export PED2_FEAT=/path/to/ped2_feat
export PED2_RAW=/path/to/UCSDped2
```

## 4. Requirements

See `requirements.txt`. In short:

```
numpy, scipy        # the premise test, the statistics, the figures' data
matplotlib          # the figures
torch, torchvision, Pillow, clip, scikit-learn
                    # feature extraction, the image controls and
                    # complementarity/ only
```

`premise_test/spectral.py` deliberately depends on **numpy only**, so the
central claim can be checked without installing a deep-learning stack.
`clip` is the OpenAI CLIP package (`pip install
git+https://github.com/openai/CLIP.git`); the package of that name on PyPI is a
different one.

## 5. Reproducing the numbers

### 5a. The measurement code (no GPU)

```bash
# the premise test and its calibration (Ped2, CLIP ViT-B/16)
python premise_test/spectral.py            # -> results/spectrum_ped2.txt

# the same statistic with the VIDEO as the unit (skips Avenue without AVENUE_FEAT)
python premise_test/cluster_recheck.py     # -> results/cluster_recheck.txt

# the same question per class and per window size (needs XD_FEATURES_ZIP)
python premise_test/oscillatory_by_class.py > results/oscillatory_by_class.txt

# the four controls
python controls/backbone.py                # -> results/backbone_robustness.txt
python controls/jitter_v1.py               # -> results/jitter_floor.txt
python controls/jitter.py                  # -> results/jitter_sign.txt
python controls/compress.py                # -> results/compress.txt
python controls/shuffle_ped2.py            # -> results/shuffle_ped2.txt   (Ped2, 0.026)
python controls/shuffle_xd.py              # -> results/xd_shuffle.txt     (XD,   0.015)
python controls/shuffle_avenue.py          # -> results/avenue_spectral.txt(Avenue,0.103)

# the injection calibration on Avenue
python injection/avenue_injection.py       # -> results/avenue_injection.txt

# the resolution half
python resolution/five_seed_audit.py       # -> results/field_baseline.txt
python resolution/epoch_budget.py          # -> results/epoch_budget.txt
python resolution/mde_table.py             # -> results/mde_table.txt
python resolution/mde_and_sigma_ci.py      # -> results/sd_ci.txt (reads mde_table.txt)

# the complementarity half: what the oscillatory operator adds over a control
python complementarity/residual_test.py    # -> results/residual_test.txt
python complementarity/cluster_residual.py # -> results/cluster_residual.txt
python complementarity/mechanisms.py       # -> results/mechanisms.txt

# every claim the paper makes about the reference implementation
python protocol/r6_evidence.py             # -> results/r6_evidence.txt; exits 1 on a mismatch

# the rows re-measured at n=4 for Table 7 (tab:mechanisms), bottom block
python complementarity/mechanism_rerun.py   # -> results/mechanism_rerun.txt
python complementarity/wavefield_factorial.py logs/wavefield_factorial
                                           # -> results/wavefield_factorial.txt
python complementarity/wavefield_warm.py    logs/wavefield_warm
                                           # -> results/wavefield_warm.txt

# last, and it is the one that has caught the most: does anything in this
# repository still claim something the manuscript has stopped saying?
python check_release_consistency.py
```

`complementarity/residual_test.py` imports the residual machinery from
`complementarity/wave_complement.py` rather than reimplementing it, so the
number it persists and the verdict that script prints cannot drift apart.
`premise_test/cluster_recheck.py` is the same idea one level up: it imports
`premise_test/spectral.py` for the statistic, and aborts unless its own
window-level z reproduces the published table, so its video-level numbers
cannot be a different statistic wearing the same name.

Several scripts write their report to **standard output**; the `.txt` files in
`results/` are the captured output, e.g.

```bash
python premise_test/spectral.py > results/spectrum_ped2.txt
```

### Which artefact backs which table and figure

Every table and every figure in the paper traces to one shipped script and one
shipped result file. Nothing in the paper is quoted from an artefact that is
not in this repository. Figure and table numbers below are the **printed** ones
(the manuscript root's `F5_*.pdf`, `F19_*.pdf` … file names are the working
tree's internal ledger IDs: `F5` is printed as Fig. 3, `F19` as Fig. 6, `F20`
as Fig. 7, and so on).

| Paper artefact | Script | Shipped result |
|---|---|---|
| Table `tab:ped2` (premise test) | `premise_test/spectral.py` | `results/spectrum_ped2.txt` |
| Table `tab:cluster` (video as unit) | `premise_test/cluster_recheck.py` | `results/cluster_recheck.txt` |
| Table `tab:perclass`, Fig. 3 (`F5_perclass_spectrum`) | `premise_test/oscillatory_by_class.py` | `results/oscillatory_by_class.txt` |
| Table `tab:backbone`, Fig. 4 (`F13_backbone_robustness`) | `controls/backbone.py` | `results/backbone_robustness.txt` |
| Fig. 5 (`F14_power_matched_calibration`) | `controls/backbone.py` | `results/backbone_robustness.txt` |
| Table `tab:avenue`, Fig. 7 (`F20_avenue_replication`) | `figures/figs_avenue.py`; data from `controls/shuffle_avenue.py` and `injection/avenue_injection.py` | `results/avenue_spectral.txt`, `results/avenue_injection.txt` |
| Fig. 6 (`F19_xd_shuffle`) | `figures/figs_xd_shuffle.py`; data from `controls/shuffle_xd.py` | `results/xd_shuffle.txt` |
| Table `tab:field` (five-seed audit) | `resolution/five_seed_audit.py` | `results/field_baseline.txt` |
| Table `tab:mde` (seeds needed) | `resolution/mde_table.py` → `resolution/mde_and_sigma_ci.py` | `results/mde_table.txt`, `results/sd_ci.txt` |
| Table `tab:arms` | `resolution/arms_paired.py` | `results/arms_paired.txt` (the AP half is also in `results/field_baseline.txt`) |
| Table `tab:paired` | `resolution/arms_paired.py` | `results/arms_paired.txt` |
| Table `tab:residual` | `complementarity/residual_test.py` | `results/residual_test.txt` |
| Table `tab:mechanisms` | `complementarity/mechanisms.py` | `results/mechanisms.txt` |
| Fig. 1 (`F6_band_auc`) | `figures/figs_main.py` | — (data is in the script) |
| Fig. 2 (`F7_multiscale_arms`) | `figures/figs_main.py` | — (same per-seed values as `results/arms_paired.txt`) |
| Fig. 8 (`F9_effect_vs_noise`) | `figures/figs_main.py` | — (the noise band is `results/arms_paired.txt`) |
| `tab:checklist` | `checklist/reporting_checklist.md` | — (no numbers) |
| Table `tab:decimate` (§5.3 arm) | `architecture/src_sgl/mamba_vad_dual_model.py`, `architecture/src_sgl/xd_train_sgl.py` | `results/decimation_arm.txt` (recomputed by `resolution/decimation_arm.py`) |
| `tab:mechanisms`, bottom block (n=4 re-measurement) | `architecture/src_sgl/wave_resonator.py`, `architecture/src_sgl/wave_tv.py` | `results/mechanism_rerun.txt` (recomputed by `complementarity/mechanism_rerun.py`) |
| `tab:mechanisms`, (a) row (wave field, n=4, one batch size) | `architecture/src/wave_field.py` | `results/wavefield_rerun.txt` (recomputed by `complementarity/wavefield_rerun.py`) |
| `tab:mechanisms`, (a) rows over batch size, fusion and gain | `architecture/src/wave_field.py` | `results/wavefield_factorial.txt` (recomputed by `complementarity/wavefield_factorial.py` from `logs/wavefield_factorial/`) |
| `tab:mechanisms`, (a) warm-start rows (frozen and joint, two fusions) | `architecture/src/wave_field.py` | `results/wavefield_warm.txt` (recomputed by `complementarity/wavefield_warm.py` from `logs/wavefield_warm/`) |

### 5b. The arms and mechanisms (GPU)

`architecture/` holds two harnesses. They are **not** interchangeable and are
**not** comparable: `src/` carries the reference backbone and the wave-field
operator, `src_sgl/` carries the multi-scale branches and the other three
mechanisms. Neither contains the other's branch, which is the structural reason
the wave-field rows of Table 7 are not comparable to the rest: they are measured
within `src/` against their own controls at their own batch size, and the paper
states the resulting limits in its limitations rather than papering over them.

Each harness imports its modules top-level, so **run from inside its own
directory**, exactly as in the original working copies:

```bash
cd architecture/src_sgl
# the default arm (Table tab:arms) and the decimation arm (§5.3)
python xd_train_sgl.py --multi-scale --multi-scale-factors 1,8,16 ...
python xd_train_sgl.py --multi-scale --ms-pool-mode decimate ...
# mechanisms (b) and (c), the two re-measured at n=4
python xd_train_sgl.py --multi-scale --use-wrb --wrb-where fusion ...
python xd_train_sgl.py --multi-scale --wave-loss 2.5 ...
```

`--ms-pool-mode decimate` holds the branch factors at 1,8,16 and replaces
`F.avg_pool1d(kernel=f, stride=f)` with a pick of every *f*-th frame, so stride,
output sampling rate, output length and all weights match the default arm. It is
worth being precise about what that does **not** establish: `avg_pool1d` with
`kernel == stride` is the standard **anti-aliased** downsample, so its boxcar is
at once an average over *f* consecutive frames and a low-pass that suppresses
fold-over above the new Nyquist rate. Dropping it removes the averaging **and**
admits aliasing. The contrast therefore attributes the difference to the pooling
**kernel**, and the 64% figure in §5.3 is a ratio of two measured differences,
not an additive decomposition. Separating mixing from anti-aliasing would need a
windowed-sinc resampler, which was not run.

The `architecture/list/*.csv` paths are **relative** (`../../../dataset/...`) and
resolve against the *current working directory*, not against the CSV. Lay the
repository out so that `architecture/list/` sits three levels above your feature
root, or regenerate the lists.

Beyond `requirements.txt` you need `mamba_ssm` (2.2.4 on the machine these were
measured on; the official kernels are CUDA-only) and the CLIP ViT-B/16
checkpoint, which `clip.load` downloads on first use. Measured on one RTX 2080
Ti (11 GB), batch 64, one epoch: about 290 s per run. `--save-min-ap 0.999`
writes no checkpoint, which is how these tables were produced — they were read
from logs, not from weights.

`figures/figs_ped2.py` renders two further Ped2 diagnostics (the two-view
spectrum and the injection calibration) that the paper does not print; it is
here because it is the same instrument on the same data.

All five figure scripts write **vector PDF** — their internal `save()` helpers
rewrite the extension, so the `.png`-looking call sites still produce
`F*.pdf`, which is what the manuscript includes.

## 6. Read this before you cite `results/shuffle_ped2.txt`

There are two shuffle implementations, and only one is correct.

* `results/superseded/shuffle_ped2_v1_SUPERSEDED.txt` — the **first** attempt.
  It re-encoded every clip from raw frames, fitted PCA on test data only, and
  averaged per clip instead of per window; its `d_orig` did not reproduce the
  published premise delta (the slowest band even flipped sign). It reports
  `ratio = 1.054`, i.e. *the opposite verdict*: that the difference is a
  marginal-distribution property and the "temporal" wording is misleading.
* `results/shuffle_ped2.txt` — the **corrected** version, which asserts that
  `d_orig` matches `backbone_robustness.txt` before it trusts the comparison.
  It reports `ratio = 0.026`, the number in the paper.

The superseded file is kept deliberately, as evidence of the bug and of the
fix; the paper is partly about exactly this failure mode. Do not cite it.

## 7. The reporting checklist

See `checklist/reporting_checklist.md`. It is the paper's Table (a minimum
reporting standard for VAD evaluation) in a form you can paste into a
submission or turn into a CI check.

## 8. Licence

MIT — see `LICENSE`. The datasets and the VadCLIP reference pipeline carry
their own licences and are not covered by it.
