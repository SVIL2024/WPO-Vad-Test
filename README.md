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

Everything below is measurement code. There is no new detector here, and
nothing in this repository needs a GPU to reproduce the premise results —
they run on frozen, pre-extracted CLIP features.

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

## 2. What is *not* in here

* **The VadCLIP reference pipeline.** It is a third-party codebase with its own
  licence; we reference it, we do not redistribute it. Two things here read it:
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

`figures/figs_ped2.py` renders two further Ped2 diagnostics (the two-view
spectrum and the injection calibration) that the paper does not print; it is
here because it is the same instrument on the same data.

All five figure scripts write **vector PDF** — their internal `save()` helpers
rewrite the extension, so the `.png`-looking call sites still produce
`F*.pdf`, which is what the manuscript includes.

## 6. Read this before you cite `results/shuffle.txt`

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
