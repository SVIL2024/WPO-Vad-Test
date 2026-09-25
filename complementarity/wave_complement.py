"""DECISIVE TEST: does the wave energy carry information the trunk lacks?

WHY THIS EXISTS
The E0 probe says the wave energy separates anomalous frames with frame AUC
~0.71 (vs 0.66 for the |u_t| control) and that was read as a green light.
But the trained trunk's own frame AUC is 0.948. So "0.71 > 0.66" answers the
wrong question. The question that decides whether fusing can ever work is:

    GIVEN what the trunk already knows, does the wave energy still tell us
    something new?

Tests, in increasing order of power:
  1. headroom    - how far apart are the two detectors?
  2. correlation - high corr => the wave re-encodes the trunk
  3. oracle      - is there ANY fusion weight where A0 + w*wave beats A0?
  4. RESIDUAL    - after removing everything the trunk score explains, does
                   the leftover still separate anomaly from normal?
                   *** this is the verdict - see note below ***

WHY THE RESIDUAL TEST IS THE VERDICT (and the others are only context)
  - Oracle fusion has LOW POWER here: AP is a global metric, so when a strong
    detector dominates, a weak signal that genuinely helps on a subset gets
    swamped. It can report "redundant" for a signal that is actually useful.
  - Plain within-bin AUC is CONFOUNDED: inside a bin the trunk score still
    varies, that residual variation still correlates with the label, so ANY
    signal that merely tracks the trunk lights up. Residualising first
    (subtracting E[wave|trunk] within fine bins) removes that artefact.
  - An effect-size cut-off ("AUC > 0.55") is also wrong: a strong trunk absorbs
    most of the label signal, so even a real independent detector leaves only a
    small residual AUC. We therefore test SIGNIFICANCE against the null 0.5,
    not magnitude.

READING THE RESULT
  |z| < 3  -> REDUNDANT. The wave is a function of what the trunk already
      computes. No fusion change can fix this: the problem is the SUBSTRATE.
      Options: (a) different substrate (not raw CLIP features), (b) different
      role (regulariser / prior, not an evidence channel), (c) stop fusing and
      report the wave as a standalone unsupervised detector.
  z > 3    -> COMPLEMENTARY. The signal is real and the fusion is throwing it
      away. Fix the FUSION, not the operator.

INPUTS (both local, no GPU needed):
  logs_fetch/scores_xd_base_s234.npz  - A0 control, per-segment probs + gt
  logs_fetch/wave_energy_test.npz     - probe --save-frames output

Run from repo root:  python complementarity/wave_complement.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.environ.get('XD_PAIR_DIR', os.path.join(REPO, 'logs_fetch'))
REPEAT = 16


# --------------------------------------------------------------- the machinery
def _residual(a0, wave, nb=50):
    """wave minus E[wave | trunk score], estimated by fine quantile bins."""
    qs = np.quantile(a0, np.linspace(0, 1, nb + 1))
    resid = np.empty(len(wave), dtype=np.float64)
    for b in range(nb):
        m = ((a0 >= qs[b]) & (a0 < qs[b + 1])) if b < nb - 1 else \
            ((a0 >= qs[b]) & (a0 <= qs[b + 1]))
        resid[m] = wave[m] - wave[m].mean() if m.sum() >= 5 else 0.0
    return resid


def conditional_auc(a0, wave, y, nb=50):
    """AUC of the wave after the trunk's contribution is removed."""
    r = _residual(a0, wave, nb=nb)
    if np.std(r) < 1e-12:
        return 0.5
    return roc_auc_score(y, r)


def auc_se(y, s):
    """Hanley-McNeil standard error of an AUC (closed form)."""
    a = roc_auc_score(y, s)
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 < 2 or n0 < 2:
        return a, float('nan')
    q1 = a / (2.0 - a) if a < 2.0 else 0.0
    q2 = 2.0 * a * a / (1.0 + a)
    var = (a * (1.0 - a) + (n1 - 1.0) * (q1 - a * a)
           + (n0 - 1.0) * (q2 - a * a)) / (n1 * n0)
    return a, float(np.sqrt(max(var, 0.0)))


def conditional_z(a0, wave, y, nb=50):
    """(residual AUC, z-score vs the null 0.5)."""
    a = conditional_auc(a0, wave, y, nb=nb)
    _, se = auc_se(y, _residual(a0, wave, nb=nb))
    if not np.isfinite(se) or se < 1e-12:
        return a, 0.0
    return a, (a - 0.5) / se


# ------------------------------------------------------------------ data loading
def _flat(obj_arr):
    return np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float64))
                           for x in obj_arr])


def load_pair():
    # ALWAYS prefer the pair that _diag_align_check.py has already proven.
    # The raw probe dump is NOT safe to use directly: the probe skips videos
    # with T<8 without advancing its gt cursor, so its own labels are shifted
    # after the first skip (98% agreement here - looks fine, is not). Only use
    # the raw dump if someone has not run the alignment gate.
    aligned = os.path.join(LOGS, 'aligned_pair.npz')
    if os.path.isfile(aligned):
        d = np.load(aligned)
        print('using ALIGNED pair from _diag_align_check.py: %s' % aligned)
        return d['a0'].astype(np.float64), d['wave'].astype(np.float64), \
            d['y'].astype(int)

    print('WARNING: aligned_pair.npz absent - falling back to raw dumps.')
    print('  Run  python _diag_align_check.py (not in this release)  FIRST to prove alignment.')
    a0_path = os.path.join(LOGS, 'scores_xd_base_s234.npz')
    wv_path = os.path.join(LOGS, 'wave_energy_test.npz')
    missing = [p for p in (a0_path, wv_path) if not os.path.isfile(p)]
    if missing:
        print('MISSING:')
        for m in missing:
            print('  ' + m)
        print('\nProduce the wave dump on the server (CPU, ~5 min) with:')
        print('  cd src && python _probe_wave_field.py \\')
        print('      --csv ../list/xd_CLIP_rgbtest.csv --gt ../list/gt.npy \\')
        print('      --n 800 --modes 192 --alpha 0.1 --c 0.5 \\')
        print('      --save-frames ../wave_energy_test.npz')
        return None, None, None

    d0 = np.load(a0_path, allow_pickle=True)
    a0 = _flat(d0['probs'])
    dw = np.load(wv_path, allow_pickle=True)
    key = next((c for c in ('e_wave_c0', 'e_wave', 'e_waveD', 'e_ut')
                if c in dw.files), None)
    wave = _flat(dw[key])
    label = _flat(dw['label'])

    gt_raw = d0['gt'].ravel()
    n = min(len(a0), len(wave), len(label))
    nseg = len(a0[:n])
    gt_seg = gt_raw[:nseg * REPEAT].reshape(nseg, REPEAT).max(axis=1)
    return a0[:n], wave[:n], gt_seg


# ------------------------------------------------------------------------- main
def main():
    a0, wave, gt = load_pair()
    if a0 is None:
        return 1
    y = gt.astype(int)
    print('segments=%d  positives=%d (%.1f%%)' % (len(y), y.sum(), 100 * y.mean()))

    print('\n=== 1. Headroom ===')
    a_auc = roc_auc_score(y, a0)
    w_auc = roc_auc_score(y, wave)
    print('  A0 trunk : frame AUC=%.4f  AP=%.4f' % (a_auc, average_precision_score(y, a0)))
    print('  wave     : frame AUC=%.4f  AP=%.4f' % (w_auc, average_precision_score(y, wave)))
    print('  -> the trunk is %.3f AUC above the wave.' % (a_auc - w_auc))

    print('\n=== 2. Correlation ===')
    print('  pearson corr(a0, wave) = %.4f' % np.corrcoef(a0, wave)[0, 1])

    print('\n=== 3. Oracle fusion (context only - LOW POWER, see docstring) ===')
    base_ap = average_precision_score(y, a0)
    wr = np.argsort(np.argsort(wave)).astype(np.float64) / max(len(wave) - 1, 1)
    ar = np.argsort(np.argsort(a0)).astype(np.float64) / max(len(a0) - 1, 1)
    best = base_ap
    for w in np.arange(-1.0, 1.01, 0.1):
        best = max(best, average_precision_score(y, a0 + w * wr))
    for w in np.arange(0.0, 1.01, 0.1):
        best = max(best, average_precision_score(y, (1 - w) * ar + w * wr))
    print('  best AP=%.6f vs A0 %.6f  (delta %+.6f)' % (best, base_ap, best - base_ap))

    print('\n=== 4. RESIDUAL TEST (the verdict) ===')
    print('  residual = wave - E[wave | trunk score], then AUC vs the label.')
    ra, z = conditional_z(a0, wave, y)
    print('  residual AUC = %.4f   z = %+.2f' % (ra, z))
    for nb in (20, 50, 100):
        a2, z2 = conditional_z(a0, wave, y, nb=nb)
        print('    (bins=%3d: AUC=%.4f z=%+.2f)' % (nb, a2, z2))

    print('\n' + '=' * 64)
    if abs(z) < 3.0:
        print('VERDICT: REDUNDANT  (|z|=%.2f < 3)' % abs(z))
        print('  The wave energy is a function of what the trunk already computes.')
        print('  No fusion change can fix this - the SUBSTRATE is the problem.')
        print('  -> try a different substrate, a different role (regulariser),')
        print('     or report the wave as a standalone unsupervised detector.')
    else:
        print('VERDICT: COMPLEMENTARY  (z=%.2f)' % z)
        print('  The wave carries real information the trunk lacks; the FUSION')
        print('  is throwing it away. Fix the fusion, not the operator.')
    print('=' * 64)
    return 0


if __name__ == '__main__':
    sys.exit(main())
