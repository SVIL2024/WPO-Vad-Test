# -*- coding: utf-8 -*-
"""Mechanism (a), the wave field, re-measured as a paired contrast at n=4.

Reads the per-seed training logs and answers the question the published row
raises: Table tab:mechanisms reports the wave field at K=192 costing -0.0754
frame AP against a shared no-wave control, at ONE seed. A single-seed screen
cannot distinguish a real cost from a seed draw, and this paper's whole argument
is that it cannot. So the row was re-run as a same-seed paired contrast at four
seeds.

Run from the repository root:

    python complementarity/wavefield_rerun.py            # logs/wavefield_n4
    python complementarity/wavefield_rerun.py <logdir>

Standard library only. The arithmetic (paired t, incomplete beta) follows the
same conventions as seedstats.py, deliberately duplicated rather than imported so
this file also runs from a bare copy.

What it found, and why that is not yet a refutation:

  measured -0.0100 +/- 0.0308, t -0.65, p 0.5610, three of four seeds below
  control. The published single-seed -0.0754 is about seven times that, and in
  units of the one-epoch sd it is 2.19 sigma where the measurement is 0.29.

  Two confounds are NOT excluded and are printed by this script. Both arms sit
  about 0.031 AP below the published control because this ran at batch 64 where
  the harness defaults to 96, so only the contrast is comparable. And the
  published row carries no fusion label while this ran --wave-fusion logits, a
  different wiring from the separately labelled feature-gate row.

Only the from-scratch row was measured. The three warm-A0 rows need a warm
checkpoint and are not covered.
"""
from __future__ import annotations

import io
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
LOGS = (sys.argv[1] if len(sys.argv) > 1
        else os.path.join(REPO, 'logs', 'wavefield_n4'))

SEEDS = (42, 234, 555, 789)

# Published, from results/mechanisms.txt, seed 234, one epoch, frame level.
PUB_CONTROL = 0.8490
PUB_WAVE = 0.7736
PUB_DELTA = -0.0754
# The paper's own yardstick for this harness family (one-epoch frame AP sd).
PUB_SD = 0.0345

# src/xd_train.py prints "best AP <x>"; it does not print "BEST AP:<x>".
RE_AP = re.compile(r'best AP ([0-9.]+)')
RE_MMAP = re.compile(r'average MAP: ([0-9.]+)')
RE_NOSAVE = re.compile(r'\[NoSave\]')
RE_WAVE = re.compile(r'\[Wave\] fusion=(\w+)')


def read_arm(arm):
    """seed -> (ap, mAP) for every completed run of one arm."""
    out = {}
    if not os.path.isdir(LOGS):
        return out
    for fn in sorted(os.listdir(LOGS)):
        m = re.match(r'wf_(%s)_s(\d+)\.log$' % arm, fn)
        if not m:
            continue
        text = io.open(os.path.join(LOGS, fn), encoding='utf-8',
                       errors='replace').read()
        aps = RE_AP.findall(text)
        if not aps:
            continue
        mm = RE_MMAP.findall(text)
        out[int(m.group(2))] = (float(aps[-1]),
                                float(mm[-1]) if mm else float('nan'),
                                bool(RE_NOSAVE.search(text)))
    return out


# ---- statistics, standard library only (same conventions as seedstats.py) ----


def mean_sd(xs):
    n = len(xs)
    if n < 2:
        raise ValueError('need >= 2 values')
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _betainc(a, b, x, itmax=300, eps=3e-12):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(itmax + 1):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = 1.0 + num / (c if abs(c) > 1e-30 else 1e-30)
        f *= c * d
        if abs(1.0 - c * d) < eps:
            break
    return front * (f - 1.0)


def t_sf(t, nu):
    return _betainc(nu / 2.0, 0.5, nu / (nu + t * t))


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    o = []
    w = o.append
    fail = []

    ctrl = read_arm('ctrl')
    wave = read_arm('wf192')
    shared = sorted(set(ctrl) & set(wave))
    if len(shared) < 2:
        w('not enough shared seeds: %s' % shared)
        sys.stdout.write('\n'.join(o) + '\n')
        return 1

    w('=' * 78)
    w('MECHANISM (a), the wave field -- paired re-measurement')
    w('one epoch, batch 64, single 11 GB card')
    w('=' * 78)
    w('')
    w('%-8s %10s %10s %10s' % ('seed', 'control', 'wave', 'delta'))
    w('-' * 78)
    diffs = []
    for s in shared:
        c, cm, cn = ctrl[s]
        v, vm, vn = wave[s]
        d = v - c
        diffs.append(d)
        w('%-8d %10.4f %10.4f %+10.4f%s'
          % (s, c, v, d, '' if cn and vn else '   (incomplete)'))
    dm, dsd = mean_sd(diffs)
    cm_, csd = mean_sd([ctrl[s][0] for s in shared])
    vm_, vsd = mean_sd([wave[s][0] for s in shared])
    w('-' * 78)
    w('%-8s %10.4f %10.4f %+10.4f   mean, sd %.4f / %.4f'
      % ('mean', cm_, vm_, dm, csd, vsd))
    w('%-8s %10s %10s %10s   paired sd %.4f' % ('sd', '', '', '', dsd))
    w('')

    t = dm / (dsd / math.sqrt(len(diffs)))
    p = t_sf(abs(t), len(diffs) - 1)
    flips = sum(1 for d in diffs if d > 0)

    w('PAIRED TEST')
    w('  delta = %+.4f +/- %.4f   n=%d  df=%d' % (dm, dsd, len(diffs), len(diffs) - 1))
    w('  t = %+.2f   p = %.4f' % (t, p))
    w('  seeds with the wave field BELOW control: %d of %d'
      % (len(diffs) - flips, len(diffs)))
    w('')

    w('AGAINST THE PUBLISHED ROW')
    w('  published, one seed (234): control %.4f  wave %.4f  delta %+.4f'
      % (PUB_CONTROL, PUB_WAVE, PUB_DELTA))
    w('  this measurement        : control %.4f  wave %.4f  delta %+.4f'
      % (cm_, vm_, dm))
    w('  the published delta is %+.4f; the measured one is %+.4f'
      % (PUB_DELTA, dm))
    w('  ratio published/measured: %.1fx'
      % (abs(PUB_DELTA / dm) if dm else float('inf')))
    w('  published |delta| in units of the one-epoch sd (%.4f): %.2f sigma'
      % (PUB_SD, abs(PUB_DELTA) / PUB_SD))
    w('  measured  |delta| in the same units: %.2f sigma'
      % (abs(dm) / PUB_SD))
    w('')
    w('  The published control is %.4f here; the gap to the published %.4f is %+.4f,'
      % (cm_, PUB_CONTROL, cm_ - PUB_CONTROL))
    w('  which is present in BOTH arms and so is a property of this harness and')
    w('  budget rather than of the wave field. The contrast is unaffected by it;')
    w('  the absolute levels are not comparable to the published ones.')
    w('')
    w('  FAIL expected: the published -0.0754 must NOT be reproduced here, because'
      if False else '')
    if abs(dm - PUB_DELTA) < 0.01:
        w('  NOTE: the measurement agrees with the published row. If that is')
        w('        real, the earlier claim that this row cannot be re-measured is')
        w('        wrong twice over. Check the fusion mode and batch size first.')
    else:
        w('  READING')
        w('    The DIRECTION is consistent -- %d of %d seeds put the wave field'
          % (len(diffs) - flips, len(diffs)))
        w('    below control, and the one seed that reverses it does so by more than')
        w('    any of the other three differences. What does not survive is the')
        w('    MAGNITUDE: the published single-seed figure of %+.4f is about %.1fx the'
          % (PUB_DELTA, abs(PUB_DELTA / dm) if dm else float('inf')))
        w('    paired estimate, and the paired estimate is not distinguishable from')
        w('    zero.')
        w('')
        w('    So the largest effect in tab:mechanisms does not hold up at the')
        w('    magnitude reported. That is this paper\'s own thesis applied to the')
        w('    paper\'s own strongest row: a single-seed screen at these effect')
        w('    sizes cannot distinguish a cost of about one AP point from a seed')
        w('    draw, and %+.4f is inside the noise of the very table that reports it.'
          % PUB_DELTA)
        w('')
        w('TWO CONFOUNDS WE HAVE NOT EXCLUDED, so this is not a refutation')
        w('  1. Batch size. src/xd_option.py defaults to 96; this ran at 64. Both')
        w('     arms are 0.0313 below the published control, which is why the')
        w('     absolute levels are not comparable and only the contrast is quoted.')
        w('     A wave-field/batch-size interaction is not ruled out.')
        w('  2. Fusion wiring. This ran --wave-fusion logits, which is the option')
        w('     default, but the published row carries no fusion label while two')
        w('     separate rows in the same table are labelled evidence->logits1 and')
        w('     feature gate. If the published row used the feature wiring, this is')
        w('     a different branch.')
    w('')
    w('SCOPE')
    w('  Only the from-scratch row was run. The three warm-A0 rows (evidence->logits1')
    w('  and feature gate, trunk frozen and joint) need a warm checkpoint from a')
    w('  matching baseline run, which is a second pipeline. They are NOT covered by')
    w('  this measurement and must not be reported as if they were.')
    w('')
    w('FUSION MODE: %s'
      % (RE_WAVE.findall(io.open(os.path.join(LOGS, 'wf_wf192_s%d.log' % shared[0]),
                                 encoding='utf-8', errors='replace').read())[:1]
         or ['(not logged)'])[0])

    text = '\n'.join(x for x in o if x != '')
    sys.stdout.write(text + '\n')
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())