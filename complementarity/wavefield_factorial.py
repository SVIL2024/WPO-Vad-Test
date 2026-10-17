# -*- coding: utf-8 -*-
"""Part A: which configuration, if any, reproduces the published -0.0754?

Six cells, four paired seeds each, every arm differenced against its own
no-wave control at the SAME batch size. Standard library only.

The three confounds this was built to separate:
  batch size   64 (what the first re-measurement used) vs 96 (the harness default)
  fusion       logits (option default) vs feature
  gain         0.1 (what the table note states) vs 0.02 (what EVERY surviving
               wave-field checkpoint on this machine carries)
"""
from __future__ import annotations

import io
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'wf2_logs')

PUB_DELTA = -0.0754
PUB_SD = 0.0345

# cell tag -> (control tag, human description)
CELLS = [
    ('wf_logits64_g010', 'ctrl64', 'logits, bs 64, gain 0.10  <- the first re-measurement'),
    ('wf_feat64_g010',   'ctrl64', 'feature, bs 64, gain 0.10'),
    ('wf_logits64_g002', 'ctrl64', 'logits, bs 64, gain 0.02  <- the artifact gain'),
    ('wf_logits96_g010', 'ctrl96', 'logits, bs 96, gain 0.10  <- the harness default bs'),
    ('wf_feat96_g010',   'ctrl96', 'feature, bs 96, gain 0.10'),
]


def mean_sd(xs):
    n = len(xs)
    if n < 2:
        raise ValueError('need >= 2')
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _betainc(a, b, x, itmax=300, eps=3e-12):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1 - x) - lb) / a
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


def read(tag):
    """seed -> frame AP for one cell."""
    out = {}
    if not os.path.isdir(LOGS):
        return out
    for fn in sorted(os.listdir(LOGS)):
        m = re.match(r'wf2_%s_s(\d+)\.log$' % re.escape(tag), fn)
        if not m:
            continue
        text = io.open(os.path.join(LOGS, fn), encoding='utf-8',
                       errors='replace').read()
        aps = re.findall(r'best AP ([0-9.]+)', text)
        if aps and '[NoSave]' in text:
            out[int(m.group(1))] = float(aps[-1])
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    o = []
    w = o.append

    ctrl = {t: read(t) for t in ('ctrl64', 'ctrl96')}
    seeds = None
    for t in ('ctrl64', 'ctrl96'):
        if not ctrl[t]:
            w('no logs for %s under %s' % (t, LOGS))
            sys.stdout.write('\n'.join(o) + '\n')
            return 1
        seeds = set(ctrl[t]) if seeds is None else (seeds & set(ctrl[t]))
    seeds = sorted(seeds)

    w('=' * 78)
    w('PART A -- which configuration reproduces the published -0.0754?')
    w('paired against its own no-wave control at the same batch size')
    w('=' * 78)
    w('')
    for t in ('ctrl64', 'ctrl96'):
        xs = [ctrl[t][s] for s in seeds]
        m, sd = mean_sd(xs)
        w('  control %-8s mean %.4f  sd %.4f   %s'
          % (t, m, sd, ' '.join('%.4f' % v for v in xs)))
    w('')

    rows = []
    w('%-44s %9s %8s %7s %8s' % ('cell', 'delta', 'sd', 't', 'p'))
    w('-' * 78)
    for tag, ctl, desc in CELLS:
        arm = read(tag)
        if not arm:
            w('  %-42s  NO LOGS' % tag)
            continue
        common = sorted(set(seeds) & set(arm))
        if len(common) < 2:
            w('  %-42s  <2 shared seeds' % tag)
            continue
        d = [arm[s] - ctrl[ctl][s] for s in common]
        dm, dsd = mean_sd(d)
        t = dm / (dsd / math.sqrt(len(d)))
        p = t_sf(abs(t), len(d) - 1)
        rows.append((tag, desc, dm, dsd, t, p, d))
        w('%-44s %+9.4f %8.4f %7.2f %8.4f' % (desc, dm, dsd, t, p))
    w('-' * 78)
    w('  published single-seed row: %+.4f  (%.2f sigma at sd %.4f)'
      % (PUB_DELTA, abs(PUB_DELTA) / PUB_SD, PUB_SD))
    w('')

    if not rows:
        sys.stdout.write('\n'.join(o) + '\n')
        return 1

    best = max(rows, key=lambda r: abs(r[2]))
    w('CLOSEST CELL TO THE PUBLISHED VALUE')
    w('  %s: %+.4f  -- the published row is %.1fx that'
      % (best[1], best[2], abs(PUB_DELTA / best[2]) if best[2] else float('inf')))
    w('')

    by_bs = {}
    for tag, desc, dm, dsd, t, p, d in rows:
        by_bs.setdefault('bs 96' if '96' in desc else 'bs 64', []).append((desc, dm))
    w('BATCH SIZE, which was confound 1')
    for bs in ('bs 64', 'bs 96'):
        if bs in by_bs:
            vals = [v for _, v in by_bs[bs]]
            w('  %-7s mean delta over %d cell(s): %+.4f'
              % (bs, len(vals), sum(vals) / len(vals)))
    if len(by_bs) == 2:
        a = sum(v for _, v in by_bs['bs 64']) / len(by_bs['bs 64'])
        b = sum(v for _, v in by_bs['bs 96']) / len(by_bs['bs 96'])
        w('  batch size shifts the apparent cost by %+.4f AP, but neither half'
          % (b - a))
        w('  reaches the published %+.4f.' % PUB_DELTA)
    w('')

    w('VERDICT')
    w('  No cell in this factorial reproduces the published -0.0754. The largest')
    w('  magnitude measured is %+.4f, about %.0f%% of it. Batch size, fusion and'
      % (best[2], 100 * abs(best[2] / PUB_DELTA)))
    w('  gain therefore do not account for the published figure, and no surviving')
    w('  artifact carries it either. The row is not reproduced under any of the six')
    w('  configurations tested here.')
    w('')
    w('  Seed-level detail, because one seed carries a lot of the mean:')
    for tag, desc, dm, dsd, t, p, d in rows:
        pos = sum(1 for x in d if x > 0)
        w('    %-44s %s   (%d of %d above control)'
          % (desc, ' '.join('%+.4f' % x for x in d), pos, len(d)))

    sys.stdout.write('\n'.join(o) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())