#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single source of truth for the seed statistics and for log parsing.

Every script in this repository that reports a mean, a standard deviation, a
paired *t* test, a Holm--Bonferroni corrected p-value, or a value read out of a
training log goes through this file.  That is deliberate: these quantities are
quoted in more than one table of the paper, and the failure mode we are guarding
against is two tables computing the same number by different routes and drifting
apart.  If you change how a statistic is computed, change it here.

Implemented on the standard library alone, with no numpy or scipy, so that
"the repository contains no third-party code" stays true and the release runs on
a bare interpreter.
"""

from __future__ import annotations

import io
import math
import os
import re

# The paper rounds every reported statistic to four decimals, so a reproduction
# within half of the last printed digit counts as agreement.
TOL = 5e-4


# ------------------------------------------------------------------ descriptives

def mean_sd(xs):
    """Sample mean and sample standard deviation (ddof = 1).

    Refuses n < 2 rather than returning a zero sd, which would silently turn a
    paired test into a division by zero further downstream.
    """
    n = len(xs)
    if n < 2:
        raise ValueError("need at least 2 values, got %d" % n)
    m = sum(xs) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def close(got, want, tol=TOL):
    """True if ``got`` reproduces ``want`` to the precision the paper prints."""
    if want != want:          # NaN expected -> nothing to compare
        return True
    return abs(got - want) <= tol


# ------------------------------------------------------------------ t distribution

def _betainc(a, b, x, itmax=300, eps=3e-12):
    """Regularised incomplete beta I_x(a, b), Lentz continued fraction."""
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
    """Two-sided tail probability of Student's t on ``nu`` degrees of freedom."""
    return _betainc(nu / 2.0, 0.5, nu / (nu + t * t))


def paired(ctrl, arm):
    """Same-seed paired contrast ``arm - ctrl``.

    ``ctrl`` and ``arm`` are ``{seed: value}`` mappings.  Same-seed differencing
    is the point: the seed-to-seed sd dominates the unpaired sd, and pairing
    cancels it.  The paper never compares arms across different seed sets.

    Returns ``(delta, sd, t, p, n, seeds)``.
    """
    seeds = sorted(set(ctrl) & set(arm))
    if len(seeds) < 2:
        raise ValueError("need >= 2 shared seeds, got %s" % seeds)
    diffs = [arm[s] - ctrl[s] for s in seeds]
    delta, sd = mean_sd(diffs)
    t = delta / (sd / math.sqrt(len(diffs)))
    p = t_sf(abs(t), len(diffs) - 1)
    return delta, sd, t, p, len(diffs), seeds


def holm(ps):
    """Holm--Bonferroni step-down adjusted p-values, input order preserved."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out = [0.0] * len(ps)
    running, m = 0.0, len(ps)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out


# ------------------------------------------------------------------ training logs

_RE_BEST = re.compile(r"BEST AP:([\d.]+)\(e(\d+)\)")
_RE_AUC = re.compile(r"AUC:([\d.]+)")
_RE_CFG = re.compile(r"multi_scale=(scales=[\d,]+(?:\([^)]*\))?|bands=[^,]+|off)")


def read_run(path):
    """Return ``(best_ap, auc, config_fragment)`` from one training log.

    The AP column comes from ``BEST AP`` rather than from the trailing ``AP:`` on
    the epoch line.  Those coincide at a one-epoch budget, but only ``BEST AP``
    stays correct if the budget is ever raised, so both the eight-seed arm run and
    the later re-measurements are read on one definition.
    """
    best, auc, cfg = None, None, None
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _RE_BEST.search(line)
            if m:
                best = float(m.group(1))
            m = _RE_AUC.search(line)
            if m:
                auc = float(m.group(1))
            if cfg is None:
                m = _RE_CFG.search(line)
                if m:
                    cfg = m.group(1)
    return best, auc, cfg


def read_dir(directory, pattern):
    """Map ``seed -> (best_ap, auc, config)`` for every log matching ``pattern``."""
    out = {}
    if not os.path.isdir(directory):
        return out
    rx = re.compile(pattern)
    for name in sorted(os.listdir(directory)):
        m = rx.match(name)
        if not m:
            continue
        best, auc, cfg = read_run(os.path.join(directory, name))
        if best is not None:
            out[int(m.group(1))] = (best, auc, cfg)
    return out


def logs_dir(here=None):
    """The repository's ``logs/`` directory.

    ``here`` defaults to the directory holding this module, which is the
    repository root, so the path is ``<repo>/logs``.
    """
    root = here or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(root, "logs")