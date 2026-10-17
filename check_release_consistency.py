#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check that this release does not contradict the manuscript.

Every claim this repository makes about the paper is also made in the paper, and
the two drift apart silently: the manuscript is edited by hand, the code and its
captured output are not. Three separate times during the wave-field work the
release said something the manuscript had already stopped saying:

  * README.md claimed twice that the wave-field rows "cannot be re-measured",
    which was true of an earlier draft and false of the current one;
  * complementarity/mechanism_rerun.py printed "the three wave-field rows cannot
    be re-measured at any seed count", and that false sentence was then frozen
    into results/mechanism_rerun.txt as shipped evidence;
  * the README described the wave-field re-measurement without the
    Holm-Bonferroni correction, i.e. it quoted only the two smallest p-values.

None of these would be caught by running the scripts -- they all ran clean and
exited 0. They are claims, not computations, so they need a check of their own.

usage:  python check_release_consistency.py
exits 0 if the release is consistent, 1 otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Directories whose contents are evidence, not claims: raw logs and the
# vendored reference pipeline must be searched for the literal strings anyway
# (a stale sentence printed into a log is still a claim), but they are not
# expected to follow the README's wording.
SKIP_PREFIXES = ('architecture/',)
LOG_PREFIXES = ('logs/',)

# (substring, why it is wrong now). Each is a sentence that was true once.
BANNED = [
    ('cannot be re-measured',
     'the wave-field rows are re-measured at n=4 over batch size, fusion and '
     'gain, plus four warm-start rows; the manuscript says so'),
    ('cannot re-measure at any',
     'same: the wave-field rows are re-measured at four seeds'),
    ('two unresolved candidate',
     'the factorial resolved them: batch size was the cause and fusion and '
     'gain do not matter'),
    ('not reproduced at its stated magnitude',
     'replaced by a specific claim -- a small cost at the harness default, with '
     'every published figure 2.5x to 7.5x larger than anything reproducible'),
]

# Statements that must be present somewhere in the release, because the
# manuscript now asserts them and a reader arriving at the code needs to find
# the correction rather than only the result.
REQUIRED = [
    ('0.0074',
     'the batch-96 wave-field row, the smallest uncorrected p in the table'),
    ('0.074',
     'the same row after Holm-Bonferroni over the ten re-measured rows'),
    ('0.203',
     'the batch-96 feature row after the same correction'),
]

failures = []
checked = 0


def text_files():
    try:
        out = subprocess.run(['git', 'ls-files'], cwd=HERE, check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL).stdout
        names = out.decode('utf-8', 'replace').splitlines()
    except Exception:
        names = []
        for base, _dirs, files in os.walk(HERE):
            for fn in files:
                rel = os.path.relpath(os.path.join(base, fn), HERE)
                rel = rel.replace('\\', '/')
                if not rel.startswith(('.git', SKIP_PREFIXES)):
                    names.append(rel)
    keep = []
    for n in names:
        if n.startswith(SKIP_PREFIXES):
            continue
        if not re.search(r'\.(md|py|txt|sh)$', n):
            continue
        keep.append(n)
    return keep


def note(ok, label, detail=''):
    tag = 'ok  ' if ok else 'FAIL'
    line = '  %s %s' % (tag, label)
    if detail:
        line += '   %s' % detail
    print(line)
    if not ok:
        failures.append(label)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    files = text_files()
    print('scanning %d text files under %s' % (len(files), HERE))
    print()

    blobs = {}
    for rel in files:
        p = os.path.join(HERE, rel)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding='utf-8', errors='replace') as fh:
                blobs[rel] = fh.read()
        except OSError:
            continue

    print('A. no stale claim survives anywhere in the release')
    for needle, why in BANNED:
        # A line that quotes the phrase in order to retract it is fine.
        offenders = []
        for rel, body in blobs.items():
            for m in re.finditer(re.escape(needle), body):
                lo = body.rfind('\n', 0, m.start()) + 1
                hi = body.find('\n', m.end())
                line = body[lo:hi if hi > 0 else len(body)]
                if re.search(r'\bwas false\b|\bthat was wrong\b|earlier (?:version|'
                             r'claim|draft)|said those rows|used to say|wrong and '
                             r'contradicted|does not claim', line, re.I):
                    continue
                offenders.append('%s:%d' % (rel, body.count('\n', 0, m.start()) + 1))
        note(not offenders, 'no live "%s"' % needle,
             '%s   -- %s' % (', '.join(offenders[:4]), why)
             if offenders else why)

    print()
    print('B. the manuscript\'s current correction is reachable from the release')
    whole = '\n'.join(blobs.values())
    for needle, why in REQUIRED:
        note(needle in whole, 'release carries %s' % needle, why)

    print()
    print('C. the results the manuscript quotes are actually derivable')
    # The factorial and warm reports must be shipped, and must name the two
    # batch-96 rows, or the reader cannot get from the table to the evidence.
    for rel in ('results/wavefield_factorial.txt', 'results/wavefield_warm.txt'):
        body = blobs.get(rel)
        note(body is not None, '%s is shipped' % rel,
             'the manuscript quotes it' if body else 'missing')
        if body is None:
            continue
        if 'factorial' in rel:
            note('0.0277' in body and '0.0085' in body,
                 '%s carries the batch-96 row' % rel)
        else:
            note('0.0927' in body and '0.3910' in body,
                 '%s carries the warm-row p values' % rel)

    print()
    print('C2. the raw logs behind those reports are shipped too')
    for d, n in (('logs/wavefield_factorial', 28), ('logs/wavefield_warm', 20)):
        p = os.path.join(HERE, d)
        cnt = len(os.listdir(p)) if os.path.isdir(p) else 0
        note(cnt == n, '%s holds %d logs' % (d, n),
             'found %d' % cnt)

    print()
    if failures:
        print('FAILED: %d of the release\'s claims contradict the manuscript'
              % len(failures))
        for f in failures:
            print('  - %s' % f)
        return 1
    print('PASS: the release does not contradict the manuscript')
    return 0


if __name__ == '__main__':
    sys.exit(main())