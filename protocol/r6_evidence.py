# -*- coding: utf-8 -*-
"""R6 (peer review): pin every claim about the reference codebase to a line.

The paper accuses a published reference implementation of two defects.  An
accusation like that is only checkable if it comes with file and line, so this
script locates each cited line in the source and prints it.  It asserts that
the expected text is found, so it fails loudly if the upstream code moves or
if the checkout is a different revision.

Read-only: nothing here writes to the reference tree.

Writes results/r6_evidence.txt
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.environ.get("VADCLIP_SRC", r"D:\program\VadCLIP-main\src")
OUT = os.environ.get("R6_OUT",
                     os.path.join(REPO, "results", "r6_evidence.txt"))

# (relative path, 1-based line, substring that must be on that line, why)
CLAIMS = [
    ("xd_test.py", 85, "return ROC1, AP2",
     "XD returns AUC from column 1 but AP from column 2"),
    ("ucf_test.py", 108, "return ROC1, AP1",
     "UCF returns AUC and AP from column 1"),
    ("ucf_train.py", 143, "= test(",
     "UCF unpacks the tuple positionally"),
    ("ucf_train.py", 144, "AP = AUC",
     "UCF then overwrites AP with AUC outright"),
    ("xd_train.py", 99, "= test(",
     "XD unpacks the same tuple positionally"),
    ("utils/dataset.py", 27, "process_feat",
     "training path resamples the clip"),
    ("utils/dataset.py", 29, "process_split",
     "test path takes native-rate windows"),
    ("utils/tools.py", 60, "def uniform_extract",
     "the resampler the training path uses"),
    ("utils/tools.py", 62, "np.linspace(0, len(feat), t_max+1",
     "compresses the whole clip into t_max frames"),
    ("utils/tools.py", 82, "def process_feat",
     "training entry point"),
    ("utils/tools.py", 92, "def process_split",
     "test entry point"),
    ("xd_option.py", 7, "visual-length",
     "default window length, XD"),
    ("ucf_option.py", 7, "visual-length",
     "default window length, UCF"),
]

lines = []


def emit(s=""):
    print(s)
    lines.append(s)


def read(rel):
    p = os.path.join(SRC, rel)
    with io.open(p, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read().splitlines()


def main():
    emit("=" * 74)
    emit("R6 -- every claim about the reference implementation, with its line")
    emit("=" * 74)
    emit("  source tree: %s" % SRC)
    emit("")
    bad = 0
    for rel, ln, want, why in CLAIMS:
        try:
            body = read(rel)
        except IOError:
            emit("  MISSING  %s" % rel)
            bad += 1
            continue
        got = body[ln - 1] if 0 <= ln - 1 < len(body) else "<no such line>"
        ok = want in got
        if not ok:
            bad += 1
        emit("  [%s] %s:%d" % ("ok " if ok else "BAD", rel, ln))
        emit("        %s" % got.strip())
        emit("        -> %s" % why)
    emit("")
    if bad:
        emit("  %d claim(s) did NOT verify -- do not put these lines in the paper"
             % bad)
    else:
        emit("  all %d claims verified against the source tree" % len(CLAIMS))

    with io.open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s" % OUT)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
