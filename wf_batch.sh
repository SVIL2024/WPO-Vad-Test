#!/bin/bash
# ============================================================================
# Mechanism (a) -- the wave field -- re-measured as a paired contrast.
#
# Why this batch exists.  An earlier limitation note claimed the wave-field rows
# of Table 7 could not be re-measured at any seed count, on the grounds that the
# wave-field module and the multi-scale branches live in different source trees.
# That reasoning conflated two different things.  The true statement is that the
# two branches cannot coexist inside a single training run.  Each is runnable on
# its own, and the wave field is wired into src/xd_train.py through
# --wave-field / --wave-field-gain / --wave-fusion.  So the rows CAN be
# re-measured, and this batch does it.
#
# The published rows (results/mechanisms.txt, seed 234, one epoch, frame level):
#
#   (control) no-wave baseline        fAP 0.8490
#   (a) wave field K=192, gain 0.1    fAP 0.7736   delta -0.0754   from scratch
#   (a) evidence->logits1, trunk frozen fAP 0.8491  delta +0.0001   warm A0
#   (a) evidence->logits1, joint       fAP 0.7834  delta -0.0656   warm A0
#   (a) feature gate,     trunk frozen fAP 0.8500  delta +0.0010   warm A0
#   (a) feature gate,     joint       fAP 0.7852  delta -0.0638   warm A0
#
# Only the from-scratch row is run here.  The three warm-A0 rows need a warm
# checkpoint from a matching baseline run, which is a second pipeline; doing the
# from-scratch row first is enough to settle whether the -0.0754 survives, and it
# needs no checkpoint at all.
#
#   Phase 0  replication gate   2 runs   seed 234 only.
#                                   Hard gate: the measured delta must land near
#                                   the published -0.0754. If it does not, the
#                                   harness here is not the harness that produced
#                                   the table, and n=4 would be measuring a
#                                   different thing.
#   Phase 1  paired n=4          8 runs   seeds 42 234 555 789, control and
#                                   wave-field paired within each seed.
#
# Same one-epoch budget the screen used, so the comparison is like-for-like.
# Same four seeds as the (b)/(c) re-measurement, so all three share a control set.
#
# Ground rules, all learned on earlier boxes:
#   * CWD must be src/ (imports are top-level).
#   * python is NOT on PATH in a non-interactive shell -> $PY.
#   * --save-min-ap 0.999 => no checkpoint is ever written. Disk is the scarce
#     resource on a billed instance; we only need the log lines.
#   * Resumable: a run whose log already carries [EPOCH] is skipped.
#   * Stop on first failure rather than retrying forever.
#   * With WITH_N=8 set, phase 1 runs the full eight seeds instead of four.
# ============================================================================
set -u
cd /root/autodl-tmp/program/waveClipVad/src || exit 1
PY=/root/miniconda3/bin/python

LOGDIR=/root/autodl-tmp/logs
mkdir -p "$LOGDIR"
SUM="$LOGDIR/wf_summary.txt"
[ -s "$SUM" ] || : > "$SUM"

SEEDS4="42 234 555 789"
SEEDS8="42 234 555 789 1234 2024 3131 999999"
if [ "${WITH_N:-0}" = "1" ]; then SEEDS="$SEEDS8"; else SEEDS="$SEEDS4"; fi

# --- COMMON, from src/xd_option.py defaults plus the batch and budget we use ---
# --save-min-ap 0.999 is what makes this write no checkpoints at all.
# NOTE: no --eval-steps here. That flag exists in src_sgl but NOT in src/xd_option.py,
# so copying it over from the previous batch would make argparse reject every run.
#
# Every path default in src/xd_option.py is written for CWD = repo root
# ('list/xd_CLIP_rgb.csv', 'model/model_xd.pth', ...), but this harness must be run
# from src/ because its imports are top-level. The two are inconsistent in the
# source tree itself; we resolve it by passing each path explicitly as ../<path>
# rather than by editing a default, which would change behaviour for every other
# caller. The data rows inside the CSVs are relative too, and they resolve to the
# same /root/autodl-tmp/dataset/ from src/ as they do from src_sgl/ -- verified,
# 800 test rows, 0 missing.
COMMON="--batch-size 64 --max-epoch 1 --save-min-ap 0.999 \
--train-list ../list/xd_CLIP_rgb.csv \
--test-list ../list/xd_CLIP_rgbtest.csv \
--gt-path ../list/gt.npy \
--gt-segment-path ../list/gt_segment.npy \
--gt-label-path ../list/gt_label.npy \
--model-path model/model_xd.pth"

# The two arms differ ONLY in --wave-field. gain 0.1 is the published value.
WAVE="--wave-field 192 --wave-field-gain 0.1"

run () {
    local tag="$1"; shift
    local log="$LOGDIR/wf_${tag}.log"
    # src/xd_train.py does NOT print an [EPOCH] line and does NOT print "BEST AP:<x>".
    # Verified against a real log: the completion marker is [NoSave] and the AP is
    # "best AP <x>". Using the src_sgl formats here silently produced bestAP=NA on
    # every run and made the resume check a no-op, so both are pinned to this
    # tree's actual output.
    if grep -q '\[NoSave\]' "$log" 2>/dev/null; then
        echo "[skip] $tag" | tee -a "$SUM"; return 0
    fi
    local t0=$(date +%s)
    echo "[run] $tag  $(date '+%F %T')" | tee -a "$SUM"
    $PY -u xd_train.py "$@" > "$log" 2>&1
    local rc=$?; local t1=$(date +%s)
    local ap auc
    ap=$(grep -oE 'best AP [0-9.]+' "$log" | tail -1 | awk '{print $3}')
    auc=$(grep -oE 'best AUC [0-9.]+' "$log" | tail -1 | awk '{print $3}')
    echo "[SUMMARY] tag=$tag bestAP=${ap:-NA} bestAUC=${auc:-NA} secs=$((t1-t0)) rc=$rc" | tee -a "$SUM"
    rm -f model/*.pt best_model/*.pt 2>/dev/null
    if [ "$rc" -ne 0 ]; then
        echo "[run] $tag FAILED rc=$rc" | tee -a "$SUM"
        tail -25 "$log" | sed 's/^/    /'
        return 1
    fi
    return 0
}

echo "=== wf_batch start $(date '+%F %T') ==="
echo "[env] $($PY -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0), "tf32=%s"%torch.backends.cuda.matmul.allow_tf32)')"
echo "[disk] $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free"
echo "[seeds] $SEEDS"
echo "[gate] phase 0 must land near the published delta -0.0754"

# ---------------------------------------------------------------------------
# Phase 0 -- replication gate at the published seed.
# ---------------------------------------------------------------------------
echo ""
echo "--- phase 0: replication gate, seed 234 ---"
run "ctrl_s234" $COMMON --seed 234 \
    || { echo "PHASE 0 control FAILED"; exit 1; }
run "wf192_s234" $COMMON $WAVE --seed 234 \
    || { echo "PHASE 0 wave FAILED"; exit 1; }

C=$(grep -oE 'tag=ctrl_s234 bestAP=[0-9.]+' "$SUM" | cut -d= -f3)
W=$(grep -oE 'tag=wf192_s234 bestAP=[0-9.]+' "$SUM" | cut -d= -f3)
echo ""
echo "  published: control 0.8490, wave 0.7736, delta -0.0754"
echo "  measured : control ${C:-NA}, wave ${W:-NA}"
if [ -n "$C" ] && [ -n "$W" ]; then
    echo "$C $W" | awk '{d=$2-$1; printf "  measured delta %.4f\n", d; if (d < -0.045 && d > -0.105) {print "  GATE PASS: same sign and within 0.03 of the published delta"} else {print "  GATE FAIL: not the same harness, or the row does not reproduce"}}'
fi
echo "  (the gate is informative, not automatic: phase 1 still runs so the n=4"
echo "   estimate can be read either way. Decide after seeing both.)"

# ---------------------------------------------------------------------------
# Phase 1 -- paired n=4 (or n=8 with WITH_N=1). Seeds 234 is already done by
# phase 0, so it is skipped by the resume check rather than re-run.
# ---------------------------------------------------------------------------
echo ""
echo "--- phase 1: paired, seeds $SEEDS ---"
for s in $SEEDS; do
  run "ctrl_s$s" $COMMON --seed "$s" \
      || { echo "PHASE 1 control broke at seed $s"; break; }
  run "wf192_s$s" $COMMON $WAVE --seed "$s" \
      || { echo "PHASE 1 wave broke at seed $s"; break; }
done

echo ""
echo "=== wf_batch done $(date '+%F %T') ==="
echo "  runs completed: $(grep -c '^\[SUMMARY\]' "$SUM")"
grep '^\[SUMMARY\]' "$SUM" | sed 's/^/  /'
echo "=== disk ==="; df -h /root/autodl-tmp | tail -1
echo "=== checkpoints written (must be 0) ==="
ls -1 model/*.pt best_model/*.pt 2>/dev/null | wc -l
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd /root/autodl-tmp/logs && tar czf /root/autodl-tmp/wf_logs.tgz wf_*.log wf_summary.txt && echo "=== logs tarred ==="
echo "WF BATCH DONE"