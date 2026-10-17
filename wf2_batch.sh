#!/bin/bash
# ============================================================================
# Mechanism (a), the wave field: close the confounds, then the warm-start rows.
#
# Where this comes from.  The published row for the wave field reads -0.0754 frame
# AP against a no-wave control at ONE seed.  Re-measured as a paired contrast at
# four seeds it came to -0.0100 +/- 0.0308 (p 0.5610), about a seventh of that.
# Three candidate explanations were identified, and none is excluded yet:
#
#   1. batch size     this ran at 64; src/xd_option.py defaults to 96. Both arms
#                     sat ~0.031 AP below the published control, which is the
#                     fingerprint of a harness/budget difference rather than an
#                     operator effect.
#   2. fusion wiring  the published row carries NO fusion label, while
#                     --wave-fusion defaults to 'logits' and the same table has a
#                     separately labelled 'feature' gate row. They are different
#                     branches and may cost different amounts.
#   3. gain           the table note says "field gain 0.1", and 0.1 was run. But
#                     EVERY surviving wave-field checkpoint on this machine is
#                     tagged g0.02 -- including the two rows whose AP reproduces
#                     the published table exactly. So 0.1 may be a transcription
#                     of 0.02, or a genuinely different setting. There is no
#                     surviving artifact with AP ~ 0.7736 at all.
#
# Part A is a 2x2 over (batch size) x (fusion) at gain 0.1, plus one cell at gain
# 0.02 to test (3), each paired against its own no-wave control. That is the
# minimum that can say WHICH configuration, if any, produces -0.0754. Without it
# the manuscript can only say "not reproduced, cause unknown", which is a weaker
# and less useful statement than naming the cause.
#
# Part B (opt-in, WITH_WARM=1) covers the four warm-start rows. These need a
# converged A0 checkpoint per seed, which does not exist for any seed but one, so
# Part B trains them first: --max-epoch 4, matching the surviving ep4 artifact.
# src/xd_option.py has NO early-stopping flag, so the budget is a fixed cap; that
# is stated here rather than papered over.
#
# Cells and cost (RTX 2080 Ti 11 GB, one-epoch runs ~235 s at bs 64):
#
#   reused from the previous batch, not re-run:
#     ctrl64                 no-wave control, batch 64          (4 seeds)
#     wf_logits64_g010       wave field, logits, gain 0.1, bs 64 (4 seeds)
#
#   Part A, 20 new runs, ~1.4 h:
#     ctrl96                 no-wave control, batch 96          (4)
#     wf_logits96_g010       logits, gain 0.1, bs 96           (4)
#     wf_feat64_g010         feature, gain 0.1, bs 64           (4)
#     wf_feat96_g010         feature, gain 0.1, bs 96           (4)
#     wf_logits64_g002       logits, gain 0.02, bs 64          (4)
#
#   Part B, opt-in, 20 new runs, ~2.2 h:
#     a0                     control to ep4, SAVES a checkpoint (4)
#     warm_logits_joint      --init-ckpt A0, whole net trains   (4)
#     warm_logits_frozen     --init-ckpt A0, --freeze-trunk 1   (4)
#     warm_feat_joint                                           (4)
#     warm_feat_frozen                                          (4)
#
# Ground rules, each of them learned on an earlier box:
#   * CWD must be src/ (imports are top-level).
#   * python is NOT on PATH in a non-interactive shell -> $PY.
#   * EVERY path default in src/xd_option.py is written for CWD = repo root
#     ('list/xd_CLIP_rgb.csv', 'model/model_xd.pth', ...), which contradicts the
#     documented "run from src/". Each is passed explicitly as ../<path> rather
#     than by editing a default, which would change behaviour for every other
#     caller.
#   * src/xd_train.py prints "best AP <x>" and finishes with [NoSave]. It does NOT
#     print the "[EPOCH] ..." or "BEST AP:<x>" formats src_sgl does -- an earlier
#     batch parsed the wrong ones and silently logged bestAP=NA on every run while
#     its resume check became a no-op.
#   * There is no --eval-steps in this tree. Passing one makes argparse reject
#     every run before it starts.
#   * Part A writes no checkpoints (--save-min-ap 0.999). Part B must save, so it
#     uses the real default and clears each seed's A0 once its warm runs are done:
#     679 MB each, and the box has ~16 GB free.
#   * Resumable on [NoSave]; stop on first failure.
# ============================================================================
set -u
cd /root/autodl-tmp/program/waveClipVad/src || exit 1
PY=/root/miniconda3/bin/python

LOGDIR=/root/autodl-tmp/logs
mkdir -p "$LOGDIR"
SUM="$LOGDIR/wf2_summary.txt"
[ -s "$SUM" ] || : > "$SUM"

WITH_WARM="${WITH_WARM:-0}"
SEEDS="42 234 555 789"

# Paths, all explicit, all relative to CWD = src/. See the header.
PATHS="--train-list ../list/xd_CLIP_rgb.csv \
--test-list ../list/xd_CLIP_rgbtest.csv \
--gt-path ../list/gt.npy \
--gt-segment-path ../list/gt_segment.npy \
--gt-label-path ../list/gt_label.npy \
--model-path model/model_xd.pth"

# One-epoch screen budget, no checkpoints. Mirrors the batch that produced the
# published table, which is the only like-for-like comparison available.
COMMON="--max-epoch 1 $PATHS"

run () {
    local tag="$1"; shift
    local log="$LOGDIR/wf2_${tag}.log"
    if grep -q '\[NoSave\]' "$log" 2>/dev/null; then
        echo "[skip] $tag" | tee -a "$SUM"; return 0
    fi
    local t0=$(date +%s)
    echo "[run] $tag  $(date '+%F %T')" | tee -a "$SUM"
    $PY -u xd_train.py "$@" > "$log" 2>&1
    local rc=$?; local t1=$(date +%s)
    local ap
    ap=$(grep -oE 'best AP [0-9.]+' "$log" | tail -1 | awk '{print $3}')
    # The log markers that prove the arm was actually built as intended. A cell
    # that silently ran as something else would otherwise look like a clean null.
    local fused frz init
    fused=$(grep -oE '\[Wave\] fusion=[a-z]+' "$log" | head -1 | awk -F= '{print $2}')
    frz=$(grep -c '\[Freeze\] trunk frozen' "$log")
    init=$(grep -c '\[InitCkpt\]' "$log")
    echo "[SUMMARY] tag=$tag bestAP=${ap:-NA} secs=$((t1-t0)) rc=$rc fusion=${fused:-none} freeze=$frz initckpt=$init" | tee -a "$SUM"
    if [ "$rc" -ne 0 ]; then
        echo "[run] $tag FAILED rc=$rc" | tee -a "$SUM"
        tail -25 "$log" | sed 's/^/    /'
        return 1
    fi
    return 0
}

echo "=== wf2_batch start $(date '+%F %T') ==="
echo "[env] $($PY -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0))')"
echo "[disk] $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free"
echo "[seeds] $SEEDS"
echo "[WITH_WARM] $WITH_WARM"

# ===========================================================================
# Part A -- from-scratch factorial over batch size and fusion, plus gain 0.02.
# ===========================================================================
echo ""
echo "--- part A: from-scratch factorial (no checkpoints) ---"

for s in $SEEDS; do
  # The two cells measured by the previous batch. They are invoked so that the
  # reuse is a real, logged skip rather than a claim in prose: if the reused logs
  # are not present under the expected names, these RUN instead, and the summary
  # will say so, which is the failure we want to see rather than a silently
  # missing control.
  run "ctrl64_s$s"          $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
      || { echo "PART A ctrl64 broke at seed $s"; break; }
  run "wf_logits64_g010_s$s" $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
      --wave-field 192 --wave-field-gain 0.1 --wave-fusion logits \
      || { echo "PART A logits64-g010 broke at seed $s"; break; }
  # control at bs 96. There is no fusion or gain in a no-wave run, so one control
  # per batch size serves every arm at that batch size.
  run "ctrl96_s$s"        $COMMON --save-min-ap 0.999 --batch-size 96 --seed "$s" \
      || { echo "PART A ctrl96 broke at seed $s"; break; }
  run "wf_logits96_g010_s$s" $COMMON --save-min-ap 0.999 --batch-size 96 --seed "$s" \
      --wave-field 192 --wave-field-gain 0.1 --wave-fusion logits \
      || { echo "PART A logits96 broke at seed $s"; break; }
  run "wf_feat64_g010_s$s"   $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
      --wave-field 192 --wave-field-gain 0.1 --wave-fusion feature \
      || { echo "PART A feat64 broke at seed $s"; break; }
  run "wf_feat96_g010_s$s"   $COMMON --save-min-ap 0.999 --batch-size 96 --seed "$s" \
      --wave-field 192 --wave-field-gain 0.1 --wave-fusion feature \
      || { echo "PART A feat96 broke at seed $s"; break; }
  run "wf_logits64_g002_s$s" $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
      --wave-field 192 --wave-field-gain 0.02 --wave-fusion logits \
      || { echo "PART A logits64-g002 broke at seed $s"; break; }
done

echo ""
echo "  part A cells expected to have been SKIPPED as already measured:"
echo "    ctrl64_s*            (no-wave control at bs 64, previous batch)"
echo "    wf_logits64_g010_s*  (wave field, logits, gain 0.1, bs 64, previous batch)"
echo "  If those show as [run] rather than [skip], the reused logs were not found"
echo "  under $LOGDIR and the control the 2x2 is paired against is missing."

# ===========================================================================
# Part B -- opt-in. Warm-start rows. Needs a converged A0 per seed first.
# ===========================================================================
if [ "$WITH_WARM" = "1" ]; then
  echo ""
  echo "--- part B: warm-start rows (A0 must be trained and saved) ---"
  for s in $SEEDS; do
    local_ck="$LOGDIR/a0_s${s}.pt"

    if [ ! -e "$local_ck" ]; then
      # ep4 matches the surviving A0 artifact naming. No early-stop flag exists
      # in this tree, so the cap is fixed.
      #
      # --save-min-ap is 0.0 here, NOT the option default of 0.83. An earlier
      # version used the default and died on the first seed: that seed's A0
      # reached only 0.7597, below 0.83, so nothing was written and the whole
      # seed loop stopped. The bar exists to keep dead weight off a billed disk
      # during the paper's own runs; here the A0 is the *input* to four
      # comparisons, so it must be written whatever it scores.
      $PY -u xd_train.py --max-epoch 4 $PATHS --batch-size 64 --seed "$s" \
          --save-min-ap 0.0 > "$LOGDIR/wf2_a0_s${s}.log" 2>&1
      rc=$?
      ap=$(grep -oE 'best AP [0-9.]+' "$LOGDIR/wf2_a0_s${s}.log" | tail -1 | awk '{print $3}')
      echo "[SUMMARY] tag=a0_s$s bestAP=${ap:-NA} secs=converged rc=$rc fusion=none freeze=0 initckpt=0" | tee -a "$SUM"
      if [ "$rc" -ne 0 ]; then
          echo "PART B a0 FAILED at seed $s -- skipping this seed, not the rest" | tee -a "$SUM"
          tail -25 "$LOGDIR/wf2_a0_s${s}.log" | sed 's/^/    /'
          continue
      fi
      # The trainer writes model/xd_best_*.pt then copies the best into
      # --model-path. Copy that, so the A0 for this seed is one file we control.
      if [ -e model/model_xd.pth ]; then
          cp model/model_xd.pth "$local_ck"
          echo "[A0] seed $s -> $local_ck ($(stat -c%s "$local_ck") B, AP ${ap:-NA})" | tee -a "$SUM"
      else
          # Same rule: one unusable seed must not cost the other three.
          echo "[A0] seed $s FAILED: model/model_xd.pth absent after a converged run -- skipping this seed" | tee -a "$SUM"
          rm -f model/*.pt 2>/dev/null
          continue
      fi
      rm -f model/*.pt 2>/dev/null
    else
      echo "[A0] seed $s already have $(stat -c%s "$local_ck") B" | tee -a "$SUM"
    fi

    seed_ok=1
    for f in logits feature; do
      run "warm_${f}_joint_s$s" $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
          --wave-field 192 --wave-field-gain 0.02 --wave-fusion "$f" \
          --init-ckpt "$local_ck" \
          || { echo "PART B warm_${f}_joint broke at seed $s -- skipping this seed" | tee -a "$SUM"; seed_ok=0; break; }
      run "warm_${f}_frozen_s$s" $COMMON --save-min-ap 0.999 --batch-size 64 --seed "$s" \
          --wave-field 192 --wave-field-gain 0.02 --wave-fusion "$f" \
          --init-ckpt "$local_ck" --freeze-trunk 1 \
          || { echo "PART B warm_${f}_frozen broke at seed $s -- skipping this seed" | tee -a "$SUM"; seed_ok=0; break; }
    done
    [ "$seed_ok" = "1" ] || echo "  (seed $s incomplete: $(ls -1 "$LOGDIR"/wf2_warm_*_s${s}.log 2>/dev/null | wc -l)/4 warm cells)" | tee -a "$SUM"

    # Each A0 is 679 MB and they are per-seed, so release it once its two fusion
    # modes and two freeze settings are done.
    if [ -e "$local_ck" ]; then
        rm -f "$local_ck"
        echo "[A0] seed $s released" | tee -a "$SUM"
    fi
    rm -f model/*.pt 2>/dev/null
  done
else
  echo ""
  echo "--- part B SKIPPED (WITH_WARM=0). Re-run with WITH_WARM=1 to include it. ---"
fi

echo ""
echo "=== wf2_batch done $(date '+%F %T') ==="
echo "  runs completed: $(grep -c '^\[SUMMARY\]' "$SUM")"
grep '^\[SUMMARY\]' "$SUM" | sed 's/^/  /'
echo "=== disk ==="; df -h /root/autodl-tmp | tail -1
echo "=== stray checkpoints in src/model (must be 0) ==="
ls -1 model/*.pt 2>/dev/null | wc -l
echo "=== A0 checkpoints retained (must be 0 after part B) ==="
ls -1 "$LOGDIR"/a0_s*.pt 2>/dev/null | wc -l
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used --format=csv,noheader
cd "$LOGDIR" && tar czf /root/autodl-tmp/wf2_logs.tgz wf2_*.log wf2_summary.txt 2>/dev/null && echo "=== logs tarred ==="
echo "WF2 BATCH DONE"

