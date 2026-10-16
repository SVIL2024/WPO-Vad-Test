#!/bin/bash
# ============================================================================
# Revision batch for the DebiasVAD / WPO-Vad-Test paper.
#
# Everything that is still open after Stage 1/2 and E4b, in ONE batch, with the
# cheap decisive work first so that stopping early still buys something.
#
#   Phase 0  alignment gate        1 run    ~5 min    Is the decimation branch
#                                                      actually the same model?
#   Phase 1  E3 decimation arm     8 runs   ~40 min   Reviewer 1 #3
#   Phase 2  cheap mechanism arms  8 runs   ~45 min   Reviewer 1 #2, partial
#   Phase 3  SSM omega pair        8 runs   ~7.5 h    Reviewer 1 #2, decisive
#
# Phases 0-2 are ON by default (~1.5 h).  Phase 3 is opt-in because it is the
# expensive one: WaveSSM must run at batch 16 (its state tensor is ~12x mamba2's
# and batch 64 OOMs on a 10.8 GiB card), which is 4x the steps, measured at
# 50-65 min per run on this hardware.  Enable with  WITH_SSM=1 bash batch_rev.sh
#
# Ground rules, all of them learned the hard way on earlier boxes:
#   * CWD must be src_sgl (top-level imports).
#   * python is NOT on PATH in a non-interactive shell -> $PY.
#   * --save-min-ap 0.999 => no checkpoint is ever written.  Disk is the scarce
#     resource on a billed instance; we only need the log lines.
#   * Resumable: a run whose log already carries [SUMMARY] is skipped, so the
#     batch can be relaunch after a crash without paying for finished work.
#   * Stop on first failure rather than retrying forever.
#   * The 1-epoch budget is used wherever the published table it will be compared
#     against was measured at 1 epoch, so the comparison is like-for-like.
# ============================================================================
set -u
cd /root/autodl-tmp/program/waveClipVad/src_sgl || exit 1
PY=/root/miniconda3/bin/python

LOGDIR=/root/autodl-tmp/logs
mkdir -p "$LOGDIR"
SUM="$LOGDIR/batch_summary.txt"
[ -s "$SUM" ] || : > "$SUM"

WITH_SSM="${WITH_SSM:-0}"
SEEDS8="42 234 555 789 1234 2024 3131 999999"
SEEDS4="42 234 555 789"          # the paper's own four, see phase 3 note below

# --- COMMON, verbatim from waveClipVad/ops/_job_e1e2.sh L31-34 ---------------
COMMON="--batch-size 64 --lr 1e-05 --mamba-layers 2 --d-state 8 --expand 2 \
--mlp-scale 4 --mamba-version mamba2 --bidirectional --future-weight 1.0 \
--eval-steps 0 --proto-weight 4.0 --dropout 0.5 --weight-decay 0.001 \
--fixed-timebase --save-min-ap 0.999 --multi-scale --max-epoch 1"

# --- bs=16 recipe, verbatim from waveClipVad/ops/_job_wave_ctrl.sh L31-38 ----
# The ONLY difference between the two omega arms is --wave-omega-init.
SSM="--batch-size 16 --lr 1e-05 --mamba-layers 2 --d-state 8 --expand 2 \
--mlp-scale 4 --mamba-version mamba2 --multi-scale --multi-scale-factors 1,8,16 \
--bidirectional --future-weight 1.0 --eval-steps 0 --proto-weight 4.0 \
--dropout 0.5 --weight-decay 0.001 --max-epoch 1 --fixed-timebase \
--save-min-ap 0.999 --wave-ssm --wave-checkpoint-scan --wave-state-chunk 2"

run () {
    local tag="$1"; shift
    local log="$LOGDIR/br_${tag}.log"
    if grep -q '^\[SUMMARY\]' "$log" 2>/dev/null; then
        echo "[skip] $tag" | tee -a "$SUM"; return 0
    fi
    local t0=$(date +%s)
    echo "[run] $tag  $(date '+%F %T')" | tee -a "$SUM"
    $PY -u xd_train_sgl.py "$@" > "$log" 2>&1
    local rc=$?; local t1=$(date +%s)
    local bestap auc
    bestap=$(grep -oE 'BEST AP:[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    auc=$(grep -oE 'AUC:[0-9.]+' "$log" | tail -1 | cut -d: -f2)
    echo "[SUMMARY] tag=$tag bestAP=${bestap:-NA} auc=${auc:-NA} secs=$((t1-t0)) rc=$rc" | tee -a "$SUM"
    rm -f model/*.pt 2>/dev/null
    if [ "$rc" -ne 0 ]; then
        echo "[run] $tag FAILED rc=$rc" | tee -a "$SUM"
        tail -20 "$log" | sed 's/^/    /'
        return 1
    fi
    return 0
}

echo "=== batch_rev start $(date '+%F %T') ==="
echo "[env] $($PY -c 'import torch;print(torch.__version__, torch.cuda.get_device_name(0), "tf32=%s"%torch.backends.cuda.matmul.allow_tf32)')"
echo "[disk] $(df -h /root/autodl-tmp | tail -1 | awk '{print $4}') free"
echo "[ssm]  WITH_SSM=$WITH_SSM"

# ---------------------------------------------------------------------------
# Phase 0 -- alignment gate.
# Before spending 40 min on phase 1, prove the decimation branch really differs
# ONLY in how the branch is pooled.  The check is on the log's own [Config] line
# plus the parameter count, both of which must match the pooled arm exactly; if
# the numbers moved, the control is dirty and phase 1 is not worth running.
# ---------------------------------------------------------------------------
echo ""
echo "--- phase 0: decimation alignment gate ---"
run "p0_decim_s42" $COMMON --seed 42 --multi-scale-factors 1,8,16 --ms-pool-mode decimate \
    || { echo "PHASE 0 FAILED"; exit 1; }
grep -oE 'multi_scale=[^,]*, multi_scale_factors[^,]*' "$LOGDIR/br_p0_decim_s42.log" | sed 's/^/  /'
echo "  (pooled reference, from the same command without --ms-pool-mode):"
echo "    multi_scale=True, multi_scale_factors=scales=1,8,16(pool)"
echo "  the two must differ ONLY in the '(pool)' / '(decimate)' suffix."

# ---------------------------------------------------------------------------
# Phase 1 -- E3, the decimation arm, n=8, one epoch.  Answers Reviewer 1 #3.
#
# Already measured on this hardware at n=8, one epoch (box region-41:16797):
#   default (1,8,16) pool  AP 0.7961 +/- 0.0255
#   fine    (1,2,4)  pool  AP 0.7705 +/- 0.0262
# Adding the decimation view of (1,8,16) makes a third cell:
#   if decimate lands near fine      -> cross-frame mixing is the cause, and the
#                                      paper's "band exposure" reading is wrong
#   if decimate lands near default    -> mixing is not the cause, and the
#                                      original reading survives
# ---------------------------------------------------------------------------
echo ""
echo "--- phase 1: E3 decimation arm, n=8 ---"
for s in $SEEDS8; do
  run "p1_decim_s$s" $COMMON --seed "$s" --multi-scale-factors 1,8,16 --ms-pool-mode decimate \
    || { echo "PHASE 1 broke at seed $s"; break; }
done

# ---------------------------------------------------------------------------
# Phase 2 -- the two CHEAP mechanism arms of tab:mechanisms, n=4.
#
# These are the two that do NOT need the expensive WaveSSM memory path:
#   (b) wave resonator bank, fused (no second Mamba pass -- costs one rFFT/irFFT
#       per band, per the comment at mamba_vad_dual_model.py L401-404)
#   (c) PDE residual regulariser, lambda 2.5 as in the published table
# Both are run on the default arm, whose n=8 one-epoch values are already known,
# so the contrast is same-seed paired with a control we have.
#
# Mechanism (a), the wave field, is NOT in this tree: wave_field.py lives in
# src/ while multi-scale lives in src_sgl/, and src/ has no multi-scale at all.
# Mixing the two harnesses would introduce a new confound, so (a) is left out
# and must be declared as a limit rather than re-measured here.
# ---------------------------------------------------------------------------
echo ""
echo "--- phase 2: cheap mechanism arms, n=4 ---"
for s in $SEEDS4; do
  run "p2_wrb_s$s" $COMMON --seed "$s" --multi-scale-factors 1,8,16 \
      --use-wrb --wrb-where fusion --wrb-kind resonator --wrb-periods '64,16' --wrb-q '4,4' \
      || { echo "PHASE 2 (b) broke at seed $s"; break; }
  run "p2_pde_s$s" $COMMON --seed "$s" --multi-scale-factors 1,8,16 \
      --wave-loss 2.5 \
      || { echo "PHASE 2 (c) broke at seed $s"; break; }
done

# ---------------------------------------------------------------------------
# Phase 3 -- the state-space pair, n=4, opt-in.  The decisive one for R1 #2.
#
# Published: omega != 0 beats omega = 0 by +0.0420 AP at ONE seed, which the
# paper itself calls 1.2 sigma and set aside.  n=4 is the seed count the paper
# already uses everywhere else (tab:arms, tab:paired), so this is a like-for-like
# upgrade rather than a retreat; n=8 would double an already ~7.5 h bill.
#
# Seeds 42 234 555 789 are the same four the paper used for its arm tables, so
# the pair can also be restricted to them for a direct comparison.
# ---------------------------------------------------------------------------
if [ "$WITH_SSM" = "1" ]; then
  echo ""
  echo "--- phase 3: SSM omega pair, n=4 (bs=16, ~55 min/run) ---"
  for s in $SEEDS4; do
    run "p3_ssm_spread_s$s" $SSM --seed "$s" --wave-omega-init spread \
        || { echo "PHASE 3 spread broke at seed $s"; break; }
    run "p3_ssm_omega0_s$s" $SSM --seed "$s" --wave-omega-init 0 \
        || { echo "PHASE 3 omega0 broke at seed $s"; break; }
  done
else
  echo ""
  echo "--- phase 3 SKIPPED (WITH_SSM=0). Re-run with WITH_SSM=1 to include it. ---"
fi

echo ""
echo "=== batch_rev done $(date '+%F %T') ==="
echo "  runs completed: $(grep -c '^\[SUMMARY\]' "$SUM")"
grep '^\[SUMMARY\]' "$SUM" | sed 's/^/  /'
echo "=== disk ==="; df -h /root/autodl-tmp | tail -1
echo "=== model dir (must be 0) ==="; ls -1 model 2>/dev/null | wc -l
echo "=== GPU ==="; nvidia-smi --query-gpu=memory.used --format=csv,noheader
echo "BATCH DONE"