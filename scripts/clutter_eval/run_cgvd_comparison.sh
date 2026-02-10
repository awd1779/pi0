#!/bin/bash
# Compare old (hard compositing) vs new (soft compositing + fixes) CGVD behavior
# on the carrot_on_plate task with 2 semantic distractors.
#
# Usage:
#   ./scripts/clutter_eval/run_cgvd_comparison.sh
#   ./scripts/clutter_eval/run_cgvd_comparison.sh --episodes 10 --runs 3
#   ./scripts/clutter_eval/run_cgvd_comparison.sh --quick   # 5 episodes, 1 run, debug images

set -euo pipefail

# === Defaults ===
TASK="widowx_carrot_on_plate"
CHECKPOINT="checkpoints/bridge_beta.pt"
NUM_EPISODES=20
NUM_RUNS=1
NUM_DISTRACTORS=2
DISTRACTORS_FILE="scripts/clutter_eval/distractors/distractors_carrot_semantic.txt"
QUICK_MODE=false
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_LOG_DIR="logs/cgvd_comparison/${TIMESTAMP}"

# === Parse arguments ===
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)       TASK="$2"; shift 2 ;;
        --episodes|-e)   NUM_EPISODES="$2"; shift 2 ;;
        --runs|-r)       NUM_RUNS="$2"; shift 2 ;;
        --distractors|-n) NUM_DISTRACTORS="$2"; shift 2 ;;
        --checkpoint)    CHECKPOINT="$2"; shift 2 ;;
        --quick)         QUICK_MODE=true; NUM_EPISODES=5; NUM_RUNS=1; shift ;;
        --output|-o)     BASE_LOG_DIR="$2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# === Load distractors from file ===
DISTRACTORS=""
count=0
while IFS= read -r line; do
    # Skip comments and empty lines
    [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
    line=$(echo "$line" | xargs)  # trim whitespace
    [[ -z "$line" ]] && continue
    DISTRACTORS="$DISTRACTORS $line"
    count=$((count + 1))
    [[ $NUM_DISTRACTORS -gt 0 && $count -ge $NUM_DISTRACTORS ]] && break
done < "$DISTRACTORS_FILE"
DISTRACTORS=$(echo "$DISTRACTORS" | xargs)  # trim leading space

echo "=============================================="
echo "CGVD Old vs New Comparison"
echo "=============================================="
echo "Task:          $TASK"
echo "Checkpoint:    $CHECKPOINT"
echo "Episodes:      $NUM_EPISODES"
echo "Runs:          $NUM_RUNS"
echo "Distractors:   $DISTRACTORS"
echo "Output:        $BASE_LOG_DIR"
echo "Quick mode:    $QUICK_MODE"
echo "=============================================="

# === Common arguments ===
COMMON_ARGS="--task $TASK --checkpoint_path $CHECKPOINT --use_bf16"
COMMON_ARGS="$COMMON_ARGS --num_episodes $NUM_EPISODES"
COMMON_ARGS="$COMMON_ARGS --distractors $DISTRACTORS"
COMMON_ARGS="$COMMON_ARGS --external_asset_scale 0.1"
COMMON_ARGS="$COMMON_ARGS --recording"

# CGVD base args (shared between old and new)
CGVD_BASE="--use_cgvd --cgvd_update_freq 1 --cgvd_verbose"

# Add debug images in quick mode
if [[ "$QUICK_MODE" == true ]]; then
    CGVD_BASE="$CGVD_BASE --cgvd_save_debug"
fi

# === Run configs ===
# 1. Baseline (no CGVD, no distractors) — clean reference
# 2. Baseline (no CGVD, with distractors) — distractor impact
# 3. Old CGVD (hard compositing, no cache refresh, dilation inside LaMa)
# 4. New CGVD (soft compositing, cache refresh, dilation before safe-set)

declare -A CONFIGS
CONFIGS[1_clean_baseline]=""
CONFIGS[2_distractor_baseline]=""
CONFIGS[3_cgvd_old]="$CGVD_BASE --cgvd_blend_sigma 0 --cgvd_lama_dilation 0 --cgvd_cache_refresh 0"
CONFIGS[4_cgvd_new]="$CGVD_BASE --cgvd_blend_sigma 5.0 --cgvd_lama_dilation 11 --cgvd_cache_refresh 50"

# Config 1 (clean) has no distractors
CLEAN_ARGS="--task $TASK --checkpoint_path $CHECKPOINT --use_bf16"
CLEAN_ARGS="$CLEAN_ARGS --num_episodes $NUM_EPISODES --recording"

# === Results arrays ===
declare -A ALL_RATES

# === Run all configs ===
for ((run=0; run<NUM_RUNS; run++)); do
    SEED=$((42 + run))
    echo ""
    echo "=============================================="
    echo "RUN $((run + 1))/$NUM_RUNS (Seed: $SEED)"
    echo "=============================================="

    # --- Config 1: Clean baseline (no distractors, no CGVD) ---
    echo ""
    echo ">>> [1/4] Clean baseline (no distractors)"
    RUN_DIR="${BASE_LOG_DIR}/run_${run}/1_clean_baseline"
    mkdir -p "$RUN_DIR"

    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/try_checkpoint_in_simpler.py \
        $CLEAN_ARGS --seed $SEED \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "${RUN_DIR}/eval.log"

    RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/eval.log" | tail -1 | grep -o "[0-9.]*")
    ALL_RATES["1_clean_${run}"]=${RATE:-0}
    echo ">>> Clean baseline: ${ALL_RATES["1_clean_${run}"]}%"

    # --- Config 2: Distractor baseline (with distractors, no CGVD) ---
    echo ""
    echo ">>> [2/4] Distractor baseline (with distractors, no CGVD)"
    RUN_DIR="${BASE_LOG_DIR}/run_${run}/2_distractor_baseline"
    mkdir -p "$RUN_DIR"

    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/try_checkpoint_in_simpler.py \
        $COMMON_ARGS --seed $SEED \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "${RUN_DIR}/eval.log"

    RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/eval.log" | tail -1 | grep -o "[0-9.]*")
    ALL_RATES["2_dist_${run}"]=${RATE:-0}
    echo ">>> Distractor baseline: ${ALL_RATES["2_dist_${run}"]}%"

    # --- Config 3: Old CGVD (hard compositing) ---
    echo ""
    echo ">>> [3/4] Old CGVD (hard compositing, blend_sigma=0, no cache refresh)"
    RUN_DIR="${BASE_LOG_DIR}/run_${run}/3_cgvd_old"
    mkdir -p "$RUN_DIR"

    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/try_checkpoint_in_simpler.py \
        $COMMON_ARGS --seed $SEED \
        ${CONFIGS[3_cgvd_old]} \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "${RUN_DIR}/eval.log"

    RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/eval.log" | tail -1 | grep -o "[0-9.]*")
    ALL_RATES["3_old_${run}"]=${RATE:-0}
    echo ">>> Old CGVD: ${ALL_RATES["3_old_${run}"]}%"

    # --- Config 4: New CGVD (soft compositing + all fixes) ---
    echo ""
    echo ">>> [4/4] New CGVD (soft compositing, blend_sigma=5, cache_refresh=50, dilation=11)"
    RUN_DIR="${BASE_LOG_DIR}/run_${run}/4_cgvd_new"
    mkdir -p "$RUN_DIR"

    xvfb-run -a -s "-screen 0 1024x768x24" python scripts/try_checkpoint_in_simpler.py \
        $COMMON_ARGS --seed $SEED \
        ${CONFIGS[4_cgvd_new]} \
        --output_dir "$RUN_DIR" \
        2>&1 | tee "${RUN_DIR}/eval.log"

    RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/eval.log" | tail -1 | grep -o "[0-9.]*")
    ALL_RATES["4_new_${run}"]=${RATE:-0}
    echo ">>> New CGVD: ${ALL_RATES["4_new_${run}"]}%"
done

# === Summary Report ===
REPORT="${BASE_LOG_DIR}/comparison_report.md"
cat > "$REPORT" << 'HEADER'
# CGVD Old vs New Comparison Report

## Goal
CGVD + distractors should match the **clean baseline** (no distractors).
The gap between CGVD and clean baseline is the "cost" of CGVD's image manipulation.

HEADER

echo "## Results" >> "$REPORT"
echo "" >> "$REPORT"
echo "| Config | $(for ((r=0; r<NUM_RUNS; r++)); do printf "Run %d | " $((r+1)); done)Avg |" >> "$REPORT"
echo "|--------|$(for ((r=0; r<NUM_RUNS; r++)); do printf -- "------|"; done)-----|" >> "$REPORT"

for config_name in "1_clean" "2_dist" "3_old" "4_new"; do
    case $config_name in
        1_clean) label="Clean baseline (no distractors)" ;;
        2_dist)  label="Distractor baseline (no CGVD)" ;;
        3_old)   label="CGVD old (hard compositing)" ;;
        4_new)   label="CGVD new (soft + fixes)" ;;
    esac

    sum=0
    row="| $label |"
    for ((r=0; r<NUM_RUNS; r++)); do
        val=${ALL_RATES["${config_name}_${r}"]:-0}
        row="$row ${val}% |"
        sum=$(echo "$sum + $val" | bc)
    done
    avg=$(echo "scale=1; $sum / $NUM_RUNS" | bc)
    row="$row ${avg}% |"
    echo "$row" >> "$REPORT"
done

echo "" >> "$REPORT"
echo "## Key Metrics" >> "$REPORT"
echo "" >> "$REPORT"
echo "- **Old CGVD gap from clean:** measures artifact cost of hard compositing" >> "$REPORT"
echo "- **New CGVD gap from clean:** should be smaller (target: <2%)" >> "$REPORT"
echo "- **Success:** new CGVD ≥ old CGVD AND closer to clean baseline" >> "$REPORT"
echo "" >> "$REPORT"
echo "## Config Details" >> "$REPORT"
echo "" >> "$REPORT"
echo "- Task: $TASK" >> "$REPORT"
echo "- Episodes: $NUM_EPISODES per run" >> "$REPORT"
echo "- Runs: $NUM_RUNS" >> "$REPORT"
echo "- Distractors: $DISTRACTORS" >> "$REPORT"
echo "- Old CGVD: blend_sigma=0, lama_dilation=0 (inside LaMa), cache_refresh=0" >> "$REPORT"
echo "- New CGVD: blend_sigma=5.0, lama_dilation=11 (before safe-set), cache_refresh=50" >> "$REPORT"

echo ""
echo "=============================================="
echo "COMPARISON COMPLETE"
echo "=============================================="
echo ""
cat "$REPORT"
echo ""
echo "Full report: $REPORT"
echo "Logs: $BASE_LOG_DIR"
