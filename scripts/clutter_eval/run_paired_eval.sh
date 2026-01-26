#!/bin/bash
# Paired evaluation: runs BOTH baseline and CGVD on identical scenarios
# Usage: ./run_paired_eval.sh [NUM_EPISODES] [NUM_RUNS] [START_SEED]
#
# This ensures fair comparison by using the same:
#   - Random seed (episode IDs and distractor placement)
#   - Distractor objects
#   - Number of episodes
#
# Multiple runs with different seeds provide statistical significance.

NUM_EPISODES=${1:-10}
NUM_RUNS=${2:-3}
START_SEED=${3:-42}

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd /home/ubuntu/open-pi-zero

# Shared configuration
TASK="widowx_spoon_on_towel"
CHECKPOINT="/home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt"
# Distractor object IDs with optional per-object scales (format: object_id:scale)
# Names are auto-derived: rc_fork_11 -> "fork", rc_knife_26 -> "knife", etc.
# Utensils default to scale=1.0, use :0.5 etc to adjust size
DISTRACTORS="ycb_032_knife:0.3 ycb_030_fork:0.3 rc_fork_11:0.1 rc_knife_26:0.1"

# Create log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/clutter_eval/paired_${TIMESTAMP}"
mkdir -p $LOG_DIR

# Calculate seeds for display
SEEDS=""
for ((run=0; run<NUM_RUNS; run++)); do
    if [ $run -gt 0 ]; then
        SEEDS="${SEEDS}, "
    fi
    SEEDS="${SEEDS}$((START_SEED + run))"
done

echo "=============================================="
echo "PAIRED EVALUATION (Multi-Run)"
echo "=============================================="
echo "Episodes per run: $NUM_EPISODES"
echo "Number of runs: $NUM_RUNS"
echo "Seeds: $SEEDS"
echo "Distractors: $DISTRACTORS"
echo "CGVD mode: Safe-set protection (auto-derived names)"
echo "Log dir: $LOG_DIR"
echo "=============================================="

# Save config for reference
cat > "${LOG_DIR}/config.txt" << EOF
TASK=$TASK
CHECKPOINT=$CHECKPOINT
DISTRACTORS=$DISTRACTORS
CGVD_MODE=safe-set-protection (distractor names auto-derived)
NUM_EPISODES=$NUM_EPISODES
NUM_RUNS=$NUM_RUNS
START_SEED=$START_SEED
SEEDS=$SEEDS
TIMESTAMP=$TIMESTAMP
EOF

# Arrays to store results from each run
declare -a BASELINE_RATES
declare -a CGVD_RATES
declare -a RUN_SEEDS

# Pre-compute seeds
for ((run=0; run<NUM_RUNS; run++)); do
    RUN_SEEDS[$run]=$((START_SEED + run))
done

# Create run directories
for ((run=0; run<NUM_RUNS; run++)); do
    mkdir -p "${LOG_DIR}/run_${run}"
done

# =============================================
# PHASE 1: Run all BASELINE evaluations
# =============================================
echo ""
echo "=============================================="
echo "PHASE 1: BASELINE (${NUM_RUNS} runs)"
echo "=============================================="

for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    RUN_DIR="${LOG_DIR}/run_${run}"

    echo ""
    echo ">>> BASELINE Run $((run + 1))/${NUM_RUNS} (Seed: $SEED)"
    echo ""

    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task $TASK \
        --checkpoint_path $CHECKPOINT \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${RUN_DIR}/baseline" \
        --use_bf16 \
        2>&1 | tee "${RUN_DIR}/baseline.log"

    # Extract success rate
    BASELINE_RATE=$(grep -o "Success rate: [0-9.]*" "${RUN_DIR}/baseline.log" | tail -1 | grep -o "[0-9.]*")
    BASELINE_RATES[$run]=${BASELINE_RATE:-0}
    echo ">>> BASELINE Run $((run + 1)) Result: ${BASELINE_RATES[$run]}%"
done

# =============================================
# PHASE 2: Run all CGVD evaluations
# =============================================
echo ""
echo "=============================================="
echo "PHASE 2: CGVD (${NUM_RUNS} runs)"
echo "=============================================="

for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    RUN_DIR="${LOG_DIR}/run_${run}"

    echo ""
    echo ">>> CGVD Run $((run + 1))/${NUM_RUNS} (Seed: $SEED)"
    echo ""

    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task $TASK \
        --checkpoint_path $CHECKPOINT \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${RUN_DIR}/cgvd" \
        --use_bf16 \
        --use_cgvd \
        --cgvd_blur_sigma 15.0 \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.5 \
        --cgvd_verbose \
        --cgvd_save_debug \
        2>&1 | tee "${RUN_DIR}/cgvd.log"

    # Extract success rate
    CGVD_RATE=$(grep -o "Success rate: [0-9.]*" "${RUN_DIR}/cgvd.log" | tail -1 | grep -o "[0-9.]*")
    CGVD_RATES[$run]=${CGVD_RATE:-0}
    echo ">>> CGVD Run $((run + 1)) Result: ${CGVD_RATES[$run]}%"
done

# =============================================
# Summary of all runs
# =============================================
echo ""
echo "=============================================="
echo "ALL RUNS COMPLETE"
echo "=============================================="
echo ""
echo "Results by seed:"
for ((run=0; run<NUM_RUNS; run++)); do
    echo "  Seed ${RUN_SEEDS[$run]}: Baseline=${BASELINE_RATES[$run]}%, CGVD=${CGVD_RATES[$run]}%"
done

# Calculate statistics using awk
calc_stats() {
    local arr=("$@")
    local n=${#arr[@]}

    # Calculate mean
    local sum=0
    local min=${arr[0]}
    local max=${arr[0]}
    for val in "${arr[@]}"; do
        sum=$(echo "$sum + $val" | bc -l)
        if (( $(echo "$val < $min" | bc -l) )); then min=$val; fi
        if (( $(echo "$val > $max" | bc -l) )); then max=$val; fi
    done
    local mean=$(echo "scale=2; $sum / $n" | bc -l)

    # Calculate std
    local sq_sum=0
    for val in "${arr[@]}"; do
        local diff=$(echo "$val - $mean" | bc -l)
        sq_sum=$(echo "$sq_sum + ($diff * $diff)" | bc -l)
    done
    local std=$(echo "scale=2; sqrt($sq_sum / $n)" | bc -l)

    echo "$mean $std $min $max"
}

# Get statistics
BASELINE_STATS=$(calc_stats "${BASELINE_RATES[@]}")
CGVD_STATS=$(calc_stats "${CGVD_RATES[@]}")

BASELINE_MEAN=$(echo $BASELINE_STATS | cut -d' ' -f1)
BASELINE_STD=$(echo $BASELINE_STATS | cut -d' ' -f2)
BASELINE_MIN=$(echo $BASELINE_STATS | cut -d' ' -f3)
BASELINE_MAX=$(echo $BASELINE_STATS | cut -d' ' -f4)

CGVD_MEAN=$(echo $CGVD_STATS | cut -d' ' -f1)
CGVD_STD=$(echo $CGVD_STATS | cut -d' ' -f2)
CGVD_MIN=$(echo $CGVD_STATS | cut -d' ' -f3)
CGVD_MAX=$(echo $CGVD_STATS | cut -d' ' -f4)

# Calculate average improvement
AVG_IMPROVEMENT=$(echo "scale=2; $CGVD_MEAN - $BASELINE_MEAN" | bc -l)
if (( $(echo "$AVG_IMPROVEMENT >= 0" | bc -l) )); then
    AVG_IMPROVEMENT="+${AVG_IMPROVEMENT}"
fi

echo ""
echo "=============================================="
echo "SUMMARY STATISTICS"
echo "=============================================="
printf "Baseline: Mean=%.2f%% Std=%.2f%% (Min=%.1f%%, Max=%.1f%%)\n" "$BASELINE_MEAN" "$BASELINE_STD" "$BASELINE_MIN" "$BASELINE_MAX"
printf "CGVD:     Mean=%.2f%% Std=%.2f%% (Min=%.1f%%, Max=%.1f%%)\n" "$CGVD_MEAN" "$CGVD_STD" "$CGVD_MIN" "$CGVD_MAX"
echo "Average Improvement: ${AVG_IMPROVEMENT}%"
echo "=============================================="

# Generate Markdown comparison report
REPORT_FILE="${LOG_DIR}/comparison_report.md"
cat > "$REPORT_FILE" << EOF
# Paired Evaluation Report

**Generated:** $(date "+%Y-%m-%d %H:%M:%S")

## Configuration

| Parameter | Value |
|-----------|-------|
| Task | \`$TASK\` |
| Episodes per run | $NUM_EPISODES |
| Number of runs | $NUM_RUNS |
| Seeds | $SEEDS |
| Checkpoint | \`$(basename $CHECKPOINT)\` |

### Distractors
EOF

# Add distractors as bullet list
for d in $DISTRACTORS; do
    echo "- \`$d\`" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

### CGVD Parameters
- Mode: **Safe-set protection** (distractor names auto-derived from asset IDs)
- Algorithm: \`final_mask = distractor_mask AND (NOT safe_mask)\`
- Safe set: target + anchor + robot (parsed from instruction)
- Blur sigma: 15.0
- Update frequency: 1
- Presence threshold: 0.5

## Results by Run

| Run | Seed | Baseline | CGVD | Improvement |
|-----|------|----------|------|-------------|
EOF

# Add per-run results
for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    B_RATE=${BASELINE_RATES[$run]}
    C_RATE=${CGVD_RATES[$run]}
    IMPROVEMENT=$(echo "scale=1; $C_RATE - $B_RATE" | bc -l)
    if (( $(echo "$IMPROVEMENT >= 0" | bc -l) )); then
        IMPROVEMENT="+${IMPROVEMENT}"
    fi
    echo "| $((run + 1)) | $SEED | ${B_RATE}% | ${C_RATE}% | ${IMPROVEMENT}% |" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

## Summary Statistics

| Method | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Baseline | ${BASELINE_MEAN}% | ${BASELINE_STD}% | ${BASELINE_MIN}% | ${BASELINE_MAX}% |
| CGVD | ${CGVD_MEAN}% | ${CGVD_STD}% | ${CGVD_MIN}% | ${CGVD_MAX}% |

**Average Improvement: ${AVG_IMPROVEMENT}%**

## Per-Episode Details

EOF

# Add per-episode details for each run
for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    RUN_DIR="${LOG_DIR}/run_${run}"

    echo "### Run $((run + 1)) (Seed: $SEED)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "| Episode | Baseline | CGVD |" >> "$REPORT_FILE"
    echo "|---------|----------|------|" >> "$REPORT_FILE"

    # Extract per-episode results from logs
    BASELINE_EPISODES=$(grep -E "^Episode [0-9]+:" "${RUN_DIR}/baseline.log" 2>/dev/null || echo "")
    CGVD_EPISODES=$(grep -E "^Episode [0-9]+:" "${RUN_DIR}/cgvd.log" 2>/dev/null || echo "")

    for ((ep=1; ep<=NUM_EPISODES; ep++)); do
        B_RESULT=$(echo "$BASELINE_EPISODES" | grep "^Episode $ep:" | sed 's/.*: //' | cut -d' ' -f1)
        C_RESULT=$(echo "$CGVD_EPISODES" | grep "^Episode $ep:" | sed 's/.*: //' | cut -d' ' -f1)
        B_RESULT=${B_RESULT:-"N/A"}
        C_RESULT=${C_RESULT:-"N/A"}
        echo "| $ep | $B_RESULT | $C_RESULT |" >> "$REPORT_FILE"
    done
    echo "" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF
---

## File Locations

- Log directory: \`$LOG_DIR\`
EOF

for ((run=0; run<NUM_RUNS; run++)); do
    echo "- Run $((run + 1)): \`${LOG_DIR}/run_${run}/\`" >> "$REPORT_FILE"
done

echo ""
echo "=============================================="
echo "REPORT GENERATED"
echo "=============================================="
echo "Comparison report: $REPORT_FILE"
echo ""
echo "Results saved to: $LOG_DIR"
for ((run=0; run<NUM_RUNS; run++)); do
    echo "  - Run $((run + 1)): ${LOG_DIR}/run_${run}/"
done
echo "=============================================="
