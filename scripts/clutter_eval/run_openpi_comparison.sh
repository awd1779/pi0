#!/bin/bash
# OpenPI Pi0 Evaluation: Baseline vs CGVD
#
# This script evaluates the OpenPI pi0_fast_droid model on SimplerEnv tasks.
# pi0_fast_droid is a cross-embodiment model trained on DROID data.
#
# Usage:
#   ./run_openpi_comparison.sh --task widowx_carrot_on_plate --episodes 10
#   ./run_openpi_comparison.sh --task widowx_spoon_on_towel --episodes 5 --seed 100
#
# Options:
#   --task, -t       Task name (default: widowx_carrot_on_plate)
#   --episodes, -e   Number of episodes (default: 10)
#   --seed, -s       Random seed (default: 42)
#   --model, -m      OpenPI model name (default: pi0_base)
#   --distractors    Path to distractors file (auto-selected by task if not specified)
#   --baseline-only  Only run baseline (no CGVD)
#   --cgvd-only      Only run CGVD evaluation

set -e

# Defaults
TASK="widowx_carrot_on_plate"
NUM_EPISODES=10
SEED=42
MODEL="pi0_fast_droid"
DISTRACTORS_FILE=""
RUN_BASELINE=true
RUN_CGVD=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)
            TASK="$2"
            shift 2
            ;;
        --episodes|-e)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --seed|-s)
            SEED="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --distractors)
            DISTRACTORS_FILE="$2"
            shift 2
            ;;
        --baseline-only)
            RUN_CGVD=false
            shift
            ;;
        --cgvd-only)
            RUN_BASELINE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Auto-select distractors file based on task if not specified
if [[ -z "$DISTRACTORS_FILE" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    case "$TASK" in
        *spoon*)    DISTRACTORS_FILE="${SCRIPT_DIR}/distractors_spoon.txt" ;;
        *eggplant*) DISTRACTORS_FILE="${SCRIPT_DIR}/distractors_eggplant.txt" ;;
        *carrot*)   DISTRACTORS_FILE="${SCRIPT_DIR}/distractors_carrot.txt" ;;
        *cube*)     DISTRACTORS_FILE="${SCRIPT_DIR}/distractors_cube.txt" ;;
    esac
fi

# Load distractors
if [[ -n "$DISTRACTORS_FILE" && -f "$DISTRACTORS_FILE" ]]; then
    DISTRACTORS=$(grep -v '^#' "$DISTRACTORS_FILE" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
else
    # Default distractors if no file
    DISTRACTORS="ycb_030_fork:0.7 rc_fork_11:0.1 ycb_037_scissors:0.70"
fi

# Environment variables
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd /home/ubuntu/open-pi-zero

# Create log directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/openpi_${MODEL}_${TIMESTAMP}"
mkdir -p $LOG_DIR

echo "=============================================="
echo "OpenPI EVALUATION: Baseline vs CGVD"
echo "=============================================="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "Distractors: $DISTRACTORS"
echo "Log dir: $LOG_DIR"
echo "Run baseline: $RUN_BASELINE"
echo "Run CGVD: $RUN_CGVD"
echo "=============================================="

# Save config
cat > "${LOG_DIR}/config.txt" << EOF
MODEL=$MODEL
TASK=$TASK
DISTRACTORS=$DISTRACTORS
NUM_EPISODES=$NUM_EPISODES
SEED=$SEED
TIMESTAMP=$TIMESTAMP
EOF

# Results arrays
declare -A RESULTS

# =============================================
# Baseline Evaluation (with distractors, no CGVD)
# =============================================
if [[ "$RUN_BASELINE" == true ]]; then
    echo ""
    echo "=============================================="
    echo "OpenPI BASELINE (with distractors)"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/baseline"
    xvfb-run -a -s "-screen 0 1024x768x24" conda run -n openpi python scripts/eval_openpi.py \
        --task $TASK \
        --model $MODEL \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/baseline" \
        2>&1 | tee "${LOG_DIR}/baseline.log"

    RESULTS["baseline"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/baseline.log" | tail -1 | grep -o "[0-9.]*" || echo "0")
fi

# =============================================
# CGVD Evaluation
# =============================================
if [[ "$RUN_CGVD" == true ]]; then
    echo ""
    echo "=============================================="
    echo "OpenPI + CGVD"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/cgvd"
    xvfb-run -a -s "-screen 0 1024x768x24" conda run -n openpi python scripts/eval_openpi.py \
        --task $TASK \
        --model $MODEL \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/cgvd" \
        --use_cgvd \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.15 \
        --cgvd_robot_threshold 0.05 \
        --cgvd_distractor_threshold 0.3 \
        --cgvd_darken_strength 1 \
        --cgvd_verbose \
        --cgvd_save_debug \
        2>&1 | tee "${LOG_DIR}/cgvd.log"

    RESULTS["cgvd"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/cgvd.log" | tail -1 | grep -o "[0-9.]*" || echo "0")
fi

# =============================================
# Summary
# =============================================
echo ""
echo "=============================================="
echo "OpenPI COMPARISON SUMMARY"
echo "=============================================="
echo "Model: $MODEL"
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "----------------------------------------------"

BASELINE_RATE=${RESULTS["baseline"]:-"N/A"}
CGVD_RATE=${RESULTS["cgvd"]:-"N/A"}

if [[ "$BASELINE_RATE" != "N/A" && "$CGVD_RATE" != "N/A" ]]; then
    DELTA=$(echo "scale=1; $CGVD_RATE - $BASELINE_RATE" | bc -l)
    [[ $(echo "$DELTA >= 0" | bc -l) -eq 1 ]] && DELTA="+${DELTA}"
else
    DELTA="N/A"
fi

echo "Results:"
echo "  Baseline (+ distractors): ${BASELINE_RATE}%"
echo "  CGVD (+ distractors):     ${CGVD_RATE}%"
echo "  Improvement:              ${DELTA}%"
echo "----------------------------------------------"

# Generate report
REPORT_FILE="${LOG_DIR}/comparison_report.md"
cat > "$REPORT_FILE" << EOF
# OpenPI Evaluation Report

**Generated:** $(date "+%Y-%m-%d %H:%M:%S")

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | \`$MODEL\` |
| Task | \`$TASK\` |
| Episodes | $NUM_EPISODES |
| Seed | $SEED |

### Distractors
EOF

for d in $DISTRACTORS; do
    echo "- \`$d\`" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

## Results

| Condition | Success Rate | Delta |
|-----------|--------------|-------|
| Baseline (+ distractors) | ${BASELINE_RATE}% | - |
| CGVD (+ distractors) | ${CGVD_RATE}% | ${DELTA}% |

## Key Observations

### Why This Test Matters

The \`$MODEL\` model is trained on 10k+ hours of diverse robot data that does NOT include Bridge tasks.
This makes it an ideal candidate for testing CGVD effectiveness because:

1. **Out-of-distribution**: The model hasn't seen these specific tasks during training
2. **More susceptible to distractors**: Without task-specific training, visual distractors are more likely to confuse the policy
3. **Validates CGVD generalization**: If CGVD helps this model, it demonstrates value beyond just fine-tuned models

### Interpretation

- **Baseline low**: Expected - model doesn't know Bridge tasks
- **CGVD improvement**: Shows CGVD value for OOD scenarios
- **No improvement**: May need to tune CGVD params or model can't do the task at all

## File Locations

- Log directory: \`$LOG_DIR\`
- Baseline videos: \`${LOG_DIR}/baseline/\`
- CGVD videos: \`${LOG_DIR}/cgvd/\`
EOF

echo ""
echo "Report saved to: $REPORT_FILE"
echo "=============================================="
