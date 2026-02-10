#!/bin/bash
# Full 4-way comparison: Pi0 vs GR00T, with and without CGVD
#
# Compares:
#   1. Pi0 (baseline)
#   2. Pi0 + CGVD
#   3. GR00T (baseline)
#   4. GR00T + CGVD
#
# Usage:
#   ./run_comparison.sh --task widowx_spoon_on_towel --episodes 10
#   ./run_comparison.sh --task widowx_carrot_on_plate --episodes 5 --seed 100
#
# Options:
#   --task, -t       Task name (default: widowx_spoon_on_towel)
#   --episodes, -e   Number of episodes (default: 10)
#   --seed, -s       Random seed (default: 42)
#   --distractors    Path to distractors file (auto-selected by task if not specified)
#   --pi0-only       Only run Pi0 evaluations
#   --groot-only     Only run GR00T evaluations

set -e

# Defaults
TASK="widowx_spoon_on_towel"
NUM_EPISODES=10
SEED=42
DISTRACTORS_FILE=""
RUN_PI0=true
RUN_GROOT=true

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
        --distractors)
            DISTRACTORS_FILE="$2"
            shift 2
            ;;
        --pi0-only)
            RUN_GROOT=false
            shift
            ;;
        --groot-only)
            RUN_PI0=false
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
LOG_DIR="logs/comparison_${TIMESTAMP}"
mkdir -p $LOG_DIR

# Determine checkpoint based on task
if [[ "$TASK" == widowx_* ]]; then
    PI0_CHECKPOINT="/home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt"
    EMBODIMENT="bridge"
elif [[ "$TASK" == google_robot_* ]]; then
    PI0_CHECKPOINT="/home/ubuntu/open-pi-zero/checkpoints/fractal.pt"
    EMBODIMENT="fractal"
else
    echo "Unknown task type: $TASK"
    exit 1
fi

echo "=============================================="
echo "4-WAY COMPARISON: Pi0 vs GR00T"
echo "=============================================="
echo "Task: $TASK"
echo "Embodiment: $EMBODIMENT"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "Distractors: $DISTRACTORS"
echo "Log dir: $LOG_DIR"
echo "Run Pi0: $RUN_PI0"
echo "Run GR00T: $RUN_GROOT"
echo "=============================================="

# Save config
cat > "${LOG_DIR}/config.txt" << EOF
TASK=$TASK
EMBODIMENT=$EMBODIMENT
PI0_CHECKPOINT=$PI0_CHECKPOINT
DISTRACTORS=$DISTRACTORS
NUM_EPISODES=$NUM_EPISODES
SEED=$SEED
TIMESTAMP=$TIMESTAMP
EOF

# Results arrays
declare -A RESULTS

# =============================================
# Pi0 Evaluations
# =============================================
if [[ "$RUN_PI0" == true ]]; then
    echo ""
    echo "=============================================="
    echo "Pi0 BASELINE"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/pi0_baseline"
    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task $TASK \
        --checkpoint_path $PI0_CHECKPOINT \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/pi0_baseline" \
        --use_bf16 \
        2>&1 | tee "${LOG_DIR}/pi0_baseline.log"

    RESULTS["pi0_baseline"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/pi0_baseline.log" | tail -1 | grep -o "[0-9.]*" || echo "0")

    echo ""
    echo "=============================================="
    echo "Pi0 + CGVD"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/pi0_cgvd"
    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task $TASK \
        --checkpoint_path $PI0_CHECKPOINT \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/pi0_cgvd" \
        --use_bf16 \
        --use_cgvd \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.15 \
        --cgvd_robot_threshold 0.05 \
        --cgvd_distractor_threshold 0.3 \
        --cgvd_darken_strength 1 \
        --cgvd_verbose \
        --cgvd_save_debug \
        2>&1 | tee "${LOG_DIR}/pi0_cgvd.log"

    RESULTS["pi0_cgvd"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/pi0_cgvd.log" | tail -1 | grep -o "[0-9.]*" || echo "0")
fi

# =============================================
# GR00T Evaluations
# =============================================
if [[ "$RUN_GROOT" == true ]]; then
    echo ""
    echo "=============================================="
    echo "GR00T BASELINE"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/groot_baseline"
    xvfb-run -a -s "-screen 0 1024x768x24" conda run -n groot python scripts/eval_groot.py \
        --task $TASK \
        --model_path nvidia/GR00T-N1.6-3B \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/groot_baseline" \
        --use_bf16 \
        2>&1 | tee "${LOG_DIR}/groot_baseline.log"

    RESULTS["groot_baseline"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/groot_baseline.log" | tail -1 | grep -o "[0-9.]*" || echo "0")

    echo ""
    echo "=============================================="
    echo "GR00T + CGVD"
    echo "=============================================="

    mkdir -p "${LOG_DIR}/groot_cgvd"
    xvfb-run -a -s "-screen 0 1024x768x24" conda run -n groot python scripts/eval_groot.py \
        --task $TASK \
        --model_path nvidia/GR00T-N1.6-3B \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        --distractors $DISTRACTORS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${LOG_DIR}/groot_cgvd" \
        --use_bf16 \
        --use_cgvd \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.15 \
        --cgvd_robot_threshold 0.05 \
        --cgvd_distractor_threshold 0.3 \
        --cgvd_darken_strength 1 \
        --cgvd_verbose \
        --cgvd_save_debug \
        2>&1 | tee "${LOG_DIR}/groot_cgvd.log"

    RESULTS["groot_cgvd"]=$(grep -oi "SUCCESS RATE: [0-9.]*" "${LOG_DIR}/groot_cgvd.log" | tail -1 | grep -o "[0-9.]*" || echo "0")
fi

# =============================================
# Summary
# =============================================
echo ""
echo "=============================================="
echo "COMPARISON SUMMARY"
echo "=============================================="
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "----------------------------------------------"

if [[ "$RUN_PI0" == true ]]; then
    PI0_BASE=${RESULTS["pi0_baseline"]:-"N/A"}
    PI0_CGVD=${RESULTS["pi0_cgvd"]:-"N/A"}
    if [[ "$PI0_BASE" != "N/A" && "$PI0_CGVD" != "N/A" ]]; then
        PI0_DIFF=$(echo "scale=1; $PI0_CGVD - $PI0_BASE" | bc -l)
        [[ $(echo "$PI0_DIFF >= 0" | bc -l) -eq 1 ]] && PI0_DIFF="+${PI0_DIFF}"
    else
        PI0_DIFF="N/A"
    fi
    echo "Pi0:"
    echo "  Baseline: ${PI0_BASE}%"
    echo "  CGVD:     ${PI0_CGVD}%"
    echo "  Delta:    ${PI0_DIFF}%"
fi

if [[ "$RUN_GROOT" == true ]]; then
    GROOT_BASE=${RESULTS["groot_baseline"]:-"N/A"}
    GROOT_CGVD=${RESULTS["groot_cgvd"]:-"N/A"}
    if [[ "$GROOT_BASE" != "N/A" && "$GROOT_CGVD" != "N/A" ]]; then
        GROOT_DIFF=$(echo "scale=1; $GROOT_CGVD - $GROOT_BASE" | bc -l)
        [[ $(echo "$GROOT_DIFF >= 0" | bc -l) -eq 1 ]] && GROOT_DIFF="+${GROOT_DIFF}"
    else
        GROOT_DIFF="N/A"
    fi
    echo "GR00T:"
    echo "  Baseline: ${GROOT_BASE}%"
    echo "  CGVD:     ${GROOT_CGVD}%"
    echo "  Delta:    ${GROOT_DIFF}%"
fi

echo "----------------------------------------------"

# Generate report
REPORT_FILE="${LOG_DIR}/comparison_report.md"
cat > "$REPORT_FILE" << EOF
# 4-Way Comparison Report

**Generated:** $(date "+%Y-%m-%d %H:%M:%S")

## Configuration

| Parameter | Value |
|-----------|-------|
| Task | \`$TASK\` |
| Embodiment | $EMBODIMENT |
| Episodes | $NUM_EPISODES |
| Seed | $SEED |

### Distractors
EOF

for d in $DISTRACTORS; do
    echo "- \`$d\`" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

## Results

| Model | CGVD | Success Rate | Improvement |
|-------|------|--------------|-------------|
EOF

if [[ "$RUN_PI0" == true ]]; then
    echo "| Pi0 | No | ${PI0_BASE}% | - |" >> "$REPORT_FILE"
    echo "| Pi0 | Yes | ${PI0_CGVD}% | ${PI0_DIFF}% |" >> "$REPORT_FILE"
fi
if [[ "$RUN_GROOT" == true ]]; then
    echo "| GR00T | No | ${GROOT_BASE}% | - |" >> "$REPORT_FILE"
    echo "| GR00T | Yes | ${GROOT_CGVD}% | ${GROOT_DIFF}% |" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Key Findings

EOF

if [[ "$RUN_PI0" == true && "$RUN_GROOT" == true ]]; then
    cat >> "$REPORT_FILE" << EOF
### Baseline Comparison (no CGVD)
- Pi0: ${PI0_BASE}%
- GR00T: ${GROOT_BASE}%

### With CGVD
- Pi0: ${PI0_CGVD}%
- GR00T: ${GROOT_CGVD}%

### CGVD Improvement
- Pi0: ${PI0_DIFF}%
- GR00T: ${GROOT_DIFF}%
EOF
fi

cat >> "$REPORT_FILE" << EOF

## File Locations

- Log directory: \`$LOG_DIR\`
EOF

if [[ "$RUN_PI0" == true ]]; then
    echo "- Pi0 baseline: \`${LOG_DIR}/pi0_baseline/\`" >> "$REPORT_FILE"
    echo "- Pi0 CGVD: \`${LOG_DIR}/pi0_cgvd/\`" >> "$REPORT_FILE"
fi
if [[ "$RUN_GROOT" == true ]]; then
    echo "- GR00T baseline: \`${LOG_DIR}/groot_baseline/\`" >> "$REPORT_FILE"
    echo "- GR00T CGVD: \`${LOG_DIR}/groot_cgvd/\`" >> "$REPORT_FILE"
fi

echo ""
echo "Report saved to: $REPORT_FILE"
echo "=============================================="
