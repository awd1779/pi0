#!/bin/bash
# Paired evaluation: runs BOTH baseline and CGVD on identical scenarios
#
# Usage (named args - recommended):
#   ./run_paired_eval.sh --task widowx_put_eggplant_in_basket --episodes 1 --runs 1
#   ./run_paired_eval.sh --task widowx_carrot_on_plate --model openpi --episodes 10
#   ./run_paired_eval.sh --task widowx_spoon_on_towel --model groot --episodes 5
#
# Usage (positional args - legacy):
#   ./run_paired_eval.sh [NUM_EPISODES] [NUM_RUNS] [START_SEED] [DISTRACTORS_FILE] [TASK]
#
# Options:
#   --task, -t           Task name (default: widowx_spoon_on_towel)
#   --episodes, -e       Episodes per run (default: 10)
#   --runs, -r           Number of runs (default: 3)
#   --seed, -s           Starting seed (default: 42)
#   --distractors        Path to distractors file (auto-selected by task if not specified)
#   --model, -m          Model backend: pi0 (default), openpi, or groot
#   --category, -c       Distractor category: semantic, visual, control (default: semantic)
#   --num_distractors, -n  Number of distractors to use (default: 0 = all from file)
#   --randomize_distractors  Randomly sample distractors per episode from pool
#
# CGVD Ablation Flags:
#   --cgvd_no_inpaint    Disable LaMa inpainting (mask only)
#   --cgvd_disable_safeset   Disable safe-set protection
#   --cgvd_no_robot_mask     Don't mask robot gripper
#   --cgvd_high_threshold    Use higher masking threshold (0.8)
#   --cgvd_safe_threshold    Safe-set detection threshold (default: 0.3)
#   --cgvd_dist_threshold    Distractor detection threshold (default: 0.6)
#
# This ensures fair comparison by using the same:
#   - Random seed (episode IDs and distractor placement)
#   - Distractor objects
#   - Number of episodes

# Defaults
NUM_EPISODES=10
NUM_RUNS=3
START_SEED=2
DISTRACTORS_FILE=""
TASK="widowx_spoon_on_towel"
MODEL="pi0"
CATEGORY="semantic"
NUM_DISTRACTORS=0  # 0 means use all from file
RANDOMIZE_DISTRACTORS=false

# CGVD ablation flags
CGVD_USE_INPAINT=true
CGVD_DISABLE_SAFESET=false
CGVD_ROBOT_THRESHOLD=0.3
CGVD_DISTRACTOR_THRESHOLD=0.30
CGVD_SAFE_THRESHOLD=0.15  # Safe-set (target/anchor) threshold
CGVD_BLEND_SIGMA=5.0
CGVD_LAMA_DILATION=11
CGVD_CACHE_REFRESH=50
CGVD_DISTRACTOR_IOU=0.15
ABLATION_TAG=""

# Parse arguments
POSITIONAL_ARGS=()
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
        --runs|-r)
            NUM_RUNS="$2"
            shift 2
            ;;
        --seed|-s)
            START_SEED="$2"
            shift 2
            ;;
        --distractors)
            DISTRACTORS_FILE="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --category|-c)
            CATEGORY="$2"
            shift 2
            ;;
        --num_distractors|-n)
            NUM_DISTRACTORS="$2"
            shift 2
            ;;
        --cgvd_no_inpaint)
            CGVD_USE_INPAINT=false
            ABLATION_TAG="_no_inpaint"
            shift
            ;;
        --cgvd_disable_safeset)
            CGVD_DISABLE_SAFESET=true
            ABLATION_TAG="_no_safeset"
            shift
            ;;
        --cgvd_no_robot_mask)
            CGVD_ROBOT_THRESHOLD=1.0
            ABLATION_TAG="_no_robot"
            shift
            ;;
        --cgvd_high_threshold)
            CGVD_DISTRACTOR_THRESHOLD=0.8
            ABLATION_TAG="_high_thresh"
            shift
            ;;
        --cgvd_safe_threshold)
            CGVD_SAFE_THRESHOLD="$2"
            shift 2
            ;;
        --cgvd_dist_threshold)
            CGVD_DISTRACTOR_THRESHOLD="$2"
            shift 2
            ;;
        --cgvd_blend_sigma)
            CGVD_BLEND_SIGMA="$2"
            shift 2
            ;;
        --cgvd_lama_dilation)
            CGVD_LAMA_DILATION="$2"
            shift 2
            ;;
        --cgvd_cache_refresh)
            CGVD_CACHE_REFRESH="$2"
            shift 2
            ;;
        --randomize_distractors)
            RANDOMIZE_DISTRACTORS=true
            shift
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# Handle positional args for backwards compatibility
if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    [[ -n "${POSITIONAL_ARGS[0]}" ]] && NUM_EPISODES="${POSITIONAL_ARGS[0]}"
    [[ -n "${POSITIONAL_ARGS[1]}" ]] && NUM_RUNS="${POSITIONAL_ARGS[1]}"
    [[ -n "${POSITIONAL_ARGS[2]}" ]] && START_SEED="${POSITIONAL_ARGS[2]}"
    [[ -n "${POSITIONAL_ARGS[3]}" ]] && DISTRACTORS_FILE="${POSITIONAL_ARGS[3]}"
    [[ -n "${POSITIONAL_ARGS[4]}" ]] && TASK="${POSITIONAL_ARGS[4]}"
fi

# Auto-select distractors file based on task AND category if not specified
if [[ -z "$DISTRACTORS_FILE" ]]; then
    case "$TASK" in
        *spoon*)    BASE="spoon" ;;
        *eggplant*) BASE="eggplant" ;;
        *carrot*)   BASE="carrot" ;;
        *banana*)   BASE="banana" ;;
        *cube*)     BASE="cube" ;;
        *)          BASE="" ;;
    esac
    if [[ -n "$BASE" ]]; then
        # Try category-specific file first, fall back to non-categorized
        CATEGORIZED_FILE="scripts/clutter_eval/distractors/distractors_${BASE}_${CATEGORY}.txt"
        FALLBACK_FILE="scripts/clutter_eval/distractors/distractors_${BASE}.txt"
        if [[ -f "$CATEGORIZED_FILE" ]]; then
            DISTRACTORS_FILE="$CATEGORIZED_FILE"
        elif [[ -f "$FALLBACK_FILE" ]]; then
            DISTRACTORS_FILE="$FALLBACK_FILE"
        fi
    fi
fi

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd /home/ubuntu/open-pi-zero

# Model-specific configuration
case "$MODEL" in
    pi0)
        CHECKPOINT="/home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt"
        EVAL_SCRIPT="scripts/try_checkpoint_in_simpler.py"
        RUN_CMD="uv run python"
        MODEL_DISPLAY="Pi0 (bridge_beta.pt)"
        ;;
    openpi)
        # Select the right OpenPI config based on task type
        if [[ "$TASK" == widowx_* ]]; then
            CHECKPOINT="pi0_widowx"  # WidowX/Bridge tasks
            MODEL_DISPLAY="OpenPI (pi0_widowx / pi0_base checkpoint)"
        elif [[ "$TASK" == google_robot_* ]]; then
            CHECKPOINT="pi0_fractal"  # Fractal/Google Robot tasks
            MODEL_DISPLAY="OpenPI (pi0_fractal / pi0_base checkpoint)"
        else
            echo "Unknown task type for OpenPI: $TASK"
            exit 1
        fi
        EVAL_SCRIPT="scripts/eval_openpi.py"
        RUN_CMD="conda run -n openpi python"
        ;;
    groot)
        CHECKPOINT="nvidia/GR00T-N1.6-3B"
        EVAL_SCRIPT="scripts/eval_groot.py"
        RUN_CMD="conda run -n groot python"
        MODEL_DISPLAY="GR00T (N1.6-3B)"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Supported models: pi0, openpi, groot"
        exit 1
        ;;
esac

# Load distractors from file or use default
# File format: one object per line (e.g., "ycb_030_fork:0.7")
# Lines starting with # are ignored
DEFAULT_DISTRACTORS_FILE="scripts/clutter_eval/distractors/distractors.txt"
if [[ -n "$DISTRACTORS_FILE" && -f "$DISTRACTORS_FILE" ]]; then
    # Read from specified file
    if [[ "$RANDOMIZE_DISTRACTORS" == true ]]; then
        # Pass full pool to Python - it will sample per episode
        DISTRACTORS=$(grep -v '^#' "$DISTRACTORS_FILE" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
        echo "[Randomize] Full pool: $DISTRACTORS"
        echo "[Randomize] Will sample $NUM_DISTRACTORS per episode"
    elif [[ $NUM_DISTRACTORS -gt 0 ]]; then
        # Current behavior: take only first N distractors from file
        DISTRACTORS=$(grep -v '^#' "$DISTRACTORS_FILE" | grep -v '^$' | head -n $NUM_DISTRACTORS | tr '\n' ' ' | sed 's/ $//')
    else
        DISTRACTORS=$(grep -v '^#' "$DISTRACTORS_FILE" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
    fi
elif [[ -f "$DEFAULT_DISTRACTORS_FILE" ]]; then
    # Read from default file
    if [[ "$RANDOMIZE_DISTRACTORS" == true ]]; then
        DISTRACTORS=$(grep -v '^#' "$DEFAULT_DISTRACTORS_FILE" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
        echo "[Randomize] Full pool: $DISTRACTORS"
        echo "[Randomize] Will sample $NUM_DISTRACTORS per episode"
    elif [[ $NUM_DISTRACTORS -gt 0 ]]; then
        DISTRACTORS=$(grep -v '^#' "$DEFAULT_DISTRACTORS_FILE" | grep -v '^$' | head -n $NUM_DISTRACTORS | tr '\n' ' ' | sed 's/ $//')
    else
        DISTRACTORS=$(grep -v '^#' "$DEFAULT_DISTRACTORS_FILE" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
    fi
else
    # Fallback to hardcoded defaults
    DISTRACTORS="ycb_030_fork:0.7 rc_fork_11:0.1 ycb_037_scissors:0.70 ycb_040_large_marker:0.70 ycb_032_knife:0.7 ycb_043_phillips_screwdriver:0.70 ycb_042_adjustable_wrench:0.7"
fi

# Check if we have distractors - skip CGVD if none
SKIP_CGVD=false
if [[ -z "$DISTRACTORS" ]]; then
    SKIP_CGVD=true
    echo "NOTE: No distractors specified, will skip CGVD (baseline only)"
fi

# Create hierarchical log directory
# Structure: logs/clutter_eval/{model}/{task_short}/{category}/n{num}_e{eps}_r{runs}_{timestamp}/
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Extract short task name (e.g., widowx_spoon_on_towel -> spoon)
case "$TASK" in
    *spoon*)    TASK_SHORT="spoon" ;;
    *eggplant*) TASK_SHORT="eggplant" ;;
    *carrot*)   TASK_SHORT="carrot" ;;
    *banana*)   TASK_SHORT="banana" ;;
    *cube*)     TASK_SHORT="cube" ;;
    *)          TASK_SHORT=$(echo "$TASK" | sed 's/widowx_//; s/google_robot_//') ;;
esac

LOG_DIR="logs/clutter_eval/${MODEL}/${TASK_SHORT}/${CATEGORY}/n${NUM_DISTRACTORS}_e${NUM_EPISODES}_r${NUM_RUNS}${ABLATION_TAG}_${TIMESTAMP}"
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
echo "Model: $MODEL_DISPLAY"
echo "Task: $TASK"
echo "Category: $CATEGORY"
echo "Num Distractors: $NUM_DISTRACTORS"
echo "Randomize Distractors: $RANDOMIZE_DISTRACTORS"
echo "Episodes per run: $NUM_EPISODES"
echo "Number of runs: $NUM_RUNS"
echo "Seeds: $SEEDS"
echo "Distractors: $DISTRACTORS"
echo "----------------------------------------------"
echo "CGVD Settings:"
echo "  Use Inpaint: $CGVD_USE_INPAINT"
echo "  Disable Safeset: $CGVD_DISABLE_SAFESET"
echo "  Robot Threshold: $CGVD_ROBOT_THRESHOLD"
echo "  Distractor Threshold: $CGVD_DISTRACTOR_THRESHOLD"
echo "  Safe Threshold: $CGVD_SAFE_THRESHOLD"
if [[ -n "$ABLATION_TAG" ]]; then
    echo "  Ablation: $ABLATION_TAG"
fi
echo "----------------------------------------------"
echo "Log dir: $LOG_DIR"
echo "=============================================="

# Save config for reference
cat > "${LOG_DIR}/config.txt" << EOF
MODEL=$MODEL
MODEL_DISPLAY=$MODEL_DISPLAY
TASK=$TASK
TASK_SHORT=$TASK_SHORT
CHECKPOINT=$CHECKPOINT
EVAL_SCRIPT=$EVAL_SCRIPT
RUN_CMD=$RUN_CMD
DISTRACTORS=$DISTRACTORS
DISTRACTORS_FILE=$DISTRACTORS_FILE
CATEGORY=$CATEGORY
NUM_DISTRACTORS=$NUM_DISTRACTORS
RANDOMIZE_DISTRACTORS=$RANDOMIZE_DISTRACTORS
CGVD_USE_INPAINT=$CGVD_USE_INPAINT
CGVD_DISABLE_SAFESET=$CGVD_DISABLE_SAFESET
CGVD_ROBOT_THRESHOLD=$CGVD_ROBOT_THRESHOLD
CGVD_DISTRACTOR_THRESHOLD=$CGVD_DISTRACTOR_THRESHOLD
CGVD_SAFE_THRESHOLD=$CGVD_SAFE_THRESHOLD
ABLATION_TAG=$ABLATION_TAG
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

    # Build model-specific arguments
    if [[ "$MODEL" == "pi0" ]]; then
        MODEL_ARGS="--checkpoint_path $CHECKPOINT --use_bf16"
    elif [[ "$MODEL" == "openpi" ]]; then
        MODEL_ARGS="--model $CHECKPOINT"
    elif [[ "$MODEL" == "groot" ]]; then
        MODEL_ARGS="--model_path $CHECKPOINT --use_bf16"
    fi

    # Build distractor arguments
    DISTRACTOR_ARGS="--distractors $DISTRACTORS"
    if [[ "$RANDOMIZE_DISTRACTORS" == true ]]; then
        DISTRACTOR_ARGS="$DISTRACTOR_ARGS --num_distractors $NUM_DISTRACTORS --randomize_distractors"
    fi

    xvfb-run -a -s "-screen 0 1024x768x24" $RUN_CMD $EVAL_SCRIPT \
        --task $TASK \
        $MODEL_ARGS \
        --seed $SEED \
        --num_episodes $NUM_EPISODES \
        $DISTRACTOR_ARGS \
        --external_asset_scale 0.1 \
        --recording \
        --output_dir "${RUN_DIR}/baseline" \
        2>&1 | tee "${RUN_DIR}/baseline.log"

    # Extract success rate (case-insensitive to match "SUCCESS RATE:")
    BASELINE_RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/baseline.log" | tail -1 | grep -o "[0-9.]*")
    BASELINE_RATES[$run]=${BASELINE_RATE:-0}
    echo ">>> BASELINE Run $((run + 1)) Result: ${BASELINE_RATES[$run]}%"
done

# =============================================
# PHASE 2: Run all CGVD evaluations
# =============================================
if [[ "$SKIP_CGVD" == true ]]; then
    echo ""
    echo "=============================================="
    echo "PHASE 2: CGVD - SKIPPED (no distractors)"
    echo "=============================================="
    # Set CGVD rates to N/A for reporting
    for ((run=0; run<NUM_RUNS; run++)); do
        CGVD_RATES[$run]="N/A"
    done
else
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

        # Build model-specific arguments
        if [[ "$MODEL" == "pi0" ]]; then
            MODEL_ARGS="--checkpoint_path $CHECKPOINT --use_bf16"
        elif [[ "$MODEL" == "openpi" ]]; then
            MODEL_ARGS="--model $CHECKPOINT"
        elif [[ "$MODEL" == "groot" ]]; then
            MODEL_ARGS="--model_path $CHECKPOINT --use_bf16"
        fi

        # Build CGVD-specific arguments based on ablation settings
        CGVD_ARGS="--use_cgvd --cgvd_update_freq 1 --cgvd_verbose --cgvd_save_debug"
        CGVD_ARGS="$CGVD_ARGS --cgvd_presence_threshold $CGVD_SAFE_THRESHOLD"
        CGVD_ARGS="$CGVD_ARGS --cgvd_robot_threshold $CGVD_ROBOT_THRESHOLD"
        CGVD_ARGS="$CGVD_ARGS --cgvd_distractor_threshold $CGVD_DISTRACTOR_THRESHOLD"
        CGVD_ARGS="$CGVD_ARGS --cgvd_blend_sigma $CGVD_BLEND_SIGMA"
        CGVD_ARGS="$CGVD_ARGS --cgvd_lama_dilation $CGVD_LAMA_DILATION"
        CGVD_ARGS="$CGVD_ARGS --cgvd_cache_refresh $CGVD_CACHE_REFRESH"
        CGVD_ARGS="$CGVD_ARGS --cgvd_distractor_iou_threshold $CGVD_DISTRACTOR_IOU"

        # Ablation: Disable inpainting (use mean-color fill instead)
        if [[ "$CGVD_USE_INPAINT" == false ]]; then
            CGVD_ARGS="$CGVD_ARGS --cgvd_disable_inpaint"
        fi

        # Ablation: Disable safe-set protection
        if [[ "$CGVD_DISABLE_SAFESET" == true ]]; then
            CGVD_ARGS="$CGVD_ARGS --cgvd_disable_safeset"
        fi

        # Build distractor arguments
        DISTRACTOR_ARGS="--distractors $DISTRACTORS"
        if [[ "$RANDOMIZE_DISTRACTORS" == true ]]; then
            DISTRACTOR_ARGS="$DISTRACTOR_ARGS --num_distractors $NUM_DISTRACTORS --randomize_distractors"
        fi

        xvfb-run -a -s "-screen 0 1024x768x24" $RUN_CMD $EVAL_SCRIPT \
            --task $TASK \
            $MODEL_ARGS \
            --seed $SEED \
            --num_episodes $NUM_EPISODES \
            $DISTRACTOR_ARGS \
            --external_asset_scale 0.1 \
            --recording \
            --output_dir "${RUN_DIR}/cgvd" \
            $CGVD_ARGS \
            2>&1 | tee "${RUN_DIR}/cgvd.log"

        # Extract success rate (case-insensitive to match "SUCCESS RATE:")
        CGVD_RATE=$(grep -oi "SUCCESS RATE: [0-9.]*" "${RUN_DIR}/cgvd.log" | tail -1 | grep -o "[0-9.]*")
        CGVD_RATES[$run]=${CGVD_RATE:-0}
        echo ">>> CGVD Run $((run + 1)) Result: ${CGVD_RATES[$run]}%"
    done
fi

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

BASELINE_MEAN=$(echo $BASELINE_STATS | cut -d' ' -f1)
BASELINE_STD=$(echo $BASELINE_STATS | cut -d' ' -f2)
BASELINE_MIN=$(echo $BASELINE_STATS | cut -d' ' -f3)
BASELINE_MAX=$(echo $BASELINE_STATS | cut -d' ' -f4)

if [[ "$SKIP_CGVD" == true ]]; then
    CGVD_MEAN="N/A"
    CGVD_STD="N/A"
    CGVD_MIN="N/A"
    CGVD_MAX="N/A"
    AVG_IMPROVEMENT="N/A"
else
    CGVD_STATS=$(calc_stats "${CGVD_RATES[@]}")
    CGVD_MEAN=$(echo $CGVD_STATS | cut -d' ' -f1)
    CGVD_STD=$(echo $CGVD_STATS | cut -d' ' -f2)
    CGVD_MIN=$(echo $CGVD_STATS | cut -d' ' -f3)
    CGVD_MAX=$(echo $CGVD_STATS | cut -d' ' -f4)

    # Calculate average improvement
    AVG_IMPROVEMENT=$(echo "scale=2; $CGVD_MEAN - $BASELINE_MEAN" | bc -l)
    if (( $(echo "$AVG_IMPROVEMENT >= 0" | bc -l) )); then
        AVG_IMPROVEMENT="+${AVG_IMPROVEMENT}"
    fi
fi

echo ""
echo "=============================================="
echo "SUMMARY STATISTICS"
echo "=============================================="
printf "Baseline: Mean=%.2f%% Std=%.2f%% (Min=%.1f%%, Max=%.1f%%)\n" "$BASELINE_MEAN" "$BASELINE_STD" "$BASELINE_MIN" "$BASELINE_MAX"
if [[ "$SKIP_CGVD" == true ]]; then
    echo "CGVD:     SKIPPED (no distractors)"
else
    printf "CGVD:     Mean=%.2f%% Std=%.2f%% (Min=%.1f%%, Max=%.1f%%)\n" "$CGVD_MEAN" "$CGVD_STD" "$CGVD_MIN" "$CGVD_MAX"
    echo "Average Improvement: ${AVG_IMPROVEMENT}%"
fi
echo "=============================================="

# Generate Markdown comparison report
REPORT_FILE="${LOG_DIR}/comparison_report.md"
cat > "$REPORT_FILE" << EOF
# Paired Evaluation Report

**Generated:** $(date "+%Y-%m-%d %H:%M:%S")

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | $MODEL_DISPLAY |
| Task | \`$TASK\` |
| Episodes per run | $NUM_EPISODES |
| Number of runs | $NUM_RUNS |
| Seeds | $SEEDS |
| Checkpoint | \`$CHECKPOINT\` |

### Distractors
EOF

# Add distractors as bullet list
for d in $DISTRACTORS; do
    echo "- \`$d\`" >> "$REPORT_FILE"
done

cat >> "$REPORT_FILE" << EOF

### CGVD Parameters
- Use Inpaint: **$CGVD_USE_INPAINT**
- Disable Safeset: **$CGVD_DISABLE_SAFESET**
- Algorithm: \`final_mask = distractor_mask AND (NOT safe_mask)\`
- Safe set: target + anchor + robot (parsed from instruction)
- Update frequency: 1
- Safe-set threshold: $CGVD_SAFE_THRESHOLD
- Robot threshold: $CGVD_ROBOT_THRESHOLD
- Distractor threshold: $CGVD_DISTRACTOR_THRESHOLD
$(if [[ -n "$ABLATION_TAG" ]]; then echo "- **Ablation:** $ABLATION_TAG"; fi)

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

# Generate CSV file for easy analysis/plotting
CSV_FILE="${LOG_DIR}/results.csv"
echo "run,seed,episode,episode_id,baseline_success,cgvd_success,baseline_time,cgvd_time" > "$CSV_FILE"

for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    RUN_DIR="${LOG_DIR}/run_${run}"

    # Extract per-episode results and timing from logs
    for ((ep=0; ep<NUM_EPISODES; ep++)); do
        EP_NUM=$((ep + 1))
        EPISODE_ID=$(( (SEED + ep) % 21 ))

        # Baseline results
        B_LINE=$(grep "^Episode ${EP_NUM}:" "${RUN_DIR}/baseline.log" 2>/dev/null | head -1)
        if echo "$B_LINE" | grep -q "SUCCESS"; then
            B_SUCCESS=1
        else
            B_SUCCESS=0
        fi
        B_TIME=$(echo "$B_LINE" | grep -oE "time=[0-9.]+" | grep -oE "[0-9.]+" || echo "")

        # CGVD results
        if [[ "$SKIP_CGVD" == true ]]; then
            C_SUCCESS=""
            C_TIME=""
        else
            C_LINE=$(grep "^Episode ${EP_NUM}:" "${RUN_DIR}/cgvd.log" 2>/dev/null | head -1)
            if echo "$C_LINE" | grep -q "SUCCESS"; then
                C_SUCCESS=1
            else
                C_SUCCESS=0
            fi
            C_TIME=$(echo "$C_LINE" | grep -oE "time=[0-9.]+" | grep -oE "[0-9.]+" || echo "")
        fi

        echo "${run},${SEED},${ep},${EPISODE_ID},${B_SUCCESS},${C_SUCCESS},${B_TIME},${C_TIME}" >> "$CSV_FILE"
    done
done

# Generate run-level summary CSV
SUMMARY_CSV="${LOG_DIR}/summary.csv"
echo "run,seed,baseline_success_rate,cgvd_success_rate,improvement" > "$SUMMARY_CSV"
for ((run=0; run<NUM_RUNS; run++)); do
    SEED=${RUN_SEEDS[$run]}
    B_RATE=${BASELINE_RATES[$run]}
    C_RATE=${CGVD_RATES[$run]}
    if [[ "$C_RATE" == "N/A" ]]; then
        IMPROVEMENT=""
    else
        IMPROVEMENT=$(echo "scale=2; $C_RATE - $B_RATE" | bc -l)
    fi
    echo "${run},${SEED},${B_RATE},${C_RATE},${IMPROVEMENT}" >> "$SUMMARY_CSV"
done

cat >> "$REPORT_FILE" << EOF
---

## Data Files

- \`results.csv\`: Per-episode data (run, seed, episode, success, timing)
- \`summary.csv\`: Per-run summary (success rates, improvement)

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
