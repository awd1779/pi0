#!/bin/bash
# Fast category + distractor count sweep using batch evaluation
#
# This script uses batch_eval.py which loads the model ONCE for the entire sweep,
# providing ~95% reduction in model loading overhead compared to run_category_sweep.sh
#
# Usage:
#   ./run_category_sweep_fast.sh [--task TASK] [--dry-run]
#
# Options:
#   --task, -t       Task to evaluate (default: widowx_carrot_on_plate)
#   --episodes, -e   Episodes per configuration (default: 21)
#   --runs, -r       Runs per configuration (default: 10)
#   --seed           Starting seed (default: 0)
#   --categories     Comma-separated categories (default: semantic,visual,control)
#   --counts         Comma-separated distractor counts (default: 0,1,3,5,7,9)
#   --dry-run        Print commands without executing
#   --recording      Save video recordings of each episode
#   --cgvd_save_debug Save CGVD debug images (original/mask/distilled)
#   --cgvd_verbose   Print verbose CGVD output
#   --randomize_distractors  Randomly sample distractors per episode from pool
#   --save_attention         Save attention map visualizations for each episode
#   --cgvd_safe_threshold    Safe-set detection threshold (default: 0.3)
#   --cgvd_robot_threshold   Robot detection threshold (default: 0.3)
#   --cgvd_distractor_threshold  Distractor detection threshold (default: 0.20)
#
# Comparison with run_category_sweep.sh:
#   - Old: 360 separate Python processes, each loading model (5-30s × 360 = 30-180 min)
#   - New: 1 Python process, model loaded once (5-30s × 1 = 5-30 sec)
#   - Estimated speedup: 70-85% reduction in total execution time

set -e

# Defaults (same as run_category_sweep.sh for consistency)
TASK="widowx_carrot_on_plate"
EPISODES=21
RUNS=10
START_SEED=0
CATEGORIES="semantic,visual,control"
DISTRACTOR_COUNTS="0,1,3,5,7,9"
DRY_RUN=""
RECORDING=""
CGVD_DEBUG=""
CGVD_VERBOSE=""
RANDOMIZE_DISTRACTORS=""
SAVE_ATTENTION=""

# CGVD thresholds
CGVD_SAFE_THRESHOLD="0.3"
CGVD_ROBOT_THRESHOLD="0.3"
CGVD_DISTRACTOR_THRESHOLD="0.20"

# Prompt / concept overrides
PROMPT_OVERRIDE=""
CGVD_TARGET=""
CGVD_ANCHOR=""

# Source object override
SOURCE_OBJ=""

# Overlay variant aggregation
OVERLAY_VARIANTS="off"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)
            TASK="$2"
            shift 2
            ;;
        --episodes|-e)
            EPISODES="$2"
            shift 2
            ;;
        --runs|-r)
            RUNS="$2"
            shift 2
            ;;
        --seed)
            START_SEED="$2"
            shift 2
            ;;
        --categories|-c)
            CATEGORIES="$2"
            shift 2
            ;;
        --counts)
            DISTRACTOR_COUNTS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry_run"
            shift
            ;;
        --recording)
            RECORDING="--recording"
            shift
            ;;
        --cgvd_save_debug|--debug)
            CGVD_DEBUG="--cgvd_save_debug"
            shift
            ;;
        --cgvd_verbose|--verbose)
            CGVD_VERBOSE="--cgvd_verbose"
            shift
            ;;
        --randomize_distractors)
            RANDOMIZE_DISTRACTORS="--randomize_distractors"
            shift
            ;;
        --save_attention)
            SAVE_ATTENTION="--save_attention"
            shift
            ;;
        --cgvd_safe_threshold)
            CGVD_SAFE_THRESHOLD="$2"
            shift 2
            ;;
        --cgvd_robot_threshold)
            CGVD_ROBOT_THRESHOLD="$2"
            shift 2
            ;;
        --cgvd_distractor_threshold)
            CGVD_DISTRACTOR_THRESHOLD="$2"
            shift 2
            ;;
        --prompt)
            PROMPT_OVERRIDE="$2"
            shift 2
            ;;
        --cgvd_target)
            CGVD_TARGET="$2"
            shift 2
            ;;
        --cgvd_anchor)
            CGVD_ANCHOR="$2"
            shift 2
            ;;
        --source_obj)
            SOURCE_OBJ="$2"
            shift 2
            ;;
        --overlay_variants)
            OVERLAY_VARIANTS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment setup
# Use PROJECT_ROOT to support both local (/home/ubuntu/open-pi-zero) and Docker (/workspace/open-pi-zero)
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/../.cache/transformers}"
export VLA_LOG_DIR="${VLA_LOG_DIR:-${PROJECT_ROOT}/logs}"
export VLA_WANDB_ENTITY="${VLA_WANDB_ENTITY:-none}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "$PROJECT_ROOT"

# Convert comma-separated to space-separated for Python args
CATEGORIES_ARGS=$(echo $CATEGORIES | tr ',' ' ')
COUNTS_ARGS=$(echo $DISTRACTOR_COUNTS | tr ',' ' ')

# Select checkpoint based on task
if [[ "$TASK" == widowx_* ]]; then
    CHECKPOINT="${PROJECT_ROOT}/checkpoints/bridge_beta.pt"
elif [[ "$TASK" == google_robot_* ]]; then
    CHECKPOINT="${PROJECT_ROOT}/checkpoints/fractal_beta.pt"
else
    echo "Unknown task type: $TASK"
    exit 1
fi

# Calculate expected workload
IFS=',' read -ra CAT_ARR <<< "$CATEGORIES"
IFS=',' read -ra COUNT_ARR <<< "$DISTRACTOR_COUNTS"
NUM_CATS=${#CAT_ARR[@]}
NUM_COUNTS=${#COUNT_ARR[@]}
TOTAL_CONFIGS=$((NUM_CATS * NUM_COUNTS * RUNS))
TOTAL_EPISODES=$((TOTAL_CONFIGS * EPISODES * 2))  # x2 for baseline + CGVD

echo "=============================================="
echo "FAST CATEGORY + DISTRACTOR SWEEP"
echo "(Using batch_eval.py - single model load)"
echo "=============================================="
echo "Task: $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Categories: $CATEGORIES"
echo "Distractor counts: $DISTRACTOR_COUNTS"
echo "Episodes per config: $EPISODES"
echo "Runs per config: $RUNS"
echo "Starting seed: $START_SEED"
echo "----------------------------------------------"
echo "Total configurations: $TOTAL_CONFIGS"
echo "Total episodes: $TOTAL_EPISODES"
echo "----------------------------------------------"
if [[ -n "$DRY_RUN" ]]; then
    echo "MODE: DRY RUN (will not execute)"
fi
echo "=============================================="

# Create output directory (matching run_paired_eval.sh base path)
# Structure will be: logs/clutter_eval/pi0/{task_short}/{category}/n{num}_e{eps}_r{runs}_{timestamp}/
OUTPUT_DIR="logs/clutter_eval/pi0"

# Use 'uv run' locally, plain 'python' in Docker
if command -v uv &>/dev/null && [ -f "uv.lock" ]; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# Build command
CMD="xvfb-run -a -s \"-screen 0 1024x768x24\" $PYTHON_CMD scripts/clutter_eval/batch_eval.py \
    --task $TASK \
    --checkpoint_path $CHECKPOINT \
    --categories $CATEGORIES_ARGS \
    --distractor_counts $COUNTS_ARGS \
    --episodes $EPISODES \
    --runs $RUNS \
    --start_seed $START_SEED \
    --output_dir $OUTPUT_DIR \
    --use_bf16 \
    --cgvd_safe_threshold $CGVD_SAFE_THRESHOLD \
    --cgvd_robot_threshold $CGVD_ROBOT_THRESHOLD \
    --cgvd_distractor_threshold $CGVD_DISTRACTOR_THRESHOLD \
    $DRY_RUN $RECORDING $CGVD_DEBUG $CGVD_VERBOSE $RANDOMIZE_DISTRACTORS $SAVE_ATTENTION"

# Append optional overrides (only if set)
if [[ -n "$PROMPT_OVERRIDE" ]]; then
    CMD="$CMD --prompt \"$PROMPT_OVERRIDE\""
fi
if [[ -n "$CGVD_TARGET" ]]; then
    CMD="$CMD --cgvd_target \"$CGVD_TARGET\""
fi
if [[ -n "$CGVD_ANCHOR" ]]; then
    CMD="$CMD --cgvd_anchor \"$CGVD_ANCHOR\""
fi
if [[ -n "$SOURCE_OBJ" ]]; then
    CMD="$CMD --source_obj \"$SOURCE_OBJ\""
fi
if [[ "$OVERLAY_VARIANTS" != "off" ]]; then
    CMD="$CMD --overlay_variants $OVERLAY_VARIANTS"
fi

echo ""
echo "Running: $CMD"
echo ""

# Record start time
START_TIME=$(date +%s)

# Execute
eval $CMD

# Record end time
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(((TOTAL_ELAPSED % 3600) / 60))
SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=============================================="
echo "SWEEP COMPLETE"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
