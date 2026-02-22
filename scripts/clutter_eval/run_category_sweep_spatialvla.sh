#!/bin/bash
# Category + distractor count sweep for SpatialVLA using batch evaluation
#
# This script uses batch_eval_spatialvla.py which loads the model ONCE for the entire sweep.
# Must be run in the 'spatialvla' conda environment (or via conda run).
#
# Usage:
#   ./run_category_sweep_spatialvla.sh [--task TASK] [--dry-run]
#
# Options:
#   --task, -t       Task to evaluate (default: widowx_spoon_on_towel)
#   --episodes, -e   Episodes per configuration (default: 21)
#   --runs, -r       Runs per configuration (default: 10)
#   --seed           Starting seed (default: 0)
#   --categories     Comma-separated categories (default: semantic,visual,control)
#   --counts         Comma-separated distractor counts (default: 0,1,3,5,7,9)
#   --dry-run        Print commands without executing
#   --recording      Save video recordings of each episode
#   --cgvd_save_debug Save CGVD debug images
#   --cgvd_verbose   Print verbose CGVD output
#   --randomize_distractors  Randomly sample distractors per episode from pool
#   --model_path     SpatialVLA model path (default: IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge)
#   --unnorm_key     Unnormalization key (default: bridge_orig/1.0.0)
#   --act_steps      Action steps per inference (default: 4)
#   --cgvd_safe_threshold    Safe-set detection threshold (default: 0.3)
#   --cgvd_robot_threshold   Robot detection threshold (default: 0.3)
#   --cgvd_distractor_threshold  Distractor detection threshold (default: 0.20)

set -e

# Defaults
TASK="widowx_spoon_on_towel"
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
MODEL_PATH="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge"
UNNORM_KEY="bridge_orig/1.0.0"
ACT_STEPS=4

# CGVD thresholds
CGVD_SAFE_THRESHOLD="0.3"
CGVD_ROBOT_THRESHOLD="0.3"
CGVD_DISTRACTOR_THRESHOLD="0.20"

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
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --unnorm_key)
            UNNORM_KEY="$2"
            shift 2
            ;;
        --act_steps)
            ACT_STEPS="$2"
            shift 2
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment setup
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "$PROJECT_ROOT"

# Convert comma-separated to space-separated for Python args
CATEGORIES_ARGS=$(echo $CATEGORIES | tr ',' ' ')
COUNTS_ARGS=$(echo $DISTRACTOR_COUNTS | tr ',' ' ')

# Calculate expected workload
IFS=',' read -ra CAT_ARR <<< "$CATEGORIES"
IFS=',' read -ra COUNT_ARR <<< "$DISTRACTOR_COUNTS"
NUM_CATS=${#CAT_ARR[@]}
NUM_COUNTS=${#COUNT_ARR[@]}
TOTAL_CONFIGS=$((NUM_CATS * NUM_COUNTS * RUNS))
TOTAL_EPISODES=$((TOTAL_CONFIGS * EPISODES * 2))  # x2 for baseline + CGVD

echo "=============================================="
echo "SpatialVLA CATEGORY + DISTRACTOR SWEEP"
echo "(Using batch_eval_spatialvla.py - single model load)"
echo "=============================================="
echo "Task: $TASK"
echo "Model: $MODEL_PATH"
echo "Unnorm key: $UNNORM_KEY"
echo "Act steps: $ACT_STEPS"
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

# Output directory
OUTPUT_DIR="logs/clutter_eval/spatialvla"

# Build command — use conda run to ensure spatialvla env
CMD="xvfb-run -a -s \"-screen 0 1024x768x24\" conda run --no-capture-output -n spatialvla python scripts/clutter_eval/batch_eval_spatialvla.py \
    --task $TASK \
    --model_path $MODEL_PATH \
    --unnorm_key $UNNORM_KEY \
    --act_steps $ACT_STEPS \
    --categories $CATEGORIES_ARGS \
    --distractor_counts $COUNTS_ARGS \
    --episodes $EPISODES \
    --runs $RUNS \
    --start_seed $START_SEED \
    --output_dir $OUTPUT_DIR \
    --cgvd_safe_threshold $CGVD_SAFE_THRESHOLD \
    --cgvd_robot_threshold $CGVD_ROBOT_THRESHOLD \
    --cgvd_distractor_threshold $CGVD_DISTRACTOR_THRESHOLD \
    $DRY_RUN $RECORDING $CGVD_DEBUG $CGVD_VERBOSE $RANDOMIZE_DISTRACTORS"

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
