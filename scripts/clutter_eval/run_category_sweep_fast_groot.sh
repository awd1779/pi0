#!/bin/bash
# Fast category + distractor count sweep for GR00T N1.6 using batch_eval_groot.py
#
# This is the GR00T equivalent of run_category_sweep_fast.sh.
# It uses batch_eval_groot.py which loads the GR00T model ONCE for the entire sweep.
#
# IMPORTANT: Must be run with the 'groot' conda environment available.
# If using CGVD, start the SAM3 server first in a separate terminal:
#   conda activate sam3 && python scripts/sam3_server.py
#
# Usage:
#   ./scripts/clutter_eval/run_category_sweep_fast_groot.sh [OPTIONS]
#
# Options:
#   --task, -t       Task to evaluate (default: widowx_carrot_on_plate)
#   --episodes, -e   Episodes per configuration (default: 21)
#   --runs, -r       Runs per configuration (default: 10)
#   --seed           Starting seed (default: 0)
#   --categories     Comma-separated categories (default: semantic,visual,control)
#   --counts         Comma-separated distractor counts (default: 0,1,3,5,7,9)
#   --model_path     GR00T model path (default: auto-select based on task)
#   --act_steps      Action steps per inference (default: 4)
#   --dry-run        Print commands without executing
#   --recording      Save video recordings of each episode
#   --cgvd_save_debug Save CGVD debug images
#   --cgvd_verbose   Print verbose CGVD output
#   --randomize_distractors  Randomly sample distractors per episode
#   --cgvd_safe_threshold    Safe-set detection threshold (default: 0.6)
#   --cgvd_robot_threshold   Robot detection threshold (default: 0.3)
#   --cgvd_distractor_threshold  Distractor detection threshold (default: 0.20)
#
# Examples:
#   # Full sweep (same conditions as pi0)
#   ./scripts/clutter_eval/run_category_sweep_fast_groot.sh
#
#   # Quick test
#   ./scripts/clutter_eval/run_category_sweep_fast_groot.sh -e 5 -r 2 --counts 0,3
#
#   # Dry run to see configuration
#   ./scripts/clutter_eval/run_category_sweep_fast_groot.sh --dry-run

set -e

# Defaults (same as run_category_sweep_fast.sh for consistency)
TASK="widowx_carrot_on_plate"
EPISODES=21
RUNS=10
START_SEED=0
CATEGORIES="semantic,visual,control"
DISTRACTOR_COUNTS="0,1,3,5,7,9"
MODEL_PATH=""
ACT_STEPS=4
DRY_RUN=""
RECORDING=""
CGVD_DEBUG=""
CGVD_VERBOSE=""
RANDOMIZE_DISTRACTORS=""

# CGVD thresholds
CGVD_SAFE_THRESHOLD="0.6"
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
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --act_steps)
            ACT_STEPS="$2"
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
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd /home/ubuntu/open-pi-zero

# Convert comma-separated to space-separated for Python args
CATEGORIES_ARGS=$(echo $CATEGORIES | tr ',' ' ')
COUNTS_ARGS=$(echo $DISTRACTOR_COUNTS | tr ',' ' ')

# Auto-select model path based on task if not specified
if [[ -z "$MODEL_PATH" ]]; then
    if [[ "$TASK" == widowx_* ]]; then
        MODEL_PATH="nvidia/GR00T-N1.6-bridge"
    elif [[ "$TASK" == google_robot_* ]]; then
        MODEL_PATH="nvidia/GR00T-N1.6-3B"
    else
        echo "Unknown task type: $TASK (cannot auto-select model)"
        exit 1
    fi
fi

# Calculate expected workload
IFS=',' read -ra CAT_ARR <<< "$CATEGORIES"
IFS=',' read -ra COUNT_ARR <<< "$DISTRACTOR_COUNTS"
NUM_CATS=${#CAT_ARR[@]}
NUM_COUNTS=${#COUNT_ARR[@]}
TOTAL_CONFIGS=$((NUM_CATS * NUM_COUNTS * RUNS))
TOTAL_EPISODES=$((TOTAL_CONFIGS * EPISODES * 2))  # x2 for baseline + CGVD

echo "=============================================="
echo "GR00T FAST CATEGORY + DISTRACTOR SWEEP"
echo "(Using batch_eval_groot.py - single model load)"
echo "=============================================="
echo "Task: $TASK"
echo "Model: $MODEL_PATH"
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
OUTPUT_DIR="logs/clutter_eval/gr00t"

# Build optional model_path argument
MODEL_PATH_ARG=""
if [[ -n "$MODEL_PATH" ]]; then
    MODEL_PATH_ARG="--model_path $MODEL_PATH"
fi

# Build command
CMD="xvfb-run -a -s \"-screen 0 1024x768x24\" conda run -n groot python scripts/clutter_eval/batch_eval_groot.py \
    --task $TASK \
    $MODEL_PATH_ARG \
    --act_steps $ACT_STEPS \
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
echo "GR00T SWEEP COMPLETE"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
