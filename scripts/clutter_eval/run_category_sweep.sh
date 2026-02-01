#!/bin/bash
# Full category + distractor count sweep
#
# Usage: ./run_category_sweep.sh [--task TASK] [--dry-run]
#
# Sweeps across:
# - Categories: semantic, visual, control
# - Distractor counts: 0, 1, 3, 5, 7, 9

set -e

# Defaults
EPISODES=21
RUNS=10
START_SEED=0
TASKS=("widowx_carrot_on_plate")
CATEGORIES=("semantic" "visual" "control")
DISTRACTOR_COUNTS=(0 1 3 5 7 9)
DRY_RUN=""
SINGLE_TASK=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)
            SINGLE_TASK="$2"
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
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --categories|-c)
            IFS=',' read -ra CATEGORIES <<< "$2"
            shift 2
            ;;
        --counts)
            IFS=',' read -ra DISTRACTOR_COUNTS <<< "$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# If single task specified, only run that one
if [[ -n "$SINGLE_TASK" ]]; then
    TASKS=("$SINGLE_TASK")
fi

cd /home/ubuntu/open-pi-zero

# Calculate total runs
NUM_COUNTS=${#DISTRACTOR_COUNTS[@]}
NUM_CATEGORIES=${#CATEGORIES[@]}
NUM_TASKS=${#TASKS[@]}
TOTAL_CONFIGS=$((NUM_TASKS * NUM_CATEGORIES * NUM_COUNTS))
TOTAL_EVALS=$((TOTAL_CONFIGS * 2))  # x2 for baseline+CGVD
TOTAL_EPISODES=$((TOTAL_EVALS * RUNS * EPISODES))

echo "=============================================="
echo "FULL CATEGORY + DISTRACTOR COUNT SWEEP"
echo "=============================================="
echo "Tasks: ${TASKS[@]}"
echo "Categories: ${CATEGORIES[@]}"
echo "Distractor counts: ${DISTRACTOR_COUNTS[@]}"
echo "Episodes per run: $EPISODES"
echo "Runs per config: $RUNS"
echo "Starting seed: $START_SEED"
echo ""
echo "Total configurations: $TOTAL_CONFIGS"
echo "Total evaluation runs: $TOTAL_EVALS (baseline + CGVD each)"
echo "Total episodes: $TOTAL_EPISODES"
echo "=============================================="

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "=== DRY RUN - Commands that would be executed ==="
    echo ""
fi

# Track progress
START_TIME=$(date +%s)
CONFIG_NUM=0

for TASK in "${TASKS[@]}"; do
    for CATEGORY in "${CATEGORIES[@]}"; do
        for NUM_DISTRACTORS in "${DISTRACTOR_COUNTS[@]}"; do
            CONFIG_NUM=$((CONFIG_NUM + 1))

            echo ""
            echo "######################################################"
            echo "# [$CONFIG_NUM/$TOTAL_CONFIGS] $TASK | $CATEGORY | ${NUM_DISTRACTORS} distractors"
            echo "######################################################"
            echo ""

            CMD="./scripts/clutter_eval/run_paired_eval.sh \
                --task $TASK \
                --category $CATEGORY \
                --num_distractors $NUM_DISTRACTORS \
                --episodes $EPISODES \
                --runs $RUNS \
                --seed $START_SEED"

            if [[ "$DRY_RUN" == "true" ]]; then
                echo "Would run: $CMD"
            else
                echo "Running: $CMD"
                echo ""
                eval $CMD
            fi

            # Show elapsed time
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))
            HOURS=$((ELAPSED / 3600))
            MINS=$(((ELAPSED % 3600) / 60))
            echo ""
            echo ">>> Completed $CONFIG_NUM/$TOTAL_CONFIGS (Elapsed: ${HOURS}h ${MINS}m)"
        done
    done
done

# Final summary
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
echo "Results in: logs/clutter_eval/pi0/"
echo "=============================================="
