#!/bin/bash
# Sweep across different numbers of distractors: 1, 3, 5, 7, 9
#
# Usage: ./run_distractor_count_sweep.sh --task TASK --category CATEGORY [--episodes N] [--runs N]
#
# Example:
#   ./run_distractor_count_sweep.sh --task widowx_carrot_on_plate --category semantic

set -e

# Defaults
TASK=""
CATEGORY="semantic"
EPISODES=21
RUNS=10
START_SEED=0
DISTRACTOR_COUNTS=(1 3 5 7 9)
DRY_RUN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)
            TASK="$2"
            shift 2
            ;;
        --category|-c)
            CATEGORY="$2"
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
        --seed|-s)
            START_SEED="$2"
            shift 2
            ;;
        --counts)
            IFS=',' read -ra DISTRACTOR_COUNTS <<< "$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "Error: --task is required"
    echo "Usage: ./run_distractor_count_sweep.sh --task TASK --category CATEGORY"
    exit 1
fi

cd /home/ubuntu/open-pi-zero

# Calculate totals
TOTAL_RUNS=$((${#DISTRACTOR_COUNTS[@]} * RUNS * 2))  # x2 for baseline+CGVD
TOTAL_EPISODES=$((TOTAL_RUNS * EPISODES))

echo "=============================================="
echo "DISTRACTOR COUNT SWEEP"
echo "=============================================="
echo "Task: $TASK"
echo "Category: $CATEGORY"
echo "Distractor counts: ${DISTRACTOR_COUNTS[@]}"
echo "Episodes per run: $EPISODES"
echo "Runs per count: $RUNS"
echo "Starting seed: $START_SEED"
echo ""
echo "Total evaluation runs: $TOTAL_RUNS"
echo "Total episodes: $TOTAL_EPISODES"
echo "=============================================="

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "=== DRY RUN MODE ==="
    echo ""
fi

START_TIME=$(date +%s)
COUNT_NUM=0
TOTAL_COUNTS=${#DISTRACTOR_COUNTS[@]}

for NUM_DISTRACTORS in "${DISTRACTOR_COUNTS[@]}"; do
    COUNT_NUM=$((COUNT_NUM + 1))

    echo ""
    echo "######################################################"
    echo "# [$COUNT_NUM/$TOTAL_COUNTS] Distractors: $NUM_DISTRACTORS"
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

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    echo ""
    echo ">>> Completed $COUNT_NUM/$TOTAL_COUNTS (Elapsed: ${HOURS}h ${MINS}m)"
done

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(((TOTAL_ELAPSED % 3600) / 60))
SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=============================================="
echo "DISTRACTOR COUNT SWEEP COMPLETE"
echo "=============================================="
echo "Task: $TASK"
echo "Category: $CATEGORY"
echo "Counts tested: ${DISTRACTOR_COUNTS[@]}"
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo ""
echo "Results in: logs/clutter_eval/pi0/"
echo "=============================================="
