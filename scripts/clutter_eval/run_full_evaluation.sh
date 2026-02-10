#!/bin/bash
# Full multi-task CGVD evaluation across all WidowX pick-and-place tasks
# Runs distractor sweep for each task with task-specific distractors
#
# Usage: ./run_full_evaluation.sh [--dry-run]
#
# Tasks evaluated:
#   - widowx_spoon_on_towel      (distractors: utensils)
#   - widowx_put_eggplant_in_basket (distractors: elongated vegetables/fruits)
#   - widowx_carrot_on_plate     (distractors: orange/yellow elongated objects)
#   - widowx_stack_cube          (distractors: small geometric objects)

set -e

# Parse arguments
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "=== DRY RUN MODE ==="
    echo ""
fi

# All tasks to evaluate
TASKS=(
    "widowx_spoon_on_towel"
    "widowx_put_eggplant_in_basket"
    "widowx_carrot_on_plate"
    "widowx_stack_cube"
)

cd /home/ubuntu/open-pi-zero

echo "=============================================="
echo "FULL MULTI-TASK CGVD EVALUATION"
echo "=============================================="
echo "Tasks to evaluate: ${#TASKS[@]}"
for task in "${TASKS[@]}"; do
    echo "  - $task"
done
echo "=============================================="
echo ""

# Track start time
START_TIME=$(date +%s)

# Run sweep for each task
TASK_NUM=0
for TASK in "${TASKS[@]}"; do
    TASK_NUM=$((TASK_NUM + 1))
    echo ""
    echo "############################################"
    echo "# TASK $TASK_NUM/${#TASKS[@]}: $TASK"
    echo "############################################"
    echo ""

    ./scripts/clutter_eval/run_distractor_sweep.sh "$TASK" $DRY_RUN

    echo ""
    echo ">>> Task $TASK complete"
    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=============================================="
echo "FULL EVALUATION COMPLETE"
echo "=============================================="
echo "Tasks evaluated: ${#TASKS[@]}"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results directories:"
for task in "${TASKS[@]}"; do
    case "$task" in
        *spoon*)    short="spoon" ;;
        *eggplant*) short="eggplant" ;;
        *carrot*)   short="carrot" ;;
        *cube*)     short="cube" ;;
        *)          short="unknown" ;;
    esac
    echo "  - logs/distractor_sweep_${short}_*/"
done
echo "=============================================="
