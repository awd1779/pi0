#!/bin/bash
# Distractor sweep: test with 0, 1, 3, 5, 7, 9 distractors
# Usage: ./run_distractor_sweep.sh --task widowx_spoon_on_towel --category semantic
#        ./run_distractor_sweep.sh --task widowx_spoon_on_towel --all-categories
#        ./run_distractor_sweep.sh --task widowx_spoon_on_towel --model openpi --all-categories -e 100

TASK="widowx_spoon_on_towel"
CATEGORY="semantic"
EPISODES=24
RUNS=1
MODEL="pi0"
ALL_CATEGORIES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t) TASK="$2"; shift 2 ;;
        --category|-c) CATEGORY="$2"; shift 2 ;;
        --episodes|-e) EPISODES="$2"; shift 2 ;;
        --runs|-r) RUNS="$2"; shift 2 ;;
        --model|-m) MODEL="$2"; shift 2 ;;
        --all-categories) ALL_CATEGORIES=true; shift ;;
        *) shift ;;
    esac
done

# Determine which categories to run
if [[ "$ALL_CATEGORIES" == true ]]; then
    CATEGORIES=("semantic" "visual" "control")
else
    CATEGORIES=("$CATEGORY")
fi

# Run sweep: loop over categories and distractor counts
for CAT in "${CATEGORIES[@]}"; do
    echo ""
    echo "############################################"
    echo "# MODEL: $MODEL | CATEGORY: $CAT"
    echo "############################################"

    for NUM in 0 1 3 5 7 9; do
        echo "=========================================="
        echo "Testing with $NUM distractors ($CAT)"
        echo "=========================================="

        ./scripts/clutter_eval/run_paired_eval.sh \
            --task "$TASK" \
            --model "$MODEL" \
            --category "$CAT" \
            --num_distractors $NUM \
            --episodes $EPISODES \
            --runs $RUNS
    done
done

echo ""
echo "Sweep complete!"
echo "Model: $MODEL"
echo "Task: $TASK"
if [[ "$ALL_CATEGORIES" == true ]]; then
    echo "Categories: semantic, visual, control"
else
    echo "Category: $CATEGORY"
fi
echo "Distractor counts: 0, 1, 3, 5, 7, 9"
