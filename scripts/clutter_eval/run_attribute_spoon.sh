#!/bin/bash
# Attribute Distractor Experiment — spoon_on_towel
#
# Tests CGVD's ability to distinguish attribute-level differences:
# target = green-handled spoon, distractors = metal/different-looking spoons
#
# Usage:
#   ./run_attribute_spoon.sh                          # full experiment
#   ./run_attribute_spoon.sh --dry-run                # preview configs
#   ./run_attribute_spoon.sh --episodes 5 --runs 1    # quick test
#   ./run_attribute_spoon.sh --counts 0,1             # subset of counts
#   ./run_attribute_spoon.sh --skip_baseline           # CGVD only
#   ./run_attribute_spoon.sh --prompt "put spoon on towel"  # custom VLA prompt
#   ./run_attribute_spoon.sh --cgvd_target "green spoon"    # custom CGVD target
#   ./run_attribute_spoon.sh --start_seed 5                # start from seed 5
#
# Output: logs/attribute_spoon/

set -e

# ---------- Defaults ----------
EPISODES=20
RUNS=10
START_SEED=0
DISTRACTOR_COUNTS=(0 1 2 3)
DRY_RUN=""
EXTRA_FLAGS=""

# ---------- Parse arguments ----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes|-e)
            EPISODES="$2"
            shift 2
            ;;
        --runs|-r)
            RUNS="$2"
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
        --skip_baseline)
            EXTRA_FLAGS="$EXTRA_FLAGS --skip_baseline"
            shift
            ;;
        --cgvd_save_debug)
            EXTRA_FLAGS="$EXTRA_FLAGS --cgvd_save_debug"
            shift
            ;;
        --prompt)
            PROMPT_OVERRIDE="$2"
            shift 2
            ;;
        --cgvd_target)
            TARGET_OVERRIDE="$2"
            shift 2
            ;;
        --start_seed)
            START_SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------- Fixed configuration ----------
TASK="widowx_spoon_on_towel"
CATEGORY="attribute"
PROMPT="${PROMPT_OVERRIDE:-put spoon on towel}"
CGVD_TARGET="${TARGET_OVERRIDE:-spoon with green handle}"

# ---------- Environment ----------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/../.cache/transformers}"
export VLA_LOG_DIR="${VLA_LOG_DIR:-${PROJECT_ROOT}/logs}"
export VLA_WANDB_ENTITY="${VLA_WANDB_ENTITY:-none}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "$PROJECT_ROOT"

CHECKPOINT="${PROJECT_ROOT}/checkpoints/bridge_beta.pt"
OUTPUT_BASE="logs/attribute_spoon"

# Use 'uv run' locally, plain 'python' in Docker
if command -v uv &>/dev/null && [ -f "uv.lock" ]; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# ---------- Summary ----------
NUM_COUNTS=${#DISTRACTOR_COUNTS[@]}
TOTAL_CONFIGS=$NUM_COUNTS
TOTAL_EPISODES=$((TOTAL_CONFIGS * 2 * RUNS * EPISODES))

echo "=============================================="
echo "ATTRIBUTE DISTRACTOR EXPERIMENT — spoon_on_towel"
echo "=============================================="
echo "Task:        $TASK"
echo "Category:    $CATEGORY"
echo "Prompt:      $PROMPT"
echo "CGVD target: $CGVD_TARGET"
echo "Dist counts: ${DISTRACTOR_COUNTS[*]}"
echo "Episodes:    $EPISODES"
echo "Runs:        $RUNS"
echo "Seeds:       $START_SEED .. $((START_SEED + RUNS - 1))"
echo "Output:      $OUTPUT_BASE/"
echo ""
echo "Total configurations: $TOTAL_CONFIGS"
echo "Total episodes (est): $TOTAL_EPISODES"
echo "=============================================="

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "=== DRY RUN — Commands that would be executed ==="
    echo ""
fi

# ---------- Run ----------
START_TIME=$(date +%s)
CONFIG_NUM=0

for NUM_DIST in "${DISTRACTOR_COUNTS[@]}"; do
    CONFIG_NUM=$((CONFIG_NUM + 1))

    echo ""
    echo "######################################################"
    echo "# [$CONFIG_NUM/$TOTAL_CONFIGS] attribute | ${NUM_DIST} distractors"
    echo "######################################################"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Would run:"
        echo "  xvfb-run -a -s \"-screen 0 1024x768x24\" $PYTHON_CMD scripts/clutter_eval/batch_eval.py \\"
        echo "      --task $TASK \\"
        echo "      --checkpoint_path $CHECKPOINT \\"
        echo "      --categories $CATEGORY \\"
        echo "      --distractor_counts $NUM_DIST \\"
        echo "      --episodes $EPISODES \\"
        echo "      --runs $RUNS \\"
        echo "      --start_seed $START_SEED \\"
        echo "      --output_dir $OUTPUT_BASE \\"
        echo "      --use_bf16 \\"
        echo "      --randomize_distractors \\"
        echo "      --prompt \"$PROMPT\" \\"
        echo "      --cgvd_target \"$CGVD_TARGET\" \\"
        echo "      --placement spread \\"
        echo "      $EXTRA_FLAGS"
    else
        echo "Running: batch_eval.py --categories $CATEGORY --distractor_counts $NUM_DIST --prompt \"$PROMPT\" --cgvd_target \"$CGVD_TARGET\""
        echo ""
        xvfb-run -a -s "-screen 0 1024x768x24" $PYTHON_CMD scripts/clutter_eval/batch_eval.py \
            --task "$TASK" \
            --checkpoint_path "$CHECKPOINT" \
            --categories "$CATEGORY" \
            --distractor_counts "$NUM_DIST" \
            --episodes "$EPISODES" \
            --runs "$RUNS" \
            --start_seed "$START_SEED" \
            --output_dir "$OUTPUT_BASE" \
            --use_bf16 \
            --randomize_distractors \
            --prompt "$PROMPT" \
            --cgvd_target "$CGVD_TARGET" \
            --placement spread \
            $EXTRA_FLAGS
    fi

    # Show elapsed time
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINS=$(((ELAPSED % 3600) / 60))
    echo ""
    echo ">>> Completed $CONFIG_NUM/$TOTAL_CONFIGS (Elapsed: ${HOURS}h ${MINS}m)"
done

# ---------- Summary ----------
END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
HOURS=$((TOTAL_ELAPSED / 3600))
MINS=$(((TOTAL_ELAPSED % 3600) / 60))
SECS=$((TOTAL_ELAPSED % 60))

echo ""
echo "=============================================="
echo "ATTRIBUTE EXPERIMENT COMPLETE"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results in: $OUTPUT_BASE/"
echo "=============================================="
