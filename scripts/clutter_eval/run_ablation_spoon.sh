#!/bin/bash
# CGVD Ablation Study — spoon_on_towel semantic
#
# Conditions:
#   cgvd_no_crossval — CGVD without cross-validation       (--disable_crossval)
#   cgvd_no_robot    — CGVD without robot mask handling    (--disable_robot)
#   cgvd_no_inpaint  — Mean-color fill instead of LaMa     (--disable_inpaint)
#
# Usage:
#   ./run_ablation_spoon.sh [CONDITION] [NUM_DISTRACTORS] [EPISODES] [RUNS]
#
# Examples:
#   ./run_ablation_spoon.sh all                  # all conditions, 10 dist, 20 eps, 10 runs
#   ./run_ablation_spoon.sh all 5                # all, 5 distractors
#   ./run_ablation_spoon.sh cgvd_no_crossval     # one condition, defaults
#   ./run_ablation_spoon.sh cgvd_no_robot 5 5 1  # one condition, 5 dist, 5 eps, 1 run
#   ./run_ablation_spoon.sh baseline 10 20 1     # baseline only, 1 run
#
# Output: logs/ablation_spoon_semantic_{NUM_DISTRACTORS}dist/{condition}/

set -e

# ---------- Arguments ----------
CONDITION="${1:-all}"
NUM_DIST="${2:-10}"
EPISODES="${3:-20}"
RUNS="${4:-10}"

# ---------- Configuration ----------
TASK="widowx_spoon_on_towel"
CATEGORY="semantic"
START_SEED=0

# ---------- Environment ----------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${PROJECT_ROOT}/../.cache/transformers}"
export VLA_LOG_DIR="${VLA_LOG_DIR:-${PROJECT_ROOT}/logs}"
export VLA_WANDB_ENTITY="${VLA_WANDB_ENTITY:-none}"
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

cd "$PROJECT_ROOT"

CHECKPOINT="${PROJECT_ROOT}/checkpoints/bridge_beta.pt"
OUTPUT_BASE="logs/ablation_spoon_semantic_${NUM_DIST}dist"

# Use 'uv run' locally, plain 'python' in Docker
if command -v uv &>/dev/null && [ -f "uv.lock" ]; then
    PYTHON_CMD="uv run python"
else
    PYTHON_CMD="python"
fi

# ---------- Helper ----------
run_condition() {
    local COND_NAME="$1"
    shift
    local EXTRA_FLAGS="$@"

    echo "----------------------------------------------"
    echo "CONDITION: $COND_NAME"
    echo "Flags:     $EXTRA_FLAGS"
    echo "----------------------------------------------"

    local OUT_DIR="${OUTPUT_BASE}/${COND_NAME}"

    local CMD="xvfb-run -a -s \"-screen 0 1024x768x24\" $PYTHON_CMD scripts/clutter_eval/batch_eval.py \
        --task $TASK \
        --checkpoint_path $CHECKPOINT \
        --categories $CATEGORY \
        --distractor_counts $NUM_DIST \
        --episodes $EPISODES \
        --runs $RUNS \
        --start_seed $START_SEED \
        --output_dir $OUT_DIR \
        --use_bf16 \
        --randomize_distractors \
        --cgvd_save_debug \
        $EXTRA_FLAGS"

    echo "Running: $CMD"
    echo ""
    eval $CMD
    echo ""
    echo "$COND_NAME done."
    echo ""
}

# ---------- Condition dispatch ----------
run_by_name() {
    case "$1" in
        cgvd_no_crossval) run_condition "cgvd_no_crossval" "--disable_crossval --skip_baseline" ;;
        cgvd_no_robot)    run_condition "cgvd_no_robot" "--disable_robot --skip_baseline" ;;
        cgvd_no_inpaint)  run_condition "cgvd_no_inpaint" "--disable_inpaint --skip_baseline" ;;
        *)
            echo "Unknown condition: $1"
            echo ""
            echo "Valid conditions: all, cgvd_no_crossval, cgvd_no_robot, cgvd_no_inpaint"
            exit 1
            ;;
    esac
}

# ---------- Header ----------
echo "=============================================="
echo "CGVD ABLATION STUDY — spoon_on_towel semantic"
echo "=============================================="
echo "Task:        $TASK"
echo "Category:    $CATEGORY"
echo "Distractors: $NUM_DIST"
echo "Condition:   $CONDITION"
echo "Episodes:    $EPISODES"
echo "Runs:        $RUNS"
echo "Seeds:       $START_SEED .. $((START_SEED + RUNS - 1))"
echo "Output:      $OUTPUT_BASE/"
echo "=============================================="
echo ""

START_TIME=$(date +%s)

# ---------- Run ----------
if [[ "$CONDITION" == "all" ]]; then
    run_by_name "cgvd_no_crossval"
    run_by_name "cgvd_no_robot"
    run_by_name "cgvd_no_inpaint"
else
    run_by_name "$CONDITION"
fi

# ---------- Summary ----------
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(((ELAPSED % 3600) / 60))
SECS=$((ELAPSED % 60))

echo "=============================================="
echo "ABLATION STUDY COMPLETE"
echo "=============================================="
echo "Total time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results in: $OUTPUT_BASE/"
echo "=============================================="
