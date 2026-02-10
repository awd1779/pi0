#!/bin/bash
# Run GR00T N1.6 Bridge evaluation on SimplerEnv
#
# This script runs GR00T in the 'groot' conda environment.
# Make sure to set up the environment first:
#   conda create -n groot python=3.10 -y
#   conda activate groot
#   git clone https://github.com/NVIDIA/Isaac-GR00T.git ~/Isaac-GR00T
#   cd ~/Isaac-GR00T && pip install -e .
#   pip install -e ~/allenzren_SimplerEnv
#   pip install -e ~/allenzren_SimplerEnv/ManiSkill2_real2sim
#   pip install -e ~/open-pi-zero
#
# Available Bridge/WidowX tasks:
#   - widowx_spoon_on_towel
#   - widowx_carrot_on_plate
#   - widowx_put_eggplant_in_basket
#   - widowx_stack_cube
#
# Usage:
#   ./run_groot_bridge_eval.sh [TASK] [NUM_EPISODES] [OPTIONS]
#
# Options:
#   --cgvd                    Enable CGVD distractor suppression
#   --distractors FILE        Path to distractor file (e.g., scripts/clutter_eval/distractors/distractors_carrot.txt)
#   --distractors OBJ1,OBJ2   Comma-separated distractor objects
#
# Examples:
#   ./run_groot_bridge_eval.sh widowx_carrot_on_plate 10
#   ./run_groot_bridge_eval.sh widowx_carrot_on_plate 10 --distractors scripts/clutter_eval/distractors/distractors_carrot.txt
#   ./run_groot_bridge_eval.sh widowx_carrot_on_plate 10 --distractors scripts/clutter_eval/distractors/distractors_carrot.txt --cgvd

set -e

# Default values
TASK=${1:-"widowx_carrot_on_plate"}
NUM_EPISODES=${2:-10}
USE_CGVD=""
DISTRACTORS=""
DISTRACTOR_ARGS=""

# Parse optional arguments
shift 2 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --cgvd)
            # Use SAM3 server since SAM3 requires transformers >= 5.0.0 (GR00T needs 4.53.0)
            USE_CGVD="--use_cgvd --cgvd_update_freq 1 --cgvd_presence_threshold 0.5 --cgvd_verbose --cgvd_save_debug --cgvd_use_server"
            shift
            ;;
        --distractors)
            DISTRACTORS="$2"
            # Check if it's a file path or direct object list
            if [[ -f "$DISTRACTORS" ]]; then
                # Read from file, filter comments and empty lines (space-separated for nargs="*")
                DISTRACTOR_LIST=$(grep -v '^#' "$DISTRACTORS" | grep -v '^$' | tr '\n' ' ' | sed 's/ $//')
                DISTRACTOR_ARGS="--distractors $DISTRACTOR_LIST"
            else
                # Convert comma-separated to space-separated if needed
                DISTRACTOR_LIST=$(echo "$DISTRACTORS" | tr ',' ' ')
                DISTRACTOR_ARGS="--distractors $DISTRACTOR_LIST"
            fi
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Environment variables
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR

cd /home/ubuntu/open-pi-zero

# Determine output directory based on flags (with timestamp)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -n "$DISTRACTOR_ARGS" ]] && [[ -n "$USE_CGVD" ]]; then
    OUTPUT_DIR="videos/groot_distractors_cgvd/${TASK}_${TIMESTAMP}"
elif [[ -n "$DISTRACTOR_ARGS" ]]; then
    OUTPUT_DIR="videos/groot_distractors/${TASK}_${TIMESTAMP}"
elif [[ -n "$USE_CGVD" ]]; then
    OUTPUT_DIR="videos/groot_cgvd/${TASK}_${TIMESTAMP}"
else
    OUTPUT_DIR="videos/groot_baseline/${TASK}_${TIMESTAMP}"
fi
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "GR00T Bridge Evaluation"
echo "=============================================="
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "Distractors: $([ -n "$DISTRACTOR_ARGS" ] && echo "$DISTRACTORS" || echo 'none')"
echo "CGVD: $([ -n "$USE_CGVD" ] && echo 'enabled' || echo 'disabled')"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Run evaluation in groot conda environment
xvfb-run -a -s "-screen 0 1024x768x24" conda run -n groot python scripts/eval_groot.py \
    --task $TASK \
    --model_path nvidia/GR00T-N1.6-bridge \
    --num_episodes $NUM_EPISODES \
    --recording \
    --output_dir "$OUTPUT_DIR" \
    --use_bf16 \
    --external_asset_scale 0.1 \
    $DISTRACTOR_ARGS \
    $USE_CGVD
