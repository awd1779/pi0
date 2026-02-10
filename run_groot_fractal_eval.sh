#!/bin/bash
# Run GR00T N1.6 Fractal/Google Robot evaluation on SimplerEnv
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
# Available Fractal/Google Robot tasks:
#   - google_robot_pick_horizontal_coke_can
#   - google_robot_pick_vertical_coke_can
#   - google_robot_pick_standing_coke_can
#   - google_robot_move_near_v0
#   - google_robot_open_drawer
#   - google_robot_close_drawer
#   - google_robot_place_apple_in_closed_top_drawer
#
# Usage:
#   ./run_groot_fractal_eval.sh [TASK] [NUM_EPISODES] [--cgvd]
#
# Examples:
#   ./run_groot_fractal_eval.sh google_robot_pick_horizontal_coke_can 10
#   ./run_groot_fractal_eval.sh google_robot_move_near_v0 10 --cgvd

set -e

# Default values
TASK=${1:-"google_robot_pick_horizontal_coke_can"}
NUM_EPISODES=${2:-10}
USE_CGVD=""

# Check for --cgvd flag
for arg in "$@"; do
    if [[ "$arg" == "--cgvd" ]]; then
        USE_CGVD="--use_cgvd --cgvd_update_freq 1 --cgvd_presence_threshold 0.5 --cgvd_verbose --cgvd_save_debug"
    fi
done

# Environment variables
export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR

cd /home/ubuntu/open-pi-zero

# Determine output directory based on CGVD flag
if [[ -n "$USE_CGVD" ]]; then
    OUTPUT_DIR="videos/groot_fractal_cgvd"
else
    OUTPUT_DIR="videos/groot_fractal_baseline"
fi

echo "=============================================="
echo "GR00T Fractal Evaluation"
echo "=============================================="
echo "Task: $TASK"
echo "Episodes: $NUM_EPISODES"
echo "CGVD: $([ -n "$USE_CGVD" ] && echo 'enabled' || echo 'disabled')"
echo "Output: $OUTPUT_DIR"
echo "=============================================="

# Run evaluation in groot conda environment
xvfb-run -a -s "-screen 0 1024x768x24" conda run -n groot python scripts/eval_groot.py \
    --task $TASK \
    --model_path nvidia/GR00T-N1.6-3B \
    --num_episodes $NUM_EPISODES \
    --recording \
    --output_dir "$OUTPUT_DIR" \
    --use_bf16 \
    $USE_CGVD
