#!/bin/bash
# Clutter evaluation with 4 distractor objects (with CGVD)
# Usage: ./run_clutter_4.sh [NUM_EPISODES]

NUM_EPISODES=${1:-10}

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR

cd /home/ubuntu/open-pi-zero

# Create log directory and filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/clutter_eval"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/cgvd_clutter_4_${TIMESTAMP}.log"

echo "Running CGVD evaluation with 4 distractors, $NUM_EPISODES episodes"
echo "Log file: $LOG_FILE"

# Run evaluation and tee output to both console and log file
{
    echo "=============================================="
    echo "CLUTTER EVALUATION LOG"
    echo "=============================================="
    echo "Timestamp: $(date)"
    echo "Mode: CGVD"
    echo "Distractors: 4"
    echo "Episodes: $NUM_EPISODES"
    echo "=============================================="
    echo ""

    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task widowx_spoon_on_towel \
        --checkpoint_path /home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt \
        --num_episodes $NUM_EPISODES \
        --distractors green_cube_3cm yellow_cube_3cm eggplant bridge_carrot_generated_modified \
        --recording \
        --output_dir videos/cgvd_clutter_4 \
        --use_bf16 \
        --use_cgvd \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.15 \
        --cgvd_verbose \
        --cgvd_save_debug

    EXIT_CODE=$?

    echo ""
    echo "=============================================="
    echo "EVALUATION COMPLETED"
    echo "=============================================="
    echo "End time: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo "Log saved to: $LOG_FILE"
    echo "=============================================="

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
