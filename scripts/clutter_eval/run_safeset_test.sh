#!/bin/bash
# Test safe-set protection for CGVD
# This tests the safe-set subtraction feature which prevents SAM3 from
# accidentally blurring the target object when it confuses it with a distractor.
#
# Usage: ./run_safeset_test.sh [NUM_EPISODES]

NUM_EPISODES=${1:-5}

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR

cd /home/ubuntu/open-pi-zero

# Create log directory and filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/safeset_eval"
mkdir -p $LOG_DIR
LOG_FILE="${LOG_DIR}/safeset_test_${TIMESTAMP}.log"

echo "Running safe-set protection test with RC kitchen utensils"
echo "Log file: $LOG_FILE"
echo ""
echo "Test scenario:"
echo "  Task: put the spoon on the towel"
echo "  Distractors: fork, knife, spatula (similar to spoon)"
echo "  Expected: spoon stays SHARP, distractors get blurred"

# Run evaluation and tee output to both console and log file
{
    echo "=============================================="
    echo "SAFE-SET PROTECTION TEST"
    echo "=============================================="
    echo "Timestamp: $(date)"
    echo "Episodes: $NUM_EPISODES"
    echo ""
    echo "Distractor assets: rc_fork_11, rc_knife_26, rc_spatula_1"
    echo "Auto-derived names: fork, knife, spatula"
    echo ""
    echo "Expected behavior:"
    echo "  1. SAM3 queries: fork, knife, spatula (distractors)"
    echo "  2. SAM3 queries: spoon, towel, robot (safe set)"
    echo "  3. Final mask = distractor AND (NOT safe)"
    echo "  4. Spoon NEVER blurred even if SAM3 confuses it"
    echo "=============================================="
    echo ""

    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task widowx_spoon_on_towel \
        --checkpoint_path /home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt \
        --num_episodes $NUM_EPISODES \
        --distractors rc_fork_11 rc_knife_26 rc_spatula_1 \
        --recording \
        --output_dir videos/safeset_test \
        --use_bf16 \
        --use_cgvd \
        --cgvd_update_freq 1 \
        --cgvd_presence_threshold 0.15 \
        --cgvd_verbose \
        --cgvd_save_debug

    EXIT_CODE=$?

    echo ""
    echo "=============================================="
    echo "TEST COMPLETED"
    echo "=============================================="
    echo "End time: $(date)"
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Check debug images at: cgvd_debug/widowx_spoon_on_towel/"
    echo "  - 5 columns: Original | Distractors | Safe Set | Final | Distilled"
    echo "  - Verify spoon appears in 'Safe Set' column"
    echo "  - Verify spoon is REMOVED from 'Final' column"
    echo ""
    echo "Check videos at: videos/safeset_test/"
    echo "  - Verify spoon is SHARP throughout"
    echo "=============================================="

} 2>&1 | tee "$LOG_FILE"

echo ""
echo "Log saved to: $LOG_FILE"
