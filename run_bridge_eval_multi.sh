#!/bin/bash
# Run open-pi-zero Bridge evaluation on SimplerEnv - Multiple episodes

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR
mkdir -p /home/ubuntu/open-pi-zero/results

cd /home/ubuntu/open-pi-zero

successes=0
failures=0

for i in $(seq 0 9); do
    echo "======== Running episode $i ========"

    xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
        --task widowx_spoon_on_towel \
        --checkpoint_path /home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt \
        --seed $i \
        --recording \
        --use_bf16 2>&1 | tee /tmp/episode_$i.log

    # Check if success
    if grep -q "Success: True" /tmp/episode_$i.log; then
        ((successes++))
        echo "Episode $i: SUCCESS"
    else
        ((failures++))
        echo "Episode $i: FAILURE"
    fi
done

echo ""
echo "======== FINAL RESULTS ========"
echo "Successes: $successes / 10"
echo "Failures: $failures / 10"
echo "Success Rate: $(echo "scale=1; $successes * 10" | bc)%"
