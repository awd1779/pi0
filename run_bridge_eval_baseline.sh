#!/bin/bash
# Run open-pi-zero Bridge evaluation on SimplerEnv (BASELINE - no CGVD)
#
# Available tasks:
# WidowX (Bridge) - use with bridge_beta.pt:
#   - widowx_spoon_on_towel
#   - widowx_carrot_on_plate
#   - widowx_put_eggplant_in_basket
#   - widowx_stack_cube
#
# Google Robot (Fractal) - use with fractal_*.pt:
#   - google_robot_pick_horizontal_coke_can
#   - google_robot_pick_vertical_coke_can
#   - google_robot_pick_standing_coke_can
#   - google_robot_move_near_v0
#   - google_robot_open_drawer
#   - google_robot_close_drawer
#   - google_robot_place_apple_in_closed_top_drawer

export TRANSFORMERS_CACHE=/home/ubuntu/.cache/transformers
export VLA_LOG_DIR=/home/ubuntu/open-pi-zero/logs
export VLA_WANDB_ENTITY=none
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia

mkdir -p $VLA_LOG_DIR

cd /home/ubuntu/open-pi-zero

# Run evaluation WITHOUT CGVD (baseline) with same distractors for fair comparison
xvfb-run -a -s "-screen 0 1024x768x24" uv run python scripts/try_checkpoint_in_simpler.py \
    --task widowx_spoon_on_towel \
    --checkpoint_path /home/ubuntu/open-pi-zero/checkpoints/bridge_beta.pt \
    --num_episodes 10 \
    --distractors apple orange sponge \
    --recording \
    --output_dir videos/baseline \
    --use_bf16
