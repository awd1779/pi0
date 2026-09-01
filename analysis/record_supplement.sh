#!/bin/bash
# Record paired episodes for the submission supplement (seed 0, matched conditions).
set -u
REPO=/home/ubuntu/open-pi-zero
DR="docker run --rm --gpus all \
  -v $REPO:/app/open-pi-zero \
  -v /home/ubuntu/.cache:/hostcache \
  -e TRANSFORMERS_CACHE=/hostcache/transformers -e HF_HOME=/hostcache/huggingface \
  -e VLA_LOG_DIR=/app/open-pi-zero/logs -e VLA_WANDB_ENTITY=none \
  -w /app/open-pi-zero drodii/open-pi-zero:latest"
COMMON="--task widowx_spoon_on_towel --checkpoint_path checkpoints/bridge_beta.pt \
  --use_bf16 --randomize_distractors --placement spread --runs 1 --start_seed 0 --recording"

echo "=== [V1] spoon semantic n18 seed0 x10 (paired) $(date)"
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 --episodes 10 \
  --output_dir logs/supplement_videos/spoon_n18
echo "=== [V2] carrot semantic n18 seed0 x10 (paired) $(date)"
$DR python scripts/clutter_eval/batch_eval.py \
  --task widowx_carrot_on_plate --checkpoint_path checkpoints/bridge_beta.pt \
  --use_bf16 --randomize_distractors --placement spread --runs 1 --start_seed 0 --recording \
  --categories semantic --distractor_counts 18 --episodes 10 \
  --output_dir logs/supplement_videos/carrot_n18
echo "=== [V3] attribute n4 complex seed0 x11 (paired) $(date)"
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories attribute --distractor_counts 4 --episodes 11 \
  --prompt "put spoon with green handle on towel" \
  --output_dir logs/supplement_videos/attr_n4
echo "=== [V4] BYOVLA spoon n18 seed0 x2 (flicker clip) $(date)"
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 --episodes 2 --skip_baseline \
  --eval_byovla --byovla_thresh 0.002 \
  --output_dir logs/supplement_videos/byovla_n18
echo "=== recording complete $(date)"
