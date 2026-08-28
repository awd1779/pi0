#!/bin/bash
# Stage 2 of the revision GPU queue — run AFTER run_revision_queue.sh finishes.
#   Job 5  R33: SAM3 per-frame robot mask substitution @ n18 semantic (~1 h;
#          per-frame SAM3 adds wall-clock). Same seeds/episodes as the headline
#          cell; CGVD arm only. Per-frame latency lines land in cgvd.log.
set -u

REPO=/home/ubuntu/open-pi-zero
IMG=drodii/open-pi-zero:latest
DR="docker run --rm --gpus all \
  -v $REPO:/app/open-pi-zero \
  -v /home/ubuntu/.cache:/hostcache \
  -e TRANSFORMERS_CACHE=/hostcache/transformers \
  -e HF_HOME=/hostcache/huggingface \
  -e VLA_LOG_DIR=/app/open-pi-zero/logs \
  -e VLA_WANDB_ENTITY=none \
  -w /app/open-pi-zero $IMG"

echo "=== [5] SAM3 robot-mask substitution (10 seeds, CGVD arm only) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py \
  --task widowx_spoon_on_towel --checkpoint_path checkpoints/bridge_beta.pt \
  --use_bf16 --randomize_distractors --placement spread \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline \
  --cgvd_robot_mask_source sam3 \
  --output_dir logs/robotmask_sam3_n18
echo "=== [5] done rc=$? $(date) ==="
