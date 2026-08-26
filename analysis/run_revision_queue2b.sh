#!/bin/bash
# Stage 2b — corrections queue (run after stage 2 finishes):
#   Job 6  OOV cell A re-run with FROZEN override (the first attempt was
#          silently clobbered by CGVDWrapper.reset()'s per-episode
#          spawned-names re-derivation; fixed via freeze_distractor_names)
#   Job 7  OOV cell B re-run (same fix)
#   Job 8  Attribute SIMPLE arm, descriptive SAM3 target variant
#          (--cgvd_target "spoon with green handle") — hypothesis: the
#          original cloud run used the descriptive target; the project's
#          own lesson log says short targets ("green spoon") under-detect.
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

COMMON="--task widowx_spoon_on_towel --checkpoint_path checkpoints/bridge_beta.pt \
  --use_bf16 --randomize_distractors --placement spread"

echo "=== [6] OOV cell A re-run, frozen D=ladle,whisk,bowl,cup $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline \
  --cgvd_distractor_names "ladle,whisk,bowl,cup" \
  --output_dir logs/oov_n18_miss_v2
echo "=== [6] done rc=$? $(date) ==="

echo "=== [7] OOV cell B re-run, frozen generic vocabulary $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline \
  --cgvd_distractor_names "fork,knife,spatula,scissors,ladle,whisk,plate,bowl,cup,mug,pan,pot" \
  --output_dir logs/oov_n18_generic_v2
echo "=== [7] done rc=$? $(date) ==="

echo "=== [8] Attribute SIMPLE arm, descriptive SAM3 target $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories attribute --distractor_counts 0 1 2 3 4 \
  --episodes 20 --runs 5 --start_seed 0 \
  --prompt "put green spoon on towel" --cgvd_target "spoon with green handle" \
  --output_dir logs/attribute_spoon_simple_desc
echo "=== [8] done rc=$? $(date) ==="
