#!/bin/bash
# Sequential GPU queue for the CGVD revision (single-GPU host, Docker image
# drodii/open-pi-zero:latest). Each job mirrors the paper runs' flags
# (--use_bf16, --randomize_distractors, placement=spread, thresholds 0.3/0.3/0.20
# via batch_eval defaults). Logs land under logs/ like the original runs.
#
#   Job 1  attribute Simple arm  (Table I missing arm)      ~3.5 h
#   Job 2  n18 replication + mask/erasure audit (R28/R29)   ~0.8 h
#   Job 3  OOV cell A: vocabulary-miss D override (R32)     ~0.7 h
#   Job 4  OOV cell B: generic-vocabulary D override (R32)  ~0.7 h
set -u  # (no -e: one failed job should not kill the queue)

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

echo "=== QUEUE START $(date) ==="

echo "=== [1/4] Attribute SIMPLE arm (5 counts x 5 seeds x 20 eps, both arms) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories attribute --distractor_counts 0 1 2 3 4 \
  --episodes 20 --runs 5 --start_seed 0 \
  --prompt "put green spoon on towel" --cgvd_target "green spoon" \
  --output_dir logs/attribute_spoon_simple
echo "=== [1/4] done rc=$? $(date) ==="

echo "=== [2/4] n18 semantic replication + erasure audit (10 seeds, CGVD arm only) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline --cgvd_save_debug \
  --output_dir logs/replication_masks_n18
echo "=== [2/4] done rc=$? $(date) ==="

echo "=== [3/4] OOV cell A: D misses the injected clutter (10 seeds, CGVD arm only) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline \
  --cgvd_distractor_names "ladle,whisk,bowl,cup" \
  --output_dir logs/oov_n18_miss
echo "=== [3/4] done rc=$? $(date) ==="

echo "=== [4/4] OOV cell B: generic tabletop vocabulary (10 seeds, CGVD arm only) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py $COMMON \
  --categories semantic --distractor_counts 18 \
  --episodes 20 --runs 10 --start_seed 0 --skip_baseline \
  --cgvd_distractor_names "fork,knife,spatula,scissors,ladle,whisk,plate,bowl,cup,mug,pan,pot" \
  --output_dir logs/oov_n18_generic
echo "=== [4/4] done rc=$? $(date) ==="

echo "=== QUEUE COMPLETE $(date) ==="
