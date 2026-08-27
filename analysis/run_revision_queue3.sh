#!/bin/bash
# Stage 3 of the revision GPU queue — BYOVLA comparator (R31 / Phase 5).
# Vocabulary-matched reimplementation (see src/cgvd/byovla_intervention.py).
# Usage:
#   ./run_revision_queue3.sh smoke   # 1 seed x 2 episodes (validates + times)
#   ./run_revision_queue3.sh full    # 10 seeds x 20 episodes (~8-13 h)
set -u

MODE="${1:-smoke}"
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

if [ "$MODE" = "smoke" ]; then
  EPS=2; RUNS=1; OUT=logs/byovla_n18_smoke
else
  EPS=20; RUNS=10; OUT=logs/byovla_n18
fi

echo "=== [BYOVLA/$MODE] n18 semantic, $RUNS seeds x $EPS eps (BYOVLA arm only) $(date) ==="
$DR python scripts/clutter_eval/batch_eval.py \
  --task widowx_spoon_on_towel --checkpoint_path checkpoints/bridge_beta.pt \
  --use_bf16 --randomize_distractors --placement spread \
  --categories semantic --distractor_counts 18 \
  --episodes $EPS --runs $RUNS --start_seed 0 --skip_baseline \
  --eval_byovla --byovla_thresh 0.002 \
  --output_dir $OUT
echo "=== [BYOVLA/$MODE] done rc=$? $(date) ==="
