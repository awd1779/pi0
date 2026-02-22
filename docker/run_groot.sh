#!/bin/bash
# Run the open-pi-zero-groot Docker container with GPU support.
#
# Usage:
#   bash docker/run_groot.sh                                    # interactive shell
#   bash docker/run_groot.sh ./scripts/clutter_eval/run_category_sweep_fast_groot.sh  # run sweep
#
# Environment variables:
#   CHECKPOINT_DIR  — path to checkpoints/ (default: ./checkpoints)
#   LOG_DIR         — path for output logs (default: ./logs)
#   CACHE_DIR       — path for HuggingFace/transformers cache (default: ~/.cache)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
CACHE_DIR="${CACHE_DIR:-${HOME}/.cache}"

# Validate checkpoint dir
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "WARNING: Checkpoint dir not found: $CHECKPOINT_DIR"
    echo "  Set CHECKPOINT_DIR=/path/to/checkpoints or place checkpoints in ./checkpoints/"
fi

# Create log dir if needed
mkdir -p "$LOG_DIR"

echo "=== Running open-pi-zero-groot container ==="
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Logs:        $LOG_DIR"
echo "Cache:       $CACHE_DIR"
echo ""

# If no command given, start interactive shell
if [ $# -eq 0 ]; then
    INTERACTIVE="-it"
    CMD=""
else
    INTERACTIVE="-it"
    CMD="-c"
    ARGS="$*"
fi

docker run --gpus all $INTERACTIVE \
    --shm-size=16g \
    -v "${CHECKPOINT_DIR}:/workspace/open-pi-zero/checkpoints:ro" \
    -v "${LOG_DIR}:/workspace/open-pi-zero/logs" \
    -v "${CACHE_DIR}:/workspace/cache:rw" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    open-pi-zero-groot \
    ${CMD:+$CMD "$ARGS"}
