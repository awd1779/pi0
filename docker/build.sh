#!/bin/bash
# Build the open-pi-zero Docker image.
#
# Because Docker build context must contain both:
#   - open-pi-zero/  (this repo)
#   - allenzren_SimplerEnv/  (sibling editable dependency)
#
# We create a temporary build context that symlinks/copies SimplerEnv
# into the repo so the Dockerfile can COPY it.
#
# Usage:
#   cd /path/to/open-pi-zero
#   bash docker/build.sh [--no-cache]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIMPLER_ENV="${PROJECT_DIR}/../allenzren_SimplerEnv"

if [ ! -d "$SIMPLER_ENV" ]; then
    echo "ERROR: allenzren_SimplerEnv not found at: $SIMPLER_ENV"
    echo "Expected directory layout:"
    echo "  parent/"
    echo "    allenzren_SimplerEnv/"
    echo "    open-pi-zero/"
    exit 1
fi

echo "=== Building open-pi-zero Docker image ==="
echo "Project:    $PROJECT_DIR"
echo "SimplerEnv: $SIMPLER_ENV"

# Temporarily copy SimplerEnv into build context (excluding .git to save space)
TEMP_SIMPLER="${PROJECT_DIR}/allenzren_SimplerEnv"
if [ -e "$TEMP_SIMPLER" ]; then
    echo "Cleaning up previous build artifact..."
    rm -rf "$TEMP_SIMPLER"
fi

echo "Copying SimplerEnv into build context (excluding .git)..."
rsync -a --exclude='.git' "$SIMPLER_ENV/" "$TEMP_SIMPLER/"

# Cleanup on exit
cleanup() {
    echo "Cleaning up temporary SimplerEnv copy..."
    rm -rf "$TEMP_SIMPLER"
}
trap cleanup EXIT

# Build
echo "Starting Docker build..."
docker build \
    -t open-pi-zero \
    -f "$PROJECT_DIR/Dockerfile" \
    "$@" \
    "$PROJECT_DIR"

echo ""
echo "=== Build complete ==="
echo "Image: open-pi-zero"
echo ""
echo "Run with:"
echo "  docker run --gpus all -it \\"
echo "    -v /path/to/checkpoints:/workspace/open-pi-zero/checkpoints \\"
echo "    -v /path/to/logs:/workspace/open-pi-zero/logs \\"
echo "    open-pi-zero ./scripts/clutter_eval/run_category_sweep_fast.sh"
