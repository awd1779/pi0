#!/bin/bash
# Build the open-pi-zero-groot Docker image.
#
# Because Docker build context must contain:
#   - open-pi-zero/       (this repo)
#   - allenzren_SimplerEnv/  (sibling editable dependency)
#   - Isaac-GR00T/           (NVIDIA GR00T package)
#
# We create a temporary build context that copies both sibling dirs
# into the repo so the Dockerfile can COPY them.
#
# Usage:
#   cd /path/to/open-pi-zero
#   bash docker/build_groot.sh [--no-cache]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIMPLER_ENV="${PROJECT_DIR}/../allenzren_SimplerEnv"
ISAAC_GROOT="${PROJECT_DIR}/../Isaac-GR00T"

if [ ! -d "$SIMPLER_ENV" ]; then
    echo "ERROR: allenzren_SimplerEnv not found at: $SIMPLER_ENV"
    echo "Expected directory layout:"
    echo "  parent/"
    echo "    allenzren_SimplerEnv/"
    echo "    Isaac-GR00T/"
    echo "    open-pi-zero/"
    exit 1
fi

if [ ! -d "$ISAAC_GROOT" ]; then
    echo "ERROR: Isaac-GR00T not found at: $ISAAC_GROOT"
    echo "Expected directory layout:"
    echo "  parent/"
    echo "    allenzren_SimplerEnv/"
    echo "    Isaac-GR00T/"
    echo "    open-pi-zero/"
    exit 1
fi

echo "=== Building open-pi-zero-groot Docker image ==="
echo "Project:    $PROJECT_DIR"
echo "SimplerEnv: $SIMPLER_ENV"
echo "Isaac-GR00T: $ISAAC_GROOT"

# Temporarily copy sibling dirs into build context (excluding .git to save space)
TEMP_SIMPLER="${PROJECT_DIR}/allenzren_SimplerEnv"
TEMP_GROOT="${PROJECT_DIR}/Isaac-GR00T"

for dir in "$TEMP_SIMPLER" "$TEMP_GROOT"; do
    if [ -e "$dir" ]; then
        echo "Cleaning up previous build artifact: $dir"
        rm -rf "$dir"
    fi
done

echo "Copying SimplerEnv into build context (excluding .git)..."
rsync -a --exclude='.git' "$SIMPLER_ENV/" "$TEMP_SIMPLER/"

echo "Copying Isaac-GR00T into build context (excluding .git)..."
rsync -a --exclude='.git' "$ISAAC_GROOT/" "$TEMP_GROOT/"

# Cleanup on exit
cleanup() {
    echo "Cleaning up temporary copies..."
    rm -rf "$TEMP_SIMPLER" "$TEMP_GROOT"
}
trap cleanup EXIT

# Build
echo "Starting Docker build..."
docker build \
    -t open-pi-zero-groot \
    -f "$PROJECT_DIR/Dockerfile.groot" \
    "$@" \
    "$PROJECT_DIR"

echo ""
echo "=== Build complete ==="
echo "Image: open-pi-zero-groot"
echo ""
echo "Run with:"
echo "  docker run --gpus all -it \\"
echo "    -v /path/to/logs:/workspace/logs \\"
echo "    open-pi-zero-groot"
