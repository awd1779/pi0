#!/bin/bash
# Setup script for GR00T N1.6 environment
#
# This script creates a conda environment with Isaac-GR00T and all dependencies
# needed to run eval_groot.py with SimplerEnv.
#
# Usage:
#   ./scripts/setup_groot_env.sh
#
# After running, activate with:
#   conda activate groot
#
# Note: GR00T requires a separate conda environment to avoid dependency
# conflicts with Pi0 (different transformers versions, etc.)

set -e

ENV_NAME="groot"
GROOT_DIR="/home/ubuntu/Isaac-GR00T"

echo "=============================================="
echo "GR00T N1.6 Environment Setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install conda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "Keeping existing environment. Will try to install missing packages."
    fi
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo ">>> Creating conda environment: $ENV_NAME (Python 3.10)"
    conda create -n $ENV_NAME python=3.10 -y
fi

# Get conda base for activation
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo ""
echo ">>> Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Clone Isaac-GR00T if not present
if [[ ! -d "$GROOT_DIR" ]]; then
    echo ""
    echo ">>> Cloning Isaac-GR00T repository"
    cd /home/ubuntu
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
fi

# Install Isaac-GR00T
echo ""
echo ">>> Installing Isaac-GR00T"
cd "$GROOT_DIR"
pip install -e .

# Install SimplerEnv
echo ""
echo ">>> Installing SimplerEnv"
if [[ -d "/home/ubuntu/allenzren_SimplerEnv" ]]; then
    pip install -e /home/ubuntu/allenzren_SimplerEnv
else
    echo "WARNING: SimplerEnv not found at /home/ubuntu/allenzren_SimplerEnv"
    echo "Please clone it manually and run: pip install -e /path/to/SimplerEnv"
fi

# Install ManiSkill2_real2sim
echo ""
echo ">>> Installing ManiSkill2_real2sim"
if [[ -d "/home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim" ]]; then
    pip install -e /home/ubuntu/allenzren_SimplerEnv/ManiSkill2_real2sim
else
    echo "WARNING: ManiSkill2_real2sim not found"
fi

# Install open-pi-zero for CGVD support
echo ""
echo ">>> Installing open-pi-zero (for CGVD wrapper)"
if [[ -d "/home/ubuntu/open-pi-zero" ]]; then
    pip install -e /home/ubuntu/open-pi-zero
else
    echo "WARNING: open-pi-zero not found at /home/ubuntu/open-pi-zero"
fi

# Install additional dependencies
echo ""
echo ">>> Installing additional dependencies"
pip install imageio imageio-ffmpeg opencv-python sentencepiece

# Check for transformers VideoInput issue and attempt to fix
echo ""
echo ">>> Checking transformers compatibility..."
TRANSFORMERS_VERSION=$(pip show transformers | grep Version | awk '{print $2}')
echo "Current transformers version: $TRANSFORMERS_VERSION"

# The VideoInput issue may occur with certain transformers versions
# Try upgrading to latest
echo ">>> Upgrading transformers to latest version..."
pip install --upgrade transformers

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To use GR00T:"
echo "  conda activate groot"
echo ""
echo "To run evaluation:"
echo "  ./run_groot_bridge_eval.sh widowx_carrot_on_plate 10"
echo "  ./run_groot_fractal_eval.sh google_robot_pick_horizontal_coke_can 10"
echo ""
echo "Or directly:"
echo "  conda activate groot"
echo "  python scripts/eval_groot.py --task widowx_carrot_on_plate --num_episodes 5"
echo ""
echo "=============================================="
echo "TROUBLESHOOTING"
echo "=============================================="
echo ""
echo "If you see 'VideoInput' import errors, try:"
echo "  1. conda activate groot"
echo "  2. pip install --upgrade transformers"
echo ""
echo "If issues persist, check the cached processor at:"
echo "  ~/.cache/huggingface/modules/transformers_modules/Eagle-Block2A-2B-v2/"
echo ""
