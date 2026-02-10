#!/bin/bash
# Setup script for OpenPI environment
#
# This script creates a conda environment with OpenPI and all dependencies
# needed to run eval_openpi.py with SimplerEnv.
#
# Usage:
#   ./scripts/setup_openpi_env.sh
#
# After running, activate with:
#   conda activate openpi

set -e

ENV_NAME="openpi"
OPENPI_DIR="/home/ubuntu/openpi"

echo "=============================================="
echo "OpenPI Environment Setup"
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
    echo ">>> Creating conda environment: $ENV_NAME (Python 3.11 required by OpenPI)"
    conda create -n $ENV_NAME python=3.11 -y
fi

# Get conda base for activation
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo ""
echo ">>> Activating environment: $ENV_NAME"
conda activate $ENV_NAME

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo ""
    echo ">>> Installing uv package manager"
    pip install uv
fi

# Clone OpenPI if not present
if [[ ! -d "$OPENPI_DIR" ]]; then
    echo ""
    echo ">>> Cloning OpenPI repository"
    cd /home/ubuntu
    git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git
else
    echo ""
    echo ">>> OpenPI already cloned at $OPENPI_DIR"
    cd "$OPENPI_DIR"
    echo ">>> Updating submodules..."
    git submodule update --init --recursive
fi

# Install OpenPI
echo ""
echo ">>> Installing OpenPI"
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

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

# Add open-pi-zero to PYTHONPATH (don't pip install - Python version conflict)
echo ""
echo ">>> Adding open-pi-zero to PYTHONPATH"
# Create activation script to set PYTHONPATH
ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACTIVATE_DIR"
echo 'export PYTHONPATH="/home/ubuntu/open-pi-zero:$PYTHONPATH"' > "$ACTIVATE_DIR/open-pi-zero.sh"
export PYTHONPATH="/home/ubuntu/open-pi-zero:$PYTHONPATH"

# Install additional dependencies that might be missing
echo ""
echo ">>> Installing additional dependencies"
pip install imageio imageio-ffmpeg opencv-python pytest s3fs

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "To use OpenPI:"
echo "  conda activate openpi"
echo ""
echo "To run evaluation:"
echo "  ./scripts/clutter_eval/run_paired_eval.sh --model openpi --task widowx_carrot_on_plate --episodes 10"
echo ""
echo "Or directly:"
echo "  conda activate openpi"
echo "  python scripts/eval_openpi.py --task widowx_carrot_on_plate --model pi0_base --num_episodes 5"
echo ""
