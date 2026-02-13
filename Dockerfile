# ==============================================================================
# open-pi-zero Docker image for GPU evaluation
# Base: RunPod PyTorch (compatible with RunPod container runtime)
# Usage:
#   docker build -t open-pi-zero .
#   docker run --gpus all \
#     -v /path/to/checkpoints:/app/open-pi-zero/checkpoints \
#     -v /path/to/logs:/app/open-pi-zero/logs \
#     open-pi-zero ./scripts/clutter_eval/run_category_sweep_fast.sh
# ==============================================================================

FROM runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=all

# ---------- System packages ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    libxrandr2 \
    libjpeg-dev \
    libpng-dev \
    libvulkan1 \
    mesa-vulkan-drivers \
    tmux \
    unzip \
    vim \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# ---------- Vulkan / EGL config ----------
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
RUN mkdir -p /etc/vulkan/icd.d && \
    cp /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json

# ---------- Python 3.10 venv (base image has 3.10 + 3.12; we need 3.10) ----------
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/opt/venv"
RUN pip install --no-cache-dir --upgrade pip

# ---------- Source layout (under /app, safe from RunPod /workspace volume) ----------
#   /app/
#     allenzren_SimplerEnv/   (sibling, for editable install)
#     open-pi-zero/           (main project)
WORKDIR /app

# ---------- Install Python deps (frozen from pi0_fresh conda env) ----------
# Copy requirements first for Docker layer caching
COPY docker/requirements_pi0_fresh.txt /tmp/requirements_pi0_fresh.txt
RUN pip install --no-cache-dir -r /tmp/requirements_pi0_fresh.txt && \
    pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git@main"

# ---------- Copy SimplerEnv (editable dependency) ----------
COPY allenzren_SimplerEnv/ /app/allenzren_SimplerEnv/

# ---------- Copy project source ----------
COPY . /app/open-pi-zero/

WORKDIR /app/open-pi-zero

# ---------- Install editable packages ----------
RUN pip install --no-cache-dir -e /app/allenzren_SimplerEnv && \
    pip install --no-cache-dir -e /app/allenzren_SimplerEnv/ManiSkill2_real2sim && \
    pip install --no-cache-dir -e .

# ---------- Environment variables ----------
ENV TRANSFORMERS_CACHE=/workspace/cache/transformers
ENV VLA_LOG_DIR=/app/open-pi-zero/logs
ENV VLA_WANDB_ENTITY=none
ENV VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia

# ---------- Symlink checkpoints/logs to /workspace for persistence ----------
# RunPod mounts a volume at /workspace that persists across restarts.
# Symlink so checkpoints and logs survive pod restarts.
RUN rm -rf /app/open-pi-zero/checkpoints /app/open-pi-zero/logs && \
    mkdir -p /workspace/checkpoints /workspace/logs && \
    ln -sf /workspace/checkpoints /app/open-pi-zero/checkpoints && \
    ln -sf /workspace/logs /app/open-pi-zero/logs

# ---------- Entrypoint ----------
WORKDIR /app/open-pi-zero
CMD ["sleep", "infinity"]
