#!/bin/bash

# --- HuggingFace authentication (for gated models like SAM3) ---
HF_TOKEN_FILE="/workspace/cache/huggingface/token"
if [ -n "$HF_TOKEN" ]; then
    mkdir -p /workspace/cache/huggingface
    echo -n "$HF_TOKEN" > "$HF_TOKEN_FILE"
    echo "HF token written from environment variable."
elif [ -f "$HF_TOKEN_FILE" ]; then
    echo "HF token already at $HF_TOKEN_FILE"
else
    echo "WARNING: No HF_TOKEN set and no cached token found."
    echo "  Gated models (SAM3) will fail to download."
    echo "  Set HF_TOKEN in RunPod pod environment variables."
fi

# --- PaliGemma tokenizer (baked into image at /app/, copy to /workspace on first boot) ---
PALIGEMMA_DST="/workspace/cache/transformers/paligemma-3b-pt-224"
if [ ! -d "$PALIGEMMA_DST" ]; then
    echo "Copying PaliGemma tokenizer to /workspace..."
    mkdir -p /workspace/cache/transformers
    cp -r /app/paligemma-3b-pt-224 "$PALIGEMMA_DST"
    echo "PaliGemma tokenizer ready."
else
    echo "PaliGemma tokenizer already at $PALIGEMMA_DST"
fi

# --- Bridge-Beta checkpoint (~11 GB, downloaded from HuggingFace) ---
CKPT="/workspace/checkpoints/bridge_beta.pt"
if [ ! -f "$CKPT" ]; then
    echo "Downloading bridge_beta checkpoint..."
    mkdir -p /workspace/checkpoints
    wget -q --show-progress https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_beta_step19296_2024-12-26_22-30_42.pt -O "$CKPT"
    echo "Download complete."
else
    echo "Checkpoint already exists at $CKPT"
fi

exec sleep infinity
