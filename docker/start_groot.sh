#!/bin/bash

# --- Pull latest code from GitHub ---
echo "Updating code from GitHub..."
cd /app
git clone --depth 1 https://github.com/awd1779/pi0.git open-pi-zero-tmp 2>/dev/null
if [ -d open-pi-zero-tmp ]; then
    rsync -a --exclude='checkpoints' --exclude='logs' open-pi-zero-tmp/ open-pi-zero/
    rm -rf open-pi-zero-tmp
    echo "Code updated to latest."
else
    echo "WARNING: Could not clone repo, using baked-in code."
fi
cd /app/open-pi-zero

# --- Symlinks to /workspace (re-created every boot, volume mount breaks them) ---
mkdir -p /workspace/checkpoints /workspace/logs
rm -rf /app/open-pi-zero/checkpoints /app/open-pi-zero/logs
ln -sf /workspace/checkpoints /app/open-pi-zero/checkpoints
ln -sf /workspace/logs /app/open-pi-zero/logs
echo "Symlinks: checkpoints/ and logs/ -> /workspace"

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

# --- Start SAM3 server in background (uses separate venv with transformers git main) ---
echo "Starting SAM3 server..."
/opt/sam3_venv/bin/python scripts/sam3_server.py &
SAM3_PID=$!

# --- Wait for SAM3 server to be ready (up to 120s for model download on first boot) ---
echo "Waiting for SAM3 server..."
for i in $(seq 1 120); do
    if [ -f /tmp/sam3_server/ready ]; then
        echo "SAM3 server ready."
        break
    fi
    sleep 1
done

if [ ! -f /tmp/sam3_server/ready ]; then
    echo "WARNING: SAM3 server did not signal ready within 120s."
    echo "  Check logs: SAM3_PID=$SAM3_PID"
fi

exec sleep infinity
