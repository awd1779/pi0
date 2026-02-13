# Deploying open-pi-zero on RunPod

## Overview

```
┌─ Local / AWS machine ────────────────────┐
│  1. Build Docker image                    │
│  2. Push to Docker Hub                    │
└───────────────┬──────────────────────────┘
                ▼
┌─ RunPod GPU Pod ─────────────────────────┐
│  3. Launch pod with your image            │
│  4. Download checkpoint from HuggingFace  │
│  5. Run evaluation sweep                  │
│  6. Download results                      │
└───────────────────────────────────────────┘
```

---

## Prerequisites

- Docker Hub account (https://hub.docker.com — free tier is fine)
- RunPod account with credits (https://www.runpod.io)
- This repo cloned with `allenzren_SimplerEnv` as a sibling directory:
  ```
  parent/
    allenzren_SimplerEnv/
    open-pi-zero/          ← you are here
  ```

---

## Step 1: Build the Docker image

```bash
cd ~/open-pi-zero
bash docker/build.sh
```

This will:
- Copy `allenzren_SimplerEnv` into the build context temporarily
- Build the image with RunPod base (CUDA 12.8.1, Ubuntu 22.04), Python 3.10 venv, and all dependencies
- Clean up the temporary copy when done

Build takes ~10-20 minutes depending on network speed.

---

## Step 2: Push to Docker Hub

```bash
# Log in to Docker Hub
docker login

# Tag the image (replace YOUR_USERNAME with your Docker Hub username)
docker tag open-pi-zero YOUR_USERNAME/open-pi-zero:latest

# Push (~5-8 GB, takes 10-20 min on a good connection)
docker push YOUR_USERNAME/open-pi-zero:latest
```

### Updating after code changes

If you modify the source code, rebuild and push:

```bash
bash docker/build.sh && \
  docker tag open-pi-zero YOUR_USERNAME/open-pi-zero:latest && \
  docker push YOUR_USERNAME/open-pi-zero:latest
```

Rebuilds are fast — Docker caches the dependency layers and only re-copies changed source files.

---

## Step 3: Launch a GPU pod on RunPod

1. Go to https://www.runpod.io/console/pods
2. Click **"+ GPU Pod"**
3. **Select GPU:**

   | GPU | VRAM | Spot Price | Notes |
   |-----|------|------------|-------|
   | RTX 4090 | 24 GB | ~$0.35-0.45/hr | Cheapest, should fit with bf16 |
   | A100 40GB | 40 GB | ~$1.0/hr | Safe choice |
   | A100 80GB | 80 GB | ~$1.5/hr | Most headroom |
   | H100 80GB | 80 GB | ~$2.5/hr | Fastest |

4. **Configure the pod:**
   - **Container Image:** `YOUR_USERNAME/open-pi-zero:latest`
   - **Volume Disk:** `20 GB`
   - **Network Volume:** (optional) Create a `50 GB` volume if you want checkpoints/results to persist across pod restarts. Mount at `/workspace`.
   - **Expose Ports:** Enable TCP port `22` if you want SSH access

5. Under **"Advanced"** (if available):
   - Ensure shared memory is at least `16 GB`

6. Click **"Deploy"**

---

## Step 4: Connect to the pod

From the RunPod console, click your pod → **"Connect"**:

- **Web Terminal** — works immediately in the browser
- **SSH** — use the provided SSH command:
  ```bash
  ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
  ```

---

## Step 5: Download the checkpoint

Inside the pod terminal:

```bash
mkdir -p /workspace/open-pi-zero/checkpoints
cd /workspace/open-pi-zero/checkpoints

# Bridge-Beta checkpoint (~11 GB)
wget https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_beta_step19296_2024-12-26_22-30_42.pt \
  -O bridge_beta.pt
```

Other available checkpoints:

```bash
# Bridge-Uniform
wget https://huggingface.co/allenzren/open-pi-zero/resolve/main/bridge_uniform_step19296_2024-12-26_22-31_42.pt \
  -O bridge_uniform.pt

# Fractal-Beta (for google_robot_* tasks)
wget https://huggingface.co/allenzren/open-pi-zero/resolve/main/fractal_beta_step29576_2024-12-29_13-10_42.pt \
  -O fractal_beta.pt

# Fractal-Uniform
wget https://huggingface.co/allenzren/open-pi-zero/resolve/main/fractal_uniform_step29576_2024-12-31_22-26_42.pt \
  -O fractal_uniform.pt
```

If you're using a **Network Volume**, the checkpoint persists across pod restarts — you only download once.

---

## Step 6: Run the evaluation sweep

```bash
cd /workspace/open-pi-zero

# Verify GPU is detected
nvidia-smi

# Verify checkpoint exists
ls -lh checkpoints/bridge_beta.pt

# Dry run first (no GPU needed, just prints what would run)
./scripts/clutter_eval/run_category_sweep_fast.sh --dry-run

# Quick smoke test (2 episodes, 1 run, no distractors)
./scripts/clutter_eval/run_category_sweep_fast.sh --episodes 2 --runs 1 --counts 0
```

### Full sweep (use tmux so it survives SSH disconnect)

```bash
# Start a tmux session
tmux new -s sweep

# Run the full sweep
./scripts/clutter_eval/run_category_sweep_fast.sh

# Detach from tmux: press Ctrl+B, then D
# Reattach later:   tmux attach -t sweep
```

### Sweep options

```bash
# Custom categories and distractor counts
./scripts/clutter_eval/run_category_sweep_fast.sh \
  --categories semantic,visual \
  --counts 0,1,3,5

# Save video recordings
./scripts/clutter_eval/run_category_sweep_fast.sh --recording

# Save CGVD debug images
./scripts/clutter_eval/run_category_sweep_fast.sh --cgvd_save_debug

# Different task
./scripts/clutter_eval/run_category_sweep_fast.sh --task widowx_spoon_on_towel
```

---

## Step 7: Download results

Results are saved to `logs/clutter_eval/pi0/` inside the pod.

### Option A: runpodctl (peer-to-peer, easiest)

On the **RunPod pod:**
```bash
cd /workspace/open-pi-zero
runpodctl send logs/
# Output: Code is: 8-galaxy-rocket-fish
```

On your **local machine:**
```bash
cd ~/open-pi-zero
runpodctl receive 8-galaxy-rocket-fish
```

Install runpodctl locally if needed:
```bash
wget -qO runpodctl https://github.com/runpod/runpodctl/releases/latest/download/runpodctl-linux-amd64
chmod +x runpodctl
```

### Option B: rsync over SSH

```bash
# From your local machine
rsync -avP root@<RUNPOD_IP>:<PORT>:/workspace/open-pi-zero/logs/ ~/open-pi-zero/logs/
```

### Option C: scp

```bash
scp -P <PORT> -r root@<RUNPOD_IP>:/workspace/open-pi-zero/logs/ ~/open-pi-zero/logs/
```

---

## Step 8: Clean up

1. **Stop the pod** when the sweep finishes (you're billed per-minute)
2. **Keep the Network Volume** if you plan to run more experiments (costs ~$0.07/GB/month)
3. **Terminate the pod** if done — Network Volume data persists independently

---

## Docker Image Details

- **Base image:** `runpod/base:1.0.3-cuda1281-ubuntu2204` (CUDA 12.8.1, Ubuntu 22.04)
- **Python:** 3.10 (in `/opt/venv`)
- **Key packages:** torch 2.5.0, tensorflow 2.15.0, transformers (git main), sapien 2.2.2
- **Baked in:** All source code, all Python deps, SimplerEnv, ManiSkill2, xvfb, Vulkan
- **NOT baked in:** Checkpoints (download via wget), HuggingFace model cache (downloaded on first run)

---

## Troubleshooting

### Out of Memory (OOM)

If you hit OOM on a 24GB GPU (RTX 4090):
```bash
# Reduce episode count
./scripts/clutter_eval/run_category_sweep_fast.sh --episodes 10 --runs 5

# Or switch to a larger GPU (A100 40/80GB)
```

### Vulkan / rendering errors

```bash
# Verify Vulkan is working
vulkaninfo 2>&1 | head -5

# If not, check NVIDIA driver
nvidia-smi
```

### Container can't find checkpoint

```bash
# If using Network Volume mounted at /workspace, symlink:
ln -sf /workspace/checkpoints /workspace/open-pi-zero/checkpoints

# Or pass checkpoint path explicitly:
./scripts/clutter_eval/run_category_sweep_fast.sh  # auto-detects from PROJECT_ROOT
```

### HuggingFace / transformers cache

First run will download SAM3 and other model weights (~2-3 GB). To persist cache across pod restarts:
```bash
export TRANSFORMERS_CACHE=/workspace/cache/transformers
mkdir -p /workspace/cache/transformers
```

---

## Cost Estimates

Full sweep: 3 categories x 6 distractor counts x 10 runs x 21 episodes = **3,780 episodes**

| GPU | Estimated Time | Spot Cost |
|-----|---------------|-----------|
| RTX 4090 | 8-15 hours | $3-7 |
| A100 80GB | 6-12 hours | $9-18 |
| H100 80GB | 4-8 hours | $10-20 |
