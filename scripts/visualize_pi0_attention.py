#!/usr/bin/env python3
"""
Visualize attention maps from local Pi0 VLA model.

This script extracts and visualizes where Pi0's action tokens attend to
in the input image, helping to understand attention leakage and model focus.

Pi0 architecture:
- Vision: SigLIP ViT → 256 image tokens (16x16 grid)
- Text: Tokenized instruction
- Proprio: Proprioception tokens
- Action: Flow-matching decoded action tokens

Block attention pattern:
- Image/text attend to themselves
- Proprio/action attend to image/text
- Action attends to proprio and itself

Usage:
    # With test image
    uv run python scripts/visualize_pi0_attention.py \
        --checkpoint checkpoints/bridge_beta.pt \
        --test \
        --output attention_map.png

    # With real image
    uv run python scripts/visualize_pi0_attention.py \
        --checkpoint checkpoints/bridge_beta.pt \
        --image path/to/image.png \
        --instruction "pick up the red block" \
        --output attention_map.png
"""

import argparse
import math
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf


class Pi0AttentionVisualizer:
    """Extract and visualize attention from Pi0's action tokens to image patches."""

    # Default number of image tokens (16x16 grid from SigLIP ViT)
    DEFAULT_NUM_IMAGE_TOKENS = 256

    def __init__(
        self,
        model: nn.Module,
        num_image_tokens: int = DEFAULT_NUM_IMAGE_TOKENS,
        target_mixtures: Optional[List[str]] = None,
    ):
        """
        Initialize the attention visualizer.

        Args:
            model: The PiZero model
            num_image_tokens: Number of image tokens (default 256 for 16x16 grid)
            target_mixtures: Which mixtures to hook ('vlm', 'action', 'proprio').
                           Default: ['vlm'] since that's where image tokens are processed.
        """
        self.model = model
        self.num_image_tokens = num_image_tokens
        self.target_mixtures = target_mixtures or ['vlm']
        self.attention_weights: List[Dict] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._model_info: Dict = {}

        self._analyze_model()
        self._register_hooks()

    def _analyze_model(self):
        """Analyze the model structure."""
        info = {
            'model_type': type(self.model).__name__,
            'mixtures': [],
            'attention_layers': [],
        }

        # Check for joint_model with mixtures
        if hasattr(self.model, 'joint_model') and hasattr(self.model.joint_model, 'mixtures'):
            mixtures = self.model.joint_model.mixtures
            info['mixtures'] = list(mixtures.keys())

            for mix_name, mixture in mixtures.items():
                for name, module in mixture.named_modules():
                    if 'self_attn' in name or 'attn' in name.lower():
                        info['attention_layers'].append({
                            'mixture': mix_name,
                            'name': name,
                            'type': type(module).__name__,
                            'has_q_proj': hasattr(module, 'q_proj'),
                        })

        self._model_info = info

    def _register_hooks(self):
        """Register forward hooks on attention layers to capture weights."""
        hooked = []

        if not hasattr(self.model, 'joint_model'):
            print("[Warning] Model doesn't have joint_model - can't hook attention")
            return

        mixtures = self.model.joint_model.mixtures

        for mix_name in self.target_mixtures:
            if mix_name not in mixtures:
                print(f"[Warning] Mixture '{mix_name}' not found")
                continue

            mixture = mixtures[mix_name]

            # Hook into each layer's self_attn
            for layer_idx, layer in enumerate(mixture.layers):
                if hasattr(layer, 'self_attn'):
                    attn_module = layer.self_attn
                    hook_name = f"{mix_name}.layer{layer_idx}.self_attn"
                    hook = attn_module.register_forward_hook(
                        self._create_attention_hook(hook_name, attn_module)
                    )
                    self.hooks.append(hook)
                    hooked.append(hook_name)

        self._model_info['hooked_layers'] = hooked
        print(f"[AttentionVisualizer] Hooked {len(hooked)} attention layers")

    def _create_attention_hook(self, layer_name: str, attn_module):
        """Create a hook that computes and captures attention weights."""
        def hook(module, inputs, outputs):
            try:
                # Get the input hidden states
                hidden_states = inputs[0] if isinstance(inputs, tuple) else inputs

                # Get Q, K projections
                q = module.q_proj(hidden_states)
                k = module.k_proj(hidden_states)

                bsz, seq_len, hidden_dim = q.shape

                # Get attention config
                num_heads = getattr(module, 'num_heads', 16)
                head_dim = hidden_dim // num_heads

                # Reshape: [B, L, H*D] -> [B, H, L, D]
                q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

                # Compute attention scores (without mask for visualization)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = torch.softmax(attn_weights, dim=-1)

                self.attention_weights.append({
                    'layer': layer_name,
                    'weights': attn_weights.detach().cpu(),
                    'seq_len': seq_len,
                })

            except Exception as e:
                print(f"[Warning] Failed to capture attention in {layer_name}: {e}")

        return hook

    def get_action_to_image_attention(
        self,
        action_token_idx: int = -1,
        aggregate: str = 'mean',
    ) -> np.ndarray:
        """
        Extract attention from action token to image tokens.

        Args:
            action_token_idx: Which action token (-1 = last/first predicted)
            aggregate: 'mean', 'max', or 'last' layer

        Returns:
            attention: [num_image_tokens] array
        """
        if not self.attention_weights:
            raise ValueError("No attention weights captured. Run inference first.")

        all_attn = []
        for layer_data in self.attention_weights:
            attn = layer_data['weights']  # [batch, heads, seq_q, seq_k]

            # Extract attention to image tokens (first num_image_tokens positions)
            try:
                action_to_image = attn[:, :, action_token_idx, :self.num_image_tokens]
                all_attn.append(action_to_image)
            except IndexError:
                continue

        if not all_attn:
            raise ValueError("Could not extract action→image attention")

        stacked = torch.stack(all_attn, dim=0)  # [layers, batch, heads, img_tokens]

        if aggregate == 'mean':
            aggregated = stacked.mean(dim=(0, 2))  # [batch, img_tokens]
        elif aggregate == 'max':
            aggregated = stacked.max(dim=0)[0].max(dim=1)[0]
        elif aggregate == 'last':
            aggregated = stacked[-1].mean(dim=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregate}")

        return aggregated[0].numpy()

    def visualize(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
        action_token_idx: int = -1,
        aggregate: str = 'mean',
        cmap: str = 'jet',
        alpha: float = 0.5,
        show: bool = True,
    ):
        """Create attention heatmap overlay on image."""
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        attention = self.get_action_to_image_attention(
            action_token_idx=action_token_idx,
            aggregate=aggregate,
        )

        # Reshape to 16x16 grid
        grid_size = int(np.sqrt(len(attention)))
        attn_map = attention[:grid_size * grid_size].reshape(grid_size, grid_size)

        # Prepare image
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_image = PILImage.fromarray(image)

        # Resize attention to image size
        attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_pil = PILImage.fromarray((attn_norm * 255).astype(np.uint8))
        attn_resized = np.array(
            attn_pil.resize(pil_image.size, PILImage.BILINEAR)
        ).astype(np.float32) / 255.0

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(pil_image)
        axes[0].set_title('Input Image')
        axes[0].axis('off')

        im = axes[1].imshow(attn_map, cmap=cmap)
        axes[1].set_title(f'Attention Map ({grid_size}x{grid_size})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(pil_image)
        axes[2].imshow(attn_resized, alpha=alpha, cmap=cmap)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def visualize_per_layer(
        self,
        image: np.ndarray,
        save_path: Optional[str] = None,
        num_layers: int = 4,
        cmap: str = 'jet',
        show: bool = True,
    ):
        """Visualize attention from multiple layers."""
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        if not self.attention_weights:
            raise ValueError("No attention weights captured")

        total_layers = len(self.attention_weights)
        layer_indices = np.linspace(0, total_layers - 1, num_layers, dtype=int)

        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        pil_image = PILImage.fromarray(image)

        fig, axes = plt.subplots(1, num_layers + 1, figsize=(4 * (num_layers + 1), 4))

        axes[0].imshow(pil_image)
        axes[0].set_title('Input')
        axes[0].axis('off')

        for i, layer_idx in enumerate(layer_indices):
            attn = self.attention_weights[layer_idx]['weights']
            layer_name = self.attention_weights[layer_idx]['layer']

            action_to_image = attn[:, :, -1, :self.num_image_tokens]
            avg_attn = action_to_image.mean(dim=1)[0].numpy()

            grid_size = int(np.sqrt(len(avg_attn)))
            attn_map = avg_attn[:grid_size * grid_size].reshape(grid_size, grid_size)
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

            attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8))
            attn_resized = np.array(
                attn_pil.resize(pil_image.size, PILImage.BILINEAR)
            ).astype(np.float32) / 255.0

            axes[i + 1].imshow(pil_image)
            axes[i + 1].imshow(attn_resized, alpha=0.5, cmap=cmap)
            axes[i + 1].set_title(f'Layer {layer_idx}\n({layer_name.split(".")[-1]})')
            axes[i + 1].axis('off')

        plt.tight_layout()

        if save_path:
            base, ext = os.path.splitext(save_path)
            per_layer_path = f"{base}_per_layer{ext}"
            plt.savefig(per_layer_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {per_layer_path}")

        if show:
            plt.show()

        return fig

    def get_model_info(self) -> Dict:
        """Return model structure info."""
        return self._model_info

    def clear(self):
        """Clear captured attention weights."""
        self.attention_weights = []

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def load_model(checkpoint_path: str, device: str = "cuda:0", use_bf16: bool = True):
    """Load Pi0 model from checkpoint."""
    from src.model.vla.pizero import PiZeroInference

    # Determine config based on checkpoint name
    if "fractal" in checkpoint_path:
        cfg_path = "config/eval/fractal_apple.yaml"
    else:
        cfg_path = "config/eval/bridge.yaml"

    print(f"[Model] Loading config: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)

    # Create model
    model = PiZeroInference(cfg, use_ddp=False)

    # Load checkpoint
    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    data["model"] = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}
    model.load_state_dict(data["model"], strict=True)

    # Setup
    dtype = torch.bfloat16 if use_bf16 else torch.float32
    model.to(dtype)
    model.to(device)
    model.eval()

    print(f"[Model] Loaded on {device} with dtype {dtype}")
    return model, cfg, dtype


def create_test_image() -> np.ndarray:
    """Create a test image with colored blocks."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)

    # Colored blocks
    image[20:80, 20:80] = [255, 0, 0]      # Red - upper left
    image[20:80, 144:204] = [0, 255, 0]    # Green - upper right
    image[82:142, 82:142] = [0, 0, 255]    # Blue - center
    image[144:204, 20:80] = [255, 255, 0]  # Yellow - lower left
    image[144:204, 144:204] = [0, 255, 255] # Cyan - lower right

    # Gray background
    mask = np.all(image == 0, axis=-1)
    image[mask] = [128, 128, 128]

    return image


def main():
    parser = argparse.ArgumentParser(description="Visualize Pi0 attention maps")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (e.g., checkpoints/bridge_beta.pt)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--instruction', type=str, default='pick up the object',
                        help='Language instruction')
    parser.add_argument('--output', type=str, default='attention_map.png',
                        help='Output path for visualization')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--test', action='store_true',
                        help='Use test image with colored blocks')
    parser.add_argument('--aggregate', type=str, default='mean',
                        choices=['mean', 'max', 'last'])
    parser.add_argument('--per-layer', action='store_true',
                        help='Also generate per-layer visualization')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display (only save)')
    parser.add_argument('--use-bf16', action='store_true', default=True)
    parser.add_argument('--mixtures', type=str, nargs='+', default=['vlm'],
                        choices=['vlm', 'action', 'proprio'],
                        help='Which mixtures to visualize attention from')

    args = parser.parse_args()

    # Load image
    if args.test:
        print("[Info] Using test image")
        image = create_test_image()
    elif args.image:
        from PIL import Image as PILImage
        print(f"[Info] Loading: {args.image}")
        image = np.array(PILImage.open(args.image).convert('RGB'))
        # Resize to 224x224 if needed
        if image.shape[:2] != (224, 224):
            pil_img = PILImage.fromarray(image)
            image = np.array(pil_img.resize((224, 224), PILImage.BILINEAR))
    else:
        print("[Error] Provide --image or use --test")
        return

    print(f"[Info] Image shape: {image.shape}")

    # Load model
    model, cfg, dtype = load_model(args.checkpoint, args.device, args.use_bf16)

    # Create visualizer
    print(f"[Info] Creating visualizer for mixtures: {args.mixtures}")
    visualizer = Pi0AttentionVisualizer(model, target_mixtures=args.mixtures)

    # Print model info
    info = visualizer.get_model_info()
    print(f"\n[Model Info]")
    print(f"  Type: {info.get('model_type')}")
    print(f"  Mixtures: {info.get('mixtures')}")
    print(f"  Hooked: {len(info.get('hooked_layers', []))} layers")

    # Prepare inputs using the processor
    from transformers import AutoTokenizer
    from src.model.vla.processing import VLAProcessor

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_path, padding_side="right"
    )
    num_image_tokens = cfg.vision.config.num_image_tokens
    processor = VLAProcessor(tokenizer, num_image_tokens, cfg.max_seq_len)

    # Process image and text
    image_tensor = torch.as_tensor(image.transpose(2, 0, 1)).unsqueeze(0)  # [1, 3, H, W]
    model_inputs = processor(text=[args.instruction], images=image_tensor)

    # Build masks and position IDs
    attention_mask = model_inputs["attention_mask"]
    causal_mask, vlm_pos, proprio_pos, action_pos = model.build_causal_mask_and_position_ids(
        attention_mask, dtype=dtype
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

    # Dummy proprio (zeros)
    proprio = torch.zeros(1, cfg.cond_steps, cfg.action_dim, dtype=dtype)

    inputs = {
        "input_ids": model_inputs["input_ids"].to(args.device),
        "pixel_values": model_inputs["pixel_values"].to(dtype).to(args.device),
        "image_text_proprio_mask": image_text_proprio_mask.to(args.device),
        "action_mask": action_mask.to(args.device),
        "vlm_position_ids": vlm_pos.to(args.device),
        "proprio_position_ids": proprio_pos.to(args.device),
        "action_position_ids": action_pos.to(args.device),
        "proprios": proprio.to(args.device),
    }

    # Run inference
    print(f"\n[Info] Running inference: '{args.instruction}'")
    visualizer.clear()

    with torch.inference_mode():
        actions = model(**inputs)

    print(f"[Info] Actions shape: {actions.shape}")
    print(f"[Info] Captured attention from {len(visualizer.attention_weights)} layers")

    if not visualizer.attention_weights:
        print("\n[Error] No attention captured!")
        visualizer.remove_hooks()
        return

    # Show first few layer shapes
    for i, layer_data in enumerate(visualizer.attention_weights[:3]):
        print(f"  Layer {i}: {layer_data['layer']}, shape: {list(layer_data['weights'].shape)}")
    if len(visualizer.attention_weights) > 3:
        print(f"  ... and {len(visualizer.attention_weights) - 3} more")

    # Visualize
    print(f"\n[Info] Creating visualization...")
    try:
        visualizer.visualize(
            image,
            save_path=args.output,
            aggregate=args.aggregate,
            show=not args.no_show,
        )

        if args.per_layer:
            visualizer.visualize_per_layer(
                image,
                save_path=args.output,
                show=not args.no_show,
            )
    except Exception as e:
        print(f"[Error] Visualization failed: {e}")
        import traceback
        traceback.print_exc()

    visualizer.remove_hooks()
    print("\n[Info] Done!")


if __name__ == "__main__":
    main()
