"""Attention capture utilities for Pi0 and GR00T visualization."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AttentionCapture:
    """Lightweight attention capture for batch evaluation."""

    def __init__(self, model: nn.Module, num_image_tokens: int = 256):
        self.model = model
        self.num_image_tokens = num_image_tokens
        self.attention_weights: List[Dict] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._q_cache: Dict[str, torch.Tensor] = {}
        self._k_cache: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self):
        """Hook into VLM mixture attention layers."""
        if not hasattr(self.model, 'joint_model'):
            print("[AttentionCapture] Model has no joint_model attribute")
            return

        mixtures = self.model.joint_model.mixtures
        if 'vlm' not in mixtures:
            print("[AttentionCapture] No 'vlm' in mixtures")
            return

        vlm = mixtures['vlm']
        # Store Q and K outputs per layer for later combination
        self._q_cache = {}
        self._k_cache = {}

        for layer_idx, layer in enumerate(vlm.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                # Hook q_proj and k_proj Linear layers directly
                if hasattr(attn, 'q_proj'):
                    hook_q = attn.q_proj.register_forward_hook(
                        self._create_proj_hook(f"vlm.layer{layer_idx}", "q", attn)
                    )
                    self.hooks.append(hook_q)
                if hasattr(attn, 'k_proj'):
                    hook_k = attn.k_proj.register_forward_hook(
                        self._create_proj_hook(f"vlm.layer{layer_idx}", "k", attn)
                    )
                    self.hooks.append(hook_k)

        print(f"[AttentionCapture] Hooked {len(self.hooks)} projection layers")

    def _create_proj_hook(self, layer_name: str, proj_type: str, attn_module):
        """Create hook for Q or K projection layer."""
        def hook(module, inputs, outputs):
            try:
                # outputs is the projected tensor [B, S, H*D]
                if proj_type == "q":
                    self._q_cache[layer_name] = outputs.detach()
                elif proj_type == "k":
                    self._k_cache[layer_name] = outputs.detach()

                # When we have both Q and K for this layer, compute attention
                if layer_name in self._q_cache and layer_name in self._k_cache:
                    q = self._q_cache.pop(layer_name)
                    k = self._k_cache.pop(layer_name)

                    bsz, seq_len, _ = q.shape
                    num_heads = attn_module.num_heads
                    num_kv_heads = attn_module.num_key_value_heads
                    head_dim = attn_module.head_dim

                    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
                    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

                    # Expand K to match Q heads (GQA)
                    if num_kv_heads != num_heads:
                        num_groups = num_heads // num_kv_heads
                        k = k.unsqueeze(2).expand(bsz, num_kv_heads, num_groups, seq_len, head_dim)
                        k = k.reshape(bsz, num_heads, seq_len, head_dim)

                    attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                    attn = torch.softmax(attn, dim=-1)
                    self.attention_weights.append({
                        'layer': layer_name,
                        'weights': attn.detach().cpu().float(),  # Convert bf16 -> float32
                    })
            except Exception as e:
                print(f"[AttentionCapture] Hook error in {layer_name}.{proj_type}: {e}")
        return hook

    def get_attention_map(self, aggregate: str = 'mean') -> Optional[np.ndarray]:
        """Get aggregated attention from last token to image tokens."""
        if not self.attention_weights:
            return None

        all_attn = []
        for layer_data in self.attention_weights:
            attn = layer_data['weights']
            try:
                # Last query token attending to image tokens
                action_to_img = attn[:, :, -1, :self.num_image_tokens]
                all_attn.append(action_to_img)
            except IndexError:
                continue

        if not all_attn:
            return None

        stacked = torch.stack(all_attn, dim=0)
        if aggregate == 'mean':
            agg = stacked.mean(dim=(0, 2))
        elif aggregate == 'last':
            agg = stacked[-1].mean(dim=1)
        else:
            agg = stacked.mean(dim=(0, 2))

        return agg[0].float().numpy()

    def save_attention_map(self, image: np.ndarray, save_path: str, aggregate: str = 'mean'):
        """Save attention heatmap overlay."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        attn = self.get_attention_map(aggregate)
        if attn is None:
            return False

        # Reshape to 16x16 grid
        grid_size = int(np.sqrt(len(attn)))
        attn_map = attn[:grid_size * grid_size].reshape(grid_size, grid_size)

        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Prepare image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = PILImage.fromarray(image)

        # Resize attention
        attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8))
        attn_resized = np.array(attn_pil.resize(pil_image.size, PILImage.BILINEAR)).astype(np.float32) / 255.0

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(pil_image)
        axes[0].set_title('Input')
        axes[0].axis('off')

        im = axes[1].imshow(attn_map, cmap='jet')
        axes[1].set_title(f'Attention ({grid_size}x{grid_size})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(pil_image)
        axes[2].imshow(attn_resized, alpha=0.5, cmap='jet')
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return True

    def clear(self):
        """Clear captured attention."""
        self.attention_weights = []
        self._q_cache = {}
        self._k_cache = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class GR00TAttentionCapture:
    """Attention capture for GR00T N1.6 DiT cross-attention layers.

    Hooks into AlternateVLDiT transformer blocks that attend to image tokens
    (from Eagle backbone) to visualize where the model focuses in the input image.

    Architecture path:
        model.policy.model.action_head.model -> AlternateVLDiT
        AlternateVLDiT.transformer_blocks -> BasicTransformerBlock[]
        BasicTransformerBlock.attn1 -> diffusers Attention (to_q, to_k)

    Block layout (interleaved self-attention):
        Even blocks: cross-attention (action queries -> VL encoder keys)
        Odd blocks: self-attention (action tokens only)
    Cross-attention alternates via attend_text_every_n_blocks:
        idx % (2 * N) == 0 -> text tokens
        idx % (2 * N) != 0 -> image tokens
    """

    def __init__(self, model, image_size: Tuple[int, int] = (256, 256)):
        """
        Args:
            model: GR00TInference instance
            image_size: (W, H) of input image for reshaping attention grid
        """
        self.model = model
        self.image_size = image_size
        self.attention_weights: List[Dict] = []
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._q_cache: Dict[str, torch.Tensor] = {}
        self._k_cache: Dict[str, torch.Tensor] = {}
        self._image_mask: Optional[torch.Tensor] = None
        self._num_hooked_blocks: int = 0
        self._register_hooks()

    def _register_hooks(self):
        """Hook into DiT cross-attention blocks that attend to image tokens."""
        try:
            dit = self.model.policy.model.action_head.model
            self.transformer_blocks = dit.transformer_blocks
            attend_text_every_n = getattr(dit, 'attend_text_every_n_blocks', 2)
        except AttributeError as e:
            print(f"[GR00TAttentionCapture] Cannot access DiT transformer blocks: {e}")
            return

        # Hook only image-attending cross-attention blocks
        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 0:  # Even = cross-attention
                # Image-attending: idx % (2 * attend_text_every_n) != 0
                if idx % (2 * attend_text_every_n) != 0:
                    attn = block.attn1
                    hook_q = attn.to_q.register_forward_hook(
                        self._create_proj_hook(f"dit.block{idx}", "q", attn)
                    )
                    hook_k = attn.to_k.register_forward_hook(
                        self._create_proj_hook(f"dit.block{idx}", "k", attn)
                    )
                    self.hooks.extend([hook_q, hook_k])
                    self._num_hooked_blocks += 1

        # Monkey-patch AlternateVLDiT.forward to capture image_mask
        original_forward = dit.forward
        capture_ref = self

        def patched_forward(*args, **kwargs):
            if 'image_mask' in kwargs and kwargs['image_mask'] is not None:
                capture_ref._image_mask = kwargs['image_mask'].detach().cpu()
            return original_forward(*args, **kwargs)

        dit.forward = patched_forward

        print(f"[GR00TAttentionCapture] Hooked {self._num_hooked_blocks} image cross-attention "
              f"blocks ({len(self.hooks)} projection layers)")

    def _create_proj_hook(self, layer_name: str, proj_type: str, attn_module):
        """Create hook for Q or K projection in a diffusers Attention module."""
        def hook(module, inputs, outputs):
            try:
                if proj_type == "q":
                    self._q_cache[layer_name] = outputs.detach()
                elif proj_type == "k":
                    self._k_cache[layer_name] = outputs.detach()

                # When we have both Q and K for this layer, compute attention
                if layer_name in self._q_cache and layer_name in self._k_cache:
                    q = self._q_cache.pop(layer_name)
                    k = self._k_cache.pop(layer_name)

                    num_heads = attn_module.heads
                    head_dim = attn_module.inner_dim // num_heads

                    bsz, q_len, _ = q.shape
                    _, k_len, _ = k.shape
                    q = q.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
                    k = k.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)

                    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

                    # Apply image_mask before softmax for accurate normalization
                    if self._image_mask is not None:
                        mask = self._image_mask[0].bool().to(attn_scores.device)
                        # Mask out non-image tokens (set to -inf)
                        attn_scores[:, :, :, ~mask] = float('-inf')

                    attn = torch.softmax(attn_scores, dim=-1)
                    self.attention_weights.append({
                        'layer': layer_name,
                        'weights': attn.detach().cpu().float(),
                    })
            except Exception as e:
                print(f"[GR00TAttentionCapture] Hook error in {layer_name}.{proj_type}: {e}")
        return hook

    def get_last_step_attention(self) -> List[Dict]:
        """Get attention weights from the last denoising step only.

        The DiT runs N denoising steps, each triggering all hooked blocks.
        The last step produces the most refined attention.
        """
        if not self.attention_weights or self._num_hooked_blocks == 0:
            return []
        return self.attention_weights[-self._num_hooked_blocks:]

    def get_attention_map(self, aggregate: str = 'mean') -> Optional[np.ndarray]:
        """Get aggregated attention from action tokens to image tokens.

        Averages across heads and image-attending cross-attention layers
        from the last denoising step.

        Returns:
            1D numpy array of attention weights per image token, or None.
        """
        last_step = self.get_last_step_attention()
        if not last_step:
            return None

        # Determine image token positions from captured mask
        if self._image_mask is not None:
            image_positions = self._image_mask[0].bool()
            num_image_tokens = image_positions.sum().item()
        else:
            print("[GR00TAttentionCapture] No image_mask captured")
            return None

        all_attn = []
        for layer_data in last_step:
            attn = layer_data['weights']  # [B, heads, q_len, k_len]
            try:
                attn_b0 = attn[0]  # [heads, q_len, k_len]
                # Average across all action query tokens
                attn_avg_q = attn_b0.mean(dim=1)  # [heads, k_len]
                # Select only image token columns
                attn_img = attn_avg_q[:, image_positions]  # [heads, num_image_tokens]
                all_attn.append(attn_img)
            except (IndexError, RuntimeError):
                continue

        if not all_attn:
            return None

        # Stack across layers and average across layers and heads
        stacked = torch.stack(all_attn, dim=0)  # [layers, heads, num_image_tokens]
        agg = stacked.mean(dim=(0, 1))  # [num_image_tokens]

        return agg.float().numpy()

    def save_attention_map(self, image: np.ndarray, save_path: str, aggregate: str = 'mean') -> bool:
        """Save 3-panel attention visualization: Input | Attention Grid | Overlay.

        Args:
            image: Input RGB image (H, W, 3) uint8
            save_path: Output PNG path
            aggregate: Aggregation method ('mean')

        Returns:
            True if saved successfully, False otherwise.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage

        attn = self.get_attention_map(aggregate)
        if attn is None:
            return False

        # Compute grid dimensions from token count
        num_tokens = len(attn)
        grid_size = int(math.sqrt(num_tokens))
        if grid_size * grid_size == num_tokens:
            grid_h = grid_w = grid_size
        else:
            # Non-square: find closest factorization
            grid_h = grid_size
            grid_w = grid_size
            for g in range(grid_size, 0, -1):
                if num_tokens % g == 0:
                    grid_h, grid_w = g, num_tokens // g
                    break

        attn_map = attn[:grid_h * grid_w].reshape(grid_h, grid_w)

        # Normalize to [0, 1]
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        # Prepare image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        pil_image = PILImage.fromarray(image)

        # Resize attention map to image size
        attn_pil = PILImage.fromarray((attn_map * 255).astype(np.uint8))
        attn_resized = np.array(
            attn_pil.resize(pil_image.size, PILImage.BILINEAR)
        ).astype(np.float32) / 255.0

        # Create 3-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(pil_image)
        axes[0].set_title('Input')
        axes[0].axis('off')

        im = axes[1].imshow(attn_map, cmap='jet')
        axes[1].set_title(f'Attention ({grid_h}x{grid_w})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        axes[2].imshow(pil_image)
        axes[2].imshow(attn_resized, alpha=0.5, cmap='jet')
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return True

    def clear(self):
        """Clear captured attention weights between inference calls."""
        self.attention_weights = []
        self._q_cache = {}
        self._k_cache = {}

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
