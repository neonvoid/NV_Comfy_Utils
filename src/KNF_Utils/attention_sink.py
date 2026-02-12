"""
Attention Sink Persistence for Cross-Chunk Consistency

Implements the "attention sink" concept from Context Forcing (arxiv 2602.06028):
the first few frames of a video act as identity anchors in the attention mechanism.
By persisting these frames' K/V cache entries across all chunks, every chunk maintains
direct attention access to the original video identity.

Architecture:
  - Uses ComfyUI's `optimized_attention_override` hook (via @wrap_attn decorator)
  - Captures K/V for the first N latent frames during chunk 0
  - Prepends cached K/V before attention computation in chunks 1+
  - Memory cost: ~4 latent frames of KV cache per tracked block (minimal)

RoPE Note:
  In Wan models, RoPE is applied BEFORE optimized_attention (in WanSelfAttention.forward).
  This means captured K tensors have chunk 0's temporal RoPE baked in. For chunks far
  from chunk 0, this creates a positional mismatch. The proper fix is bounded positional
  encoding (Context Forcing Eq. 8), which remaps all positions to [0, max_trained_pos]
  so that sink K/V always has valid positions. Until that's implemented, sink persistence
  still helps via V-channel identity anchoring (V is RoPE-free).

References:
  - Context Forcing Section 3.3: Context Management System
  - Context Forcing Equation 8: Bounded positional indexing
  - attention_guidance_implementation_plan.md: optimized_attention_override mechanism
"""

import os
import torch
from typing import Dict, Set, Optional, Tuple


class AttentionSinkManager:
    """
    Manages attention sink K/V cache across chunks.

    Lifecycle:
      1. Create manager with configuration
      2. For chunk 0: call set_mode("capture"), patch model, run sampling
         -> After sampling, sink_kv_cache is populated
      3. For chunk 1+: call set_mode("apply"), patch model, run sampling
         -> Sink K/V is prepended to each attention computation

    The manager creates an optimized_attention_override function that
    handles both capture and apply modes based on current state.
    """

    def __init__(self,
                 num_sink_latent_frames: int = 1,
                 target_blocks: Optional[Set[int]] = None,
                 capture_at_step: str = "last"):
        """
        Args:
            num_sink_latent_frames: Number of latent frames to capture as sinks.
                1 latent frame = 4 video frames. Default 1 captures first 4 video frames.
            target_blocks: Which transformer block indices to track.
                Default captures from 5 evenly-spaced blocks across the network.
                Wan 2.1 14B has 40 blocks; 1.3B has 30.
            capture_at_step: When to capture sinks during sampling.
                "last" = capture only at the final denoising step (cleanest signal)
                "first" = capture at step 0 (noisiest, but available earliest)
                int = capture at specific step number
        """
        self.num_sink_latent_frames = num_sink_latent_frames
        self.target_blocks = target_blocks or {0, 5, 10, 15, 19}
        self.capture_at_step = capture_at_step

        # State
        self.sink_kv_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._mode = "idle"  # "idle", "capture", "apply"
        self._current_step = [0]
        self._total_steps = [0]
        self._capture_complete = False
        self._apply_count = [0]

    def set_mode(self, mode: str):
        """Set operating mode: 'capture' for chunk 0, 'apply' for chunks 1+."""
        assert mode in ("idle", "capture", "apply"), f"Invalid mode: {mode}"
        self._mode = mode
        self._current_step[0] = 0
        self._apply_count[0] = 0

        if mode == "capture":
            self._capture_complete = False
            print(f"[AttentionSinkManager] Mode: CAPTURE "
                  f"(sink_frames={self.num_sink_latent_frames}, "
                  f"blocks={sorted(self.target_blocks)}, "
                  f"capture_at={self.capture_at_step})")
        elif mode == "apply":
            n_cached = len(self.sink_kv_cache)
            if n_cached == 0:
                print("[AttentionSinkManager] WARNING: No sinks cached! Run capture first.")
            else:
                # Report cache contents
                sample_block = next(iter(self.sink_kv_cache))
                k_shape = self.sink_kv_cache[sample_block][0].shape
                print(f"[AttentionSinkManager] Mode: APPLY "
                      f"({n_cached} blocks cached, K shape per block: {list(k_shape)})")

    def clear(self):
        """Clear all cached sinks. Call between videos."""
        self.sink_kv_cache.clear()
        self._mode = "idle"
        self._capture_complete = False
        self._current_step[0] = 0
        self._apply_count[0] = 0

    @property
    def has_sinks(self) -> bool:
        return len(self.sink_kv_cache) > 0

    def _should_capture_at_step(self, step: int) -> bool:
        """Check if we should capture sinks at this step."""
        if self._capture_complete:
            return False

        if self.capture_at_step == "last":
            # Capture at the final step (cleanest denoised signal)
            return step >= self._total_steps[0] - 1
        elif self.capture_at_step == "first":
            return step == 0
        elif isinstance(self.capture_at_step, int):
            return step == self.capture_at_step
        else:
            # Parse percentage like "80%"
            if isinstance(self.capture_at_step, str) and self.capture_at_step.endswith("%"):
                pct = int(self.capture_at_step[:-1])
                target = int(self._total_steps[0] * pct / 100)
                return step == target
        return False

    def _extract_sink_tokens(self, tensor: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Extract the first N latent frames worth of tokens from a sequence tensor.

        In Wan, the sequence dimension contains tokens arranged as:
        [frame_0_spatial..., frame_1_spatial..., ..., frame_T_spatial...]

        We need to figure out how many tokens correspond to num_sink_latent_frames.
        Since we don't know spatial_size a priori, we estimate it from the total
        sequence length and the expected number of latent frames.

        Args:
            tensor: Shape [B, S, D] where S = num_latent_frames * spatial_tokens
            seq_dim: Which dimension is the sequence (default 1)

        Returns:
            Sliced tensor with only sink frame tokens
        """
        seq_len = tensor.shape[seq_dim]

        # We don't know exact spatial_size, so we use a heuristic:
        # For typical Wan resolutions, spatial_size = H/2 * W/2 (after 2x spatial patching)
        # e.g., 480x832 -> 240x416 -> spatial = 99,840 tokens... that's too many
        # Actually: Wan patches are 2x2, so spatial = (H/patch) * (W/patch)
        # For 480x832 with patch_size=2: 240 * 416 = 99840... no, that's huge
        # The actual Wan patchification: embed_dim patches of size patch_size
        # Let's just use the proportion: if we have T latent frames and S total tokens,
        # tokens_per_frame = S / T. But we don't know T here either.
        #
        # Better approach: just take the first `ratio` of tokens where ratio =
        # num_sink_latent_frames / total_latent_frames. We'll set total_latent_frames
        # during capture from the actual latent shape.
        #
        # For now, store the sink token count from the first capture call.
        return tensor

    def create_override(self):
        """
        Create the optimized_attention_override function.

        This function is set on model_options["transformer_options"]["optimized_attention_override"]
        and intercepts ALL attention computations in the model.

        Returns:
            Callable that matches the optimized_attention_override signature:
            (original_func, q, k, v, heads, **kwargs) -> output
        """
        manager = self
        sink_token_count = {}  # block_idx -> num tokens to capture (set dynamically)

        def attention_sink_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]
            if len(args) > 3:
                heads = args[3]
            else:
                heads = kwargs.get("heads", 1)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)

            # Skip blocks we're not tracking
            if block_idx not in manager.target_blocks:
                return original_func(*args, **kwargs)

            # === CAPTURE MODE ===
            if manager._mode == "capture":
                step = manager._current_step[0]

                if manager._should_capture_at_step(step) and block_idx not in manager.sink_kv_cache:
                    # Determine how many tokens to capture
                    # We need to estimate tokens_per_latent_frame from the sequence length
                    seq_len = k.shape[1]  # [B, S, D]

                    if block_idx not in sink_token_count:
                        # First time seeing this block at capture step
                        # We'll capture a proportion of the sequence
                        # Heuristic: for Wan, total_latent_frames is typically known
                        # from the latent tensor shape. We'll use a configurable
                        # fraction as fallback.
                        total_latent_frames = t_opts.get("_sink_total_latent_frames", None)
                        if total_latent_frames and total_latent_frames > 0:
                            tokens_per_frame = seq_len // total_latent_frames
                            n_tokens = manager.num_sink_latent_frames * tokens_per_frame
                        else:
                            # Fallback: assume ~20 latent frames for 81 video frames
                            # Take first 1/20th of tokens per sink latent frame
                            estimated_frames = 20
                            tokens_per_frame = seq_len // estimated_frames
                            n_tokens = manager.num_sink_latent_frames * tokens_per_frame

                        n_tokens = min(n_tokens, seq_len)
                        sink_token_count[block_idx] = n_tokens

                    n_tokens = sink_token_count[block_idx]

                    # Capture K and V for sink tokens
                    # K has RoPE baked in (from WanSelfAttention.forward)
                    # V is RoPE-free
                    k_sink = k[:, :n_tokens, :].detach().clone().cpu()
                    v_sink = v[:, :n_tokens, :].detach().clone().cpu()

                    manager.sink_kv_cache[block_idx] = (k_sink, v_sink)
                    print(f"[AttentionSink] Captured block {block_idx}: "
                          f"{n_tokens} tokens (of {seq_len}), "
                          f"K shape {list(k_sink.shape)}")

                    if len(manager.sink_kv_cache) == len(manager.target_blocks):
                        manager._capture_complete = True
                        total_bytes = sum(
                            kk.nbytes + vv.nbytes
                            for kk, vv in manager.sink_kv_cache.values()
                        )
                        print(f"[AttentionSink] Capture complete! "
                              f"{len(manager.sink_kv_cache)} blocks, "
                              f"{total_bytes / 1024 / 1024:.1f} MB total")

                # Always call original during capture (no modification)
                return original_func(*args, **kwargs)

            # === APPLY MODE ===
            elif manager._mode == "apply":
                if block_idx in manager.sink_kv_cache:
                    k_sink, v_sink = manager.sink_kv_cache[block_idx]

                    # Move to device
                    k_sink_dev = k_sink.to(device=k.device, dtype=k.dtype)
                    v_sink_dev = v_sink.to(device=v.device, dtype=v.dtype)

                    # Prepend sink K/V to current K/V
                    # Q stays unchanged (current chunk queries attend to sinks + current)
                    k_extended = torch.cat([k_sink_dev, k], dim=1)
                    v_extended = torch.cat([v_sink_dev, v], dim=1)

                    # Call original attention with extended K/V
                    # Q shape: [B, S_current, D]
                    # K shape: [B, S_sink + S_current, D]
                    # V shape: [B, S_sink + S_current, D]
                    # The attention will compute: softmax(Q @ K^T) @ V
                    # Each current query can attend to both sink tokens and current tokens
                    new_args = list(args)
                    new_args[1] = k_extended  # k
                    new_args[2] = v_extended  # v

                    manager._apply_count[0] += 1

                    return original_func(*new_args, **kwargs)

            # Default: pass through unchanged
            return original_func(*args, **kwargs)

        return attention_sink_override

    def get_step_callback(self, original_callback=None):
        """
        Create a sampling step callback that tracks the current step.

        Args:
            original_callback: Optional existing callback to chain with

        Returns:
            Callback function compatible with ComfyUI's sampler
        """
        manager = self

        def step_callback(step, x0, x, total_steps):
            manager._current_step[0] = step
            manager._total_steps[0] = total_steps
            if original_callback:
                return original_callback(step, x0, x, total_steps)

        return step_callback

    def get_cache_stats(self) -> dict:
        """Get statistics about the current cache state."""
        if not self.sink_kv_cache:
            return {"cached_blocks": 0, "total_bytes": 0, "total_mb": 0}

        total_bytes = sum(
            k.nbytes + v.nbytes
            for k, v in self.sink_kv_cache.values()
        )
        sample_block = next(iter(self.sink_kv_cache))
        k_shape = list(self.sink_kv_cache[sample_block][0].shape)

        return {
            "cached_blocks": len(self.sink_kv_cache),
            "block_indices": sorted(self.sink_kv_cache.keys()),
            "k_shape_per_block": k_shape,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1024 / 1024,
        }

    def to_serializable(self) -> dict:
        """Convert to a dict that can be saved with torch.save."""
        return {
            "sink_kv_cache": {
                block_idx: (k.cpu(), v.cpu())
                for block_idx, (k, v) in self.sink_kv_cache.items()
            },
            "config": {
                "num_sink_latent_frames": self.num_sink_latent_frames,
                "target_blocks": sorted(self.target_blocks),
                "capture_at_step": self.capture_at_step,
            },
            "stats": self.get_cache_stats(),
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "AttentionSinkManager":
        """Reconstruct from a saved dict."""
        config = data.get("config", {})
        manager = cls(
            num_sink_latent_frames=config.get("num_sink_latent_frames", 1),
            target_blocks=set(config.get("target_blocks", [0, 5, 10, 15, 19])),
            capture_at_step=config.get("capture_at_step", "last"),
        )
        for block_idx, (k, v) in data.get("sink_kv_cache", {}).items():
            manager.sink_kv_cache[int(block_idx)] = (k.cpu(), v.cpu())
        return manager


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class NV_AttentionSinkCapture:
    """
    Capture attention sinks from the first chunk's generation.

    Patches the model to capture K/V cache entries for the first N latent frames
    during sampling. These sinks can then be applied to subsequent chunks to
    maintain identity consistency.

    Use this on chunk 0. Connect the SINK_CACHE output to NV_AttentionSinkApply
    for subsequent chunks.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "num_sink_video_frames": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 32,
                    "tooltip": "Number of VIDEO frames to capture as sinks (converted to latent frames via 4:1)"
                }),
                "target_blocks": ("STRING", {
                    "default": "0,5,10,15,19",
                    "tooltip": "Transformer block indices to capture. Comma-separated. "
                               "Wan 14B has 40 blocks, 1.3B has 30."
                }),
                "capture_at_step": ("STRING", {
                    "default": "last",
                    "tooltip": "When to capture: 'last' (cleanest), 'first' (earliest), "
                               "or a step number, or '80%'"
                }),
                "total_latent_frames": ("INT", {
                    "default": 21,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Total latent frames in the generation (for token count calculation). "
                               "81 video frames = 21 latent frames."
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "SINK_CACHE")
    RETURN_NAMES = ("model", "sink_cache")
    FUNCTION = "capture"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = ("Capture attention sink K/V from chunk 0 for cross-chunk identity persistence. "
                   "Based on Context Forcing (arxiv 2602.06028).")

    def capture(self, model, num_sink_video_frames=4, target_blocks="0,5,10,15,19",
                capture_at_step="last", total_latent_frames=21):

        # Parse target blocks
        blocks = set()
        for x in target_blocks.split(","):
            try:
                blocks.add(int(x.strip()))
            except ValueError:
                pass

        # Convert video frames to latent frames
        num_sink_latent = max(1, num_sink_video_frames // 4)

        # Parse capture step
        try:
            cap_step = int(capture_at_step)
        except ValueError:
            cap_step = capture_at_step  # "last", "first", or "80%"

        # Create manager
        manager = AttentionSinkManager(
            num_sink_latent_frames=num_sink_latent,
            target_blocks=blocks,
            capture_at_step=cap_step,
        )
        manager.set_mode("capture")

        # Patch model
        patched_model = model.clone()
        if "transformer_options" not in patched_model.model_options:
            patched_model.model_options["transformer_options"] = {}

        # Store total_latent_frames hint for token count estimation
        patched_model.model_options["transformer_options"]["_sink_total_latent_frames"] = total_latent_frames

        # Set the override
        patched_model.model_options["transformer_options"]["optimized_attention_override"] = manager.create_override()

        # Store manager reference for step tracking
        patched_model._attention_sink_manager = manager

        print(f"[NV_AttentionSinkCapture] Model patched for sink capture")
        print(f"  Sink frames: {num_sink_video_frames} video / {num_sink_latent} latent")
        print(f"  Target blocks: {sorted(blocks)}")
        print(f"  Capture at step: {cap_step}")
        print(f"  Total latent frames hint: {total_latent_frames}")

        return (patched_model, manager)


class NV_AttentionSinkApply:
    """
    Apply captured attention sinks to a model for chunks 1+.

    Prepends the captured K/V cache entries before every attention computation,
    giving the model direct attention access to the original video's identity
    frames. This helps prevent character/style drift across chunks.

    Connect SINK_CACHE from NV_AttentionSinkCapture (after chunk 0 has run).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sink_cache": ("SINK_CACHE",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = ("Apply attention sinks from chunk 0 to maintain identity across chunks. "
                   "Based on Context Forcing (arxiv 2602.06028).")

    def apply(self, model, sink_cache):
        manager = sink_cache  # The SINK_CACHE type is an AttentionSinkManager instance

        if not manager.has_sinks:
            print("[NV_AttentionSinkApply] WARNING: No sinks in cache! Was capture run?")
            return (model.clone(),)

        manager.set_mode("apply")

        # Patch model
        patched_model = model.clone()
        if "transformer_options" not in patched_model.model_options:
            patched_model.model_options["transformer_options"] = {}

        patched_model.model_options["transformer_options"]["optimized_attention_override"] = manager.create_override()
        patched_model._attention_sink_manager = manager

        stats = manager.get_cache_stats()
        print(f"[NV_AttentionSinkApply] Model patched with {stats['cached_blocks']} sink blocks "
              f"({stats['total_mb']:.1f} MB)")

        return (patched_model,)


class NV_SaveAttentionSinks:
    """Save captured attention sinks to disk for parallel chunk workflows."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sink_cache": ("SINK_CACHE",),
                "output_path": ("STRING", {
                    "default": "attention_sinks.pt",
                    "tooltip": "Path to save the sink cache"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "NV_Utils/attention"

    def save(self, sink_cache, output_path):
        manager = sink_cache

        if not manager.has_sinks:
            print("[NV_SaveAttentionSinks] WARNING: No sinks to save!")
            return (output_path,)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data = manager.to_serializable()
        torch.save(data, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        stats = manager.get_cache_stats()
        print(f"[NV_SaveAttentionSinks] Saved to {output_path}")
        print(f"  Blocks: {stats['cached_blocks']} ({stats['block_indices']})")
        print(f"  File size: {size_mb:.1f} MB")

        return (output_path,)


class NV_LoadAttentionSinks:
    """Load attention sinks from disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "attention_sinks.pt",
                    "tooltip": "Path to the saved sink cache"
                }),
            }
        }

    RETURN_TYPES = ("SINK_CACHE",)
    RETURN_NAMES = ("sink_cache",)
    FUNCTION = "load"
    CATEGORY = "NV_Utils/attention"

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attention sinks not found: {path}")

        data = torch.load(path, map_location='cpu', weights_only=False)
        manager = AttentionSinkManager.from_serializable(data)

        stats = manager.get_cache_stats()
        print(f"[NV_LoadAttentionSinks] Loaded from {path}")
        print(f"  Blocks: {stats['cached_blocks']} ({stats.get('block_indices', [])})")
        print(f"  K shape per block: {stats.get('k_shape_per_block', [])}")
        print(f"  Total: {stats['total_mb']:.1f} MB")

        return (manager,)


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_AttentionSinkCapture": NV_AttentionSinkCapture,
    "NV_AttentionSinkApply": NV_AttentionSinkApply,
    "NV_SaveAttentionSinks": NV_SaveAttentionSinks,
    "NV_LoadAttentionSinks": NV_LoadAttentionSinks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_AttentionSinkCapture": "NV Attention Sink Capture",
    "NV_AttentionSinkApply": "NV Attention Sink Apply",
    "NV_SaveAttentionSinks": "NV Save Attention Sinks",
    "NV_LoadAttentionSinks": "NV Load Attention Sinks",
}
