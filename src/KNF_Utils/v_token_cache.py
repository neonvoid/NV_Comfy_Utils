"""
V-Token Capture/Inject System for Cross-Chunk Consistency

Captures V (Value) tokens from a small full-resolution pre-render of strategic frames
(window centers) and injects them into the chunked high-resolution refinement pass.
This gives each context window a pre-computed "ground truth" V representation for
its center frame, anchoring the spatial layout of repetitive background elements
that would otherwise shift between chunks.

Why V-only (not K):
  V has no RoPE — carries pure content with no positional encoding.
  K has temporal RoPE baked in, creating position collisions when injected
  at different temporal positions within a context window.

Architecture:
  Workflow A (pre-pass):
    [NV_VTokenCapture] → [KSampler on center frames] → [NV_VTokenSave] → .pt file
  Workflow B (chunked high-res):
    [NV_VTokenLoad] → [NV_VTokenInject] → [WAN Context Windows] → [KSampler]

Based on the AttentionSinkManager pattern from attention_sink.py.
"""

import os
import torch
from typing import Optional


def _parse_target_blocks(target_blocks_str):
    """Parse target_blocks string into a set of block indices or None for 'all'."""
    s = target_blocks_str.strip().lower()
    if s == "all" or s == "":
        return None  # None means all blocks
    try:
        return set(int(x.strip()) for x in s.split(",") if x.strip())
    except ValueError:
        print(f"[VTokenCache] WARNING: Could not parse target_blocks '{target_blocks_str}', using all blocks")
        return None


def _compute_window_centers(total_frames, context_length, context_overlap):
    """
    Calculate center frame indices for static standard context windows.
    Replicates create_windows_static_standard logic from context_windows.py.

    Args:
        total_frames: Total latent frames in the generation
        context_length: Frames per context window
        context_overlap: Overlap between adjacent windows

    Returns:
        Sorted list of unique center frame indices (latent space)
    """
    delta = context_length - context_overlap
    if delta <= 0:
        print(f"[VTokenCache] WARNING: context_overlap ({context_overlap}) >= context_length ({context_length}), "
              f"using delta=1")
        delta = 1

    centers = []
    for start_idx in range(0, total_frames, delta):
        ending = start_idx + context_length
        if ending >= total_frames:
            # Last window gets pulled back to fit
            final_start = max(0, start_idx - (ending - total_frames))
            center = final_start + context_length // 2
            centers.append(min(center, total_frames - 1))
            break
        center = start_idx + context_length // 2
        centers.append(center)

    result = sorted(set(centers))
    return result


def _latent_to_video_frames(latent_indices, total_video_frames):
    """
    Map latent frame indices to video frame indices for Wan's 3D VAE.

    Wan's temporal compression is 4:1 with special first frame:
      Latent 0 → Video frame 0 (1 frame)
      Latent k (k>0) → Video frames (k-1)*4+1 through k*4 (4 frames each)
    """
    video_frames = []
    for lat_idx in latent_indices:
        if lat_idx == 0:
            video_frames.append(0)
        else:
            start_vf = (lat_idx - 1) * 4 + 1
            end_vf = lat_idx * 4 + 1  # exclusive
            for vf in range(start_vf, min(end_vf, total_video_frames)):
                video_frames.append(vf)
    return video_frames


def _parse_capture_frames(capture_frames_str, total_latent_frames, context_length, context_overlap):
    """
    Parse capture_frames input. Returns list of latent frame indices.

    "auto" → compute window centers from context params
    "3,6,10,16" → explicit latent frame indices
    """
    s = capture_frames_str.strip().lower()
    if s == "auto":
        frames = _compute_window_centers(total_latent_frames, context_length, context_overlap)
        print(f"[VTokenCache] Auto window centers: {frames} "
              f"(total={total_latent_frames}, ctx_len={context_length}, overlap={context_overlap})")
        return frames
    try:
        frames = sorted(set(int(x.strip()) for x in s.split(",") if x.strip()))
        # Validate frame indices
        frames = [f for f in frames if 0 <= f < total_latent_frames]
        if not frames:
            print(f"[VTokenCache] WARNING: No valid frame indices in '{capture_frames_str}' "
                  f"(total_latent_frames={total_latent_frames})")
        return frames
    except ValueError:
        print(f"[VTokenCache] WARNING: Could not parse capture_frames '{capture_frames_str}', "
              f"falling back to auto")
        return _compute_window_centers(total_latent_frames, context_length, context_overlap)


# ============================================================================
# VTokenManager — Internal state manager
# ============================================================================

class VTokenManager:
    """
    Manages V-token capture and injection state across workflow nodes.

    Storage: (block_idx, frame_idx) → V tensor on CPU
    Each V tensor shape: [B, tokens_per_frame, D]
    """

    def __init__(self, target_blocks, capture_frames, strength):
        self._mode = "idle"  # "idle" | "capture" | "inject"
        self.target_blocks = target_blocks  # set of int or None (all)
        self.capture_frames = capture_frames  # list of global latent frame indices
        self.strength = strength

        # Storage: (block_idx, frame_idx) → V tensor on CPU
        self.v_cache = {}

        # Computed at capture time
        self.tokens_per_frame = None

        # Debug tracking
        self._capture_logged = False
        self._inject_count = 0

    def set_mode(self, mode):
        assert mode in ("idle", "capture", "inject"), f"Invalid mode: {mode}"
        self._mode = mode
        self._capture_logged = False
        self._inject_count = 0

        if mode == "capture":
            self.v_cache.clear()
            print(f"[VTokenManager] Mode: CAPTURE "
                  f"(frames={self.capture_frames}, "
                  f"blocks={'all' if self.target_blocks is None else sorted(self.target_blocks)}, "
                  f"strength={self.strength})")
        elif mode == "inject":
            n_entries = len(self.v_cache)
            if n_entries == 0:
                print("[VTokenManager] WARNING: No V tokens cached! Run capture first.")
            else:
                n_blocks = len(set(b for b, _ in self.v_cache.keys()))
                n_frames = len(set(f for _, f in self.v_cache.keys()))
                sample_key = next(iter(self.v_cache))
                v_shape = self.v_cache[sample_key].shape
                print(f"[VTokenManager] Mode: INJECT "
                      f"({n_entries} entries: {n_blocks} blocks × {n_frames} frames, "
                      f"V shape per entry: {list(v_shape)}, strength={self.strength})")

    @property
    def has_cache(self):
        return len(self.v_cache) > 0

    def get_cache_stats(self):
        if not self.v_cache:
            return {"entries": 0, "blocks": 0, "frames": 0, "total_bytes": 0, "total_mb": 0}
        blocks = sorted(set(b for b, _ in self.v_cache.keys()))
        frames = sorted(set(f for _, f in self.v_cache.keys()))
        total_bytes = sum(v.nbytes for v in self.v_cache.values())
        sample_key = next(iter(self.v_cache))
        return {
            "entries": len(self.v_cache),
            "blocks": len(blocks),
            "block_indices": blocks,
            "frames": len(frames),
            "frame_indices": frames,
            "v_shape_per_entry": list(self.v_cache[sample_key].shape),
            "total_bytes": total_bytes,
            "total_mb": total_bytes / 1024 / 1024,
        }

    def to_serializable(self):
        """Convert to a dict that can be saved with torch.save."""
        return {
            "v_cache": {
                f"{block_idx},{frame_idx}": v.cpu()
                for (block_idx, frame_idx), v in self.v_cache.items()
            },
            "config": {
                "target_blocks": sorted(self.target_blocks) if self.target_blocks is not None else "all",
                "capture_frames": self.capture_frames,
                "strength": self.strength,
                "tokens_per_frame": self.tokens_per_frame,
            },
            "stats": self.get_cache_stats(),
        }

    @classmethod
    def from_serializable(cls, data):
        """Reconstruct from a saved dict."""
        config = data.get("config", {})
        tb = config.get("target_blocks", "all")
        target_blocks = None if tb == "all" else set(tb)
        manager = cls(
            target_blocks=target_blocks,
            capture_frames=config.get("capture_frames", []),
            strength=config.get("strength", 1.0),
        )
        manager.tokens_per_frame = config.get("tokens_per_frame", None)
        for key_str, v in data.get("v_cache", {}).items():
            block_idx, frame_idx = key_str.split(",")
            manager.v_cache[(int(block_idx), int(frame_idx))] = v.cpu()
        return manager


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class NV_VTokenCapture:
    """
    Captures V tokens from a small full-resolution pre-render of strategic frames.

    Connect BEFORE a KSampler that renders only the center frames.
    The KSampler should use the same denoise/conditions as the chunked pass.

    Workflow A: [Model] → [NV_VTokenCapture] → [KSampler] → [NV_VTokenSave]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "capture_frames": ("STRING", {
                    "default": "auto",
                    "tooltip": "'auto' calculates window center frames from context params, "
                               "or comma-separated latent frame indices like '3,6,10,16'. "
                               "Latent frames = (video_frames - 1) / 4 + 1 for Wan."
                }),
                "total_video_frames": ("INT", {
                    "default": 65,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total video frames in the full generation (e.g. 65, 81). "
                               "Converted to latent frames internally using Wan formula: "
                               "(n-1)/4+1. 65 video = 17 latent, 81 video = 21 latent."
                }),
                "context_window_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 400,
                    "step": 4,
                    "tooltip": "Context window size in VIDEO frames. Wire directly from "
                               "NV Chunk Loader's context_window_size output. Must match "
                               "your WAN Context Windows node. Only used when "
                               "capture_frames='auto'. Converted to latent internally."
                }),
                "context_overlap": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 200,
                    "step": 4,
                    "tooltip": "Context window overlap in VIDEO frames. Wire directly from "
                               "NV Chunk Loader's context_overlap output. Must match your "
                               "WAN Context Windows node. Only used when "
                               "capture_frames='auto'. Converted to latent internally."
                }),
                "target_blocks": ("STRING", {
                    "default": "all",
                    "tooltip": "'all' for every transformer block, or comma-separated indices "
                               "like '0,5,10,15,19'. More blocks = better quality but more "
                               "disk/RAM (~15MB per block per frame at 1280p)."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Scale factor for injected V values at injection time. "
                               ">1 strengthens cached V influence, <1 weakens it. "
                               "Stored in cache, applied by NV_VTokenInject."
                }),
                "latent_prefix_frames": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Number of LATENT frames prepended before the actual video frames "
                               "(e.g., VACE reference frames from NV_VacePrePassReference). "
                               "Wire from the 'trim_latent' output of NV_VacePrePassReference. "
                               "The capture will skip this many frames at the start of the V tensor "
                               "to avoid capturing reference-region V tokens instead of center frames."
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "V_TOKEN_CACHE")
    RETURN_NAMES = ("model", "v_cache")
    FUNCTION = "capture"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Captures V tokens from strategic frames during a small full-resolution "
        "pre-render. Connect before a KSampler, then save with NV_VTokenSave. "
        "The inject node uses these tokens to anchor background consistency "
        "in the chunked high-res pass."
    )

    def capture(self, model, capture_frames, total_video_frames,
                context_window_size, context_overlap, target_blocks, strength,
                latent_prefix_frames=0):
        model = model.clone()
        blocks = _parse_target_blocks(target_blocks)

        # Convert video frames to latent frames (Wan VAE: (n-1)//4+1)
        total_latent_frames = max(((total_video_frames - 1) // 4) + 1, 1)
        latent_ctx_len = max(((context_window_size - 1) // 4) + 1, 1)
        latent_ctx_overlap = max(((context_overlap - 1) // 4) + 1, 0)
        print(f"[VTokenCapture] Video→Latent conversion: "
              f"total {total_video_frames}→{total_latent_frames}, "
              f"ctx_len {context_window_size}→{latent_ctx_len}, "
              f"overlap {context_overlap}→{latent_ctx_overlap}")

        frames = _parse_capture_frames(
            capture_frames, total_latent_frames, latent_ctx_len, latent_ctx_overlap
        )

        if not frames:
            print("[VTokenCapture] WARNING: No frames to capture. Returning unpatched model.")
            manager = VTokenManager(blocks, [], strength)
            return (model, manager)

        manager = VTokenManager(blocks, frames, strength)
        manager.set_mode("capture")

        # Store prefix offset for the capture closure
        prefix_offset = latent_prefix_frames
        if prefix_offset > 0:
            print(f"[VTokenCapture] Skipping {prefix_offset} prepended reference latent frames in V tensor")

        # Track which step we're on for capture-at-last-step logic
        # We capture on EVERY step (V changes each step), last step's V is final
        # But we only need to keep the last captured values — overwrite each step

        # Check for existing override to chain with
        existing_override = None
        if "transformer_options" in model.model_options:
            existing_override = model.model_options["transformer_options"].get(
                "optimized_attention_override"
            )

        def _call_next(original_func, args, kwargs):
            if existing_override is not None:
                return existing_override(original_func, *args, **kwargs)
            return original_func(*args, **kwargs)

        # Debug state
        debug_state = {
            "first_call": True,
            "captures_this_step": 0,
            "last_sigma": None,
        }

        def capture_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]

            # Skip cross-attention (Q and K have different seq_len)
            if q.shape[1] != k.shape[1]:
                return _call_next(original_func, args, kwargs)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)
            grid_sizes = t_opts.get("grid_sizes", None)
            sigmas = t_opts.get("sigmas", None)

            # Skip non-targeted blocks
            if blocks is not None and block_idx not in blocks:
                return _call_next(original_func, args, kwargs)

            # Detect new step via sigma change
            current_sigma = sigmas[0].item() if sigmas is not None and len(sigmas) > 0 else None
            if current_sigma is not None and current_sigma != debug_state["last_sigma"]:
                if debug_state["captures_this_step"] > 0:
                    print(f"[VTokenCapture] Step complete: captured {debug_state['captures_this_step']} blocks")
                debug_state["last_sigma"] = current_sigma
                debug_state["captures_this_step"] = 0

            # Need grid_sizes to compute tokens_per_frame
            if grid_sizes is None:
                if debug_state["first_call"]:
                    print("[VTokenCapture] WARNING: No grid_sizes in transformer_options")
                    debug_state["first_call"] = False
                return _call_next(original_func, args, kwargs)

            tokens_per_frame = grid_sizes[1] * grid_sizes[2]
            manager.tokens_per_frame = tokens_per_frame

            # First call: full state dump
            if debug_state["first_call"]:
                print(f"[VTokenCapture] === First capture call ===")
                print(f"[VTokenCapture] grid_sizes: {grid_sizes} "
                      f"(T={grid_sizes[0]}, H={grid_sizes[1]}, W={grid_sizes[2]})")
                print(f"[VTokenCapture] tokens_per_frame: {tokens_per_frame}")
                print(f"[VTokenCapture] V shape: {v.shape} "
                      f"(B={v.shape[0]}, seq_len={v.shape[1]}, dim={v.shape[2]})")
                print(f"[VTokenCapture] dtype: {v.dtype}, device: {v.device}")
                print(f"[VTokenCapture] Capturing frames: {frames}")
                expected_seq = len(frames) * tokens_per_frame
                print(f"[VTokenCapture] Expected seq_len: {len(frames)} frames × "
                      f"{tokens_per_frame} tokens = {expected_seq} "
                      f"(actual: {v.shape[1]}, prefix_offset={prefix_offset})")
                if current_sigma is not None:
                    print(f"[VTokenCapture] sigma: {current_sigma:.4f}")
                debug_state["first_call"] = False

            # Capture: slice V tensor per frame and store on CPU
            # During Pass 1.5, no context windows — V contains all strategic frames
            # Frame order matches the latent tensor order
            # prefix_offset skips prepended reference frames (e.g., from NV_VacePrePassReference)
            num_frames_in_tensor = v.shape[1] // tokens_per_frame

            for i, frame_idx in enumerate(frames):
                local_pos = i + prefix_offset  # skip reference prefix
                if local_pos >= num_frames_in_tensor:
                    break
                start = local_pos * tokens_per_frame
                end = start + tokens_per_frame
                if end > v.shape[1]:
                    break
                manager.v_cache[(block_idx, frame_idx)] = (
                    v[:, start:end, :].detach().clone().cpu()
                )

            debug_state["captures_this_step"] += 1

            # Run attention normally — capture doesn't modify anything
            return _call_next(original_func, args, kwargs)

        # Apply the override
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = capture_override

        block_info = f"blocks {target_blocks}" if blocks is not None else "all blocks"
        chain_info = " (chained with existing override)" if existing_override else ""
        print(f"[VTokenCapture] Patched model: {len(frames)} frames to capture, "
              f"{block_info}, strength={strength}{chain_info}")

        return (model, manager)


class NV_VTokenSave:
    """
    Saves captured V tokens to a .pt file on disk.

    Connect after the KSampler in Workflow A (capture pass).
    The saved file can be loaded in Workflow B by NV_VTokenLoad.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "v_cache": ("V_TOKEN_CACHE",),
                "output_path": ("STRING", {
                    "default": "v_token_cache.pt",
                    "tooltip": "File path for saving the V token cache. "
                               "Relative paths are relative to ComfyUI's working directory."
                }),
            },
            "optional": {
                "dep_latent": ("LATENT", {
                    "tooltip": "Connect from KSampler/VAEDecode output to force execution order. "
                               "This input is not used for data — it only ensures VTokenSave "
                               "runs AFTER the sampler populates the V cache."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "save"
    CATEGORY = "NV_Utils/attention"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Saves captured V tokens to a .pt file. Required when capture and "
        "inject happen in separate workflows."
    )

    def save(self, v_cache, output_path, dep_latent=None):
        manager = v_cache

        if not manager.has_cache:
            print("[VTokenSave] WARNING: No V tokens to save! Was capture run?")
            return (output_path,)

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        data = manager.to_serializable()
        torch.save(data, output_path)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        stats = manager.get_cache_stats()
        print(f"[VTokenSave] Saved to {output_path}")
        print(f"  Entries: {stats['entries']} ({stats['blocks']} blocks × {stats['frames']} frames)")
        print(f"  Frames: {stats.get('frame_indices', [])}")
        print(f"  Blocks: {stats.get('block_indices', [])}")
        print(f"  V shape per entry: {stats.get('v_shape_per_entry', [])}")
        print(f"  File size: {size_mb:.1f} MB")

        return (output_path,)


class NV_VTokenLoad:
    """
    Loads V token cache from a .pt file on disk.

    Use in Workflow B to load tokens saved by NV_VTokenSave in Workflow A.
    Connect output to NV_VTokenInject.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "v_token_cache.pt",
                    "tooltip": "Path to the saved V token cache file."
                }),
            },
        }

    RETURN_TYPES = ("V_TOKEN_CACHE",)
    RETURN_NAMES = ("v_cache",)
    FUNCTION = "load"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Loads V token cache from a .pt file saved by NV_VTokenSave. "
        "Connect output to NV_VTokenInject."
    )

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"V token cache not found: {path}")

        data = torch.load(path, map_location='cpu', weights_only=False)
        manager = VTokenManager.from_serializable(data)

        stats = manager.get_cache_stats()
        print(f"[VTokenLoad] Loaded from {path}")
        print(f"  Entries: {stats['entries']} ({stats['blocks']} blocks × {stats['frames']} frames)")
        print(f"  Frames: {stats.get('frame_indices', [])}")
        print(f"  Blocks: {stats.get('block_indices', [])}")
        print(f"  V shape per entry: {stats.get('v_shape_per_entry', [])}")
        print(f"  Total: {stats['total_mb']:.1f} MB")

        return (manager,)


class NV_VTokenInject:
    """
    Injects pre-captured V tokens into the chunked high-resolution pass.

    For each context window, looks up which cached frames overlap with the
    window's frame range and prepends those V tokens to the window's V tensor.
    This gives each window a pre-computed "ground truth" for its center frame.

    Connect BEFORE context windows and sampler in Workflow B:
    [NV_VTokenLoad] → [NV_VTokenInject] → [WAN Context Windows] → [KSampler]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "v_cache": ("V_TOKEN_CACHE",),
                "latent_prefix_frames": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1,
                                         "tooltip": "Number of prepended reference latent frames to skip "
                                                    "(e.g., from NV_VacePrePassReference trim_latent output). "
                                                    "Context window index_list includes these prefix frames "
                                                    "before actual video frames."}),
                "chunk_start_video_frame": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1,
                                            "tooltip": "Starting VIDEO frame index for this chunk "
                                                       "(wire from chunk loader start_frame output). "
                                                       "Used to map chunk-local indices to global latent "
                                                       "frame indices that match the V cache keys."}),
                "sigma_start": ("FLOAT", {
                    "default": 1000.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Start injecting at this sigma and below. 1000.0 = inject from "
                               "the very first step. Lower values skip early high-noise steps."
                }),
                "sigma_end": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.01,
                    "tooltip": "Stop injecting below this sigma. 0.0 = inject at all steps. "
                               "Raise to skip late steps where fine detail forms — prevents "
                               "temporal 'pop' artifacts at injected frame positions. "
                               "Try 0.3-0.5 to anchor spatial layout without contaminating detail."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "inject"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Injects pre-captured V tokens into the chunked high-res pass. "
        "Each context window gets its matching cached V tokens replaced in-place, "
        "anchoring background element positions. Use sigma_end to gate injection "
        "to early denoising steps only (prevents fine-detail artifacts)."
    )

    def inject(self, model, v_cache, latent_prefix_frames, chunk_start_video_frame,
               sigma_start, sigma_end):
        model = model.clone()
        manager = v_cache

        if not manager.has_cache:
            print("[VTokenInject] WARNING: No V tokens in cache! Was capture/load run?")
            return (model,)

        manager.set_mode("inject")

        # Compute chunk's global latent frame offset from video frame start
        # Wan VAE: latent 0 = 1 video frame, latent k>0 = 4 video frames each
        # For chunk starting at video frame V: latent offset = (V-1)//4 + 1 if V>0, else 0
        if chunk_start_video_frame > 0:
            chunk_latent_offset = ((chunk_start_video_frame - 1) // 4) + 1
        else:
            chunk_latent_offset = 0

        prefix_offset = latent_prefix_frames

        print(f"[VTokenInject] Chunk mapping: video_start={chunk_start_video_frame} "
              f"→ latent_offset={chunk_latent_offset}, prefix={prefix_offset}")
        if sigma_end > 0.0 or sigma_start < 1000.0:
            print(f"[VTokenInject] Sigma gating: inject when {sigma_end} <= sigma <= {sigma_start}")

        # Build lookup: which frame indices have cached V?
        cached_frames = set(f for _, f in manager.v_cache.keys())
        target_blocks = manager.target_blocks
        strength = manager.strength

        # Check for existing override to chain with
        existing_override = None
        if "transformer_options" in model.model_options:
            existing_override = model.model_options["transformer_options"].get(
                "optimized_attention_override"
            )

        def _call_next(original_func, args, kwargs):
            if existing_override is not None:
                return existing_override(original_func, *args, **kwargs)
            return original_func(*args, **kwargs)

        # Debug state
        debug_state = {
            "first_call": True,
            "inject_count": 0,
            "last_sigma": None,
            "step_inject_count": 0,
            "sigma_gated_count": 0,
        }

        def inject_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]

            # Skip cross-attention (K shorter than Q = text tokens as keys).
            # Allow K >= Q: AnchorKVCache prepends anchor tokens to K/V,
            # making K longer than Q. This is still self-attention.
            if k.shape[1] < q.shape[1]:
                return _call_next(original_func, args, kwargs)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)
            window = t_opts.get("context_window", None)
            grid_sizes = t_opts.get("grid_sizes", None)
            sigmas = t_opts.get("sigmas", None)

            # Skip non-targeted blocks
            if target_blocks is not None and block_idx not in target_blocks:
                return _call_next(original_func, args, kwargs)

            # Detect new step
            current_sigma = sigmas[0].item() if sigmas is not None and len(sigmas) > 0 else None
            if current_sigma is not None and current_sigma != debug_state["last_sigma"]:
                if debug_state["step_inject_count"] > 0 or debug_state["sigma_gated_count"] > 0:
                    print(f"[VTokenInject] Step summary: "
                          f"injected {debug_state['step_inject_count']}, "
                          f"sigma-gated {debug_state['sigma_gated_count']}")
                debug_state["last_sigma"] = current_sigma
                debug_state["step_inject_count"] = 0
                debug_state["sigma_gated_count"] = 0

            # Sigma gating: only inject when sigma is within [sigma_end, sigma_start].
            # At high sigma (early steps), V encodes spatial layout — good to anchor.
            # At low sigma (late steps), V encodes fine detail — should be computed
            # naturally for temporal consistency (prevents "pop" at injected frames).
            if current_sigma is not None:
                if current_sigma > sigma_start or current_sigma < sigma_end:
                    if debug_state["sigma_gated_count"] == 0 and block_idx == 0:
                        print(f"[VTokenInject] Sigma gated: σ={current_sigma:.4f} "
                              f"outside [{sigma_end}, {sigma_start}], skipping injection")
                    debug_state["sigma_gated_count"] += 1
                    return _call_next(original_func, args, kwargs)

            # No context window = no injection target
            if window is None or grid_sizes is None:
                if debug_state["first_call"] and block_idx == 0:
                    reason = "no context_window" if window is None else "no grid_sizes"
                    print(f"[VTokenInject] Passthrough: {reason}")
                    debug_state["first_call"] = False
                return _call_next(original_func, args, kwargs)

            # Get frame indices for this window
            index_list = getattr(window, "index_list", None)
            if index_list is None:
                return _call_next(original_func, args, kwargs)

            # First call: full state dump
            if debug_state["first_call"]:
                print(f"[VTokenInject] === First inject call ===")
                print(f"[VTokenInject] grid_sizes: {grid_sizes}")
                print(f"[VTokenInject] V shape: {v.shape}")
                print(f"[VTokenInject] Window index_list (local): "
                      f"{index_list[:8]}{'...' if len(index_list) > 8 else ''} "
                      f"({len(index_list)} frames)")
                # Show the global mapping for video frames
                global_map = []
                for li in index_list:
                    if li < prefix_offset:
                        global_map.append(f"L{li}=ref")
                    else:
                        gf = (li - prefix_offset) + chunk_latent_offset
                        global_map.append(f"L{li}=G{gf}")
                print(f"[VTokenInject] Local→Global map: "
                      f"{global_map[:8]}{'...' if len(global_map) > 8 else ''}")
                print(f"[VTokenInject] Cached frames (global): {sorted(cached_frames)}")
                print(f"[VTokenInject] Prefix offset: {prefix_offset}, "
                      f"Chunk latent offset: {chunk_latent_offset}")
                print(f"[VTokenInject] Strength: {strength}")
                if current_sigma is not None:
                    print(f"[VTokenInject] sigma: {current_sigma:.4f}")
                debug_state["first_call"] = False

            # Find which cached frames overlap with this window
            # index_list contains LOCAL indices into the chunk's combined latent tensor:
            #   [0..prefix-1] = reference prefix frames (NOT video)
            #   [prefix..] = video frames, where local video index 0 = chunk_latent_offset globally
            # V cache keys use GLOBAL latent frame indices, so we must map:
            #   global_latent = (local_idx - prefix_offset) + chunk_latent_offset
            tokens_per_frame = grid_sizes[1] * grid_sizes[2]

            # Detect anchor prefix in V tensor (from AnchorKVCache prepend).
            # If V has more tokens than expected from the window's frame count,
            # the excess is anchor tokens prepended by an outer override.
            expected_window_tokens = len(index_list) * tokens_per_frame
            anchor_token_offset = max(0, v.shape[1] - expected_window_tokens)

            matching = []  # list of (local_pos, global_frame_idx) for frames with cached V
            for pos, local_idx in enumerate(index_list):
                if local_idx < prefix_offset:
                    continue  # skip reference prefix region
                global_frame = (local_idx - prefix_offset) + chunk_latent_offset
                if global_frame in cached_frames and (block_idx, global_frame) in manager.v_cache:
                    matching.append((pos, global_frame))

            if not matching:
                # No cached frames in this window — pass through
                return _call_next(original_func, args, kwargs)

            # In-place V replacement: blend cached V into matching frame positions.
            # This preserves K/V sequence length alignment (no prepend) and is
            # compatible with AnchorKVCache or any other override in the chain.
            v_modified = v.clone()
            replaced_count = 0

            for pos, global_frame in matching:
                key = (block_idx, global_frame)
                v_cached = manager.v_cache[key].to(device=v.device, dtype=v.dtype)

                # Handle batch size mismatch (e.g., B=1 at capture, B=2 with CFG at inject)
                if v_cached.shape[0] < v.shape[0]:
                    v_cached = v_cached.expand(v.shape[0], -1, -1)

                # Account for anchor tokens prepended by AnchorKVCache
                start = anchor_token_offset + pos * tokens_per_frame
                end = start + tokens_per_frame
                if end > v.shape[1] or v_cached.shape[1] != tokens_per_frame:
                    if block_idx == 0:
                        print(f"[VTokenInject] GUARD SKIP: pos={pos}, G{global_frame}, "
                              f"start={start}, end={end}, v.shape[1]={v.shape[1]}, "
                              f"v_cached.shape[1]={v_cached.shape[1]}, tpf={tokens_per_frame}, "
                              f"anchor_offset={anchor_token_offset}")
                    continue

                # Blend: strength=1.0 fully replaces, strength=0.5 averages
                if strength >= 1.0:
                    v_modified[:, start:end, :] = v_cached
                else:
                    v_modified[:, start:end, :] = (
                        (1.0 - strength) * v[:, start:end, :] + strength * v_cached
                    )
                replaced_count += 1

            if replaced_count == 0:
                return _call_next(original_func, args, kwargs)

            debug_state["step_inject_count"] += 1
            debug_state["inject_count"] += 1

            # Log first injection per step
            if debug_state["step_inject_count"] == 1:
                matched_info = [f"L{pos}→G{gf}" for pos, gf in matching]
                print(f"[VTokenInject] INJECT: window "
                      f"(local {index_list[:5]}{'...' if len(index_list) > 5 else ''}) "
                      f"block {block_idx}")
                print(f"[VTokenInject]   Replaced V: {matched_info} "
                      f"({replaced_count} × {tokens_per_frame} tokens, strength={strength})")

            # V replaced in-place — K/V lengths unchanged, compatible with all overrides
            new_args = list(args)
            new_args[2] = v_modified

            return _call_next(original_func, tuple(new_args), kwargs)

        # Apply the override
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = inject_override

        stats = manager.get_cache_stats()
        chain_info = " (chained with existing override)" if existing_override else ""
        sigma_info = ""
        if sigma_end > 0.0 or sigma_start < 1000.0:
            sigma_info = f", sigma=[{sigma_end}, {sigma_start}]"
        print(f"[VTokenInject] Patched model: {stats['frames']} cached frames, "
              f"{stats['blocks']} blocks, strength={strength}{sigma_info}{chain_info}")

        return (model,)


class NV_VTokenFrameSlice:
    """
    Extracts context window center frames from latent and/or image tensors
    for the V-token capture pass (Pass 1.5).

    Calculates which frames are context window centers using the same
    formula as WAN Context Windows (static standard schedule), then
    slices inputs to just those frames.

    Use this node once per input that needs slicing:
      - The main upscaled pre-pass latent (connect to 'latent')
      - Each VACE control video (connect to 'images')

    Wire context_window_size and context_overlap from NV Chunk Loader
    to ensure settings match the chunked pass.

    Workflow:
      [Upscaled Latent] → [NV_VTokenFrameSlice] → sliced latent → [KSampler]
      [Control Video]   → [NV_VTokenFrameSlice] → sliced images → [WanVaceToVideo]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_video_frames": ("INT", {
                    "default": 65,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Total video frames in the full generation (e.g. 65, 81)."
                }),
                "context_window_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 400,
                    "step": 4,
                    "tooltip": "Context window size in VIDEO frames. Wire from chunk loader."
                }),
                "context_overlap": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 200,
                    "step": 4,
                    "tooltip": "Context overlap in VIDEO frames. Wire from chunk loader."
                }),
                "capture_frames": ("STRING", {
                    "default": "auto",
                    "tooltip": "'auto' for window centers, or comma-separated latent frame "
                               "indices like '6,14'."
                }),
            },
            "optional": {
                "latent": ("LATENT",),
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "INT")
    RETURN_NAMES = ("latent", "images", "frame_info", "num_center_frames")
    FUNCTION = "slice_frames"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Extracts context window center frames from latent or image tensors "
        "for the V-token capture pass (Pass 1.5). Use once per input: "
        "connect latent for the upscaled pre-pass, or images for VACE controls. "
        "Wire context settings from NV Chunk Loader."
    )

    def slice_frames(self, total_video_frames, context_window_size, context_overlap,
                     capture_frames, latent=None, images=None):
        # Convert video to latent frames
        total_latent = max(((total_video_frames - 1) // 4) + 1, 1)
        latent_ctx_len = max(((context_window_size - 1) // 4) + 1, 1)
        latent_ctx_overlap = max(((context_overlap - 1) // 4) + 1, 0)

        # Calculate center frames in latent space
        centers = _parse_capture_frames(
            capture_frames, total_latent, latent_ctx_len, latent_ctx_overlap
        )

        info_lines = [
            f"Video→Latent: total {total_video_frames}→{total_latent}, "
            f"ctx {context_window_size}→{latent_ctx_len}, "
            f"overlap {context_overlap}→{latent_ctx_overlap}",
            f"Center latent frames: {centers}",
        ]

        # --- Slice LATENT (T dimension in latent frame space) ---
        sliced_latent = latent
        if latent is not None and centers:
            samples = latent["samples"]  # [B, C, T, H, W]
            valid_centers = [c for c in centers if c < samples.shape[2]]
            if valid_centers:
                sliced_samples = samples[:, :, valid_centers, :, :]
                sliced_latent = {"samples": sliced_samples}
                # Preserve other latent keys (noise_mask, etc.)
                for k, v in latent.items():
                    if k != "samples":
                        sliced_latent[k] = v
                info_lines.append(
                    f"Latent: [{samples.shape[2]}] → [{sliced_samples.shape[2]}] frames "
                    f"(indices {valid_centers})"
                )
            else:
                info_lines.append(
                    f"WARNING: No valid latent centers (T={samples.shape[2]}, "
                    f"centers={centers})"
                )

        # --- Slice IMAGE (frame dimension, mapped from latent centers) ---
        sliced_images = images
        if images is not None and centers:
            video_indices = _latent_to_video_frames(centers, total_video_frames)
            valid_indices = [i for i in video_indices if i < images.shape[0]]
            if valid_indices:
                sliced_images = images[valid_indices]
                info_lines.append(
                    f"Images: [{images.shape[0]}] → [{sliced_images.shape[0]}] frames "
                    f"(video indices {valid_indices})"
                )
            else:
                info_lines.append(
                    f"WARNING: No valid video indices (N={images.shape[0]}, "
                    f"mapped={video_indices})"
                )

        frame_info = "\n".join(info_lines)
        print(f"[VTokenFrameSlice] {frame_info}")

        return (sliced_latent, sliced_images, frame_info, len(centers))


# ============================================================================
# Node Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_VTokenCapture": NV_VTokenCapture,
    "NV_VTokenSave": NV_VTokenSave,
    "NV_VTokenLoad": NV_VTokenLoad,
    "NV_VTokenInject": NV_VTokenInject,
    "NV_VTokenFrameSlice": NV_VTokenFrameSlice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VTokenCapture": "NV V-Token Capture",
    "NV_VTokenSave": "NV V-Token Save",
    "NV_VTokenLoad": "NV V-Token Load",
    "NV_VTokenInject": "NV V-Token Inject",
    "NV_VTokenFrameSlice": "NV V-Token Frame Slice",
}
