"""
WAN Memory Estimator

Estimates VRAM requirements for WAN video generation workflows.
Accepts actual MODEL input to measure real parameter sizes (including LoRAs).

Memory Components:
1. Model Weights - from actual loaded model
2. Activations - proportional to resolution and batch size
3. VACE Cache - if using VACE conditioning with context windows
4. VAE Decode - output accumulation (or streaming)
5. Latent Tensor - intermediate latent storage
6. VAE Encoder feat_cache - causal convolution cache during encoding (CRITICAL!)

NOTE: The VAE encoder's feat_cache is often the OOM bottleneck for long videos.
Even with streaming encode (which streams OUTPUT to CPU), the feat_cache
accumulates on GPU throughout the entire encoding process. This grows with:
  num_causalconv3d_layers × feature_channels × frame_count × H/8 × W/8
"""

import torch


def format_bytes(num_bytes):
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def get_model_size(model):
    """
    Get actual model size in bytes from a ComfyUI MODEL.

    This measures the real loaded weights, including any LoRAs applied.
    """
    try:
        # ComfyUI MODEL wrapper -> actual model
        if hasattr(model, 'model'):
            actual_model = model.model
        else:
            actual_model = model

        # Get state dict and sum parameter sizes
        if hasattr(actual_model, 'state_dict'):
            state_dict = actual_model.state_dict()
            total_bytes = 0
            for key, param in state_dict.items():
                if isinstance(param, torch.Tensor):
                    total_bytes += param.numel() * param.element_size()
            return total_bytes

        # Fallback: iterate parameters
        total_bytes = 0
        for param in actual_model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    except Exception as e:
        print(f"[WAN Memory Estimator] Could not measure model: {e}")
        return 0


def get_vae_size(vae):
    """Get VAE model size in bytes."""
    try:
        if hasattr(vae, 'first_stage_model'):
            vae_model = vae.first_stage_model
        else:
            vae_model = vae

        total_bytes = 0
        for param in vae_model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    except Exception as e:
        print(f"[WAN Memory Estimator] Could not measure VAE: {e}")
        return 0


def count_causalconv3d_encoder(vae):
    """
    Count CausalConv3d layers in VAE encoder and get their output channels.

    Returns: list of output channel counts for each CausalConv3d layer
    """
    try:
        from comfy.ldm.wan.vae import CausalConv3d

        if hasattr(vae, 'first_stage_model'):
            vae_model = vae.first_stage_model
        else:
            vae_model = vae

        if not hasattr(vae_model, 'encoder'):
            return []

        channels = []
        for m in vae_model.encoder.modules():
            if isinstance(m, CausalConv3d):
                # Get output channels from the conv weight shape
                if hasattr(m, 'conv') and hasattr(m.conv, 'weight'):
                    out_channels = m.conv.weight.shape[0]
                    channels.append(out_channels)
        return channels
    except Exception as e:
        print(f"[WAN Memory Estimator] Could not count CausalConv3d: {e}")
        return []


def estimate_feat_cache_size(vae, pixel_frames, height, width):
    """
    Estimate the feat_cache memory used during VAE encoding.

    The feat_cache stores intermediate features for causal convolutions.
    Each CausalConv3d layer caches features at its CURRENT spatial resolution
    (not just latent space). Early encoder layers operate at full or partial
    resolution before downsampling completes.

    This is the primary OOM bottleneck for long video encoding.

    Returns: estimated bytes for feat_cache
    """
    # WAN VAE encoder has progressive spatial downsampling:
    # - Input: H x W (pixel space)
    # - After block 1: H/2 x W/2
    # - After block 2: H/4 x W/4
    # - After block 3: H/8 x W/8 (latent space)
    # Each block has multiple CausalConv3d layers

    # Approximate layer distribution across spatial resolutions
    # Based on typical VAE encoder structure with 3 downsampling blocks
    spatial_levels = [
        # (num_layers, spatial_divisor, channels)
        (4, 1, 128),    # Full resolution layers
        (4, 2, 256),    # Half resolution layers
        (4, 4, 512),    # Quarter resolution layers
        (4, 8, 512),    # Latent resolution layers
    ]

    # Override with actual channels if VAE provided
    if vae is not None:
        detected_channels = count_causalconv3d_encoder(vae)
        if detected_channels:
            # Distribute detected channels across spatial levels
            num_layers = len(detected_channels)
            layers_per_level = max(1, num_layers // 4)
            spatial_levels = []
            for i, divisor in enumerate([1, 2, 4, 8]):
                start_idx = i * layers_per_level
                end_idx = (i + 1) * layers_per_level if i < 3 else num_layers
                level_channels = detected_channels[start_idx:end_idx]
                if level_channels:
                    avg_ch = sum(level_channels) // len(level_channels)
                    spatial_levels.append((len(level_channels), divisor, avg_ch))

    # Each CausalConv3d caches: [B, C, cache_t, H/div, W/div]
    # cache_t is kernel temporal size - 1 (typically 2 frames for kernel_t=3)
    cache_temporal_frames = 2

    total_bytes = 0
    for num_layers, spatial_div, channels in spatial_levels:
        layer_h = height // spatial_div
        layer_w = width // spatial_div
        # [1, C, cache_t, H, W] in fp16/bf16 (2 bytes per element)
        layer_bytes = num_layers * 1 * channels * cache_temporal_frames * layer_h * layer_w * 2
        total_bytes += layer_bytes

    # Peak memory overhead during torch.cat([cache_x, x], dim=2)
    # Both cache_x and the current chunk x must exist simultaneously
    # x has shape [1, C, chunk_frames, H, W] where chunk_frames = 4 for WAN
    chunk_frames = 4
    for num_layers, spatial_div, channels in spatial_levels:
        layer_h = height // spatial_div
        layer_w = width // spatial_div
        # This is the peak overhead per layer for the cat operation
        peak_overhead = num_layers * 1 * channels * chunk_frames * layer_h * layer_w * 2
        total_bytes += peak_overhead * 0.3  # Not all layers peak simultaneously

    # The full input tensor must be on GPU during streaming encode
    # Shape: [1, 3, pixel_frames, H, W] in fp16
    input_tensor_bytes = 1 * 3 * pixel_frames * height * width * 2
    total_bytes += input_tensor_bytes

    return int(total_bytes)


def estimate_max_frames_for_vram(available_vram_gb, height, width, vae=None):
    """
    Estimate maximum pixel frames that can be encoded given available VRAM.

    This is useful for determining safe chunk sizes for streaming encode.

    Args:
        available_vram_gb: Available VRAM in GB for encoding
        height: Video height in pixels
        width: Video width in pixels
        vae: Optional VAE model for accurate channel counts

    Returns: estimated max pixel frames
    """
    available_bytes = available_vram_gb * 1024**3

    # Binary search for max frames
    low, high = 1, 1000
    while low < high:
        mid = (low + high + 1) // 2
        cache_bytes = estimate_feat_cache_size(vae, mid, height, width)
        if cache_bytes <= available_bytes:
            low = mid
        else:
            high = mid - 1

    return low


class NV_WorkflowFeasibilityChecker:
    """
    Pre-flight check for WAN VACE workflows.

    Analyzes your settings and tells you BEFORE running:
    - Will this workflow fit in VRAM?
    - What are the optimal chunk and context window sizes?
    - What's the expected memory usage per phase?

    Outputs can be connected directly to other nodes:
    - recommended_chunk_size → NV_ParallelChunkPlanner
    - recommended_context_window → WAN Context Windows node
    - recommended_context_overlap → WAN Context Windows node
    - recommended_chunk_overlap → NV_ParallelChunkPlanner
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_frames": ("INT", {
                    "default": 480, "min": 1, "max": 10000,
                    "tooltip": "Total video frames to process"
                }),
                "height": ("INT", {
                    "default": 480, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Video height in pixels"
                }),
                "width": ("INT", {
                    "default": 832, "min": 64, "max": 2048, "step": 16,
                    "tooltip": "Video width in pixels"
                }),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "For accurate model size measurement"}),
                "vae": ("VAE", {"tooltip": "For accurate VAE feat_cache estimation"}),
                "available_vram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 128.0, "step": 0.1,
                    "tooltip": "0 = auto-detect GPU VRAM"
                }),
                "use_vace": ("BOOLEAN", {"default": True}),
                "num_vace_inputs": ("INT", {"default": 1, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "STRING", "BOOLEAN",)
    RETURN_NAMES = (
        "recommended_chunk_size",
        "recommended_context_window",
        "recommended_context_overlap",
        "recommended_chunk_overlap",
        "feasibility_report",
        "will_fit",
    )
    OUTPUT_NODE = True
    FUNCTION = "check_feasibility"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Pre-flight check: Will your workflow fit in VRAM? Get optimal chunk and context window settings before running."

    def check_feasibility(self, total_frames, height, width,
                          model=None, vae=None,
                          available_vram_gb=0.0, use_vace=True, num_vace_inputs=1):
        """
        Analyze workflow memory requirements and recommend optimal settings.

        Checks two phases:
        1. VACE Encoding - feat_cache memory bottleneck
        2. KSampler Inference - activation and VACE window memory bottleneck

        Returns recommended settings that will fit in available VRAM.
        """

        # Auto-detect VRAM
        if available_vram_gb <= 0:
            if torch.cuda.is_available():
                available_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            else:
                available_vram_gb = 8.0

        # Get model size
        if model is not None:
            model_size_gb = get_model_size(model) / 1024**3
        else:
            model_size_gb = 5.2  # WAN 1.3B default

        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("WORKFLOW FEASIBILITY CHECK")
        report_lines.append("=" * 50)
        report_lines.append(f"Total VRAM: {available_vram_gb:.1f} GB")
        report_lines.append(f"Resolution: {width}x{height}")
        report_lines.append(f"Total frames: {total_frames}")
        report_lines.append(f"VACE: {'Yes' if use_vace else 'No'}")
        if use_vace and num_vace_inputs > 1:
            report_lines.append(f"VACE inputs: {num_vace_inputs}")
        report_lines.append("")

        # === PHASE 1: ENCODING FEASIBILITY ===
        report_lines.append("PHASE 1: VACE ENCODING")
        report_lines.append("-" * 30)

        # Calculate max safe chunk size for encoding
        # Reserve memory for VAE weights (~500MB) and some headroom
        encoding_budget_gb = (available_vram_gb - 0.5) * 0.4  # 40% of remaining for encoding
        if use_vace:
            encoding_budget_gb /= 2  # 2 encodes (inactive + reactive)

        max_encode_frames = estimate_max_frames_for_vram(encoding_budget_gb, height, width, vae)
        safe_chunk_size = int(max_encode_frames * 0.8)  # 80% safety margin
        # Align to WAN's temporal pattern: (n-1) mod 4 + 1 = 1, so n = 4k+1
        safe_chunk_size = ((safe_chunk_size - 1) // 4) * 4 + 1
        safe_chunk_size = max(17, safe_chunk_size)  # Minimum viable chunk

        encode_feasible = total_frames <= safe_chunk_size or safe_chunk_size >= 65

        report_lines.append(f"  Encoding budget: {encoding_budget_gb:.1f} GB")
        report_lines.append(f"  Max safe frames: {max_encode_frames}")
        report_lines.append(f"  Recommended chunk_size: {safe_chunk_size}")

        if total_frames <= safe_chunk_size:
            report_lines.append("  -> Can encode full video in one pass")
        else:
            # Estimate number of chunks needed (accounting for overlap)
            effective_chunk = safe_chunk_size - 20  # Assume ~20 frame overlap
            num_chunks = 1 + max(0, (total_frames - safe_chunk_size + effective_chunk - 1) // effective_chunk)
            report_lines.append(f"  -> Need ~{num_chunks} chunks for parallel processing")
        report_lines.append("")

        # === PHASE 2: INFERENCE FEASIBILITY ===
        report_lines.append("PHASE 2: KSAMPLER INFERENCE")
        report_lines.append("-" * 30)

        # Calculate optimal context window
        latent_h = height // 8
        latent_w = width // 8
        model_bytes = model_size_gb * 1024**3
        vae_bytes = 0.5 * 1024**3
        fixed_memory = model_bytes + vae_bytes

        inference_budget = available_vram_gb * 1024**3 - fixed_memory

        # Find optimal context window size
        # Try from largest (best quality) to smallest, find first that fits
        optimal_context = 17  # Fallback minimum
        optimal_window_memory = 0

        for ctx in range(193, 16, -4):  # 193, 189, 185, ... down to 17
            window_latent = (ctx + 3) // 4

            # Activation memory: scales with window size and resolution
            activation_factor = 2.5 * (window_latent / 21)  # Normalized to 81 pixel frames (21 latent)
            resolution_factor = (latent_h * latent_w) / (60 * 104)  # Normalized to 480x832
            activation = model_bytes * activation_factor * resolution_factor * 0.3

            # Per-window latent tensor: [1, 16, window_latent, H/8, W/8] in fp16
            latent = 1 * 16 * window_latent * latent_h * latent_w * 2

            # Per-window VACE conditioning (sliced by patcher)
            if use_vace:
                # vace_frames: [1, 32, T, H/8, W/8]
                # vace_mask: [1, 64, T, H/8, W/8]
                # vace_context: [1, num_vace, 96, T, H/8, W/8]
                vace = (
                    32 * window_latent * latent_h * latent_w * 2 +
                    64 * window_latent * latent_h * latent_w * 2 +
                    num_vace_inputs * 96 * window_latent * latent_h * latent_w * 2
                )
            else:
                vace = 0

            window_total = activation + latent + vace

            if window_total < inference_budget * 0.8:  # 80% safety margin
                optimal_context = ctx
                optimal_window_memory = window_total
                break

        # Calculate optimal overlap (20-25% of context window for temporal coherence)
        context_overlap = max(8, optimal_context // 4)
        context_overlap = (context_overlap // 4) * 4  # Align to WAN temporal pattern

        # Chunk overlap for stitching (should be >= context_overlap for smooth blends)
        chunk_overlap = max(16, optimal_context // 2)
        chunk_overlap = (chunk_overlap // 4) * 4

        inference_feasible = optimal_context >= 17

        report_lines.append(f"  Model size: {model_size_gb:.1f} GB")
        report_lines.append(f"  Inference budget: {inference_budget / 1024**3:.1f} GB")
        report_lines.append(f"  Optimal context_window_size: {optimal_context}")
        report_lines.append(f"  Per-window memory: {format_bytes(optimal_window_memory)}")
        report_lines.append(f"  Recommended context_overlap: {context_overlap}")
        report_lines.append(f"  Recommended chunk_overlap: {chunk_overlap}")

        # Estimate windows per chunk
        effective_context = optimal_context - context_overlap
        if safe_chunk_size > optimal_context and effective_context > 0:
            num_windows = 1 + (safe_chunk_size - optimal_context) // effective_context
        else:
            num_windows = 1
        report_lines.append(f"  Windows per chunk: ~{num_windows}")
        report_lines.append("")

        # === OVERALL FEASIBILITY ===
        will_fit = encode_feasible and inference_feasible

        report_lines.append("=" * 50)
        report_lines.append("VERDICT")
        report_lines.append("=" * 50)

        if will_fit:
            report_lines.append("WORKFLOW WILL FIT IN VRAM")
            report_lines.append("")
            report_lines.append("Recommended settings:")
            report_lines.append(f"  chunk_size: {safe_chunk_size}")
            report_lines.append(f"  context_window_size: {optimal_context}")
            report_lines.append(f"  context_overlap: {context_overlap}")
            report_lines.append(f"  chunk_overlap: {chunk_overlap}")
        else:
            report_lines.append("WARNING: WORKFLOW MAY OOM")
            report_lines.append("")
            if not encode_feasible:
                report_lines.append("-> Encoding: Reduce resolution or use smaller chunks")
                report_lines.append(f"   Max safe encode: {max_encode_frames} frames @ {width}x{height}")
            if not inference_feasible:
                report_lines.append("-> Inference: Need smaller context windows or lower resolution")
                report_lines.append(f"   Available for inference: {inference_budget / 1024**3:.1f} GB")

        report_lines.append("")
        report_lines.append("Connect outputs to workflow nodes:")
        report_lines.append("  recommended_chunk_size -> NV_ParallelChunkPlanner")
        report_lines.append("  recommended_context_window -> WAN Context Windows")
        report_lines.append("  recommended_context_overlap -> WAN Context Windows")

        report = "\n".join(report_lines)
        print(report)

        return {
            "ui": {"text": [report]},
            "result": (
                safe_chunk_size,
                optimal_context,
                context_overlap,
                chunk_overlap,
                report,
                will_fit,
            )
        }


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_WorkflowFeasibilityChecker": NV_WorkflowFeasibilityChecker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WorkflowFeasibilityChecker": "NV Workflow Feasibility Checker",
}
