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


class NV_WANMemoryEstimator:
    """
    Estimates VRAM requirements for WAN video generation.

    Connect your actual MODEL (with LoRAs applied) to get accurate size estimates.
    The node measures real parameter counts rather than using hardcoded values.

    Memory breakdown:
    - Model Weights: Actual size of loaded diffusion model + LoRAs
    - VAE Weights: Size of loaded VAE model
    - Latent Tensor: Working memory for latent video
    - Activations: Estimated transformer activation memory
    - VACE Cache: If using VACE + context windows (can be CPU offloaded)
    - VAE Output: Decoded video frames (can be streamed)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixel_frames": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of output video frames (pixel space)"
                }),
                "height": ("INT", {
                    "default": 480,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Video height in pixels (should be divisible by 16)"
                }),
                "width": ("INT", {
                    "default": 832,
                    "min": 64,
                    "max": 2048,
                    "step": 16,
                    "tooltip": "Video width in pixels (should be divisible by 16)"
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "The diffusion model (with LoRAs). If not connected, uses estimate."
                }),
                "vae": ("VAE", {
                    "tooltip": "The VAE model. If not connected, uses estimate."
                }),
                "use_vace": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable VACE conditioning memory estimation"
                }),
                "num_vace_inputs": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "tooltip": "Number of VACE video inputs (each adds to cache size)"
                }),
                "use_context_windows": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Using context windows for long video generation"
                }),
                "context_window_size": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": 200,
                    "tooltip": "Context window size in pixel frames"
                }),
                "cpu_offload_vace": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "VACE cache offloaded to CPU (NV_VACEContextWindowPatcher)"
                }),
                "streaming_vae_decode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Using NV_StreamingVAEDecode to stream output to CPU"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("memory_report",)
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Estimates VRAM requirements for WAN video generation. Connect your MODEL to get accurate size including LoRAs."

    def estimate(self, pixel_frames, height, width,
                 model=None, vae=None,
                 use_vace=False, num_vace_inputs=1,
                 use_context_windows=False, context_window_size=81,
                 cpu_offload_vace=True, streaming_vae_decode=False):
        """
        Calculate memory estimates for the workflow.
        """

        # Latent space dimensions (WAN: 4x temporal, 8x spatial compression)
        latent_frames = (pixel_frames + 3) // 4  # Ceiling division
        latent_h = height // 8
        latent_w = width // 8
        latent_channels = 16  # WAN latent channels

        report_lines = []
        report_lines.append("=" * 50)
        report_lines.append("WAN VRAM ESTIMATION REPORT")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Input parameters
        report_lines.append("INPUT PARAMETERS:")
        report_lines.append(f"  Pixel frames: {pixel_frames}")
        report_lines.append(f"  Resolution: {width}x{height}")
        report_lines.append(f"  Latent size: {latent_frames} frames x {latent_h}x{latent_w}")
        report_lines.append("")

        total_gpu_memory = 0
        total_cpu_memory = 0

        # 1. Model Weights
        report_lines.append("MODEL WEIGHTS:")
        if model is not None:
            model_bytes = get_model_size(model)
            report_lines.append(f"  Diffusion model: {format_bytes(model_bytes)} (measured)")

            # Check for LoRAs in model options
            if hasattr(model, 'model_options'):
                patches = model.model_options.get('patches', {})
                if patches:
                    report_lines.append(f"  (includes {len(patches)} patch types)")
        else:
            # Estimate: WAN 1.3B = ~5.2GB fp16, 14B = ~28GB fp16
            model_bytes = 5.2 * 1024**3  # Default to 1.3B estimate
            report_lines.append(f"  Diffusion model: {format_bytes(model_bytes)} (estimated 1.3B)")
            report_lines.append("  (Connect MODEL input for accurate size with LoRAs)")

        total_gpu_memory += model_bytes

        if vae is not None:
            vae_bytes = get_vae_size(vae)
            report_lines.append(f"  VAE model: {format_bytes(vae_bytes)} (measured)")
        else:
            vae_bytes = 0.5 * 1024**3  # ~500MB estimate
            report_lines.append(f"  VAE model: {format_bytes(vae_bytes)} (estimated)")

        total_gpu_memory += vae_bytes
        report_lines.append("")

        # 2. Latent Tensor
        report_lines.append("LATENT TENSOR:")
        # [B, C, T, H, W] in fp16/bf16
        latent_bytes = 1 * latent_channels * latent_frames * latent_h * latent_w * 2  # 2 bytes for fp16
        report_lines.append(f"  Shape: [1, {latent_channels}, {latent_frames}, {latent_h}, {latent_w}]")
        report_lines.append(f"  Size: {format_bytes(latent_bytes)}")
        total_gpu_memory += latent_bytes
        report_lines.append("")

        # 3. Activation Memory (rough estimate)
        report_lines.append("ACTIVATION MEMORY:")
        # Very rough estimate: ~2-4x model size for activations during forward pass
        # With context windows, only window_size frames are processed at once
        if use_context_windows:
            active_frames = (context_window_size + 3) // 4  # Latent frames in window
            activation_multiplier = 2.5
        else:
            active_frames = latent_frames
            activation_multiplier = 3.0

        # Activation scales with frames being processed and resolution
        resolution_factor = (latent_h * latent_w) / (60 * 104)  # Normalized to 480x832
        frame_factor = active_frames / 21  # Normalized to 81 pixel frames
        activation_bytes = model_bytes * activation_multiplier * resolution_factor * frame_factor * 0.3

        report_lines.append(f"  Active frames: {active_frames} latent frames")
        report_lines.append(f"  Estimated: {format_bytes(activation_bytes)}")
        total_gpu_memory += activation_bytes
        report_lines.append("")

        # 4. VACE Cache (if applicable)
        if use_vace:
            report_lines.append("VACE CACHE:")
            # vace_frames: [B, 32, T, H, W] per input
            vace_frames_bytes = num_vace_inputs * 1 * 32 * latent_frames * latent_h * latent_w * 2
            # vace_mask: [B, 64, T, H, W]
            vace_mask_bytes = 1 * 64 * latent_frames * latent_h * latent_w * 2
            # vace_context: [B, num_inputs, 96, T, H, W]
            vace_context_bytes = 1 * num_vace_inputs * 96 * latent_frames * latent_h * latent_w * 2

            total_vace_bytes = vace_frames_bytes + vace_mask_bytes + vace_context_bytes

            report_lines.append(f"  vace_frames: {format_bytes(vace_frames_bytes)} ({num_vace_inputs} inputs)")
            report_lines.append(f"  vace_mask: {format_bytes(vace_mask_bytes)}")
            report_lines.append(f"  vace_context: {format_bytes(vace_context_bytes)}")
            report_lines.append(f"  Total VACE: {format_bytes(total_vace_bytes)}")

            if use_context_windows and cpu_offload_vace:
                report_lines.append(f"  Location: CPU (offloaded)")
                total_cpu_memory += total_vace_bytes
                # Only window-size cache on GPU
                window_vace = total_vace_bytes * (context_window_size / pixel_frames)
                report_lines.append(f"  GPU (per window): {format_bytes(window_vace)}")
                total_gpu_memory += window_vace
            else:
                report_lines.append(f"  Location: GPU")
                total_gpu_memory += total_vace_bytes
            report_lines.append("")

        # 5. VAE Decode Output
        report_lines.append("VAE DECODE OUTPUT:")
        # Decoded: [B, C, T, H, W] -> [T, H, W, 3] in fp32
        decode_bytes = pixel_frames * height * width * 3 * 4  # fp32
        report_lines.append(f"  Decoded frames: {pixel_frames} x {height} x {width} x 3")
        report_lines.append(f"  Total: {format_bytes(decode_bytes)}")

        if streaming_vae_decode:
            report_lines.append(f"  Location: CPU (streaming)")
            total_cpu_memory += decode_bytes
            # Only ~1 frame on GPU at a time
            frame_bytes = height * width * 3 * 4
            report_lines.append(f"  GPU (per frame): {format_bytes(frame_bytes)}")
            total_gpu_memory += frame_bytes
        else:
            report_lines.append(f"  Location: GPU")
            total_gpu_memory += decode_bytes
        report_lines.append("")

        # Summary
        report_lines.append("=" * 50)
        report_lines.append("SUMMARY")
        report_lines.append("=" * 50)
        report_lines.append(f"  Estimated GPU VRAM: {format_bytes(total_gpu_memory)}")
        if total_cpu_memory > 0:
            report_lines.append(f"  Estimated CPU RAM: {format_bytes(total_cpu_memory)}")
        report_lines.append("")

        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if total_gpu_memory > 24 * 1024**3:
            report_lines.append("  ⚠ Estimated VRAM exceeds 24GB!")
            if not use_context_windows:
                report_lines.append("  → Enable context windows for long videos")
            if use_vace and not cpu_offload_vace:
                report_lines.append("  → Enable VACE CPU offload (NV_VACEContextWindowPatcher)")
            if not streaming_vae_decode:
                report_lines.append("  → Use NV_StreamingVAEDecode for decode")
        elif total_gpu_memory > 12 * 1024**3:
            report_lines.append("  ⚠ Estimated VRAM exceeds 12GB")
            report_lines.append("  Consider enabling optimizations for smaller GPUs")
        else:
            report_lines.append("  ✓ Should fit in most modern GPUs (12GB+)")

        report = "\n".join(report_lines)
        print(report)

        return {"ui": {"text": [report]}, "result": (report,)}


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_WANMemoryEstimator": NV_WANMemoryEstimator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WANMemoryEstimator": "NV WAN Memory Estimator",
}
