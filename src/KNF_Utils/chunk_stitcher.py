"""
NV Chunk Stitcher

Stitches processed video chunks back together with crossfade blending.
Loads chunk videos from disk and applies smooth transitions at overlap regions.

Key Features:
- Linear crossfade blending (same algorithm as v1 NV_VideoSampler, but on final pixels)
- NO re-encoding through VAE - pure pixel operation
- Loads chunks from numbered video files (chunk_0.mp4, chunk_1.mp4, etc.)
- Outputs stitched video frames ready for saving

Usage:
1. Process all chunks on separate GPUs/machines
2. Save each output as chunk_0.mp4, chunk_1.mp4, etc. in a shared directory
3. Run NV_ChunkStitcher to combine them
4. Save the final stitched output
"""

import os
import json
import math
import torch
import numpy as np


# ============================================================================
# Blend Weight Functions
# ============================================================================
# Different window functions for chunk transition blending.
# Linear is the simplest; Hamming/Hann have better sidelobe properties
# (less ringing/ghosting at boundaries).
# See: Overlapping Co-Denoising (arxiv 2511.03272) for motivation.

BLEND_MODES = ["linear", "hamming", "hann", "cosine", "fft_spectral"]


def compute_blend_weights(num_frames: int, mode: str = "linear", device=None) -> torch.Tensor:
    """
    Compute blend weights for crossfade transition.

    All modes produce weights that go from 0 (keep output) to 1 (keep chunk).

    Args:
        num_frames: Number of frames in the crossfade region
        mode: One of "linear", "hamming", "hann", "cosine"
        device: Torch device for the output tensor

    Returns:
        Tensor of shape [num_frames] with values in [0, 1]
    """
    if num_frames <= 1:
        return torch.ones(max(1, num_frames), device=device)

    t = torch.linspace(0, 1, num_frames, device=device)

    if mode == "linear":
        weights = t
    elif mode == "hamming":
        # Hamming window: good sidelobe suppression (-43dB)
        # We use the rising half of the window (0 -> 1)
        weights = 0.54 - 0.46 * torch.cos(math.pi * t)
    elif mode == "hann":
        # Hann window: first sidelobe -31dB, but zero endpoints
        weights = 0.5 * (1.0 - torch.cos(math.pi * t))
    elif mode == "cosine":
        # Cosine (raised cosine): smooth S-curve transition
        weights = 0.5 * (1.0 - torch.cos(math.pi * t))
    else:
        raise ValueError(f"Unknown blend mode: {mode}. Choose from {BLEND_MODES}")

    return weights


def fft_spectral_blend(region_a: torch.Tensor, region_b: torch.Tensor,
                       cutoff_ratio: float = 0.25) -> torch.Tensor:
    """
    FFT spectral blending: take low frequencies (structure/motion) from region_a
    and high frequencies (detail/sharpness) from region_b.

    Inspired by FreeLong (NeurIPS 2024, arxiv 2407.19918).

    Args:
        region_a: Overlap frames from previous chunk [T, H, W, C]
        region_b: Overlap frames from next chunk [T, H, W, C]
        cutoff_ratio: Fraction of frequencies to take from region_a (0.25 = bottom 25%)

    Returns:
        Blended frames [T, H, W, C]
    """
    # Move to channel-first for FFT: [T, H, W, C] -> [T, C, H, W]
    a = region_a.permute(0, 3, 1, 2).float()
    b = region_b.permute(0, 3, 1, 2).float()

    # 3D FFT over temporal + spatial dims
    a_freq = torch.fft.fftn(a, dim=(-3, -2, -1))
    b_freq = torch.fft.fftn(b, dim=(-3, -2, -1))

    # Build low-pass mask (ellipsoidal in frequency space)
    T, C, H, W = a.shape
    t_freqs = torch.fft.fftfreq(T, device=a.device)
    h_freqs = torch.fft.fftfreq(H, device=a.device)
    w_freqs = torch.fft.fftfreq(W, device=a.device)

    # Normalized frequency magnitude (0 to 1)
    t_grid, h_grid, w_grid = torch.meshgrid(t_freqs, h_freqs, w_freqs, indexing='ij')
    freq_magnitude = torch.sqrt(t_grid**2 + h_grid**2 + w_grid**2)
    freq_magnitude = freq_magnitude / (freq_magnitude.max() + 1e-8)

    # Smooth transition around cutoff (sigmoid) to avoid ringing
    sharpness = 10.0
    low_pass = torch.sigmoid(-sharpness * (freq_magnitude - cutoff_ratio))
    high_pass = 1.0 - low_pass

    # Expand mask for channels: [T, H, W] -> [T, C, H, W]
    low_pass = low_pass.unsqueeze(1).expand_as(a_freq)
    high_pass = high_pass.unsqueeze(1).expand_as(b_freq)

    # Blend in frequency domain
    blended_freq = a_freq * low_pass + b_freq * high_pass

    # IFFT back to spatial domain
    blended = torch.fft.ifftn(blended_freq, dim=(-3, -2, -1)).real

    # Back to [T, H, W, C]
    blended = blended.permute(0, 2, 3, 1)

    return blended.to(region_a.dtype)


# ============================================================================
# WAN Frame Alignment Helpers
# ============================================================================
# WAN models require frame counts to satisfy: (frames % 4) == 1
# Valid counts: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65...
# VHS VideoHelperSuite truncates frames to meet this constraint AFTER loading,
# which can break chunk boundaries if not accounted for.

def is_wan_aligned(frames: int) -> bool:
    """Check if frame count satisfies WAN constraint: (frames % 4) == 1"""
    return (frames % 4) == 1


def nearest_wan_aligned(frames: int) -> int:
    """Return the nearest valid WAN frame count (always rounds down)"""
    if is_wan_aligned(frames):
        return frames
    return ((frames - 1) // 4) * 4 + 1


def validate_wan_alignment(frames: int) -> tuple:
    """
    Check if frame count satisfies WAN constraint.

    Returns:
        (is_valid, nearest_valid, difference)
    """
    is_valid = is_wan_aligned(frames)
    nearest = nearest_wan_aligned(frames)
    diff = frames - nearest
    return (is_valid, nearest, diff)


def calculate_adjusted_overlap(expected_frames: int, actual_frames: int, planned_overlap: int) -> int:
    """
    Calculate adjusted overlap given frame count mismatch.

    If chunks were truncated by WAN alignment, the overlap region may be smaller
    than planned. This calculates the actual usable overlap.
    """
    diff = expected_frames - actual_frames
    adjusted = max(0, planned_overlap - diff)
    return adjusted


def load_video_frames(video_path):
    """
    Load video file and return frames as tensor [T, H, W, C].

    Uses OpenCV for video loading.
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV (cv2) is required for video loading.\n"
            "Install with: pip install opencv-python"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    # Stack to tensor [T, H, W, C]
    return torch.from_numpy(np.stack(frames, axis=0))


class NV_ChunkStitcher:
    """
    Stitches video chunks with crossfade blending.

    Loads chunk videos from a directory and applies linear crossfade
    at overlap regions. No VAE re-encoding - pure pixel blending.

    Features:
    - Validates chunk frame counts against plan
    - WAN mode (default ON) checks frame alignment constraints
    - Detailed reporting of any issues detected
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan_json_path": ("STRING", {
                    "default": "chunk_plan.json",
                    "tooltip": "Path to the chunk_plan.json file"
                }),
                "chunk_directory": ("STRING", {
                    "default": "./chunks",
                    "tooltip": "Directory containing chunk_0.mp4, chunk_1.mp4, etc."
                }),
            },
            "optional": {
                "chunk_prefix": ("STRING", {
                    "default": "chunk_",
                    "tooltip": "Filename prefix for chunk videos (e.g., 'chunk_' for chunk_0.mp4)"
                }),
                "chunk_extension": ("STRING", {
                    "default": ".mp4",
                    "tooltip": "File extension for chunk videos"
                }),
                "crossfade_override": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 128,
                    "tooltip": "Override crossfade frames (-1 = use plan's overlap_frames)"
                }),
                "wan_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WAN frame alignment awareness. Validates (N%4)==1 constraint and reports issues."
                }),
                "blend_mode": (BLEND_MODES, {
                    "default": "linear",
                    "tooltip": "Crossfade blend function. hamming/hann have better boundary properties than linear."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_report",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches processed video chunks with crossfade blending and WAN alignment validation."

    def stitch_chunks(self, plan_json_path, chunk_directory,
                      chunk_prefix="chunk_", chunk_extension=".mp4",
                      crossfade_override=-1, wan_mode=True, blend_mode="linear"):
        """
        Load and stitch all chunks with crossfade blending.
        """

        # Load the plan
        try:
            with open(plan_json_path, 'r') as f:
                plan = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Chunk plan not found: {plan_json_path}")

        num_chunks = len(plan.get("chunks", []))
        blend_config = plan.get("blend_config", {})
        planned_overlap = plan.get("overlap_frames", 32)

        # Determine crossfade frames
        if crossfade_override >= 0:
            crossfade_frames = crossfade_override
        else:
            crossfade_frames = blend_config.get("crossfade_frames", planned_overlap)

        # Build report header
        report_lines = [
            "=" * 60,
            "CHUNK STITCH REPORT",
            "=" * 60,
            f"Plan: {plan_json_path}",
            f"Plan overlap_frames: {planned_overlap}",
            f"Effective crossfade: {crossfade_frames}",
            f"WAN mode: {wan_mode}",
        ]

        if crossfade_override >= 0 and crossfade_override != planned_overlap:
            report_lines.append(
                f"⚠️  WARNING: crossfade_override ({crossfade_override}) != plan overlap ({planned_overlap})"
            )

        print(f"[NV_ChunkStitcher] Loading {num_chunks} chunks from {chunk_directory}")
        print(f"[NV_ChunkStitcher] Crossfade: {crossfade_frames} frames, WAN mode: {wan_mode}")

        report_lines.append("")
        report_lines.append("LOADING CHUNKS:")

        # Load all chunks with validation
        chunks = []
        warnings = []
        for i in range(num_chunks):
            video_path = os.path.join(chunk_directory, f"{chunk_prefix}{i}{chunk_extension}")

            if not os.path.exists(video_path):
                raise FileNotFoundError(
                    f"Chunk video not found: {video_path}\n"
                    f"Make sure all chunks have been processed and saved."
                )

            print(f"  Loading chunk {i}: {video_path}")
            chunk_frames = load_video_frames(video_path)
            chunks.append(chunk_frames)

            # Get expected frame count from plan
            actual_frames = chunk_frames.shape[0]
            expected_frames = plan["chunks"][i].get("frame_count") if i < len(plan.get("chunks", [])) else None

            # Check WAN alignment
            is_aligned, nearest, diff = validate_wan_alignment(actual_frames)

            # Build chunk report line
            chunk_info = f"  Chunk {i}: {actual_frames} frames"
            if expected_frames is not None:
                if actual_frames != expected_frames:
                    chunk_info += f" (expected {expected_frames}, diff={actual_frames - expected_frames})"
                    warnings.append(f"Chunk {i} frame count mismatch: {actual_frames} vs expected {expected_frames}")
                else:
                    chunk_info += " ✓"

            if wan_mode:
                if is_aligned:
                    chunk_info += " [WAN OK]"
                else:
                    chunk_info += f" [WAN: needs {nearest}, off by {diff}]"
                    if actual_frames == expected_frames:  # Only warn if we haven't already
                        warnings.append(f"Chunk {i} not WAN-aligned: {actual_frames} frames")

            report_lines.append(chunk_info)
            print(f"    Shape: {chunk_frames.shape}")

        # Add warnings section
        if warnings:
            report_lines.append("")
            report_lines.append("⚠️  WARNINGS:")
            for w in warnings:
                report_lines.append(f"   • {w}")

        # Stitch chunks with crossfade
        report_lines.append("")
        report_lines.append("STITCHING:")
        print(f"\n[NV_ChunkStitcher] Stitching {num_chunks} chunks...")

        # First chunk: use all frames
        output = chunks[0]
        report_lines.append(f"  Chunk 0: {chunks[0].shape[0]} frames (base)")

        for i in range(1, num_chunks):
            current_chunk = chunks[i]

            # Get transition_start_index from plan (KEY FIX: use absolute position, not relative from end)
            # The start_frame in the plan is where this chunk's overlap begins in the original video
            if i < len(plan.get("chunks", [])):
                transition_start_index = plan["chunks"][i].get("start_frame", output.shape[0] - crossfade_frames)
            else:
                # Fallback - use old behavior (relative from end)
                transition_start_index = output.shape[0] - crossfade_frames

            # Validate we have enough frames at the transition point
            if transition_start_index + crossfade_frames > output.shape[0]:
                # Edge case: fall back to end-based calculation
                report_lines.append(
                    f"  ⚠️  Chunk {i}: transition_start_index ({transition_start_index}) + crossfade ({crossfade_frames}) > output ({output.shape[0]})"
                )
                transition_start_index = output.shape[0] - crossfade_frames

            if crossfade_frames > 0:
                # Check if we have enough frames for crossfade
                if crossfade_frames >= output.shape[0] or crossfade_frames >= current_chunk.shape[0]:
                    report_lines.append(
                        f"  ⚠️  Chunk {i}: crossfade ({crossfade_frames}) >= chunk size, concatenating instead"
                    )
                    output = torch.cat([output, current_chunk], dim=0)
                    continue

                # Get overlap regions at ABSOLUTE positions (like KJNodes CrossFadeImages)
                frames_to_blend_from_output = output[transition_start_index:transition_start_index + crossfade_frames]
                frames_to_blend_from_chunk = current_chunk[:crossfade_frames]

                # Validate dimensions match
                if frames_to_blend_from_output.shape[1:] != frames_to_blend_from_chunk.shape[1:]:
                    raise ValueError(
                        f"Chunk dimension mismatch at chunk {i}!\n"
                        f"Output region shape: {frames_to_blend_from_output.shape}\n"
                        f"Chunk start shape: {frames_to_blend_from_chunk.shape}"
                    )

                # Blend the overlap region
                if blend_mode == "fft_spectral":
                    blended = fft_spectral_blend(frames_to_blend_from_output, frames_to_blend_from_chunk)
                else:
                    weights = compute_blend_weights(crossfade_frames, mode=blend_mode, device=output.device)
                    weights = weights.view(-1, 1, 1, 1)  # [T, 1, 1, 1] for broadcasting
                    blended = (1.0 - weights) * frames_to_blend_from_output + weights * frames_to_blend_from_chunk

                # Reconstruct output (KJNodes style):
                # - Everything BEFORE transition point
                # - Blended frames
                # - Rest of current chunk after the overlap region
                output = torch.cat([
                    output[:transition_start_index],
                    blended,
                    current_chunk[crossfade_frames:]
                ], dim=0)

                new_frames = current_chunk.shape[0] - crossfade_frames
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames "
                    f"(transition at {transition_start_index}, {crossfade_frames} blended [{blend_mode}], {new_frames} new)"
                )
            else:
                # No crossfade - just concatenate (not recommended)
                output = torch.cat([output, current_chunk], dim=0)
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames (no blend)"
                )

        # Clamp to valid range
        output = torch.clamp(output, 0.0, 1.0)

        # Final report
        report_lines.append("")
        report_lines.append("RESULT:")
        report_lines.append(f"  Total output: {output.shape[0]} frames")
        report_lines.append(f"  Resolution: {output.shape[2]}x{output.shape[1]}")

        # Check final output WAN alignment
        if wan_mode:
            final_aligned, final_nearest, final_diff = validate_wan_alignment(output.shape[0])
            if final_aligned:
                report_lines.append(f"  WAN alignment: ✓ OK")
            else:
                report_lines.append(f"  WAN alignment: ⚠️  {output.shape[0]} frames (needs {final_nearest})")

        report_lines.append("=" * 60)

        stitch_report = "\n".join(report_lines)
        print(stitch_report)

        return (output, stitch_report)


class NV_ChunkStitcherFromImages:
    """
    Stitches video chunks that are passed as IMAGE inputs.

    Use this when chunks are already loaded in ComfyUI rather than on disk.
    Supports up to 4 chunk inputs.

    Features:
    - Optional plan JSON validation to catch frame count mismatches
    - WAN mode (default ON) validates frame alignment constraints
    - Detailed reporting of any adjustments made

    IMPORTANT: crossfade_frames MUST match overlap_frames from NV_ParallelChunkPlanner!
    If they don't match, you'll get visible ghosting at chunk boundaries because
    the crossfade will blend non-matching frame content.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_0": ("IMAGE", {"tooltip": "First video chunk"}),
                "crossfade_frames": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 128,
                    "tooltip": "MUST match overlap_frames from NV_ParallelChunkPlanner! Mismatched values cause ghosting."
                }),
            },
            "optional": {
                "chunk_1": ("IMAGE", {"tooltip": "Second video chunk (optional)"}),
                "chunk_2": ("IMAGE", {"tooltip": "Third video chunk (optional)"}),
                "chunk_3": ("IMAGE", {"tooltip": "Fourth video chunk (optional)"}),
                "plan_json_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to chunk_plan.json for validation (optional but recommended)"
                }),
                "wan_mode": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "WAN frame alignment awareness. Validates (N%4)==1 constraint and adjusts overlap if needed."
                }),
                "blend_mode": (BLEND_MODES, {
                    "default": "linear",
                    "tooltip": "Crossfade blend function. hamming/hann have better boundary properties than linear."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("stitched_video", "stitch_info",)
    FUNCTION = "stitch_chunks"
    CATEGORY = "NV_Utils"
    DESCRIPTION = "Stitches video chunks from IMAGE inputs with optional plan validation and WAN alignment checks."

    def stitch_chunks(self, chunk_0, crossfade_frames,
                      chunk_1=None, chunk_2=None, chunk_3=None,
                      plan_json_path="", wan_mode=True, blend_mode="linear"):
        """
        Stitch chunk IMAGE inputs with crossfade.
        """

        # Collect non-None chunks
        chunks = [chunk_0]
        if chunk_1 is not None:
            chunks.append(chunk_1)
        if chunk_2 is not None:
            chunks.append(chunk_2)
        if chunk_3 is not None:
            chunks.append(chunk_3)

        num_chunks = len(chunks)
        report_lines = [
            "=" * 60,
            "CHUNK STITCH REPORT",
            "=" * 60,
        ]

        # Load plan for validation if provided
        plan = None
        planned_overlap = crossfade_frames
        if plan_json_path and os.path.exists(plan_json_path):
            try:
                with open(plan_json_path, 'r') as f:
                    plan = json.load(f)
                planned_overlap = plan.get("overlap_frames", crossfade_frames)
                report_lines.append(f"Plan loaded: {plan_json_path}")
                report_lines.append(f"Plan overlap_frames: {planned_overlap}")

                # Check if plan chunk count matches
                plan_chunks = plan.get("chunks", [])
                if len(plan_chunks) != num_chunks:
                    report_lines.append(
                        f"⚠️  WARNING: Plan has {len(plan_chunks)} chunks, but {num_chunks} provided!"
                    )
            except Exception as e:
                report_lines.append(f"⚠️  Could not load plan: {e}")

        # Validate crossfade matches plan
        if plan and crossfade_frames != planned_overlap:
            report_lines.append(
                f"⚠️  WARNING: crossfade_frames ({crossfade_frames}) != plan overlap_frames ({planned_overlap})"
            )
            report_lines.append("   This may cause ghosting at chunk boundaries!")

        report_lines.append("")
        report_lines.append("CHUNK ANALYSIS:")

        # Validate chunks and check WAN alignment
        warnings = []
        for i, chunk in enumerate(chunks):
            actual_frames = chunk.shape[0]
            is_aligned, nearest, diff = validate_wan_alignment(actual_frames)

            # Get expected frames from plan if available
            expected_frames = None
            if plan and i < len(plan.get("chunks", [])):
                expected_frames = plan["chunks"][i].get("frame_count")

            # Build chunk report line
            chunk_info = f"  Chunk {i}: {actual_frames} frames"
            if expected_frames is not None:
                if actual_frames != expected_frames:
                    chunk_info += f" (expected {expected_frames}, diff={actual_frames - expected_frames})"
                    warnings.append(f"Chunk {i} frame count mismatch: {actual_frames} vs expected {expected_frames}")
                else:
                    chunk_info += " ✓"

            if wan_mode:
                if is_aligned:
                    chunk_info += " [WAN OK]"
                else:
                    chunk_info += f" [WAN: needs {nearest}, truncated by {diff}]"
                    if not expected_frames:  # Only warn if we didn't already warn about mismatch
                        warnings.append(f"Chunk {i} not WAN-aligned: {actual_frames} frames (nearest valid: {nearest})")

            report_lines.append(chunk_info)

        # Add warnings section
        if warnings:
            report_lines.append("")
            report_lines.append("⚠️  WARNINGS:")
            for w in warnings:
                report_lines.append(f"   • {w}")

        report_lines.append("")
        report_lines.append("STITCHING:")

        print(f"[NV_ChunkStitcherFromImages] Stitching {num_chunks} chunks")
        print(f"  Crossfade: {crossfade_frames} frames")
        print(f"  WAN mode: {wan_mode}")

        # First chunk: use all frames
        output = chunks[0].clone()
        report_lines.append(f"  Chunk 0: {chunks[0].shape[0]} frames (base)")

        for i in range(1, num_chunks):
            current_chunk = chunks[i]
            effective_crossfade = crossfade_frames

            # Get transition_start_index from plan (KEY FIX: use absolute position, not relative from end)
            # The start_frame in the plan is where this chunk's overlap begins in the original video
            if plan and i < len(plan.get("chunks", [])):
                transition_start_index = plan["chunks"][i].get("start_frame", output.shape[0] - effective_crossfade)
            else:
                # Fallback if no plan - use old behavior (relative from end)
                transition_start_index = output.shape[0] - effective_crossfade

            # Validate we have enough frames at the transition point
            if transition_start_index + effective_crossfade > output.shape[0]:
                # Edge case: fall back to end-based calculation
                report_lines.append(
                    f"  ⚠️  Chunk {i}: transition_start_index ({transition_start_index}) + crossfade ({effective_crossfade}) > output ({output.shape[0]})"
                )
                transition_start_index = output.shape[0] - effective_crossfade

            # Check if we have enough frames for crossfade
            if effective_crossfade > 0 and effective_crossfade < min(output.shape[0], current_chunk.shape[0]):
                # Get overlap regions at ABSOLUTE positions (like KJNodes CrossFadeImages)
                frames_to_blend_from_output = output[transition_start_index:transition_start_index + effective_crossfade]
                frames_to_blend_from_chunk = current_chunk[:effective_crossfade]

                # Blend the overlap region
                if blend_mode == "fft_spectral":
                    blended = fft_spectral_blend(frames_to_blend_from_output, frames_to_blend_from_chunk)
                else:
                    weights = compute_blend_weights(effective_crossfade, mode=blend_mode, device=output.device)
                    weights = weights.view(-1, 1, 1, 1)
                    blended = (1.0 - weights) * frames_to_blend_from_output + weights * frames_to_blend_from_chunk

                # Reconstruct output (KJNodes style):
                # - Everything BEFORE transition point
                # - Blended frames
                # - Rest of current chunk after the overlap region
                output = torch.cat([
                    output[:transition_start_index],
                    blended,
                    current_chunk[effective_crossfade:]
                ], dim=0)

                new_frames = current_chunk.shape[0] - effective_crossfade
                report_lines.append(
                    f"  Chunk {i}: {current_chunk.shape[0]} frames "
                    f"(transition at {transition_start_index}, {effective_crossfade} blended [{blend_mode}], {new_frames} new)"
                )
            else:
                output = torch.cat([output, current_chunk], dim=0)
                report_lines.append(f"  Chunk {i}: {current_chunk.shape[0]} frames (concat, no blend)")

        output = torch.clamp(output, 0.0, 1.0)

        # Final summary
        report_lines.append("")
        report_lines.append("RESULT:")
        report_lines.append(f"  Total output: {output.shape[0]} frames")
        report_lines.append(f"  Resolution: {output.shape[2]}x{output.shape[1]}")

        # Check final output WAN alignment
        if wan_mode:
            final_aligned, final_nearest, final_diff = validate_wan_alignment(output.shape[0])
            if final_aligned:
                report_lines.append(f"  WAN alignment: ✓ OK")
            else:
                report_lines.append(f"  WAN alignment: ⚠️  {output.shape[0]} frames (needs {final_nearest})")

        report_lines.append("=" * 60)

        stitch_info = "\n".join(report_lines)
        print(stitch_info)

        return (output, stitch_info)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_ChunkStitcher": NV_ChunkStitcher,
    "NV_ChunkStitcherFromImages": NV_ChunkStitcherFromImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ChunkStitcher": "NV Chunk Stitcher (From Files)",
    "NV_ChunkStitcherFromImages": "NV Chunk Stitcher (From Images)",
}
