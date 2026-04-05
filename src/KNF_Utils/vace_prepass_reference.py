"""
NV VACE Pre-Pass Reference

Multi-frame VACE reference conditioning for cascaded video generation.

Accepts reference frames (current chunk's Kling output) and an optional
identity anchor (chunk 0's Kling output, frozen across all chunks) to
combat identity drift in chunked pipelines.

Prepend order: [identity_anchor, reference_frames, control_video]
- identity_anchor at t=0: exploits WAN 2.2 training prior (t=0 = identity source)
- reference_frames after: RoPE locality gives stronger attention to generation start
- control_video last: grey masked inpaint target (mask=1, generate regions)

Key differences from native WanVaceToVideo:
- Multiple reference frames (not just 1)
- Optional cross-chunk identity anchor for drift prevention
- Sharpness quality floor on identity anchor frames
- Frame repeat (4x default) for 3D VAE temporal compression preservation
- Uniform or adaptive (IFS) sampling of N reference frames from each pool
- Internal upscaling of reference frames to target resolution

Based on research validation:
- SkyReels-A2 (arXiv 2504.02436): frame_repeat=4 prevents 6.5% identity drop
- FlashVideo (arXiv 2502.05179): cascaded is HIGHER quality than single-stage
- HiStream (arXiv 2512.21338): cascaded validated on Wan 2.1
- VINs (arXiv 2503.17539): parallel chunks converge with shared global signal
- LongDiff (CVPR 2025): Informative Frame Selection +1.8% subject consistency vs uniform

Identity-anchor-first ordering validated against WAN 2.2 VACE architecture:
- Conv3d(kernel=(1,2,2)): no temporal mixing in patch embedding
- Full self-attention (non-causal) in VaceWanAttentionBlock
- RoPE encodes relative (t,h,w) distance — locality bias for refs near generation
- Training prior: t=0 = identity anchor (R2V conditioning pattern)
See: ref_selection_debate/vace_ordering_synthesis.md

Output format is identical to WanVaceToVideo:
- vace_frames: [1, 32, T_ref+T_ctrl, H/8, W/8]
- vace_mask: [1, 64, T_ref+T_ctrl, H/8, W/8]
- vace_strength: scalar float
"""

import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
import comfy.model_management
import comfy.latent_formats
import node_helpers
from .streaming_vace_to_video import streaming_vae_encode


def _score_frames_ifs(frames):
    """LongDiff-style Informative Frame Selection (IFS).

    Score(k) = normalized_entropy(k) + normalized_SAD(k)

    - Entropy: Shannon entropy of grayscale histogram — high = visually complex
    - SAD: Sum of Absolute Differences with previous frame — high = content change

    Args:
        frames: [T, H, W, C] tensor in [0, 1]
    Returns:
        [T] numpy array of scores (higher = more informative)
    """
    # Work on CPU numpy for histogram speed
    gray = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]).cpu().numpy()
    T = gray.shape[0]

    # Entropy per frame: Shannon entropy of 256-bin histogram
    entropies = np.zeros(T)
    for i in range(T):
        hist, _ = np.histogram(gray[i], bins=256, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total > 0:
            p = hist / total
            p = p[p > 0]
            entropies[i] = -np.sum(p * np.log2(p))

    # SAD per frame: sum of absolute differences with previous frame
    # Frame 0 gets max SAD so it's always a strong candidate (first-frame dominance)
    sads = np.zeros(T)
    for i in range(1, T):
        sads[i] = np.abs(gray[i] - gray[i - 1]).mean()
    sads[0] = sads.max() if T > 1 else 1.0

    # Normalize each to [0, 1] before combining
    e_min, e_max = entropies.min(), entropies.max()
    if e_max > e_min:
        entropies = (entropies - e_min) / (e_max - e_min)
    else:
        entropies[:] = 1.0

    s_min, s_max = sads.min(), sads.max()
    if s_max > s_min:
        sads = (sads - s_min) / (s_max - s_min)
    else:
        sads[:] = 1.0

    return entropies + sads


def _compute_sharpness(frames):
    """Compute per-frame sharpness via Laplacian variance (no OpenCV dependency).

    Uses a 3×3 Laplacian kernel convolved over grayscale frames. Higher variance
    = sharper image. Returns one score per frame.

    Args:
        frames: [T, H, W, C] tensor in [0, 1]
    Returns:
        [T] numpy array of sharpness scores (Laplacian variance)
    """
    gray = (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])
    # Scale to [0, 255] so Laplacian variance matches OpenCV convention
    # (threshold=50 is meaningful; [0,1] range would give ~0.001 values)
    gray = (gray * 255.0).to(torch.float32).unsqueeze(1)  # [T, 1, H, W]
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=gray.device
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.flatten(1).var(dim=1).cpu().numpy()


def _sample_frames(frames, count, mode):
    """Sample frames using uniform or adaptive (IFS) selection.

    Args:
        frames: [T, H, W, C] tensor
        count: number of frames to select
        mode: "uniform" or "adaptive (IFS)"
    Returns:
        (indices, scores_or_none) — list of selected frame indices, and IFS scores if adaptive
    """
    total = frames.shape[0]
    count = min(count, total)

    if count >= total:
        return list(range(total)), None

    if mode == "uniform":
        indices = torch.linspace(0, total - 1, count).long().tolist()
        return indices, None

    # Adaptive (IFS): score all frames, pick best per temporal bin
    scores = _score_frames_ifs(frames)
    bin_edges = np.linspace(0, total, count + 1).astype(int)
    indices = []
    for i in range(count):
        start, end = bin_edges[i], bin_edges[i + 1]
        if start >= end:
            start = max(0, end - 1)
        best_in_bin = start + int(scores[start:end].argmax())
        indices.append(best_in_bin)
    return indices, scores


class NV_VacePrePassReference:
    """
    Multi-frame VACE reference conditioning for pre-pass cascaded generation.

    Accepts reference frames (current chunk's Kling output) and an optional
    identity anchor (chunk 0's Kling output) to combat cross-chunk identity drift.

    Prepend order: [identity_anchor, reference_frames, previous_chunk_tail, control_video]
    - identity_anchor at t=0: WAN 2.2 training prior treats t=0 as identity source
    - reference_frames after: RoPE locality for pose/expression guidance
    - previous_chunk_tail last: maximum RoPE locality to generation start for seam continuity
    - control_video: grey masked inpaint target (mask=1 generate)

    Wiring per chunk:
    - Chunk 0: reference_frames=Chunk0 Kling, identity_anchor=not connected, tail=not connected
    - Chunk N: reference_frames=ChunkN Kling, identity_anchor=Chunk0 Kling, tail=ChunkN-1 last frames
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 8192, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4,
                           "tooltip": "Number of output video frames (pixel space)"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1000.0, "step": 0.01,
                            "tooltip": "VACE conditioning strength for the control video. "
                                       "This is the entry-level vace_strength applied to both references and control."}),
                "ref_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                "tooltip": "Effective strength for reference frames. When different from 'strength', "
                                           "reference latents are pre-scaled so their effective influence equals this value "
                                           "after the model applies vace_strength. E.g., strength=0.35 + ref_strength=1.0 "
                                           "means beauty control at 0.35, references at 1.0."}),
                "reference_frames": ("IMAGE", {
                    "tooltip": "This chunk's Kling output — the primary identity/pose/expression reference. "
                               "These frames tell VACE what the face/body should look like in the masked region. "
                               "Sampled down to num_refs frames using ref_sampling mode."
                }),
                "num_refs": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1,
                             "tooltip": "Number of reference frames to sample from reference_frames. "
                                        "1-5 is the sweet spot; >5 gives diminishing returns."}),
                "ref_sampling": (["uniform", "adaptive (IFS)"], {
                    "default": "uniform",
                    "tooltip": "How to select reference frames. 'uniform' = evenly spaced. "
                               "'adaptive (IFS)' = LongDiff-style Informative Frame Selection: "
                               "scores each frame by image entropy + temporal change, picks the "
                               "best per temporal bin. Better for content with uneven motion."
                }),
                "frame_repeat": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1,
                                "tooltip": "Repeat each reference N times for 3D VAE temporal compression "
                                           "(4 recommended for Wan 2.1, prevents 6.5%% identity drop per SkyReels-A2)"}),
            },
            "optional": {
                "identity_anchor": ("IMAGE", {
                    "tooltip": "Cross-chunk identity lock — typically chunk 0's Kling output, frozen for all chunks. "
                               "Prepended FIRST at t=0, exploiting WAN 2.2's training prior that treats t=0 as the "
                               "primary identity source. Only needed for chunks 1+ to prevent identity drift. "
                               "Subject to sharpness quality floor (min_sharpness)."
                }),
                "num_anchors": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1,
                                "tooltip": "Number of identity anchor frames to sample. "
                                           "1-3 recommended for identity anchoring."}),
                "anchor_sampling": (["uniform", "adaptive (IFS)"], {
                    "default": "uniform",
                    "tooltip": "How to select identity anchor frames."
                }),
                "min_sharpness": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                                  "tooltip": "Minimum Laplacian variance for identity anchor frames. Frames below this "
                                             "threshold are rejected before sampling. Set to 0 to disable. "
                                             "reference_frames bypass this filter (current chunk's own output)."}),
                "previous_chunk_tail": ("IMAGE", {
                    "tooltip": "Last N frames of previous chunk's raw Kling crop output (pre-stitch). "
                               "Prepended LAST in the ref block, immediately before control_video, for "
                               "maximum RoPE locality to the first generated frames. Provides fine-detail "
                               "continuity across chunk boundaries via late-block VACE skip connections. "
                               "Leave unconnected for chunk 0 or single-chunk runs."
                }),
                "num_tail_frames": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1,
                    "tooltip": "Number of tail frames to take from the END of previous_chunk_tail. "
                               "0 = disabled (tail input ignored). At frame_repeat=4, each tail frame "
                               "= 1 latent frame. 2 is the recommended default (March 2026 synthesis)."}),
                "control_video": ("IMAGE", {
                    "tooltip": "Grey masked inpaint video — the generation target. Mask=1 regions are where "
                               "VACE will paint new content using the reference frames as identity guide."
                }),
                "control_masks": ("MASK", {
                    "tooltip": "Inpaint mask for control video. mask=1 = generate, mask=0 = preserve."
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "LATENT", "INT",)
    RETURN_NAMES = ("positive", "negative", "latent", "ref_latent", "trim_latent",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Multi-frame VACE reference for cascaded pre-pass workflows. "
        "Wire this chunk's Kling output to reference_frames. "
        "Optionally wire chunk 0's Kling output to identity_anchor for cross-chunk identity lock. "
        "Optionally wire previous chunk's last frames to previous_chunk_tail for seam continuity. "
        "Wire grey masked inpaint video to control_video."
    )

    def execute(self, positive, negative, vae, width, height, length, batch_size, strength,
                ref_strength, reference_frames, num_refs, ref_sampling, frame_repeat,
                identity_anchor=None, num_anchors=3, anchor_sampling="uniform",
                min_sharpness=50.0, previous_chunk_tail=None, num_tail_frames=2,
                control_video=None, control_masks=None):

        latent_length = ((length - 1) // 4) + 1

        # === Step 1a: Sample reference frames (current chunk's Kling output) ===
        ref_indices, ref_scores = _sample_frames(reference_frames, num_refs, ref_sampling)
        sampled_refs = reference_frames[ref_indices]
        print(f"[NV_VacePrePassReference] Sampled {len(ref_indices)} reference frames "
              f"(indices: {ref_indices}, mode: {ref_sampling}) from {reference_frames.shape[0]} available")
        if ref_scores is not None:
            print(f"[NV_VacePrePassReference] Ref IFS scores: "
                  f"{', '.join(f'[{idx}]={ref_scores[idx]:.3f}' for idx in ref_indices)}")

        # === Step 1b: Sample identity anchor frames (optional, with quality floor) ===
        sampled_anchors = None
        if identity_anchor is not None:
            total_anchors_available = identity_anchor.shape[0]

            # Apply sharpness quality floor (reference_frames bypass this — current chunk's own output)
            if min_sharpness > 0:
                sharpness = _compute_sharpness(identity_anchor)
                passed_mask = sharpness >= min_sharpness
                passed_count = int(passed_mask.sum())
                print(f"[NV_VacePrePassReference] Identity anchor sharpness filter: "
                      f"{passed_count}/{total_anchors_available} passed "
                      f"(threshold={min_sharpness:.1f}, "
                      f"range={sharpness.min():.2f}-{sharpness.max():.2f})")

                if passed_count == 0:
                    print(f"[NV_VacePrePassReference] WARNING: All identity anchor frames below sharpness threshold. "
                          f"Skipping identity anchor entirely.")
                else:
                    passed_indices = np.where(passed_mask)[0]
                    passed_frames = identity_anchor[passed_indices]
                    anchor_count = min(num_anchors, passed_count)
                    sub_indices, anchor_scores = _sample_frames(passed_frames, anchor_count, anchor_sampling)
                    original_indices = [int(passed_indices[i]) for i in sub_indices]
                    sampled_anchors = identity_anchor[original_indices]
                    print(f"[NV_VacePrePassReference] Sampled {len(original_indices)} identity anchor frames "
                          f"(original indices: {original_indices}, mode: {anchor_sampling})")
                    if anchor_scores is not None:
                        print(f"[NV_VacePrePassReference] Anchor IFS scores: "
                              f"{', '.join(f'[{sub_indices[j]}]={anchor_scores[sub_indices[j]]:.3f}' for j in range(len(sub_indices)))}")
            else:
                anchor_indices, anchor_scores = _sample_frames(identity_anchor, num_anchors, anchor_sampling)
                sampled_anchors = identity_anchor[anchor_indices]
                print(f"[NV_VacePrePassReference] Sampled {len(anchor_indices)} identity anchor frames "
                      f"(indices: {anchor_indices}, mode: {anchor_sampling}) "
                      f"from {total_anchors_available} available")
                if anchor_scores is not None:
                    print(f"[NV_VacePrePassReference] Anchor IFS scores: "
                          f"{', '.join(f'[{idx}]={anchor_scores[idx]:.3f}' for idx in anchor_indices)}")

        # === Step 1b-tail: Extract last N frames of previous chunk (optional) ===
        # Tail frames provide fine-detail continuity across chunk boundaries.
        # Positioned LAST in the ref block = closest to control_video = maximum RoPE
        # locality to first generated frames. VACE late-block skip connections (10 layers
        # across 40 DiT blocks) reassert tail detail during late denoising where
        # micro-texture (hair, fabric, speculars) is resolved.
        sampled_tail = None
        if previous_chunk_tail is not None and num_tail_frames > 0 and previous_chunk_tail.shape[0] > 0:
            total_tail_available = previous_chunk_tail.shape[0]
            actual_tail_count = min(num_tail_frames, total_tail_available)
            # Explicit start index to avoid Python's -0 == 0 gotcha
            # (tensor[-0:] returns the FULL tensor, not empty)
            start_idx = total_tail_available - actual_tail_count
            sampled_tail = previous_chunk_tail[start_idx:]
            print(f"[NV_VacePrePassReference] Tail: {actual_tail_count} frames from end of "
                  f"previous chunk ({total_tail_available} available)")

        # === Step 1c: Concatenate [identity_anchor, reference_frames, tail] ===
        # Order matters for RoPE temporal locality + WAN training prior:
        # - anchor FIRST: exploits WAN t=0 training prior for identity anchoring
        # - refs MIDDLE: pose/expression guidance with moderate RoPE locality
        # - tail LAST: maximum RoPE locality to generation start for seam continuity
        # Conv3d kernel_t=1 at patch embedding means no temporal mixing at entry —
        # ordering only affects attention (Stage 8) and skip injection (Stage 9).
        # Validate spatial dimensions match before concat (different crop
        # resolutions will cause a raw torch.cat RuntimeError otherwise)
        ref_shape = sampled_refs.shape[1:]  # [H, W, C]
        if sampled_anchors is not None and sampled_anchors.shape[1:] != ref_shape:
            raise ValueError(
                f"[NV_VacePrePassReference] Spatial mismatch: identity_anchor {list(sampled_anchors.shape[1:])} "
                f"vs reference_frames {list(ref_shape)}. Resize before connecting.")
        if sampled_tail is not None and sampled_tail.shape[1:] != ref_shape:
            raise ValueError(
                f"[NV_VacePrePassReference] Spatial mismatch: previous_chunk_tail {list(sampled_tail.shape[1:])} "
                f"vs reference_frames {list(ref_shape)}. Resize before connecting.")

        refs_list = []
        if sampled_anchors is not None:
            refs_list.append(sampled_anchors)
        refs_list.append(sampled_refs)
        if sampled_tail is not None:
            refs_list.append(sampled_tail)

        all_refs = torch.cat(refs_list, dim=0) if len(refs_list) > 1 else refs_list[0]

        # Log the composition
        log_parts = []
        if sampled_anchors is not None:
            log_parts.append(f"{sampled_anchors.shape[0]} anchor")
        log_parts.append(f"{sampled_refs.shape[0]} ref")
        if sampled_tail is not None:
            log_parts.append(f"{sampled_tail.shape[0]} tail")
        print(f"[NV_VacePrePassReference] Prepend: {' + '.join(log_parts)} = "
              f"{all_refs.shape[0]} total (order: {'→'.join(log_parts)})")

        # Frame repeat: duplicate each reference frame_repeat times in pixel space
        # [r1,r1,r1,r1, r2,r2,r2,r2, ...] for 3D VAE temporal compression
        # SkyReels-A2: "before VAE" assembly with 4x repeat is critical
        if frame_repeat > 1:
            repeated = all_refs.unsqueeze(1).expand(-1, frame_repeat, -1, -1, -1)
            repeated = repeated.reshape(-1, *all_refs.shape[1:])
        else:
            repeated = all_refs

        total_ref_pixels = repeated.shape[0]
        ref_latent_length = ((total_ref_pixels - 1) // 4) + 1

        print(f"[NV_VacePrePassReference] Frame repeat {frame_repeat}x -> "
              f"{total_ref_pixels} pixel frames -> {ref_latent_length} latent frames")

        # Upscale reference frames to target resolution
        repeated = comfy.utils.common_upscale(
            repeated.movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)

        # VAE encode reference frames (streaming for OOM safety)
        print(f"[NV_VacePrePassReference] Encoding {total_ref_pixels} reference frames...")
        ref_latent = streaming_vae_encode(vae, repeated[:, :, :, :3])
        # ref_latent shape: [1, 16, ref_latent_length, H/8, W/8]

        # Save 16ch encoded reference for the ref_latent output (used by
        # NV_PrependReferenceLatent in V2V workflows). Must be real encoded
        # content, NOT zeros — zeros create a massive latent-space discontinuity
        # at the reference/video boundary that causes noise bleed artifacts.
        ref_latent_16ch = ref_latent.clone()

        # Free pixel-space reference tensors
        del repeated, all_refs, sampled_refs
        if sampled_anchors is not None:
            del sampled_anchors

        # Build 32ch reference: 16ch encoded + 16ch neutral reactive
        # This matches native WanVaceToVideo reference_image encoding:
        #   cat([ref_latent, process_out(zeros_like(ref_latent))], dim=1)
        # The neutral reactive channel signals "nothing to generate here"
        ref_latent = torch.cat([
            ref_latent,
            comfy.latent_formats.Wan21().process_out(torch.zeros_like(ref_latent))
        ], dim=1)
        # ref_latent shape: [1, 32, ref_latent_length, H/8, W/8]

        # === Step 1d: Pre-scale reference latent for independent strength control ===
        # The entry-level vace_strength applies uniformly to the entire VACE entry
        # (both references and control frames). To give references a different effective
        # strength, we pre-scale the reference latent so that:
        #   effective_ref = ref_latent * ref_scale * vace_strength = ref_latent * ref_strength
        # This allows e.g. beauty control at 0.35 with references at 1.0
        if strength > 0 and abs(ref_strength - strength) > 1e-6:
            ref_scale = min(ref_strength / strength, 10.0)  # cap at 10x to prevent latent blowout
            ref_latent = ref_latent * ref_scale
            print(f"[NV_VacePrePassReference] Pre-scaled reference latent by {ref_scale:.2f}x "
                  f"(ref_strength={ref_strength}, vace_strength={strength}, "
                  f"effective reference influence={ref_strength})")

        # === Step 2: Process control video (same as native WanVaceToVideo) ===
        if control_video is not None:
            control_video = comfy.utils.common_upscale(
                control_video[:length].movedim(-1, 1),
                width, height, "bilinear", "center"
            ).movedim(1, -1)
            if control_video.shape[0] < length:
                control_video = torch.nn.functional.pad(
                    control_video,
                    (0, 0, 0, 0, 0, 0, 0, length - control_video.shape[0]),
                    value=0.5
                )
        else:
            control_video = torch.ones((length, height, width, 3)) * 0.5

        # === Step 3: Process masks ===
        if control_masks is None:
            mask = torch.ones((length, height, width, 1))
        else:
            mask = control_masks
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            mask = comfy.utils.common_upscale(
                mask[:length], width, height, "bilinear", "center"
            ).movedim(1, -1)
            if mask.shape[0] < length:
                mask = torch.nn.functional.pad(
                    mask,
                    (0, 0, 0, 0, 0, 0, 0, length - mask.shape[0]),
                    value=1.0
                )

        # === Step 4: Concept decoupling for control video ===
        # Split into inactive (preserved regions) and reactive (generated regions)
        control_video = control_video - 0.5
        inactive = (control_video * (1 - mask)) + 0.5
        reactive = (control_video * mask) + 0.5

        del control_video  # Free pixel-space control video

        # VAE encode control video (streaming for long videos)
        print(f"[NV_VacePrePassReference] Encoding {length} control frames (inactive)...")
        inactive = streaming_vae_encode(vae, inactive[:, :, :, :3])
        print(f"[NV_VacePrePassReference] Encoding {length} control frames (reactive)...")
        reactive = streaming_vae_encode(vae, reactive[:, :, :, :3])

        control_video_latent = torch.cat((inactive, reactive), dim=1)
        # control_video_latent shape: [1, 32, latent_length, H/8, W/8]

        del inactive, reactive  # Free intermediate latents

        # === Step 5: Prepend reference latent to control latent ===
        control_video_latent = torch.cat((ref_latent, control_video_latent), dim=2)
        # combined shape: [1, 32, ref_latent_length + latent_length, H/8, W/8]

        del ref_latent  # Free standalone reference latent

        # === Step 6: Mask downscaling to latent space ===
        # Same logic as native WanVaceToVideo (nodes_wan.py lines 346-352)
        vae_stride = 8
        height_mask = height // vae_stride
        width_mask = width // vae_stride
        mask = mask.view(length, height_mask, vae_stride, width_mask, vae_stride)
        mask = mask.permute(2, 4, 0, 1, 3)
        mask = mask.reshape(vae_stride * vae_stride, length, height_mask, width_mask)
        mask = torch.nn.functional.interpolate(
            mask.unsqueeze(0),
            size=(latent_length, height_mask, width_mask),
            mode='nearest-exact'
        ).squeeze(0)

        # Prepend zero-mask for reference frames (mask=0 = preserve)
        # Note: can't use zeros_like with slice — ref_latent_length may exceed mask.shape[1]
        mask_pad = torch.zeros(
            mask.shape[0], ref_latent_length, mask.shape[2], mask.shape[3],
            device=mask.device, dtype=mask.dtype
        )
        mask = torch.cat((mask_pad, mask), dim=1)

        # Update total latent length
        total_latent_length = latent_length + ref_latent_length
        trim_latent = ref_latent_length

        mask = mask.unsqueeze(0)

        # === Step 7: Set conditioning ===
        positive = node_helpers.conditioning_set_values(
            positive,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True
        )
        negative = node_helpers.conditioning_set_values(
            negative,
            {"vace_frames": [control_video_latent], "vace_mask": [mask], "vace_strength": [strength]},
            append=True
        )

        # === Step 8: Create output latents ===
        intermediate_device = comfy.model_management.intermediate_device()
        latent = torch.zeros(
            [batch_size, 16, total_latent_length, height // 8, width // 8],
            device=intermediate_device
        )
        out_latent = {"samples": latent}

        # Reference-only latent: real VAE-encoded reference frames for prepending
        # to a custom denoised latent in V2V workflows. Using real content (not zeros)
        # prevents latent-space discontinuity at the ref/video boundary that causes
        # noise bleed artifacts in the first frames after trimming.
        out_ref_latent = {"samples": ref_latent_16ch.to(intermediate_device)}
        del ref_latent_16ch

        print(f"[NV_VacePrePassReference] Done.")
        print(f"  VACE latent shape: {control_video_latent.shape}")
        print(f"  VACE mask shape: {mask.shape}")
        print(f"  Output latent shape: {latent.shape}")
        print(f"  ref_latent shape: {out_ref_latent['samples'].shape} (VAE-encoded reference for V2V prepend)")
        print(f"  trim_latent: {trim_latent} (reference latent frames to trim from output)")

        return (positive, negative, out_latent, out_ref_latent, trim_latent)


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_VacePrePassReference": NV_VacePrePassReference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VacePrePassReference": "NV VACE Pre-Pass Reference",
}
