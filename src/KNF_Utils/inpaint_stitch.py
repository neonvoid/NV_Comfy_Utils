"""
NV Inpaint Stitch v2 — Composites inpainted crops back into the original frames.

Pairs with NV_InpaintCrop v2 (inpaint_crop.py). Takes the STITCHER dict produced
by the crop node plus the inpainted image batch, and blends each frame back into
its canvas at the stored coordinates using the stored blend mask.

Handles both stitcher formats:
  - v2 (NV_InpaintCrop v2): 'resize_algorithm' (single key)
  - v1 (comfy-inpaint-crop-fork): 'downscale_algorithm' / 'upscale_algorithm'

Frame skipping: if the crop node skipped frames (empty mask), the stitch node
reinserts the original frames at their correct positions in the output batch.
"""

import gc

import torch
import torch.nn.functional as TF
import comfy.model_management

from .mask_ops import rescale_image, rescale_mask
from .multiband_blend_stitch import multiband_blend
from .guided_filter import refine_mask


# =============================================================================
# Memory-efficient stack
# =============================================================================

def _memory_efficient_stack(tensor_list, dim=0):
    """Pre-allocate output and copy frame-by-frame, releasing inputs as we go.

    torch.stack builds the output AND keeps all inputs alive simultaneously —
    peak memory = 2 × output_size. For 277-frame 1080p RGB that's ~13.8 GB
    during the stack call alone, which triggers Windows access violations
    under RAM pressure (e.g. during back-to-back param sweeps).

    This variant allocates the output once, copies each input into its slot,
    then nulls the list entry to release the reference. Peak ≈ output_size +
    one frame. Safe because the caller immediately `del`s the list after
    the stack call (see NV_InpaintStitch.stitch).

    Only supports dim=0 (the only case we hit). Falls back to torch.stack
    for other dims.
    """
    if not tensor_list:
        raise RuntimeError("Cannot stack empty list")
    if dim != 0:
        return torch.stack(tensor_list, dim=dim)

    first = tensor_list[0]
    n = len(tensor_list)
    out_shape = (n,) + tuple(first.shape)
    out = torch.empty(out_shape, dtype=first.dtype, device=first.device)
    for i in range(n):
        out[i] = tensor_list[i]
        tensor_list[i] = None  # release reference so GC can reclaim
    return out


# =============================================================================
# Inverse Content Warp
# =============================================================================

def _inverse_content_warp(image, mask, warp_mode, warp_entry):
    """Apply inverse content warp to undo crop stabilization before blending.

    Args:
        image: [1, H, W, C] — resized inpainted crop.
        mask: [1, H, W, 1] or None — optional blend mask. If None, only image is warped.
        warp_mode: 'centroid' or 'optical_flow'.
        warp_entry: dict with warp data for this frame.

    Returns:
        (image, mask) with inverse warp applied. Returned mask is None if input was None.
    """
    device = image.device
    _, H, W, C = image.shape

    if warp_mode == "centroid":
        dx = -warp_entry["dx"]
        dy = -warp_entry["dy"]
        norm_dx = 2.0 * dx / W
        norm_dy = 2.0 * dy / H
        theta = torch.tensor([
            [1.0, 0.0, norm_dx],
            [0.0, 1.0, norm_dy]
        ], device=device, dtype=torch.float32).unsqueeze(0)
        grid = TF.affine_grid(theta, (1, 1, H, W), align_corners=False)

        # Use zeros for both image and mask — compositing is strictly mask-gated,
        # so zero-filled edges are invisible in the final output. This avoids both
        # border-stretching artifacts AND reflection mirror ghosts.
        img_nchw = image.permute(0, 3, 1, 2)
        image = TF.grid_sample(img_nchw, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)

        if mask is not None:
            mask_nchw = mask.permute(0, 3, 1, 2)
            mask = TF.grid_sample(mask_nchw, grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=False).permute(0, 2, 3, 1)

    elif warp_mode == "optical_flow":
        flow = warp_entry["flow"].to(device)  # Forward flow: maps src→ref, used as-is for inverse grid
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)

        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        xx_warped = xx + flow[0, 0]
        yy_warped = yy + flow[0, 1]
        xx_norm = 2.0 * xx_warped / (W - 1) - 1.0
        yy_norm = 2.0 * yy_warped / (H - 1) - 1.0
        grid = torch.stack([xx_norm, yy_norm], dim=-1).unsqueeze(0)

        img_nchw = image.permute(0, 3, 1, 2)
        image = TF.grid_sample(img_nchw, grid, mode='bilinear',
                               padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)

        if mask is not None:
            mask_nchw = mask.permute(0, 3, 1, 2)
            mask = TF.grid_sample(mask_nchw, grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)

    return image, mask


# =============================================================================
# Core Stitch Function
# =============================================================================

def stitch_single_frame(canvas_image, inpainted_image, mask,
                        ctc_x, ctc_y, ctc_w, ctc_h,
                        cto_x, cto_y, cto_w, cto_h,
                        resize_algorithm,
                        warp_mode=None, warp_entry=None,
                        blend_mode="multiband", multiband_levels=5,
                        guided_refine=False, guided_radius=8, guided_eps=0.001,
                        guided_strength=0.7):
    """Blend one inpainted crop back into its canvas and extract the original region.

    Args:
        canvas_image: [H, W, C] or [1, H, W, C] — the padded original frame.
        inpainted_image: [1, H, W, C] — the inpainted crop at target resolution.
        mask: [H, W] or [1, H, W] — blend mask at crop resolution.
        ctc_x/y/w/h: Where the crop sits on the canvas.
        cto_x/y/w/h: Where the original image sits on the canvas.
        resize_algorithm: Interpolation method for resizing crop back to canvas scale.
        warp_mode: Optional content warp mode ('centroid' or 'optical_flow').
        warp_entry: Optional per-frame warp data dict.
        blend_mode: 'alpha' (standard), 'multiband' (Laplacian pyramid), or 'hard' (no blend).
        multiband_levels: Pyramid levels for multiband mode (default 5).
        guided_strength: Lerp strength for guided filter (0.0=original, 1.0=fully refined).

    Returns:
        Tuple of:
          [1, cto_h, cto_w, C] output image (original image region with inpainted area blended in).
          [1, ctc_h, ctc_w] final blend mask used for compositing (after guided refinement if enabled).
    """
    device = canvas_image.device

    # Ensure canvas has batch dim
    if canvas_image.dim() == 3:
        canvas_image = canvas_image.unsqueeze(0)

    # Ensure mask has batch dim
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # Inverse content warp BEFORE resize: dx/dy are in the crop's working resolution
    # (target res from InpaintCrop), so must be applied before downsizing to canvas scale.
    # Only warp the IMAGE, not the blend mask — the mask was created at InpaintCrop time
    # in original (non-stabilized) coordinates and was never forward-warped by CoTrackerBridge.
    # After inverse-warping the image back to original coordinates, the mask already aligns.
    if warp_mode is not None and warp_entry is not None:
        inpainted_image, _ = _inverse_content_warp(
            inpainted_image, None, warp_mode, warp_entry
        )

    # Resize inpainted image and mask to canvas crop size
    _, h, w, _ = inpainted_image.shape
    if ctc_w != w or ctc_h != h:
        resized_image = rescale_image(inpainted_image, ctc_w, ctc_h, resize_algorithm)
        resized_mask = rescale_mask(mask, ctc_w, ctc_h, resize_algorithm)
    else:
        resized_image = inpainted_image
        resized_mask = mask

    # Clamp mask
    resized_mask = resized_mask.clamp(0, 1)

    # Extract canvas region
    canvas_crop = canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :]

    # Guided filter: refine blend mask using canvas crop edges (applied before blend)
    if guided_refine:
        resized_mask = refine_mask(
            mask=resized_mask, guide_image=canvas_crop,
            radius=guided_radius, eps=guided_eps, strength=guided_strength, mode="color",
        )

    # Expand mask to match image channels
    resized_mask = resized_mask.unsqueeze(-1)  # [1, H, W, 1]

    if blend_mode == "multiband":
        # Laplacian pyramid: blend each frequency band at appropriate spatial scale.
        # Operates on [B,C,H,W] — convert from [1,H,W,C].
        inp_nchw = resized_image.permute(0, 3, 1, 2)
        cvs_nchw = canvas_crop.permute(0, 3, 1, 2)
        mask_nchw = resized_mask.permute(0, 3, 1, 2)[:, :1]  # [1, 1, H, W]
        blended_nchw = multiband_blend(inp_nchw, cvs_nchw, mask_nchw, num_levels=multiband_levels)
        blended = blended_nchw.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    elif blend_mode == "hard":
        # Hard paste: binary threshold mask, no feathering
        hard_mask = (resized_mask > 0.5).float()
        blended = hard_mask * resized_image + (1.0 - hard_mask) * canvas_crop
    else:
        # Standard alpha composite
        blended = resized_mask * resized_image + (1.0 - resized_mask) * canvas_crop

    canvas_image[:, ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w, :] = blended

    # Extract the original image area from canvas
    output_image = canvas_image[:, cto_y:cto_y + cto_h, cto_x:cto_x + cto_w, :]

    # Return the final blend mask (after guided refinement) for consistent stitch_mask output
    # resized_mask is [1, H, W, 1] at this point — squeeze channel dim
    final_mask = resized_mask.squeeze(-1)  # [1, H, W]

    return output_image, final_mask


def _build_stitch_mask(blend_mask, ctc_x, ctc_y, ctc_w, ctc_h,
                       cto_x, cto_y, cto_w, cto_h, resize_algorithm):
    """Build a per-frame mask [cto_h, cto_w] showing where inpainted content was composited.

    1.0 = inpainted region, 0.0 = original content. Useful for downstream
    BoundaryColorMatch or StitchBoundaryMask nodes.
    """
    # Resize blend mask to canvas crop size
    if blend_mask.dim() == 2:
        blend_mask = blend_mask.unsqueeze(0)

    if ctc_w != blend_mask.shape[-1] or ctc_h != blend_mask.shape[-2]:
        blend_mask = rescale_mask(blend_mask, ctc_w, ctc_h, resize_algorithm)

    blend_mask = blend_mask.clamp(0, 1)

    # Create full canvas-sized mask
    canvas_h = cto_y + cto_h
    canvas_w = cto_x + cto_w
    # Canvas must be at least large enough for both crop region and orig region
    canvas_h = max(canvas_h, ctc_y + ctc_h)
    canvas_w = max(canvas_w, ctc_x + ctc_w)

    canvas_mask = torch.zeros(canvas_h, canvas_w, device=blend_mask.device)
    canvas_mask[ctc_y:ctc_y + ctc_h, ctc_x:ctc_x + ctc_w] = blend_mask.squeeze(0)

    # Extract the original image region
    return canvas_mask[cto_y:cto_y + cto_h, cto_x:cto_x + cto_w]


def _get_resize_algorithm(stitcher):
    """Extract resize algorithm from stitcher, handling both v1 and v2 formats."""
    # v2 format (NV_InpaintCrop v2): single 'resize_algorithm'
    if 'resize_algorithm' in stitcher:
        return stitcher['resize_algorithm']
    # v1 format (fork): separate up/downscale — just use upscale as default
    return stitcher.get('upscale_algorithm', stitcher.get('downscale_algorithm', 'bicubic'))


# =============================================================================
# Node Class
# =============================================================================

class NV_InpaintStitch:
    """Stitch inpainted crops back into original frames using STITCHER metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "inpainted_image": ("IMAGE",),
                "blend_mode": (["multiband", "alpha", "hard"], {
                    "default": "multiband",
                    "tooltip": "multiband: Laplacian pyramid — blends low freq broadly, "
                               "high freq narrowly (best for structural seam hiding, recommended default). "
                               "alpha: standard feathered blend (simpler, faster). "
                               "hard: binary mask paste, no feathering."
                }),
            },
            "optional": {
                "multiband_levels": ("INT", {
                    "default": 5, "min": 2, "max": 8, "step": 1,
                    "tooltip": "Pyramid levels for multiband mode. 5 = good for 720p-1080p. "
                               "More levels = broader low-frequency blending."
                }),
                "guided_refine": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Refine the blend mask using guided filter before compositing. "
                               "Snaps mask edges to actual image boundaries — reduces ghosting at "
                               "hair/clothing edges. Uses canvas crop as guide (stable source)."
                }),
                "guided_radius": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Guided filter window radius. 8 = good for 512px crops, "
                               "12-16 for 720p+. Larger = broader smoothing."
                }),
                "guided_eps": ("FLOAT", {
                    "default": 0.001, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Guided filter edge sensitivity. Lower = sharper snap to edges. "
                               "0.001 = default, 0.01 = smoother (less aggressive edge tracking)."
                }),
                "guided_strength": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Lerp between original mask (0.0) and guided-refined mask (1.0). "
                               "0.7 = safe default that prevents mask erosion from misleading gradients. "
                               "1.0 = fully trust the guided filter. 0.0 = no refinement."
                }),
                "output_dtype": (["fp16", "fp32"], {
                    "default": "fp16",
                    "tooltip": "Output tensor dtype. fp16 (default, since 2026-04-24): halves CPU "
                               "RAM (3.45 GB instead of 6.86 GB for 277-frame 1080p). Output goes "
                               "to video encoder which converts to uint8 anyway, so fp16 has zero "
                               "visible impact on final delivery. fp32: legacy behavior — only "
                               "needed if downstream nodes require fp32 precision (e.g., feeding "
                               "into another high-precision color operation before video save). "
                               "If you OOM with fp32 (allocator can't allocate ~6.9 GB twice for "
                               "back-to-back stitches), drop to fp16."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "stitch_mask")
    FUNCTION = "stitch"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Composites inpainted crops back into the original frames. "
        "Pairs with NV Inpaint Crop v2. Handles frame skipping automatically. "
        "Supports alpha, multiband (Laplacian pyramid), or hard blend modes."
    )

    def stitch(self, stitcher, inpainted_image, blend_mode="multiband", multiband_levels=5,
               guided_refine=False, guided_radius=8, guided_eps=0.001, guided_strength=0.7,
               output_dtype="fp16"):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        resize_algorithm = _get_resize_algorithm(stitcher)
        # Resolve output dtype (default fp16 — see pre-allocation block below for why)
        output_dtype_torch = torch.float16 if output_dtype == "fp16" else torch.float32

        skipped_indices_raw = stitcher.get('skipped_indices', [])
        skipped_indices = set(skipped_indices_raw)
        original_frames = stitcher.get('original_frames', [])
        total_frames = stitcher.get('total_frames', inpainted_image.shape[0])

        # Content warp data (backward-compatible — absent in old stitchers)
        content_warp_mode = stitcher.get('content_warp_mode', None)
        content_warp_data = stitcher.get('content_warp_data', None)

        # Detect single_stitcher mode (one coordinate set broadcast across batch)
        num_coords = len(stitcher.get('cropped_to_canvas_x', []))
        batch_size = inpainted_image.shape[0]
        single_stitcher = (num_coords == 1 and batch_size > 1)

        # --- Entry debug logging ---
        print(f"[NV_InpaintStitch] Starting: {batch_size} inpainted frames, "
              f"total_frames={total_frames}, skipped={len(skipped_indices)}, "
              f"coords={num_coords}, single_stitcher={single_stitcher}, "
              f"warp={content_warp_mode or 'none'}, blend={blend_mode}")
        print(f"[NV_InpaintStitch]   inpainted shape: {list(inpainted_image.shape)}, "
              f"canvas count: {len(stitcher.get('canvas_image', []))}, "
              f"warp_data: {len(content_warp_data) if content_warp_data else 0} entries")

        # --- Frame count validation ---
        # Prevents silent overrun when upstream changes frame count (e.g., tail prepend,
        # WAN 4k+1 snapping, trim mismatch). Without this, the loop walks past the end
        # of inpainted_image and produces empty [0,H,W,C] slices that crash.
        expected_inpainted = total_frames - len(skipped_indices)

        if not single_stitcher and batch_size != expected_inpainted:
            # Build detailed diagnostic
            diag = (
                f"[NV_InpaintStitch] FRAME MISMATCH!\n"
                f"  Expected {expected_inpainted} inpainted frames "
                f"(total_frames={total_frames} - skipped={len(skipped_indices)})\n"
                f"  Received {batch_size} frames, shape={list(inpainted_image.shape)}\n"
                f"  Stitcher metadata: canvas_images={len(stitcher.get('canvas_image', []))}, "
                f"coordinates={num_coords}, "
                f"masks={len(stitcher.get('cropped_mask_for_blend', []))}, "
                f"warp_entries={len(content_warp_data) if content_warp_data else 0}\n"
                f"  Check: was tail_trim applied? Did WAN snap frame count (4k+1 rule)?"
            )
            print(diag)
            raise ValueError(diag)

        # --- Metadata length validation ---
        if not single_stitcher and len(skipped_indices) == 0:
            # Validate all coordinate arrays match expected count
            for key in ['canvas_image', 'cropped_mask_for_blend',
                        'cropped_to_canvas_x', 'cropped_to_canvas_y',
                        'cropped_to_canvas_w', 'cropped_to_canvas_h',
                        'canvas_to_orig_x', 'canvas_to_orig_y',
                        'canvas_to_orig_w', 'canvas_to_orig_h']:
                arr = stitcher.get(key, [])
                if len(arr) != expected_inpainted and len(arr) != 1:
                    raise ValueError(
                        f"[NV_InpaintStitch] Metadata length mismatch: "
                        f"stitcher['{key}'] has {len(arr)} entries, "
                        f"expected {expected_inpainted} (or 1 for broadcast)")
            if content_warp_data and len(content_warp_data) < expected_inpainted:
                raise ValueError(
                    f"[NV_InpaintStitch] Warp data mismatch: "
                    f"{len(content_warp_data)} warp entries for {expected_inpainted} frames")

        # --- Reclaim VRAM before full-res stitch operations ---
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

        # --- Pre-allocate output batch in fp16 (replaces the list-append-then-stack pattern).
        # Two-stage history of this code path:
        #
        #   v1 (original): `results = []` list grown per-frame via `.to(intermediate)`.
        #     Failure: Windows native access violation mid-loop on back-to-back stitch
        #     calls — accumulating ~6 GB list of CPU tensors collided with downstream
        #     held outputs (D-105 crash log, frame 239/277 in second call).
        #
        #   v2 (D-105 fix): pre-allocate one contiguous output tensor at peek_canvas
        #     dtype (fp32 in practice) and `.copy_()` per-frame. Fixed the access
        #     violation but a SECOND OOM appeared 2026-04-24 (night) — when two
        #     InpaintStitch calls produce 1936x1072 fp32 outputs, both held by
        #     downstream branches simultaneously: 277 × 1936 × 1072 × 3 × 4 bytes
        #     = 6.86 GB × 2 = 13.7 GB, hitting the system RAM ceiling on the
        #     SECOND `torch.empty` call.
        #
        #   v3 (this version): default output dtype to fp16. The output goes to a
        #     video encoder which converts to uint8 anyway — fp16 has zero visible
        #     impact on final delivery. Cuts per-stitch RAM in half (3.45 GB at
        #     1080p × 277 frames). Two parallel stitches now total ~6.9 GB instead
        #     of 13.7 GB — fits comfortably on a 32 GB system. fp32 still available
        #     via the `output_dtype` widget for users who specifically need it
        #     (e.g., feeding into another high-precision color operation).
        peek_canvas = stitcher['canvas_image'][0]
        out_H, out_W = peek_canvas.shape[0], peek_canvas.shape[1]
        out_C = peek_canvas.shape[2] if peek_canvas.dim() == 3 else 3
        # Output dtype: fp16 default for RAM efficiency. Resolved from the
        # widget value below (passed via execute() default).
        out_dtype = output_dtype_torch
        result_batch = torch.empty((total_frames, out_H, out_W, out_C), dtype=out_dtype, device=intermediate)
        stitch_mask_batch = torch.empty((total_frames, out_H, out_W), dtype=out_dtype, device=intermediate)

        if len(skipped_indices) == 0:
            # Simple path: all frames were inpainted
            for b in range(batch_size):
                idx = 0 if single_stitcher else b
                warp_entry = content_warp_data[idx] if (content_warp_data and not single_stitcher) else None

                # Debug logging: first frame, every 20th, and last frame
                if b == 0 or (b + 1) % 20 == 0 or b == batch_size - 1:
                    ctc_w = stitcher['cropped_to_canvas_w'][idx]
                    ctc_h = stitcher['cropped_to_canvas_h'][idx]
                    warp_info = "none"
                    if warp_entry and content_warp_mode == "centroid":
                        warp_info = f"centroid(dx={warp_entry.get('dx', 0):.1f},dy={warp_entry.get('dy', 0):.1f})"
                    elif warp_entry and content_warp_mode == "optical_flow":
                        warp_info = "optical_flow"
                    print(f"[NV_InpaintStitch] Frame {b}/{batch_size}: "
                          f"crop=({ctc_w}x{ctc_h}), warp={warp_info}")

                try:
                    # Clone canvas to prevent in-place mutation from accumulating across frames
                    canvas = stitcher['canvas_image'][idx].to(device).clone()
                    out, final_mask = stitch_single_frame(
                        canvas,
                        inpainted_image[b:b+1].to(device),
                        stitcher['cropped_mask_for_blend'][idx].to(device),
                        stitcher['cropped_to_canvas_x'][idx],
                        stitcher['cropped_to_canvas_y'][idx],
                        stitcher['cropped_to_canvas_w'][idx],
                        stitcher['cropped_to_canvas_h'][idx],
                        stitcher['canvas_to_orig_x'][idx],
                        stitcher['canvas_to_orig_y'][idx],
                        stitcher['canvas_to_orig_w'][idx],
                        stitcher['canvas_to_orig_h'][idx],
                        resize_algorithm,
                        warp_mode=content_warp_mode,
                        warp_entry=warp_entry,
                        blend_mode=blend_mode,
                        multiband_levels=multiband_levels,
                        guided_refine=guided_refine,
                        guided_radius=guided_radius,
                        guided_eps=guided_eps,
                        guided_strength=guided_strength,
                    )
                    # Direct GPU -> pre-allocated CPU slot. No intermediate CPU tensor
                    # is created; `copy_` streams straight into `result_batch[b]`.
                    result_batch[b].copy_(out.squeeze(0), non_blocking=False)
                    sm = _build_stitch_mask(
                        final_mask,
                        stitcher['cropped_to_canvas_x'][idx], stitcher['cropped_to_canvas_y'][idx],
                        stitcher['cropped_to_canvas_w'][idx], stitcher['cropped_to_canvas_h'][idx],
                        stitcher['canvas_to_orig_x'][idx], stitcher['canvas_to_orig_y'][idx],
                        stitcher['canvas_to_orig_w'][idx], stitcher['canvas_to_orig_h'][idx],
                        resize_algorithm,
                    )
                    stitch_mask_batch[b].copy_(sm, non_blocking=False)
                except Exception as e:
                    inp_shape = list(inpainted_image[b:b+1].shape)
                    raise RuntimeError(
                        f"[NV_InpaintStitch] FAILED at frame {b}/{batch_size} "
                        f"(idx={idx}, single_stitcher={single_stitcher}): "
                        f"inpainted_slice={inp_shape}, "
                        f"crop=({stitcher['cropped_to_canvas_w'][idx]}x{stitcher['cropped_to_canvas_h'][idx]}), "
                        f"warp={content_warp_mode}. Error: {e}"
                    ) from e

                # Periodic GPU cleanup to prevent VRAM fragmentation on long sequences
                if (b + 1) % 32 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        else:
            # Reconstruct full batch with skipped frames reinserted
            inpainted_idx = 0
            original_idx = 0

            for frame_idx in range(total_frames):
                if frame_idx in skipped_indices:
                    # Pass-through original frame; zero stitch mask to mark un-inpainted
                    result_batch[frame_idx].copy_(original_frames[original_idx], non_blocking=False)
                    stitch_mask_batch[frame_idx].zero_()
                    original_idx += 1
                else:
                    warp_entry = content_warp_data[inpainted_idx] if content_warp_data else None

                    # Debug logging: first, every 20th, last
                    if inpainted_idx == 0 or (inpainted_idx + 1) % 20 == 0 or frame_idx == total_frames - 1:
                        ctc_w = stitcher['cropped_to_canvas_w'][inpainted_idx]
                        ctc_h = stitcher['cropped_to_canvas_h'][inpainted_idx]
                        print(f"[NV_InpaintStitch] Frame {frame_idx}/{total_frames} "
                              f"(inpainted_idx={inpainted_idx}): crop=({ctc_w}x{ctc_h})")

                    try:
                        # Clone canvas in skip path too (consistency with no-skip path)
                        canvas = stitcher['canvas_image'][inpainted_idx].to(device).clone()
                        out, final_mask = stitch_single_frame(
                            canvas,
                            inpainted_image[inpainted_idx:inpainted_idx+1].to(device),
                            stitcher['cropped_mask_for_blend'][inpainted_idx].to(device),
                            stitcher['cropped_to_canvas_x'][inpainted_idx],
                            stitcher['cropped_to_canvas_y'][inpainted_idx],
                            stitcher['cropped_to_canvas_w'][inpainted_idx],
                            stitcher['cropped_to_canvas_h'][inpainted_idx],
                            stitcher['canvas_to_orig_x'][inpainted_idx],
                            stitcher['canvas_to_orig_y'][inpainted_idx],
                            stitcher['canvas_to_orig_w'][inpainted_idx],
                            stitcher['canvas_to_orig_h'][inpainted_idx],
                            resize_algorithm,
                            warp_mode=content_warp_mode,
                            warp_entry=warp_entry,
                            blend_mode=blend_mode,
                            multiband_levels=multiband_levels,
                            guided_refine=guided_refine,
                            guided_radius=guided_radius,
                            guided_eps=guided_eps,
                            guided_strength=guided_strength,
                        )
                        result_batch[frame_idx].copy_(out.squeeze(0), non_blocking=False)
                        sm = _build_stitch_mask(
                            final_mask,
                            stitcher['cropped_to_canvas_x'][inpainted_idx], stitcher['cropped_to_canvas_y'][inpainted_idx],
                            stitcher['cropped_to_canvas_w'][inpainted_idx], stitcher['cropped_to_canvas_h'][inpainted_idx],
                            stitcher['canvas_to_orig_x'][inpainted_idx], stitcher['canvas_to_orig_y'][inpainted_idx],
                            stitcher['canvas_to_orig_w'][inpainted_idx], stitcher['canvas_to_orig_h'][inpainted_idx],
                            resize_algorithm,
                        )
                        stitch_mask_batch[frame_idx].copy_(sm, non_blocking=False)
                    except Exception as e:
                        inp_shape = list(inpainted_image[inpainted_idx:inpainted_idx+1].shape)
                        raise RuntimeError(
                            f"[NV_InpaintStitch] FAILED at frame_idx={frame_idx}/{total_frames} "
                            f"(inpainted_idx={inpainted_idx}/{batch_size}): "
                            f"inpainted_slice={inp_shape}, "
                            f"crop=({stitcher['cropped_to_canvas_w'][inpainted_idx]}x"
                            f"{stitcher['cropped_to_canvas_h'][inpainted_idx]}), "
                            f"warp={content_warp_mode}. Error: {e}"
                        ) from e

                    inpainted_idx += 1

        # Reclaim GPU memory before returning
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"[NV_InpaintStitch] Stitched {result_batch.shape[0]} frames "
              f"({len(skipped_indices)} skipped, {batch_size} inpainted, "
              f"blend={blend_mode})")

        return (result_batch, stitch_mask_batch)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_InpaintStitch2": NV_InpaintStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_InpaintStitch2": "NV Inpaint Stitch v2",
}
