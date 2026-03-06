"""
NV Kling Stitch Adapter — Bridges Kling API output back to InpaintStitch2.

Handles the two mismatches that occur when routing InpaintCrop2 → Kling → InpaintStitch2:

  1. Resolution: Kling outputs at its own resolution (720p/1080p), not the crop's
     target resolution.  The adapter resizes back to the crop target so that
     content-warp (CoTracker / centroid) dx/dy values are applied at the correct
     pixel scale by InpaintStitch2.  This is critical because InpaintStitch2
     applies inverse warp BEFORE resize (per the 2026-03-05 double-head fix),
     so the image must arrive at the resolution the dx/dy were computed at.

  2. Frame count: Kling always outputs 24fps and may add/drop frames relative to
     the input.  The adapter resamples to match the stitcher's expected count
     using nearest-frame selection (preserves sharpness, no blending artifacts).

Important: The adapter NEVER modifies the stitcher dict.  Content warp data
(content_warp_mode, content_warp_data) belongs to CoTrackerBridge/centroid and
is consumed by InpaintStitch2 — the adapter must not touch it.

Workflow:
  InpaintCrop2 → KlingUploadPreview → KlingEditVideo → **KlingStitchAdapter** → InpaintStitch2
"""

import torch

from .inpaint_crop import rescale_image


class NV_KlingStitchAdapter:
    """Resize + resample Kling output to match what InpaintStitch2 expects."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "kling_images": ("IMAGE", {
                    "tooltip": "Output frames from NV Kling Edit Video.",
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "STITCHER dict from NV Inpaint Crop v2.",
                }),
            },
            "optional": {
                "upload_config": ("KLING_UPLOAD_CONFIG", {
                    "tooltip": (
                        "Config from NV Kling Upload Preview. "
                        "Used to read the pre-Kling crop resolution if available. "
                        "Falls back to stitcher crop_target_w/h if not connected."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STITCHER")
    RETURN_NAMES = ("images", "stitcher")
    FUNCTION = "adapt"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Bridges Kling API output back to Inpaint Stitch v2. "
        "Resizes frames to the crop's target resolution (so content-warp "
        "data stays valid) and resamples frame count if Kling changed it."
    )

    def adapt(self, kling_images, stitcher, upload_config=None):
        kling_b, kling_h, kling_w, kling_c = kling_images.shape

        # --- Determine target resolution ---
        # Priority: stitcher crop_target > upload_config input_resolution > as-is
        target_w = stitcher.get("crop_target_w")
        target_h = stitcher.get("crop_target_h")

        if target_w is None or target_h is None:
            if upload_config is not None and "input_resolution" in upload_config:
                target_w, target_h = upload_config["input_resolution"]
            else:
                # No resolution info — pass through at Kling resolution
                target_w, target_h = kling_w, kling_h

        # Use stitcher's resize algorithm for consistency with InpaintStitch2
        resize_algorithm = stitcher.get("resize_algorithm", "lanczos")

        # --- Determine expected frame count ---
        # Expected = non-skipped frames (stitcher has per-frame data for these)
        skipped = set(stitcher.get("skipped_indices", []))
        total_frames = stitcher.get("total_frames", 0)
        expected_frames = total_frames - len(skipped) if total_frames > 0 else kling_b

        # Also validate against per-frame stitcher data length
        num_canvas = len(stitcher.get("canvas_image", []))
        if num_canvas > 0 and num_canvas != expected_frames:
            print(
                f"[NV_KlingStitchAdapter] Warning: stitcher canvas_image count "
                f"({num_canvas}) != expected ({expected_frames}). Using canvas count."
            )
            expected_frames = num_canvas

        # --- Resample frames if count differs ---
        if kling_b != expected_frames and expected_frames > 0:
            print(
                f"[NV_KlingStitchAdapter] Frame count mismatch: "
                f"Kling={kling_b}, expected={expected_frames}. "
                f"Resampling with nearest-frame selection."
            )
            indices = torch.linspace(0, kling_b - 1, expected_frames).round().long()
            kling_images = kling_images[indices]
            kling_b = expected_frames

        # --- Resize to crop target resolution ---
        # Must happen AFTER frame resampling — rescale_image uses GPU, minimize work.
        # InpaintStitch2 applies inverse warp at this resolution BEFORE its own resize
        # to canvas crop size, so the image MUST arrive at crop_target dims.
        if kling_w != target_w or kling_h != target_h:
            print(
                f"[NV_KlingStitchAdapter] Resizing {kling_w}x{kling_h} → "
                f"{target_w}x{target_h} (crop target, algo={resize_algorithm})"
            )
            kling_images = rescale_image(kling_images, target_w, target_h, resize_algorithm)

        print(
            f"[NV_KlingStitchAdapter] Output: {kling_images.shape[0]} frames "
            f"@ {target_w}x{target_h}"
        )

        return (kling_images, stitcher)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_KlingStitchAdapter": NV_KlingStitchAdapter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_KlingStitchAdapter": "NV Kling Stitch Adapter",
}
