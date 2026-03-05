"""
NV AE Tracking Bridge — Apply After Effects / Mocha tracking data to InpaintCrop pipeline.

Takes per-frame point tracking positions from AE or Mocha, converts to crop-space
translations, applies stabilization to cropped images, and stores inverse warp data
in the stitcher for InpaintStitch2.

Replaces the built-in centroid stabilization with external tracking data for higher
quality (sub-pixel AE/Mocha point tracker vs mask centroid).

Set content_stabilize=off on InpaintCrop2 when using this node.

Supported tracking data formats:
  - Simple: one line per frame with "frame x y" (tab or comma separated)
  - JSON: [{"frame": 0, "x": 960.0, "y": 540.0}, ...]
  - AE paste: full keyframe data copied from After Effects tracker panel
"""

import math
import json
import re
import torch
import torch.nn.functional as TF
import comfy.model_management

from .inpaint_crop import rescale_image


# =============================================================================
# Tracking Data Parser
# =============================================================================

def parse_tracking_data(text):
    """Parse tracking data from various formats into list of (frame, x, y) tuples.

    Supports:
      - JSON array: [{"frame": 0, "x": 960, "y": 540}, ...]
      - AE paste format: header + tab-separated Frame/X/Y data
      - Simple lines: "frame x y" or "x y" (tab, comma, or space separated)

    Returns:
        list of (frame_int, x_float, y_float)
    """
    text = text.strip()
    if not text:
        return []

    # Try JSON
    if text[0] in '{[':
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            return [(int(d.get('frame', i)), float(d['x']), float(d['y']))
                    for i, d in enumerate(data)]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    lines = text.split('\n')
    results = []
    in_data = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip AE header lines
        if 'Adobe After Effects' in line or 'End of Keyframe Data' in line:
            continue
        if line.startswith(('Units', 'Source', 'Comp', 'Effects', 'Transform')):
            continue
        # Detect data header row
        if 'Frame' in line and ('X pixels' in line or 'x pixels' in line.lower()):
            in_data = True
            continue

        # Parse as numbers (tab, comma, or space separated)
        parts = re.split(r'[,\t]+', line)
        nums = []
        for p in parts:
            p = p.strip()
            try:
                nums.append(float(p))
            except ValueError:
                pass

        if len(nums) >= 3:
            # frame, x, y (and possibly z for AE 3D)
            results.append((int(nums[0]), nums[1], nums[2]))
        elif len(nums) == 2:
            # x, y only — auto-number frames
            results.append((len(results), nums[0], nums[1]))

    return results


# =============================================================================
# Helpers
# =============================================================================

def _build_translation_grid(dx, dy, H, W, device):
    """Build affine_grid for pure translation. dx/dy in pixels."""
    norm_dx = 2.0 * dx / W
    norm_dy = 2.0 * dy / H
    theta = torch.tensor([
        [1.0, 0.0, norm_dx],
        [0.0, 1.0, norm_dy]
    ], device=device, dtype=torch.float32).unsqueeze(0)
    return TF.affine_grid(theta, (1, 1, H, W), align_corners=False)


def _stitcher_val(stitcher, key, b):
    """Get stitcher value at index b, falling back to 0 for single-stitcher mode."""
    vals = stitcher[key]
    return vals[b] if b < len(vals) else vals[0]


# =============================================================================
# Node Class
# =============================================================================

class NV_AETrackingBridge:
    """Apply After Effects / Mocha point tracking data to stabilize InpaintCrop output.

    Replaces centroid stabilization with external tracking for higher quality.
    Set content_stabilize=off on InpaintCrop2 when using this node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "cropped_image": ("IMAGE",),
                "tracking_data": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Per-frame tracking point positions from AE or Mocha. "
                               "Coordinates in SOURCE VIDEO pixels (not crop space). "
                               "Formats: simple (frame x y per line), "
                               "JSON ([{frame, x, y}, ...]), "
                               "or AE paste (copy from tracker panel)."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stabilization strength. "
                               "1.0 = full lock to reference position. "
                               "< 1.0 = partial correction. "
                               "> 1.0 = overcorrect (compensate for tracking drift)."
                }),
                "start_frame": ("INT", {
                    "default": 0, "min": 0,
                    "tooltip": "Frame number in tracking data that corresponds to the "
                               "first cropped image. Use when tracking data spans the "
                               "full video but crop is a subset of frames."
                }),
            },
            "optional": {
                "cropped_mask": ("MASK",),
                "cropped_mask_processed": ("MASK",),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "stabilized_image", "cropped_mask",
                    "cropped_mask_processed", "info")
    FUNCTION = "apply_tracking"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Applies After Effects / Mocha point tracking data to stabilize "
        "InpaintCrop2 output. Set content_stabilize=off on InpaintCrop2 "
        "when using this node. Tracking coordinates must be in source "
        "video pixel space. The inverse warp is automatically applied "
        "by InpaintStitch2."
    )

    def apply_tracking(self, stitcher, cropped_image, tracking_data,
                       strength=1.0, start_frame=0,
                       cropped_mask=None, cropped_mask_processed=None):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        B, H, W, C = cropped_image.shape

        # Parse tracking data
        tracks = parse_tracking_data(tracking_data)
        if not tracks:
            info = "No tracking data parsed — images passed through unchanged."
            print(f"[NV_AETrackingBridge] {info}")
            return (stitcher, cropped_image, cropped_mask,
                    cropped_mask_processed, info)

        # Build frame → position lookup
        track_map = {int(f): (x, y) for f, x, y in tracks}

        # Convert tracking positions from source video space to crop space
        crop_positions = []
        for b in range(B):
            frame_idx = start_frame + b

            # Find tracking position (nearest available if missing)
            if frame_idx in track_map:
                src_x, src_y = track_map[frame_idx]
            else:
                available = sorted(track_map.keys())
                nearest = min(available, key=lambda f: abs(f - frame_idx))
                src_x, src_y = track_map[nearest]

            # Source video → canvas space
            cto_x = _stitcher_val(stitcher, 'canvas_to_orig_x', b)
            cto_y = _stitcher_val(stitcher, 'canvas_to_orig_y', b)
            canvas_x = src_x + cto_x
            canvas_y = src_y + cto_y

            # Canvas → crop space (at canvas resolution)
            ctc_x = _stitcher_val(stitcher, 'cropped_to_canvas_x', b)
            ctc_y = _stitcher_val(stitcher, 'cropped_to_canvas_y', b)
            ctc_w = _stitcher_val(stitcher, 'cropped_to_canvas_w', b)
            ctc_h = _stitcher_val(stitcher, 'cropped_to_canvas_h', b)

            crop_x = (canvas_x - ctc_x) * W / ctc_w
            crop_y = (canvas_y - ctc_y) * H / ctc_h

            crop_positions.append((crop_x, crop_y))

        # Reference = temporal median
        ref_x = sorted(p[0] for p in crop_positions)[B // 2]
        ref_y = sorted(p[1] for p in crop_positions)[B // 2]

        # Compute displacements (strength-scaled)
        displacements = [(
            (crop_positions[b][0] - ref_x) * strength,
            (crop_positions[b][1] - ref_y) * strength,
        ) for b in range(B)]

        # Margin for expansion
        max_dx = max(abs(d[0]) for d in displacements)
        max_dy = max(abs(d[1]) for d in displacements)
        margin = int(math.ceil(max(max_dx, max_dy))) + 1

        # Can we do expand-crop-trim?
        use_expansion = (margin > 0
                         and len(stitcher.get('canvas_image', [])) >= B)

        warp_data = []
        warped_imgs = []
        warped_mo = [] if cropped_mask is not None else None
        warped_mp = [] if cropped_mask_processed is not None else None
        n_clamped = 0

        for b in range(B):
            dx, dy = displacements[b]

            if use_expansion:
                canvas = stitcher['canvas_image'][b].to(device)
                if canvas.dim() == 3:
                    canvas = canvas.unsqueeze(0)
                ctc_x = _stitcher_val(stitcher, 'cropped_to_canvas_x', b)
                ctc_y = _stitcher_val(stitcher, 'cropped_to_canvas_y', b)
                ctc_w = _stitcher_val(stitcher, 'cropped_to_canvas_w', b)
                ctc_h = _stitcher_val(stitcher, 'cropped_to_canvas_h', b)
                canvas_h, canvas_w = canvas.shape[1], canvas.shape[2]

                sx = W / ctc_w
                sy = H / ctc_h

                # Available margins in target space
                avail_l = ctc_x * sx
                avail_r = (canvas_w - ctc_x - ctc_w) * sx
                avail_t = ctc_y * sy
                avail_b = (canvas_h - ctc_y - ctc_h) * sy

                # Clamp displacement to available canvas
                dx_orig, dy_orig = dx, dy
                dx = max(-avail_l, min(avail_r, dx))
                dy = max(-avail_t, min(avail_b, dy))
                if dx != dx_orig or dy != dy_orig:
                    n_clamped += 1

                # Per-side margins based on clamped displacement
                need_l = (int(math.ceil(abs(dx))) + 1) if dx < 0 else 1
                need_r = (int(math.ceil(abs(dx))) + 1) if dx > 0 else 1
                need_t = (int(math.ceil(abs(dy))) + 1) if dy < 0 else 1
                need_b = (int(math.ceil(abs(dy))) + 1) if dy > 0 else 1

                mc_l = int(math.ceil(need_l / sx)) + 1
                mc_r = int(math.ceil(need_r / sx)) + 1
                mc_t = int(math.ceil(need_t / sy)) + 1
                mc_b = int(math.ceil(need_b / sy)) + 1

                ex = max(0, ctc_x - mc_l)
                ey = max(0, ctc_y - mc_t)
                ex2 = min(canvas_w, ctc_x + ctc_w + mc_r)
                ey2 = min(canvas_h, ctc_y + ctc_h + mc_b)

                actual_l_c = ctc_x - ex
                actual_t_c = ctc_y - ey
                total_c_w = ex2 - ex
                total_c_h = ey2 - ey
                expanded_w_b = int(round(total_c_w * sx))
                expanded_h_b = int(round(total_c_h * sy))
                trim_left = int(round(actual_l_c * sx))
                trim_top = int(round(actual_t_c * sy))

                exp_img = canvas[:, ey:ey2, ex:ex2, :]
                exp_img = rescale_image(exp_img, expanded_w_b, expanded_h_b, 'lanczos')
                img_nchw = exp_img.permute(0, 3, 1, 2)

                grid = _build_translation_grid(dx, dy, expanded_h_b, expanded_w_b, device)

                wi = TF.grid_sample(img_nchw, grid, mode='bilinear',
                                    padding_mode='zeros', align_corners=False)
                wi = wi[:, :, trim_top:trim_top+H, trim_left:trim_left+W]

                # Warp masks if provided
                if cropped_mask is not None:
                    pad_right = max(0, expanded_w_b - trim_left - W)
                    pad_bottom = max(0, expanded_h_b - trim_top - H)
                    mo_4d = TF.pad(cropped_mask[b].unsqueeze(0).unsqueeze(0),
                                   (trim_left, pad_right, trim_top, pad_bottom),
                                   mode='constant', value=0)
                    wmo = TF.grid_sample(mo_4d.to(device), grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False)
                    wmo = wmo[:, :, trim_top:trim_top+H, trim_left:trim_left+W]
                    warped_mo.append(wmo.squeeze(0).squeeze(0).to(intermediate))

                if cropped_mask_processed is not None:
                    pad_right = max(0, expanded_w_b - trim_left - W)
                    pad_bottom = max(0, expanded_h_b - trim_top - H)
                    mp_4d = TF.pad(cropped_mask_processed[b].unsqueeze(0).unsqueeze(0),
                                   (trim_left, pad_right, trim_top, pad_bottom),
                                   mode='constant', value=0)
                    wmp = TF.grid_sample(mp_4d.to(device), grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False)
                    wmp = wmp[:, :, trim_top:trim_top+H, trim_left:trim_left+W]
                    warped_mp.append(wmp.squeeze(0).squeeze(0).to(intermediate))
            else:
                # No expansion — pass through unchanged
                wi = cropped_image[b:b+1].permute(0, 3, 1, 2).to(device)
                dx, dy = 0.0, 0.0
                if warped_mo is not None:
                    warped_mo.append(cropped_mask[b])
                if warped_mp is not None:
                    warped_mp.append(cropped_mask_processed[b])

            warp_data.append({"dx": dx, "dy": dy})
            warped_imgs.append(wi.permute(0, 2, 3, 1).squeeze(0).to(intermediate))

        # Store warp data in stitcher for InpaintStitch2 inverse warp
        stitcher['content_warp_mode'] = 'centroid'  # Same inverse type (translation)
        stitcher['content_warp_data'] = warp_data

        result_images = torch.stack(warped_imgs, dim=0)
        result_mo = torch.stack(warped_mo, dim=0) if warped_mo else cropped_mask
        result_mp = torch.stack(warped_mp, dim=0) if warped_mp else cropped_mask_processed

        max_disp = max(max(abs(d["dx"]) for d in warp_data),
                       max(abs(d["dy"]) for d in warp_data))
        clamp_str = f", {n_clamped} edge-clamped" if n_clamped > 0 else ""
        mode_str = "expand-crop-trim" if use_expansion else "passthrough"

        info_lines = [
            f"AE tracking: {len(tracks)} points parsed, {B} frames stabilized",
            f"Strength: {strength:.2f}, max_disp: {max_disp:.1f}px ({mode_str}{clamp_str})",
            f"Reference: ({ref_x:.1f}, {ref_y:.1f}) in crop space",
            f"Source points: frames {min(track_map.keys())}-{max(track_map.keys())}",
        ]
        info = "\n".join(info_lines)

        print(f"[NV_AETrackingBridge] {B} frames, strength={strength:.2f}, "
              f"max_disp={max_disp:.1f}px, {len(tracks)} track points "
              f"({mode_str}{clamp_str})")

        return (stitcher, result_images, result_mo, result_mp, info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_AETrackingBridge": NV_AETrackingBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_AETrackingBridge": "NV AE Tracking Bridge",
}
