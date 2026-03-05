"""
NV CoTracker Bridge — Auto-stabilize InpaintCrop output using CoTracker3 point tracking.

Runs CoTracker3 offline inference on cropped frames to get sub-pixel point trajectories,
then applies the same expand-crop-trim stabilization as NV_AETrackingBridge. Stores
inverse warp data in the stitcher so InpaintStitch2 undoes the stabilization automatically.

Drop-in replacement for NV_AETrackingBridge — no manual tracking data required.
Set content_stabilize=off on InpaintCrop2 when using this node.

Workflow:
  InpaintCrop2 (stabilize=off) -> NV_CoTrackerBridge -> denoise -> InpaintStitch2
"""

import json
import math
import torch
import torch.nn.functional as TF
import comfy.model_management

from .ae_tracking_bridge import _build_translation_grid, _stitcher_val
from .inpaint_crop import rescale_image


# =============================================================================
# CoTracker3 Model Cache
# =============================================================================

_cotracker_model = None


def _get_cotracker_model():
    """Load CoTracker3 offline model (lazy, cached at module level)."""
    global _cotracker_model
    if _cotracker_model is None:
        print("[NV_CoTrackerBridge] Loading CoTracker3 offline model (first run downloads ~100MB)...")
        _cotracker_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        _cotracker_model.eval()
        print("[NV_CoTrackerBridge] CoTracker3 model loaded.")
    return _cotracker_model


# =============================================================================
# Helpers
# =============================================================================

def _compute_mask_centroid(mask_2d):
    """Compute weighted centroid of a 2D mask tensor [H, W]. Returns (cx, cy) floats."""
    H, W = mask_2d.shape
    if mask_2d.sum() < 1e-6:
        return W / 2.0, H / 2.0

    ys = torch.arange(H, device=mask_2d.device, dtype=torch.float32)
    xs = torch.arange(W, device=mask_2d.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    total = mask_2d.sum()
    cx = (xx * mask_2d).sum() / total
    cy = (yy * mask_2d).sum() / total
    return cx.item(), cy.item()


def _interpolate_invisible(positions, visibility, threshold=0.5):
    """Fill invisible frames via linear interpolation from nearest visible neighbours.

    Args:
        positions: list of (x, y) tuples, length B
        visibility: list of float confidence values, length B
        threshold: minimum visibility to trust a position

    Returns:
        list of (x, y) with invisible frames interpolated. None if all invisible.
    """
    B = len(positions)
    visible = [i for i in range(B) if visibility[i] >= threshold]

    if not visible:
        return None

    result = list(positions)
    for i in range(B):
        if visibility[i] >= threshold:
            continue

        # Find nearest visible neighbours
        left = max((v for v in visible if v < i), default=None)
        right = min((v for v in visible if v > i), default=None)

        if left is not None and right is not None:
            t = (i - left) / (right - left)
            result[i] = (
                positions[left][0] + t * (positions[right][0] - positions[left][0]),
                positions[left][1] + t * (positions[right][1] - positions[left][1]),
            )
        elif left is not None:
            result[i] = positions[left]
        elif right is not None:
            result[i] = positions[right]

    return result


# =============================================================================
# Node Class
# =============================================================================

class NV_CoTrackerBridge:
    """Auto-stabilize InpaintCrop output using CoTracker3 point tracking.

    Drop-in replacement for NV_AETrackingBridge — no manual tracking data required.
    Set content_stabilize=off on InpaintCrop2 when using this node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitcher": ("STITCHER",),
                "cropped_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Stabilization strength. 1.0 = full lock to reference. "
                               "<1.0 = partial correction. >1.0 = overcorrect."
                }),
            },
            "optional": {
                "cropped_mask": ("MASK",),
                "cropped_mask_processed": ("MASK",),
                "tracking_points": ("STRING", {
                    "forceInput": True,
                    "tooltip": "JSON array of {x, y} points from NV_PointPicker. When provided, tracks all points "
                               "and averages their trajectories for more robust stabilization."
                }),
                "query_x": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 8192.0, "step": 0.5,
                    "tooltip": "Manual query point X in crop pixels. -1 = auto from mask centroid on frame 0. "
                               "Ignored when tracking_points is connected."
                }),
                "query_y": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 8192.0, "step": 0.5,
                    "tooltip": "Manual query point Y in crop pixels. -1 = auto from mask centroid on frame 0. "
                               "Ignored when tracking_points is connected."
                }),
            }
        }

    RETURN_TYPES = ("STITCHER", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("stitcher", "stabilized_image", "cropped_mask",
                    "cropped_mask_processed", "info")
    FUNCTION = "apply_tracking"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Auto-stabilizes InpaintCrop2 output using CoTracker3 point tracking. "
        "No manual tracking data required — tracks a point automatically. "
        "Set content_stabilize=off on InpaintCrop2 when using this node. "
        "The inverse warp is automatically applied by InpaintStitch2."
    )

    def apply_tracking(self, stitcher, cropped_image, strength=1.0,
                       cropped_mask=None, cropped_mask_processed=None,
                       tracking_points=None, query_x=-1.0, query_y=-1.0):
        device = comfy.model_management.get_torch_device()
        intermediate = comfy.model_management.intermediate_device()
        B, H, W, C = cropped_image.shape

        # Single frame — nothing to stabilize
        if B <= 1:
            info = "Single frame — passed through unchanged."
            print(f"[NV_CoTrackerBridge] {info}")
            return (stitcher, cropped_image, cropped_mask, cropped_mask_processed, info)

        # =====================================================================
        # Step 1: Determine query points
        # =====================================================================
        query_list = []  # list of (x, y)

        # Priority 1: tracking_points from NV_PointPicker
        if tracking_points:
            try:
                parsed = json.loads(tracking_points)
                for p in parsed:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        query_list.append((float(p["x"]), float(p["y"])))
            except (json.JSONDecodeError, TypeError):
                pass

        # Priority 2: manual query_x/query_y
        if not query_list and query_x >= 0 and query_y >= 0:
            query_list.append((float(query_x), float(query_y)))

        # Priority 3: mask centroid
        if not query_list and cropped_mask is not None:
            cx, cy = _compute_mask_centroid(cropped_mask[0])
            query_list.append((cx, cy))

        # Priority 4: image center
        if not query_list:
            query_list.append((W / 2.0, H / 2.0))
            print("[NV_CoTrackerBridge] WARNING: No tracking points, mask, or manual query. Using image center.")

        n_queries = len(query_list)
        query_source = f"point_picker({n_queries})" if tracking_points else ("manual" if query_x >= 0 else "mask_centroid")
        if not tracking_points and query_x < 0 and cropped_mask is None:
            query_source = "image_center (no mask)"

        # queries: (1, N, 3) = [[frame_idx, x, y], ...]
        queries = torch.tensor([[[0.0, qx, qy] for qx, qy in query_list]], dtype=torch.float32)

        # =====================================================================
        # Step 2: Run CoTracker3 inference
        # =====================================================================
        try:
            model = _get_cotracker_model()
            model = model.to(device)

            # IMAGE [B,H,W,C] -> CoTracker [1,T,C,H,W]
            video = cropped_image.permute(0, 3, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_tracks, pred_visibility = model(video, queries=queries.to(device))
            # pred_tracks: [1, B, 1, 2], pred_visibility: [1, B, 1, 1] or [1, B, 1]

            model.cpu()
            torch.cuda.empty_cache()

        except Exception as e:
            info = f"CoTracker3 inference failed: {e}. Images passed through unchanged."
            print(f"[NV_CoTrackerBridge] ERROR: {info}")
            return (stitcher, cropped_image, cropped_mask, cropped_mask_processed, info)

        # =====================================================================
        # Step 3: Extract positions and handle visibility (multi-point averaging)
        # =====================================================================
        # pred_tracks: [1, B, N, 2], pred_visibility: [1, B, N] or [1, B, N, 1]
        all_tracks = pred_tracks[0].cpu()   # [B, N, 2]
        all_vis = pred_visibility[0].cpu()  # [B, N] or [B, N, 1]
        if all_vis.dim() > 2:
            all_vis = all_vis.squeeze(-1)   # [B, N]

        N = all_tracks.shape[1]

        # For each query point, get interpolated trajectory, then average across points
        per_point_positions = []  # list of N trajectory lists (each length B)
        per_point_vis = []
        for qi in range(N):
            positions_qi = [(all_tracks[b, qi, 0].item(), all_tracks[b, qi, 1].item()) for b in range(B)]
            vis_qi = [all_vis[b, qi].item() for b in range(B)]
            interp = _interpolate_invisible(positions_qi, vis_qi, threshold=0.5)
            if interp is not None:
                per_point_positions.append(interp)
                per_point_vis.append(vis_qi)

        if not per_point_positions:
            info = "All frames invisible to CoTracker across all query points — passed through unchanged."
            print(f"[NV_CoTrackerBridge] WARNING: {info}")
            return (stitcher, cropped_image, cropped_mask, cropped_mask_processed, info)

        # Average displacement across all valid tracked points per frame
        # Each point's displacement = position[b] - position[ref_frame]
        # We average the displacements (not raw positions) so different points in different
        # locations contribute their motion equally.
        n_tracked = len(per_point_positions)

        # Compute per-point reference (median position)
        per_point_ref = []
        for qi in range(n_tracked):
            xs = sorted(p[0] for p in per_point_positions[qi])
            ys = sorted(p[1] for p in per_point_positions[qi])
            per_point_ref.append((xs[B // 2], ys[B // 2]))

        # Average displacement across points, weighted by visibility
        positions = []  # averaged displacement as (dx, dy) relative to zero
        for b in range(B):
            total_w = 0.0
            sum_dx = 0.0
            sum_dy = 0.0
            for qi in range(n_tracked):
                w = per_point_vis[qi][b] if per_point_vis[qi][b] >= 0.5 else 0.3  # lower weight for interpolated
                dx_qi = per_point_positions[qi][b][0] - per_point_ref[qi][0]
                dy_qi = per_point_positions[qi][b][1] - per_point_ref[qi][1]
                sum_dx += dx_qi * w
                sum_dy += dy_qi * w
                total_w += w
            if total_w > 0:
                positions.append((sum_dx / total_w, sum_dy / total_w))
            else:
                positions.append((0.0, 0.0))

        # Count visible frames (any point visible counts)
        n_visible = 0
        for b in range(B):
            if any(per_point_vis[qi][b] >= 0.5 for qi in range(n_tracked)):
                n_visible += 1

        # =====================================================================
        # Step 4: Apply strength to displacements
        # =====================================================================
        # positions[] are already averaged displacements relative to median reference
        displacements = [
            (positions[b][0] * strength, positions[b][1] * strength)
            for b in range(B)
        ]

        max_dx = max(abs(d[0]) for d in displacements)
        max_dy = max(abs(d[1]) for d in displacements)
        margin = int(math.ceil(max(max_dx, max_dy))) + 1

        # =====================================================================
        # Step 5: Expand-crop-trim warp (from ae_tracking_bridge.py)
        # =====================================================================
        use_expansion = (margin > 0 and len(stitcher.get('canvas_image', [])) >= B)

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
                need_b_m = (int(math.ceil(abs(dy))) + 1) if dy > 0 else 1

                mc_l = int(math.ceil(need_l / sx)) + 1
                mc_r = int(math.ceil(need_r / sx)) + 1
                mc_t = int(math.ceil(need_t / sy)) + 1
                mc_b = int(math.ceil(need_b_m / sy)) + 1

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
                wi = wi[:, :, trim_top:trim_top + H, trim_left:trim_left + W]

                # Warp masks if provided
                if cropped_mask is not None:
                    pad_right = max(0, expanded_w_b - trim_left - W)
                    pad_bottom = max(0, expanded_h_b - trim_top - H)
                    mo_4d = TF.pad(cropped_mask[b].unsqueeze(0).unsqueeze(0),
                                   (trim_left, pad_right, trim_top, pad_bottom),
                                   mode='constant', value=0)
                    wmo = TF.grid_sample(mo_4d.to(device), grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False)
                    wmo = wmo[:, :, trim_top:trim_top + H, trim_left:trim_left + W]
                    warped_mo.append(wmo.squeeze(0).squeeze(0).to(intermediate))

                if cropped_mask_processed is not None:
                    pad_right = max(0, expanded_w_b - trim_left - W)
                    pad_bottom = max(0, expanded_h_b - trim_top - H)
                    mp_4d = TF.pad(cropped_mask_processed[b].unsqueeze(0).unsqueeze(0),
                                   (trim_left, pad_right, trim_top, pad_bottom),
                                   mode='constant', value=0)
                    wmp = TF.grid_sample(mp_4d.to(device), grid, mode='bilinear',
                                         padding_mode='zeros', align_corners=False)
                    wmp = wmp[:, :, trim_top:trim_top + H, trim_left:trim_left + W]
                    warped_mp.append(wmp.squeeze(0).squeeze(0).to(intermediate))
            else:
                # No expansion available — pass through unchanged
                wi = cropped_image[b:b + 1].permute(0, 3, 1, 2).to(device)
                dx, dy = 0.0, 0.0
                if warped_mo is not None:
                    warped_mo.append(cropped_mask[b])
                if warped_mp is not None:
                    warped_mp.append(cropped_mask_processed[b])

            warp_data.append({"dx": dx, "dy": dy})
            warped_imgs.append(wi.permute(0, 2, 3, 1).squeeze(0).to(intermediate))

        # =====================================================================
        # Step 6: Store warp data in stitcher
        # =====================================================================
        stitcher['content_warp_mode'] = 'centroid'
        stitcher['content_warp_data'] = warp_data

        result_images = torch.stack(warped_imgs, dim=0)
        result_mo = torch.stack(warped_mo, dim=0) if warped_mo else cropped_mask
        result_mp = torch.stack(warped_mp, dim=0) if warped_mp else cropped_mask_processed

        max_disp = max(max(abs(d["dx"]) for d in warp_data),
                       max(abs(d["dy"]) for d in warp_data))
        clamp_str = f", {n_clamped} edge-clamped" if n_clamped > 0 else ""
        mode_str = "expand-crop-trim" if use_expansion else "passthrough"

        query_coords = ", ".join(f"({qx:.0f},{qy:.0f})" for qx, qy in query_list)
        info_lines = [
            f"CoTracker3: {B} frames, {n_tracked}/{n_queries} points tracked, query={query_source}",
            f"Points: {query_coords}",
            f"Strength: {strength:.2f}, max_disp: {max_disp:.1f}px ({mode_str}{clamp_str})",
            f"Visibility: {n_visible}/{B} frames visible (threshold=0.5)",
        ]
        info = "\n".join(info_lines)

        print(f"[NV_CoTrackerBridge] {B} frames, {n_tracked} pts tracked, strength={strength:.2f}, "
              f"max_disp={max_disp:.1f}px, query={query_source} "
              f"({mode_str}{clamp_str}), {n_visible}/{B} visible")

        return (stitcher, result_images, result_mo, result_mp, info)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_CoTrackerBridge": NV_CoTrackerBridge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_CoTrackerBridge": "NV CoTracker Bridge",
}
