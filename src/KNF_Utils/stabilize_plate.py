"""
NV Stabilize Plate / Inverse Stabilize — Content-aware crop stabilization.

Standard VFX technique (After Effects "Stabilize Plate", Mocha Pro "Reverse Stabilize"):
Before cropping, compute a per-frame translation that aligns the mask centroid to a
fixed reference position. Warp the video + mask so the subject is pixel-locked.
After inpainting + stitching, inverse-warp to restore original motion.

This addresses the ROOT CAUSE of temporal jitter: the diffusion model currently sees
jittery input because bbox smoothing stabilizes the *window*, not the *content within
the window*. A locked plate gives the diffusion model temporally coherent spatial
positions.

Pipeline:
    Input Video → NV_StabilizePlate → (stable plate) → NV_InpaintCrop → KSampler
                                                                            ↓
    Output Video ← NV_InverseStabilize ← NV_InpaintStitch ← ← ← ← ← ← ← ←

Performance: ~8-12ms for 81 frames at 720p (negligible vs diffusion).

Two smoothing options for the centroid trajectory:
    one_euro  — velocity-adaptive, best for variable-speed motion
    kalman_rts — globally optimal offline, best for offline batch processing

Upgrade path: If rotation matters (subject turning), compute rigid transform via
mask PCA instead of just centroid translation.
"""

import torch
import torch.nn.functional as F

from .mask_tracking_bbox import one_euro_smooth_1d


def _compute_centroids(mask):
    """Compute per-frame mask centroid.

    Args:
        mask: [B, H, W] binary or soft mask.

    Returns:
        cx, cy: Lists of float centroid coordinates.
        valid: List of bool — True if frame had mask content.
    """
    B, H, W = mask.shape
    cx_list = []
    cy_list = []
    valid = []

    for b in range(B):
        m = mask[b]
        total = m.sum()
        if total < 1.0:
            valid.append(False)
            cx_list.append(W / 2.0)
            cy_list.append(H / 2.0)
        else:
            ys = torch.arange(H, device=m.device, dtype=m.dtype)
            xs = torch.arange(W, device=m.device, dtype=m.dtype)
            cy = (m.sum(dim=1) * ys).sum() / total
            cx = (m.sum(dim=0) * xs).sum() / total
            valid.append(True)
            cx_list.append(float(cx.item()))
            cy_list.append(float(cy.item()))

    # Forward/backward fill invalid frames
    last = None
    for i in range(B):
        if valid[i]:
            last = (cx_list[i], cy_list[i])
        elif last is not None:
            cx_list[i], cy_list[i] = last
    last = None
    for i in range(B - 1, -1, -1):
        if valid[i]:
            last = (cx_list[i], cy_list[i])
        elif last is not None:
            cx_list[i], cy_list[i] = last

    return cx_list, cy_list, valid


def _smooth_centroids(cx_list, cy_list, mode, min_cutoff=0.05, beta=0.7,
                      valid=None, q_pos=4.0, r_pos=9.0):
    """Smooth centroid trajectory.

    Args:
        cx_list, cy_list: Raw centroid sequences.
        mode: "one_euro" or "kalman_rts".
        min_cutoff, beta: One-Euro parameters.
        valid: Per-frame detection flags (for Kalman).
        q_pos, r_pos: Kalman noise parameters (position only, no dimensions).

    Returns:
        Smoothed cx_list, cy_list.
    """
    B = len(cx_list)
    if B <= 2:
        return cx_list, cy_list

    if mode == "kalman_rts" and B >= 10:
        from .kalman_rts_smoother import kalman_rts_smooth
        if valid is None:
            valid = [True] * B
        # Use Kalman on centroid as if it were a 1-pixel-wide bbox
        # We trick it: x1=cx, y1=cy, x2=cx+1, y2=cy+1
        x1s = cx_list
        y1s = cy_list
        x2s = [cx + 1.0 for cx in cx_list]
        y2s = [cy + 1.0 for cy in cy_list]
        sx1, sy1, sx2, sy2 = kalman_rts_smooth(
            x1s, y1s, x2s, y2s, valid,
            q_pos=q_pos, q_dim=0.01, r_pos=r_pos, r_dim=0.01
        )
        # Recover centroid from smoothed "corners"
        return sx1, sy1

    # Default: One-Euro
    cx_smooth = one_euro_smooth_1d(cx_list, min_cutoff, beta)
    cy_smooth = one_euro_smooth_1d(cy_list, min_cutoff, beta)
    return cx_smooth, cy_smooth


def _build_translation_grid(dx, dy, H, W, device):
    """Build a sampling grid for a pure translation warp.

    Args:
        dx, dy: Pixel displacement (positive = shift right/down).
        H, W: Image dimensions.
        device: Torch device.

    Returns:
        grid: [1, H, W, 2] normalized sampling grid for grid_sample.
    """
    # Normalized displacement: grid_sample uses [-1, 1] range
    # dx in pixels → normalized = 2 * dx / W
    norm_dx = 2.0 * dx / W
    norm_dy = 2.0 * dy / H

    # Identity grid
    theta = torch.tensor([
        [1.0, 0.0, norm_dx],
        [0.0, 1.0, norm_dy]
    ], device=device, dtype=torch.float32).unsqueeze(0)

    grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
    return grid


class NV_StabilizePlate:
    """Stabilize video by locking mask centroid to a fixed reference position.

    Standard VFX "stabilize plate" technique. Warps each frame so the tracked
    subject stays pixel-locked, eliminating the jitter that causes cascading
    diffusion artifacts. After inpainting, use NV_InverseStabilize to restore
    original motion.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Video frames [B, H, W, C] to stabilize."
                }),
                "mask": ("MASK", {
                    "tooltip": "Per-frame segmentation masks [B, H, W]. "
                               "Centroid of this mask defines the tracking target."
                }),
                "smooth_mode": (["kalman_rts", "one_euro"], {
                    "default": "kalman_rts",
                    "tooltip": "kalman_rts: globally optimal offline smoothing (recommended). "
                               "one_euro: velocity-adaptive, faster but has causal lag."
                }),
                "max_displacement_pct": ("FLOAT", {
                    "default": 0.20, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "Maximum displacement as fraction of image dimension. "
                               "Clamps translation to prevent border encroachment. "
                               "0.20 = up to 20% of width/height."
                }),
            },
            "optional": {
                "reference_cx": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 8192.0, "step": 1.0,
                    "tooltip": "Fixed reference X position (pixels). "
                               "-1 = auto (temporal median of smoothed centroids)."
                }),
                "reference_cy": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 8192.0, "step": 1.0,
                    "tooltip": "Fixed reference Y position (pixels). "
                               "-1 = auto (temporal median of smoothed centroids)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STABILIZE_TRANSFORMS", "STRING")
    RETURN_NAMES = ("stable_image", "stable_mask", "transforms", "info")
    FUNCTION = "stabilize"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Stabilize video by locking mask centroid to a fixed position. "
        "Standard VFX technique: warps frames so the tracked subject is pixel-locked, "
        "eliminating jitter that cascades through diffusion. Use NV_InverseStabilize "
        "after inpainting to restore original motion."
    )

    def stabilize(self, image, mask, smooth_mode, max_displacement_pct,
                  reference_cx=-1.0, reference_cy=-1.0):
        B, H, W, C = image.shape
        device = image.device

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.shape[0] == 1 and B > 1:
            mask = mask.expand(B, -1, -1)

        info_lines = [f"[NV_StabilizePlate] {B} frames, {W}x{H}px, mode={smooth_mode}"]

        # Early exit for single frame
        if B <= 1:
            info_lines.append("  Single frame — no stabilization needed")
            transforms = {
                'dx': [0.0], 'dy': [0.0],
                'H': H, 'W': W,
            }
            info = "\n".join(info_lines)
            print(info)
            return (image, mask, transforms, info)

        # Compute raw centroids
        cx_raw, cy_raw, valid = _compute_centroids(mask)
        info_lines.append(
            f"  Raw centroids: x=[{min(cx_raw):.1f}, {max(cx_raw):.1f}], "
            f"y=[{min(cy_raw):.1f}, {max(cy_raw):.1f}]"
        )

        # Smooth centroid trajectory
        cx_smooth, cy_smooth = _smooth_centroids(
            cx_raw, cy_raw, smooth_mode, valid=valid
        )

        # Reference position (where we lock the centroid to)
        if reference_cx < 0:
            # Temporal median of smoothed centroids
            ref_cx = float(sorted(cx_smooth)[len(cx_smooth) // 2])
        else:
            ref_cx = reference_cx

        if reference_cy < 0:
            ref_cy = float(sorted(cy_smooth)[len(cy_smooth) // 2])
        else:
            ref_cy = reference_cy

        info_lines.append(f"  Reference position: ({ref_cx:.1f}, {ref_cy:.1f})")

        # Compute per-frame displacements
        max_dx = W * max_displacement_pct
        max_dy = H * max_displacement_pct

        dx_list = []
        dy_list = []

        for b in range(B):
            dx = ref_cx - cx_smooth[b]
            dy = ref_cy - cy_smooth[b]

            # Clamp to max displacement
            dx = max(-max_dx, min(max_dx, dx))
            dy = max(-max_dy, min(max_dy, dy))

            dx_list.append(dx)
            dy_list.append(dy)

        max_disp = max(
            max(abs(d) for d in dx_list),
            max(abs(d) for d in dy_list)
        )
        info_lines.append(
            f"  Displacement range: max {max_disp:.1f}px "
            f"(limit: {max_dx:.0f}x{max_dy:.0f}px)"
        )

        # Warp each frame
        stable_images = []
        stable_masks = []

        for b in range(B):
            grid = _build_translation_grid(dx_list[b], dy_list[b], H, W, device)

            # Warp image: [1, H, W, C] → [1, C, H, W] for grid_sample
            img_nchw = image[b:b+1].permute(0, 3, 1, 2)
            warped_img = F.grid_sample(
                img_nchw, grid,
                mode='bilinear', padding_mode='border', align_corners=False
            )
            stable_images.append(warped_img.permute(0, 2, 3, 1).squeeze(0))

            # Warp mask: [1, 1, H, W]
            mask_4d = mask[b:b+1].unsqueeze(1)
            warped_mask = F.grid_sample(
                mask_4d, grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            stable_masks.append(warped_mask.squeeze(0).squeeze(0))

        stable_image = torch.stack(stable_images, dim=0)
        stable_mask = torch.stack(stable_masks, dim=0)

        # Store transforms for inverse warp
        transforms = {
            'dx': dx_list,
            'dy': dy_list,
            'H': H,
            'W': W,
        }

        info = "\n".join(info_lines)
        print(info)

        return (stable_image, stable_mask, transforms, info)


class NV_InverseStabilize:
    """Reverse the stabilization warp to restore original motion.

    After inpainting the stabilized plate and stitching back, apply the
    inverse transform to put pixels back in their original positions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                    "tooltip": "Inpainted+stitched frames to de-stabilize [B, H, W, C]."
                }),
                "transforms": ("STABILIZE_TRANSFORMS", {
                    "tooltip": "Transforms from NV_StabilizePlate."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inverse"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Reverse the stabilization warp from NV_StabilizePlate. "
        "Apply after inpainting + stitching to restore original motion."
    )

    def inverse(self, image, transforms):
        B, H, W, C = image.shape
        device = image.device

        dx_list = transforms['dx']
        dy_list = transforms['dy']

        # Handle batch size mismatch (single-stitcher case)
        if len(dx_list) == 1 and B > 1:
            dx_list = dx_list * B
            dy_list = dy_list * B

        result_frames = []

        for b in range(B):
            # Inverse = negate the displacement
            grid = _build_translation_grid(-dx_list[b], -dy_list[b], H, W, device)

            img_nchw = image[b:b+1].permute(0, 3, 1, 2)
            warped = F.grid_sample(
                img_nchw, grid,
                mode='bilinear', padding_mode='zeros', align_corners=False
            )
            result_frames.append(warped.permute(0, 2, 3, 1).squeeze(0))

        result = torch.stack(result_frames, dim=0)

        print(f"[NV_InverseStabilize] De-stabilized {B} frames")

        return (result,)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "NV_StabilizePlate": NV_StabilizePlate,
    "NV_InverseStabilize": NV_InverseStabilize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_StabilizePlate": "NV Stabilize Plate",
    "NV_InverseStabilize": "NV Inverse Stabilize",
}
