"""
NV Mask Diff Viz — Highlight the EXACT pixels that differ between two masks.

A diagnostic node for "what did this mask operation actually change?" Wire the BEFORE
mask to mask_a and the AFTER mask to mask_b. The viz output paints REMOVED pixels in
one bright color (default red) and ADDED pixels in another (default green), against
either an optional base image or a neutral gray background. The preserved (common)
region can be dimmed so the diff colors really pop.

SET-LOGIC INTERPRETATION (after binary_threshold):
  - A only      = pixels in A but NOT in B  → REMOVED by the operation
  - B only      = pixels in B but NOT in A  → ADDED by the operation
  - Common      = pixels in both A and B    → preserved
  - Symmetric Δ = A_only OR B_only          → "what changed"

USE CASES:
  - What did NV_MaskBinaryCleanup.fill_holes actually fill?
        union_output → mask_a, cleaned_output → mask_b
  - What did NV_TemporalMaskStabilizer_V2 repair this frame?
        raw_per_frame_mask → mask_a, stabilized_mask → mask_b
  - What did NV_MaskUnion.post_close_px close?
        union(close_off) → mask_a, union(close=8) → mask_b
  - Drift between two SAM3 prompts on the same scene
  - Source-actor mask vs warped mask alignment

OUTPUTS:
  viz                  IMAGE  — the colored diff overlay
  a_only_mask          MASK   — REMOVED pixels (in A, not in B)
  b_only_mask          MASK   — ADDED pixels (in B, not in A)
  common_mask          MASK   — preserved pixels (in both)
  symmetric_diff_mask  MASK   — A_only OR B_only (the full "what changed" mask, ready
                                 to feed into other nodes — e.g., a sweep diagnostic)
  summary              STRING — pixel counts, percentages, IoU, Dice
  per_frame_csv        STRING — per-frame stats for animated masks

BOUNDARY_ONLY MODE:
  Most mask operations (close, fill_holes, erode/dilate, temporal stabilizer) primarily
  change boundary pixels. When boundary_only=True, the viz restricts visible diff to
  pixels within `boundary_width_px` of either mask's edge — interior fills (e.g. closed
  holes far from any edge) are hidden so you see ONLY edge-level deltas. The MASK outputs
  still represent the full diff regardless of this toggle.
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F


# Pure / saturated diff colors — designed for unmissable highlight against any base image.
_BRIGHT_COLORS = OrderedDict([
    ("bright_red",     (1.00, 0.00, 0.00)),
    ("bright_green",   (0.00, 1.00, 0.00)),
    ("bright_blue",    (0.00, 0.10, 1.00)),
    ("bright_yellow",  (1.00, 1.00, 0.00)),
    ("bright_magenta", (1.00, 0.00, 1.00)),
    ("bright_cyan",    (0.00, 1.00, 1.00)),
    ("white",          (1.00, 1.00, 1.00)),
])
_BRIGHT_NAMES = list(_BRIGHT_COLORS.keys())

_COMMON_MODES = ["dim", "tint", "off"]


class NV_MaskDiffViz:
    """Symmetric-difference visualization for two masks. Bright colors on the changed pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK", {
                    "tooltip": "BEFORE mask. Pixels here but not in mask_b will be highlighted as REMOVED."
                }),
                "mask_b": ("MASK", {
                    "tooltip": "AFTER mask. Pixels here but not in mask_a will be highlighted as ADDED."
                }),
                "a_only_color": (_BRIGHT_NAMES, {
                    "default": "bright_red",
                    "tooltip": "Color for REMOVED pixels (in A, not in B). Pure-saturated for unmissable highlight."
                }),
                "b_only_color": (_BRIGHT_NAMES, {
                    "default": "bright_green",
                    "tooltip": "Color for ADDED pixels (in B, not in A)."
                }),
                "diff_opacity": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Opacity of the bright diff colors over the base. 1.0 = fully replace, 0.5 = half-blend."
                }),
                "common_mode": (_COMMON_MODES, {
                    "default": "dim",
                    "tooltip": "How to render preserved (common) pixels. "
                               "dim: darken so diffs pop. "
                               "tint: blend toward common_color. "
                               "off: leave preserved pixels untouched on the base."
                }),
                "common_opacity": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Strength of dim/tint applied to common (preserved) pixels."
                }),
                "common_color": (_BRIGHT_NAMES, {
                    "default": "white",
                    "tooltip": "Color used by common_mode='tint'. Ignored for dim/off modes."
                }),
                "binary_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "Threshold for binarizing soft masks before diffing. 0.5 = standard."
                }),
                "boundary_only": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, hide diff pixels deep inside a mask (e.g., filled holes) and show only edge deltas. "
                               "MASK outputs are unaffected by this toggle."
                }),
                "boundary_width_px": ("INT", {
                    "default": 4, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Width of the boundary band when boundary_only=True. Larger = thicker visible band."
                }),
                "print_summary": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Print pixel-count summary to stdout."
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Base image to overlay diffs on (typically the source crop). "
                               "If absent, a neutral gray background is generated."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("viz", "a_only_mask", "b_only_mask", "common_mask", "symmetric_diff_mask",
                    "summary", "per_frame_csv")
    FUNCTION = "diff"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Highlight the exact pixels that differ between two masks (e.g., before/after a mask "
        "operation). Bright pure colors on the symmetric difference; dimmed common region. "
        "Outputs the diff masks individually plus the combined symmetric_diff_mask for downstream use."
    )

    # ===================================================================
    # Helpers
    # ===================================================================

    @staticmethod
    def _promote_mask(mask, name):
        """[H,W] or [B,H,W] → [B,H,W] float in [0, 1].

        Normalizes uint8 / [0, 255]-ranged masks to [0, 1] so binary_threshold=0.5
        works correctly. Without this, a uint8 mask of values 0/255 would treat almost
        every nonzero pixel as True at threshold 0.5 (because 255 > 0.5 trivially).
        """
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim != 3:
            raise ValueError(f"[NV_MaskDiffViz] {name} must be 2D or 3D, got {mask.ndim}D shape={tuple(mask.shape)}")
        mask = mask.float()
        # Auto-normalize if values are in [0, 255] range (uint8 or float "as bytes").
        if mask.numel() > 0 and float(mask.max()) > 1.5:
            mask = mask / 255.0
        return mask

    @staticmethod
    def _align_batches(a, b, image):
        """Broadcast B=1 masks/images up to the largest batch among them."""
        Bs = [a.shape[0], b.shape[0]]
        if image is not None:
            Bs.append(image.shape[0])
        B_max = max(Bs)
        def expand(t, name):
            if t is None:
                return None
            if t.shape[0] == B_max:
                return t
            if t.shape[0] == 1:
                return t.expand(B_max, *t.shape[1:])
            raise ValueError(f"[NV_MaskDiffViz] {name} batch {t.shape[0]} doesn't match {B_max}")
        return expand(a, "mask_a"), expand(b, "mask_b"), expand(image, "image")

    @staticmethod
    def _resize_to(t, H, W, mode):
        """Resize [B, H_in, W_in] → [B, H, W] (mask) or [B, H_in, W_in, 3] → [B, H, W, 3] (image)."""
        if t is None or t.shape[1] == H and t.shape[2] == W:
            return t
        if t.ndim == 3:  # mask
            x = t.unsqueeze(1).float()
            x = F.interpolate(x, size=(H, W), mode=mode)
            return x.squeeze(1)
        # image
        x = t.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1)

    @staticmethod
    def _dilate_bool(x_bool, k):
        """Morphological dilate via max_pool2d. x_bool: [B, H, W] bool. k: kernel size (odd)."""
        if k <= 1:
            return x_bool
        x = x_bool.float().unsqueeze(1)
        pad = k // 2
        out = F.max_pool2d(x, kernel_size=k, stride=1, padding=pad)
        return out.squeeze(1) > 0.5

    @classmethod
    def _erode_bool(cls, x_bool, k):
        """Morphological erode via dilation of complement."""
        if k <= 1:
            return x_bool
        return ~cls._dilate_bool(~x_bool, k)

    @classmethod
    def _boundary_band(cls, a_bin, b_bin, width_px):
        """Compute a band of `width_px` pixels on EACH side of either mask's boundary.

        Direct construction: `dilate(mask, k) & ~erode(mask, k)` produces a symmetric band
        extending ~width_px pixels INWARD and ~width_px pixels OUTWARD from the true edge.
        Note: an all-1 mask has no boundary by this definition (erosion stays all-1, so the
        XOR is empty). Image-domain borders are NOT treated as object boundaries.
        """
        k = 2 * width_px + 1
        band_a = cls._dilate_bool(a_bin, k) & (~cls._erode_bool(a_bin, k))
        band_b = cls._dilate_bool(b_bin, k) & (~cls._erode_bool(b_bin, k))
        return band_a | band_b

    # ===================================================================
    # Main
    # ===================================================================

    def diff(self, mask_a, mask_b, a_only_color, b_only_color, diff_opacity,
             common_mode, common_opacity, common_color, binary_threshold,
             boundary_only, boundary_width_px, print_summary, image=None):

        a = self._promote_mask(mask_a, "mask_a")
        b = self._promote_mask(mask_b, "mask_b")
        img = image if image is not None else None

        # Validate image shape BEFORE batch alignment so the user sees the correct
        # error ("wrong rank/channels") rather than a misleading "batch mismatch"
        # when they pass a [H, W, 3] tensor (rank 3, where shape[0] would be H).
        if img is not None and (img.ndim != 4 or img.shape[-1] not in (3, 4)):
            raise ValueError(
                f"[NV_MaskDiffViz] image must be [B, H, W, 3] or [B, H, W, 4], "
                f"got rank {img.ndim} with shape {tuple(img.shape)}"
            )

        # Align batches and spatial dims
        a, b, img = self._align_batches(a, b, img)

        # Establish reference H, W from mask_a; resize others to match.
        # NOTE: mask_a is the resolution authority. If you need a different reference,
        # resize upstream before wiring into this node.
        B, H, W = a.shape
        if b.shape[1:] != (H, W):
            b = self._resize_to(b, H, W, mode="nearest")
        if img is not None:
            img = img[..., :3].contiguous().float()
            if img.shape[1:3] != (H, W):
                img = self._resize_to(img, H, W, mode="bilinear")

        device = a.device
        a_bin = a > binary_threshold
        b_bin = b > binary_threshold

        a_only = a_bin & (~b_bin)
        b_only = b_bin & (~a_bin)
        common = a_bin & b_bin
        sym_diff = a_only | b_only

        # ---- per-frame stats ----
        per_frame = self._compute_stats(a_bin, b_bin, a_only, b_only, common, sym_diff, H, W)
        agg = self._aggregate_stats(per_frame)

        # ---- viz ----
        viz = self._build_viz(
            a_only=a_only, b_only=b_only, common=common,
            base_image=img, B=B, H=H, W=W, device=device,
            a_only_color=a_only_color, b_only_color=b_only_color, common_color=common_color,
            diff_opacity=float(diff_opacity), common_mode=common_mode,
            common_opacity=float(common_opacity),
            boundary_only=bool(boundary_only), boundary_width_px=int(boundary_width_px),
            a_bin=a_bin, b_bin=b_bin,
        )

        # ---- format outputs ----
        a_only_mask = a_only.float()
        b_only_mask = b_only.float()
        common_mask = common.float()
        symmetric_diff_mask = sym_diff.float()

        summary = self._make_summary(B, H, W, agg, boundary_only, boundary_width_px)
        per_frame_csv = self._make_csv(per_frame)

        if print_summary:
            print(f"\n[NV_MaskDiffViz]\n{summary}\n")

        return (viz, a_only_mask, b_only_mask, common_mask, symmetric_diff_mask, summary, per_frame_csv)

    # ===================================================================
    # Per-frame statistics
    # ===================================================================

    @staticmethod
    def _compute_stats(a_bin, b_bin, a_only, b_only, common, sym_diff, H, W):
        B = a_bin.shape[0]
        frame_total = float(H * W)
        rows = []
        # Vectorized counts per frame [B]
        a_count = a_bin.view(B, -1).sum(dim=1).float()
        b_count = b_bin.view(B, -1).sum(dim=1).float()
        a_only_count = a_only.view(B, -1).sum(dim=1).float()
        b_only_count = b_only.view(B, -1).sum(dim=1).float()
        common_count = common.view(B, -1).sum(dim=1).float()
        union_count = (a_bin | b_bin).view(B, -1).sum(dim=1).float()
        sym_count = sym_diff.view(B, -1).sum(dim=1).float()

        # IoU/Dice: both empty masks ARE identical by definition → return 1.0, not 0.
        # Without this guard, 0/clamp(0,min=1) = 0, which would falsely report total mismatch.
        iou = torch.where(
            union_count > 0,
            common_count / union_count.clamp(min=1.0),
            torch.ones_like(union_count),
        )
        dice_den = a_count + b_count
        dice = torch.where(
            dice_den > 0,
            (2.0 * common_count) / dice_den.clamp(min=1.0),
            torch.ones_like(dice_den),
        )
        # Symmetric-diff percentage: 0% when both masks empty (no diff at all). Keep raw 0/clamp.
        sym_pct_of_union = sym_count / union_count.clamp(min=1.0) * 100.0
        a_only_pct_of_a = a_only_count / a_count.clamp(min=1.0) * 100.0
        b_only_pct_of_b = b_only_count / b_count.clamp(min=1.0) * 100.0

        # Single CPU transfer
        a_count_np = a_count.detach().cpu().numpy()
        b_count_np = b_count.detach().cpu().numpy()
        a_only_np = a_only_count.detach().cpu().numpy()
        b_only_np = b_only_count.detach().cpu().numpy()
        common_np = common_count.detach().cpu().numpy()
        union_np = union_count.detach().cpu().numpy()
        sym_np = sym_count.detach().cpu().numpy()
        iou_np = iou.detach().cpu().numpy()
        dice_np = dice.detach().cpu().numpy()
        sym_pct_np = sym_pct_of_union.detach().cpu().numpy()
        a_only_pct_np = a_only_pct_of_a.detach().cpu().numpy()
        b_only_pct_np = b_only_pct_of_b.detach().cpu().numpy()

        for i in range(B):
            rows.append({
                "frame": i,
                "a_pixels": int(a_count_np[i]),
                "b_pixels": int(b_count_np[i]),
                "a_only_pixels": int(a_only_np[i]),
                "b_only_pixels": int(b_only_np[i]),
                "common_pixels": int(common_np[i]),
                "union_pixels": int(union_np[i]),
                "sym_diff_pixels": int(sym_np[i]),
                "iou": round(float(iou_np[i]), 4),
                "dice": round(float(dice_np[i]), 4),
                "sym_diff_pct_of_union": round(float(sym_pct_np[i]), 2),
                "a_only_pct_of_a": round(float(a_only_pct_np[i]), 2),
                "b_only_pct_of_b": round(float(b_only_pct_np[i]), 2),
                "frame_pixels": int(frame_total),
            })
        return rows

    @staticmethod
    def _aggregate_stats(per_frame):
        if not per_frame:
            return {}
        arr = lambda key: np.array([f[key] for f in per_frame], dtype=np.float64)
        agg = {}
        for key in ("a_pixels", "b_pixels", "a_only_pixels", "b_only_pixels",
                   "common_pixels", "sym_diff_pixels", "iou", "dice",
                   "sym_diff_pct_of_union", "a_only_pct_of_a", "b_only_pct_of_b"):
            v = arr(key)
            agg[key] = {
                "min": float(v.min()),
                "max": float(v.max()),
                "mean": float(v.mean()),
                "p50": float(np.percentile(v, 50)),
            }
        # Worst-frame indices
        # IoU: lowest = most diverged
        ious = [f["iou"] for f in per_frame]
        agg["iou"]["worst_frame"] = int(np.argmin(ious))
        # sym_diff_pct: highest = most diverged
        syms = [f["sym_diff_pct_of_union"] for f in per_frame]
        agg["sym_diff_pct_of_union"]["worst_frame"] = int(np.argmax(syms))
        return agg

    # ===================================================================
    # Visualization
    # ===================================================================

    def _build_viz(self, a_only, b_only, common, base_image, B, H, W, device,
                   a_only_color, b_only_color, common_color,
                   diff_opacity, common_mode, common_opacity,
                   boundary_only, boundary_width_px, a_bin, b_bin):
        # Base
        if base_image is None:
            base = torch.full((B, H, W, 3), 0.30, device=device, dtype=torch.float32)
        else:
            base = base_image.to(device).float()

        out = base.clone()

        # Apply common region treatment
        if common_mode != "off":
            common_f = common.float().unsqueeze(-1)  # [B, H, W, 1]
            ca = max(0.0, min(1.0, float(common_opacity)))
            if common_mode == "dim":
                # Lerp toward black at common_opacity
                target = torch.zeros_like(out)
            else:  # tint
                tint = torch.tensor(_BRIGHT_COLORS[common_color], device=device, dtype=out.dtype).view(1, 1, 1, 3)
                target = tint.expand_as(out)
            out = out * (1.0 - common_f * ca) + target * (common_f * ca)

        # Apply diff highlights — optionally restricted to boundary band
        if boundary_only:
            band = self._boundary_band(a_bin, b_bin, boundary_width_px)
            a_show = a_only & band
            b_show = b_only & band
        else:
            a_show = a_only
            b_show = b_only

        a_color = torch.tensor(_BRIGHT_COLORS[a_only_color], device=device, dtype=out.dtype).view(1, 1, 1, 3)
        b_color = torch.tensor(_BRIGHT_COLORS[b_only_color], device=device, dtype=out.dtype).view(1, 1, 1, 3)
        da = max(0.0, min(1.0, float(diff_opacity)))

        a_show_f = a_show.float().unsqueeze(-1)
        b_show_f = b_show.float().unsqueeze(-1)
        out = out * (1.0 - a_show_f * da) + a_color.expand_as(out) * (a_show_f * da)
        out = out * (1.0 - b_show_f * da) + b_color.expand_as(out) * (b_show_f * da)

        return out.clamp(0, 1)

    # ===================================================================
    # Output formatters
    # ===================================================================

    @staticmethod
    def _make_csv(per_frame):
        if not per_frame:
            return ""
        keys = list(per_frame[0].keys())
        lines = [",".join(keys)]
        for f in per_frame:
            row = []
            for k in keys:
                v = f[k]
                if isinstance(v, float):
                    row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
            lines.append(",".join(row))
        return "\n".join(lines)

    @staticmethod
    def _make_summary(B, H, W, agg, boundary_only, boundary_width_px):
        bar = "=" * 80
        frame_total = H * W
        lines = [
            bar,
            f" NV_MaskDiffViz — {B} frames | resolution {W}x{H}",
            f" boundary_only={'YES' if boundary_only else 'no'}"
            + (f" (band={boundary_width_px} px)" if boundary_only else ""),
            bar,
        ]
        if not agg:
            lines.append("  (no frames)")
            lines.append(bar)
            return "\n".join(lines)

        def pct_of_frame(v):
            return v / frame_total * 100.0

        lines.append("  Pixel counts (averaged across frames):")
        lines.append(f"    Mask A pixels       : avg {agg['a_pixels']['mean']:>10,.0f}   ({pct_of_frame(agg['a_pixels']['mean']):5.2f}% of frame)")
        lines.append(f"    Mask B pixels       : avg {agg['b_pixels']['mean']:>10,.0f}   ({pct_of_frame(agg['b_pixels']['mean']):5.2f}% of frame)")
        lines.append(f"    A only (REMOVED)    : avg {agg['a_only_pixels']['mean']:>10,.0f}   ({agg['a_only_pct_of_a']['mean']:5.2f}% of A)   max {agg['a_only_pixels']['max']:>10,.0f}")
        lines.append(f"    B only (ADDED)      : avg {agg['b_only_pixels']['mean']:>10,.0f}   ({agg['b_only_pct_of_b']['mean']:5.2f}% of B)   max {agg['b_only_pixels']['max']:>10,.0f}")
        lines.append(f"    Common (preserved)  : avg {agg['common_pixels']['mean']:>10,.0f}")
        lines.append(f"    Symmetric diff      : avg {agg['sym_diff_pixels']['mean']:>10,.0f}   ({agg['sym_diff_pct_of_union']['mean']:5.2f}% of union)")
        lines.append("")
        lines.append("  Similarity:")
        lines.append(f"    IoU(A,B)  : mean={agg['iou']['mean']:.4f}  min={agg['iou']['min']:.4f}  worst@frame={agg['iou']['worst_frame']}")
        lines.append(f"    Dice(A,B) : mean={agg['dice']['mean']:.4f}  min={agg['dice']['min']:.4f}")
        lines.append("")
        # Sanity hint for trivial / degenerate inputs so the user doesn't think the node is broken.
        if agg["sym_diff_pixels"]["mean"] == 0:
            if agg["a_pixels"]["mean"] == 0 and agg["b_pixels"]["mean"] == 0:
                lines.append("  ⓘ Both masks are empty across all frames — no diff to show.")
            else:
                lines.append("  ⓘ Masks are identical across all frames after threshold — viz will show no diff highlights.")
            lines.append("")

        lines.append("  Reading guide:")
        lines.append("    IoU > 0.99  = nearly identical (operation barely changed anything)")
        lines.append("    IoU 0.95-0.99 = small targeted change (e.g., light cleanup, hole fill)")
        lines.append("    IoU 0.85-0.95 = noticeable change (e.g., dilation/erosion or seam close)")
        lines.append("    IoU < 0.85  = large change — verify this is what you intended")
        lines.append(bar)
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "NV_MaskDiffViz": NV_MaskDiffViz,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_MaskDiffViz": "NV Mask Diff Viz",
}
