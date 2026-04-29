"""
NV Image Diff Analyzer — Pixel-by-pixel comparison of two images.

Outputs an amplified difference heatmap and per-channel statistics.
Optionally takes a mask to EXCLUDE the generated/inpainted region,
measuring only the kept pixels (where source and result should match).

Use case: diagnosing VAE roundtrip degradation, diffusion color drift,
or stitch seam color mismatch by visualizing exactly where and how much
pixels change.

ARCHITECTURE NOTE: this is a HUMAN-DEBUG visualization wrapper. Numeric
metrics that feed the agent corpus go through diff_ops.compute_diff_metrics
via NV_ShotMeasure / NV_ShotRecord. The boundary-mask helper is shared
(imported from diff_ops below) so the chart numbers and the JSONL numbers
can never disagree on zone definitions.
"""

import torch
import torch.nn.functional as F

from .diff_ops import (
    _compute_boundary_mask as _diff_ops_boundary_mask,
    _percentile as _diff_ops_percentile,
)


class NV_ImageDiffAnalyzer:
    """Pixel-by-pixel image comparison with amplified heatmap output."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image (e.g., original source)"}),
                "image_b": ("IMAGE", {"tooltip": "Second image (e.g., VAE roundtripped or stitched result)"}),
                "amplify": ("FLOAT", {
                    "default": 10.0, "min": 1.0, "max": 100.0, "step": 1.0,
                    "tooltip": "Multiplier for the difference visualization. Higher = more visible subtle differences."
                }),
                "mode": (["absolute_diff", "signed_rgb", "luminance_diff", "heatmap"], {
                    "default": "heatmap",
                    "tooltip": "absolute_diff: |A-B| amplified. signed_rgb: red=B brighter, blue=A brighter. "
                               "luminance_diff: grayscale brightness delta. heatmap: magnitude as cool-to-hot gradient."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Mask of the GENERATED region (mask=1 = inpainted area). "
                               "Stats are computed only on UNMASKED pixels (mask=0 = kept region). "
                               "The heatmap still shows all pixels but stats focus on the kept area."
                }),
                "stitcher": ("STITCHER", {
                    "tooltip": "Connect the STITCHER from InpaintCrop2. Extracts the blend mask, crop coords, "
                               "and stitch parameters for analysis. Overrides the 'mask' input if both connected."
                }),
                "mask_config": ("MASK_PROCESSING_CONFIG", {
                    "tooltip": "Connect NV_MaskProcessingConfig to log the mask processing settings "
                               "used in the pipeline (erode/dilate, blend pixels, etc.)."
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Pixels with mask value below this are considered 'kept' (measured)."
                }),
                "boundary_width": ("INT", {
                    "default": 16, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Width in pixels of the boundary zone around the mask edge. "
                               "Used to split stats into 'boundary' vs 'interior' kept pixels."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING",)
    RETURN_NAMES = ("diff_heatmap", "masked_diff", "stats_text",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = "Pixel-by-pixel comparison of two images. Outputs amplified diff heatmap and per-channel statistics."

    def execute(self, image_a, image_b, amplify, mode, mask=None, stitcher=None,
                mask_config=None, mask_threshold=0.5, boundary_width=16):
        # Ensure same shape — resize B to match A if needed
        if image_a.shape != image_b.shape:
            b_resized = image_b.permute(0, 3, 1, 2)
            b_resized = F.interpolate(b_resized, size=(image_a.shape[1], image_a.shape[2]), mode="bilinear", align_corners=False)
            image_b = b_resized.permute(0, 2, 3, 1)

        diff = image_b.float() - image_a.float()
        abs_diff = diff.abs()

        # Extract mask from stitcher if provided (overrides manual mask)
        stitcher_info = None
        if stitcher is not None:
            mask, stitcher_info = self._extract_stitcher_mask(stitcher, image_a.shape)

        # Build visualization based on mode
        if mode == "absolute_diff":
            vis = (abs_diff * amplify).clamp(0, 1)
        elif mode == "signed_rgb":
            lum_diff = diff.mean(dim=-1, keepdim=True)
            red = lum_diff.clamp(min=0) * amplify
            green = torch.zeros_like(red)
            blue = (-lum_diff).clamp(min=0) * amplify
            vis = torch.cat([red, green, blue], dim=-1).clamp(0, 1)
        elif mode == "luminance_diff":
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image_a.device)
            lum_a = (image_a * weights).sum(dim=-1, keepdim=True)
            lum_b = (image_b * weights).sum(dim=-1, keepdim=True)
            lum_diff = (lum_b - lum_a).abs() * amplify
            vis = lum_diff.expand(-1, -1, -1, 3).clamp(0, 1)
        elif mode == "heatmap":
            magnitude = abs_diff.mean(dim=-1)
            magnitude = (magnitude * amplify).clamp(0, 1)
            vis = self._apply_heatmap(magnitude)

        # Prepare mask zones — mirror diff_ops.compute_diff_metrics exactly so
        # the legacy chart numbers can never disagree with the JSONL numbers.
        # 2D→3D promote handles ComfyUI's bare [H, W] mask shape.
        # `nearest` keeps mask edges crisp before threshold; bilinear would
        # bias zone occupancy and split this node's stats from the agent path.
        m_resized = None
        if mask is not None:
            m = mask.float()
            if m.dim() == 2:
                m = m.unsqueeze(0)
            if m.shape[1:] != image_a.shape[1:3]:
                m = F.interpolate(m.unsqueeze(1), size=(image_a.shape[1], image_a.shape[2]), mode="nearest").squeeze(1)
            if m.shape[0] == 1 and vis.shape[0] > 1:
                m = m.expand(vis.shape[0], -1, -1)
            m_resized = m
            keep = (m < mask_threshold).float().unsqueeze(-1)
            masked_vis = vis * keep
        else:
            masked_vis = vis

        # Compute statistics
        stats = self._compute_stats(
            image_a, image_b, abs_diff, diff, m_resized, mask_threshold, boundary_width,
            stitcher_info=stitcher_info, mask_config=mask_config,
        )

        return (vis, masked_vis, stats,)

    def _extract_stitcher_mask(self, stitcher, image_shape):
        """Extract blend mask from STITCHER dict and build a full-frame mask.

        The stitcher carries per-frame blend masks at crop resolution + coordinates.
        We reconstruct the full-frame mask showing where blending occurs.

        Returns: (mask [B, H, W], stitcher_info dict for stats)
        """
        B, H, W, C = image_shape
        device = "cpu"

        blend_masks = stitcher.get("cropped_mask_for_blend", [])
        skipped = set(stitcher.get("skipped_indices", []))
        total_frames = stitcher.get("total_frames", B)
        single_stitcher = len(stitcher.get("cropped_to_canvas_x", [])) == 1 and B > 1

        # Collect stitcher metadata for the report
        info = {
            "total_frames": total_frames,
            "skipped_frames": len(skipped),
            "stitch_source": stitcher.get("stitch_source", "unknown"),
            "resize_algorithm": stitcher.get("resize_algorithm", stitcher.get("upscale_algorithm", "unknown")),
            "num_blend_masks": len(blend_masks),
            "single_stitcher": single_stitcher,
        }

        # Sample crop coords for the report
        if stitcher.get("cropped_to_canvas_w"):
            info["crop_w"] = stitcher["cropped_to_canvas_w"][0]
            info["crop_h"] = stitcher["cropped_to_canvas_h"][0]
        if stitcher.get("canvas_to_orig_w"):
            info["orig_w"] = stitcher["canvas_to_orig_w"][0]
            info["orig_h"] = stitcher["canvas_to_orig_h"][0]

        # Build full-frame mask: 0 = untouched, blend_value = blended, 1 = fully generated
        full_mask = torch.zeros(B, H, W, device=device)

        inpaint_idx = 0
        for frame in range(min(B, total_frames)):
            if frame in skipped:
                continue

            idx = 0 if single_stitcher else inpaint_idx

            if idx >= len(blend_masks):
                inpaint_idx += 1
                continue

            bm = blend_masks[idx].float()  # [crop_H, crop_W] or [1, crop_H, crop_W]
            if bm.dim() == 3:
                bm = bm.squeeze(0)

            # Get coordinates to place in full frame
            cto_x = stitcher["canvas_to_orig_x"][idx]
            cto_y = stitcher["canvas_to_orig_y"][idx]
            cto_w = stitcher["canvas_to_orig_w"][idx]
            cto_h = stitcher["canvas_to_orig_h"][idx]

            # Resize blend mask from crop space to original bbox space
            if bm.shape[0] != cto_h or bm.shape[1] != cto_w:
                bm_resized = F.interpolate(
                    bm.unsqueeze(0).unsqueeze(0), size=(cto_h, cto_w), mode="bilinear", align_corners=False
                ).squeeze(0).squeeze(0)
            else:
                bm_resized = bm

            # Clip to frame bounds
            src_y_start = max(0, -cto_y)
            src_x_start = max(0, -cto_x)
            dst_y_start = max(0, cto_y)
            dst_x_start = max(0, cto_x)
            copy_h = min(cto_h - src_y_start, H - dst_y_start)
            copy_w = min(cto_w - src_x_start, W - dst_x_start)

            if copy_h > 0 and copy_w > 0:
                full_mask[frame, dst_y_start:dst_y_start + copy_h, dst_x_start:dst_x_start + copy_w] = \
                    bm_resized[src_y_start:src_y_start + copy_h, src_x_start:src_x_start + copy_w]

            inpaint_idx += 1

        return full_mask, info

    def _apply_heatmap(self, magnitude):
        """Convert [B, H, W] magnitude (0-1) to [B, H, W, 3] RGB heatmap."""
        r = torch.zeros_like(magnitude)
        g = torch.zeros_like(magnitude)
        b = torch.zeros_like(magnitude)

        # 0.0-0.2: black to blue
        t = (magnitude / 0.2).clamp(0, 1)
        mask0 = magnitude <= 0.2
        b = torch.where(mask0, t, b)

        # 0.2-0.4: blue to cyan
        t = ((magnitude - 0.2) / 0.2).clamp(0, 1)
        mask1 = (magnitude > 0.2) & (magnitude <= 0.4)
        b = torch.where(mask1, torch.ones_like(b), b)
        g = torch.where(mask1, t, g)

        # 0.4-0.6: cyan to yellow
        t = ((magnitude - 0.4) / 0.2).clamp(0, 1)
        mask2 = (magnitude > 0.4) & (magnitude <= 0.6)
        r = torch.where(mask2, t, r)
        g = torch.where(mask2, torch.ones_like(g), g)
        b = torch.where(mask2, 1.0 - t, b)

        # 0.6-0.8: yellow to red
        t = ((magnitude - 0.6) / 0.2).clamp(0, 1)
        mask3 = (magnitude > 0.6) & (magnitude <= 0.8)
        r = torch.where(mask3, torch.ones_like(r), r)
        g = torch.where(mask3, 1.0 - t, g)

        # 0.8-1.0: red to white
        t = ((magnitude - 0.8) / 0.2).clamp(0, 1)
        mask4 = magnitude > 0.8
        r = torch.where(mask4, torch.ones_like(r), r)
        g = torch.where(mask4, t, g)
        b = torch.where(mask4, t, b)

        return torch.stack([r, g, b], dim=-1)

    def _compute_boundary_mask(self, mask_binary, width):
        """Boundary zone — kept pixels within `width` of the mask edge.

        Delegates to the shared op so the legacy viz node and the new
        agent-corpus path use the same zone definition.
        """
        return _diff_ops_boundary_mask(mask_binary, width)

    def _zone_stats(self, abs_diff, signed_diff, zone_mask, channel_names=("Red", "Green", "Blue")):
        """Compute stats for a specific zone. Returns list of formatted lines."""
        lines = []
        count = zone_mask.sum().item()
        if count == 0:
            lines.append("  (no pixels in this zone)")
            return lines, count

        for c, name in enumerate(channel_names):
            ch_abs = abs_diff[..., c][zone_mask]
            ch_signed = signed_diff[..., c][zone_mask]

            mean_abs = ch_abs.mean().item()
            max_abs = ch_abs.max().item()
            median_abs = ch_abs.median().item()
            mean_signed = ch_signed.mean().item()
            std_signed = ch_signed.std().item()

            # Percentiles via shared helper — matches diff_ops indexing convention
            # (round(q * (n-1))) so this node's text report and the JSONL
            # numbers agree on small zones.
            p50 = _diff_ops_percentile(ch_abs, 0.50)
            p90 = _diff_ops_percentile(ch_abs, 0.90)
            p95 = _diff_ops_percentile(ch_abs, 0.95)
            p99 = _diff_ops_percentile(ch_abs, 0.99)

            lines.append(f"  {name}:")
            lines.append(f"    Mean |diff|:  {mean_abs:.6f}  ({mean_abs*255:.2f}/255)")
            lines.append(f"    Median |diff|:{median_abs:.6f}  ({median_abs*255:.2f}/255)")
            lines.append(f"    Max |diff|:   {max_abs:.6f}  ({max_abs*255:.2f}/255)")
            lines.append(f"    Percentiles:  p50={p50*255:.1f}  p90={p90*255:.1f}  p95={p95*255:.1f}  p99={p99*255:.1f}  (all /255)")
            lines.append(f"    Mean signed:  {mean_signed:+.6f}  ({mean_signed*255:+.2f}/255)  {'(B brighter)' if mean_signed > 0 else '(A brighter)'}")
            lines.append(f"    Std signed:   {std_signed:.6f}  ({std_signed*255:.2f}/255)")

        return lines, count

    def _compute_stats(self, image_a, image_b, abs_diff, signed_diff, mask, mask_threshold, boundary_width,
                        stitcher_info=None, mask_config=None):
        """Compute verbose per-channel, per-zone, and per-frame statistics."""
        lines = []
        B, H, W, C = image_a.shape

        lines.append("=" * 70)
        lines.append("NV IMAGE DIFF ANALYZER — DIAGNOSTIC REPORT")
        lines.append("=" * 70)
        lines.append("")

        # --- Input info ---
        lines.append(f"Image A shape: {list(image_a.shape)}  (B={B}, H={H}, W={W}, C={C})")
        lines.append(f"Image B shape: {list(image_b.shape)}")
        lines.append(f"Frames (batch): {B}")
        lines.append("")

        # --- Stitcher info ---
        if stitcher_info is not None:
            lines.append("STITCHER CONFIG:")
            lines.append(f"  Stitch source:     {stitcher_info.get('stitch_source', '?')}")
            lines.append(f"  Resize algorithm:  {stitcher_info.get('resize_algorithm', '?')}")
            lines.append(f"  Total frames:      {stitcher_info.get('total_frames', '?')}")
            lines.append(f"  Skipped frames:    {stitcher_info.get('skipped_frames', '?')}")
            lines.append(f"  Single stitcher:   {stitcher_info.get('single_stitcher', '?')}")
            if "crop_w" in stitcher_info:
                lines.append(f"  Crop size:         {stitcher_info['crop_w']}x{stitcher_info['crop_h']} (cropped_to_canvas)")
            if "orig_w" in stitcher_info:
                lines.append(f"  Original bbox:     {stitcher_info['orig_w']}x{stitcher_info['orig_h']} (canvas_to_orig)")
                if "crop_w" in stitcher_info:
                    scale_x = stitcher_info["orig_w"] / max(stitcher_info["crop_w"], 1)
                    scale_y = stitcher_info["orig_h"] / max(stitcher_info["crop_h"], 1)
                    lines.append(f"  Resize factor:     {scale_x:.2f}x × {scale_y:.2f}x (crop→orig)")
            lines.append(f"  Mask source:       STITCHER blend_mask (continuous 0-1, not binary)")
            lines.append("")

        # --- Mask config info ---
        if mask_config is not None:
            lines.append("MASK PROCESSING CONFIG:")
            for key, val in sorted(mask_config.items()):
                lines.append(f"  {key}: {val}")
            lines.append("")

        # --- Mask info ---
        blend_zone = None
        if mask is not None:
            gen_mask = mask >= mask_threshold  # [B, H, W] bool — True = generated
            keep_mask = ~gen_mask  # True = kept
            total_px = gen_mask.numel()
            kept_px = keep_mask.sum().item()
            gen_px = gen_mask.sum().item()

            # For continuous masks (from stitcher), also compute blend transition zone
            # Blend zone: pixels that are partially blended (0 < mask < threshold)
            is_continuous = (mask > 0).sum().item() != (mask >= mask_threshold).sum().item()
            if is_continuous:
                blend_zone = (mask > 0.01) & (mask < mask_threshold)  # partially blended kept pixels
                untouched_zone = mask <= 0.01  # truly untouched pixels
                blend_px = blend_zone.sum().item()
                untouched_px = untouched_zone.sum().item()

                lines.append(f"Mask: continuous (from stitcher blend mask, threshold={mask_threshold})")
                lines.append(f"  Total pixels:     {total_px:>10,}")
                lines.append(f"  Untouched (=0):   {int(untouched_px):>10,}  ({100*untouched_px/total_px:.1f}%)  [mask ≤ 0.01]")
                lines.append(f"  Blend zone:       {int(blend_px):>10,}  ({100*blend_px/total_px:.1f}%)  [0.01 < mask < {mask_threshold}]")
                lines.append(f"  Generated:        {int(gen_px):>10,}  ({100*gen_px/total_px:.1f}%)  [mask ≥ {mask_threshold}]")
                lines.append(f"  Kept (measured):  {int(kept_px):>10,}  ({100*kept_px/total_px:.1f}%)  [untouched + blend]")

                # Mask value distribution in blend zone
                if blend_px > 0:
                    bz_vals = mask[blend_zone]
                    lines.append(f"  Blend mask range: {bz_vals.min().item():.3f} — {bz_vals.max().item():.3f}")
                    lines.append(f"  Blend mask mean:  {bz_vals.mean().item():.3f}")
            else:
                lines.append(f"Mask: binary (threshold={mask_threshold})")
                lines.append(f"  Total pixels:     {total_px:>10,}")
                lines.append(f"  Kept pixels:      {int(kept_px):>10,}  ({100*kept_px/total_px:.1f}%)")
                lines.append(f"  Generated pixels: {int(gen_px):>10,}  ({100*gen_px/total_px:.1f}%)")

            # Compute boundary zone (based on binary gen_mask edge)
            boundary_zone = self._compute_boundary_mask(gen_mask, boundary_width)
            interior_zone = keep_mask & (~boundary_zone)
            bnd_px = boundary_zone.sum().item()
            int_px = interior_zone.sum().item()

            lines.append(f"  Boundary zone:    {int(bnd_px):>10,}  ({100*bnd_px/max(kept_px,1):.1f}% of kept)  [within {boundary_width}px of mask edge]")
            lines.append(f"  Interior zone:    {int(int_px):>10,}  ({100*int_px/max(kept_px,1):.1f}% of kept)  [farther than {boundary_width}px from edge]")
        else:
            keep_mask = None
            boundary_zone = None
            interior_zone = None
            lines.append("Mask: none (measuring ALL pixels)")

        lines.append("")

        # --- Global kept-pixel stats ---
        lines.append("-" * 70)
        lines.append("GLOBAL STATS (all kept pixels)")
        lines.append("-" * 70)

        if keep_mask is not None:
            zone_lines, _ = self._zone_stats(abs_diff, signed_diff, keep_mask)
        else:
            all_mask = torch.ones(B, H, W, dtype=torch.bool, device=image_a.device)
            zone_lines, _ = self._zone_stats(abs_diff, signed_diff, all_mask)
        lines.extend(zone_lines)

        # Luminance
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image_a.device)
        lum_a = (image_a * weights).sum(dim=-1)
        lum_b = (image_b * weights).sum(dim=-1)
        lum_diff = lum_b - lum_a

        if keep_mask is not None:
            lum_vals = lum_diff[keep_mask]
        else:
            lum_vals = lum_diff.flatten()

        if lum_vals.numel() > 0:
            mse = (lum_vals ** 2).mean().item()
            psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))).item()
            lp90 = _diff_ops_percentile(lum_vals.abs(), 0.90)
            lp99 = _diff_ops_percentile(lum_vals.abs(), 0.99)

            lines.append(f"  Luminance (BT.709):")
            lines.append(f"    Mean shift:   {lum_vals.mean().item():+.6f}  ({lum_vals.mean().item()*255:+.2f}/255)")
            lines.append(f"    Std:          {lum_vals.std().item():.6f}  ({lum_vals.std().item()*255:.2f}/255)")
            lines.append(f"    Max |shift|:  {lum_vals.abs().max().item():.6f}  ({lum_vals.abs().max().item()*255:.2f}/255)")
            lines.append(f"    p90 |shift|:  {lp90:.6f}  ({lp90*255:.2f}/255)")
            lines.append(f"    p99 |shift|:  {lp99:.6f}  ({lp99*255:.2f}/255)")
            lines.append(f"    PSNR:         {psnr:.1f} dB  {'(excellent >40, good >30, visible <30, bad <20)' if True else ''}")

        # --- Boundary vs Interior split ---
        if boundary_zone is not None and interior_zone is not None:
            lines.append("")
            lines.append("-" * 70)
            lines.append(f"BOUNDARY ZONE (kept pixels within {boundary_width}px of mask edge)")
            lines.append("-" * 70)
            bnd_lines, bnd_count = self._zone_stats(abs_diff, signed_diff, boundary_zone)
            lines.extend(bnd_lines)
            if bnd_count > 0:
                bnd_lum = lum_diff[boundary_zone]
                bnd_mse = (bnd_lum ** 2).mean().item()
                bnd_psnr = 10 * torch.log10(torch.tensor(1.0 / max(bnd_mse, 1e-10))).item()
                lines.append(f"  Luminance PSNR: {bnd_psnr:.1f} dB")

            lines.append("")
            lines.append("-" * 70)
            lines.append(f"INTERIOR ZONE (kept pixels farther than {boundary_width}px from mask edge)")
            lines.append("-" * 70)
            int_lines, int_count = self._zone_stats(abs_diff, signed_diff, interior_zone)
            lines.extend(int_lines)
            if int_count > 0:
                int_lum = lum_diff[interior_zone]
                int_mse = (int_lum ** 2).mean().item()
                int_psnr = 10 * torch.log10(torch.tensor(1.0 / max(int_mse, 1e-10))).item()
                lines.append(f"  Luminance PSNR: {int_psnr:.1f} dB")

        # --- Blend transition zone (for continuous masks from stitcher) ---
        if blend_zone is not None and blend_zone.sum().item() > 0:
            lines.append("")
            lines.append("-" * 70)
            lines.append("BLEND TRANSITION ZONE (kept pixels with 0 < mask < threshold)")
            lines.append("-" * 70)
            blz_lines, blz_count = self._zone_stats(abs_diff, signed_diff, blend_zone)
            lines.extend(blz_lines)
            if blz_count > 0:
                blz_lum = lum_diff[blend_zone]
                blz_mse = (blz_lum ** 2).mean().item()
                blz_psnr = 10 * torch.log10(torch.tensor(1.0 / max(blz_mse, 1e-10))).item()
                lines.append(f"  Luminance PSNR: {blz_psnr:.1f} dB")

            # Also show untouched-only stats
            untouched = mask <= 0.01
            if untouched.sum().item() > 0:
                lines.append("")
                lines.append("-" * 70)
                lines.append("UNTOUCHED ZONE (mask ≤ 0.01, should be identical to source)")
                lines.append("-" * 70)
                ut_lines, ut_count = self._zone_stats(abs_diff, signed_diff, untouched)
                lines.extend(ut_lines)
                if ut_count > 0:
                    ut_lum = lum_diff[untouched]
                    ut_mse = (ut_lum ** 2).mean().item()
                    ut_psnr = 10 * torch.log10(torch.tensor(1.0 / max(ut_mse, 1e-10))).item()
                    lines.append(f"  Luminance PSNR: {ut_psnr:.1f} dB")
                    if ut_mse < 1e-8:
                        lines.append(f"  => PERFECT MATCH — untouched pixels are bit-identical")
                    elif ut_psnr > 60:
                        lines.append(f"  => NEAR-PERFECT — untouched pixels have negligible difference (rounding)")
                    elif ut_psnr > 40:
                        lines.append(f"  => VERY GOOD — untouched pixels barely changed")
                    else:
                        lines.append(f"  => WARNING — untouched pixels should be identical but show {ut_lum.abs().mean().item()*255:.1f}/255 mean diff")
                        lines.append(f"     This suggests the stitch is modifying pixels outside the blend zone!")

        # --- Per-frame stats (if batch > 1) ---
        if B > 1:
            lines.append("")
            lines.append("-" * 70)
            lines.append("PER-FRAME SUMMARY")
            lines.append("-" * 70)
            lines.append(f"{'Frame':>6} | {'Mean|diff|':>12} | {'MeanSigned':>12} | {'MaxDiff':>10} | {'PSNR':>8} | {'KeptPx':>8}")
            lines.append("-" * 70)

            for b in range(B):
                frame_abs = abs_diff[b]  # [H, W, C]
                frame_signed = signed_diff[b]
                frame_lum = lum_diff[b]

                if keep_mask is not None:
                    fm = keep_mask[b]  # [H, W]
                    if fm.sum() == 0:
                        lines.append(f"{b:>6} | {'(no kept pixels)':>50}")
                        continue
                    f_abs_vals = frame_abs[fm]
                    f_signed_vals = frame_signed[fm]
                    f_lum_vals = frame_lum[fm]
                    f_kept = fm.sum().item()
                else:
                    f_abs_vals = frame_abs.reshape(-1, C)
                    f_signed_vals = frame_signed.reshape(-1, C)
                    f_lum_vals = frame_lum.flatten()
                    f_kept = H * W

                f_mean_abs = f_abs_vals.mean().item()
                f_mean_signed = f_signed_vals.mean().item()
                f_max = f_abs_vals.max().item()
                f_mse = (f_lum_vals ** 2).mean().item()
                f_psnr = 10 * torch.log10(torch.tensor(1.0 / max(f_mse, 1e-10))).item()

                lines.append(
                    f"{b:>6} | {f_mean_abs*255:>10.2f}/255 | {f_mean_signed*255:>+10.2f}/255 | {f_max*255:>8.1f}/255 | {f_psnr:>6.1f} dB | {int(f_kept):>8,}"
                )

            # Flag outlier frames
            if B > 3:
                frame_means = []
                for b in range(B):
                    if keep_mask is not None:
                        fm = keep_mask[b]
                        if fm.sum() == 0:
                            frame_means.append(0.0)
                            continue
                        frame_means.append(abs_diff[b][fm].mean().item())
                    else:
                        frame_means.append(abs_diff[b].mean().item())

                fm_tensor = torch.tensor(frame_means)
                median_val = fm_tensor.median().item()
                if median_val > 0:
                    outliers = [(i, v) for i, v in enumerate(frame_means) if v > median_val * 2.0]
                    if outliers:
                        lines.append("")
                        lines.append(f"  OUTLIER FRAMES (>2x median diff of {median_val*255:.1f}/255):")
                        for i, v in outliers:
                            lines.append(f"    Frame {i}: mean|diff| = {v*255:.1f}/255  ({v/max(median_val,1e-10):.1f}x median)")

        # --- Diagnostic interpretation ---
        lines.append("")
        lines.append("=" * 70)
        lines.append("DIAGNOSTIC INTERPRETATION")
        lines.append("=" * 70)

        if keep_mask is not None and lum_vals.numel() > 0:
            mean_abs_all = abs_diff[keep_mask].mean().item()
            mean_signed_all = signed_diff[keep_mask].mean().item()
            std_all = signed_diff[keep_mask].std().item()
            ratio = abs(mean_signed_all) / max(std_all, 1e-10)

            if mean_abs_all * 255 < 1.0:
                lines.append("Result: EXCELLENT — kept pixels are nearly identical (<1/255 mean diff)")
            elif mean_abs_all * 255 < 5.0:
                lines.append("Result: GOOD — small differences, likely imperceptible (<5/255)")
            elif mean_abs_all * 255 < 15.0:
                lines.append("Result: MODERATE — noticeable on close inspection (5-15/255)")
            else:
                lines.append("Result: SIGNIFICANT — clearly visible differences (>15/255)")

            lines.append("")

            if ratio > 0.5:
                direction = "brighter" if mean_signed_all > 0 else "darker"
                lines.append(f"Pattern: GLOBAL TONE SHIFT — B is consistently {direction} than A")
                lines.append(f"  Mean signed / std ratio: {ratio:.2f} (>0.5 = systematic shift)")
                lines.append(f"  Diagnosis: the diffusion model is producing a different overall exposure/color")
                lines.append(f"  Fix candidates: color transfer (Reinhard/OKLab), latent mean correction")
            else:
                lines.append(f"Pattern: LOCAL/SPATIAL VARIATION — no consistent global shift")
                lines.append(f"  Mean signed / std ratio: {ratio:.2f} (<0.5 = not a uniform offset)")
                lines.append(f"  Diagnosis: differences are spatially varying (e.g., resize interpolation,")
                lines.append(f"  blend zone feathering, or content-dependent VAE distortion)")
                lines.append(f"  Fix candidates: better interpolation, narrower blend zone, Poisson blending")

            # Boundary vs interior comparison
            if boundary_zone is not None and interior_zone is not None:
                bnd_vals = abs_diff[boundary_zone]
                int_vals = abs_diff[interior_zone]
                if bnd_vals.numel() > 0 and int_vals.numel() > 0:
                    bnd_mean = bnd_vals.mean().item()
                    int_mean = int_vals.mean().item()
                    ratio_bi = bnd_mean / max(int_mean, 1e-10)
                    lines.append("")
                    lines.append(f"Boundary vs Interior ratio: {ratio_bi:.2f}x")
                    if ratio_bi > 3.0:
                        lines.append(f"  => STRONG boundary concentration — differences are at the seam, not spread evenly")
                        lines.append(f"  => This points to blend zone / feathering as the dominant source")
                    elif ratio_bi > 1.5:
                        lines.append(f"  => MODERATE boundary concentration — seam is worse than interior but interior has drift too")
                    else:
                        lines.append(f"  => UNIFORM distribution — differences spread evenly, not concentrated at boundary")
                        lines.append(f"  => This points to resize interpolation or VAE roundtrip as the source")

        lines.append("")
        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "NV_ImageDiffAnalyzer": NV_ImageDiffAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ImageDiffAnalyzer": "NV Image Diff Analyzer",
}
