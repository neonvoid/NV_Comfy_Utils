"""
NV Image Diff Analyzer — Pixel-by-pixel comparison of two images.

Outputs an amplified difference heatmap and per-channel statistics.
Optionally takes a mask to EXCLUDE the generated/inpainted region,
measuring only the kept pixels (where source and result should match).

Use case: diagnosing VAE roundtrip degradation, diffusion color drift,
or stitch seam color mismatch by visualizing exactly where and how much
pixels change.
"""

import torch
import torch.nn.functional as F


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
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Pixels with mask value below this are considered 'kept' (measured)."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING",)
    RETURN_NAMES = ("diff_heatmap", "masked_diff", "stats_text",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = "Pixel-by-pixel comparison of two images. Outputs amplified diff heatmap and per-channel statistics."

    def execute(self, image_a, image_b, amplify, mode, mask=None, mask_threshold=0.5):
        # Ensure same shape — resize B to match A if needed
        if image_a.shape != image_b.shape:
            # image: [B, H, W, C]
            b_resized = image_b.permute(0, 3, 1, 2)  # [B, C, H, W]
            b_resized = F.interpolate(b_resized, size=(image_a.shape[1], image_a.shape[2]), mode="bilinear", align_corners=False)
            image_b = b_resized.permute(0, 2, 3, 1)  # [B, H, W, C]

        diff = image_b.float() - image_a.float()  # signed diff
        abs_diff = diff.abs()

        # Build visualization based on mode
        if mode == "absolute_diff":
            vis = (abs_diff * amplify).clamp(0, 1)

        elif mode == "signed_rgb":
            # Red channel = where B is brighter, Blue = where A is brighter
            pos = diff.clamp(min=0) * amplify  # B brighter
            neg = (-diff).clamp(min=0) * amplify  # A brighter
            lum_diff = diff.mean(dim=-1, keepdim=True)
            red = lum_diff.clamp(min=0) * amplify
            green = torch.zeros_like(red)
            blue = (-lum_diff).clamp(min=0) * amplify
            vis = torch.cat([red, green, blue], dim=-1).clamp(0, 1)

        elif mode == "luminance_diff":
            # Perceptual luminance weights (BT.709)
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image_a.device)
            lum_a = (image_a * weights).sum(dim=-1, keepdim=True)
            lum_b = (image_b * weights).sum(dim=-1, keepdim=True)
            lum_diff = (lum_b - lum_a).abs() * amplify
            vis = lum_diff.expand(-1, -1, -1, 3).clamp(0, 1)

        elif mode == "heatmap":
            # Magnitude → cool (blue) to hot (red) colormap
            magnitude = abs_diff.mean(dim=-1)  # [B, H, W]
            magnitude = (magnitude * amplify).clamp(0, 1)
            vis = self._apply_heatmap(magnitude)  # [B, H, W, 3]

        # Build masked version — zero out the generated region so only kept pixels show
        if mask is not None:
            # mask: [B, H, W] or [1, H, W], resize to match
            m = mask.float()
            if m.shape[1:] != image_a.shape[1:3]:
                m = F.interpolate(m.unsqueeze(1), size=(image_a.shape[1], image_a.shape[2]), mode="bilinear", align_corners=False).squeeze(1)
            if m.shape[0] == 1 and vis.shape[0] > 1:
                m = m.expand(vis.shape[0], -1, -1)
            # mask=1 is generated (exclude), mask=0 is kept (show)
            keep = (m < mask_threshold).float().unsqueeze(-1)  # [B, H, W, 1]
            masked_vis = vis * keep
        else:
            keep = None
            masked_vis = vis

        # Compute statistics
        stats = self._compute_stats(image_a, image_b, abs_diff, diff, mask, mask_threshold)

        return (vis, masked_vis, stats,)

    def _apply_heatmap(self, magnitude):
        """Convert [B, H, W] magnitude (0-1) to [B, H, W, 3] RGB heatmap."""
        # Simple 5-stop colormap: black → blue → cyan → yellow → red → white
        B, H, W = magnitude.shape
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

    def _compute_stats(self, image_a, image_b, abs_diff, signed_diff, mask, mask_threshold):
        """Compute per-channel and summary statistics."""
        lines = []

        # Determine which pixels to measure
        if mask is not None:
            m = mask.float()
            if m.shape[1:] != image_a.shape[1:3]:
                m = F.interpolate(m.unsqueeze(1), size=(image_a.shape[1], image_a.shape[2]), mode="bilinear", align_corners=False).squeeze(1)
            keep_mask = (m < mask_threshold)  # [B, H, W] bool
            total_px = keep_mask.sum().item()
            lines.append(f"Measuring KEPT pixels only (mask < {mask_threshold})")
            lines.append(f"Kept pixels: {int(total_px):,} / {keep_mask.numel():,} ({100*total_px/max(keep_mask.numel(),1):.1f}%)")
        else:
            keep_mask = None
            total_px = image_a[..., 0].numel()
            lines.append("Measuring ALL pixels (no mask provided)")
            lines.append(f"Total pixels: {int(total_px):,}")

        lines.append("")

        channel_names = ["Red", "Green", "Blue"]
        for c, name in enumerate(channel_names):
            ch_abs = abs_diff[..., c]  # [B, H, W]
            ch_signed = signed_diff[..., c]

            if keep_mask is not None:
                vals_abs = ch_abs[keep_mask]
                vals_signed = ch_signed[keep_mask]
            else:
                vals_abs = ch_abs.flatten()
                vals_signed = ch_signed.flatten()

            if vals_abs.numel() == 0:
                lines.append(f"{name}: no pixels to measure")
                continue

            mean_abs = vals_abs.mean().item()
            max_abs = vals_abs.max().item()
            mean_signed = vals_signed.mean().item()
            std_signed = vals_signed.std().item()

            # Convert to 0-255 scale for intuition
            lines.append(f"{name}:")
            lines.append(f"  Mean |diff|:   {mean_abs:.6f}  ({mean_abs*255:.2f}/255)")
            lines.append(f"  Max |diff|:    {max_abs:.6f}  ({max_abs*255:.2f}/255)")
            lines.append(f"  Mean signed:   {mean_signed:+.6f}  ({mean_signed*255:+.2f}/255)  {'(B brighter)' if mean_signed > 0 else '(A brighter)'}")
            lines.append(f"  Std signed:    {std_signed:.6f}  ({std_signed*255:.2f}/255)")
            lines.append("")

        # Overall luminance stats
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=image_a.device)
        lum_a = (image_a * weights).sum(dim=-1)
        lum_b = (image_b * weights).sum(dim=-1)
        lum_diff = lum_b - lum_a

        if keep_mask is not None:
            lum_vals = lum_diff[keep_mask]
        else:
            lum_vals = lum_diff.flatten()

        if lum_vals.numel() > 0:
            lines.append("Luminance (BT.709):")
            lines.append(f"  Mean shift:    {lum_vals.mean().item():+.6f}  ({lum_vals.mean().item()*255:+.2f}/255)")
            lines.append(f"  Std:           {lum_vals.std().item():.6f}  ({lum_vals.std().item()*255:.2f}/255)")
            lines.append(f"  Max |shift|:   {lum_vals.abs().max().item():.6f}  ({lum_vals.abs().max().item()*255:.2f}/255)")

            # PSNR
            mse = (lum_vals ** 2).mean().item()
            if mse > 0:
                psnr = 10 * torch.log10(torch.tensor(1.0 / mse)).item()
                lines.append(f"  PSNR:          {psnr:.1f} dB")
            else:
                lines.append(f"  PSNR:          inf (identical)")

        return "\n".join(lines)


NODE_CLASS_MAPPINGS = {
    "NV_ImageDiffAnalyzer": NV_ImageDiffAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ImageDiffAnalyzer": "NV Image Diff Analyzer",
}
