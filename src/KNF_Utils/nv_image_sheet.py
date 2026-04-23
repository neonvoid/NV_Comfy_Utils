"""NV Image Composite Sheet — combine 1-4 IMAGEs into a single character-sheet layout.

Primary use case: bypass Seedance 2.0's real-person ref-image gate by
presenting multiple angles of a subject as a single "character reference sheet"
image (multi-view illustration aesthetic rather than a privacy-triggering
portrait). Anecdotal — not guaranteed to pass the gate, but widely reported to
work on photoreal face content that fails as a solo portrait.

Output is a single IMAGE [1, H, W, 3] that plugs into anywhere an IMAGE is
expected — wire it into NV Seedance Prep's `reference_images` to use as a
single seedance ref (counted as 1 image, not N).

Layouts:
  - `auto`:       picks based on image count (1→passthrough, 2→horizontal, 3-4→grid_2x2)
  - `horizontal`: single row
  - `vertical`:   single column  (note: aspect may fall outside seedance's 0.4-2.5 range)
  - `grid_2x2`:   2×2 grid, missing slots filled with bg color

Each input is letterbox-padded to `cell_size × cell_size` to preserve aspect
without cropping. Gap between cells is `padding_px` in `bg_color`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


_BG_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "white": (1.0, 1.0, 1.0),
    "gray": (0.5, 0.5, 0.5),
    "dark_gray": (0.15, 0.15, 0.15),
    "light_gray": (0.85, 0.85, 0.85),
}


def _letterbox(img: torch.Tensor, cell: int, bg_rgb: tuple[float, float, float]) -> torch.Tensor:
    """Resize an IMAGE [1,H,W,3] to fit in cell×cell preserving aspect; pad with bg_rgb."""
    _, h, w, _ = img.shape
    scale = min(cell / h, cell / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    # Resize via F.interpolate which wants [B,C,H,W]
    x = img.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1)

    # Pad to cell×cell with bg color
    canvas = torch.full((1, cell, cell, 3), 0.0, dtype=img.dtype, device=img.device)
    for c in range(3):
        canvas[..., c] = bg_rgb[c]
    y_off = (cell - new_h) // 2
    x_off = (cell - new_w) // 2
    canvas[:, y_off:y_off + new_h, x_off:x_off + new_w, :] = x
    return canvas


def _make_gap(h: int, w: int, bg_rgb, dtype, device) -> torch.Tensor:
    gap = torch.zeros(1, h, w, 3, dtype=dtype, device=device)
    for c in range(3):
        gap[..., c] = bg_rgb[c]
    return gap


class NV_ImageCompositeSheet:
    """Combine up to 4 IMAGE tensors into a single character-sheet composite."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image (required)."}),
                "cell_size": ("INT", {
                    "default": 768, "min": 128, "max": 3000, "step": 32,
                    "tooltip": (
                        "Per-cell size in pixels. Each input is letterboxed into a cell of this size. "
                        "For Seedance compatibility, a 2×2 grid at cell_size=768 produces a ~1536×1536 sheet "
                        "(well within [300,6000] range + 1:1 aspect is gate-friendly)."
                    ),
                }),
                "padding_px": ("INT", {
                    "default": 8, "min": 0, "max": 128, "step": 1,
                    "tooltip": "Gap between cells, filled with bg_color.",
                }),
                "layout": (["auto", "horizontal", "vertical", "grid_2x2"], {
                    "default": "auto",
                    "tooltip": (
                        "auto: 1→passthrough, 2→horizontal, 3-4→grid_2x2. "
                        "vertical can push aspect below Seedance's 0.4 floor — prefer grid_2x2 for 3+ imgs."
                    ),
                }),
                "bg_color": (list(_BG_COLORS.keys()), {
                    "default": "dark_gray",
                    "tooltip": "Letterbox + gap fill color. Neutral grays look most 'reference sheet'-like.",
                }),
            },
            "optional": {
                "image_b": ("IMAGE", {"tooltip": "Second image."}),
                "image_c": ("IMAGE", {"tooltip": "Third image."}),
                "image_d": ("IMAGE", {"tooltip": "Fourth image."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sheet",)
    FUNCTION = "composite"
    CATEGORY = "NV_Utils/image"
    DESCRIPTION = (
        "Combine 1-4 IMAGEs into a single character-sheet composite. Each input "
        "is letterboxed to cell_size, then arranged in a row / column / 2×2 grid. "
        "Designed to bypass Seedance 2.0's real-person ref-image gate by presenting "
        "multiple angles as a single 'reference art' image."
    )

    def composite(
        self,
        image_a: torch.Tensor,
        cell_size: int,
        padding_px: int,
        layout: str,
        bg_color: str,
        image_b: torch.Tensor | None = None,
        image_c: torch.Tensor | None = None,
        image_d: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor]:
        bg_rgb = _BG_COLORS[bg_color]

        # Collect non-None inputs, promote any 3D [H,W,3] to 4D [1,H,W,3]
        raw = [("a", image_a), ("b", image_b), ("c", image_c), ("d", image_d)]
        imgs: list[torch.Tensor] = []
        for _, img in raw:
            if img is None:
                continue
            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.ndim != 4 or img.shape[-1] < 3:
                raise ValueError(
                    f"[NV_ImageCompositeSheet] expected IMAGE [B,H,W,3 or 4], got {tuple(img.shape)}"
                )
            # Collapse [B,H,W,C>3] alpha channels to RGB
            img = img[..., :3]
            # If B>1, take first frame only — sheet is a static composite.
            if img.shape[0] > 1:
                img = img[:1]
            imgs.append(img)

        n = len(imgs)
        if n == 0:
            raise ValueError("[NV_ImageCompositeSheet] at least one image is required.")

        # Align device/dtype to first image
        ref = imgs[0]
        imgs = [i.to(device=ref.device, dtype=ref.dtype) for i in imgs]

        # Resolve auto layout
        if layout == "auto":
            if n == 1:
                layout = "horizontal"  # passthrough (single letterboxed cell)
            elif n == 2:
                layout = "horizontal"
            else:
                layout = "grid_2x2"

        # Letterbox every input into a uniform cell
        cells = [_letterbox(i, cell_size, bg_rgb) for i in imgs]

        # Fill missing grid slots with blank cells (for grid_2x2 with n<4)
        if layout == "grid_2x2":
            while len(cells) < 4:
                cells.append(_make_gap(cell_size, cell_size, bg_rgb, ref.dtype, ref.device))

        dev, dt = ref.device, ref.dtype

        if layout == "horizontal":
            # Row: [cell, gap, cell, gap, cell, ...]
            parts: list[torch.Tensor] = []
            for idx, c in enumerate(cells):
                if idx > 0 and padding_px > 0:
                    parts.append(_make_gap(cell_size, padding_px, bg_rgb, dt, dev))
                parts.append(c)
            out = torch.cat(parts, dim=2)

        elif layout == "vertical":
            parts = []
            for idx, c in enumerate(cells):
                if idx > 0 and padding_px > 0:
                    parts.append(_make_gap(padding_px, cell_size, bg_rgb, dt, dev))
                parts.append(c)
            out = torch.cat(parts, dim=1)

        elif layout == "grid_2x2":
            # Build top row (a, b) and bottom row (c, d) with horizontal gaps,
            # then stack the two rows with a vertical gap.
            top_parts = [cells[0]]
            if padding_px > 0:
                top_parts.append(_make_gap(cell_size, padding_px, bg_rgb, dt, dev))
            top_parts.append(cells[1])
            top_row = torch.cat(top_parts, dim=2)

            bot_parts = [cells[2]]
            if padding_px > 0:
                bot_parts.append(_make_gap(cell_size, padding_px, bg_rgb, dt, dev))
            bot_parts.append(cells[3])
            bot_row = torch.cat(bot_parts, dim=2)

            if padding_px > 0:
                v_gap = _make_gap(padding_px, top_row.shape[2], bg_rgb, dt, dev)
                out = torch.cat([top_row, v_gap, bot_row], dim=1)
            else:
                out = torch.cat([top_row, bot_row], dim=1)
        else:
            raise ValueError(f"[NV_ImageCompositeSheet] unknown layout: {layout!r}")

        # Clamp + coerce to float32 (ComfyUI IMAGE convention)
        out = out.clamp(0.0, 1.0)
        if out.dtype != torch.float32:
            out = out.to(dtype=torch.float32)

        h, w = out.shape[1], out.shape[2]
        print(f"[NV_ImageCompositeSheet] {n} input(s) → {layout} → {w}×{h} sheet")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "NV_ImageCompositeSheet": NV_ImageCompositeSheet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_ImageCompositeSheet": "NV Image Composite Sheet",
}
