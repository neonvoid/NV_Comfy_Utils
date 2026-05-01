"""
NV Multi-Band Blend Stitch — Laplacian pyramid blending for seam-free stitching.

Decomposes stitched and original images into frequency bands (Laplacian pyramid),
then blends each band with a progressively wider mask. Low frequencies blend broadly
(catches color/tone shift), high frequencies blend narrowly (preserves edge detail).

Standard technique from panorama stitching (Burt & Adelson 1983), adapted for
video inpaint stitch seam repair. Operates on IMAGE tensors after InpaintStitch2.
"""

import torch
import torch.nn.functional as F


# Default batch chunk size for memory-bounded pyramid operations.
# Caps the peak transient allocation inside F.pad / F.interpolate / subtraction
# during pyramid build + collapse. Matches the existing pattern in
# texture_harmonize.py:_chunked_gaussian_blur_sep.
#
# Why chunking: at full batch (e.g. B=277, 1080p, fp32), F.pad must allocate a
# single ~7 GB contiguous host buffer. On Windows late in a long-lived ComfyUI
# session, virtual address space fragmentation from upstream NV_InpaintStitch2
# torch.stack calls (each 6.9 GB contiguous) makes that allocation fail with an
# access violation. Chunk=16 caps the transient at ~400 MB which always fits.
_PYRAMID_CHUNK_SIZE = 16


def _validate_chunk_kernel(chunk_size, kernel):
    """Defensive validation for chunked pyramid helpers.

    chunk_size <= 0 would skip the chunked loop entirely and return uninitialized
    garbage from torch.empty(). Even-sized kernels would shape-mismatch the
    pre-allocated output (conv with pad=K//2 adds an extra row/col, ::2 then
    rounds differently than the (H+1)//2 pre-alloc formula).
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError(f"chunk_size must be a positive int (got {chunk_size!r})")
    K = kernel.shape[-1]
    if K % 2 == 0:
        raise ValueError(f"kernel size must be odd (got {K}); chunked path requires odd K for shape consistency")


def _batch_slices(batch_size, chunk_size):
    """Yield (start, end) slice pairs covering [0, batch_size) in chunks of chunk_size."""
    for s in range(0, batch_size, chunk_size):
        yield s, min(s + chunk_size, batch_size)


def _gaussian_downsample(img, kernel, chunk_size=_PYRAMID_CHUNK_SIZE):
    """Blur then 2x downsample. img: [B, C, H, W], kernel: [1, 1, K, K].

    Memory-bounded: chunks the F.pad + conv over the batch dimension when B
    exceeds chunk_size. Reflect padding is along H/W only, so chunked output
    is bit-identical to the unchunked path (frames are independent).
    """
    _validate_chunk_kernel(chunk_size, kernel)
    B, C, H, W = img.shape
    k = kernel.expand(C, -1, -1, -1)
    pad = kernel.shape[-1] // 2

    if B <= chunk_size:
        padded = F.pad(img, (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(padded, k, groups=C)
        return blurred[:, :, ::2, ::2]

    out_h = (H + 1) // 2
    out_w = (W + 1) // 2
    out = torch.empty(B, C, out_h, out_w, device=img.device, dtype=img.dtype)
    for s, e in _batch_slices(B, chunk_size):
        padded = F.pad(img[s:e], (pad, pad, pad, pad), mode='reflect')
        blurred = F.conv2d(padded, k, groups=C)
        out[s:e].copy_(blurred[:, :, ::2, ::2])
        del padded, blurred
    return out


def _gaussian_upsample(img, kernel, target_h, target_w, chunk_size=_PYRAMID_CHUNK_SIZE):
    """2x upsample then blur. img: [B, C, H, W], kernel: [1, 1, K, K].

    Memory-bounded: chunks F.interpolate + F.pad + conv over the batch dimension.
    """
    _validate_chunk_kernel(chunk_size, kernel)
    B, C = img.shape[:2]
    k = kernel.expand(C, -1, -1, -1)
    pad = kernel.shape[-1] // 2

    if B <= chunk_size:
        up = F.interpolate(img, size=(target_h, target_w), mode='bilinear', align_corners=False)
        padded = F.pad(up, (pad, pad, pad, pad), mode='reflect')
        return F.conv2d(padded, k, groups=C)

    out = torch.empty(B, C, target_h, target_w, device=img.device, dtype=img.dtype)
    for s, e in _batch_slices(B, chunk_size):
        up = F.interpolate(img[s:e], size=(target_h, target_w), mode='bilinear', align_corners=False)
        padded = F.pad(up, (pad, pad, pad, pad), mode='reflect')
        out[s:e].copy_(F.conv2d(padded, k, groups=C))
        del up, padded
    return out


def _make_gaussian_kernel_2d(size=5):
    """Create a 2D Gaussian kernel for pyramid operations."""
    sigma = size / 6.0
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]


def build_gaussian_pyramid(img, num_levels, kernel, chunk_size=_PYRAMID_CHUNK_SIZE):
    """Build Gaussian pyramid. Returns list of [B, C, H, W] tensors."""
    pyramid = [img]
    current = img
    for _ in range(num_levels - 1):
        current = _gaussian_downsample(current, kernel, chunk_size=chunk_size)
        pyramid.append(current)
    return pyramid


def build_laplacian_pyramid(img, num_levels, kernel, chunk_size=_PYRAMID_CHUNK_SIZE):
    """Build Laplacian pyramid. Returns list of [B, C, H, W] difference images + base.

    Memory-bounded: when B > chunk_size, level Laplacians are built into
    pre-allocated buffers via chunked upsample + torch.sub(out=), avoiding
    full-batch transient subtraction temporaries.
    """
    gauss = build_gaussian_pyramid(img, num_levels, kernel, chunk_size=chunk_size)
    laplacian = []

    for i in range(len(gauss) - 1):
        cur = gauss[i]
        nxt = gauss[i + 1]
        B, C, H, W = cur.shape

        if B <= chunk_size:
            expanded = _gaussian_upsample(nxt, kernel, H, W, chunk_size=chunk_size)
            laplacian.append(cur - expanded)
            continue

        # Pre-allocate level Laplacian; fill chunk-by-chunk via in-place subtract
        lap = torch.empty_like(cur)
        for s, e in _batch_slices(B, chunk_size):
            expanded = _gaussian_upsample(nxt[s:e], kernel, H, W, chunk_size=chunk_size)
            torch.sub(cur[s:e], expanded, out=lap[s:e])
            del expanded
        laplacian.append(lap)

    laplacian.append(gauss[-1])  # Base level (lowest frequency)
    return laplacian


def collapse_laplacian_pyramid(pyramid, kernel, chunk_size=_PYRAMID_CHUNK_SIZE):
    """Reconstruct image from Laplacian pyramid.

    Memory-bounded: when B > chunk_size, each level's reconstruction is built
    into a pre-allocated buffer via chunked upsample + torch.add(out=).
    """
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        target = pyramid[i]
        B, C, H, W = target.shape

        if B <= chunk_size:
            img = _gaussian_upsample(img, kernel, H, W, chunk_size=chunk_size) + target
            continue

        out = torch.empty_like(target)
        for s, e in _batch_slices(B, chunk_size):
            expanded = _gaussian_upsample(img[s:e], kernel, H, W, chunk_size=chunk_size)
            torch.add(expanded, target[s:e], out=out[s:e])
            del expanded
        img = out

    return img


def build_mask_pyramid(mask, num_levels, kernel, chunk_size=_PYRAMID_CHUNK_SIZE):
    """Build Gaussian pyramid of the blend mask (progressively blurred at each level)."""
    pyramid = [mask]
    current = mask
    for _ in range(num_levels - 1):
        current = _gaussian_downsample(current, kernel, chunk_size=chunk_size)
        pyramid.append(current)
    return pyramid


def multiband_blend(stitched, original, mask, num_levels=5, kernel_size=5,
                    chunk_size=_PYRAMID_CHUNK_SIZE):
    """Perform multi-band blending using Laplacian pyramids.

    Args:
        stitched: [B, C, H, W] — the stitched image (inpainted composite).
        original: [B, C, H, W] — the original unmodified image.
        mask: [B, 1, H, W] — blend mask (1.0 = use stitched, 0.0 = use original).
        num_levels: Number of pyramid levels.
        kernel_size: Gaussian kernel size for pyramid operations.
        chunk_size: Batch chunk size for memory-bounded pyramid ops (default 16).

    Returns:
        [B, C, H, W] — blended result.
    """
    device = stitched.device
    kernel = _make_gaussian_kernel_2d(kernel_size).to(device)

    # Clamp pyramid levels to avoid too-small images
    min_dim = min(stitched.shape[2], stitched.shape[3])
    max_levels = max(1, int(torch.tensor(float(min_dim)).log2().item()) - 2)
    num_levels = min(num_levels, max_levels)

    # Build Laplacian pyramids for both images
    lap_stitched = build_laplacian_pyramid(stitched, num_levels, kernel, chunk_size=chunk_size)
    lap_original = build_laplacian_pyramid(original, num_levels, kernel, chunk_size=chunk_size)

    # Build Gaussian pyramid for mask
    mask_pyr = build_mask_pyramid(mask, num_levels, kernel, chunk_size=chunk_size)

    # Blend each level
    blended_pyr = []
    for i in range(num_levels):
        m = mask_pyr[i]
        blended_pyr.append(m * lap_stitched[i] + (1.0 - m) * lap_original[i])

    # Collapse
    return collapse_laplacian_pyramid(blended_pyr, kernel, chunk_size=chunk_size)


# Standalone NV_MultiBandBlendStitch node deleted 2026-05-01 (see node_notes/cleanup/
# 2026-05-01_post_process_node_cleanup.md). Helper functions above (multiband_blend,
# build_laplacian_pyramid, etc.) are still imported by inpaint_stitch.py, crop_color_fix.py,
# and texture_harmonize.py.
