"""
Shared mask spatial operations — morphology, blur, resize.

Single source of truth for mask processing functions used across:
  - NV_InpaintCrop2
  - NV_TemporalMaskStabilizer
  - NV_VaceControlVideoPrep
  - NV_MaskTrackingBBox
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
import torchvision.transforms.v2 as T


# =============================================================================
# GPU-native morphology helpers (F.max_pool2d based)
# =============================================================================

def _pad_for_pool(x, kernel_size):
    """Same-padding for pooling ops (odd kernels only). Even kernels forced odd via |1."""
    kernel_size = kernel_size | 1  # Force odd - even kernels break output shape with stride=1
    pad = kernel_size // 2
    _, _, h, w = x.shape
    mode = "reflect" if (h > pad and w > pad) else "replicate"
    return F.pad(x, (pad, pad, pad, pad), mode=mode)


def _grey_dilate_gpu(x, kernel_size):
    """Grey dilation via max_pool2d. x: [B, 1, H, W]."""
    if kernel_size <= 1:
        return x
    return F.max_pool2d(_pad_for_pool(x, kernel_size), kernel_size=kernel_size, stride=1)


def _grey_erode_gpu(x, kernel_size):
    """Grey erosion via negated max_pool2d. x: [B, 1, H, W]."""
    if kernel_size <= 1:
        return x
    return -F.max_pool2d(_pad_for_pool(-x, kernel_size), kernel_size=kernel_size, stride=1)


# =============================================================================
# Morphology (GPU when CUDA available, scipy CPU fallback)
# =============================================================================

def mask_erode_dilate(mask, amount):
    """Erode (negative) or dilate (positive) mask. GPU-accelerated when on CUDA."""
    if amount == 0:
        return mask

    size = max(1, int(abs(amount)))

    if mask.is_cuda:
        src = mask.unsqueeze(1).to(dtype=torch.float32)
        out = _grey_erode_gpu(src, size) if amount < 0 else _grey_dilate_gpu(src, size)
        return out.squeeze(1).to(dtype=mask.dtype)

    device = mask.device
    results = []
    for m in mask:
        m_np = m.cpu().numpy()
        if amount < 0:
            m_np = scipy.ndimage.grey_erosion(m_np, size=(size, size))
        else:
            m_np = scipy.ndimage.grey_dilation(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))
    return torch.stack(results, dim=0)


def mask_fill_holes(mask, size):
    """Fill holes using grey closing (dilate then erode). GPU-accelerated when on CUDA."""
    if size == 0:
        return mask
    size = max(1, int(size))

    if mask.is_cuda:
        src = mask.unsqueeze(1).to(dtype=torch.float32)
        out = _grey_erode_gpu(_grey_dilate_gpu(src, size), size)
        return out.squeeze(1).to(dtype=mask.dtype)

    device = mask.device
    results = []
    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_closing(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))
    return torch.stack(results, dim=0)


def mask_remove_noise(mask, size):
    """Remove isolated pixels using grey opening (erode then dilate). GPU-accelerated when on CUDA."""
    if size == 0:
        return mask
    size = max(1, int(size))

    if mask.is_cuda:
        src = mask.unsqueeze(1).to(dtype=torch.float32)
        out = _grey_dilate_gpu(_grey_erode_gpu(src, size), size)
        return out.squeeze(1).to(dtype=mask.dtype)

    device = mask.device
    results = []
    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_opening(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))
    return torch.stack(results, dim=0)


# =============================================================================
# Connected Components Gating (CorridorKey-style clean_matte)
# =============================================================================

def mask_connected_components_gate(mask, threshold=0.5, min_area=300, dilate_radius=15, blur_kernel=5, connectivity=8):
    """Gate a soft mask using connected component analysis — kills noise while preserving soft edges.

    Steps:
      1. Threshold mask to binary for CC analysis
      2. Remove connected components smaller than min_area
      3. Dilate surviving binary regions to create a safe zone
      4. Gaussian blur the gate for smooth falloff
      5. Multiply original soft alpha by the gate

    Args:
        mask: [B, H, W] float tensor in [0, 1]
        threshold: binarization threshold for CC analysis
        min_area: minimum component area in pixels to keep
        dilate_radius: binary dilation radius for the gate
        blur_kernel: Gaussian blur kernel size for the gate
        connectivity: 4 or 8 for OpenCV CC analysis

    Returns:
        [B, H, W] gated mask — noise killed, soft edges preserved in valid regions
    """
    squeeze = False
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
        squeeze = True

    src = mask.clamp(0.0, 1.0)
    if src.shape[0] == 0:
        return src.squeeze(0) if squeeze else src
    device = src.device
    dtype = src.dtype
    gates = []

    for frame in src:
        frame_np = (frame > float(threshold)).to(torch.uint8).cpu().numpy()
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame_np, connectivity=connectivity, ltype=cv2.CV_32S)

        keep = np.zeros_like(frame_np, dtype=np.uint8)
        for label_idx in range(1, num_labels):
            if int(stats[label_idx, cv2.CC_STAT_AREA]) >= int(min_area):
                keep[labels == label_idx] = 1

        gates.append(torch.from_numpy(keep).to(device=device, dtype=torch.float32))

    gate = torch.stack(gates, dim=0).unsqueeze(1)  # [B, 1, H, W]

    if dilate_radius > 0:
        kernel_size = 2 * int(dilate_radius) + 1
        gate = _grey_dilate_gpu(gate, kernel_size) if gate.is_cuda else gate
        if not gate.is_cuda:
            gate_np = gate.squeeze(1).cpu().numpy()
            for i in range(gate_np.shape[0]):
                gate_np[i] = scipy.ndimage.grey_dilation(gate_np[i], size=(kernel_size, kernel_size))
            gate = torch.from_numpy(gate_np).to(device=device).unsqueeze(1)

    gate = gate.clamp(0.0, 1.0)

    if blur_kernel and blur_kernel > 1:
        bk = int(blur_kernel)
        if bk % 2 == 0:
            bk += 1
        gate = T.functional.gaussian_blur(gate, bk).clamp(0.0, 1.0)

    result = src * gate.squeeze(1).to(dtype=dtype)
    return result.squeeze(0) if squeeze else result


def mask_smooth(mask, amount):
    """Smooth mask edges by binarizing then blurring."""
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    binary = mask > 0.5
    smoothed = T.functional.gaussian_blur(binary.unsqueeze(1).float(), amount).squeeze(1)
    return smoothed


def mask_blur(mask, amount):
    """Direct Gaussian blur on mask (preserves gradients). Used for blend feathering."""
    if amount == 0:
        return mask

    if amount % 2 == 0:
        amount += 1

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    blurred = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)
    return blurred


# =============================================================================
# Resize
# =============================================================================

def rescale_image(samples, width, height, algorithm='bicubic'):
    """Resize image tensor [B, H, W, C] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bicubic')

    samples = samples.movedim(-1, 1)
    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    return samples.movedim(1, -1)


def rescale_mask(samples, width, height, algorithm='bilinear'):
    """Resize mask tensor [B, H, W] using GPU."""
    algorithm_map = {
        'nearest': 'nearest',
        'bilinear': 'bilinear',
        'bicubic': 'bicubic',
        'lanczos': 'bicubic',
        'area': 'area',
    }
    mode = algorithm_map.get(algorithm.lower(), 'bilinear')

    samples = samples.unsqueeze(1)
    samples = torch.nn.functional.interpolate(
        samples,
        size=(height, width),
        mode=mode,
        align_corners=False if mode in ['bilinear', 'bicubic'] else None
    )
    return samples.squeeze(1)
