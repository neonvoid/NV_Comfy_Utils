"""
Shared mask spatial operations — morphology, blur, resize.

Single source of truth for mask processing functions used across:
  - NV_InpaintCrop2
  - NV_TemporalMaskStabilizer
  - NV_VaceControlVideoPrep
  - NV_MaskTrackingBBox
"""

import torch
import numpy as np
import scipy.ndimage
import torchvision.transforms.v2 as T


# =============================================================================
# Morphology (scipy grey operations — preserve gradients)
# =============================================================================

def mask_erode_dilate(mask, amount):
    """Erode (negative) or dilate (positive) mask using scipy grey morphology."""
    if amount == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        if amount < 0:
            m_np = scipy.ndimage.grey_erosion(m_np, size=(-amount, -amount))
        else:
            m_np = scipy.ndimage.grey_dilation(m_np, size=(amount, amount))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_fill_holes(mask, size):
    """Fill holes using grey closing (dilate then erode)."""
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_closing(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


def mask_remove_noise(mask, size):
    """Remove isolated pixels using grey opening (erode then dilate)."""
    if size == 0:
        return mask

    device = mask.device
    results = []

    for m in mask:
        m_np = m.cpu().numpy()
        m_np = scipy.ndimage.grey_opening(m_np, size=(size, size))
        results.append(torch.from_numpy(m_np).to(device))

    return torch.stack(results, dim=0)


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
