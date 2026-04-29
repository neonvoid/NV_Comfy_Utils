"""Seam-continuity metric computation — extracted from seam_analyzer.py.

Operates on a contiguous IMAGE batch [B, H, W, 3] and a list of seam frame
indices (0-indexed; each value is the LAST frame of the chunk before the
seam, so the seam pair is (i, i+1)).

Returns dict of FLOAT scalars: PSNR/SSIM/flow/Laplacian aggregated over all
seams in the render. Worst-case (min/max/p95) emphasized over means since
one bad seam can invalidate a whole render.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_gray_uint8(frame_hwc):
    gray = 0.299 * frame_hwc[..., 0] + 0.587 * frame_hwc[..., 1] + 0.114 * frame_hwc[..., 2]
    return (gray * 255).clamp(0, 255).to(torch.uint8)


def _psnr(a, b):
    mse = (a.float() - b.float()).pow(2).mean()
    if mse < 1e-10:
        return 100.0
    return float(10 * torch.log10(1.0 / mse))


def _ssim_approx(a, b):
    a_gray = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).float()
    b_gray = (0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]).float()
    mu_a, mu_b = a_gray.mean(), b_gray.mean()
    sa, sb = a_gray.std(), b_gray.std()
    sab = ((a_gray - mu_a) * (b_gray - mu_b)).mean()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sa ** 2 + sb ** 2 + C2)
    return float(num / den)


def _flow_magnitude(prev_gray_uint8, curr_gray_uint8):
    """Mean optical-flow magnitude between two grayscale uint8 frames.
    Returns -1.0 if cv2 unavailable (so caller can drop from aggregate).
    """
    try:
        import cv2
        prev_np = prev_gray_uint8.cpu().numpy()
        curr_np = curr_gray_uint8.cpu().numpy()
        flow = cv2.calcOpticalFlowFarneback(
            prev_np, curr_np, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(mag.mean())
    except ImportError:
        return -1.0


def _laplacian_variance(frame_hwc):
    gray = (0.299 * frame_hwc[..., 0] + 0.587 * frame_hwc[..., 1] + 0.114 * frame_hwc[..., 2])
    gray = (gray * 255.0).float().unsqueeze(0).unsqueeze(0)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32, device=gray.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return float(lap.var())


def parse_seam_indices(chunk_plan_json):
    """Extract seam frame indices from a chunk plan JSON string.

    Expected format (NV_AspectChunkPlanner / similar):
        {"chunks": [{"start": 0, "end": 80}, {"start": 65, "end": 160}, ...]}
    Seam index = end-1 of each chunk except the last (so seam pair = (end-1, end)
    of consecutive chunks aligned to the output frame numbering).

    Returns list[int] or [] if unparseable.
    """
    if not chunk_plan_json or not chunk_plan_json.strip():
        return []
    try:
        import json
        plan = json.loads(chunk_plan_json)
        chunks = plan.get("chunks", []) if isinstance(plan, dict) else []
        if len(chunks) < 2:
            return []
        seams = []
        for c in chunks[:-1]:
            end = c.get("end")
            if isinstance(end, int) and end > 0:
                seams.append(end - 1)
        return seams
    except Exception:
        return []


def compute_seam_metrics(images, seam_indices=None, chunk_plan_json=None):
    """Aggregate seam-continuity scalars over all seam pairs in the render.

    Args:
        images: IMAGE [B, H, W, 3] for the full output.
        seam_indices: list[int] (frame index of last frame before each seam),
            or None to parse from chunk_plan_json.
        chunk_plan_json: STRING from NV_AspectChunkPlanner (used if seam_indices None).

    Returns dict of scalars; empty dict if no seams.
    """
    out = {}
    try:
        if seam_indices is None:
            seam_indices = parse_seam_indices(chunk_plan_json)
        if not seam_indices:
            return {"seam_count": 0}

        if not torch.is_tensor(images) or images.dim() != 4:
            return {"compute_error": "expected IMAGE [B,H,W,3]"}

        B = images.shape[0]
        # Filter seam indices to valid range (need both frame i and i+1)
        seam_indices = [s for s in seam_indices if 0 <= s < B - 1]
        if not seam_indices:
            return {"seam_count": 0}

        psnrs, ssims, flows, lap_deltas = [], [], [], []
        for s in seam_indices:
            a = images[s]
            b = images[s + 1]
            psnrs.append(_psnr(a, b))
            ssims.append(_ssim_approx(a, b))
            f = _flow_magnitude(_to_gray_uint8(a), _to_gray_uint8(b))
            if f >= 0:
                flows.append(f)
            lap_deltas.append(abs(_laplacian_variance(b) - _laplacian_variance(a)))

        out["seam_count"] = len(seam_indices)
        out["psnr_min"]  = round(min(psnrs), 3)
        out["psnr_mean"] = round(sum(psnrs) / len(psnrs), 3)
        out["ssim_min"]  = round(min(ssims), 4)
        out["ssim_mean"] = round(sum(ssims) / len(ssims), 4)
        if flows:
            out["flow_mag_max"]  = round(max(flows), 3)
            out["flow_mag_mean"] = round(sum(flows) / len(flows), 3)
            # p95 only meaningful if multiple seams
            if len(flows) > 1:
                sorted_flows = sorted(flows)
                k = int(round(0.95 * (len(sorted_flows) - 1)))
                out["flow_mag_p95"] = round(sorted_flows[k], 3)
        if lap_deltas:
            out["laplacian_var_delta_max"]  = round(max(lap_deltas), 3)
            out["laplacian_var_delta_mean"] = round(sum(lap_deltas) / len(lap_deltas), 3)
    except Exception as e:
        out["compute_error"] = f"{type(e).__name__}: {e}"
    return out
