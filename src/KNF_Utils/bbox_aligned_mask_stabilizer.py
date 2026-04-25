"""
NV BboxAligned Mask Stabilizer — Track B of the D-098 ownership-principle fix.

Stabilizes a per-frame SAM3 silhouette mask by motion-compensating consecutive
frames using the temporally stable bbox center trajectory from
NV_OptimizeCropTrajectory, then fusing aligned neighbors via a streaming
weighted-mean accumulator (or optional median).

Architecture (after 3-round multi-AI debate 2026-04-24, rotation upgrade
2026-04-25):

  motion model:    bbox-center translation by default (general-purpose,
                   works for any mask type). Optional similarity-transform
                   upgrade via rotation_mode="pca" — opt-in for shots where
                   the subject genuinely rotates frame-to-frame (face nods,
                   body twists, rolling objects).
                   2026-04-25 multi-AI debate (round 2) flagged that
                   translation-only alignment fails on head-nod failure
                   modes — bbox center barely moves but silhouette rotates
                   5–15° about the neck. Translation-aligned weighted-mean
                   over rotated silhouettes produces visible boundary smear
                   that no consensus mode can recover. Rotation is
                   PCA-derived from the silhouette mass (eigenvector of
                   covariance), confidence-gated by eccentricity, and
                   per-pair gracefully degraded above max_rotation_deg.
                   Default is "off" so the node remains a safe drop-in
                   master mask for general workflows; face/rotation-sensitive
                   workflows opt in to "pca" (and optionally lower
                   head_region_frac to focus PCA on the rotating subregion).
                   - scale alignment is still rejected: vacuous under
                     NV_OptimizeCropTrajectory's default size_mode=lock_largest
                     (per-frame extents constant), and harmful under unlocked
                     sizing because bbox-extent fluctuation is annotation
                     noise, not camera motion
                   - flow rejected: aperture problem on uniform mask interiors

  consensus:       distance-weighted mean (default), median (opt-in)
                   - weighted-mean encodes uncertainty as soft transition,
                     which all downstream consumers in this pipeline want
                     (NV_FaceHarmonizePyramid, NV_FrameTemporalStabilize,
                     InpaintCrop2, InpaintStitch2 — none binarize the master
                     mask before further processing; VACE dilates 80-100px
                     before VAE encoding so any soft band is killed upstream)
                   - median preserves morphological boundaries; offered as
                     escape hatch if downstream binarization ever appears in
                     the pipeline

  window:          fixed ±N (default 2 = 5-frame consensus) with dynamic
                   valid-count shrinking — boundary frames and trajectory
                   gaps just have fewer survivors

  neighbor reject: center-jump magnitude only (no IoU magic numbers).
                   Skips neighbors whose bbox-center delta exceeds
                   `max_center_jump_frac × min(H, W)`.

  failure modes:   - bbox missing for ≤3 frames → linear-interpolate centers
                   - longer bbox gaps → raw-mask passthrough on those frames
                   - subject enter/exit → window naturally shrinks via
                     valid-count, no zero-padding pollution

  output:          soft mask (default); binary as opt-in. Aux quality scalar
                   tensor opt-in for production debugging.

  performance:     streaming weighted-mean accumulator — O(H×W) memory, no
                   T×K×H×W stack materialization. Median path uses small
                   per-frame stack (≤5 × H × W ≤ ~50 MB at 1080p).

Key references in this repo:
  - bbox_ops.extract_bboxes — extract per-frame bbox extents from a MASK
  - bbox_ops.forward_backward_fill — pattern for gap-filling per-frame data
  - low_freq_recompose / face_harmonize_pyramid — sibling post-decode nodes
"""

import math

import torch
import torch.nn.functional as F

from .bbox_ops import extract_bboxes


LOG_PREFIX = "[NV_BboxAlignedMaskStabilizer]"

# Fixed weighted-mean weights table for offsets [0, 1, 2, 3, 4] from current.
# Mirrored on negative side via abs(dt). Values >= 5 unused in practice (window
# capped at 4 by the schema).
_WEIGHTS_TABLE = [0.50, 0.30, 0.20, 0.12, 0.07]

# Below this eccentricity, PCA orientation is too noisy to use (near-circular
# silhouette — back-of-head only, etc.). Falls back to translation-only for
# affected neighbor pairs. 0.35 ≈ axis-ratio 2:1.4 — production face/head crops
# typically live in [0.4, 0.7] when the orientation signal is informative.
# Raised from 0.15 → 0.35 per multi-AI review (2026-04-25): 0.15 was nearly a
# perfect circle and allowed noisy angles to fire on near-circular crops.
_PCA_ECC_MIN = 0.35


# =============================================================================
# Center extraction + interpolation
# =============================================================================

def _extract_centers(bbox_mask):
    """Per-frame (cx, cy) from a bbox MASK. Returns list[Optional[(cx, cy)]]."""
    x1s, y1s, x2s, y2s, present = extract_bboxes(bbox_mask, info_lines=None)
    centers = []
    for x1, y1, x2, y2, p in zip(x1s, y1s, x2s, y2s, present):
        if not p:
            centers.append(None)
        else:
            centers.append((0.5 * (x1 + x2), 0.5 * (y1 + y2)))
    return centers


def _interpolate_short_gaps(centers, max_gap):
    """Linearly interpolate (cx, cy) across gaps of length <= max_gap.

    Gaps that are too long, or that touch the clip start/end without anchors
    on both sides, are left as None — those frames will fall back to raw-mask
    passthrough downstream.
    """
    if max_gap <= 0:
        return list(centers)

    interpolated = list(centers)
    T = len(interpolated)

    i = 0
    while i < T:
        if interpolated[i] is not None:
            i += 1
            continue
        # Find span of consecutive None entries
        gap_start = i
        while i < T and interpolated[i] is None:
            i += 1
        gap_end = i  # exclusive
        gap_len = gap_end - gap_start

        if gap_len > max_gap:
            continue
        left = interpolated[gap_start - 1] if gap_start > 0 else None
        right = interpolated[gap_end] if gap_end < T else None
        if left is None or right is None:
            continue  # boundary-touching gap — leave as None
        for k, idx in enumerate(range(gap_start, gap_end)):
            alpha = (k + 1) / (gap_len + 1)
            cx = left[0] * (1.0 - alpha) + right[0] * alpha
            cy = left[1] * (1.0 - alpha) + right[1] * alpha
            interpolated[idx] = (cx, cy)
    return interpolated


# =============================================================================
# PCA orientation (per-frame silhouette long-axis angle)
# =============================================================================

def _build_head_region_mask(raw_mask_THW, head_mass_frac, aspect_threshold=1.3):
    """Build a [T, H, W] head-region mask via per-frame MASS-fraction cutoff
    on the raw silhouette — but ONLY for frames where the silhouette is tall
    enough that head/torso separation is needed.

    Logic per frame:
      - Tall silhouette (vertical_extent / horizontal_extent > aspect_threshold):
        head + body present → apply top head_mass_frac cutoff to isolate head.
      - Square or wide silhouette (aspect <= aspect_threshold):
        head-only crop (no torso to subtract) → use FULL silhouette as PCA
        support. Applying the cutoff here would amputate jawline / lower-face
        geometry and leave a near-circular forehead-only region, which biases
        PCA toward instability (round-3 multi-AI review, 2026-04-25, A5).

    Mass-fraction cutoff (when applied):
      For each frame, finds the y-row where head_mass_frac of the total
      silhouette mass lies above (image-space y=0 is the top). Marks all
      rows above that cutoff as the head region.

    Silhouette-geometry-driven, not bbox-geometry-driven. Adapts to per-frame
    actual head extent — robust across nods, profile shots, medium shots,
    loose framing, AND tight head-only crops.

    head_mass_frac=0 or >=1 disables (PCA uses full silhouette).
    """
    if head_mass_frac <= 0.0 or head_mass_frac >= 1.0:
        return None
    T, H, W = raw_mask_THW.shape
    mf = raw_mask_THW.float().clamp(0.0, 1.0)

    # Per-frame aspect-ratio detection: which frames have body mass below
    # the head that would otherwise dominate the PCA major axis?
    binary = (mf > 0.1).float()  # [T, H, W]
    rows_present = binary.any(dim=2).float()  # [T, H]: 1 where any column in row has mask
    cols_present = binary.any(dim=1).float()  # [T, W]: 1 where any row in col has mask
    v_extent = rows_present.sum(dim=1)  # [T]: count of rows containing foreground
    h_extent = cols_present.sum(dim=1).clamp_min(1.0)  # [T]
    aspect = v_extent / h_extent  # [T]: vertical/horizontal bounding-box ratio
    needs_restriction = (aspect > aspect_threshold)  # [T] bool

    # Mass-fraction cutoff (computed for ALL frames; gated below).
    row_mass = mf.sum(dim=2)  # [T, H]
    total_mass = row_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [T, 1]
    cum_mass = torch.cumsum(row_mass, dim=1)  # [T, H]
    cutoff = head_mass_frac * total_mass  # [T, 1]
    above_cutoff = (cum_mass >= cutoff).float()
    y_cut = above_cutoff.argmax(dim=1)  # [T]

    y_idx = torch.arange(H, device=raw_mask_THW.device).unsqueeze(0)  # [1, H]
    keep_rows_restricted = (y_idx <= y_cut.unsqueeze(1)).float()  # [T, H]
    keep_rows_full = torch.ones_like(keep_rows_restricted)  # [T, H]

    # Per-frame select between restricted (tall silhouette) and full (head-only).
    keep_rows = torch.where(
        needs_restriction.unsqueeze(1), keep_rows_restricted, keep_rows_full
    )

    head = keep_rows.unsqueeze(-1).expand(T, H, W).contiguous()  # [T, H, W]
    # Frames with no foreground mass → empty region; PCA gets zero ecc and
    # routes to translation-only for those.
    valid = (total_mass.squeeze(1) > 1e-3).float().view(T, 1, 1)
    return head * valid


def _compute_pca_orientations(mask_THW, region_weight_THW=None, eps=1e-6):
    """Per-frame PCA orientation + eccentricity of the silhouette.

    For each frame, computes the mass-weighted 2x2 covariance of foreground
    pixel coordinates. The major eigenvector's angle (vs the +x axis) is the
    silhouette's long-axis orientation; eccentricity = (λ_major - λ_minor) /
    (λ_major + λ_minor) is a [0, 1] confidence proxy.

    Args:
        mask_THW: [T, H, W] float in [0, 1]. Soft masks are weighted by value.
        region_weight_THW: optional [T, H, W] mask that further restricts which
            pixels contribute to the orientation. Used to isolate the head
            region from neck/shoulder mass that would otherwise dominate the
            PCA major axis under nodding (D-098 follow-up, 2026-04-25).

    Returns:
        thetas: [T] float32 tensor, axis angle in radians, range (-pi/2, pi/2].
        eccs:   [T] float32 tensor in [0, 1].
    """
    T, H, W = mask_THW.shape
    device = mask_THW.device

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    mf = mask_THW.float().clamp(0.0, 1.0)
    if region_weight_THW is not None:
        mf = mf * region_weight_THW.clamp(0.0, 1.0)

    mass = mf.sum(dim=(1, 2), keepdim=True).clamp_min(eps)  # [T, 1, 1]
    cx = (mf * xs).sum(dim=(1, 2), keepdim=True) / mass
    cy = (mf * ys).sum(dim=(1, 2), keepdim=True) / mass
    dx = xs - cx
    dy = ys - cy
    c_xx = (mf * dx * dx).sum(dim=(1, 2)) / mass.squeeze(-1).squeeze(-1)
    c_yy = (mf * dy * dy).sum(dim=(1, 2)) / mass.squeeze(-1).squeeze(-1)
    c_xy = (mf * dx * dy).sum(dim=(1, 2)) / mass.squeeze(-1).squeeze(-1)

    tr = c_xx + c_yy
    det = c_xx * c_yy - c_xy * c_xy
    disc = (tr * tr - 4.0 * det).clamp_min(0.0).sqrt()
    l1 = 0.5 * (tr + disc)
    l2 = 0.5 * (tr - disc)

    # Major-axis angle for symmetric 2x2 matrix [[a, b], [b, d]]:
    # theta = 0.5 * atan2(2b, a - d)
    thetas = 0.5 * torch.atan2(2.0 * c_xy, c_xx - c_yy)
    eccs = ((l1 - l2) / (l1 + l2 + eps)).clamp(0.0, 1.0)

    return thetas, eccs


def _smooth_axis_angles(thetas, window):
    """Median-smooth axis angles temporally. Handles mod-pi wrap by operating
    on the (cos(2θ), sin(2θ)) representation, which lives on a continuous
    unit circle without discontinuities.

    Reduces frame-to-frame θ flicker caused by SAM3 boundary jitter
    re-shaping the per-frame silhouette covariance — without this smoothing,
    raw PCA θ from a noisy mask becomes itself a source of rotational judder
    when fed back into the warp (multi-AI review, 2026-04-25).

    window=1 is identity. Replicate-padded at the temporal edges.
    """
    if window <= 1 or thetas.shape[0] < 3:
        return thetas
    cos2 = torch.cos(2.0 * thetas)
    sin2 = torch.sin(2.0 * thetas)
    pad = window // 2
    cos_padded = F.pad(cos2.view(1, 1, -1), (pad, pad), mode="replicate").view(-1)
    sin_padded = F.pad(sin2.view(1, 1, -1), (pad, pad), mode="replicate").view(-1)
    cos_un = cos_padded.unfold(0, window, 1)  # [T, window]
    sin_un = sin_padded.unfold(0, window, 1)
    cos_med = cos_un.median(dim=-1).values
    sin_med = sin_un.median(dim=-1).values
    return 0.5 * torch.atan2(sin_med, cos_med)


def _axis_angle_diff(theta_a, theta_b):
    """Smallest difference between two AXIS angles (mod pi).

    PCA orientation is unique mod pi (the long axis has no head/tail
    distinction). Naive subtraction produces spurious large differences when
    the angles straddle the +/-pi/2 boundary (e.g. theta_a=+85°, theta_b=-85°
    actually represent rotations only ~10° apart).

    Returns the SIGNED smallest difference in (-pi/2, pi/2].
    """
    d = theta_a - theta_b
    # Wrap into (-pi/2, pi/2]
    while d > math.pi / 2:
        d -= math.pi
    while d <= -math.pi / 2:
        d += math.pi
    return d


# =============================================================================
# Sub-pixel mask warp (similarity transform via grid_sample)
# =============================================================================

def _warp_mask_similarity(
    mask_HW, cx_t, cy_t, cx_j, cy_j, delta_theta, kernel_grid_xs, kernel_grid_ys
):
    """Warp mask_HW from neighbor frame's coordinate space into current frame's
    coordinate space via similarity transform (translation + rotation about the
    bbox center).

    out[x_t, y_t] = mask[x_j, y_j] where the mapping is:
        (x_j, y_j) = R(-delta_theta) @ ((x_t, y_t) - (cx_t, cy_t)) + (cx_j, cy_j)

    delta_theta is the rotation FROM neighbor frame TO current frame
    (i.e., theta_t - theta_j). The inverse is applied during the lookup so
    that pixels at angle theta_t in the output read from angle theta_j in the
    source.

    delta_theta=0 reduces exactly to the prior translation-only case
    (cos=1, sin=0 → src = (x_t, y_t) - (cx_t, cy_t) + (cx_j, cy_j) =
    (x_t, y_t) - (dx, dy) where dx = cx_t - cx_j).

    Padding mode "zeros" — out-of-bounds reads as 0 (no subject). reflection
    or border would leak silhouette content into the padded region.
    """
    H, W = mask_HW.shape
    cos_inv = math.cos(-delta_theta)
    sin_inv = math.sin(-delta_theta)

    dxs = kernel_grid_xs - cx_t
    dys = kernel_grid_ys - cy_t
    src_x = cos_inv * dxs - sin_inv * dys + cx_j
    src_y = sin_inv * dxs + cos_inv * dys + cy_j

    norm_x = (src_x / max(W - 1, 1)) * 2.0 - 1.0
    norm_y = (src_y / max(H - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
    mask_bchw = mask_HW.unsqueeze(0).unsqueeze(0)
    warped = F.grid_sample(
        mask_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return warped.squeeze(0).squeeze(0)


# =============================================================================
# Per-frame consensus
# =============================================================================

def _frame_consensus_weighted_mean(
    raw_mask, t, cx_t, cy_t, valid_dts, neighbor_warps,
    kernel_grid_xs, kernel_grid_ys,
):
    """Streaming weighted-mean accumulator over valid neighbors.

    neighbor_warps: list of (cx_j, cy_j, delta_theta) per neighbor — paired
    1:1 with valid_dts. delta_theta=0 reduces to translation-only.

    Returns (out_HW, weight_sum, valid_count). If weight_sum == 0, caller
    should fall back to raw_mask[t].
    """
    H, W = raw_mask.shape[1], raw_mask.shape[2]
    accumulator = torch.zeros((H, W), device=raw_mask.device, dtype=raw_mask.dtype)
    weight_sum = 0.0
    for dt, (cx_j, cy_j, dtheta) in zip(valid_dts, neighbor_warps):
        w = _WEIGHTS_TABLE[min(abs(dt), len(_WEIGHTS_TABLE) - 1)]
        if dt == 0:
            aligned = raw_mask[t]
        else:
            aligned = _warp_mask_similarity(
                raw_mask[t + dt], cx_t, cy_t, cx_j, cy_j, dtheta,
                kernel_grid_xs, kernel_grid_ys,
            )
        accumulator.add_(aligned, alpha=w)
        weight_sum += w
    return accumulator, weight_sum, len(valid_dts)


def _frame_consensus_median(
    raw_mask, t, cx_t, cy_t, valid_dts, neighbor_warps,
    kernel_grid_xs, kernel_grid_ys,
):
    """Median consensus over valid neighbors. Builds a small per-frame stack
    (max ~5 entries × H × W) and calls torch.median(dim=0).

    neighbor_warps: list of (cx_j, cy_j, delta_theta) per neighbor.

    Important: torch.median on an EVEN-sized dimension returns the LOWER of
    the two middle values, not the mean of them. For soft masks in [0, 1]
    this acts like a logical-AND / morphological erosion, aggressively
    shrinking the mask. We special-case k=2 to use the mean explicitly,
    which matches the architecture's promise of "graceful degradation to
    mean at low valid-count" (multi-AI review caught this 2026-04-24).

    At k=1 median is identity. At k=3+ median works as expected.
    """
    if not valid_dts:
        return None, 0
    candidates = []
    for dt, (cx_j, cy_j, dtheta) in zip(valid_dts, neighbor_warps):
        if dt == 0:
            candidates.append(raw_mask[t])
        else:
            candidates.append(
                _warp_mask_similarity(
                    raw_mask[t + dt], cx_t, cy_t, cx_j, cy_j, dtheta,
                    kernel_grid_xs, kernel_grid_ys,
                )
            )
    stack = torch.stack(candidates, dim=0)  # [K, H, W]
    if stack.shape[0] == 2:
        # torch.median picks lower-of-two; explicit mean is what we want
        return stack.mean(dim=0), 2
    return torch.median(stack, dim=0).values, len(candidates)


# =============================================================================
# Node
# =============================================================================

class NV_BboxAlignedMaskStabilizer:
    """Stabilize a per-frame mask by motion-compensating neighbors using the
    smoothed bbox-center trajectory, then fusing via weighted-mean (default)
    or median.

    Pipeline placement (the master mask fan-out):

        SAM3 → raw_mask
        NV_PointDrivenBBox → CoTracker3 bbox
        NV_OptimizeCropTrajectory(bbox) → bbox_optimized (stable signal)
        NV_BboxAlignedMaskStabilizer(raw_mask, bbox_optimized) → MASTER_MASK
                                ┌────────────────────────────┐
              MASTER_MASK ──────┤── InpaintCrop2.mask
                                ├── VaceControlVideoPrep.mask
                                ├── NV_FaceHarmonizePyramid.mask
                                ├── NV_FrameTemporalStabilize.mask
                                └── (InpaintStitch2 reads via stitcher)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_mask": ("MASK", {
                    "tooltip": "Per-frame SAM3 silhouette [T, H, W]. The unstable signal — "
                               "this is what we're stabilizing."
                }),
                "bbox_optimized": ("MASK", {
                    "tooltip": "Smoothed bbox MASK from NV_OptimizeCropTrajectory [T, H, W]. "
                               "Provides the stable per-frame motion signal (bbox center)."
                }),
                "temporal_window": ("INT", {
                    "default": 2, "min": 1, "max": 4, "step": 1,
                    "tooltip": "± window size. 2 = 5-frame consensus (default). 1 = 3-frame "
                               "(faster, less smoothing). 3-4 = wider temporal smoothing but "
                               "slower and risks oversmooth on fast motion."
                }),
            },
            "optional": {
                "consensus": (["weighted_mean", "median"], {
                    "default": "weighted_mean",
                    "tooltip": "weighted_mean (default): streaming distance-weighted accumulator. "
                               "Soft transitions where mask oscillates between contours. Best for "
                               "downstream soft-mask consumers (this pipeline). "
                               "median: morphologically decisive boundaries. Opt-in when downstream "
                               "binarizes the mask. Slower (builds per-frame stack), but exact."
                }),
                "max_center_jump_frac": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Skip neighbor if bbox-center delta exceeds this fraction of "
                               "min(H, W). Prevents alignment when motion is so large that "
                               "translation is meaningless (scene cut, fast pan). 0.25 = 25%% of "
                               "the shorter side."
                }),
                "interpolate_gap": ("INT", {
                    "default": 3, "min": 0, "max": 20, "step": 1,
                    "tooltip": "Linearly interpolate bbox-center across gaps up to this length. "
                               "Longer gaps → raw-mask passthrough on those frames. 0 disables "
                               "interpolation entirely."
                }),
                "rotation_mode": (["off", "pca"], {
                    "default": "off",
                    "tooltip": "off (default): translation-only alignment (legacy D-106 behavior, "
                               "general-purpose, known-safe across any mask type). "
                               "pca: derive per-frame orientation from PCA on the silhouette "
                               "mass (eigenvector of mass-weighted covariance). Promotes alignment "
                               "from translation (dx, dy) to similarity (dx, dy, theta). Opt in "
                               "for shots where the subject ROTATES between frames — face nods, "
                               "body twists, rolling objects. For non-rotating subjects, the "
                               "translation-only default is faster and incurs no risk. "
                               "Confidence-gated by per-frame eccentricity (>=0.35)."
                }),
                "max_rotation_deg": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 90.0, "step": 1.0,
                    "tooltip": "Skip neighbor if absolute rotation delta exceeds this (degrees). "
                               "Mirrors max_center_jump_frac for the rotation axis. 30° handles "
                               "natural head pose variation; raise for shots with large "
                               "deliberate head turns, lower for static-camera tracking."
                }),
                "head_region_frac": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Top fraction of the silhouette MASS used for PCA orientation "
                               "(only active when rotation_mode=pca). "
                               "1.0 (default): use the FULL silhouette as PCA support — correct "
                               "for any general-purpose mask (clothing/body swap, object swap, "
                               "subject mask without distinct rotating subregion). "
                               "<1.0: face/head workflow tuning. Restricts covariance to the top "
                               "head_region_frac of mask MASS, excluding static body mass that "
                               "would otherwise anchor the major axis. Adapts per-frame to actual "
                               "silhouette extent. 0.5 = top 50%% of mass (typical face+torso "
                               "shot → head + upper neck). Adaptive aspect-ratio gate auto-skips "
                               "the cutoff for square head-only crops where restriction would "
                               "amputate jawline."
                }),
                "theta_smooth_window": ("INT", {
                    "default": 5, "min": 1, "max": 15, "step": 2,
                    "tooltip": "Temporal median window for PCA θ smoothing. Reduces frame-to-frame "
                               "θ flicker from SAM3 boundary jitter — without this, raw PCA θ "
                               "from a noisy silhouette can itself become a source of rotational "
                               "judder when fed into the warp. 1 = no smoothing; 5 = ±2 frames. "
                               "Should be ≥ temporal_window for consensus path."
                }),
                "output_mode": (["soft", "binary"], {
                    "default": "soft",
                    "tooltip": "soft: passthrough float [0, 1] consensus output. "
                               "binary: threshold at 0.5 (loses gradient information; only "
                               "useful if a downstream consumer requires a hard mask)."
                }),
                "return_quality": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, embed per-frame valid-count quality stats in the info "
                               "string output. Useful for debugging which frames had degraded "
                               "consensus (boundaries, trajectory gaps, extreme motion)."
                }),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("master_mask", "info")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Mask"
    DESCRIPTION = (
        "Stabilize a per-frame mask by motion-compensating neighbors via the "
        "smoothed bbox trajectory, then fusing via weighted-mean (default) or "
        "median. The 'master mask' upstream of all temporal-decision-making nodes."
    )

    def execute(self, raw_mask, bbox_optimized, temporal_window,
                consensus="weighted_mean",
                max_center_jump_frac=0.25,
                interpolate_gap=3,
                rotation_mode="off",
                max_rotation_deg=30.0,
                head_region_frac=1.0,
                theta_smooth_window=5,
                output_mode="soft",
                return_quality=False):
        info_lines = []

        # ── Param validation (defensive — schema clamps to 1-4 but bypassable) ─
        if not (1 <= int(temporal_window) <= 4):
            raise ValueError(
                f"temporal_window must be in [1, 4]; got {temporal_window}. "
                f"_WEIGHTS_TABLE only defines weights for offsets 0-4."
            )

        # ── Shape validation + 2D auto-promotion ────────────────────────────
        # ComfyUI MASK type can arrive as [H, W] for single-frame inputs.
        if raw_mask.dim() == 2:
            raw_mask = raw_mask.unsqueeze(0)
        if bbox_optimized.dim() == 2:
            bbox_optimized = bbox_optimized.unsqueeze(0)
        if raw_mask.dim() != 3 or bbox_optimized.dim() != 3:
            raise ValueError(
                f"raw_mask and bbox_optimized must be [T, H, W] (or [H, W] for single frame); "
                f"got {list(raw_mask.shape)} / {list(bbox_optimized.shape)}"
            )
        if raw_mask.shape != bbox_optimized.shape:
            raise ValueError(
                f"raw_mask {list(raw_mask.shape)} must match bbox_optimized "
                f"{list(bbox_optimized.shape)} in T, H, W"
            )
        T, H, W = raw_mask.shape

        # ── NaN / inf sanitization (production safety) ──────────────────────
        # Real pipelines occasionally produce NaN at mask boundaries from
        # upstream warps. One NaN poisons the consensus accumulator, so we
        # sanitize up front and report it.
        n_nonfinite_raw = int((~torch.isfinite(raw_mask)).sum().item())
        if n_nonfinite_raw > 0:
            info_lines.append(f"WARN: sanitized {n_nonfinite_raw} non-finite values in raw_mask")
            raw_mask = torch.nan_to_num(raw_mask, nan=0.0, posinf=1.0, neginf=0.0)

        # Single-frame fast path — nothing to stabilize against. Always emit
        # float32 [0, 1] regardless of input dtype (ComfyUI MASK convention).
        if T < 2:
            info_lines.append(f"passthrough (T={T})")
            info = f"{LOG_PREFIX} " + " | ".join(info_lines)
            print(info)
            return (raw_mask.clone().float().clamp(0.0, 1.0), info)

        device = raw_mask.device
        work_mask = raw_mask.float()  # accumulator math wants fp32 stability
        max_jump = float(max_center_jump_frac) * float(min(H, W))

        # ── Center extraction + gap interpolation ───────────────────────────
        # IMPORTANT: keep `original_centers` separate from the interpolated
        # list. We allow interpolated centers to define alignment for THIS
        # frame's coordinate space, but only ORIGINALLY-present frames get to
        # contribute their raw mask as a neighbor — interpolated frames have
        # an estimated center but a potentially-garbage raw_mask (SAM3 may
        # have failed for the same reason the bbox detector did). Multi-AI
        # review (2026-04-24) flagged self-pollution from contributing
        # garbage masks at dt=0 for interpolated frames.
        original_centers = _extract_centers(bbox_optimized)
        n_present = sum(1 for c in original_centers if c is not None)
        centers = _interpolate_short_gaps(list(original_centers), int(interpolate_gap))
        n_after_interp = sum(1 for c in centers if c is not None)
        n_interpolated = n_after_interp - n_present
        if n_after_interp < T:
            info_lines.append(
                f"bbox: {n_present}/{T} present, +{n_interpolated} interpolated, "
                f"{T - n_after_interp} gap-passthrough"
            )
        elif n_interpolated > 0:
            info_lines.append(f"bbox: {n_present}/{T} present, +{n_interpolated} interpolated")
        else:
            info_lines.append(f"bbox: {T}/{T} present")

        # ── Pre-build coordinate grids reused across all warp calls ─────────
        # Float32 indices to match the dtype we're operating in.
        ys_grid, xs_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # ── PCA orientations (per-frame silhouette long axis) ───────────────
        # Computed once for the entire batch. Skipped when rotation_mode="off"
        # to preserve the legacy D-106 translation-only path exactly.
        # Two safeguards beyond the legacy translation-only path (multi-AI
        # review, 2026-04-25):
        #   1. Head-region weighting: PCA on full silhouette under-reports
        #      head rotation because neck + shoulder mass anchors the major
        #      axis. Restricting to top head_region_frac of bbox recovers the
        #      head-rotation signal.
        #   2. Temporal θ smoothing: raw per-frame PCA θ on a noisy silhouette
        #      can itself become a source of rotational judder. Median-smooth
        #      θ on (cos, sin) representation before consensus consumption.
        max_rotation_rad = math.radians(float(max_rotation_deg))
        if rotation_mode == "pca":
            # Head-region weight derived from RAW MASK mass distribution, not
            # bbox geometry — adapts to per-frame actual silhouette extent.
            head_region = _build_head_region_mask(work_mask, float(head_region_frac))
            if head_region is not None:
                # head_region weights: per-frame, frames with tall silhouettes
                # get cutoff applied; head-only frames keep full silhouette.
                # Quick stat: fraction of frames where region is < full mask.
                # mean over [T, H, W] then compared against 1.0 — values <1
                # indicate restriction was applied.
                with torch.no_grad():
                    per_frame_kept = head_region.float().mean(dim=(1, 2))  # [T] in [0, 1]
                    n_restricted = int((per_frame_kept < 0.99).sum().item())
                info_lines.append(
                    f"PCA region: head-mass cutoff frac={head_region_frac:.2f}, "
                    f"adaptive-restricted {n_restricted}/{T} frames "
                    f"(others used full silhouette as head-only)"
                )
            pca_thetas, pca_eccs = _compute_pca_orientations(work_mask, head_region)
            if int(theta_smooth_window) > 1:
                pca_thetas = _smooth_axis_angles(pca_thetas, int(theta_smooth_window))
                info_lines.append(f"θ-smooth: median window={int(theta_smooth_window)}")
            theta_floor = float(_PCA_ECC_MIN)
            n_low_ecc = int((pca_eccs < theta_floor).sum().item())
            if n_low_ecc > 0:
                info_lines.append(
                    f"PCA: {n_low_ecc}/{T} frames below ecc {theta_floor:.2f} "
                    f"(translation-only on those neighbor pairs)"
                )
        else:
            pca_thetas = pca_eccs = None

        # ── Output buffer + diagnostics ─────────────────────────────────────
        out = torch.empty((T, H, W), device=device, dtype=torch.float32)
        valid_counts = [0] * T
        n_passthrough = 0
        n_rotation_fallback = 0  # neighbors skipped for excess rotation
        n_rotation_applied = 0  # neighbors actually using nonzero theta

        # ── Per-frame consensus loop ────────────────────────────────────────
        for t in range(T):
            if centers[t] is None:
                # Anchor frame has no bbox → can't define alignment for THIS
                # frame's coordinate space. Pass raw mask through (clamped).
                out[t] = work_mask[t].clamp(0.0, 1.0)
                n_passthrough += 1
                continue

            cx_t, cy_t = centers[t]
            theta_t = float(pca_thetas[t].item()) if pca_thetas is not None else 0.0
            ecc_t = float(pca_eccs[t].item()) if pca_eccs is not None else 0.0

            # Collect valid neighbors and their (cx_j, cy_j, delta_theta).
            # ONLY originally-present frames (not interpolated) contribute —
            # interpolated frames have an estimated center but their raw
            # mask might be garbage (SAM3 may have failed where bbox failed).
            valid_dts = []
            neighbor_warps = []
            for dt in range(-temporal_window, temporal_window + 1):
                j = t + dt
                if j < 0 or j >= T:
                    continue
                if original_centers[j] is None:
                    continue
                cx_j, cy_j = centers[j]
                dx = cx_t - cx_j
                dy = cy_t - cy_j
                if (dx * dx + dy * dy) ** 0.5 > max_jump:
                    continue

                # Per-pair delta_theta. Confidence-gated: only apply rotation
                # when BOTH frames have a well-defined orientation. Bad theta
                # is worse than no theta (multi-AI debate, 2026-04-25).
                if (
                    rotation_mode == "pca"
                    and pca_thetas is not None
                    and ecc_t >= _PCA_ECC_MIN
                    and float(pca_eccs[j].item()) >= _PCA_ECC_MIN
                    and dt != 0
                ):
                    theta_j = float(pca_thetas[j].item())
                    dtheta = _axis_angle_diff(theta_t, theta_j)
                    if abs(dtheta) > max_rotation_rad:
                        # Excess rotation: most likely PCA estimation noise
                        # rather than real >30° head motion. Round-2 multi-AI
                        # review (2026-04-25) flagged that DROPPING the neighbor
                        # collapses consensus support exactly at problematic
                        # frames. Graceful degradation: keep neighbor but apply
                        # translation-only (dtheta=0). Codex round-1 principle
                        # "bad theta is worse than no theta" still honored —
                        # zero theta is just translation, never a wrong angle.
                        n_rotation_fallback += 1
                        dtheta = 0.0
                    elif dtheta != 0.0:
                        n_rotation_applied += 1
                else:
                    dtheta = 0.0

                valid_dts.append(dt)
                neighbor_warps.append((cx_j, cy_j, dtheta))

            if not valid_dts:
                out[t] = work_mask[t].clamp(0.0, 1.0)
                n_passthrough += 1
                continue

            if consensus == "median":
                stabilized, k = _frame_consensus_median(
                    work_mask, t, cx_t, cy_t, valid_dts, neighbor_warps, xs_grid, ys_grid
                )
                if k == 0 or stabilized is None:
                    out[t] = work_mask[t].clamp(0.0, 1.0)
                    n_passthrough += 1
                else:
                    out[t] = stabilized.clamp(0.0, 1.0)
                    valid_counts[t] = k
            else:  # weighted_mean (default)
                accum, wsum, k = _frame_consensus_weighted_mean(
                    work_mask, t, cx_t, cy_t, valid_dts, neighbor_warps, xs_grid, ys_grid
                )
                if wsum <= 0.0:
                    out[t] = work_mask[t].clamp(0.0, 1.0)
                    n_passthrough += 1
                else:
                    out[t] = (accum / wsum).clamp(0.0, 1.0)
                    valid_counts[t] = k

        # ── Diagnostics computed BEFORE binarization ───────────────────────
        # Two metrics: whole-frame (legacy) AND in-mask. For typical face shots
        # at 15-25% mask occupancy, whole-frame delta is dominated by the
        # 75-85% zero region and reads ~0.05-0.15/255 even when the boundary
        # is wiggling visibly. The in-mask metric uses the union of
        # (raw_mask[t-1] | raw_mask[t]) > 0.1 as the relevance mask, so the
        # average reflects what's happening AT the silhouette boundary —
        # which is where SAM3 jitter actually lives.
        with torch.no_grad():
            # Whole-frame (legacy — kept for backward-compat)
            raw_delta = (work_mask[1:] - work_mask[:-1]).abs().mean().item() * 255
            out_delta = (out[1:] - out[:-1]).abs().mean().item() * 255
            reduction = (
                (1.0 - out_delta / max(raw_delta, 1e-12)) * 100.0 if raw_delta > 0 else 0.0
            )

            # In-mask boundary delta — the metric that actually shows whether
            # the stabilizer is doing useful work
            union = (
                (work_mask[1:].clamp(0.0, 1.0) > 0.1)
                | (work_mask[:-1].clamp(0.0, 1.0) > 0.1)
            ).float()
            union_count = union.sum().clamp_min(1.0)
            raw_delta_inmask = (
                ((work_mask[1:] - work_mask[:-1]).abs() * union).sum().item()
                / union_count.item() * 255
            )
            out_delta_inmask = (
                ((out[1:] - out[:-1]).abs() * union).sum().item()
                / union_count.item() * 255
            )
            reduction_inmask = (
                (1.0 - out_delta_inmask / max(raw_delta_inmask, 1e-12)) * 100.0
                if raw_delta_inmask > 0 else 0.0
            )
            mask_occupancy = float(union.mean().item())

        # ── Output mode (after diagnostics) ─────────────────────────────────
        if output_mode == "binary":
            out = (out > 0.5).float()

        if n_passthrough > 0:
            info_lines.append(f"raw passthrough: {n_passthrough}/{T} frames")

        info_lines.append(
            f"in-mask delta (boundary jitter, occupancy={mask_occupancy*100:.0f}%): "
            f"raw={raw_delta_inmask:.2f}/255 -> smoothed={out_delta_inmask:.2f}/255 "
            f"(reduction {reduction_inmask:.0f}%)"
        )
        info_lines.append(
            f"whole-frame delta (diluted by zero-BG): raw={raw_delta:.2f}/255 -> "
            f"smoothed={out_delta:.2f}/255 (reduction {reduction:.0f}%)"
        )

        info_lines.append(
            f"T={T} window=+/-{temporal_window} consensus={consensus} "
            f"output={output_mode} rotation={rotation_mode}"
        )

        if rotation_mode == "pca":
            info_lines.append(
                f"rotation: {n_rotation_applied} neighbor pairs warped, "
                f"{n_rotation_fallback} fell back to translation-only "
                f"(>{max_rotation_deg:.0f}°)"
            )

        if return_quality:
            full = sum(1 for k in valid_counts if k == 2 * temporal_window + 1)
            partial = sum(1 for k in valid_counts if 0 < k < 2 * temporal_window + 1)
            empty = sum(1 for k in valid_counts if k == 0)
            info_lines.append(
                f"quality: {full} full / {partial} partial / {empty} empty (passthrough)"
            )

        info = f"{LOG_PREFIX} " + " | ".join(info_lines)
        print(info)
        # Always emit float32 [0, 1] regardless of input dtype. ComfyUI MASK
        # convention is fp32; preserving an upstream uint8/bool dtype here
        # would quantize soft consensus values to 0/1 and destroy the entire
        # benefit of stabilization.
        return (out.float(), info)


# ── Registration ─────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {
    "NV_BboxAlignedMaskStabilizer": NV_BboxAlignedMaskStabilizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_BboxAlignedMaskStabilizer": "NV BboxAligned Mask Stabilizer (master mask)",
}
