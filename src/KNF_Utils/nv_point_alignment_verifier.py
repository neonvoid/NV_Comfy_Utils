"""
NV Point Alignment Verifier — Pre-flight verification of source/render anchor pairs.

Catches anchor-identity-mismatch failures BEFORE committing to expensive downstream
runs (CoTracker3 + NV_PreCompWarp + ~75 min of Seedance/Kling). Critical because
NV_PreCompWarp's visibility-aware solver tolerates noisy anchors gracefully but
NOT swapped/misaligned identity — anchor 3 = "L_shoulder" on source but "R_hip"
on render is catastrophic.

Multi-criterion gating (v1.2):
  HARD GATES (always enforced):
    1. Name pairing intact (same length, names, order)
    2. Spatial spread / non-collinearity (convex hull area threshold)
    3. (when quick_validate=True) ≥4 anchors with mutual visibility > 40%
    4. (when quick_validate=True) Provisional Umeyama residual < 5% body diagonal
  REPORT-ONLY (advisory; gate iff assume_upright_biped=True):
    - Vertical ordering (head_y < neck_y < shoulder_y < hip_y < knee_y)
    - L/R side consistency relative to body centroid
    - Mirror detection via cross-product of (shoulder-vec) × (spine-vec)

`gating_passed` is REPORTED, not enforced. Downstream nodes see it but don't
auto-block — user inspects match_quality_report and decides.

Designed for the v1.2 VLM-driven CoTracker anchor pipeline.
"""

import json
import time

import cv2
import numpy as np
import torch


# Default 12-color qualitative palette (matches NV_VLMLandmarkCorresponder's
# Set-of-Mark coloring — kept in sync so visualizations are identifiable).
_PALETTE_BGR = [
    (66, 102, 255), (75, 192, 75), (240, 165, 0), (180, 70, 200),
    (40, 200, 200), (220, 60, 80), (140, 110, 40), (160, 200, 100),
    (240, 90, 200), (40, 130, 240), (200, 200, 60), (120, 70, 220),
]


def _parse_points_json(s, label):
    """Parse a tracking_points JSON string into a list of dicts.

    Accepts either NV_PointPicker format `[{x,y,t}, ...]` or the extended
    NV_VLMLandmarkCorresponder format `[{x,y,t,name}, ...]`. Adds index-based
    `name` fallback if names are missing.
    """
    if not s or not s.strip():
        raise ValueError(f"[NV_PointAlignmentVerifier] {label} is empty")
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"[NV_PointAlignmentVerifier] {label} JSON parse error: {e}")
    if not isinstance(parsed, list):
        raise ValueError(
            f"[NV_PointAlignmentVerifier] {label} must be a list, got {type(parsed).__name__}"
        )
    out = []
    for i, p in enumerate(parsed):
        if not isinstance(p, dict) or "x" not in p or "y" not in p:
            raise ValueError(
                f"[NV_PointAlignmentVerifier] {label} entry {i} missing x or y"
            )
        try:
            x = float(p["x"])
            y = float(p["y"])
            t = int(p.get("t", 0))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"[NV_PointAlignmentVerifier] {label} entry {i} bad numeric values: {e}"
            )
        name = str(p.get("name", f"anchor_{i}"))
        out.append({"x": x, "y": y, "t": t, "name": name, "id": i})
    return out


def _draw_anchor(img_bgr, x, y, idx, name, palette):
    """Draw a numbered colored circle + name label on a BGR image."""
    H, W = img_bgr.shape[:2]
    xi = int(round(max(0, min(W - 1, x))))
    yi = int(round(max(0, min(H - 1, y))))
    color = palette[idx % len(palette)]
    cv2.circle(img_bgr, (xi, yi), 8, color, 2, lineType=cv2.LINE_AA)
    cv2.circle(img_bgr, (xi, yi), 2, color, -1, lineType=cv2.LINE_AA)
    label = f"{idx}:{name}"
    # Draw label with a dark background for legibility
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    bx0 = max(0, xi + 10)
    by0 = max(0, yi - th - 6)
    by1 = max(0, yi - 2)
    bx1 = min(W - 1, bx0 + tw + 4)
    cv2.rectangle(img_bgr, (bx0, by0), (bx1, by1), (0, 0, 0), -1)
    cv2.putText(img_bgr, label, (bx0 + 2, by1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)


def _image_tensor_to_bgr_uint8(image_tensor, frame_idx):
    """Pull frame_idx from IMAGE [B,H,W,3] (RGB float [0,1]) → BGR uint8 H×W×3."""
    fi = max(0, min(image_tensor.shape[0] - 1, frame_idx))
    arr = (image_tensor[fi].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _bgr_to_image_tensor(bgr_arr):
    """Convert BGR uint8 H×W×3 → IMAGE [1,H,W,3] RGB float [0,1]."""
    rgb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _convex_hull_area(points_xy):
    """Return convex-hull area in pixel² for a list of (x,y) points; 0 if <3 points."""
    if len(points_xy) < 3:
        return 0.0
    pts = np.array(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    try:
        hull = cv2.convexHull(pts)
        return float(cv2.contourArea(hull))
    except cv2.error:
        return 0.0


class NV_PointAlignmentVerifier:
    """Side-by-side anchor-pair verification with multi-criterion gating."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {
                    "tooltip": "Cropped source video frames (post InpaintCrop2). [B,H,W,3] in [0,1].",
                }),
                "source_tracking_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Source-side anchor JSON from NV_VLMLandmarkCorresponder.",
                }),
                "render_images": ("IMAGE", {
                    "tooltip": "Cropped render video frames (post InpaintCrop2). [B,H,W,3] in [0,1].",
                }),
                "render_tracking_points": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Render-side anchor JSON from NV_VLMLandmarkCorresponder.",
                }),
            },
            "optional": {
                "min_hull_area_frac": ("FLOAT", {
                    "default": 0.04, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": (
                        "Minimum convex-hull area as fraction of crop_w*crop_h for the "
                        "spatial-spread gate. Catches collinear/clustered anchors that "
                        "would collapse Umeyama solver conditioning. 0.04 = 4% of frame."
                    ),
                }),
                "assume_upright_biped": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "When True, the report-only biped checks (vertical ordering, "
                        "L/R side consistency, mirror detection) become HARD GATES. "
                        "Default False — these checks fail on legitimate non-upright "
                        "shots (parkour, reclining, rear-facing). Spatial spread + "
                        "Umeyama residual remain pose-agnostic and stay as hard gates "
                        "regardless."
                    ),
                }),
                "quick_validate_with_cotracker": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Run a cheap CoTracker3 proxy pass (every Nth frame at "
                        "validation_stride) to compute mutual visibility + Umeyama "
                        "residual. Adds ~30-60s but greatly improves the gate."
                    ),
                }),
                "validation_stride": ("INT", {
                    "default": 8, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Frame stride for the cheap CoTracker proxy pass.",
                }),
                "min_mutual_visible_anchors": ("INT", {
                    "default": 4, "min": 2, "max": 16, "step": 1,
                    "tooltip": (
                        "Minimum number of anchors with mutual visibility > 40% required "
                        "for the visibility gate (only when quick_validate=True)."
                    ),
                }),
                "max_residual_frac": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": (
                        "Maximum allowed Umeyama RMSE residual as fraction of body "
                        "diagonal at K keyframes (only when quick_validate=True)."
                    ),
                }),
                "max_keyframes_to_render": ("INT", {
                    "default": 4, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Cap on number of keyframes shown in the side-by-side preview.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "BOOLEAN")
    RETURN_NAMES = ("verification_image", "match_quality_report", "gating_passed")
    FUNCTION = "verify"
    CATEGORY = "NV_Utils/Tracking"
    DESCRIPTION = (
        "Pre-flight verification of source/render anchor pairs from "
        "NV_VLMLandmarkCorresponder. Side-by-side overlay + multi-criterion gating "
        "(name pairing, spatial spread, mutual visibility, Umeyama residual). "
        "Biped pose checks demoted to report-only by default — opt in via "
        "assume_upright_biped=True. Catches identity-mismatch BEFORE the expensive "
        "downstream PreCompWarp run."
    )

    def verify(self, source_images, source_tracking_points,
               render_images, render_tracking_points,
               min_hull_area_frac=0.04, assume_upright_biped=False,
               quick_validate_with_cotracker=False, validation_stride=8,
               min_mutual_visible_anchors=4, max_residual_frac=0.05,
               max_keyframes_to_render=4):
        t_start = time.perf_counter()

        # --- Parse + validate inputs ---------------------------------------
        src = _parse_points_json(source_tracking_points, "source_tracking_points")
        ren = _parse_points_json(render_tracking_points, "render_tracking_points")

        report_lines = ["=" * 64, "NV_PointAlignmentVerifier — Match Quality Report", "=" * 64]
        criteria = []  # list of (name, status_str, detail_str, is_hard_gate)

        # --- Hard gate 1: Name pairing intact ------------------------------
        name_pair_ok = True
        if len(src) != len(ren):
            name_pair_ok = False
            criteria.append(("name_pairing", "FAIL", f"length mismatch: src={len(src)}, ren={len(ren)}", True))
        else:
            for i, (s, r) in enumerate(zip(src, ren)):
                if s["name"] != r["name"]:
                    name_pair_ok = False
                    criteria.append(("name_pairing", "FAIL",
                                     f"name mismatch at idx {i}: src={s['name']!r}, ren={r['name']!r}",
                                     True))
                    break
            if name_pair_ok:
                criteria.append(("name_pairing", "PASS",
                                 f"{len(src)} anchors, names + order intact", True))

        # Build accepted-anchor lists (drop missing/sentinel x=-1)
        src_acc = [s for s in src if s["x"] >= 0 and s["y"] >= 0]
        ren_acc = [r for r in ren if r["x"] >= 0 and r["y"] >= 0]
        if name_pair_ok:
            mutually_acc = [
                (s, r) for s, r in zip(src, ren)
                if s["x"] >= 0 and s["y"] >= 0 and r["x"] >= 0 and r["y"] >= 0
            ]
        else:
            mutually_acc = []

        crop_h_src = source_images.shape[1]
        crop_w_src = source_images.shape[2]
        crop_h_ren = render_images.shape[1]
        crop_w_ren = render_images.shape[2]

        # --- Hard gate 2: Spatial spread (convex hull area) -----------------
        if mutually_acc:
            src_xy = [(s["x"], s["y"]) for s, _ in mutually_acc]
            ren_xy = [(r["x"], r["y"]) for _, r in mutually_acc]
            src_hull = _convex_hull_area(src_xy)
            ren_hull = _convex_hull_area(ren_xy)
            src_min = min_hull_area_frac * crop_w_src * crop_h_src
            ren_min = min_hull_area_frac * crop_w_ren * crop_h_ren
            spread_ok = src_hull >= src_min and ren_hull >= ren_min
            criteria.append((
                "spatial_spread",
                "PASS" if spread_ok else "FAIL",
                f"src_hull={src_hull:.0f}px² (min {src_min:.0f}), "
                f"ren_hull={ren_hull:.0f}px² (min {ren_min:.0f})",
                True,
            ))
        else:
            spread_ok = False
            criteria.append(("spatial_spread", "FAIL", "no mutually accepted anchors", True))

        # --- Report-only: vertical ordering ---------------------------------
        # v1.2 multi-AI review: tri-state mapping per spec.
        #   0 violations → PASS
        #   1 violation  → WARN  (default) or FAIL when assume_upright_biped=True
        #   2+ violations → FAIL (always)
        #   -1 (insufficient anchors) → PASS (cannot fail what we can't check)
        vert_viol_src, vert_msg_src = self._check_vertical_ordering(src)
        vert_viol_ren, vert_msg_ren = self._check_vertical_ordering(ren)
        max_viol = max(vert_viol_src, vert_viol_ren)
        if max_viol <= 0:
            vert_status = "PASS"
        elif max_viol == 1:
            vert_status = "FAIL" if assume_upright_biped else "WARN"
        else:  # 2+
            vert_status = "FAIL"
        criteria.append((
            "vertical_ordering",
            vert_status,
            f"src: {vert_msg_src} | ren: {vert_msg_ren}",
            assume_upright_biped,
        ))

        # --- Report-only: L/R side consistency relative to body centroid ---
        side_ok, side_msg = self._check_side_consistency(src, ren)
        criteria.append((
            "side_consistency",
            "PASS" if side_ok else ("WARN" if not assume_upright_biped else "FAIL"),
            side_msg,
            assume_upright_biped,
        ))

        # --- Report-only: mirror detection (cross product) -----------------
        mirror_warn, mirror_msg = self._check_mirror(src, ren)
        criteria.append((
            "mirror_detection",
            "WARN" if mirror_warn else "PASS",
            mirror_msg,
            assume_upright_biped,
        ))

        # --- Hard gates 3 + 4: quick CoTracker proxy validation ------------
        mutual_vis_ok = None
        residual_ok = None
        mutual_vis_msg = "skipped (quick_validate=False)"
        residual_msg = "skipped (quick_validate=False)"
        if quick_validate_with_cotracker:
            try:
                mutual_vis_ok, residual_ok, mutual_vis_msg, residual_msg = \
                    self._run_quick_cotracker_validation(
                        source_images, render_images, src, ren,
                        validation_stride, min_mutual_visible_anchors, max_residual_frac,
                    )
            except Exception as e:
                mutual_vis_ok = False
                residual_ok = False
                mutual_vis_msg = f"validation failed: {e}"
                residual_msg = f"validation failed: {e}"
        criteria.append((
            "mutual_visibility",
            "PASS" if mutual_vis_ok else ("FAIL" if quick_validate_with_cotracker else "N/A"),
            mutual_vis_msg,
            quick_validate_with_cotracker,
        ))
        # Renamed from "umeyama_residual" in v1.2 — implementation uses
        # cv2.estimateAffinePartial2D (RANSAC similarity), which is
        # similarity-class but NOT Umeyama's SVD-on-centered-points formulation.
        # The downstream NV_PreCompWarp solver is the actual Umeyama; this is
        # just a cheap pre-flight similarity-fit check.
        criteria.append((
            "similarity_fit_residual",
            "PASS" if residual_ok else ("FAIL" if quick_validate_with_cotracker else "N/A"),
            residual_msg,
            quick_validate_with_cotracker,
        ))

        # --- Compute final gating_passed ------------------------------------
        # Hard gates: criteria where is_hard_gate=True must be PASS or N/A
        gating_passed = True
        for name, status, _detail, is_hard in criteria:
            if is_hard and status not in ("PASS", "N/A"):
                gating_passed = False
                break

        # --- Build report ---------------------------------------------------
        report_lines.append("")
        report_lines.append(f"{'CRITERION':<22} {'STATUS':<6} {'GATE':<6} DETAIL")
        report_lines.append("-" * 64)
        for name, status, detail, is_hard in criteria:
            gate_label = "HARD" if is_hard else "WARN"
            report_lines.append(f"{name:<22} {status:<6} {gate_label:<6} {detail}")
        report_lines.append("")
        report_lines.append(f"OVERALL gating_passed = {gating_passed}")
        report_lines.append("")
        report_lines.append("Per-anchor:")
        for i in range(min(len(src), len(ren))):
            s, r = src[i], ren[i]
            s_status = "missing" if s["x"] < 0 else f"({s['x']:.0f},{s['y']:.0f})@t={s['t']}"
            r_status = "missing" if r["x"] < 0 else f"({r['x']:.0f},{r['y']:.0f})@t={r['t']}"
            report_lines.append(f"  {i:>2} {s['name']:<14} src={s_status:<28} ren={r_status}")

        if not gating_passed:
            report_lines.append("")
            report_lines.append("[NV_PointAlignmentVerifier] gate FAILED — see criteria above")
            print(f"[NV_PointAlignmentVerifier] gate FAILED — see match_quality_report")

        elapsed = time.perf_counter() - t_start
        report_lines.append("")
        report_lines.append(f"Verifier elapsed: {elapsed * 1000:.0f} ms")
        report = "\n".join(report_lines)

        # --- Build verification image (side-by-side per keyframe) ----------
        verification_image = self._build_verification_image(
            source_images, render_images, src, ren, max_keyframes_to_render,
        )

        return (verification_image, report, gating_passed)

    @staticmethod
    def _check_vertical_ordering(points):
        """Verify head_y < neck_y < avg(shoulder y) < avg(hip y) < avg(knee y).
        Image coords: lower y = higher in frame for typical actor poses.

        Returns (violation_count: int, msg: str). Caller maps to PASS/WARN/FAIL:
            0 → PASS, 1 → WARN, 2+ → FAIL  (per v1.2 spec).
        violation_count of -1 means "insufficient anchors to check" (treat as PASS).
        """
        by_name = {p["name"]: p for p in points if p["x"] >= 0 and p["y"] >= 0}

        def _y(name):
            return by_name[name]["y"] if name in by_name else None

        def _avg(names):
            ys = [by_name[n]["y"] for n in names if n in by_name]
            return sum(ys) / len(ys) if ys else None

        steps = [
            ("head", _y("head")),
            ("neck", _y("neck")),
            ("shoulders", _avg(["L_shoulder", "R_shoulder"])),
            ("hips", _avg(["L_hip", "R_hip"])),
            ("knees", _avg(["L_knee", "R_knee"])),
        ]
        present = [(n, v) for n, v in steps if v is not None]
        if len(present) < 2:
            return -1, "insufficient anchors for vertical check (allowed)"
        violations = []
        for (n1, v1), (n2, v2) in zip(present, present[1:]):
            if v1 >= v2:
                violations.append(f"{n1}({v1:.0f})>={n2}({v2:.0f})")
        if not violations:
            return 0, f"{len(present)} levels ordered correctly"
        return len(violations), f"{len(violations)} viol: " + ", ".join(violations[:3])

    @staticmethod
    def _check_side_consistency(src, ren):
        """For each {L_x, R_x} pair, verify same horizontal ordering in both videos."""
        src_by = {p["name"]: p for p in src if p["x"] >= 0}
        ren_by = {p["name"]: p for p in ren if p["x"] >= 0}
        pairs = [("L_shoulder", "R_shoulder"), ("L_elbow", "R_elbow"),
                 ("L_hip", "R_hip"), ("L_knee", "R_knee")]
        violations = []
        checked = 0
        for L, R in pairs:
            if L in src_by and R in src_by and L in ren_by and R in ren_by:
                checked += 1
                src_lr = src_by[L]["x"] - src_by[R]["x"]
                ren_lr = ren_by[L]["x"] - ren_by[R]["x"]
                if (src_lr >= 0) != (ren_lr >= 0):
                    violations.append(f"{L}/{R} swapped")
        if checked == 0:
            return True, "insufficient L/R pairs to check (allowed)"
        if not violations:
            return True, f"{checked}/{len(pairs)} pairs checked, all consistent"
        return False, f"{len(violations)} swap(s): " + ", ".join(violations)

    @staticmethod
    def _check_mirror(src, ren):
        """Cross-product mirror check (v1.2 — corrected from v1.1's dot product).

        z_normal = shoulder_vec.x * spine_vec.y - shoulder_vec.y * spine_vec.x
        Sign flip between source and render = horizontal mirror.
        """
        def _vec(points, A, B):
            by = {p["name"]: p for p in points if p["x"] >= 0}
            if A not in by or B not in by:
                return None
            return (by[B]["x"] - by[A]["x"], by[B]["y"] - by[A]["y"])

        def _midpoint(points, A, B):
            by = {p["name"]: p for p in points if p["x"] >= 0}
            if A not in by or B not in by:
                return None
            return ((by[A]["x"] + by[B]["x"]) / 2.0,
                    (by[A]["y"] + by[B]["y"]) / 2.0)

        def _z_normal(points):
            shoulder_vec = _vec(points, "L_shoulder", "R_shoulder")
            neck_pt = next((p for p in points if p["name"] == "neck" and p["x"] >= 0), None)
            hip_mid = _midpoint(points, "L_hip", "R_hip")
            if not all([shoulder_vec, neck_pt, hip_mid]):
                return None
            spine_vec = (neck_pt["x"] - hip_mid[0], neck_pt["y"] - hip_mid[1])
            return shoulder_vec[0] * spine_vec[1] - shoulder_vec[1] * spine_vec[0]

        z_src = _z_normal(src)
        z_ren = _z_normal(ren)
        if z_src is None or z_ren is None:
            return False, "insufficient anchors for mirror check"
        # Noise threshold: require both magnitudes well above noise floor
        # to avoid false flags on near-degenerate poses
        noise_threshold = 100.0  # px² — small relative to any real torso
        if abs(z_src) < noise_threshold or abs(z_ren) < noise_threshold:
            return False, f"degenerate pose (|z_src|={abs(z_src):.0f}, |z_ren|={abs(z_ren):.0f})"
        if (z_src > 0) != (z_ren > 0):
            return True, (f"MIRROR DETECTED: z_src={z_src:.0f}, z_ren={z_ren:.0f} "
                          f"(opposite signs). LR anchors will be wrong.")
        return False, f"orientation matches (z_src={z_src:.0f}, z_ren={z_ren:.0f})"

    def _run_quick_cotracker_validation(self, source_images, render_images,
                                         src, ren, stride, min_visible, max_resid_frac):
        """Cheap CoTracker3 proxy: subsample frames, track, compute mutual visibility +
        provisional Umeyama residual on the keyframe positions.

        Best-effort — falls back gracefully if CoTracker3 not loadable.
        """
        try:
            from .cotracker_bridge import _get_cotracker_model
            model = _get_cotracker_model()
        except Exception as e:
            raise RuntimeError(f"could not load CoTracker3: {e}")

        # Subsample images by stride. v1.2 multi-AI review BLOCKER: must also
        # include every anchor's `t` so the CoTracker queries can use exact
        # (t, x, y) tuples. Previously we re-mapped t to the nearest sub-frame
        # index but kept the original (x, y) coords — that initialized
        # CoTracker on the wrong pixels for any anchor whose t fell BETWEEN
        # subsampled frames.
        T_src = source_images.shape[0]
        T_ren = render_images.shape[0]
        T_use = min(T_src, T_ren)
        sub_idx_set = set(range(0, T_use, stride))
        sub_idx_set.add(T_use - 1)
        for p in src + ren:
            if p["x"] >= 0 and p["y"] >= 0:
                t_clamped = max(0, min(T_use - 1, int(p["t"])))
                sub_idx_set.add(t_clamped)
        sub_idx = sorted(sub_idx_set)
        # Lookup table for exact frame → sub position
        sub_pos_lookup = {t: pos for pos, t in enumerate(sub_idx)}

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        def _track(images, points):
            sub = images[sub_idx].to(device=device, dtype=dtype).movedim(-1, 1).unsqueeze(0) * 255.0
            queries = []
            for i, p in enumerate(points):
                if p["x"] < 0 or p["y"] < 0:
                    continue
                t_full = max(0, min(T_use - 1, int(p["t"])))
                # Exact lookup: t_full is guaranteed to be in sub_pos_lookup
                # because we explicitly added every anchor's t to sub_idx.
                t_sub = sub_pos_lookup[t_full]
                queries.append([t_sub, p["x"], p["y"], i])  # last col = original anchor idx
            if not queries:
                return None, None, None
            q_tensor = torch.tensor([[q[0], q[1], q[2]] for q in queries],
                                    dtype=dtype, device=device).unsqueeze(0)
            with torch.no_grad():
                tracks, vis = model(sub, queries=q_tensor)
            return tracks[0].cpu().numpy(), vis[0].cpu().numpy(), [q[3] for q in queries]

        src_tracks, src_vis, src_idx = _track(source_images, src)
        ren_tracks, ren_vis, ren_idx = _track(render_images, ren)
        if src_tracks is None or ren_tracks is None:
            raise RuntimeError("not enough valid anchors to track")

        # Compute mutual visibility per anchor (intersection of src_idx and ren_idx)
        common_anchor_indices = [i for i in src_idx if i in ren_idx]
        if not common_anchor_indices:
            return False, False, "no common anchors after sub-sampling", "no anchors for fit"

        # Mutual visibility: fraction of frames where BOTH tracks are visible above 0.5
        mutual_visible_count = 0
        for ai in common_anchor_indices:
            si = src_idx.index(ai)
            ri = ren_idx.index(ai)
            mutual = (src_vis[:, si] > 0.5) & (ren_vis[:, ri] > 0.5)
            if mutual.mean() > 0.40:
                mutual_visible_count += 1
        mutual_vis_ok = mutual_visible_count >= min_visible
        mutual_msg = (f"{mutual_visible_count}/{len(common_anchor_indices)} anchors with "
                      f"mutual_vis>40% (need {min_visible})")

        # Provisional similarity-fit RMSE at all sub-frames (v1.2 fix from
        # Gemini review: previously only first/middle/last were checked, which
        # short-circuited on occlusion in the chosen middle frame and yielded
        # "no fits could be computed". Iterating over every sub_idx position
        # is cheap (~milliseconds) and avoids that failure mode.)
        sample_frames = list(range(len(sub_idx)))
        residuals = []
        body_diags = []
        for f in sample_frames:
            from_pts = []
            to_pts = []
            for ai in common_anchor_indices:
                si = src_idx.index(ai)
                ri = ren_idx.index(ai)
                if src_vis[f, si] > 0.5 and ren_vis[f, ri] > 0.5:
                    from_pts.append(src_tracks[f, si])
                    to_pts.append(ren_tracks[f, ri])
            if len(from_pts) < 2:
                continue
            from_arr = np.asarray(from_pts, dtype=np.float64)
            to_arr = np.asarray(to_pts, dtype=np.float64)
            try:
                M, _ = cv2.estimateAffinePartial2D(from_arr, to_arr, method=cv2.RANSAC)
            except cv2.error:
                M = None
            if M is None:
                continue
            ones = np.ones((from_arr.shape[0], 1), dtype=np.float64)
            from_h = np.hstack([from_arr, ones])
            transformed = (M @ from_h.T).T
            rmse = float(np.sqrt(np.mean(np.linalg.norm(transformed - to_arr, axis=1) ** 2)))
            # Body diagonal at this frame: bbox of common anchors on render side
            x_min, y_min = to_arr.min(axis=0)
            x_max, y_max = to_arr.max(axis=0)
            diag = float(np.hypot(x_max - x_min, y_max - y_min))
            if diag > 0:
                residuals.append(rmse / diag)
                body_diags.append(diag)
        if not residuals:
            return mutual_vis_ok, False, mutual_msg, "no fits could be computed"
        max_r = max(residuals)
        residual_ok = max_r < max_resid_frac
        residual_msg = (f"max RMSE/body_diag = {max_r:.3f} (limit {max_resid_frac:.2f}); "
                        f"{len(residuals)} fits, body_diag mean={np.mean(body_diags):.0f}px")
        return mutual_vis_ok, residual_ok, mutual_msg, residual_msg

    def _build_verification_image(self, source_images, render_images, src, ren, max_kf):
        """Build side-by-side panel for up to max_kf keyframes."""
        # Pick keyframes: union of src_t and ren_t (deduped), sorted, capped
        kfs_src = sorted(set(p["t"] for p in src if p["x"] >= 0))
        kfs_ren = sorted(set(p["t"] for p in ren if p["x"] >= 0))
        kfs = sorted(set(kfs_src + kfs_ren))[:max_kf]
        if not kfs:
            kfs = [0]

        rows = []
        for t in kfs:
            t_src = max(0, min(source_images.shape[0] - 1, t))
            t_ren = max(0, min(render_images.shape[0] - 1, t))
            src_bgr = _image_tensor_to_bgr_uint8(source_images, t_src)
            ren_bgr = _image_tensor_to_bgr_uint8(render_images, t_ren)
            for s in src:
                if s["x"] >= 0 and s["y"] >= 0 and s["t"] == t:
                    _draw_anchor(src_bgr, s["x"], s["y"], s["id"], s["name"], _PALETTE_BGR)
            for r in ren:
                if r["x"] >= 0 and r["y"] >= 0 and r["t"] == t:
                    _draw_anchor(ren_bgr, r["x"], r["y"], r["id"], r["name"], _PALETTE_BGR)

            # Resize render panel to match source height for side-by-side
            sh, sw = src_bgr.shape[:2]
            rh, rw = ren_bgr.shape[:2]
            target_h = sh
            scale = target_h / float(rh)
            new_w = max(1, int(round(rw * scale)))
            ren_resized = cv2.resize(ren_bgr, (new_w, target_h),
                                     interpolation=cv2.INTER_AREA)
            label_bar = np.zeros((22, sw + new_w, 3), dtype=np.uint8)
            cv2.putText(label_bar, f"keyframe t={t}  source (left)  render (right)",
                        (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1,
                        cv2.LINE_AA)
            row = np.hstack([src_bgr, ren_resized])
            row = np.vstack([label_bar, row])
            rows.append(row)

        # Stack rows; pad shorter rows on the right
        max_w = max(r.shape[1] for r in rows)
        padded = []
        for r in rows:
            if r.shape[1] < max_w:
                pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
                r = np.hstack([r, pad])
            padded.append(r)
        full = np.vstack(padded)
        return _bgr_to_image_tensor(full)


NODE_CLASS_MAPPINGS = {
    "NV_PointAlignmentVerifier": NV_PointAlignmentVerifier,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_PointAlignmentVerifier": "NV Point Alignment Verifier",
}
