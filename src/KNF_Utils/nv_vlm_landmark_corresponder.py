"""
NV VLM Landmark Corresponder — Auto-detect matched tracking anchors on source + render.

Replaces the manual NV_PointPicker → NV_PointPicker dual-click flow for the
clothing/bag-swap pipeline (workstream C). Calls Gemini twice (source-first,
render-conditioned-on-source) and snaps each landmark to its nearest high-texture
pixel via Shi-Tomasi corner detection. Outputs paired tracking_points JSON in
NV_PointPicker format that drops directly into NV_CoTrackerTrajectoriesJSON.

Architecture (v1.2):
1. SOURCE Gemini call — multi-frame, returns [{name, x, y, t, anchor_type}]
2. SOURCE coord conversion + boundary-safe pixel clamp BEFORE Shi-Tomasi
3. SOURCE Shi-Tomasi snap (resolution-aware radius, distance-weighted ranking,
   cv2.cornerSubPix on fallback)
4. RENDER Gemini call — multi-frame, conditioned on the source's resolved list
   PLUS a single Set-of-Mark image (one source frame with all anchors plotted)
5. RENDER coord conversion + boundary-safe clamp + Shi-Tomasi snap
6. Pairing validation (single retry on schema violations)
7. Emit two tracking_points JSONs in identical name/order

Designed for v1.2 of the VLM-driven CoTracker3 anchor pipeline (research handoff
2026-05-01_vlm_cotracker_anchor_architecture.md). v1 supports SINGLE-ACTOR crops
only — multi-person/crowd behavior is undefined.
"""

import io
import json
import os
import re
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


GEMINI_MODELS = ["gemini-3.1-pro-preview", "gemini-3-flash-preview"]
DEFAULT_VOCAB = (
    "head,neck,L_shoulder,R_shoulder,L_elbow,R_elbow,"
    "L_hip,R_hip,L_knee,R_knee"
)
ALLOWED_ANCHOR_TYPES = (
    "outer_contour_corner",
    "seam_intersection",
    "joint_crease",
    "garment_edge_corner",
    "high_contrast_fold",
    "missing",
)

# RGB palette for Set-of-Mark visualization (kept in sync with NV_PointAlignmentVerifier)
_PALETTE_RGB = [
    (255, 102, 66), (75, 192, 75), (0, 165, 240), (200, 70, 180),
    (200, 200, 40), (80, 60, 220), (40, 110, 140), (100, 200, 160),
    (200, 90, 240), (240, 130, 40), (60, 200, 200), (220, 70, 120),
]

# =============================================================================
# Flexible keyframe spec parser (v1.2 fix per user feedback — accept multiple syntaxes)
# =============================================================================


def _resolve_negative(idx, T):
    """Convert negative Python-style indices: -1 → T-1, -5 → T-5."""
    return T + idx if idx < 0 else idx


def _normalize_indices(indices, T, name):
    """Clamp to [0, T-1], dedup, sort. Raise on empty result."""
    if not indices:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: parsed to empty list")
    out = sorted({max(0, min(T - 1, int(i))) for i in indices})
    if not out:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: all indices out of bounds (T={T})")
    return out


def _expand_token(tok, T, name):
    """Expand a single token into a list of ints (pre-clamp).

    Supported forms:
      "47"         → plain int
      "-1"         → negative (= T-1)
      "0-10"       → inclusive range
      "0-100:5"    → inclusive range with step
      "-5--1"      → range using negatives
    """
    # Try plain int (handles negative too)
    try:
        return [_resolve_negative(int(tok), T)]
    except ValueError:
        pass

    # Optional ":step" suffix
    range_part, step = tok, 1
    if ":" in tok:
        rp, sp = tok.rsplit(":", 1)
        try:
            step = int(sp)
        except ValueError:
            raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: bad step in token {tok!r}")
        range_part = rp

    # Find the separating dash. If range_part starts with '-', skip the leading
    # sign of the first number when searching.
    start_search = 1 if range_part.startswith("-") else 0
    sep_pos = range_part.find("-", start_search)
    if sep_pos < 0:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: unparseable token {tok!r}")
    try:
        a = int(range_part[:sep_pos])
        b = int(range_part[sep_pos + 1:])
    except ValueError:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: unparseable range in {tok!r}")

    a = _resolve_negative(a, T)
    b = _resolve_negative(b, T)
    if step <= 0:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: step must be positive, got {step}")
    if a <= b:
        return list(range(a, b + 1, step))
    return list(range(a, b - 1, -step))


def _parse_keyframe_spec(spec, T, name="keyframes"):
    """Flexibly parse a keyframe specification into a sorted unique list of valid indices.

    Accepted forms (mix freely with commas, semicolons, or whitespace):
      JSON array:        "[0, 24, 47]"
      Comma-separated:   "0, 24, 47"
      Whitespace-sep:    "0 24 47"
      Single int:        "47" or 47 (native)
      Native list/tuple: [0, 24, 47]
      Inclusive range:   "0-10"            → 0,1,2,...,10
      Range with step:   "0-100:5"         → 0,5,10,...,100
      Negative index:    "-1"              → T-1 (last frame)
      Negative range:    "-10--1"          → last 10 frames
      Mixed:             "0, 5-10, 47, 80-100:5, -1"
      Bracketed list:    "(0, 24, 47)" or "[0, 24, 47]"

    All indices are clamped to [0, T-1] then deduplicated and sorted.
    Raises ValueError on unparseable input or empty result.
    """
    if spec is None:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: spec is None")

    # Native int
    if isinstance(spec, int):
        return _normalize_indices([spec], T, name)

    # Native list/tuple — recurse into items (each may be int or sub-string)
    if isinstance(spec, (list, tuple)):
        result = []
        for item in spec:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, str):
                result.extend(_parse_keyframe_spec(item, T, name))
            else:
                raise ValueError(
                    f"[NV_VLMLandmarkCorresponder] {name}: list item {item!r} is not int/str"
                )
        return _normalize_indices(result, T, name)

    if not isinstance(spec, str):
        raise ValueError(
            f"[NV_VLMLandmarkCorresponder] {name}: expected str/int/list, got {type(spec).__name__}"
        )

    s = spec.strip()
    if not s:
        raise ValueError(f"[NV_VLMLandmarkCorresponder] {name}: empty spec")

    # Try JSON path first (preserves backwards-compat with "[0, 24, 47]" bracketed numerics)
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _parse_keyframe_spec(parsed, T, name)
        except json.JSONDecodeError:
            pass  # fall through to general parsing

    # Strip outer brackets/parens if present (for non-JSON-strict bracketed lists)
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()

    # Replace semicolons with commas (forgiving alternate separator)
    s = s.replace(";", ",")

    # Split on commas, then on whitespace within each chunk
    indices = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        for tok in chunk.split():
            if tok:
                indices.extend(_expand_token(tok, T, name))

    return _normalize_indices(indices, T, name)


ANCHOR_TYPE_DEFINITIONS = """anchor_type values (use one per landmark):
- "outer_contour_corner": a sharp corner where the actor's silhouette changes
  direction against the background (e.g. tip of shoulder seam against sky)
- "seam_intersection": where two visible garment seams meet at a high-contrast
  junction (e.g. shirt-sleeve seam meeting collar)
- "joint_crease": a fold line at an articulating joint that creates a strong
  oriented edge (e.g. elbow inner crease)
- "garment_edge_corner": a corner of a garment EDGE (collar tip, hem corner,
  pocket corner) -- internal garment landmark, not the silhouette
- "high_contrast_fold": a non-seam cloth fold with sharp local contrast and
  stable position across frames (use ONLY when no seam/crease/corner option
  is available)
- "missing": the landmark is occluded, ambiguous-side, or has no trackable
  feature in any provided frame
"""


# =============================================================================
# API key resolution + Gemini call (mirrors AVM's pattern but standalone)
# =============================================================================


def _resolve_api_key(api_config, ui_key=""):
    """Tiered key lookup: api_config (AVM_API dict) → env var → UI input."""
    if api_config is not None and isinstance(api_config, dict) and api_config.get("api_key"):
        return api_config["api_key"], api_config.get("model_name", GEMINI_MODELS[0]), api_config.get("provider", "gemini_direct")
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key, GEMINI_MODELS[0], "gemini_direct"
    if ui_key.strip():
        return ui_key.strip(), GEMINI_MODELS[0], "gemini_direct"
    raise ValueError(
        "[NV_VLMLandmarkCorresponder] No Gemini API key found. Wire an AVM API "
        "Config node, set GEMINI_API_KEY env var, or enter the key in the node UI."
    )


def _call_gemini_direct(pil_imgs_with_labels, prompt, api_key, model_name):
    """Call google-genai with interleaved text labels and images.

    pil_imgs_with_labels: list of (label_str, PIL.Image) tuples. Labels are
    inserted BEFORE each image so Gemini sees explicit "Frame N:" / "Source
    reference:" cues.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai not installed. Run: pip install google-genai"
        )
    client = genai.Client(api_key=api_key)
    parts = []
    for label, img in pil_imgs_with_labels:
        if label:
            parts.append(types.Part.from_text(text=label))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    parts.append(types.Part.from_text(text=prompt))
    response = client.models.generate_content(model=model_name, contents=parts)
    return response.text


def _call_openrouter(pil_imgs_with_labels, prompt, api_key, model_name, base_url):
    """Call OpenRouter with interleaved labels + images. Mirrors AVM's pattern."""
    import base64
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")
    content = []
    for label, img in pil_imgs_with_labels:
        if label:
            content.append({"type": "text", "text": label})
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
    }
    resp = requests.post(f"{base_url}/chat/completions", headers=headers,
                          json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_vlm(pil_imgs_with_labels, prompt, api_key, model_name, provider, base_url):
    """Dispatch to gemini_direct or openrouter based on provider."""
    if provider == "openrouter":
        return _call_openrouter(pil_imgs_with_labels, prompt, api_key, model_name,
                                 base_url or "https://openrouter.ai/api/v1")
    return _call_gemini_direct(pil_imgs_with_labels, prompt, api_key, model_name)


def _parse_json_response(raw):
    """Strip markdown fences and parse JSON. Tolerant of leading/trailing prose."""
    text = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    # Find the first [ and last ] to bracket the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        text = text[start: end + 1]
    return json.loads(text)


# =============================================================================
# Image utilities
# =============================================================================


def _image_tensor_frame_to_pil(image_tensor, frame_idx):
    """[B,H,W,3] frame → PIL.Image (RGB)."""
    fi = max(0, min(image_tensor.shape[0] - 1, frame_idx))
    arr = (image_tensor[fi].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _draw_set_of_mark(pil_img, anchors_with_pixel_coords):
    """Draw Set-of-Mark overlay: numbered colored circles + name labels.

    anchors_with_pixel_coords: list of {name, id, x_px, y_px} (pixel coords in
    THIS image, already mapped from VLM's 0-1000 to actual frame size).
    Returns a NEW PIL.Image (does not mutate input).
    """
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for a in anchors_with_pixel_coords:
        if a["x_px"] < 0 or a["y_px"] < 0:
            continue
        color = _PALETTE_RGB[a["id"] % len(_PALETTE_RGB)]
        x, y = int(round(a["x_px"])), int(round(a["y_px"]))
        # Outer ring + inner dot
        draw.ellipse([x - 9, y - 9, x + 9, y + 9], outline=color, width=2)
        draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
        # Label background + text
        label = f"{a['id']}:{a['name']}"
        try:
            tw, th = draw.textbbox((0, 0), label, font=font)[2:4]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
        tx, ty = x + 11, max(0, y - th - 2)
        draw.rectangle([tx - 2, ty - 1, tx + tw + 2, ty + th + 1], fill=(0, 0, 0))
        draw.text((tx, ty), label, fill=color, font=font)
    return img


# =============================================================================
# Coord conversion + Shi-Tomasi snap
# =============================================================================


def _normalize_to_pixels(x_norm, y_norm, W, H):
    """Convert Gemini's 0-1000 (or 0-1) to pixel coords. Tolerant of overshoot."""
    if x_norm > 1.5:
        x_px = x_norm / 1000.0 * W
    else:
        x_px = x_norm * W
    if y_norm > 1.5:
        y_px = y_norm / 1000.0 * H
    else:
        y_px = y_norm * H
    return float(x_px), float(y_px)


def _compute_radius_px(crop_h, crop_w, frac, max_cap):
    """Resolution-aware trackability radius."""
    r = max(8, int(round(min(crop_h, crop_w) * frac)))
    return min(r, max_cap)


def _boundary_safe_clamp(x_px, y_px, W, H, radius):
    """Clamp to [radius, W-radius] × [radius, H-radius]. Returns (x, y, clamp_dist)."""
    cx = max(radius, min(W - radius - 1, x_px))
    cy = max(radius, min(H - radius - 1, y_px))
    dist = float(np.hypot(cx - x_px, cy - y_px))
    return cx, cy, dist


def _shi_tomasi_snap(gray_frame, x_px, y_px, radius_px,
                      max_corners=20, alpha_dist=0.5):
    """Find the best trackable corner near (x_px, y_px) within radius_px.

    Returns (snapped_x, snapped_y, found_corner: bool, sub_pixel_refined: bool).

    If no qualifying corner found within radius, returns the input point with
    cv2.cornerSubPix applied (sub-pixel refinement on the VLM point itself).
    Falls back gracefully if even cornerSubPix can't run.
    """
    H, W = gray_frame.shape
    # Build mask for goodFeaturesToTrack restricted to the search radius
    x0 = max(0, int(round(x_px - radius_px)))
    y0 = max(0, int(round(y_px - radius_px)))
    x1 = min(W, int(round(x_px + radius_px + 1)))
    y1 = min(H, int(round(y_px + radius_px + 1)))
    if x1 - x0 <= 2 or y1 - y0 <= 2:
        # Window collapsed — return as-is
        return x_px, y_px, False, False

    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    corners = cv2.goodFeaturesToTrack(
        gray_frame, maxCorners=max_corners, qualityLevel=0.01,
        minDistance=2, mask=mask, blockSize=5, useHarrisDetector=False,
    )
    if corners is not None and len(corners) > 0:
        # Rank by corner_quality / (1 + alpha * dist_to_VLM_point).
        # v1.2 fix (multi-AI review): use real Shi-Tomasi minimum-eigenvalue
        # quality from cv2.cornerMinEigenVal so the ranking actually trades
        # quality vs proximity, not just proximity (v1.0 used uniform weight).
        # Compute eigenvalue map ONCE on the search ROI (cheap).
        roi = gray_frame[y0:y1, x0:x1]
        try:
            eig_roi = cv2.cornerMinEigenVal(roi, blockSize=5)
        except cv2.error:
            eig_roi = None
        best_pt = None
        best_score = -float("inf")
        for c in corners:
            cx, cy = float(c[0][0]), float(c[0][1])
            dist = float(np.hypot(cx - x_px, cy - y_px))
            # Look up corner quality at this pixel in the eigenvalue map.
            # Fall back to 1.0 if the cornerMinEigenVal call failed.
            if eig_roi is not None:
                ix = int(np.clip(round(cx - x0), 0, eig_roi.shape[1] - 1))
                iy = int(np.clip(round(cy - y0), 0, eig_roi.shape[0] - 1))
                quality = float(eig_roi[iy, ix])
            else:
                quality = 1.0
            score = quality / (1.0 + alpha_dist * dist / max(1.0, radius_px))
            if score > best_score:
                best_score = score
                best_pt = (cx, cy)
        if best_pt is not None:
            # Sub-pixel refine on the snapped corner
            try:
                pt_arr = np.array([[best_pt]], dtype=np.float32)
                cv2.cornerSubPix(
                    gray_frame, pt_arr,
                    winSize=(5, 5), zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                              20, 0.03),
                )
                return float(pt_arr[0, 0, 0]), float(pt_arr[0, 0, 1]), True, True
            except cv2.error:
                return best_pt[0], best_pt[1], True, False
    # No qualifying corner — fall back to VLM point with sub-pixel refinement
    try:
        sub_radius = max(2, min(8, radius_px // 2))
        # Need a window that fits — clamp tightly
        if (x_px - sub_radius < 1 or x_px + sub_radius > W - 1 or
                y_px - sub_radius < 1 or y_px + sub_radius > H - 1):
            return x_px, y_px, False, False
        pt_arr = np.array([[(x_px, y_px)]], dtype=np.float32)
        cv2.cornerSubPix(
            gray_frame, pt_arr,
            winSize=(sub_radius, sub_radius), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                      10, 0.05),
        )
        return float(pt_arr[0, 0, 0]), float(pt_arr[0, 0, 1]), False, True
    except cv2.error:
        return x_px, y_px, False, False


# =============================================================================
# Prompt builders
# =============================================================================


def _build_source_prompt(landmark_list):
    names_json = json.dumps(landmark_list)
    return f"""You are selecting seed points for dense visual point tracking, NOT segmentation.

Input: K frames from a single cropped actor video. Each frame is labeled with its frame index t. The text "Frame {{t}}:" appears immediately before each image.

Coordinate system: integer (x, y) normalized 0-1000 over each individual frame's width and height. (0, 0) is the top-left of that specific frame; (1000, 1000) is the bottom-right. Each landmark's coordinates apply to whichever frame's t you chose.

Task: For each requested landmark name, return exactly one anchor:
- choose the frame where that landmark is BOTH most visible AND most trackable
- prefer points that will REMAIN inside frame and visible for the next several frames
- prefer persistent geometric features (seams, corners, creases) over anatomically-exact centers

Trackable means:
- corner-like, high-local-texture, geometrically stable feature
- avoid: flat skin, cast shadows, specular highlights, motion-blurred regions, hair wisps, transient cloth wrinkles, frame-edge points (within 5% of any edge)

{ANCHOR_TYPE_DEFINITIONS}

Side convention: "L_" and "R_" are the actor's anatomical left/right, NOT the viewer's. Use body-orientation cues (face-front vs face-away, navel position, hand asymmetry) -- NEVER assume mirror.

Strict output rules:
- Return ONLY a JSON array. No markdown fences. No prose. No extra keys.
- Use exactly these names, exactly once each, in this order:
  {names_json}
- Each item: {{"name": "...", "x": int (0-1000), "y": int (0-1000), "t": int, "anchor_type": "..."}}
- t MUST be one of the provided frame indices.
- anchor_type MUST be one of the allowed strings listed above.
- If a landmark is occluded or not reliably trackable in ALL provided frames: {{"name":"...", "x":-1, "y":-1, "t":-1, "anchor_type":"missing"}}
"""


def _build_render_prompt(source_anchors_payload):
    payload_json = json.dumps(source_anchors_payload, indent=2)
    return f"""You are matching tracking anchors from a source video onto a rendered video of the same scene.
Same actor identity, same scene composition. Different lighting / style / exact pose.

CRITICAL: Your job is to find the SAME PHYSICAL FEATURE the source picked, NOT a generic anatomical center.
A 30-pixel discrepancy between the source's anchor and your render anchor will ruin a downstream similarity-transform fit.

Input:
- A Set-of-Mark reference image labeled "Source reference:" — ONE source frame with all source anchors plotted as colored numbered circles. Each anchor's id and name are labeled next to it.
- K frames from the cropped render video, labeled "Frame {{t}}:" immediately before each image.
- The ordered list of source anchors below, each carrying: name, id, the source's chosen frame, the source's normalized (x_src, y_src), and an anchor_type describing the kind of local feature the source picked:

{payload_json}

Coordinate system: integer (x, y) normalized 0-1000 over each render frame's width and height, top-left = (0, 0).

For each provided source anchor, return one render anchor that:
1. Identifies the ANALOGOUS physical feature on the same body part on the actor in the render video
   - if source picked "outer_contour_corner" of L_shoulder seam, find the analogous outer-contour corner on the render's L_shoulder seam -- NOT the inner clavicle hinge
   - if source picked "seam_intersection" on the chest, find the same intersection of the same garment seams on the render's chest
2. Is on the actor's same anatomical side
   - if side is ambiguous (near-frontal pose, mirrored framing), return x=-1,y=-1,t=-1 rather than guessing
   - NEVER swap left/right to satisfy visibility -- return missing instead
3. Is most visible and trackable in the chosen render frame t
4. Avoids frame edges, blur, and textureless regions (same trackability criteria as source)

If the render diverges topologically from the source (different garment, body part hidden by occluder), choose the nearest analogous stable feature on the same body part. If no such analog exists, return missing.

{ANCHOR_TYPE_DEFINITIONS}

Strict output rules:
- Return ONLY a JSON array. No markdown fences. No prose. No extra keys.
- Use EXACTLY the same names in EXACTLY the same order as the source list. No additions, no removals, no renames. Match each entry by id.
- Each item: {{"name": "...", "x": int (0-1000), "y": int (0-1000), "t": int, "anchor_type": "..."}}
- t MUST be one of the provided render frame indices.
- For occluded / ambiguous-side / topologically-absent: {{"name":"...", "x":-1, "y":-1, "t":-1, "anchor_type":"missing"}}
- DO NOT invent landmarks. DO NOT change the body side. DO NOT mirror.
"""


# =============================================================================
# Schema validation + retry
# =============================================================================


def _validate_response(parsed, expected_names, kf_indices, label):
    """Verify parsed list matches contract. Returns (ok, error_message).

    v1.2 (multi-AI review tightening):
    - x/y in [0, 1000] for non-missing entries (catches model hallucinations)
    - sentinel-coherence: x=y=t=-1 iff anchor_type='missing' (no half-states)
    """
    if not isinstance(parsed, list):
        return False, f"{label} response is not a JSON list (got {type(parsed).__name__})"
    if len(parsed) != len(expected_names):
        return False, (f"{label} response length {len(parsed)} != expected "
                       f"{len(expected_names)}")
    seen_names = set()
    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            return False, f"{label} item {i} is not a dict"
        for k in ("name", "x", "y", "t", "anchor_type"):
            if k not in item:
                return False, f"{label} item {i} missing key '{k}'"
        if item["name"] != expected_names[i]:
            return False, (f"{label} item {i}: name '{item['name']}' != "
                           f"expected '{expected_names[i]}'")
        if item["name"] in seen_names:
            return False, f"{label} item {i}: duplicate name '{item['name']}'"
        seen_names.add(item["name"])
        try:
            x = int(item["x"])
            y = int(item["y"])
            t = int(item["t"])
        except (TypeError, ValueError) as e:
            return False, f"{label} item {i} non-int coord/t: {e}"
        if t != -1 and t not in kf_indices:
            return False, (f"{label} item {i} t={t} not in keyframe indices "
                           f"{kf_indices}")
        if item["anchor_type"] not in ALLOWED_ANCHOR_TYPES:
            return False, (f"{label} item {i} anchor_type "
                           f"{item['anchor_type']!r} not in {ALLOWED_ANCHOR_TYPES}")
        # v1.2 sentinel coherence: missing iff (x==-1 AND y==-1 AND t==-1)
        is_missing_atype = item["anchor_type"] == "missing"
        is_sentinel_coords = (x == -1 and y == -1 and t == -1)
        if is_missing_atype != is_sentinel_coords:
            return False, (
                f"{label} item {i}: sentinel/anchor_type mismatch "
                f"(anchor_type={item['anchor_type']!r}, x={x}, y={y}, t={t}). "
                f"Either return all sentinels with anchor_type='missing', or all "
                f"valid coords with a non-missing anchor_type."
            )
        # v1.2 coord-range check (skip for sentinel rows)
        if not is_sentinel_coords:
            if not (0 <= x <= 1000):
                return False, (f"{label} item {i}: x={x} out of normalized range [0,1000] "
                               f"(set x=-1 with anchor_type='missing' if landmark is unfindable)")
            if not (0 <= y <= 1000):
                return False, (f"{label} item {i}: y={y} out of normalized range [0,1000] "
                               f"(set y=-1 with anchor_type='missing' if landmark is unfindable)")
    return True, "ok"


# =============================================================================
# Main node
# =============================================================================


class NV_VLMLandmarkCorresponder:
    """Sequential Gemini-driven landmark detection + Shi-Tomasi snap.

    v1 supports SINGLE-ACTOR crops only. Multi-person/crowd behavior is undefined.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_images": ("IMAGE", {
                    "tooltip": (
                        "Cropped source video frames [B,H,W,3]. v1 requires a "
                        "single-actor crop; multi-person scenes are out of scope."
                    ),
                }),
                "render_images": ("IMAGE", {
                    "tooltip": (
                        "Cropped render video frames [B,H,W,3]. May have a different "
                        "crop window from source — coord conversion uses each video's "
                        "own dimensions."
                    ),
                }),
                "keyframe_indices_source": ("STRING", {
                    "default": "[0]",
                    "multiline": False,
                    "tooltip": (
                        "Frame indices to query Gemini on for the source video. Accepts "
                        "many syntaxes (mix freely with commas, semicolons, whitespace):\n"
                        "  JSON:           [0, 24, 47]\n"
                        "  CSV:            0, 24, 47\n"
                        "  Whitespace:     0 24 47\n"
                        "  Single int:     47\n"
                        "  Range:          0-10  (inclusive both ends)\n"
                        "  Range w/ step:  0-100:5\n"
                        "  Negative:       -1   (last frame), -5--1 (last 5)\n"
                        "  Mixed:          0, 5-10, 47, 80-100:5, -1\n"
                        "Indices are auto-clamped to the video's length, deduplicated, "
                        "and sorted. Wire from NV_KeyframeSampler.keyframe_indices for "
                        "auto-selection, or type your own override."
                    ),
                }),
                "keyframe_indices_render": ("STRING", {
                    "default": "[0]",
                    "multiline": False,
                    "tooltip": (
                        "Frame indices for the render video — same flexible syntax as "
                        "keyframe_indices_source (JSON / CSV / range / negative / mixed). "
                        "Render motion peaks are usually independent of source, so wire "
                        "from a SECOND NV_KeyframeSampler — or type your own. The two "
                        "lists do NOT have to overlap; Gemini picks the best frame per "
                        "landmark on each side independently."
                    ),
                }),
                "landmark_vocab": ("STRING", {
                    "default": DEFAULT_VOCAB,
                    "multiline": False,
                    "tooltip": (
                        "Comma-separated landmark names. Default is torso-centric for "
                        "similarity-transform fitting on full-body shots. For upper-"
                        "body shots, drop knees: head,neck,L_shoulder,R_shoulder,"
                        "L_elbow,R_elbow. For head-and-shoulders: head,neck,"
                        "L_shoulder,R_shoulder."
                    ),
                }),
            },
            "optional": {
                "api": ("AVM_API", {
                    "tooltip": (
                        "Optional AVM_API config from AVM API Config node. If "
                        "unwired, falls back to GEMINI_API_KEY env var or the "
                        "ui_api_key field below."
                    ),
                }),
                "ui_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Last-resort Gemini API key entry (least secure — avoid in "
                        "shared workflows). Only used when api is unwired and "
                        "GEMINI_API_KEY env var is unset."
                    ),
                }),
                "trackability_radius_frac": ("FLOAT", {
                    "default": 0.02, "min": 0.005, "max": 0.1, "step": 0.005,
                    "tooltip": (
                        "Shi-Tomasi search radius as fraction of min(crop_w, crop_h). "
                        "0.02 = ~20px on 1024² crops, ~5px on 256² crops (floored "
                        "to 8px minimum)."
                    ),
                }),
                "trackability_max_radius_px": ("INT", {
                    "default": 48, "min": 8, "max": 128, "step": 4,
                    "tooltip": (
                        "Hard cap on Shi-Tomasi radius. Prevents the snap from "
                        "crossing body-part boundaries on huge crops."
                    ),
                }),
                "verbose_debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": (
                        "Print detailed VLM call + snap diagnostics to console. "
                        "Includes raw VLM responses, per-anchor snap distances, "
                        "and confidence breakdowns."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "source_tracking_points", "render_tracking_points",
        "correspondence_report", "raw_vlm_responses",
    )
    FUNCTION = "correspond"
    CATEGORY = "NV_Utils/Tracking"
    DESCRIPTION = (
        "Auto-detect matched tracking anchors on source + render via two sequential "
        "Gemini calls (source-first, render-conditioned-on-source) plus Shi-Tomasi "
        "trackability snap. Replaces manual NV_PointPicker dual-click for the "
        "PreCompWarp pipeline. Outputs paired tracking_points JSONs in NV_PointPicker "
        "format. v1 supports single-actor crops only."
    )

    def correspond(self, source_images, render_images,
                   keyframe_indices_source, keyframe_indices_render,
                   landmark_vocab,
                   api=None, ui_api_key="",
                   trackability_radius_frac=0.02,
                   trackability_max_radius_px=48,
                   verbose_debug=False):
        t_start = time.perf_counter()

        # --- Resolve config -------------------------------------------------
        api_key, model_name, provider = _resolve_api_key(api, ui_api_key)
        base_url = api.get("base_url", "https://openrouter.ai/api/v1") if api else "https://openrouter.ai/api/v1"

        # --- Parse vocab + keyframes ---------------------------------------
        vocab_list = [s.strip() for s in landmark_vocab.split(",") if s.strip()]
        if len(vocab_list) < 2:
            raise ValueError(
                f"[NV_VLMLandmarkCorresponder] landmark_vocab must have at least 2 names; got {vocab_list}"
            )

        T_src, H_src, W_src = source_images.shape[:3]
        T_ren, H_ren, W_ren = render_images.shape[:3]

        # Flexible parser handles JSON / CSV / ranges / negatives / mixed forms +
        # bounds-clamps, dedups, and sorts. See _parse_keyframe_spec docstring
        # for the full syntax list.
        kf_src = _parse_keyframe_spec(keyframe_indices_source, T_src,
                                       "keyframe_indices_source")
        kf_ren = _parse_keyframe_spec(keyframe_indices_render, T_ren,
                                       "keyframe_indices_render")

        if verbose_debug:
            print(f"[NV_VLMLandmarkCorresponder] vocab={vocab_list}")
            print(f"[NV_VLMLandmarkCorresponder] source: T={T_src}, H={H_src}, W={W_src}, kf={kf_src}")
            print(f"[NV_VLMLandmarkCorresponder] render: T={T_ren}, H={H_ren}, W={W_ren}, kf={kf_ren}")
            print(f"[NV_VLMLandmarkCorresponder] model={model_name}, provider={provider}")

        # --- Step 1: Source Gemini call ------------------------------------
        src_imgs_with_labels = []
        for t in kf_src:
            src_imgs_with_labels.append((f"Frame {t}:", _image_tensor_frame_to_pil(source_images, t)))
        source_prompt = _build_source_prompt(vocab_list)
        src_raw, src_parsed = self._call_with_retry(
            src_imgs_with_labels, source_prompt, vocab_list, kf_src,
            api_key, model_name, provider, base_url, "source", verbose_debug,
        )

        # --- Step 2: Source coord conversion + boundary clamp + Shi-Tomasi snap ---
        src_radius_px = _compute_radius_px(H_src, W_src, trackability_radius_frac,
                                           trackability_max_radius_px)
        src_gray = self._make_gray_frames(source_images)
        src_anchors = []
        src_snap_meta = []
        for i, item in enumerate(src_parsed):
            anchor = self._snap_one(
                item, i, src_gray, kf_src, H_src, W_src, src_radius_px,
                "source", verbose_debug,
            )
            src_anchors.append(anchor[0])
            src_snap_meta.append(anchor[1])

        # --- Step 3: Build Set-of-Mark reference image ---------------------
        som_image = self._build_set_of_mark_image(source_images, src_anchors, src_snap_meta, kf_src)

        # --- Step 4: Render Gemini call ------------------------------------
        ren_imgs_with_labels = [("Source reference (Set-of-Mark — colored numbered "
                                  "circles show source anchors):", som_image)]
        for t in kf_ren:
            ren_imgs_with_labels.append((f"Frame {t}:", _image_tensor_frame_to_pil(render_images, t)))
        # Build source payload (use VLM-original 0-1000 coords, not snapped pixel coords,
        # so the render call sees what Gemini originally returned)
        source_payload = []
        for i, (a, m) in enumerate(zip(src_anchors, src_snap_meta)):
            source_payload.append({
                "name": a["name"],
                "id": i,
                "source_t": int(a["t"]) if a["t"] >= 0 else -1,
                "source_x_norm": int(m["vlm_x_norm"]) if m["vlm_x_norm"] is not None else -1,
                "source_y_norm": int(m["vlm_y_norm"]) if m["vlm_y_norm"] is not None else -1,
                "anchor_type": a.get("anchor_type", "missing"),
            })
        render_prompt = _build_render_prompt(source_payload)
        ren_raw, ren_parsed = self._call_with_retry(
            ren_imgs_with_labels, render_prompt, vocab_list, kf_ren,
            api_key, model_name, provider, base_url, "render", verbose_debug,
        )

        # --- Step 5: Render coord conversion + clamp + snap ----------------
        ren_radius_px = _compute_radius_px(H_ren, W_ren, trackability_radius_frac,
                                           trackability_max_radius_px)
        ren_gray = self._make_gray_frames(render_images)
        ren_anchors = []
        ren_snap_meta = []
        for i, item in enumerate(ren_parsed):
            anchor = self._snap_one(
                item, i, ren_gray, kf_ren, H_ren, W_ren, ren_radius_px,
                "render", verbose_debug,
            )
            ren_anchors.append(anchor[0])
            ren_snap_meta.append(anchor[1])

        # --- Step 6: Final pairing validation ------------------------------
        if len(src_anchors) != len(ren_anchors):
            raise RuntimeError(
                f"[NV_VLMLandmarkCorresponder] post-snap length mismatch: "
                f"src={len(src_anchors)}, ren={len(ren_anchors)}"
            )
        for i, (s, r) in enumerate(zip(src_anchors, ren_anchors)):
            if s["name"] != r["name"]:
                raise RuntimeError(
                    f"[NV_VLMLandmarkCorresponder] post-snap name mismatch at idx {i}: "
                    f"src={s['name']!r}, ren={r['name']!r}"
                )

        # --- Step 7: Build outputs -----------------------------------------
        # Final tracking_points JSON: existing 4-field shape (NV_PointPicker compatible)
        # PLUS a 'name' key (NV_PointPicker tolerates extra keys).
        src_tracking = json.dumps([
            {"x": int(round(a["x"])), "y": int(round(a["y"])),
             "t": int(a["t"]), "name": a["name"]}
            for a in src_anchors
        ])
        ren_tracking = json.dumps([
            {"x": int(round(a["x"])), "y": int(round(a["y"])),
             "t": int(a["t"]), "name": a["name"]}
            for a in ren_anchors
        ])

        # --- Build correspondence_report (anchor_type + confidence breakdown) ---
        report_entries = []
        for i, (a_s, m_s, a_r, m_r) in enumerate(zip(src_anchors, src_snap_meta,
                                                       ren_anchors, ren_snap_meta)):
            confidence = m_s["confidence"] * m_r["confidence"]
            report_entries.append({
                "id": i,
                "name": a_s["name"],
                "anchor_type_source": a_s.get("anchor_type", "missing"),
                "anchor_type_render": a_r.get("anchor_type", "missing"),
                "source": {
                    "x": int(round(a_s["x"])), "y": int(round(a_s["y"])), "t": int(a_s["t"]),
                    "vlm_x_norm": m_s["vlm_x_norm"], "vlm_y_norm": m_s["vlm_y_norm"],
                    "clamp_distance_px": m_s["clamp_distance_px"],
                    "snap_found_corner": m_s["snap_found_corner"],
                    "snap_distance_px": m_s["snap_distance_px"],
                    "snap_skipped_small_crop": m_s["snap_skipped_small_crop"],
                    "clamp_factor": m_s["clamp_factor"],
                    "snap_factor": m_s["snap_factor"],
                    "geom_factor": m_s["geom_factor"],
                    "confidence": m_s["confidence"],
                },
                "render": {
                    "x": int(round(a_r["x"])), "y": int(round(a_r["y"])), "t": int(a_r["t"]),
                    "vlm_x_norm": m_r["vlm_x_norm"], "vlm_y_norm": m_r["vlm_y_norm"],
                    "clamp_distance_px": m_r["clamp_distance_px"],
                    "snap_found_corner": m_r["snap_found_corner"],
                    "snap_distance_px": m_r["snap_distance_px"],
                    "snap_skipped_small_crop": m_r["snap_skipped_small_crop"],
                    "clamp_factor": m_r["clamp_factor"],
                    "snap_factor": m_r["snap_factor"],
                    "geom_factor": m_r["geom_factor"],
                    "confidence": m_r["confidence"],
                },
                "joint_confidence": confidence,
            })

        elapsed = time.perf_counter() - t_start
        correspondence_report = json.dumps({
            "elapsed_seconds": round(elapsed, 2),
            "model": model_name,
            "provider": provider,
            "vocab": vocab_list,
            "source_keyframes": kf_src,
            "render_keyframes": kf_ren,
            "source_radius_px": src_radius_px,
            "render_radius_px": ren_radius_px,
            "anchors": report_entries,
            "notes": (
                "confidence is heuristic and treated as a relative ordering signal "
                "within a single run, not an absolute probability. It is NOT consumed "
                "by NV_CoTrackerTrajectoriesJSON."
            ),
        }, indent=2)

        raw_vlm_responses = json.dumps({
            "source_raw": src_raw,
            "render_raw": ren_raw,
        }, indent=2)

        print(f"[NV_VLMLandmarkCorresponder] done in {elapsed:.1f}s "
              f"({len(src_anchors)} anchors paired)")

        return (src_tracking, ren_tracking, correspondence_report, raw_vlm_responses)

    def _call_with_retry(self, imgs_with_labels, prompt, vocab_list, kf_indices,
                         api_key, model_name, provider, base_url, label, verbose):
        """Call Gemini with one retry on schema violation. Returns (raw, parsed)."""
        for attempt in (1, 2):
            t0 = time.perf_counter()
            try:
                raw = _call_vlm(imgs_with_labels, prompt, api_key, model_name,
                                 provider, base_url)
            except Exception as e:
                raise RuntimeError(f"[NV_VLMLandmarkCorresponder] {label} VLM call failed: {e}")
            elapsed = time.perf_counter() - t0
            if verbose:
                print(f"[NV_VLMLandmarkCorresponder] {label} VLM call attempt {attempt} "
                      f"({elapsed:.1f}s) raw response:")
                print(raw[:500] + ("..." if len(raw) > 500 else ""))
            try:
                parsed = _parse_json_response(raw)
            except json.JSONDecodeError as e:
                if attempt == 1:
                    prompt = (prompt + "\n\nIMPORTANT: previous response could not be parsed as JSON. "
                              "Return ONLY the JSON array, no markdown fences, no prose.")
                    continue
                raise RuntimeError(f"[NV_VLMLandmarkCorresponder] {label} response not parseable as JSON "
                                    f"after retry: {e}\nRaw: {raw}")
            ok, err = _validate_response(parsed, vocab_list, kf_indices, label)
            if ok:
                return raw, parsed
            if attempt == 1:
                print(f"[NV_VLMLandmarkCorresponder] {label} schema violation: {err} — retrying")
                prompt = (prompt + f"\n\nIMPORTANT: previous response had this problem: {err}. "
                          "Return ONLY a JSON array of length "
                          f"{len(vocab_list)} with names {json.dumps(vocab_list)} "
                          "in that exact order, each item having keys name/x/y/t/anchor_type.")
                continue
            raise RuntimeError(f"[NV_VLMLandmarkCorresponder] {label} schema validation failed "
                                f"after retry: {err}\nRaw: {raw}")
        raise RuntimeError("unreachable")

    def _snap_one(self, item, i, gray_frames, kf_indices, H, W, radius_px,
                   label, verbose):
        """Process one Gemini-returned anchor through coord-convert + clamp + snap.

        Returns (anchor_dict, meta_dict) where:
        - anchor_dict has the final pixel coords + name + t + anchor_type
        - meta_dict has confidence sub-scores + diagnostics
        """
        anchor = {
            "name": item["name"],
            "anchor_type": item.get("anchor_type", "missing"),
            "t": int(item["t"]),
            "x": -1.0,
            "y": -1.0,
        }
        meta = {
            "vlm_x_norm": int(item["x"]) if item["x"] != -1 else None,
            "vlm_y_norm": int(item["y"]) if item["y"] != -1 else None,
            "clamp_distance_px": 0.0,
            "snap_found_corner": False,
            "snap_distance_px": 0.0,
            "snap_skipped_small_crop": False,
            "clamp_factor": 1.0,
            "snap_factor": 1.0,
            "geom_factor": 1.0,
            "confidence": 0.0,
        }

        # Missing anchor → keep sentinels
        if item["x"] < 0 or item["y"] < 0 or item["t"] < 0 or item["anchor_type"] == "missing":
            meta["clamp_factor"] = 0.0
            meta["snap_factor"] = 0.0
            meta["geom_factor"] = 0.0
            meta["confidence"] = 0.0
            anchor["t"] = -1
            return anchor, meta

        # Convert 0-1000 to pixel coords
        x_px, y_px = _normalize_to_pixels(float(item["x"]), float(item["y"]), W, H)

        # Degenerate small crop check: if 2*radius >= min(H,W), skip Shi-Tomasi
        # but STILL run cv2.cornerSubPix on a tiny window — per v1.2 spec, fall-
        # back path keeps sub-pixel refinement so CoTracker3 doesn't pay the
        # integer-truncation cost. (Multi-AI review #1 BLOCKER on both reviewers.)
        if 2 * radius_px >= min(H, W):
            # Boundary-safe clamp to [1, W-1] × [1, H-1]
            x_safe = max(1.0, min(W - 2.0, x_px))
            y_safe = max(1.0, min(H - 2.0, y_px))
            meta["clamp_distance_px"] = float(np.hypot(x_safe - x_px, y_safe - y_px))
            meta["snap_skipped_small_crop"] = True
            anchor["x"] = x_safe
            anchor["y"] = y_safe
            meta["clamp_factor"] = max(0.0, 1.0 - min(1.0, meta["clamp_distance_px"] / 40.0))
            # Default snap_factor for raw VLM point (no refinement attempt yet)
            meta["snap_factor"] = 0.4

            # Attempt cv2.cornerSubPix on the tiniest window that always fits
            t_idx = int(item["t"])
            sub_pixel_ok = False
            if 0 <= t_idx < gray_frames.shape[0]:
                # Use winSize=(2,2) — actually the patch is (2*win+1) = 5x5,
                # which is the smallest cv2 will accept. Fall back gracefully on
                # cv2.error if the position is too close to a frame edge.
                try:
                    if (x_safe >= 3 and x_safe <= W - 4 and
                            y_safe >= 3 and y_safe <= H - 4):
                        pt_arr = np.array([[(float(x_safe), float(y_safe))]],
                                          dtype=np.float32)
                        cv2.cornerSubPix(
                            gray_frames[t_idx], pt_arr,
                            winSize=(2, 2), zeroZone=(-1, -1),
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                      10, 0.05),
                        )
                        anchor["x"] = float(pt_arr[0, 0, 0])
                        anchor["y"] = float(pt_arr[0, 0, 1])
                        meta["snap_factor"] = 0.5  # sub-pixel refinement succeeded
                        sub_pixel_ok = True
                except cv2.error:
                    pass

            meta["confidence"] = meta["clamp_factor"] * meta["snap_factor"] * meta["geom_factor"]
            if verbose:
                print(f"[NV_VLMLandmarkCorresponder]   {label} #{i} {item['name']}: "
                      f"DEGENERATE small crop (2*r={2*radius_px} >= min(H,W)={min(H,W)}) — "
                      f"Shi-Tomasi skipped, sub_pixel={sub_pixel_ok}, "
                      f"final=({anchor['x']:.2f}, {anchor['y']:.2f}), "
                      f"conf={meta['confidence']:.2f}")
            return anchor, meta

        # Boundary-safe clamp BEFORE Shi-Tomasi (prevents OOB crash)
        x_clamped, y_clamped, clamp_dist = _boundary_safe_clamp(x_px, y_px, W, H, radius_px)
        meta["clamp_distance_px"] = clamp_dist

        if clamp_dist > 40:
            # Severe clamp — VLM coord was way off-frame, low confidence
            meta["clamp_factor"] = 0.0
        else:
            meta["clamp_factor"] = max(0.0, 1.0 - min(1.0, clamp_dist / 40.0))

        # Shi-Tomasi snap on the t-frame
        t_idx = int(item["t"])
        # Map t_idx (which IS in kf_indices per validation) to gray_frames index.
        # gray_frames is full-length [T,H,W], so t_idx works directly.
        if t_idx < 0 or t_idx >= gray_frames.shape[0]:
            # Shouldn't happen given validation, but be defensive
            anchor["x"] = x_clamped
            anchor["y"] = y_clamped
            meta["snap_factor"] = 0.5
            meta["confidence"] = meta["clamp_factor"] * meta["snap_factor"] * meta["geom_factor"]
            return anchor, meta

        snapped_x, snapped_y, found_corner, sub_pixel = _shi_tomasi_snap(
            gray_frames[t_idx], x_clamped, y_clamped, radius_px,
        )
        meta["snap_found_corner"] = found_corner
        meta["snap_distance_px"] = float(np.hypot(snapped_x - x_clamped, snapped_y - y_clamped))

        if found_corner:
            meta["snap_factor"] = 1.0
        elif sub_pixel:
            meta["snap_factor"] = 0.5
        else:
            meta["snap_factor"] = 0.4  # raw VLM point, no refinement possible

        anchor["x"] = snapped_x
        anchor["y"] = snapped_y
        meta["confidence"] = meta["clamp_factor"] * meta["snap_factor"] * meta["geom_factor"]

        if verbose:
            print(f"[NV_VLMLandmarkCorresponder]   {label} #{i} {item['name']}: "
                  f"VLM ({item['x']},{item['y']})@t={t_idx} -> px ({x_px:.1f},{y_px:.1f}) "
                  f"clamp={clamp_dist:.1f} snap={meta['snap_distance_px']:.1f} "
                  f"found_corner={found_corner} subpix={sub_pixel} "
                  f"final=({snapped_x:.1f},{snapped_y:.1f}) "
                  f"conf={meta['confidence']:.2f}")
        return anchor, meta

    @staticmethod
    def _make_gray_frames(image_tensor):
        """[B,H,W,3] in [0,1] → [B,H,W] uint8 grayscale numpy."""
        gray = (image_tensor[..., 0] * 0.299
                + image_tensor[..., 1] * 0.587
                + image_tensor[..., 2] * 0.114) * 255.0
        return gray.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

    def _build_set_of_mark_image(self, source_images, src_anchors, src_snap_meta, kf_src):
        """Pick the keyframe with most in-frame anchors; plot all anchors on it.

        For anchors whose source_t differs from the chosen base frame, we still
        plot them at their (x, y) — Gemini knows from the JSON payload they
        were chosen on a different frame; the visual reference is for spatial
        identity, not temporal."""
        # Count in-frame anchors per keyframe (anchors whose t matches that keyframe)
        kf_counts = {t: 0 for t in kf_src}
        for a in src_anchors:
            if a["t"] in kf_counts and a["x"] >= 0 and a["y"] >= 0:
                kf_counts[a["t"]] += 1
        # Pick the keyframe with most in-frame anchors; tie → latest t
        best_t = max(kf_src, key=lambda t: (kf_counts.get(t, 0), t))

        base_pil = _image_tensor_frame_to_pil(source_images, best_t)
        anchors_for_draw = []
        for i, a in enumerate(src_anchors):
            if a["x"] < 0 or a["y"] < 0:
                continue
            anchors_for_draw.append({
                "id": i,
                "name": a["name"],
                "x_px": a["x"],
                "y_px": a["y"],
            })
        return _draw_set_of_mark(base_pil, anchors_for_draw)


NODE_CLASS_MAPPINGS = {
    "NV_VLMLandmarkCorresponder": NV_VLMLandmarkCorresponder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VLMLandmarkCorresponder": "NV VLM Landmark Corresponder",
}
