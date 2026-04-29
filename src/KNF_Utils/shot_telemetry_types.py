"""Shot telemetry — type constants, verdict rubric, controlled vocabularies.

Single source of truth for the agent's input contract. Bump SCHEMA_VERSION
on any breaking change to record shape; bump RUBRIC_VERSION on any change
to verdict-score semantics.
"""

SCHEMA_VERSION = "1.0"
RUBRIC_VERSION = "v1.0"

# Controlled vocabulary — tags the operator may attach via verdict input.
# Any non-empty tag here is a binary VETO for the agent (don't recommend the
# params that produced this output, regardless of metric scores).
DISQUALIFYING_ARTIFACT_TAGS = [
    "rainbow_checkerboard",   # init-noise covariance failure mode
    "ghost_edge",             # highband reinjection at high denoise
    "matte_face",             # over-aggressive color correction
    "identity_drift",         # wrong-person output
    "bg_bleed",               # background colors leaking into face
    "pulsing_blur",           # texture-band sharpness instability
    "head_jitter",            # mask-driven motion artifacts
    "seam_flash",             # visible chunk boundary
    "vae_fizz",               # synthetic high-freq noise from VAE
    "oom",                    # render failed with OOM
    "render_error",           # render failed with non-OOM error
]

# Rubric — locked semantics for verdict scores. Update RUBRIC_VERSION when
# any of these change so the agent can filter records taken under prior
# rubrics out of its training context.
VERDICT_RUBRIC = """
verdict_overall (1-5):
  5 = ship — zero distracting artifacts, identity preserved, temporally clean
  4 = good — minor flaws survivable in compositing
  3 = mixed — some shots worked, others didn't; ambiguous
  2 = poor — visible artifacts requiring rework
  1 = unusable — wrong person, broken motion, or catastrophic failure

verdict_identity (1-5):
  5 = perfect likeness, all features match
  4 = recognizable, minor feature drift
  3 = ambiguous — could pass at distance
  2 = noticeably off, viewers would notice
  1 = wrong person

verdict_temporal (1-5):
  5 = zero flicker, smooth motion
  4 = minor frame-to-frame variance, not distracting
  3 = visible jitter, distracting on close inspection
  2 = pulsing/strobing
  1 = unusable temporal failure
"""

# Bucketing thresholds — used by shot_fingerprint.py for regime classification.
# These are coarse on purpose: agent reasons about regimes, not exact values.
BG_LUMINANCE_BUCKETS = [
    (0.25, "dark"),
    (0.55, "mid"),
    (1.01, "bright"),  # 1.01 to include 1.0
]

MASK_OCCUPANCY_BUCKETS = [
    (0.10, "tight"),
    (0.25, "small"),
    (0.50, "medium"),
    (1.01, "large"),
]

# Motion class thresholds (pixels per frame on mask centroid)
MOTION_CLASS_BUCKETS = [
    (1.5,  "static"),
    (5.0,  "head_tilt"),
    (15.0, "walking"),
    (1e9,  "fast"),
]
