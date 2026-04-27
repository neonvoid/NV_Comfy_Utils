"""
NV Simple Color Match - Minimal mean-shift color anchor with temporal EMA.

Replaces NV_CropColorFix in the default post-decode chain for the new static-mask
VACE architecture. Intended to provide cross-chunk color stability without the
variance-scaling, multi-stage cascade, or per-frame morphology amplification that
caused CCF to over-correct ("matte face") on the new pipeline.

Design principles (categorically prevent each CCF failure mode by construction):

1. **Mean-shift ONLY (no std/variance scaling).** No `std_tgt / std_src` term
   anywhere in the math. The face's contrast is mathematically untouchable —
   "matte face" is impossible because the mechanism that produces it doesn't
   exist in the formula.

2. **Single-stage operation.** No boundary correction, no compositing, no LF/HF
   decomposition. Just per-frame mean shift + temporal EMA + clamp. CCF's >100
   stage interactions reduce to 0.

3. **Reference mask binarized ONCE at entry.** No per-frame morphology, no
   soft-mask amplification, no per-frame ref zone variance. The "boiling edge"
   mechanism (D-123) requires per-frame morphology on a soft mask — we don't
   do per-frame morphology, so boiling edge is impossible.

4. **EMA always advances.** On frames with insufficient ref pixels, decays
   toward neutral instead of freezing state. No recovery strobe (CCF Bug #3).

5. **Hard-clamped per-frame shift.** Blast radius bounded by max_shift
   parameter. Even adversarial input cannot produce extreme correction.

6. **RGB direct.** No Lab colorspace roundtrip. No precision loss at conversion
   boundaries.

7. **Optional seed shift inputs** for cross-chunk continuity. Output 3 final
   shift values that can wire into the next chunk's seed inputs to maintain
   color anchor across chunk boundaries (Gemini's catch from architecture
   debate — Option C beats Option D specifically because of cross-chunk
   temporal stability).

Architecture validated via 2-round adversarial multi-AI debate 2026-04-27 (D-126).
NV_CropColorFix remains in the codebase as legacy/opt-in rescue for shots that
need aggressive variance-scaling correction.
"""

import torch

from .mask_ops import mask_erode_dilate


class NV_SimpleColorMatch:
    """Per-channel mean-shift color match with temporal EMA.

    Drop-in replacement for NV_CropColorFix in the new static-mask VACE pipeline.
    Wire between VAE Decode and NV_InpaintStitch2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_crop": ("IMAGE", {
                    "tooltip": "Generated crop after VAE decode (post-KSampler)."
                }),
                "original_crop": ("IMAGE", {
                    "tooltip": "Original cropped source from NV_InpaintCrop2 (color reference)."
                }),
                "mask": ("MASK", {
                    "tooltip": "Crop-space mask. mask=1 = generated region (face), "
                               "mask=0 = preserved region (color reference zone)."
                }),
                "enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master enable. False = passthrough (zero correction). "
                               "Use to A/B test against raw VACE output."
                }),
            },
            "optional": {
                "ema_strength": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "Temporal EMA history weight. 0.0 = fully per-frame (no smoothing). "
                               "0.3 = light smoothing (default). 0.7 = heavy. 0.95 = very sticky. "
                               "Higher = more cross-chunk color stability against VACE drift."
                }),
                "max_shift": ("FLOAT", {
                    "default": 0.02, "min": 0.0, "max": 0.2, "step": 0.005,
                    "tooltip": "Hard cap on per-channel mean shift (in [0,1] image range). "
                               "0.02 = ~5/255 (subtle anchor, default). 0.05 = ~13/255 (moderate). "
                               "0.0 = effectively disabled (zero shift). Bounds blast radius — "
                               "even adversarial inputs cannot exceed this magnitude."
                }),
                "ref_threshold": ("FLOAT", {
                    "default": 0.01, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Mask values below this define the reference (mask=0) zone. "
                               "0.01 = strict (only fully-zero pixels). With binarized upstream "
                               "masks (threshold=True on VaceControlVideoPrep) any value works."
                }),
                "ref_erosion": ("INT", {
                    "default": 4, "min": 0, "max": 32, "step": 1,
                    "tooltip": "Erode reference zone inward by N pixels at NODE ENTRY (applied ONCE, "
                               "not per-frame). Excludes pixels near the boundary that may have VACE "
                               "bleed. 0 = no erosion. 4 = ½ VAE block (default). 8 = 1 VAE block. "
                               "Crucially, this morphology runs once on a binarized mask — no "
                               "per-frame ref zone variance, no boiling-edge amplification."
                }),
                "min_ref_pixels": ("INT", {
                    "default": 100, "min": 10, "max": 10000, "step": 10,
                    "tooltip": "Minimum reference pixels for a fresh per-frame shift computation. "
                               "Frames below this fall back to EMA decay (NOT skip — EMA still "
                               "advances toward neutral, preventing the recovery strobe that "
                               "afflicted NV_CropColorFix on insufficient-ref frames)."
                }),
                "neutral_decay": ("FLOAT", {
                    "default": 0.95, "min": 0.5, "max": 1.0, "step": 0.05,
                    "tooltip": "On insufficient-ref frames: ema_shift *= neutral_decay (drifts "
                               "toward 0). 0.95 = very gentle decay (recovery is invisible). "
                               "0.5 = aggressive return to neutral. Set to 1.0 to literally hold "
                               "the prior shift, but be aware that introduces stale-state risk."
                }),
                "seed_shift_r": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001,
                    "forceInput": True,
                    "tooltip": "Optional EMA seed value for R channel. Wire prior chunk's "
                               "final_shift_r output for cross-chunk color continuity. "
                               "0.0 = fresh start (no seed)."
                }),
                "seed_shift_g": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001,
                    "forceInput": True,
                    "tooltip": "Optional EMA seed value for G channel."
                }),
                "seed_shift_b": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.001,
                    "forceInput": True,
                    "tooltip": "Optional EMA seed value for B channel."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("corrected_crop", "info", "final_shift_r", "final_shift_g", "final_shift_b")
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/Inpaint"
    DESCRIPTION = (
        "Minimal mean-shift color anchor with temporal EMA. Single-stage replacement "
        "for NV_CropColorFix in the new static-mask VACE pipeline. Categorically "
        "prevents CCF's failure modes: no variance scaling (no matte-face), single "
        "binarization at entry (no boiling-edge amplification), continuous EMA (no "
        "skip strobe), hard-clamped shift (bounded blast radius). Outputs final EMA "
        "state as 3 FLOATs for chunk-to-chunk color continuity."
    )

    def execute(
        self,
        generated_crop,
        original_crop,
        mask,
        enable,
        ema_strength=0.3,
        max_shift=0.02,
        ref_threshold=0.01,
        ref_erosion=4,
        min_ref_pixels=100,
        neutral_decay=0.95,
        seed_shift_r=0.0,
        seed_shift_g=0.0,
        seed_shift_b=0.0,
    ):
        TAG = "[NV_SimpleColorMatch]"
        device = generated_crop.device

        if not enable:
            print(f"{TAG} disabled (passthrough)")
            return (generated_crop, f"{TAG} disabled (passthrough)", 0.0, 0.0, 0.0)

        B, H, W, C = generated_crop.shape
        if original_crop.shape != generated_crop.shape:
            raise ValueError(
                f"{TAG} Shape mismatch: original {original_crop.shape} vs generated {generated_crop.shape}. "
                f"Both must be [B, H, W, C] with identical batch and spatial dims."
            )

        # --- Mask normalization (strict — no silent pad/truncate) ---
        m = mask.float().to(device)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[1:] != (H, W):
            m = torch.nn.functional.interpolate(
                m.unsqueeze(1), size=(H, W), mode="nearest-exact"
            ).squeeze(1)
        if m.shape[0] != B:
            if m.shape[0] == 1:
                m = m.expand(B, -1, -1)
            else:
                raise ValueError(
                    f"{TAG} Mask frame count ({m.shape[0]}) does not match batch ({B}). "
                    f"Use a single-frame mask (broadcast to {B}) or {B} frames. "
                    f"Silent padding/truncation explicitly rejected (CCF Bug 8 lesson)."
                )

        # --- Binarize ref mask ONCE at entry (no per-frame morphology) ---
        ref_base = (m < ref_threshold).float()  # [B, H, W]
        if ref_erosion > 0:
            ref_base = mask_erode_dilate(ref_base, -ref_erosion)
        ref_mask_bool = ref_base > 0.5  # [B, H, W] bool — frozen for the rest of execution

        # --- Vectorized per-frame mean computation ---
        gen = generated_crop.float()
        orig = original_crop.float()
        gen_rgb = gen[..., :3]   # [B, H, W, 3]
        orig_rgb = orig[..., :3]

        ref_4d = ref_mask_bool.unsqueeze(-1).float()  # [B, H, W, 1]
        ref_count_per_frame = ref_mask_bool.sum(dim=(1, 2)).float()  # [B]
        count_safe = ref_count_per_frame.clamp(min=1).unsqueeze(-1)  # [B, 1]

        gen_mean = (gen_rgb * ref_4d).sum(dim=(1, 2)) / count_safe   # [B, 3]
        orig_mean = (orig_rgb * ref_4d).sum(dim=(1, 2)) / count_safe  # [B, 3]
        raw_shifts = (orig_mean - gen_mean).clamp(-max_shift, max_shift)  # [B, 3]

        # --- Sequential EMA pass (state-dependent) ---
        ema_shift = torch.tensor(
            [seed_shift_r, seed_shift_g, seed_shift_b],
            device=device, dtype=torch.float32,
        )
        ema_initialized = bool(seed_shift_r != 0.0 or seed_shift_g != 0.0 or seed_shift_b != 0.0)

        corrected = gen.clone()
        n_valid = 0
        n_decay = 0
        total_abs_shift = torch.zeros(3, device=device, dtype=torch.float32)

        for b in range(B):
            ref_count = int(ref_count_per_frame[b].item())

            if ref_count >= min_ref_pixels:
                # Fresh per-frame shift, EMA-blended with prior state
                if ema_initialized:
                    ema_shift = ema_strength * ema_shift + (1.0 - ema_strength) * raw_shifts[b]
                else:
                    ema_shift = raw_shifts[b]
                    ema_initialized = True
                n_valid += 1
            else:
                # Decay toward neutral — EMA always advances, never freezes
                ema_shift = ema_shift * neutral_decay
                n_decay += 1

            # Apply current EMA shift to this frame's RGB channels
            shift_view = ema_shift.view(1, 1, 3)  # [1, 1, 3] broadcasts to [H, W, 3]
            corrected[b, ..., :3] = (gen[b, ..., :3] + shift_view).clamp(0, 1)

            total_abs_shift = total_abs_shift + ema_shift.abs()

        # --- Diagnostics ---
        avg_shift_255 = (total_abs_shift / max(B, 1)) * 255.0
        info_lines = [
            f"{TAG} {B} frames, {H}x{W}, ema={ema_strength:.2f}, max_shift={max_shift:.3f}",
            f"  Mask: ref_threshold={ref_threshold:.3f}, ref_erosion={ref_erosion}px (applied ONCE at entry)",
            f"  Validity: {n_valid}/{B} fresh shift, {n_decay}/{B} EMA decay (insufficient ref)",
            f"  Avg applied shift (in /255): "
            f"R={avg_shift_255[0].item():.2f}, G={avg_shift_255[1].item():.2f}, B={avg_shift_255[2].item():.2f}",
            f"  Final EMA shift (chunk-continuation): "
            f"R={ema_shift[0].item():.4f}, G={ema_shift[1].item():.4f}, B={ema_shift[2].item():.4f}",
        ]
        if seed_shift_r != 0 or seed_shift_g != 0 or seed_shift_b != 0:
            info_lines.insert(2, f"  Seeded from prior chunk: "
                                 f"R={seed_shift_r:.4f}, G={seed_shift_g:.4f}, B={seed_shift_b:.4f}")
        info = "\n".join(info_lines)
        print(info)

        return (
            corrected,
            info,
            float(ema_shift[0].item()),
            float(ema_shift[1].item()),
            float(ema_shift[2].item()),
        )


NODE_CLASS_MAPPINGS = {
    "NV_SimpleColorMatch": NV_SimpleColorMatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SimpleColorMatch": "NV Simple Color Match",
}
