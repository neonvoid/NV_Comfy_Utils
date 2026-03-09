"""
NV VACE Temporal Align

Aligns all VACE conditioning entries to the same temporal dimension.

When multiple VACE source nodes (WanVaceToVideo, NV_VacePrePassReference, etc.)
add entries to conditioning, they may have different temporal lengths — e.g.,
one has T=27 (control-only) and another has T=42 (refs + control). The VACE
model requires all entries to have identical T for torch.stack() in
model_base.py WAN21_Vace.extra_conds().

This node inspects all vace_frames entries, finds the maximum T, and pads
shorter entries with neutral latent fill + mask=1 (generate / no guidance)
so they all match. Place this node between your VACE conditioning chain and
the sampler.

Padding semantics:
- vace_frames pad: plain zeros for all 32ch — matches model_base.py
  WAN21_Vace.extra_conds() default when no VACE conditioning exists.
  After process_in, both 16ch halves become (-mean/std), a symmetric
  neutral signal. Do NOT use process_out(zeros) — Wan21 adds latents_mean,
  creating an asymmetric signal.
- vace_mask pad: ones — mask=1 means "generate freely, no guidance here"
- vace_strength: unchanged (per-entry scalar, not temporal)
"""

import torch


class NV_VaceTemporalAlign:
    """
    Align VACE conditioning entries to a common temporal dimension.

    Finds the longest vace_frames entry across all conditioning entries and
    front-pads shorter entries with neutral fill so torch.stack() succeeds.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "align"
    CATEGORY = "NV_Utils/conditioning"
    DESCRIPTION = (
        "Align VACE conditioning entries to a common temporal dimension. "
        "Required when combining multiple VACE sources (e.g., WanVaceToVideo + "
        "NV_VacePrePassReference) that produce entries with different frame counts. "
        "Place between your VACE conditioning chain and the sampler."
    )

    def align(self, positive, negative):
        # Compute a single max_t across BOTH positive and negative to ensure
        # they stay in sync. WanVaceToVideo always sets both identically, but
        # multi-source pipelines could produce different entry sets.
        max_t = max(_find_max_t(positive), _find_max_t(negative))
        positive = _align_vace_entries(positive, "positive", max_t)
        negative = _align_vace_entries(negative, "negative", max_t)
        return (positive, negative)


def _find_max_t(conditioning):
    """Find the maximum temporal dimension across all vace_frames entries."""
    max_t = 0
    for cond_entry in conditioning:
        vace_frames = cond_entry[1].get("vace_frames", None)
        if vace_frames is not None:
            for vf in vace_frames:
                max_t = max(max_t, vf.shape[2])
    return max_t


def _pad_vace_entry(vf, mask_entry, target_t):
    """Pad a single vace_frames + vace_mask entry to target_t by front-prepending neutral fill.

    Args:
        vf: [1, 32, T, H/8, W/8] — VACE latent (16ch inactive + 16ch reactive)
        mask_entry: [1, 64, T, H/8, W/8] — VACE mask
        target_t: desired temporal dimension

    Returns:
        (padded_vf, padded_mask) with T = target_t
    """
    current_t = vf.shape[2]
    pad_t = target_t - current_t
    if pad_t <= 0:
        return vf, mask_entry

    # Build neutral 32ch VACE latent padding — plain zeros for both halves.
    # This matches model_base.py WAN21_Vace.extra_conds() default (line 1332):
    #   vace_frames = [torch.zeros(noise_shape)]
    # After process_in, both halves become (-mean/std) — a symmetric neutral
    # signal that the model interprets as "no VACE conditioning here."
    # NOTE: Do NOT use process_out(zeros) for one half — Wan21 process_out
    # adds latents_mean, creating an asymmetric signal after process_in.
    B = vf.shape[0]
    pad_latent = torch.zeros(
        B, 32, pad_t, vf.shape[3], vf.shape[4],
        device=vf.device, dtype=vf.dtype,
    )

    padded_vf = torch.cat([pad_latent, vf], dim=2)

    # Build mask padding: ones = "generate freely, no guidance"
    mask_pad = torch.ones(
        B, 64, pad_t, mask_entry.shape[3], mask_entry.shape[4],
        device=mask_entry.device, dtype=mask_entry.dtype,
    )
    padded_mask = torch.cat([mask_pad, mask_entry], dim=2)

    return padded_vf, padded_mask


def _align_vace_entries(conditioning, label, max_t):
    """Align all vace_frames entries in a conditioning list to the same T."""
    if max_t == 0:
        return conditioning  # No VACE entries, nothing to do

    aligned = []
    any_padded = False

    for cond_entry in conditioning:
        cond_tensor, cond_dict = cond_entry[0], cond_entry[1].copy()
        vace_frames = cond_dict.get("vace_frames", None)
        vace_masks = cond_dict.get("vace_mask", None)

        if vace_frames is None:
            aligned.append([cond_tensor, cond_dict])
            continue

        new_frames = []
        new_masks = []

        for i, vf in enumerate(vace_frames):
            mask_entry = vace_masks[i] if vace_masks is not None and i < len(vace_masks) else None

            # Ensure every entry has a mask — construct default if missing
            if mask_entry is None:
                mask_entry = torch.ones(
                    vf.shape[0], 64, vf.shape[2], vf.shape[3], vf.shape[4],
                    device=vf.device, dtype=vf.dtype,
                )

            if vf.shape[2] < max_t:
                padded_vf, padded_mask = _pad_vace_entry(vf, mask_entry, max_t)
                new_frames.append(padded_vf)
                new_masks.append(padded_mask)

                pad_amount = max_t - vf.shape[2]
                if not any_padded:
                    print(f"[NV_VaceTemporalAlign] Padding {label} entry {i}: "
                          f"T={vf.shape[2]} -> T={max_t} (+{pad_amount} neutral frames)")
                    any_padded = True
            else:
                new_frames.append(vf)
                new_masks.append(mask_entry)

        cond_dict["vace_frames"] = new_frames
        cond_dict["vace_mask"] = new_masks

        aligned.append([cond_tensor, cond_dict])

    if any_padded:
        print(f"[NV_VaceTemporalAlign] All {label} VACE entries aligned to T={max_t}")

    return aligned


# Node registration
NODE_CLASS_MAPPINGS = {
    "NV_VaceTemporalAlign": NV_VaceTemporalAlign,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceTemporalAlign": "NV VACE Temporal Align",
}
