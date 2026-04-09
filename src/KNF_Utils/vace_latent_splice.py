"""
NV VACE Latent Splice — Eliminate VAE roundtrip drift in chunked VACE inpainting.

Problem: When chunk N's tail frames are prepended to chunk N+1's control video,
they go through VAE decode (chunk N) → pixel space → VAE encode (chunk N+1).
This roundtrip introduces ~7/255 brightness/color drift. Because VACE treats
mask=0 regions as authoritative anchors, the drift biases the entire chunk.

Solution: After WanVaceToVideo encodes the control video into vace_frames
conditioning, overwrite the tail portion's inactive channels (first 16 of 32)
with the clean KSampler output latent from the previous chunk. Both are in
encoder domain — no domain mismatch.

Pipeline wiring:
  WanVaceToVideo → CONDITIONING → [NV_VaceLatentSplice] → patched CONDITIONING → KSampler
  Prev chunk KSampler → LATENT ─────────────────↑
"""

import copy
import torch


class NV_VaceLatentSplice:
    """Replace roundtrip-drifted tail latents in VACE conditioning with clean cached latents."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {
                    "tooltip": "CONDITIONING from WanVaceToVideo (contains vace_frames with roundtrip-drifted tail)."
                }),
                "prev_chunk_latent": ("LATENT", {
                    "tooltip": "LATENT output from previous chunk's KSampler. Encoder-domain, same crop resolution."
                }),
                "tail_frames": ("INT", {
                    "default": 4, "min": 4, "max": 16, "step": 4,
                    "tooltip": "Number of pixel-space tail frames prepended by VaceControlVideoPrep. "
                               "Must match tail_overlap_frames. Multiple of 4 (WAN temporal rule)."
                }),
            },
            "optional": {
                "prev_tail_trim": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 4,
                    "tooltip": "If the previous chunk itself had tail frames prepended, skip this many "
                               "pixel frames from the front of prev_chunk_latent to get its real output. "
                               "0 for chunk 0 (no prior tail). Set to chunk N-1's tail_frames value."
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "execute"
    CATEGORY = "NV_Utils/VACE"
    DESCRIPTION = (
        "Eliminates VAE roundtrip drift in chunked VACE inpainting by splicing clean "
        "encoder-domain latents from the previous chunk's KSampler output into the VACE "
        "conditioning's inactive channels. Place between WanVaceToVideo and KSampler."
    )

    def execute(self, conditioning, prev_chunk_latent, tail_frames, prev_tail_trim=0):
        TAG = "[NV_VaceLatentSplice]"

        # Enforce WAN temporal alignment (API/math nodes can bypass widget step=4)
        if tail_frames % 4 != 0:
            raise ValueError(f"{TAG} tail_frames={tail_frames} must be a multiple of 4 (WAN temporal rule).")
        if prev_tail_trim % 4 != 0:
            raise ValueError(f"{TAG} prev_tail_trim={prev_tail_trim} must be a multiple of 4 (WAN temporal rule).")

        # --- Extract previous chunk's latent ---
        prev_latent = prev_chunk_latent["samples"]  # [B, 16, T_prev, H, W] encoder domain
        if prev_latent.ndim != 5 or prev_latent.shape[1] != 16:
            raise ValueError(
                f"{TAG} prev_chunk_latent has unexpected shape {list(prev_latent.shape)}. "
                f"Expected [B, 16, T, H, W] (WAN encoder-domain latent)."
            )
        print(f"{TAG} prev_chunk_latent shape: {list(prev_latent.shape)}")

        # Convert pixel frame counts to latent temporal frames
        # WAN temporal compression: T_pixel = 4*T_latent + 1, so N pixel frames ≈ N/4 latent frames
        # (for prepended frames that are multiples of 4, this is exact)
        tail_T = tail_frames // 4
        prev_trim_T = prev_tail_trim // 4

        if tail_T <= 0:
            print(f"{TAG} tail_frames={tail_frames} → tail_T=0, nothing to splice. Passing through.")
            return (conditioning,)

        # Skip the previous chunk's own tail prefix to get its real generated output
        if prev_trim_T > 0:
            if prev_trim_T >= prev_latent.shape[2]:
                raise ValueError(
                    f"{TAG} prev_tail_trim={prev_tail_trim} → {prev_trim_T} latent frames, "
                    f"but prev_chunk_latent only has {prev_latent.shape[2]} temporal frames. "
                    f"Cannot trim more than available."
                )
            real_output = prev_latent[:, :, prev_trim_T:, :, :]
            print(f"{TAG} Trimmed {prev_trim_T} latent frames from prev chunk front → "
                  f"real output: {list(real_output.shape)}")
        else:
            real_output = prev_latent

        # Extract tail slice from end of previous chunk's real output
        if tail_T > real_output.shape[2]:
            raise ValueError(
                f"{TAG} tail_frames={tail_frames} → {tail_T} latent frames needed, "
                f"but prev chunk real output only has {real_output.shape[2]} temporal frames."
            )
        cached_tail = real_output[:, :, -tail_T:, :, :]  # [B, 16, tail_T, H, W]
        print(f"{TAG} Cached tail: {list(cached_tail.shape)} (last {tail_T} latent frames)")

        # --- Patch each conditioning entry ---
        patched = []
        spliced_count = 0

        for cond_tensor, cond_dict in conditioning:
            new_dict = copy.copy(cond_dict)

            vace_frames_list = cond_dict.get("vace_frames", None)
            if vace_frames_list is None:
                patched.append([cond_tensor, new_dict])
                continue

            new_vace_frames = []
            for vf in vace_frames_list:
                # vf shape: [B, 32, T, H, W] — first 16ch = inactive, second 16ch = reactive
                vf = vf.clone()

                # Validate dimensions
                if vf.ndim != 5 or vf.shape[1] != 32:
                    raise ValueError(
                        f"{TAG} vace_frames has unexpected shape {list(vf.shape)}. "
                        f"Expected [B, 32, T, H, W]."
                    )
                if cached_tail.shape[0] != vf.shape[0]:
                    raise ValueError(
                        f"{TAG} Batch mismatch! cached_tail batch={cached_tail.shape[0]} "
                        f"but vace_frames batch={vf.shape[0]}."
                    )
                if cached_tail.shape[3:] != vf.shape[3:]:
                    raise ValueError(
                        f"{TAG} Spatial mismatch! cached_tail is {list(cached_tail.shape[3:])} "
                        f"but vace_frames is {list(vf.shape[3:])}. "
                        f"Chunks must use the same crop resolution."
                    )
                if tail_T > vf.shape[2]:
                    raise ValueError(
                        f"{TAG} Temporal mismatch! tail_T={tail_T} but vace_frames "
                        f"only has {vf.shape[2]} latent frames."
                    )

                # Splice: overwrite inactive channels (0:16) for tail portion (0:tail_T)
                # with clean encoder-domain latent from prev chunk
                tail_device = cached_tail.to(device=vf.device, dtype=vf.dtype)
                vf[:, :16, :tail_T, :, :] = tail_device

                new_vace_frames.append(vf)
                spliced_count += 1

            new_dict["vace_frames"] = new_vace_frames
            patched.append([cond_tensor, new_dict])

        print(f"{TAG} Spliced {spliced_count} vace_frames entries: "
              f"replaced inactive channels [:16, 0:{tail_T}] with clean latent. "
              f"Zero roundtrip drift on tail.")

        return (patched,)


NODE_CLASS_MAPPINGS = {
    "NV_VaceLatentSplice": NV_VaceLatentSplice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_VaceLatentSplice": "NV VACE Latent Splice",
}
