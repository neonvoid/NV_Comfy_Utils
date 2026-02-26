"""
NV_FreeSwimPatch — Inward Sliding-Window Spatial Attention Mask

Based on FreeSwim (arXiv:2511.14712). At resolutions above training (e.g.,
1080p on Wan 14B trained at 720p), each query's spatial attention is diluted
over more tokens than training. This node restricts each query to a
training-resolution-sized patch-token window via FlexAttention's block-sparse
mask.

Edge tokens get shifted-inward windows so they see a full training-size
receptive field. Temporal dimension is completely unrestricted — orthogonal
to context windows.

Patch token dimensions are auto-derived: pixels ÷ VAE_compression ÷ patch_size.
The patch_size is read from the model at runtime (fallback: (1,2,2) for WAN).

Hook point: optimized_attention_override in transformer_options, which
intercepts Q/K/V BEFORE the attention dot product
(comfy/ldm/modules/attention.py:125-141).
"""

import torch

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("[FreeSwimPatch] WARNING: FlexAttention not available (PyTorch 2.5+ required). "
          "Node will pass through without masking.")

# Training resolution presets: (width_px, height_px)
_TRAINING_PRESETS = {
    "720p - Wan 2.1 14B (1280x720)": (1280, 720),
    "480p - Wan 2.1 1.3B (832x480)": (832, 480),
    "custom": None,
}

# VAE spatial compression factor (standard for WAN / most diffusion models)
_VAE_SPATIAL_COMPRESSION = 8


def _pixels_to_patch_tokens(px, vae_compression, spatial_patch_size):
    """Convert pixel dimension to patch-token count: px ÷ VAE ÷ patch_size."""
    return px // vae_compression // spatial_patch_size


def _extract_spatial_patch_size(model):
    """Read spatial patch_size from the diffusion model. Returns (h_patch, w_patch)."""
    try:
        dm = model.model.diffusion_model
        ps = dm.patch_size  # tuple like (1, 2, 2)
        return (ps[1], ps[2])
    except Exception:
        return (2, 2)  # WAN default


def _make_mask_mod(H_p, W_p, hw, hh, num_anchor):
    """Factory: returns a mask_mod(b, h, q_idx, kv_idx) -> bool.

    Uses a factory to capture all grid parameters as plain Python ints
    in the closure — required for FlexAttention block evaluation.

    The % spatial_tokens operation strips temporal position from the index.
    Two tokens at the same (h, w) but different t produce the same spatial
    coordinates → pass the window check. This makes the temporal dimension
    completely unrestricted (FreeSwim design: spatial-only windowing).
    """
    spatial_tokens = H_p * W_p

    def mask_mod(b, h, q_idx, kv_idx):
        # Anchor tokens (prepended by NV_AnchorKVCache): always attend
        if kv_idx < num_anchor:
            return True

        # Decompose to spatial positions (temporal unrestricted via modulo)
        real_kv = kv_idx - num_anchor
        h_q = (q_idx % spatial_tokens) // W_p
        w_q = (q_idx % spatial_tokens) % W_p
        h_k = (real_kv % spatial_tokens) // W_p
        w_k = (real_kv % spatial_tokens) % W_p

        # Inward boundary shift — edge tokens get expanded window so they
        # see a full training-size receptive field instead of a truncated one
        delta_w = max(hw - w_q, w_q - (W_p - hw - 1), 0)
        delta_h = max(hh - h_q, h_q - (H_p - hh - 1), 0)

        # Spatial window check
        return (abs(w_q - w_k) <= hw + delta_w and
                abs(h_q - h_k) <= hh + delta_h)

    return mask_mod


class NV_FreeSwimPatch:
    """
    Applies inward sliding-window spatial attention mask at resolutions
    above training (e.g., 1080p on Wan 14B trained at 720p).

    At training resolution or below, the node is a no-op. Connect before
    your sampler, and before NV_AnchorKVCache if using both:

        [NV_FreeSwimPatch] -> [NV_AnchorKVCache] -> [KSampler]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Toggle on/off for A/B testing. When disabled, model passes through unmodified."
                }),
                "training_resolution": (list(_TRAINING_PRESETS.keys()), {
                    "default": "720p - Wan 2.1 14B (1280x720)",
                    "tooltip": "Training resolution of the model. Patch token window is auto-derived. "
                               "Select 'custom' to specify pixel dimensions manually."
                }),
                "custom_width": ("INT", {
                    "default": 1280,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom training width in PIXELS. Only used when training_resolution='custom'. "
                               "Patch tokens are computed automatically: pixels ÷ 8 (VAE) ÷ patch_size (from model)."
                }),
                "custom_height": ("INT", {
                    "default": 720,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                    "tooltip": "Custom training height in PIXELS. Only used when training_resolution='custom'. "
                               "Patch tokens are computed automatically: pixels ÷ 8 (VAE) ÷ patch_size (from model)."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Inward sliding-window spatial attention mask for super-resolution generation. "
        "Keeps each query's spatial attention within training-resolution bounds. "
        "No-op at or below training resolution. Based on FreeSwim (arXiv:2511.14712)."
    )

    def patch(self, model, enabled, training_resolution, custom_width, custom_height):
        if not enabled:
            return (model,)

        if not FLEX_ATTENTION_AVAILABLE:
            print("[FreeSwimPatch] FlexAttention unavailable — returning model unmodified.")
            return (model,)

        # Resolve training resolution in pixels
        preset = _TRAINING_PRESETS.get(training_resolution)
        if preset is not None:
            width_px, height_px = preset
        else:
            width_px, height_px = custom_width, custom_height

        # Extract patch_size from model and convert pixels → patch tokens
        ps_h, ps_w = _extract_spatial_patch_size(model)
        tw = _pixels_to_patch_tokens(width_px, _VAE_SPATIAL_COMPRESSION, ps_w)
        th = _pixels_to_patch_tokens(height_px, _VAE_SPATIAL_COMPRESSION, ps_h)

        print(f"[FreeSwimPatch] {width_px}x{height_px} px → {tw}x{th} patch tokens "
              f"(VAE÷{_VAE_SPATIAL_COMPRESSION}, patch_size=({ps_h},{ps_w}))")

        model = model.clone()

        hw = tw // 2  # half-width for window centering
        hh = th // 2  # half-height for window centering

        # BlockMask cache: (S_q, S_kv, H_patch, W_patch) -> BlockMask
        mask_cache = {}

        debug_state = {
            "first_step": True,
            "last_logged_step": -1,
            "step_apply_count": 0,
            "step_skip_count": 0,
        }

        # Check for existing override to chain with
        existing_override = None
        if "transformer_options" in model.model_options:
            existing_override = model.model_options["transformer_options"].get(
                "optimized_attention_override"
            )

        def _call_next(original_func, args, kwargs):
            """Call existing override if chained, otherwise call original."""
            if existing_override is not None:
                return existing_override(original_func, *args, **kwargs)
            return original_func(*args, **kwargs)

        def freeswim_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]

            t_opts = kwargs.get("transformer_options", {})
            grid_sizes = t_opts.get("grid_sizes", None)
            block_idx = t_opts.get("block_index", -1)
            sigmas = t_opts.get("sigmas", None)

            # --- Step tracking ---
            current_step_sig = sigmas[0].item() if sigmas is not None and len(sigmas) > 0 else None
            is_new_step = (current_step_sig is not None
                           and current_step_sig != debug_state["last_logged_step"])
            if is_new_step:
                if debug_state["step_apply_count"] > 0 or debug_state["step_skip_count"] > 0:
                    print(f"[FreeSwimPatch] Step summary: "
                          f"applied {debug_state['step_apply_count']}, "
                          f"skipped {debug_state['step_skip_count']}")
                debug_state["last_logged_step"] = current_step_sig
                debug_state["step_apply_count"] = 0
                debug_state["step_skip_count"] = 0

            # --- PASSTHROUGH CASES ---

            # No grid_sizes: can't determine spatial layout
            if grid_sizes is None:
                if debug_state["first_step"] and block_idx == 0:
                    print("[FreeSwimPatch] Passthrough: no grid_sizes in transformer_options")
                    debug_state["first_step"] = False
                debug_state["step_skip_count"] += 1
                return _call_next(original_func, args, kwargs)

            H_patch, W_patch = grid_sizes[1], grid_sizes[2]

            # Cross-attention: K from text (~512) is SMALLER than Q from video (~8000+)
            if k.shape[1] < q.shape[1]:
                return _call_next(original_func, args, kwargs)

            # No-op: at or below training resolution
            if H_patch <= th and W_patch <= tw:
                if debug_state["first_step"] and block_idx == 0:
                    print(f"[FreeSwimPatch] No-op: grid {W_patch}x{H_patch} "
                          f"<= training {tw}x{th}")
                    debug_state["first_step"] = False
                debug_state["step_skip_count"] += 1
                return _call_next(original_func, args, kwargs)

            # --- APPLY FREESWIM ---

            S_q = q.shape[1]
            S_kv = k.shape[1]
            num_anchor = S_kv - S_q  # 0 if no anchor KV, >0 if anchor tokens prepended

            # Get or create cached BlockMask
            cache_key = (S_q, S_kv, H_patch, W_patch)
            if cache_key not in mask_cache:
                mask_mod_fn = _make_mask_mod(H_patch, W_patch, hw, hh, num_anchor)
                block_mask = create_block_mask(
                    mask_mod_fn,
                    B=None,
                    H=None,
                    Q_LEN=S_q,
                    KV_LEN=S_kv,
                    device=str(q.device),
                )
                mask_cache[cache_key] = block_mask

                if block_idx == 0:
                    print(f"[FreeSwimPatch] Mask created: "
                          f"Q_LEN={S_q}, KV_LEN={S_kv}, "
                          f"grid={W_patch}x{H_patch}, "
                          f"anchor={num_anchor}")

            cached_mask = mask_cache[cache_key]

            # First-step verbose logging
            if debug_state["first_step"] and block_idx == 0:
                print(f"[FreeSwimPatch] === First step activated ===")
                print(f"[FreeSwimPatch] grid: {W_patch}x{H_patch} patches, "
                      f"training window: {tw}x{th}")
                print(f"[FreeSwimPatch] Q: {q.shape}, K: {k.shape}, "
                      f"anchor tokens: {num_anchor}")
                print(f"[FreeSwimPatch] dtype: {q.dtype}, device: {q.device}")
                debug_state["first_step"] = False

            # Reshape for FlexAttention: [B, S, inner_dim] -> [B, heads, S, dim_head]
            B_size = q.shape[0]
            heads = kwargs["heads"]
            dim_head = q.shape[2] // heads

            q_4d = q.view(B_size, S_q, heads, dim_head).transpose(1, 2).contiguous()
            k_4d = k.view(B_size, S_kv, heads, dim_head).transpose(1, 2).contiguous()
            v_4d = v.view(B_size, S_kv, heads, dim_head).transpose(1, 2).contiguous()

            # FlexAttention with spatial sliding-window mask
            out = flex_attention(q_4d, k_4d, v_4d, block_mask=cached_mask)

            # Reshape back: [B, heads, S_q, dim_head] -> [B, S_q, inner_dim]
            out = out.transpose(1, 2).reshape(B_size, S_q, heads * dim_head)

            debug_state["step_apply_count"] += 1
            return out

        # Apply the override
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = freeswim_override

        chain_info = " (chained with existing override)" if existing_override else ""
        print(f"[FreeSwimPatch] Patched model: {tw}x{th} training window{chain_info}")

        return (model,)


NODE_CLASS_MAPPINGS = {
    "NV_FreeSwimPatch": NV_FreeSwimPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_FreeSwimPatch": "NV FreeSwim Patch",
}
