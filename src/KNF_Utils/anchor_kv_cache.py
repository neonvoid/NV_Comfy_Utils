"""
Anchor KV Cache for Context Windows

Caches frame 0's K/V representations from the first context window and injects
them into ALL subsequent windows within the same denoising step. This gives every
window direct attention access to the anchor frame, breaking the cross-window
isolation problem where distant windows share zero frames and rely on a "telephone
game" through intermediate overlapping windows for consistency.

Based on HiStream (arXiv 2512.21338) anchor caching, adapted for ComfyUI's
per-step parallel context window system.

Key insight: K has RoPE baked in (position 0), so injected K naturally receives
lower attention weight from distant frames — automatic temporal decay without
any scheduling. V is RoPE-free and carries pure content identity.

Hook point: optimized_attention_override in transformer_options, which intercepts
Q/K/V BEFORE the attention dot product (comfy/ldm/modules/attention.py:125-141).
"""

import torch


def _parse_target_blocks(target_blocks_str):
    """Parse target_blocks string into a set of block indices or None for 'all'."""
    s = target_blocks_str.strip().lower()
    if s == "all" or s == "":
        return None  # None means all blocks
    try:
        return set(int(x.strip()) for x in s.split(",") if x.strip())
    except ValueError:
        print(f"[AnchorKVCache] WARNING: Could not parse target_blocks '{target_blocks_str}', using all blocks")
        return None


class NV_AnchorKVCache:
    """
    Caches anchor frame K/V from the first context window and injects into
    all subsequent windows, giving every window direct attention to frame 0.

    Connect AFTER your context window node and BEFORE the sampler.

    Usage:
        [WAN Context Windows] -> [NV_AnchorKVCache] -> [KSampler]
        or
        [WAN Context Windows] -> [NV_AnchorKVCache] -> [VACE Context Window Patcher] -> [KSampler]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "num_anchor_frames": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1,
                    "tooltip": "Number of frames to cache from the start of the anchor window. "
                               "More frames = stronger anchoring but more VRAM and compute."
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Scale factor for injected V values. >1 strengthens anchor influence, "
                               "<1 weakens it. K is never scaled (preserves relevance signal)."
                }),
                "target_blocks": ("STRING", {
                    "default": "all",
                    "tooltip": "'all' for every transformer block, or comma-separated indices "
                               "like '0,5,10,15,19' to target specific blocks."
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "NV_Utils/attention"
    DESCRIPTION = (
        "Caches anchor frame K/V from the first context window and injects into "
        "all subsequent windows. Breaks cross-window isolation by giving every "
        "window direct attention access to frame 0."
    )

    def patch(self, model, num_anchor_frames, strength, target_blocks):
        model = model.clone()
        blocks = _parse_target_blocks(target_blocks)

        # Per-step KV cache: block_index -> (k_anchor, v_anchor)
        # Refreshed each step when window containing frame 0 fires
        kv_cache = {}

        # Debug state tracking
        # last_logged_step: which denoising step we last printed verbose logs for
        # step_capture_count: how many blocks captured this step (for summary)
        # step_inject_count: how many injections happened this step
        # first_step: whether this is the very first step (extra verbose)
        debug_state = {
            "last_logged_step": -1,
            "step_capture_count": 0,
            "step_inject_count": 0,
            "first_step": True,
            "cache_logged": False,
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

        def anchor_kv_override(original_func, *args, **kwargs):
            q, k, v = args[0], args[1], args[2]

            # Skip cross-attention — only cache/inject for self-attention.
            # Self-attention: Q and K have same seq_len (both are video tokens).
            # Cross-attention: Q=video tokens (long), K=text tokens (~512) — different lengths.
            # Without this check, cross-attn's 512-token K/V overwrites the correct
            # 8040-token self-attn cache since both share the same block_idx key.
            if q.shape[1] != k.shape[1]:
                return _call_next(original_func, args, kwargs)

            t_opts = kwargs.get("transformer_options", {})
            block_idx = t_opts.get("block_index", -1)
            window = t_opts.get("context_window", None)
            grid_sizes = t_opts.get("grid_sizes", None)
            sigmas = t_opts.get("sigmas", None)

            # Detect step changes from sigma values for per-step logging
            current_step_sig = sigmas[0].item() if sigmas is not None and len(sigmas) > 0 else None
            is_new_step = (current_step_sig is not None
                           and current_step_sig != debug_state["last_logged_step"])
            if is_new_step:
                # New denoising step — reset per-step counters
                if debug_state["step_capture_count"] > 0 or debug_state["step_inject_count"] > 0:
                    # Log summary of previous step (unless it's the very first)
                    print(f"[AnchorKVCache] Step summary: "
                          f"captured {debug_state['step_capture_count']} blocks, "
                          f"injected into {debug_state['step_inject_count']} block×window calls")
                debug_state["last_logged_step"] = current_step_sig
                debug_state["step_capture_count"] = 0
                debug_state["step_inject_count"] = 0
                debug_state["cache_logged"] = False

            # --- PASSTHROUGH CASES ---

            # No context window active (short video, no windowing): pass through
            if window is None or grid_sizes is None:
                if debug_state["first_step"] and block_idx == 0:
                    reason = "no context_window" if window is None else "no grid_sizes"
                    print(f"[AnchorKVCache] Passthrough: {reason} "
                          f"(context windows may not be active)")
                    debug_state["first_step"] = False
                return _call_next(original_func, args, kwargs)

            # Block not targeted: pass through
            if blocks is not None and block_idx not in blocks:
                return _call_next(original_func, args, kwargs)

            # Get index_list from the context window
            index_list = getattr(window, "index_list", None)
            if index_list is None:
                if debug_state["first_step"] and block_idx == 0:
                    print(f"[AnchorKVCache] Passthrough: window has no index_list "
                          f"(window type: {type(window).__name__})")
                return _call_next(original_func, args, kwargs)

            # Calculate tokens per frame from grid_sizes
            # grid_sizes = (T_patches, H_patches, W_patches)
            tokens_per_frame = grid_sizes[1] * grid_sizes[2]

            # First-step verbose logging: full state dump
            if debug_state["first_step"] and block_idx == 0:
                print(f"[AnchorKVCache] === First step activated ===")
                print(f"[AnchorKVCache] grid_sizes: {grid_sizes} "
                      f"(T={grid_sizes[0]}, H={grid_sizes[1]}, W={grid_sizes[2]})")
                print(f"[AnchorKVCache] tokens_per_frame: {tokens_per_frame} "
                      f"(H×W = {grid_sizes[1]}×{grid_sizes[2]})")
                print(f"[AnchorKVCache] Q/K/V shape: {q.shape} "
                      f"(B={q.shape[0]}, seq_len={q.shape[1]}, dim={q.shape[2]})")
                print(f"[AnchorKVCache] dtype: Q={q.dtype}, K={k.dtype}, V={v.dtype}")
                print(f"[AnchorKVCache] device: {k.device}")
                print(f"[AnchorKVCache] Window index_list: "
                      f"{index_list[:8]}{'...' + str(index_list[-3:]) if len(index_list) > 8 else ''} "
                      f"({len(index_list)} frames)")
                total_blocks = t_opts.get("total_blocks", "unknown")
                print(f"[AnchorKVCache] total_blocks: {total_blocks}, "
                      f"targeting: {'all' if blocks is None else sorted(blocks)}")
                if current_step_sig is not None:
                    print(f"[AnchorKVCache] sigma: {current_step_sig:.4f}")
                debug_state["first_step"] = False

            # Check if this window contains frame 0
            is_anchor_window = 0 in index_list

            if is_anchor_window:
                # === CAPTURE: Cache anchor K/V from this window ===
                frame_0_pos = index_list.index(0)
                start_token = frame_0_pos * tokens_per_frame
                end_token = start_token + (num_anchor_frames * tokens_per_frame)
                end_token = min(end_token, k.shape[1])

                # Cache on GPU (anchor KV per block is ~7MB with CFG, all 40 blocks ~280MB)
                kv_cache[block_idx] = (
                    k[:, start_token:end_token, :].clone(),
                    v[:, start_token:end_token, :].clone(),
                )
                debug_state["step_capture_count"] += 1

                # Log capture details on block 0 of each new step
                if not debug_state["cache_logged"] and block_idx == 0:
                    n_tokens = end_token - start_token
                    print(f"[AnchorKVCache] CAPTURE: anchor window "
                          f"(frames {index_list[:5]}{'...' if len(index_list) > 5 else ''})")
                    print(f"[AnchorKVCache]   frame_0 at pos {frame_0_pos}, "
                          f"tokens [{start_token}:{end_token}] = {n_tokens} tokens")
                    print(f"[AnchorKVCache]   K anchor shape: "
                          f"{kv_cache[block_idx][0].shape}, "
                          f"V anchor shape: {kv_cache[block_idx][1].shape}")

                # Log cache size summary once all blocks are captured
                if not debug_state["cache_logged"]:
                    total_blocks_count = t_opts.get("total_blocks", None)
                    expected = len(blocks) if blocks is not None else total_blocks_count
                    if expected and debug_state["step_capture_count"] >= expected:
                        total_bytes = sum(
                            kk.nbytes + vv.nbytes for kk, vv in kv_cache.values()
                        )
                        print(f"[AnchorKVCache]   Cache complete: {len(kv_cache)} blocks, "
                              f"{total_bytes / 1024 / 1024:.1f} MB GPU")
                        debug_state["cache_logged"] = True

                # Run anchor window attention normally (no injection into itself)
                return _call_next(original_func, args, kwargs)

            elif block_idx in kv_cache:
                # === INJECT: Prepend anchor K/V to this window's K/V ===
                k_anchor, v_anchor = kv_cache[block_idx]

                # Match device and dtype
                k_anchor = k_anchor.to(device=k.device, dtype=k.dtype)
                v_anchor = v_anchor.to(device=v.device, dtype=v.dtype)

                # Apply strength scaling to V only
                # K is never scaled — preserves the relevance/attention-weight signal
                # V scaling modulates how much anchor content flows through
                if strength != 1.0:
                    v_anchor = v_anchor * strength

                # Prepend anchor tokens: Q stays unchanged (asymmetric attention)
                # Each current-window query can attend to both anchor + current tokens
                k_extended = torch.cat([k_anchor, k], dim=1)
                v_extended = torch.cat([v_anchor, v], dim=1)

                debug_state["step_inject_count"] += 1

                # Log first injection per step (block 0 only to avoid spam)
                if debug_state["step_inject_count"] == 1:
                    print(f"[AnchorKVCache] INJECT: window "
                          f"(frames {index_list[:5]}{'...' if len(index_list) > 5 else ''}) "
                          f"block {block_idx}")
                    print(f"[AnchorKVCache]   K: {k.shape} -> {k_extended.shape} "
                          f"(+{k_anchor.shape[1]} anchor tokens)")

                new_args = list(args)
                new_args[1] = k_extended
                new_args[2] = v_extended

                return _call_next(original_func, tuple(new_args), kwargs)

            else:
                # No cache yet for this block — shouldn't happen if window 0 fires first
                if debug_state["step_inject_count"] == 0 and block_idx == 0:
                    print(f"[AnchorKVCache] WARNING: No cache for block {block_idx}, "
                          f"window frames {index_list[:5]}. "
                          f"Window 0 may not have fired yet this step.")
                return _call_next(original_func, args, kwargs)

        # Apply the override
        if "transformer_options" not in model.model_options:
            model.model_options["transformer_options"] = {}
        model.model_options["transformer_options"]["optimized_attention_override"] = anchor_kv_override

        chain_info = " (chained with existing override)" if existing_override else ""
        block_info = f"blocks {target_blocks}" if blocks is not None else "all blocks"
        print(f"[AnchorKVCache] Patched model: {num_anchor_frames} anchor frame(s), "
              f"strength={strength}, {block_info}{chain_info}")

        return (model,)


NODE_CLASS_MAPPINGS = {
    "NV_AnchorKVCache": NV_AnchorKVCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_AnchorKVCache": "NV Anchor KV Cache",
}
