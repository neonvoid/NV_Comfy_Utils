"""
WAN VRAM Estimator & Config Optimizer

Two nodes:
1. NV_WanVRAMEstimator — Accurate per-component VRAM estimation for a specific
   (model, resolution, frames) configuration.
2. NV_WanConfigOptimizer — Given hardware (VRAM + RAM), finds the optimal
   resolution × frame count × context window combination.

Extracts real model config at runtime (dim, ffn_dim, heads, layers, patch_size),
detects the active attention backend (flash/SDPA/sub-quad/etc.), and computes
peak VRAM from first principles.

Key insight: under torch.no_grad() inference, only ONE transformer layer's
activations are live at a time. The peak is:

    peak = model_weights + layer_peak + vace_overhead + conditioning + latent + overhead

Where layer_peak = max(self_attention_peak, ffn_peak) + residuals, computed per
the active attention backend's memory profile.
"""

import logging
import math
import psutil
import torch
from dataclasses import dataclass, field
from typing import Optional, List

import comfy.model_management

from .wan_memory_estimator import (
    format_bytes,
    get_model_size,
    get_vae_size,
    estimate_feat_cache_size,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Attention backend detection
# ---------------------------------------------------------------------------

@dataclass
class AttentionBackend:
    name: str
    complexity: str  # "O(N)" or "O(N*chunk)"
    multiplier: float  # applied to base attention memory estimate


def detect_attention_backend() -> AttentionBackend:
    """Detect the active ComfyUI attention backend and its memory profile.

    Flash/SDPA/sage: O(N) — no attention matrix materialized.
    Sub-quadratic: O(N*chunk) — dynamic chunking, higher working memory.
    Split: O(N*slice) — query slicing, moderate overhead.
    """
    if comfy.model_management.sage_attention_enabled():
        return AttentionBackend("sage", "O(N)", 1.0)

    if comfy.model_management.flash_attention_enabled():
        return AttentionBackend("flash", "O(N)", 1.0)

    if comfy.model_management.xformers_enabled():
        return AttentionBackend("xformers", "O(N)", 1.0)

    if comfy.model_management.pytorch_attention_enabled():
        if comfy.model_management.pytorch_attention_flash_attention():
            return AttentionBackend("pytorch_sdpa_flash", "O(N)", 1.0)
        return AttentionBackend("pytorch_sdpa", "O(N)", 1.2)

    # Check for split attention via args (no dedicated model_management check)
    try:
        from comfy.cli_args import args
        if getattr(args, 'use_split_cross_attention', False):
            return AttentionBackend("split", "O(N*slice)", 1.3)
    except ImportError:
        pass

    return AttentionBackend("sub_quadratic", "O(N*chunk)", 1.5)


# ---------------------------------------------------------------------------
# Model config extraction
# ---------------------------------------------------------------------------

@dataclass
class WanConfig:
    dim: int
    ffn_dim: int
    num_heads: int
    num_layers: int
    head_dim: int
    patch_size: tuple  # (t, h, w)
    in_dim: int  # input channels (16 for latent)
    is_vace: bool
    vace_layers: int
    vace_in_dim: int
    model_type: str  # 't2v', 'i2v', 'vace', etc.


def extract_wan_config(model) -> Optional[WanConfig]:
    """Extract WAN model architecture config from a ComfyUI MODEL.

    Navigates: model.model.diffusion_model → WanModel instance attributes.
    Returns None for non-WAN models (graceful fallback).
    """
    try:
        base_model = model.model
        dm = base_model.diffusion_model

        # WanModel stores these as instance attributes (set in __init__)
        if not hasattr(dm, 'dim') or not hasattr(dm, 'ffn_dim'):
            return None

        return WanConfig(
            dim=dm.dim,
            ffn_dim=dm.ffn_dim,
            num_heads=dm.num_heads,
            num_layers=dm.num_layers,
            head_dim=dm.dim // dm.num_heads,
            patch_size=tuple(dm.patch_size),
            in_dim=getattr(dm, 'in_dim', 16),
            is_vace=hasattr(dm, 'vace_blocks'),
            vace_layers=getattr(dm, 'vace_layers', 0) or 0,
            vace_in_dim=getattr(dm, 'vace_in_dim', 0) or 0,
            model_type=getattr(dm, 'model_type', 'unknown'),
        )
    except Exception as e:
        logger.warning(f"[WAN VRAM Estimator] Could not extract WAN config: {e}")
        return None


def get_model_dtype(model) -> torch.dtype:
    """Get the inference dtype of a ComfyUI MODEL."""
    try:
        base_model = model.model
        dtype = base_model.get_dtype()
        if base_model.manual_cast_dtype is not None:
            dtype = base_model.manual_cast_dtype
        return dtype
    except Exception:
        return torch.float16


from .chunk_utils import video_to_latent_frames as _vtl
from .chunk_utils import nearest_wan_aligned as _nwa


def _pixel_to_latent(pixel_frames):
    """WAN pixel→latent conversion: max(((px - 1) // 4) + 1, 1)"""
    return max(_vtl(pixel_frames), 1) if pixel_frames > 0 else 0


# ---------------------------------------------------------------------------
# Core estimation
# ---------------------------------------------------------------------------

@dataclass
class VRAMBreakdown:
    """Per-component VRAM breakdown for a single denoising step."""
    model_weights: int  # bytes
    self_attn_peak: int
    ffn_peak: int
    cross_attn_peak: int
    residuals: int
    patch_embed_fp32_peak: int  # fp32 Conv3d intermediate during patchify
    vace_overhead: int
    latent_tensor: int
    text_conditioning: int
    vace_conditioning: int
    comfyui_overhead: int
    safety_margin_bytes: int
    # Cross-reference: ComfyUI's own memory_required() estimate (activations only)
    comfyui_memory_required: int = 0

    @property
    def layer_peak(self) -> int:
        """Peak within a single transformer block (attention or FFN phase)."""
        return max(self.self_attn_peak, self.ffn_peak) + self.residuals

    @property
    def activation_peak(self) -> int:
        """Largest activation phase: max of transformer block peak vs patch embedding fp32 peak."""
        return max(self.layer_peak, self.patch_embed_fp32_peak)

    @property
    def subtotal(self) -> int:
        return (
            self.model_weights
            + self.activation_peak
            + self.vace_overhead
            + self.latent_tensor
            + self.text_conditioning
            + self.vace_conditioning
        )

    @property
    def total(self) -> int:
        return self.subtotal + self.comfyui_overhead + self.safety_margin_bytes

    @property
    def total_gb(self) -> float:
        return self.total / (1024 ** 3)


def estimate_inference_peak(
    config: WanConfig,
    model_weight_bytes: int,
    model_dtype: torch.dtype,
    total_pixel_frames: int,
    height: int,
    width: int,
    backend: AttentionBackend,
    use_vace: bool = False,
    num_vace_inputs: int = 1,
    use_cfg: bool = True,
    safety_margin_pct: float = 20.0,
    model=None,
) -> VRAMBreakdown:
    """Estimate peak VRAM for one denoising step from model config.

    Under torch.no_grad(), only ONE layer's activations are live at a time.
    The peak is dominated by the largest single-layer phase (attention or FFN)
    plus persistent tensors (weights, latent, conditioning).

    Key insight: the peak activation phase is max(transformer_block_peak,
    patch_embed_fp32_peak). The transformer block peak is dominated by the
    FFN phase due to GELU doubling: during nn.GELU, both input and output
    [B, seq, ffn_dim] coexist.
    """
    bpe = comfy.model_management.dtype_size(model_dtype)
    batch = 2 if use_cfg else 1

    # Spatial/temporal dimensions
    T_lat = _pixel_to_latent(total_pixel_frames)
    H_lat = height // 8
    W_lat = width // 8
    patch_t, patch_h, patch_w = config.patch_size

    # Sequence length after patchify (Conv3d with stride=patch_size)
    seq_len = (T_lat // max(patch_t, 1)) * (H_lat // max(patch_h, 1)) * (W_lat // max(patch_w, 1))

    # ---------------------------------------------------------------
    # Self-attention phase peak
    # ---------------------------------------------------------------
    # At the optimized_attention() call, these tensors coexist:
    #   x_outer (hidden state from block): [B, seq, dim]        = 1 unit
    #   x_attn  (addcmul result, attn input): [B, seq, dim]     = 1 unit
    #   Q: [B, seq, dim]                                         = 1 unit
    #   K: [B, seq, dim]                                         = 1 unit
    #   V (self.v(x) computed inline): [B, seq, dim]             = 1 unit
    # Total: 5 units of [B, seq, dim]. For flash/SDPA, no attention matrix.
    # For sub-quadratic, chunk overhead adds ~multiplier.
    # We report self_attn_peak as the Q+K+V+O portion (4 units) with backend
    # multiplier, and residuals as the persistent x_outer + x_attn (2 units).
    self_attn_peak = int(4 * batch * seq_len * config.dim * bpe * backend.multiplier)

    # ---------------------------------------------------------------
    # FFN phase peak (THIS IS TYPICALLY THE DOMINANT PHASE)
    # ---------------------------------------------------------------
    # Sequential: Linear(dim→ffn_dim) → GELU → Linear(ffn_dim→dim)
    # During nn.GELU(approximate='tanh'), both input AND output of shape
    # [B, seq, ffn_dim] coexist (GELU is NOT in-place).
    # Plus x_outer persists from the block.
    #
    # Peak during GELU:
    #   x_outer: [B, seq, dim]           = dim
    #   gelu_input: [B, seq, ffn_dim]    = ffn_dim
    #   gelu_output: [B, seq, ffn_dim]   = ffn_dim
    # Total: (dim + 2*ffn_dim) * batch * seq * bpe
    ffn_peak = int(batch * seq_len * (2 * config.ffn_dim + config.dim) * bpe)

    # ---------------------------------------------------------------
    # Cross-attention peak
    # ---------------------------------------------------------------
    # Q from hidden [B, seq, dim], K/V from text [B, text_len, dim]
    # Text context is small (~512 tokens, +257 for i2v CLIP)
    text_len = 512 + (257 if config.model_type == 'i2v' else 0)
    cross_attn_peak = int(
        batch * (2 * seq_len * config.dim + 2 * text_len * config.dim + seq_len * config.dim) * bpe
    )

    # ---------------------------------------------------------------
    # Residuals: tensors persisting OUTSIDE the attention/FFN call
    # ---------------------------------------------------------------
    # x_outer (block hidden state) + addcmul/norm intermediate + time embeddings
    # x_outer: [B, seq, dim], attn/FFN input: [B, seq, dim]
    residuals = int(batch * seq_len * config.dim * bpe * 2 + batch * 6 * config.dim * bpe)

    # ---------------------------------------------------------------
    # fp32 patch embedding phase (transient but can be large)
    # ---------------------------------------------------------------
    # WanModel.forward_orig line 555: x = self.patch_embedding(x.float()).to(x.dtype)
    # Conv3d runs in fp32, then .to(dtype) converts. During conversion both coexist:
    #   fp32 output: [B, dim, T_p, H_p, W_p] * 4 bytes
    #   bf16 copy:   [B, dim, T_p, H_p, W_p] * bpe bytes
    #   fp32 input:  [B, in_dim, T, H_lat, W_lat] * 4 bytes (small)
    fp32_conv_output = batch * config.dim * seq_len * 4  # fp32
    dtype_copy = batch * config.dim * seq_len * bpe  # bf16 copy during .to()
    fp32_input = batch * config.in_dim * T_lat * H_lat * W_lat * 4 if hasattr(config, 'in_dim') else 0
    patch_embed_fp32_peak = int(fp32_conv_output + dtype_copy + fp32_input)

    # ---------------------------------------------------------------
    # VACE overhead (VaceWanModel keeps extra tensors alive during block loop)
    # ---------------------------------------------------------------
    vace_overhead = 0
    if use_vace and config.is_vace and config.vace_layers > 0:
        # x_orig: kept alive for ALL blocks [batch, seq_len, dim]
        x_orig_mem = batch * seq_len * config.dim * bpe
        # c list: num_vace_inputs tensors of [batch, seq_len, dim] each
        c_list_mem = num_vace_inputs * batch * seq_len * config.dim * bpe
        # c_skip: temp per vace block [batch, seq_len, dim]
        c_skip_mem = batch * seq_len * config.dim * bpe
        # Vace block runs its own attention+FFN at same peak as a regular block
        vace_block_peak = max(self_attn_peak, ffn_peak) + residuals
        # vace_patch_embedding also runs in fp32 (line 775 of model.py)
        vace_patch_embed = num_vace_inputs * batch * config.dim * seq_len * 4  # fp32 intermediate
        vace_overhead = int(x_orig_mem + c_list_mem + c_skip_mem + vace_block_peak + vace_patch_embed)

    # ---------------------------------------------------------------
    # Conditioning tensors (persistent during sampling)
    # ---------------------------------------------------------------
    # Latent: [batch, 16, T_lat, H_lat, W_lat]
    latent_tensor = int(batch * 16 * T_lat * H_lat * W_lat * bpe)

    # Text embeddings: [batch, text_len, dim]
    text_conditioning = int(batch * text_len * config.dim * bpe)

    # VACE conditioning: vace_frames [N, 32, T, H_lat, W_lat] + vace_mask [N, 64, T, H_lat, W_lat]
    vace_conditioning = 0
    if use_vace:
        vace_conditioning = int(num_vace_inputs * (32 + 64) * T_lat * H_lat * W_lat * bpe)

    # ---------------------------------------------------------------
    # CUDA/ComfyUI overhead (scales with workload)
    # ---------------------------------------------------------------
    # Base: CUDA context + ComfyUI model management (~700-1000 MB on Windows)
    # Plus: CUDA caching allocator fragmentation (~8% of working set).
    # The allocator holds freed blocks in a cache, and fragmentation from
    # varying-sized allocations means reserved >> allocated (typically 1.2-1.3x).
    base_overhead = int(800 * 1024 * 1024)  # 800 MB base
    # Estimate working set (everything except weights, which don't fragment much)
    activation_working_set = max(self_attn_peak, ffn_peak) + residuals + latent_tensor + text_conditioning + vace_overhead
    fragmentation_overhead = int(activation_working_set * 0.10)  # 10% fragmentation
    comfyui_overhead = base_overhead + fragmentation_overhead

    # ---------------------------------------------------------------
    # Cross-reference: ComfyUI's own memory_required() estimate
    # ---------------------------------------------------------------
    comfyui_memory_required = 0
    if model is not None:
        try:
            input_shape = [batch, 16, T_lat, H_lat, W_lat]
            comfyui_memory_required = int(model.model.memory_required(input_shape))
        except Exception:
            pass

    # Build breakdown
    breakdown = VRAMBreakdown(
        model_weights=model_weight_bytes,
        self_attn_peak=self_attn_peak,
        ffn_peak=ffn_peak,
        cross_attn_peak=cross_attn_peak,
        residuals=residuals,
        patch_embed_fp32_peak=patch_embed_fp32_peak,
        vace_overhead=vace_overhead,
        latent_tensor=latent_tensor,
        text_conditioning=text_conditioning,
        vace_conditioning=vace_conditioning,
        comfyui_overhead=comfyui_overhead,
        safety_margin_bytes=0,  # computed below
        comfyui_memory_required=comfyui_memory_required,
    )

    # Safety margin applied to subtotal (not overhead)
    margin = int(breakdown.subtotal * safety_margin_pct / 100.0)
    breakdown.safety_margin_bytes = margin

    return breakdown


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(
    config: Optional[WanConfig],
    backend: AttentionBackend,
    breakdown: VRAMBreakdown,
    available_vram_bytes: int,
    total_pixel_frames: int,
    height: int,
    width: int,
    model_dtype: torch.dtype,
    use_cfg: bool,
    fits: bool,
    measured_peak_bytes: int = 0,
) -> str:
    """Build a human-readable VRAM estimation report."""
    lines = []
    lines.append("=" * 60)
    lines.append("  WAN VRAM Estimation Report")
    lines.append("=" * 60)

    if config is not None:
        T_lat = _pixel_to_latent(total_pixel_frames)
        H_lat = height // 8
        W_lat = width // 8
        pt, ph, pw = config.patch_size
        seq_len = (T_lat // max(pt, 1)) * (H_lat // max(ph, 1)) * (W_lat // max(pw, 1))

        model_desc = "VACE" if config.is_vace else config.model_type.upper()
        param_b = breakdown.model_weights / (1024 ** 3)
        lines.append(f"  Model: WAN {model_desc} ({param_b:.1f} GB weights)")
        lines.append(f"  Config: dim={config.dim} ffn={config.ffn_dim} "
                      f"heads={config.num_heads} layers={config.num_layers}")
        lines.append(f"  Patch: {config.patch_size}  dtype: {model_dtype}")
        if config.is_vace:
            lines.append(f"  VACE layers: {config.vace_layers}")
        lines.append(f"  Attention: {backend.name} [{backend.complexity}]"
                      f"{f' (x{backend.multiplier})' if backend.multiplier != 1.0 else ''}")
        lines.append("")
        lines.append(f"  Resolution: {width}x{height} (latent {W_lat}x{H_lat})")
        lines.append(f"  Frames: {total_pixel_frames} pixel -> {T_lat} latent")
        lines.append(f"  Sequence length: {seq_len:,} tokens")
        lines.append(f"  CFG: {'Yes (batch=2)' if use_cfg else 'No (batch=1)'}")
    else:
        lines.append("  Model: Non-WAN (using fallback estimation)")

    lines.append("")
    lines.append("  --- Memory Breakdown (inference peak) ---")
    fb = format_bytes

    items = [
        ("Model weights", breakdown.model_weights),
        ("Self-attention peak", breakdown.self_attn_peak),
        ("FFN peak (w/ GELU 2x)", breakdown.ffn_peak),
        ("Cross-attention", breakdown.cross_attn_peak),
        ("Residuals (x + norm)", breakdown.residuals),
        ("Patch embed fp32 peak", breakdown.patch_embed_fp32_peak),
    ]
    if breakdown.vace_overhead > 0:
        items.append(("VACE overhead", breakdown.vace_overhead))
    items += [
        ("Latent tensor", breakdown.latent_tensor),
        ("Text conditioning", breakdown.text_conditioning),
    ]
    if breakdown.vace_conditioning > 0:
        items.append(("VACE conditioning", breakdown.vace_conditioning))
    items += [
        ("CUDA/ComfyUI overhead", breakdown.comfyui_overhead),
        ("Safety margin", breakdown.safety_margin_bytes),
    ]

    for label, val in items:
        lines.append(f"    {label:<26s} {fb(val):>12s}")

    lines.append(f"    {'─' * 40}")

    # Indicate which phase dominates
    dominant_layer = "attention" if breakdown.self_attn_peak >= breakdown.ffn_peak else "FFN+GELU"
    dominant_phase = "transformer" if breakdown.layer_peak >= breakdown.patch_embed_fp32_peak else "patch_embed_fp32"
    lines.append(f"    Layer bottleneck: {dominant_layer}")
    lines.append(f"    Activation bottleneck: {dominant_phase}")
    lines.append(f"    {'TOTAL ESTIMATED':<26s} {fb(breakdown.total):>12s}")

    # Cross-references
    lines.append("")
    lines.append("  --- Cross-References ---")
    if breakdown.comfyui_memory_required > 0:
        cmr_gb = breakdown.comfyui_memory_required / (1024 ** 3)
        our_act = breakdown.activation_peak / (1024 ** 3)
        ratio = our_act / cmr_gb if cmr_gb > 0 else 0
        lines.append(f"    ComfyUI memory_required(): {fb(breakdown.comfyui_memory_required)} (activations only)")
        lines.append(f"    Our activation estimate:   {fb(breakdown.activation_peak)}")
        lines.append(f"    Ratio (ours / ComfyUI):    {ratio:.2f}x")
        lines.append(f"    Note: ComfyUI uses 1.5x safety when batching")

    if measured_peak_bytes > 0:
        measured_gb = measured_peak_bytes / (1024 ** 3)
        est_gb = breakdown.total_gb
        accuracy = (est_gb / measured_gb * 100) if measured_gb > 0 else 0
        lines.append("")
        lines.append(f"  >>> MEASURED peak VRAM:  {measured_gb:.2f} GB")
        lines.append(f"  >>> ESTIMATED:           {est_gb:.2f} GB")
        lines.append(f"  >>> Accuracy:            {accuracy:.0f}%")
        if est_gb < measured_gb:
            lines.append(f"  >>> UNDERESTIMATE by {measured_gb - est_gb:.2f} GB")
        else:
            lines.append(f"  >>> Overestimate by {est_gb - measured_gb:.2f} GB (conservative)")

    available_gb = available_vram_bytes / (1024 ** 3)
    headroom = available_vram_bytes - breakdown.total
    lines.append("")
    lines.append(f"  Available VRAM: {available_gb:.1f} GB")
    lines.append(f"  Headroom: {format_bytes(headroom) if headroom >= 0 else '-' + format_bytes(-headroom)}")
    lines.append("")
    lines.append("=" * 60)
    if fits:
        lines.append("  FITS - full video in single pass (no context windows)")
    else:
        lines.append("  DOES NOT FIT - context windows required")
    lines.append("=" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class NV_WanVRAMEstimator:
    """
    Accurate VRAM estimator for WAN model inference.

    Extracts real model config (dim, ffn_dim, heads, layers, patch_size),
    detects the active attention backend, and computes per-component VRAM
    breakdown from first principles.

    Primary output: fits_without_context_windows — wire to
    NV WAN Context Windows → bypass input to auto-skip chunking
    when the full video fits in VRAM.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "WAN model — config is extracted at runtime for "
                               "accurate estimation. Also measures actual weight bytes."
                }),
                "total_frames": ("INT", {
                    "default": 81, "min": 1, "max": 10000, "step": 1,
                    "tooltip": "Total pixel frames in the video to generate."
                }),
                "height": ("INT", {
                    "default": 480, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Video height in pixels."
                }),
                "width": ("INT", {
                    "default": 832, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Video width in pixels."
                }),
            },
            "optional": {
                "images": ("IMAGE", {
                    "tooltip": "Control/input video — auto-extracts frame count "
                               "and overrides total_frames if connected."
                }),
                "vae": ("VAE", {
                    "tooltip": "VAE model for encoding-phase feat_cache estimation."
                }),
                "available_vram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 512.0, "step": 0.1,
                    "tooltip": "0 = auto-detect from GPU. Set manually for planning "
                               "(e.g. 192 for B200, 24 for RTX 4090)."
                }),
                "use_vace": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Account for VACE conditioning memory overhead."
                }),
                "num_vace_inputs": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1,
                    "tooltip": "Number of VACE conditioning inputs."
                }),
                "safety_margin_percent": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0,
                    "tooltip": "Safety margin added to estimate (percentage of subtotal). "
                               "20%% accounts for CUDA allocator fragmentation and "
                               "transient intermediates not captured analytically."
                }),
                "use_cfg": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Whether CFG is used (doubles batch dimension for activations)."
                }),
                "run_profiler": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run one forward pass on dummy data to measure actual peak VRAM. "
                               "Takes ~5-10s. Compares measured vs estimated and outputs a "
                               "calibration_factor you can use to tune safety margins."
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "STRING",)
    RETURN_NAMES = (
        "fits_without_context_windows",
        "estimated_peak_gb",
        "available_vram_gb",
        "headroom_gb",
        "measured_peak_gb",
        "calibration_factor",
        "vram_report",
    )
    OUTPUT_NODE = True
    FUNCTION = "estimate"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Accurate VRAM estimator for WAN inference. Extracts real model config, "
        "detects the attention backend, and computes per-component memory breakdown. "
        "Wire fits_without_context_windows to NV WAN Context Windows → bypass."
    )

    def estimate(
        self,
        model,
        total_frames,
        height,
        width,
        images=None,
        vae=None,
        available_vram_gb=0.0,
        use_vace=False,
        num_vace_inputs=1,
        safety_margin_percent=20.0,
        use_cfg=True,
        run_profiler=False,
    ):
        # --- Resolve frame count (IMAGE overrides total_frames) ---
        if images is not None:
            total_frames = images.shape[0]

        # --- Detect attention backend ---
        backend = detect_attention_backend()

        # --- Extract model config ---
        config = extract_wan_config(model)
        model_dtype = get_model_dtype(model)
        weight_bytes = get_model_size(model)

        # --- Auto-detect VRAM ---
        if available_vram_gb <= 0:
            if torch.cuda.is_available():
                available_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            else:
                available_vram_bytes = int(8 * 1024 ** 3)
        else:
            available_vram_bytes = int(available_vram_gb * 1024 ** 3)

        # --- Estimate ---
        if config is not None:
            breakdown = estimate_inference_peak(
                config=config,
                model_weight_bytes=weight_bytes,
                model_dtype=model_dtype,
                total_pixel_frames=total_frames,
                height=height,
                width=width,
                backend=backend,
                use_vace=use_vace,
                num_vace_inputs=num_vace_inputs,
                use_cfg=use_cfg,
                safety_margin_pct=safety_margin_percent,
                model=model,
            )
        else:
            # Non-WAN fallback: use ComfyUI's generic memory_required()
            logger.warning("[WAN VRAM Estimator] Non-WAN model — using generic estimation")
            T_lat = _pixel_to_latent(total_frames)
            H_lat = height // 8
            W_lat = width // 8
            input_shape = (2 if use_cfg else 1, 16, T_lat, H_lat, W_lat)
            try:
                generic_bytes = int(model.model.memory_required(input_shape))
            except Exception:
                generic_bytes = 0

            breakdown = VRAMBreakdown(
                model_weights=weight_bytes,
                self_attn_peak=generic_bytes,
                ffn_peak=0,
                cross_attn_peak=0,
                residuals=0,
                patch_embed_fp32_peak=0,
                vace_overhead=0,
                latent_tensor=0,
                text_conditioning=0,
                vace_conditioning=0,
                comfyui_overhead=int(800 * 1024 * 1024),
                safety_margin_bytes=int(generic_bytes * safety_margin_percent / 100.0),
            )

        # --- Optional profiler: run one forward pass to measure actual peak ---
        measured_peak = 0
        profiler_error = ""
        if run_profiler:
            logger.info("[WAN VRAM Estimator] Running profiler — one forward pass on dummy data...")
            measured_peak, profiler_error = _profile_forward_pass(
                model, total_frames, height, width, use_cfg, model_dtype
            )
            if profiler_error:
                logger.warning(f"[WAN VRAM Estimator] Profiler: {profiler_error}")
        elif torch.cuda.is_available():
            # Fall back to reading peak from any previous CUDA work in this session
            try:
                measured_peak = torch.cuda.max_memory_allocated()
            except Exception:
                pass

        fits = breakdown.total <= available_vram_bytes
        headroom_bytes = available_vram_bytes - breakdown.total

        report = format_report(
            config=config,
            backend=backend,
            breakdown=breakdown,
            available_vram_bytes=available_vram_bytes,
            total_pixel_frames=total_frames,
            height=height,
            width=width,
            model_dtype=model_dtype,
            use_cfg=use_cfg,
            fits=fits,
            measured_peak_bytes=measured_peak,
        )

        # Append profiler status to report
        if run_profiler:
            if profiler_error:
                report += f"\n  Profiler error: {profiler_error}"
            else:
                report += "\n  Profiler: measured via single forward pass (dummy data)"

        # Print to console and show in UI
        print(report)

        estimated_gb = round(breakdown.total_gb, 3)
        avail_gb = round(available_vram_bytes / (1024 ** 3), 3)
        headroom_gb = round(headroom_bytes / (1024 ** 3), 3)
        measured_gb = round(measured_peak / (1024 ** 3), 3) if measured_peak > 0 else 0.0
        calibration = round(measured_gb / estimated_gb, 3) if estimated_gb > 0 and measured_gb > 0 else 0.0

        return {
            "ui": {"text": [report]},
            "result": (
                fits,
                estimated_gb,
                avail_gb,
                headroom_gb,
                measured_gb,
                calibration,
                report,
            ),
        }


# ---------------------------------------------------------------------------
# Config optimizer — search resolution × frames × CW frontier
# ---------------------------------------------------------------------------

# WAN temporal alignment: valid frame counts are 4k+1
def _align_wan_frames(n):
    """Round down to nearest valid WAN frame count (4k+1)."""
    return max(1, _nwa(n))


# Common aspect ratios as (name, w/h ratio)
ASPECT_RATIOS = {
    "16:9": 16.0 / 9.0,
    "9:16": 9.0 / 16.0,
    "4:3": 4.0 / 3.0,
    "3:4": 3.0 / 4.0,
    "1:1": 1.0,
    "21:9": 21.0 / 9.0,
    "9:21": 9.0 / 21.0,
}


def _generate_resolution_candidates(aspect_ratio_name, min_h, max_h):
    """Generate (height, width) pairs for an aspect ratio, aligned to 16px."""
    ratio = ASPECT_RATIOS.get(aspect_ratio_name, 16.0 / 9.0)
    candidates = []
    # Step by 16 for fine granularity, but skip near-duplicates
    for h in range(min_h, max_h + 1, 16):
        w = round(h * ratio / 16) * 16
        if w >= 64:
            candidates.append((h, w))
    return candidates


def _generate_frame_candidates(min_frames, max_frames):
    """Generate valid WAN frame counts (4k+1) in range."""
    candidates = []
    n = _align_wan_frames(min_frames)
    if n < min_frames:
        n += 4
    while n <= max_frames:
        candidates.append(n)
        n += 4
    return candidates


def _generate_cw_candidates():
    """Context window sizes to try (largest first = best quality)."""
    return [513, 385, 321, 257, 193, 161, 129, 113, 97, 81, 65, 49, 33, 17]


def estimate_ram_usage(total_pixel_frames, height, width, vae=None):
    """Estimate system RAM needed for VAE encoding + output accumulation.

    Components:
    1. VAE feat_cache (causal conv cache, the bottleneck for long videos)
    2. Decoded output frames on CPU: [frames, H, W, 3] in fp32
    3. Encoded latent on CPU: [1, 16, T_lat, H/8, W/8] in fp16
    """
    # VAE feat_cache (the primary RAM/VRAM concern during encoding)
    feat_cache = estimate_feat_cache_size(vae, total_pixel_frames, height, width)

    # Output video on CPU: [frames, H, W, 3] fp32
    output_bytes = total_pixel_frames * height * width * 3 * 4

    # Latent on CPU: [1, 16, T_lat, H/8, W/8] fp16
    T_lat = _pixel_to_latent(total_pixel_frames)
    latent_bytes = 16 * T_lat * (height // 8) * (width // 8) * 2

    return feat_cache + output_bytes + latent_bytes


@dataclass
class ConfigCandidate:
    """One viable (resolution, frames, context_window) configuration."""
    height: int
    width: int
    total_frames: int
    context_window: int  # 0 = no context windows needed
    context_overlap: int
    vram_peak_gb: float
    vram_headroom_gb: float
    ram_usage_gb: float
    ram_headroom_gb: float
    total_pixels: int = field(init=False)
    quality_score: float = field(init=False)

    def __post_init__(self):
        self.total_pixels = self.height * self.width
        # Quality score: resolution × frames, with bonus for no-CW
        # Higher CW is better when CW is needed (gentler blending)
        cw_factor = 1.3 if self.context_window == 0 else (self.context_window / 81.0) * 0.8
        self.quality_score = self.total_pixels * self.total_frames * cw_factor


def find_optimal_configs(
    config: WanConfig,
    model_weight_bytes: int,
    model_dtype: torch.dtype,
    backend: AttentionBackend,
    available_vram_bytes: int,
    available_ram_bytes: int,
    aspect_ratio: str = "16:9",
    min_height: int = 480,
    max_height: int = 1080,
    min_frames: int = 17,
    max_frames: int = 513,
    use_vace: bool = False,
    num_vace_inputs: int = 1,
    use_cfg: bool = True,
    safety_margin_pct: float = 15.0,
    vae=None,
    max_results: int = 20,
) -> List[ConfigCandidate]:
    """Search the resolution × frames × CW frontier for viable configs.

    Strategy:
    1. For each resolution (highest first):
       For each frame count (highest first):
         a. Try full-pass (no CW) first — best quality
         b. If doesn't fit, try decreasing CW sizes
         c. Also check RAM for VAE encoding
    2. Collect all viable configs, sort by quality score
    3. Return top N
    """
    resolutions = _generate_resolution_candidates(aspect_ratio, min_height, max_height)
    resolutions.reverse()  # Highest first
    frame_candidates = _generate_frame_candidates(min_frames, max_frames)
    frame_candidates.reverse()  # Most frames first
    cw_candidates = _generate_cw_candidates()

    viable = []

    for h, w in resolutions:
        for frames in frame_candidates:
            # Check RAM first (VAE encoding is often the real bottleneck)
            ram_bytes = estimate_ram_usage(frames, h, w, vae)
            ram_headroom = available_ram_bytes - ram_bytes
            if ram_headroom < 0:
                continue  # This (frames, resolution) combo won't fit in RAM

            # Try full-pass (no context windows)
            breakdown = estimate_inference_peak(
                config=config,
                model_weight_bytes=model_weight_bytes,
                model_dtype=model_dtype,
                total_pixel_frames=frames,
                height=h,
                width=w,
                backend=backend,
                use_vace=use_vace,
                num_vace_inputs=num_vace_inputs,
                use_cfg=use_cfg,
                safety_margin_pct=safety_margin_pct,
            )

            if breakdown.total <= available_vram_bytes:
                # Full video fits — best case
                viable.append(ConfigCandidate(
                    height=h,
                    width=w,
                    total_frames=frames,
                    context_window=0,
                    context_overlap=0,
                    vram_peak_gb=breakdown.total_gb,
                    vram_headroom_gb=(available_vram_bytes - breakdown.total) / (1024 ** 3),
                    ram_usage_gb=ram_bytes / (1024 ** 3),
                    ram_headroom_gb=ram_headroom / (1024 ** 3),
                ))
                continue

            # Full pass doesn't fit — try context windows (largest first)
            for cw in cw_candidates:
                if cw >= frames:
                    continue  # CW must be smaller than total frames

                # Estimate with context window (CW determines activation size)
                cw_breakdown = estimate_inference_peak(
                    config=config,
                    model_weight_bytes=model_weight_bytes,
                    model_dtype=model_dtype,
                    total_pixel_frames=cw,  # CW size determines per-window cost
                    height=h,
                    width=w,
                    backend=backend,
                    use_vace=use_vace,
                    num_vace_inputs=num_vace_inputs,
                    use_cfg=use_cfg,
                    safety_margin_pct=safety_margin_pct,
                )

                if cw_breakdown.total <= available_vram_bytes:
                    # This CW fits — use it
                    co = max(8, cw // 4)
                    co = _align_wan_frames(co)
                    viable.append(ConfigCandidate(
                        height=h,
                        width=w,
                        total_frames=frames,
                        context_window=cw,
                        context_overlap=co,
                        vram_peak_gb=cw_breakdown.total_gb,
                        vram_headroom_gb=(available_vram_bytes - cw_breakdown.total) / (1024 ** 3),
                        ram_usage_gb=ram_bytes / (1024 ** 3),
                        ram_headroom_gb=ram_headroom / (1024 ** 3),
                    ))
                    break  # Largest viable CW found for this (res, frames)

    # Sort by quality score (highest first)
    viable.sort(key=lambda c: c.quality_score, reverse=True)
    return viable[:max_results]


def format_config_table(configs: List[ConfigCandidate], available_vram_gb: float,
                        available_ram_gb: float) -> str:
    """Format a ranked table of viable configurations."""
    lines = []
    lines.append("=" * 80)
    lines.append("  WAN Config Optimizer — Viable Configurations")
    lines.append(f"  VRAM: {available_vram_gb:.1f} GB  |  RAM: {available_ram_gb:.1f} GB")
    lines.append("=" * 80)

    if not configs:
        lines.append("  No viable configurations found for this hardware.")
        lines.append("  Try: lower min_resolution, fewer frames, or more VRAM.")
        return "\n".join(lines)

    # Header
    lines.append(f"  {'#':>3s}  {'Resolution':>11s}  {'Frames':>6s}  {'CW':>5s}  "
                 f"{'CO':>4s}  {'VRAM Peak':>10s}  {'Headroom':>9s}  {'RAM':>8s}  Note")
    lines.append(f"  {'─' * 76}")

    for i, c in enumerate(configs):
        res = f"{c.width}×{c.height}"
        cw_str = "FULL" if c.context_window == 0 else str(c.context_window)
        co_str = "—" if c.context_window == 0 else str(c.context_overlap)
        vram_str = f"{c.vram_peak_gb:.1f} GB"
        head_str = f"+{c.vram_headroom_gb:.1f} GB"
        ram_str = f"{c.ram_usage_gb:.1f} GB"

        note = ""
        if c.context_window == 0:
            note = "single pass"
        elif c.vram_headroom_gb < 2.0:
            note = "tight"

        rank = f"#{i+1}"
        lines.append(f"  {rank:>3s}  {res:>11s}  {c.total_frames:>6d}  {cw_str:>5s}  "
                     f"{co_str:>4s}  {vram_str:>10s}  {head_str:>9s}  {ram_str:>8s}  {note}")

    lines.append("")

    # Highlight the top pick
    top = configs[0]
    lines.append(f"  RECOMMENDED: {top.width}×{top.height} @ {top.total_frames} frames")
    if top.context_window == 0:
        lines.append(f"    No context windows needed — full single-pass render")
    else:
        lines.append(f"    Context window: {top.context_window}  overlap: {top.context_overlap}")
    lines.append(f"    VRAM: {top.vram_peak_gb:.1f} GB (headroom: {top.vram_headroom_gb:.1f} GB)")
    lines.append(f"    RAM: {top.ram_usage_gb:.1f} GB (headroom: {top.ram_headroom_gb:.1f} GB)")

    return "\n".join(lines)


class NV_WanConfigOptimizer:
    """
    Hardware-aware configuration optimizer for WAN workflows.

    Given your GPU VRAM + system RAM, searches the resolution × frame count ×
    context window frontier to find the optimal configuration. Outputs are
    directly pluggable into other workflow nodes.

    Two modes:
    - Fixed frames: Set target_frames > 0 to find the best resolution + CW
      for a specific frame count.
    - Full search: Set target_frames = 0 to explore all combinations and
      get a ranked table of viable configs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "WAN model for architecture config extraction."
                }),
                "aspect_ratio": (list(ASPECT_RATIOS.keys()), {
                    "default": "16:9",
                    "tooltip": "Target aspect ratio for resolution search."
                }),
            },
            "optional": {
                "vae": ("VAE", {
                    "tooltip": "VAE for encoding-phase RAM estimation (feat_cache)."
                }),
                "target_frames": ("INT", {
                    "default": 0, "min": 0, "max": 10000, "step": 1,
                    "tooltip": "Target frame count. 0 = search all frame counts. "
                               "> 0 = find best resolution + CW for this frame count."
                }),
                "min_height": ("INT", {
                    "default": 480, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Minimum video height to consider."
                }),
                "max_height": ("INT", {
                    "default": 1080, "min": 64, "max": 4096, "step": 16,
                    "tooltip": "Maximum video height to consider."
                }),
                "min_frames": ("INT", {
                    "default": 17, "min": 1, "max": 10000, "step": 4,
                    "tooltip": "Minimum frame count to consider (ignored if target_frames > 0)."
                }),
                "max_frames": ("INT", {
                    "default": 513, "min": 1, "max": 10000, "step": 4,
                    "tooltip": "Maximum frame count to consider (ignored if target_frames > 0)."
                }),
                "available_vram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 512.0, "step": 0.1,
                    "tooltip": "0 = auto-detect. Set manually for planning."
                }),
                "available_ram_gb": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 2048.0, "step": 0.1,
                    "tooltip": "0 = auto-detect system RAM. Set manually for planning."
                }),
                "use_vace": ("BOOLEAN", {"default": False}),
                "num_vace_inputs": ("INT", {"default": 1, "min": 1, "max": 8}),
                "use_cfg": ("BOOLEAN", {"default": True}),
                "safety_margin_percent": ("FLOAT", {
                    "default": 20.0, "min": 0.0, "max": 50.0, "step": 1.0,
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "BOOLEAN", "STRING",)
    RETURN_NAMES = (
        "optimal_height",
        "optimal_width",
        "optimal_frames",
        "optimal_context_window",
        "optimal_context_overlap",
        "fits_without_context_windows",
        "config_report",
    )
    OUTPUT_NODE = True
    FUNCTION = "optimize"
    CATEGORY = "NV_Utils"
    DESCRIPTION = (
        "Hardware-aware config optimizer. Searches resolution × frames × CW "
        "frontier to find the best configuration for your VRAM + RAM. "
        "Outputs are directly pluggable into workflow nodes."
    )

    def optimize(
        self,
        model,
        aspect_ratio,
        vae=None,
        target_frames=0,
        min_height=480,
        max_height=1080,
        min_frames=17,
        max_frames=513,
        available_vram_gb=0.0,
        available_ram_gb=0.0,
        use_vace=False,
        num_vace_inputs=1,
        use_cfg=True,
        safety_margin_percent=15.0,
    ):
        # --- Extract model info ---
        config = extract_wan_config(model)
        if config is None:
            report = "ERROR: Non-WAN model detected. Config optimizer requires a WAN model."
            print(report)
            return {"ui": {"text": [report]}, "result": (480, 832, 81, 81, 21, False, report)}

        model_dtype = get_model_dtype(model)
        weight_bytes = get_model_size(model)
        backend = detect_attention_backend()

        # --- Auto-detect hardware ---
        if available_vram_gb <= 0:
            if torch.cuda.is_available():
                available_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            else:
                available_vram_bytes = int(8 * 1024 ** 3)
        else:
            available_vram_bytes = int(available_vram_gb * 1024 ** 3)

        if available_ram_gb <= 0:
            available_ram_bytes = psutil.virtual_memory().total
        else:
            available_ram_bytes = int(available_ram_gb * 1024 ** 3)

        actual_vram_gb = available_vram_bytes / (1024 ** 3)
        actual_ram_gb = available_ram_bytes / (1024 ** 3)

        # --- Fixed frames mode vs full search ---
        if target_frames > 0:
            aligned_frames = _align_wan_frames(target_frames)
            search_min_frames = aligned_frames
            search_max_frames = aligned_frames
        else:
            search_min_frames = min_frames
            search_max_frames = max_frames

        # --- Run search ---
        configs = find_optimal_configs(
            config=config,
            model_weight_bytes=weight_bytes,
            model_dtype=model_dtype,
            backend=backend,
            available_vram_bytes=available_vram_bytes,
            available_ram_bytes=available_ram_bytes,
            aspect_ratio=aspect_ratio,
            min_height=min_height,
            max_height=max_height,
            min_frames=search_min_frames,
            max_frames=search_max_frames,
            use_vace=use_vace,
            num_vace_inputs=num_vace_inputs,
            use_cfg=use_cfg,
            safety_margin_pct=safety_margin_percent,
            vae=vae,
        )

        # --- Build report ---
        header_lines = []
        model_desc = "VACE" if config.is_vace else config.model_type.upper()
        param_gb = weight_bytes / (1024 ** 3)
        header_lines.append(f"  Model: WAN {model_desc} ({param_gb:.1f} GB)")
        header_lines.append(f"  Attention: {backend.name} [{backend.complexity}]")
        header_lines.append(f"  Aspect ratio: {aspect_ratio}")
        if target_frames > 0:
            header_lines.append(f"  Target frames: {target_frames} (aligned: {_align_wan_frames(target_frames)})")
        header_lines.append("")

        table = format_config_table(configs, actual_vram_gb, actual_ram_gb)
        report = "\n".join(header_lines) + "\n" + table
        print(report)

        # --- Extract top result ---
        if configs:
            top = configs[0]
            return {
                "ui": {"text": [report]},
                "result": (
                    top.height,
                    top.width,
                    top.total_frames,
                    top.context_window,
                    top.context_overlap,
                    top.context_window == 0,
                    report,
                ),
            }
        else:
            return {
                "ui": {"text": [report]},
                "result": (
                    min_height,
                    round(min_height * ASPECT_RATIOS.get(aspect_ratio, 16/9) / 16) * 16,
                    _align_wan_frames(min_frames),
                    17,
                    5,
                    False,
                    report,
                ),
            }


# ---------------------------------------------------------------------------
# VRAM Profiler — Empirical validation via actual forward pass
# ---------------------------------------------------------------------------

def _profile_forward_pass(model, total_pixel_frames, height, width, use_cfg, model_dtype):
    """Run a single forward pass on dummy data and measure actual peak VRAM.

    This creates minimal dummy inputs, resets CUDA peak memory stats, runs
    one denoising step through the model, and returns the measured peak.

    Returns (measured_peak_bytes, error_message). On success, error_message is empty.
    """
    if not torch.cuda.is_available():
        return 0, "CUDA not available"

    try:
        device = comfy.model_management.get_torch_device()
        T_lat = _pixel_to_latent(total_pixel_frames)
        H_lat = height // 8
        W_lat = width // 8
        batch = 2 if use_cfg else 1

        # Create dummy inputs matching what calc_cond_batch would pass
        dummy_latent = torch.randn(
            batch, 16, T_lat, H_lat, W_lat,
            device=device, dtype=model_dtype
        )
        dummy_timestep = torch.tensor([999.0] * batch, device=device, dtype=torch.float32)

        # Minimal conditioning: just cross_attn (text embeddings)
        config = extract_wan_config(model)
        dim = config.dim if config else 2048
        text_len = 512
        dummy_context = torch.randn(batch, text_len, dim, device=device, dtype=model_dtype)

        # Build conditioning dict matching ComfyUI's format
        c = {
            'c_crossattn': dummy_context,
            'transformer_options': {},
        }

        # Ensure model is on GPU
        comfy.model_management.load_model_gpu(model)

        # Reset peak stats and run forward pass
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        with torch.no_grad():
            try:
                _ = model.model.apply_model(dummy_latent, dummy_timestep, **c)
            except Exception as e:
                # Some models need specific conditioning — fall back to just
                # measuring with whatever we can
                return 0, f"Forward pass failed: {e}"

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()

        # Clean up
        del dummy_latent, dummy_timestep, dummy_context, c
        torch.cuda.empty_cache()

        return peak, ""

    except Exception as e:
        return 0, f"Profiling error: {e}"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "NV_WanVRAMEstimator": NV_WanVRAMEstimator,
    "NV_WanConfigOptimizer": NV_WanConfigOptimizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_WanVRAMEstimator": "NV WAN VRAM Estimator",
    "NV_WanConfigOptimizer": "NV WAN Config Optimizer",
}
