"""Manifest summarization helpers — extracted from render_manifest.py.

Pure functions. No side effects. Imported by both render_manifest.py
(legacy node) and shot_telemetry.py (new node).
"""

import os
import time


def summarize_stitcher(stitcher):
    """Extract a structured summary of stitcher dict metadata."""
    if stitcher is None:
        return None
    out = {}
    try:
        ctc_x = stitcher.get('cropped_to_canvas_x', None)
        ctc_y = stitcher.get('cropped_to_canvas_y', None)
        ctc_w = stitcher.get('cropped_to_canvas_w', None)
        ctc_h = stitcher.get('cropped_to_canvas_h', None)
        if ctc_x is not None and ctc_w is not None:
            n = len(ctc_x)
            out['frame_count'] = n
            if n > 0:
                out['bbox_x_range'] = [int(min(ctc_x)), int(max(ctc_x))]
                out['bbox_y_range'] = [int(min(ctc_y)), int(max(ctc_y))]
                out['bbox_w_range'] = [int(min(ctc_w)), int(max(ctc_w))]
                out['bbox_h_range'] = [int(min(ctc_h)), int(max(ctc_h))]
                out['bbox_first'] = [int(ctc_x[0]), int(ctc_y[0]), int(ctc_w[0]), int(ctc_h[0])]
                out['bbox_last'] = [int(ctc_x[-1]), int(ctc_y[-1]), int(ctc_w[-1]), int(ctc_h[-1])]

        canvas_images = stitcher.get('canvas_image', None)
        if canvas_images is not None and len(canvas_images) > 0:
            cf = canvas_images[0]
            if hasattr(cf, 'shape'):
                if cf.dim() == 3:
                    out['canvas_dims'] = [int(cf.shape[0]), int(cf.shape[1])]
                elif cf.dim() == 4:
                    out['canvas_dims'] = [int(cf.shape[1]), int(cf.shape[2])]

        cwm = stitcher.get('content_warp_mode', None)
        out['content_warp_mode'] = str(cwm) if cwm is not None else None

        cwd = stitcher.get('content_warp_data', None)
        if cwd is not None:
            out['content_warp_present'] = True
            if isinstance(cwd, list):
                out['content_warp_frames'] = len(cwd)
        else:
            out['content_warp_present'] = False

        ra = stitcher.get('resize_algorithm', None)
        if ra is not None:
            out['resize_algorithm'] = str(ra)
    except Exception as e:
        out['extraction_error'] = str(e)
    return out


def summarize_mask_config(mask_config):
    """Extract dict summary of NV_MaskProcessingConfig override values."""
    if mask_config is None:
        return None
    if not isinstance(mask_config, dict):
        return {"raw": str(type(mask_config))}
    keys = [
        'cleanup_fill_holes', 'cleanup_remove_noise', 'cleanup_smooth',
        'crop_expand_px', 'crop_blend_feather_px',
        'vace_input_grow_px', 'vace_erosion_blocks', 'vace_feather_blocks',
        'vace_halo_px', 'vace_stitch_erosion_px', 'vace_stitch_feather_px',
    ]
    out = {}
    for k in keys:
        if k in mask_config:
            v = mask_config[k]
            try:
                out[k] = float(v) if isinstance(v, (int, float)) else str(v)
            except (ValueError, TypeError):
                out[k] = str(v)
    return out


def resolve_output_path(output_path, shot_name):
    """Resolve output_path: directory → timestamped filename. File → as-is.
    Auto-creates parent dir.
    """
    if not output_path:
        return ""
    output_path = os.path.expandvars(os.path.expanduser(output_path))
    if os.path.isdir(output_path):
        ts = time.strftime("%Y%m%d-%H%M%S")
        safe_shot = shot_name.strip().replace(os.sep, "_") if shot_name else "render"
        fname = f"{safe_shot}_{ts}_manifest.json"
        return os.path.join(output_path, fname)
    parent = os.path.dirname(output_path)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError:
            pass
    return output_path
