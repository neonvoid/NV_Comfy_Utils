"""
NV Seam Analyzer — Measures temporal continuity across chunk boundaries.

Takes the tail of chunk N and the head of chunk N+1, computes frame-to-frame
metrics across the seam window to quantify detail drift, snap-back, and
temporal smoothness.

Metrics computed:
- PSNR: Peak Signal-to-Noise Ratio (higher = more similar)
- SSIM-approx: Structural similarity via mean/std correlation (higher = more similar)
- Optical flow magnitude: Inter-frame motion (spikes = snap-back artifact)
- Laplacian variance: Sharpness per frame (drops = softening at seam)
- Per-channel mean delta: Color drift detection

All metrics are computed per-frame-pair and printed as a table, plus
output as a visual chart IMAGE for easy comparison across variants.
"""

import torch
import numpy as np

# Math helpers — single source of truth in seam_ops.py. NV_SeamAnalyzer is a
# diagnostic visualization wrapper around the same primitives that
# NV_ShotMeasure/NV_ShotRecord use, so the chart numbers and the agent
# corpus numbers can never disagree.
from .seam_ops import (
    flow_magnitude as _optical_flow_magnitude,
    laplacian_variance as _laplacian_variance,
    psnr as _psnr,
    ssim_approx as _ssim_approx,
    to_gray_uint8 as _to_gray_uint8,
)


def _channel_means(frame_hwc):
    """Per-channel mean for [H, W, C] float tensor."""
    return [frame_hwc[..., c].mean().item() for c in range(frame_hwc.shape[-1])]


def _render_chart(metrics_table, label):
    """Render metrics as a simple chart IMAGE [1, H, W, 3].

    Creates a text-based chart with colored bars for each metric.
    """
    num_pairs = len(metrics_table)
    chart_w = 700
    row_h = 18
    header_h = 60
    section_h = num_pairs * row_h + 30
    num_sections = 4  # PSNR, SSIM, Flow, Sharpness
    chart_h = header_h + num_sections * section_h + 20

    chart = torch.zeros(chart_h, chart_w, 3, dtype=torch.float32)
    chart[:] = 0.15  # dark background

    # We'll render as a simple data visualization — colored horizontal bars
    # proportional to metric values, with the seam position highlighted

    # Find seam index (where chunk_n ends and chunk_n+1 begins)
    # Convention: metrics_table[i] compares frame i and frame i+1
    # Seam is wherever the label switches from "N" to "N+1"

    y = 5
    # Header text — encode as colored pixels (simple block letters too complex,
    # just use colored bars with position encoding)

    # PSNR section
    if num_pairs > 0:
        psnr_values = [m["psnr"] for m in metrics_table]
        psnr_min = min(psnr_values)
        psnr_max = max(max(psnr_values), psnr_min + 1)

        for i, m in enumerate(metrics_table):
            bar_y = header_h + i * row_h
            # Normalize PSNR to [0, 1] range for bar length
            norm = (m["psnr"] - psnr_min) / (psnr_max - psnr_min + 1e-8)
            bar_len = int(norm * (chart_w - 100))

            # Color: green for high PSNR, red for low
            r = 1.0 - norm
            g = norm
            chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 0] = r
            chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 1] = g

            # Seam marker (white tick if this is the seam pair)
            if m.get("is_seam", False):
                chart[bar_y:bar_y + row_h - 2, 75:80, :] = 1.0  # white marker

        # Flow section
        flow_values = [m["flow_mag"] for m in metrics_table if m["flow_mag"] >= 0]
        if flow_values:
            flow_max = max(flow_values) + 0.1
            section_start = header_h + section_h
            for i, m in enumerate(metrics_table):
                if m["flow_mag"] < 0:
                    continue
                bar_y = section_start + i * row_h
                if bar_y + row_h > chart_h:
                    break
                norm = m["flow_mag"] / flow_max
                bar_len = int(norm * (chart_w - 100))

                # Color: blue for low flow (smooth), yellow for high (motion/snap)
                chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 0] = norm
                chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 1] = norm
                chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 2] = 1.0 - norm

                if m.get("is_seam", False):
                    chart[bar_y:bar_y + row_h - 2, 75:80, :] = 1.0

        # Sharpness section
        sharp_values = [m["sharpness"] for m in metrics_table]
        if sharp_values:
            sharp_max = max(sharp_values) + 0.1
            section_start = header_h + 2 * section_h
            for i, m in enumerate(metrics_table):
                bar_y = section_start + i * row_h
                if bar_y + row_h > chart_h:
                    break
                norm = m["sharpness"] / sharp_max
                bar_len = int(norm * (chart_w - 100))

                chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 1] = 0.6
                chart[bar_y:bar_y + row_h - 2, 80:80 + bar_len, 2] = norm

                if m.get("is_seam", False):
                    chart[bar_y:bar_y + row_h - 2, 75:80, :] = 1.0

    return chart.unsqueeze(0)  # [1, H, W, 3]


class NV_SeamAnalyzer:
    """Measures temporal continuity across a chunk boundary.

    Wire the last N frames of chunk N and the first M frames of chunk N+1.
    Computes frame-to-frame metrics across the seam window and outputs
    a diagnostic chart + printed table.

    Use to compare variants (baseline, D sweep, A test) objectively.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_n_tail": ("IMAGE", {
                    "tooltip": "Last N frames of chunk N's output (the 'before' side of the seam)"
                }),
                "chunk_n1_head": ("IMAGE", {
                    "tooltip": "First M frames of chunk N+1's output (the 'after' side of the seam)"
                }),
            },
            "optional": {
                "label": ("STRING", {
                    "default": "",
                    "tooltip": "Label for this variant (e.g., 'baseline', 'D_1.5', 'A_test'). Shown in output."
                }),
                "face_crop": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, also compute metrics on the center 40% crop (rough face region proxy)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("chart", "report")
    OUTPUT_NODE = True
    FUNCTION = "analyze"
    CATEGORY = "NV_Utils/Debug"
    DESCRIPTION = (
        "Measures temporal continuity across a chunk boundary. "
        "Computes PSNR, SSIM, optical flow, and sharpness frame-by-frame "
        "across the seam window. Use to objectively compare chunk-chaining variants."
    )

    def analyze(self, chunk_n_tail, chunk_n1_head, label="", face_crop=False):
        # Concatenate into one sequence: [tail..., head...]
        all_frames = torch.cat([chunk_n_tail, chunk_n1_head], dim=0)
        n_tail = chunk_n_tail.shape[0]
        n_head = chunk_n1_head.shape[0]
        total = all_frames.shape[0]
        seam_idx = n_tail - 1  # last frame of chunk N

        H, W = all_frames.shape[1], all_frames.shape[2]

        # Optional center crop for face-region proxy
        if face_crop:
            crop_h = int(H * 0.4)
            crop_w = int(W * 0.4)
            y0 = (H - crop_h) // 2
            x0 = (W - crop_w) // 2
            all_frames_crop = all_frames[:, y0:y0 + crop_h, x0:x0 + crop_w, :]
        else:
            all_frames_crop = None

        # Compute per-pair metrics
        metrics = []
        prev_gray = _to_gray_uint8(all_frames[0])

        for i in range(1, total):
            curr = all_frames[i]
            prev = all_frames[i - 1]
            curr_gray = _to_gray_uint8(curr)

            m = {
                "pair": f"{i-1}->{i}",
                "is_seam": (i - 1 == seam_idx),
                "psnr": _psnr(prev, curr),
                "ssim": _ssim_approx(prev, curr),
                "flow_mag": _optical_flow_magnitude(prev_gray, curr_gray),
                "sharpness": _laplacian_variance(curr),
                "ch_delta_r": abs(_channel_means(curr)[0] - _channel_means(prev)[0]),
                "ch_delta_g": abs(_channel_means(curr)[1] - _channel_means(prev)[1]),
                "ch_delta_b": abs(_channel_means(curr)[2] - _channel_means(prev)[2]),
            }

            # Face crop metrics
            if all_frames_crop is not None:
                m["face_psnr"] = _psnr(all_frames_crop[i - 1], all_frames_crop[i])
                m["face_ssim"] = _ssim_approx(all_frames_crop[i - 1], all_frames_crop[i])

            metrics.append(m)
            prev_gray = curr_gray

        # Print table
        variant_label = f" [{label}]" if label else ""
        print(f"\n{'=' * 90}")
        print(f"[NV_SeamAnalyzer]{variant_label} Seam analysis: {n_tail} tail + {n_head} head frames")
        print(f"  Seam between frames {seam_idx} and {seam_idx + 1}")
        print(f"{'=' * 90}")

        header = f"{'Pair':>8} {'Seam':>4} {'PSNR':>7} {'SSIM':>6} {'Flow':>6} {'Sharp':>8} {'dR':>6} {'dG':>6} {'dB':>6}"
        if face_crop:
            header += f" {'fPSNR':>7} {'fSSIM':>6}"
        print(header)
        print("-" * len(header))

        for m in metrics:
            seam_marker = " >>>" if m["is_seam"] else ""
            line = (f"{m['pair']:>8}{seam_marker:>4} {m['psnr']:>7.2f} {m['ssim']:>6.4f} "
                    f"{m['flow_mag']:>6.2f} {m['sharpness']:>8.1f} "
                    f"{m['ch_delta_r']:>6.4f} {m['ch_delta_g']:>6.4f} {m['ch_delta_b']:>6.4f}")
            if face_crop and "face_psnr" in m:
                line += f" {m['face_psnr']:>7.2f} {m['face_ssim']:>6.4f}"
            print(line)

        # Summary stats
        seam_metrics = [m for m in metrics if m["is_seam"]]
        non_seam = [m for m in metrics if not m["is_seam"]]

        if seam_metrics and non_seam:
            seam_m = seam_metrics[0]
            avg_psnr = np.mean([m["psnr"] for m in non_seam])
            avg_ssim = np.mean([m["ssim"] for m in non_seam])
            avg_flow = np.mean([m["flow_mag"] for m in non_seam if m["flow_mag"] >= 0])

            print(f"\n--- Summary ---")
            print(f"  Seam PSNR:  {seam_m['psnr']:.2f}  (avg non-seam: {avg_psnr:.2f}, delta: {seam_m['psnr'] - avg_psnr:+.2f})")
            print(f"  Seam SSIM:  {seam_m['ssim']:.4f}  (avg non-seam: {avg_ssim:.4f}, delta: {seam_m['ssim'] - avg_ssim:+.4f})")
            print(f"  Seam Flow:  {seam_m['flow_mag']:.2f}  (avg non-seam: {avg_flow:.2f}, delta: {seam_m['flow_mag'] - avg_flow:+.2f})")

            if face_crop and "face_psnr" in seam_m:
                avg_fpsnr = np.mean([m["face_psnr"] for m in non_seam if "face_psnr" in m])
                print(f"  Face PSNR:  {seam_m['face_psnr']:.2f}  (avg non-seam: {avg_fpsnr:.2f}, delta: {seam_m['face_psnr'] - avg_fpsnr:+.2f})")

        print(f"{'=' * 90}\n")

        # Build report string
        report_lines = [f"Seam Analysis{variant_label}", f"Frames: {n_tail} tail + {n_head} head", ""]
        if seam_metrics and non_seam:
            seam_m = seam_metrics[0]
            report_lines.append(f"Seam PSNR: {seam_m['psnr']:.2f} (avg: {avg_psnr:.2f}, delta: {seam_m['psnr'] - avg_psnr:+.2f})")
            report_lines.append(f"Seam SSIM: {seam_m['ssim']:.4f} (avg: {avg_ssim:.4f})")
            report_lines.append(f"Seam Flow: {seam_m['flow_mag']:.2f} (avg: {avg_flow:.2f})")
        report = "\n".join(report_lines)

        # Render chart
        chart = _render_chart(metrics, label)

        return (chart, report)


NODE_CLASS_MAPPINGS = {
    "NV_SeamAnalyzer": NV_SeamAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SeamAnalyzer": "NV Seam Analyzer",
}
