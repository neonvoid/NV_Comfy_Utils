"""
Sigma Schedule Visualizer - Diagnostic node for understanding shift/denoise/steps interaction.

Renders a visual chart showing exactly how ComfyUI computes the sigma schedule
for flow matching models (Wan, Flux, etc.), where denoise truncation occurs,
and how much of the input signal is preserved.

The math replicated here matches ComfyUI's actual implementation:
  - comfy/model_sampling.py: time_snr_shift() Möbius transformation
  - comfy/samplers.py: set_steps() denoise truncation logic
"""

import torch
import numpy as np
import io as io_module
from PIL import Image

# Try matplotlib for proper charts, fall back to PIL text rendering
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def time_snr_shift(alpha, t):
    """
    Möbius transformation for flow matching sigma schedule.
    Matches comfy/model_sampling.py line 244-247.

    sigma(t) = alpha * t / (1 + (alpha - 1) * t)

    When alpha=1: identity (no shift)
    When alpha>1: front-loads the schedule toward high sigma values
    """
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def compute_schedule(shift, steps, denoise):
    """
    Compute the exact sigma schedule matching ComfyUI's implementation.
    Replicates comfy/samplers.py set_steps() logic.

    Returns: (full_sigmas, used_sigmas, discarded_sigmas)
      - full_sigmas: all sigma values for the expanded schedule
      - used_sigmas: the tail of the schedule actually used for denoising
      - discarded_sigmas: the high-noise head that gets truncated
    """
    if denoise <= 0.0:
        return [], [], []

    if denoise >= 0.9999:
        total_steps = steps
    else:
        total_steps = int(steps / denoise)

    # Generate the full schedule using percent_to_sigma logic.
    # percent goes from 0 (= sigma_max) to 1 (= sigma=0)
    # sigma = time_snr_shift(shift, 1.0 - percent)
    full_sigmas = []
    for k in range(total_steps + 1):
        percent = k / total_steps
        if percent <= 0.0:
            sigma = 1.0
        elif percent >= 1.0:
            sigma = 0.0
        else:
            sigma = time_snr_shift(shift, 1.0 - percent)
        full_sigmas.append(sigma)

    # Truncate: keep last (steps + 1) values
    # This is the sigmas[-(steps + 1):] from set_steps()
    used_sigmas = full_sigmas[-(steps + 1):]
    discarded_sigmas = full_sigmas[:len(full_sigmas) - (steps + 1)]

    return full_sigmas, used_sigmas, discarded_sigmas


class NV_SigmaScheduleVisualizer:
    """
    Visualize the sigma schedule to understand shift/denoise/steps interaction.

    Renders charts showing:
    - The continuous sigma curve with step markers
    - Which portion of the schedule is used vs discarded by denoise
    - Step sizes (work done per step)
    - Signal preservation from the input image

    Optionally compare two different parameter sets side-by-side.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shift": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 100, "step": 1}),
                "denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "compare_shift": ("FLOAT", {"default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1}),
                "compare_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "compare_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
                "show_comparison": ("BOOLEAN", {"default": False}),
                "chart_width": ("INT", {"default": 1200, "min": 600, "max": 2400, "step": 100}),
                "chart_height": ("INT", {"default": 900, "min": 400, "max": 1800, "step": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("chart", "summary",)
    FUNCTION = "visualize"
    CATEGORY = "NV_Utils/Debug"

    def visualize(self, shift, steps, denoise,
                  compare_shift=7.0, compare_steps=20, compare_denoise=0.35,
                  show_comparison=False, chart_width=1200, chart_height=900):

        # Compute primary schedule
        full_sigmas, used_sigmas, discarded_sigmas = compute_schedule(shift, steps, denoise)

        # Compute comparison schedule if enabled
        comp_full, comp_used, comp_discarded = None, None, None
        if show_comparison:
            comp_full, comp_used, comp_discarded = compute_schedule(
                compare_shift, compare_steps, compare_denoise
            )

        # Build summary text
        summary = self._build_summary(shift, steps, denoise, used_sigmas, full_sigmas)
        if show_comparison and comp_used is not None:
            summary += "\n\n" + "=" * 50 + "\n"
            summary += "COMPARISON\n"
            summary += "=" * 50 + "\n"
            summary += self._build_summary(compare_shift, compare_steps, compare_denoise,
                                           comp_used, comp_full)

        # Render chart
        if HAS_MATPLOTLIB:
            chart_image = self._render_matplotlib(
                shift, steps, denoise, full_sigmas, used_sigmas, discarded_sigmas,
                show_comparison, compare_shift, compare_steps, compare_denoise,
                comp_full, comp_used, comp_discarded,
                chart_width, chart_height
            )
        else:
            chart_image = self._render_pil_fallback(summary, chart_width, chart_height)

        return (chart_image, summary)

    def _build_summary(self, shift, steps, denoise, used_sigmas, full_sigmas):
        """Build a text summary of the schedule."""
        if not used_sigmas:
            return f"Shift={shift}, Steps={steps}, Denoise={denoise}\nNo denoising (denoise=0)"

        sigma_start = used_sigmas[0]
        signal_preserved = (1.0 - sigma_start) * 100

        total_steps_computed = len(full_sigmas) - 1
        actual_steps = len(used_sigmas) - 1

        lines = [
            f"Shift={shift}, Steps={steps}, Denoise={denoise}",
            f"",
            f"Schedule computation:",
            f"  Expanded steps: int({steps}/{denoise:.2f}) = {total_steps_computed}",
            f"  Steps used: {actual_steps} (last {actual_steps} of {total_steps_computed})",
            f"",
            f"Sigma range: {sigma_start:.4f} -> {used_sigmas[-1]:.4f}",
            f"Signal preserved from input: {signal_preserved:.1f}%",
            f"",
            f"Sigma values at each step:",
        ]

        for i, s in enumerate(used_sigmas):
            label = f"  s[{i}] = {s:.4f}"
            if i == 0:
                label += f"  <- START ({signal_preserved:.1f}% signal)"
            elif i == len(used_sigmas) - 1:
                label += f"  <- END (clean)"
            lines.append(label)

        lines.append("")
        lines.append("Step sizes (Δσ per step):")
        for i in range(len(used_sigmas) - 1):
            delta = used_sigmas[i] - used_sigmas[i + 1]
            pct = delta / sigma_start * 100 if sigma_start > 0 else 0
            lines.append(f"  Step {i + 1}: delta={delta:.4f} ({pct:.1f}% of range)")

        return "\n".join(lines)

    # ── matplotlib rendering ──────────────────────────────────────────

    def _render_matplotlib(self, shift, steps, denoise, full_sigmas, used_sigmas,
                           discarded_sigmas, show_comparison, comp_shift, comp_steps,
                           comp_denoise, comp_full, comp_used, comp_discarded,
                           chart_width, chart_height):
        """Render charts using matplotlib."""
        dpi = 100
        fig_w = chart_width / dpi
        fig_h = chart_height / dpi

        bg_color = '#1a1a2e'
        panel_bg = '#16213e'

        if show_comparison and comp_used:
            fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_facecolor(bg_color)
            fig.suptitle('Sigma Schedule Visualizer', fontsize=16,
                         fontweight='bold', color='white', y=0.98)

            # Primary schedule
            self._draw_schedule_panel(axes[0, 0], shift, steps, denoise,
                                      full_sigmas, used_sigmas, '#2196F3', '#FF5722',
                                      panel_bg)
            axes[0, 0].set_title(
                f'Primary: shift={shift}, steps={steps}, denoise={denoise}',
                fontsize=11, color='white', pad=8)

            # Comparison schedule
            self._draw_schedule_panel(axes[0, 1], comp_shift, comp_steps, comp_denoise,
                                      comp_full, comp_used, '#4CAF50', '#FF9800',
                                      panel_bg)
            axes[0, 1].set_title(
                f'Compare: shift={comp_shift}, steps={comp_steps}, denoise={comp_denoise}',
                fontsize=11, color='white', pad=8)

            # Step sizes comparison
            self._draw_step_sizes_comparison(axes[1, 0], used_sigmas, comp_used,
                                             f'Primary', f'Compare', panel_bg)

            # Signal preservation comparison
            self._draw_signal_comparison(axes[1, 1], used_sigmas, comp_used,
                                         f'shift={shift}\nsteps={steps}\ndenoise={denoise}',
                                         f'shift={comp_shift}\nsteps={comp_steps}\ndenoise={comp_denoise}',
                                         panel_bg)
        else:
            fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            fig.patch.set_facecolor(bg_color)
            fig.suptitle('Sigma Schedule Visualizer', fontsize=16,
                         fontweight='bold', color='white', y=0.98)

            # Layout: large schedule chart on top, step sizes and info below
            gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3,
                                  left=0.08, right=0.95, top=0.92, bottom=0.08)

            ax_main = fig.add_subplot(gs[0, :])
            ax_steps = fig.add_subplot(gs[1, 0])
            ax_info = fig.add_subplot(gs[1, 1])

            # Main schedule chart
            self._draw_schedule_panel(ax_main, shift, steps, denoise,
                                      full_sigmas, used_sigmas, '#2196F3', '#FF5722',
                                      panel_bg)
            ax_main.set_title(
                f'shift={shift}  |  steps={steps}  |  denoise={denoise}',
                fontsize=13, color='white', pad=10)

            # Step sizes
            self._draw_step_sizes_single(ax_steps, used_sigmas, panel_bg)

            # Info panel
            self._draw_info_panel(ax_info, shift, steps, denoise,
                                  used_sigmas, full_sigmas, panel_bg)

        # Convert figure to image tensor
        buf = io_module.BytesIO()
        fig.savefig(buf, format='png', facecolor=bg_color, edgecolor='none')
        plt.close(fig)
        buf.seek(0)

        pil_img = Image.open(buf).convert('RGB')
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)

    def _draw_schedule_panel(self, ax, shift, steps, denoise,
                              full_sigmas, used_sigmas, used_color, disc_color,
                              panel_bg):
        """Draw the main sigma schedule visualization."""
        ax.set_facecolor(panel_bg)

        # ── Continuous sigma curves for reference ──
        t_vals = np.linspace(0.001, 1.0, 500)

        # Shift=1 reference (identity line)
        ax.plot(t_vals, t_vals, color='#444466', linewidth=1,
                linestyle=':', alpha=0.5, label='shift=1 (no shift)')

        # Current shift curve
        sigma_curve = [time_snr_shift(shift, t) for t in t_vals]
        ax.plot(t_vals, sigma_curve, color='#7777aa', linewidth=2,
                linestyle='--', alpha=0.6, label=f'shift={shift} curve')

        # ── Plot step markers on the schedule ──
        n_full = len(full_sigmas)
        n_used = len(used_sigmas)
        n_discarded = n_full - n_used

        # Map each sigma to its t value: sigma = shift*t/(1+(shift-1)*t)
        # Solving for t: t = sigma / (shift - (shift-1)*sigma)
        def sigma_to_t(sigma, s):
            if sigma >= 1.0:
                return 1.0
            if sigma <= 0.0:
                return 0.0
            return sigma / (s - (s - 1) * sigma)

        # Discarded steps (high noise, skipped by denoise)
        if n_discarded > 0:
            disc_t = [sigma_to_t(s, shift) for s in full_sigmas[:n_discarded]]
            disc_s = full_sigmas[:n_discarded]
            ax.scatter(disc_t, disc_s, color=disc_color, s=70, zorder=5,
                       alpha=0.5, marker='x', linewidths=2,
                       label=f'Skipped ({n_discarded} points)')

        # Used steps
        used_t = [sigma_to_t(s, shift) for s in used_sigmas]
        ax.scatter(used_t, used_sigmas, color=used_color, s=90, zorder=5,
                   edgecolors='white', linewidths=1.5,
                   label=f'Used ({n_used - 1} steps)')

        # Connect used steps with path
        ax.plot(used_t, used_sigmas, color=used_color, linewidth=2.5, alpha=0.8)

        # ── Denoise threshold line ──
        if used_sigmas:
            sigma_start = used_sigmas[0]
            signal_pct = (1.0 - sigma_start) * 100

            ax.axhline(y=sigma_start, color='#FFD700', linewidth=2,
                        linestyle='--', alpha=0.8)
            ax.annotate(
                f'  denoise={denoise}  →  σ_start = {sigma_start:.3f}\n'
                f'  signal preserved = {signal_pct:.1f}%',
                xy=(0.02, sigma_start), fontsize=10, color='#FFD700',
                verticalalignment='bottom', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e',
                          edgecolor='#FFD700', alpha=0.8))

            # Shade the "signal preserved" band at the bottom
            ax.axhspan(0, 1.0 - sigma_start, alpha=0.06, color='#4CAF50')
            ax.text(0.5, (1.0 - sigma_start) / 2, f'Signal: {signal_pct:.0f}%',
                    ha='center', va='center', fontsize=9, color='#4CAF50',
                    alpha=0.6, transform=ax.get_yaxis_transform())

        # ── Annotate sigma values ──
        for i, (t, s) in enumerate(zip(used_t, used_sigmas)):
            offset = 0.04 if i % 2 == 0 else -0.06
            ax.annotate(f'{s:.3f}', xy=(t, s), xytext=(t, s + offset),
                        fontsize=8, color='white', ha='center', alpha=0.85,
                        arrowprops=dict(arrowstyle='-', color='white', alpha=0.3,
                                        lw=0.5) if abs(offset) > 0.03 else None)

        ax.set_xlabel('Normalized time (t)', color='white', fontsize=11)
        ax.set_ylabel('Sigma (noise level)', color='white', fontsize=11)
        ax.set_xlim(-0.03, 1.05)
        ax.set_ylim(-0.05, 1.1)
        ax.tick_params(colors='white', labelsize=9)
        ax.legend(fontsize=9, loc='lower right', facecolor='#2a2a4e',
                  edgecolor='#555577', labelcolor='white')
        ax.grid(True, alpha=0.12, color='white')

    def _draw_step_sizes_single(self, ax, used_sigmas, panel_bg):
        """Draw step size bar chart for a single schedule."""
        ax.set_facecolor(panel_bg)

        if len(used_sigmas) < 2:
            ax.text(0.5, 0.5, 'No steps', ha='center', va='center',
                    color='white', fontsize=14)
            ax.axis('off')
            return

        deltas = [used_sigmas[i] - used_sigmas[i + 1]
                  for i in range(len(used_sigmas) - 1)]
        step_labels = [f'Step {i + 1}' for i in range(len(deltas))]

        # Color large steps red (>25% of total range) to flag potential issues
        max_delta = max(deltas) if deltas else 1
        colors = ['#FF5722' if d > 0.25 else '#2196F3' for d in deltas]

        bars = ax.barh(step_labels, deltas, color=colors,
                       edgecolor='white', linewidth=0.5)

        for bar, delta in zip(bars, deltas):
            ax.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                    f'Δσ={delta:.3f}', va='center', fontsize=9, color='white')

        ax.set_xlabel('Step Size (Δσ)', color='white', fontsize=10)
        ax.set_title('Work Per Step', color='white', fontsize=12, pad=8)
        ax.tick_params(colors='white', labelsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.12, color='white', axis='x')

    def _draw_step_sizes_comparison(self, ax, used_1, used_2, label_1, label_2,
                                     panel_bg):
        """Draw step size comparison between two schedules."""
        ax.set_facecolor(panel_bg)

        deltas_1 = ([used_1[i] - used_1[i + 1] for i in range(len(used_1) - 1)]
                    if len(used_1) > 1 else [])
        deltas_2 = ([used_2[i] - used_2[i + 1] for i in range(len(used_2) - 1)]
                    if used_2 and len(used_2) > 1 else [])

        max_steps = max(len(deltas_1), len(deltas_2))
        if max_steps == 0:
            ax.text(0.5, 0.5, 'No steps', ha='center', va='center',
                    color='white', fontsize=14)
            ax.axis('off')
            return

        x = np.arange(max_steps)
        width = 0.35

        d1 = deltas_1 + [0] * (max_steps - len(deltas_1))
        d2 = deltas_2 + [0] * (max_steps - len(deltas_2))

        ax.bar(x - width / 2, d1, width, label=label_1, color='#2196F3',
               edgecolor='white', linewidth=0.5)
        ax.bar(x + width / 2, d2, width, label=label_2, color='#4CAF50',
               edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Step Number', color='white', fontsize=10)
        ax.set_ylabel('Step Size (Δσ)', color='white', fontsize=10)
        ax.set_title('Work Per Step', color='white', fontsize=12, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i + 1}' for i in range(max_steps)], fontsize=8)
        ax.tick_params(colors='white', labelsize=9)
        ax.legend(fontsize=9, facecolor='#2a2a4e', edgecolor='#555577',
                  labelcolor='white')
        ax.grid(True, alpha=0.12, color='white', axis='y')

    def _draw_signal_comparison(self, ax, used_1, used_2, label_1, label_2,
                                 panel_bg):
        """Draw signal preservation comparison."""
        ax.set_facecolor(panel_bg)

        sig1 = (1.0 - used_1[0]) * 100 if used_1 else 0
        sig2 = (1.0 - used_2[0]) * 100 if used_2 else 0

        bars = ax.bar(['Primary', 'Compare'], [sig1, sig2],
                      color=['#2196F3', '#4CAF50'], edgecolor='white', linewidth=1)

        for bar, val, label in zip(bars, [sig1, sig2], [label_1, label_2]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', fontsize=14, color='white',
                    fontweight='bold')
            ax.text(bar.get_x() + bar.get_width() / 2, -8,
                    label, ha='center', fontsize=8, color='#aaaacc',
                    linespacing=1.3)

        ax.set_ylabel('Signal Preserved (%)', color='white', fontsize=10)
        ax.set_title('Input Signal Preservation', color='white', fontsize=12, pad=8)
        ax.set_ylim(0, 110)
        ax.tick_params(colors='white', labelsize=9)
        ax.grid(True, alpha=0.12, color='white', axis='y')

    def _draw_info_panel(self, ax, shift, steps, denoise, used_sigmas,
                          full_sigmas, panel_bg):
        """Draw info text panel with schedule details."""
        ax.set_facecolor(panel_bg)
        ax.axis('off')

        if not used_sigmas:
            ax.text(0.5, 0.5, 'No steps', ha='center', va='center',
                    color='white', fontsize=14)
            return

        sigma_start = used_sigmas[0]
        signal_pct = (1.0 - sigma_start) * 100
        total_computed = len(full_sigmas) - 1
        actual_used = len(used_sigmas) - 1

        # Build text blocks
        lines = []
        lines.append(('PARAMETERS', '', 13, '#FFD700'))
        lines.append(('Shift', f'{shift}', 11, 'white'))
        lines.append(('Steps', f'{steps}', 11, 'white'))
        lines.append(('Denoise', f'{denoise}', 11, 'white'))
        lines.append(('', '', 6, 'white'))
        lines.append(('COMPUTATION', '', 13, '#FFD700'))
        lines.append(('Expanded to', f'{total_computed} steps', 10, '#aaaacc'))
        lines.append(('Then take last', f'{actual_used} steps', 10, '#aaaacc'))
        lines.append(('', '', 6, 'white'))
        lines.append(('RESULT', '', 13, '#FFD700'))
        lines.append(('σ start', f'{sigma_start:.4f}', 11, 'white'))

        sig_color = '#4CAF50' if signal_pct > 30 else '#FF9800' if signal_pct > 15 else '#FF5722'
        lines.append(('Signal kept', f'{signal_pct:.1f}%', 14, sig_color))
        lines.append(('', '', 6, 'white'))
        lines.append(('STEP DETAILS', '', 13, '#FFD700'))

        for i in range(len(used_sigmas) - 1):
            s_from = used_sigmas[i]
            s_to = used_sigmas[i + 1]
            delta = s_from - s_to
            lines.append((f'Step {i + 1}',
                           f'{s_from:.3f} → {s_to:.3f} (Δ{delta:.3f})',
                           9, '#aaaacc'))

        y = 0.97
        for label, value, fontsize, color in lines:
            if not label and not value:
                y -= 0.02
                continue
            if value:
                ax.text(0.05, y, f'{label}: ', fontsize=fontsize, color='#888899',
                        verticalalignment='top', transform=ax.transAxes)
                ax.text(0.48, y, value, fontsize=fontsize, color=color,
                        verticalalignment='top', transform=ax.transAxes,
                        fontweight='bold')
            else:
                ax.text(0.05, y, label, fontsize=fontsize, color=color,
                        verticalalignment='top', transform=ax.transAxes,
                        fontweight='bold')
            y -= 0.065 if fontsize >= 12 else 0.05

    # ── PIL fallback ──────────────────────────────────────────────────

    def _render_pil_fallback(self, summary, width, height):
        """Simple PIL text rendering when matplotlib is not available."""
        from PIL import ImageDraw, ImageFont

        img = Image.new('RGB', (width, height), color=(26, 26, 46))
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 14)
            title_font = ImageFont.truetype("arial.ttf", 20)
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

        draw.text((20, 10), "Sigma Schedule Visualizer", font=title_font,
                  fill=(255, 215, 0))
        draw.text((20, 40), "(Install matplotlib for graphical charts)",
                  font=font, fill=(170, 170, 200))

        y = 80
        for line in summary.split('\n'):
            draw.text((20, y), line, font=font, fill=(255, 255, 255))
            y += 18

        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "NV_SigmaScheduleVisualizer": NV_SigmaScheduleVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NV_SigmaScheduleVisualizer": "NV Sigma Schedule Visualizer",
}
