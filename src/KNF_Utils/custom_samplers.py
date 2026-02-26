"""
Custom samplers for rectified flow / flow-matching models (e.g. WAN 2.2).

RF-Solver-2: 2nd-order Taylor solver with finite-difference velocity derivative.
    Paper: arxiv.org/abs/2411.04746 (ICML 2025)
    2 NFE/step, O(h³) error per step.

Flow-Solver-3: Multi-step polynomial interpolation of cached velocity evaluations.
    Paper: arxiv.org/abs/2411.07627
    1 NFE/step, O(h³) error (3rd order, graceful degradation on early steps).

Both are registered into ComfyUI's KSampler dropdown via __init__.py.
"""

import torch
from comfy.k_diffusion.sampling import to_d, default_noise_sampler
from comfy.utils import model_trange as trange


# ---------------------------------------------------------------------------
# RF-Solver-2
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_nv_rf_solver_2(model, x, sigmas, extra_args=None, callback=None,
                          disable=None, delta_t=0.01):
    """RF-Solver-2: 2nd-order Taylor solver for rectified flow ODE.

    At each step:
        1. Evaluate velocity  v = (x - D(x, σ)) / σ           (1st NFE)
        2. Probe: tiny Euler step to get v at a nearby point    (2nd NFE)
        3. Finite-difference estimate of dv/dt
        4. 2nd-order Taylor step:  x += h·v + ½h²·v'

    Falls back to Euler on the final step (σ_next = 0).

    Args:
        delta_t: Probe step size for finite-difference derivative estimation.
                 Paper default is 0.01. Smaller = more accurate but numerically
                 noisier. Larger = smoother but less precise.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma  # negative (moving toward clean)

        # 1. Velocity at current point (1st NFE)
        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)

        if callback is not None:
            callback({
                'x': x, 'i': i, 'sigma': sigma,
                'sigma_hat': sigma, 'denoised': denoised,
            })

        if sigma_next == 0:
            # Final step: standard Euler (no valid probe target at σ=0)
            x = x + d * h
        else:
            # 2. Probe: tiny Euler step to estimate velocity derivative (2nd NFE)
            #    Move in the direction of the velocity field by delta_t
            x_probe = x + delta_t * d
            sigma_probe = sigma + delta_t
            # Clamp: don't let probe sigma go below next sigma
            sigma_probe = x.new_tensor(max(float(sigma_probe), float(sigma_next)))
            denoised_probe = model(
                x_probe, sigma_probe.expand(x.shape[0]), **extra_args
            )
            d_probe = to_d(x_probe, sigma_probe, denoised_probe)

            # 3. Finite-difference velocity derivative
            d_prime = (d_probe - d) / delta_t

            # 4. 2nd-order Taylor step: x += h·v + ½h²·v'
            x = x + h * d + 0.5 * h * h * d_prime

    return x


# ---------------------------------------------------------------------------
# Flow-Solver-3  (3rd-order multi-step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_nv_flow_solver_3(model, x, sigmas, extra_args=None, callback=None,
                            disable=None):
    """Flow-Solver: multi-step polynomial interpolation for flow diffusion ODE.

    Caches the last 2 velocity evaluations and uses polynomial interpolation
    to approximate higher-order Taylor terms — achieving 3rd-order accuracy
    with only 1 NFE per step.

    Graceful degradation:
        Step 0:  1st-order (Euler)      — no history
        Step 1:  2nd-order              — 1 cached velocity
        Step 2+: 3rd-order              — 2 cached velocities

    Falls back to Euler on the final step (σ_next = 0) to avoid cache
    extrapolation artifacts near the origin.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    # Velocity cache: list of (sigma_value, velocity_tensor) tuples
    cache = []

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        h = sigma_next - sigma  # negative

        # Model call (1 NFE)
        denoised = model(x, sigma * s_in, **extra_args)
        d = to_d(x, sigma, denoised)

        if callback is not None:
            callback({
                'x': x, 'i': i, 'sigma': sigma,
                'sigma_hat': sigma, 'denoised': denoised,
            })

        sigma_f = float(sigma)

        if len(cache) == 0 or sigma_next == 0:
            # 1st-order (Euler)
            x = x + d * h

        elif len(cache) == 1:
            # 2nd-order — one previous velocity available
            sigma_prev, d_prev = cache[-1]
            D1 = d - d_prev

            # Taylor coefficient for h²/2 term
            C1 = h * h / 2.0
            denom = sigma_f - sigma_prev
            if abs(denom) > 1e-8:
                B1 = C1 / denom
            else:
                B1 = 0.0

            x = x + h * d + B1 * D1

        else:
            # 3rd-order — two previous velocities available
            sigma_p1, d_p1 = cache[-1]
            sigma_p2, d_p2 = cache[-2]
            D1 = d - d_p1
            D2 = d - d_p2

            C1 = h * h / 2.0
            C2 = h * h * h / 6.0

            # Solve 2×2 linear system:  R · [B1, B2]^T = [C1, C2]^T
            #   R = [[σ - σ_{-1},  σ - σ_{-2}],
            #        [(σ - σ_{-1})², (σ - σ_{-2})²]]
            r11 = sigma_f - sigma_p1
            r12 = sigma_f - sigma_p2
            r21 = r11 * r11
            r22 = r12 * r12
            det = r11 * r22 - r12 * r21

            if abs(det) > 1e-10:
                B1 = (r22 * C1 - r12 * C2) / det
                B2 = (-r21 * C1 + r11 * C2) / det
            else:
                # Degenerate matrix — fall back to 2nd order
                B1 = C1 / r11 if abs(r11) > 1e-8 else 0.0
                B2 = 0.0

            x = x + h * d + B1 * D1 + B2 * D2

        # Update cache (keep last 2 for 3rd-order interpolation)
        cache.append((sigma_f, d))
        if len(cache) > 2:
            cache.pop(0)

    return x
