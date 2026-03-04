"""
Kalman + Rauch-Tung-Striebel (RTS) Smoother for Bounding Box Stabilization.

8D state model: [cx, cy, w, h, vcx, vcy, vw, vh]
- Position (cx, cy) and dimensions (w, h) with velocity terms
- Constant-velocity motion model (BoT-SORT style)
- Forward Kalman filter + RTS backward smoother = globally optimal offline estimate
- 53-63% accuracy improvement over forward-only Kalman (Pale-Ramon 2022)

Key advantages over One-Euro filter:
    - Models velocity → predicts through brief occlusions
    - RTS backward pass uses future frames → no causal lag
    - Natively handles missing frames (just runs predict step)
    - Smooths position AND dimensions jointly

Default Q/R tuned for SAM3 masks at 8fps video:
    - q_pos=4.0: position acceleration noise (px/frame²)²
    - q_dim=1.0: dimension acceleration noise (size changes slowly)
    - r_pos=9.0: center measurement noise (3px std from SAM3 mask edges)
    - r_dim=25.0: width/height measurement noise (5px std)

Reference: Pale-Ramon 2022 — "Kalman vs FIR for bbox stabilization"
"""

import numpy as np


def kalman_rts_smooth(x1s, y1s, x2s, y2s, present,
                      q_pos=4.0, q_dim=1.0, r_pos=9.0, r_dim=25.0):
    """Apply Kalman + RTS smoothing to bbox coordinate sequences.

    Operates in center+size space: [cx, cy, w, h, vcx, vcy, vw, vh].

    Args:
        x1s, y1s, x2s, y2s: Per-frame corner coordinates (float lists).
        present: Per-frame boolean — True if mask was detected, False if filled.
        q_pos: Process noise for position acceleration (higher = trust measurements more).
        q_dim: Process noise for dimension acceleration (lower = more stable sizes).
        r_pos: Measurement noise for center position (higher = smoother trajectory).
        r_dim: Measurement noise for width/height (higher = more stable dimensions).

    Returns:
        Smoothed (x1s, y1s, x2s, y2s) as lists.
    """
    T = len(x1s)
    if T <= 2:
        return x1s, y1s, x2s, y2s

    # Convert corners to center+size
    cxs = [(x1 + x2) / 2.0 for x1, x2 in zip(x1s, x2s)]
    cys = [(y1 + y2) / 2.0 for y1, y2 in zip(y1s, y2s)]
    ws = [x2 - x1 for x1, x2 in zip(x1s, x2s)]
    hs = [y2 - y1 for y1, y2 in zip(y1s, y2s)]

    # State: [cx, cy, w, h, vcx, vcy, vw, vh]
    dim = 8
    meas_dim = 4

    # State transition matrix (constant velocity)
    F = np.eye(dim)
    F[0, 4] = 1.0  # cx += vcx
    F[1, 5] = 1.0  # cy += vcy
    F[2, 6] = 1.0  # w += vw
    F[3, 7] = 1.0  # h += vh

    # Measurement matrix (observe cx, cy, w, h)
    H = np.zeros((meas_dim, dim))
    H[0, 0] = 1.0  # cx
    H[1, 1] = 1.0  # cy
    H[2, 2] = 1.0  # w
    H[3, 3] = 1.0  # h

    # Process noise (acceleration-based, discrete white noise)
    Q = np.zeros((dim, dim))
    # Position block: [pos, vel] with noise q
    for i, q in [(0, q_pos), (1, q_pos), (2, q_dim), (3, q_dim)]:
        Q[i, i] += q * 0.25      # dt^4/4
        Q[i, i+4] += q * 0.5     # dt^3/2
        Q[i+4, i] += q * 0.5     # dt^3/2
        Q[i+4, i+4] += q         # dt^2

    # Measurement noise
    R = np.diag([r_pos, r_pos, r_dim, r_dim])

    # Initialize state from first present frame
    first_valid = 0
    for i in range(T):
        if present[i]:
            first_valid = i
            break

    x0 = np.array([cxs[first_valid], cys[first_valid],
                    ws[first_valid], hs[first_valid],
                    0.0, 0.0, 0.0, 0.0])
    P0 = np.diag([r_pos, r_pos, r_dim, r_dim,
                   q_pos * 10, q_pos * 10, q_dim * 10, q_dim * 10])

    # Forward Kalman filter
    x_pred = np.zeros((T, dim))
    P_pred = np.zeros((T, dim, dim))
    x_filt = np.zeros((T, dim))
    P_filt = np.zeros((T, dim, dim))

    x_curr = x0.copy()
    P_curr = P0.copy()

    for t in range(T):
        # Predict
        x_pred[t] = F @ x_curr
        P_pred[t] = F @ P_curr @ F.T + Q

        if present[t]:
            # Measurement
            z = np.array([cxs[t], cys[t], ws[t], hs[t]])

            # Innovation
            y = z - H @ x_pred[t]
            S = H @ P_pred[t] @ H.T + R

            # Kalman gain (Joseph form for numerical stability)
            K = P_pred[t] @ H.T @ np.linalg.inv(S)

            # Update
            x_filt[t] = x_pred[t] + K @ y
            IKH = np.eye(dim) - K @ H
            P_filt[t] = IKH @ P_pred[t] @ IKH.T + K @ R @ K.T
        else:
            # No measurement — prediction only
            x_filt[t] = x_pred[t]
            P_filt[t] = P_pred[t]

        x_curr = x_filt[t]
        P_curr = P_filt[t]

    # RTS backward smoother
    x_smooth = np.zeros((T, dim))
    P_smooth = np.zeros((T, dim, dim))

    x_smooth[T-1] = x_filt[T-1]
    P_smooth[T-1] = P_filt[T-1]

    for t in range(T - 2, -1, -1):
        # Smoother gain
        P_pred_inv = np.linalg.inv(P_pred[t+1])
        G = P_filt[t] @ F.T @ P_pred_inv

        # Smooth
        x_smooth[t] = x_filt[t] + G @ (x_smooth[t+1] - x_pred[t+1])
        P_smooth[t] = P_filt[t] + G @ (P_smooth[t+1] - P_pred[t+1]) @ G.T

    # Convert back to corners
    out_x1s = []
    out_y1s = []
    out_x2s = []
    out_y2s = []

    for t in range(T):
        cx, cy, w, h = x_smooth[t, :4]
        # Ensure positive dimensions
        w = max(w, 1.0)
        h = max(h, 1.0)
        out_x1s.append(cx - w / 2.0)
        out_y1s.append(cy - h / 2.0)
        out_x2s.append(cx + w / 2.0)
        out_y2s.append(cy + h / 2.0)

    return out_x1s, out_y1s, out_x2s, out_y2s
