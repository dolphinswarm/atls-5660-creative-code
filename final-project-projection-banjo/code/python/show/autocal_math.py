# autocal_math.py - pure numpy helpers for the automatic projector calibration.
#
# Frames:
#   depth frame  = Femto Bolt depth/IR camera, the frame the tracker (banjo_info) uses
#   color frame  = Femto Bolt color camera; the point cloud is expressed here while
#                  orbbec1.depthalignmode is 'software' (depth aligned to color)
#
# All image arrays are TouchDesigner numpyArray() layout: row 0 = BOTTOM of the image.

import numpy as np


def valid_mask(pc, zmin=0.2, zmax=8.0):
    z = pc[..., 2]
    return np.isfinite(z) & (z > zmin) & (z < zmax)


def fit_color_intrinsics(pc_aligned):
    """Color camera pinhole from an aligned (color-frame) point cloud.
    Returns (ax, bx, ay, by): col = ax*x/z + bx, row = ay*y/z + by, plus rms px."""
    H, W = pc_aligned.shape[:2]
    vv, uu = np.mgrid[0:H, 0:W]
    ok = valid_mask(pc_aligned)
    x, y, z = pc_aligned[ok].T
    u, v = uu[ok] + 0.5, vv[ok] + 0.5
    A = np.stack([x / z, np.ones_like(z)], 1)
    sx, *_ = np.linalg.lstsq(A, u, rcond=None)
    B = np.stack([y / z, np.ones_like(z)], 1)
    sy, *_ = np.linalg.lstsq(B, v, rcond=None)
    rms = float(np.sqrt(np.mean((A @ sx - u) ** 2 + (B @ sy - v) ** 2)))
    return (float(sx[0]), float(sx[1]), float(sy[0]), float(sy[1])), rms


def kabsch(P, Q):
    """R, t minimising |R P + t - Q|."""
    cp, cq = P.mean(0), Q.mean(0)
    Hm = (P - cp).T @ (Q - cq)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cq - R @ cp


def organized_normals(pc, step=2):
    """Per-pixel normals of an organized cloud (zeros where invalid)."""
    a = pc[2 * step:, step:-step] - pc[:-2 * step, step:-step]
    b = pc[step:-step, 2 * step:] - pc[step:-step, :-2 * step]
    n = np.cross(b, a)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(ln > 1e-9, n / np.maximum(ln, 1e-9), 0)
    out = np.zeros_like(pc)
    out[step:-step, step:-step] = n
    ok = valid_mask(pc)
    # kill normals across depth edges
    edge = np.zeros(pc.shape[:2], bool)
    edge[step:-step, step:-step] = (np.abs(pc[2 * step:, step:-step, 2] - pc[:-2 * step, step:-step, 2]) > 0.05) | \
                                   (np.abs(pc[step:-step, 2 * step:, 2] - pc[step:-step, :-2 * step, 2]) > 0.05)
    out[~ok | edge] = 0
    return out


def _rot(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def depth_to_color_icp(pc_depth, pc_color, intr, R0=None, t0=None, iters=40, stride=2):
    """Rigid transform depth-frame -> color-frame (X_c = R X_d + t) by projective
    point-to-plane ICP. pc_depth: unaligned cloud; pc_color: aligned cloud.
    Returns R, t, rms (m, point-to-plane), inlier fraction within 1 cm."""
    ax, bx, ay, by = intr
    H, W = pc_color.shape[:2]
    N = organized_normals(pc_color)
    P = pc_depth[::stride, ::stride].reshape(-1, 3)
    P = P[valid_mask(P[None])[0]]
    R = np.eye(3) if R0 is None else R0.copy()
    t = np.zeros(3) if t0 is None else t0.copy()
    thr = 0.15
    rms, frac = 1.0, 0.0
    for it in range(iters):
        X = P @ R.T + t
        z = X[:, 2]
        good = z > 0.2
        zs = np.where(good, z, 1)
        col = np.floor(ax * X[:, 0] / zs + bx).astype(int)
        row = np.floor(ay * X[:, 1] / zs + by).astype(int)
        good &= (col >= 0) & (col < W) & (row >= 0) & (row < H)
        Q = np.zeros_like(X)
        Nq = np.zeros_like(X)
        Q[good] = pc_color[row[good], col[good]]
        Nq[good] = N[row[good], col[good]]
        good &= valid_mask(Q[None])[0] & (np.abs(Nq).sum(1) > 0.5)
        d_pt = np.linalg.norm(X - Q, axis=1)
        good &= d_pt < thr
        if good.sum() < 500:
            break
        Xg, Qg, Ng = X[good], Q[good], Nq[good]
        r = np.einsum('ij,ij->i', Xg - Qg, Ng)
        rms = float(np.sqrt(np.mean(r ** 2)))
        frac = float(np.mean(np.abs(r) < 0.01))
        # robust weights (Huber-ish)
        s = max(0.004, 1.5 * np.median(np.abs(r)))
        w = 1.0 / np.maximum(1.0, np.abs(r) / s)
        A = np.hstack([np.cross(Xg, Ng), Ng])
        Aw = A * w[:, None]
        x = np.linalg.solve(Aw.T @ A + 1e-9 * np.eye(6), -(Aw.T @ r))
        dR = _rot(x[:3])
        R, t = dR @ R, dR @ t + x[3:]
        thr = max(0.03, min(thr, 4 * s))
        if np.linalg.norm(x) < 1e-6:
            break
    return R, t, rms, frac


def detect_spots(diff, max_spots, min_peak=0.12, radius=22):
    """Find up to max_spots bright spots in a difference image (H, W) float.
    Returns list of (row, col, peak, compact) with sub-pixel centroids.
    compact = share of the window's above-half-peak pixels that touch the peak blob
    (1.0 = one tidy spot; low = smeared change, e.g. a person moving)."""
    img = diff.copy()
    out = []
    H, W = img.shape
    for _ in range(max_spots):
        idx = int(np.argmax(img))
        r, c = divmod(idx, W)
        peak = float(img[r, c])
        if peak < min_peak:
            break
        r0, r1 = max(0, r - radius), min(H, r + radius + 1)
        c0, c1 = max(0, c - radius), min(W, c + radius + 1)
        patch = img[r0:r1, c0:c1]
        wts = np.clip(patch - 0.5 * peak, 0, None)
        if wts.sum() > 0:
            rr, cc = np.mgrid[r0:r1, c0:c1]
            # a projected dot is a compact blob; edge pixels of the window should be dark
            border = np.r_[patch[0], patch[-1], patch[:, 0], patch[:, -1]]
            compact = 1.0 - float(np.mean(border > 0.5 * peak))
            out.append((float((rr * wts).sum() / wts.sum()), float((cc * wts).sum() / wts.sum()),
                        peak, compact))
        img[r0:r1, c0:c1] = 0
    return out


def lookup_3d(pc, row, col, r=3):
    H, W = pc.shape[:2]
    ri, ci = int(round(row)), int(round(col))
    patch = pc[max(0, ri - r):ri + r + 1, max(0, ci - r):ci + r + 1].reshape(-1, 3)
    ok = valid_mask(patch[None])[0]
    if ok.sum() < 5:
        return None
    P = patch[ok]
    med = np.median(P, axis=0)
    # reject spots straddling a depth edge
    if np.max(np.abs(P[:, 2] - med[2])) > 0.03:
        return None
    return med


def planarity(X):
    """Smallest / largest singular value of the centered points (0 = all on one plane)."""
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    return float(s[-1] / max(s[0], 1e-9))
