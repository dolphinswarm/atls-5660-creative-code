# banjo_pose.py
# Paste into the callbacks DAT of a Script CHOP in TouchDesigner.
#
# Reads:
#   - dot_layout        Table DAT: your marker layout (name, x, y, z in meters)
#                       -> created/filled automatically by the Calibrate page
#   - info_blobs        Info DAT attached to the Blob Track TOP
#   - orbbec_pointcloud Orbbec Select TOP set to Point Cloud
#
# CALIBRATION (Calibrate page on the Script CHOP; click "Setup Parameters"
# on the Script CHOP once if the page doesn't appear):
#   1. Hold the banjo still, facing the camera. Cover the two arm-rest dots
#      so only the 3 rim dots are visible. Press "1. Capture Rim".
#   2. Uncover them so all 5 dots are visible. Press "2. Capture All".
#      dot_layout is written for you. The banjo may move between steps.
#   Head center = circle through the 3 rim dots. +x points from the center
#   toward the arm rest, +z points out of the head (toward the camera at
#   capture time), y completes the right-handed frame.
#
# Outputs (1 sample each):
#   tx ty tz   banjo-head center in meters (camera space)
#   rx ry rz   rotation in degrees (Rx Ry Rz order, TouchDesigner's default)
#   nx ny nz   unit normal of the head face
#   rms        fit error in meters (lower = better)
#   ndots      dots used in the fit
#   valid      1 = fresh pose this frame, 0 = holding last good pose
#   cal_state  0 idle, 1 capturing rim, 2 capturing all
#   cal_rim_ok 1 once step 1 has been captured
#   dbg_*      diagnostics (delete once everything works)

import numpy as np
from itertools import permutations, combinations

# ---------- settings ----------
LAYOUT_DAT = 'dot_layout'
BLOBS_DAT = 'info_blobs'
POINTCLOUD_TOP = 'orbbec_pointcloud'

ID_COLS = ('id',)
U_COLS = ('u', 'x', 'tx', 'centerx', 'posx')
V_COLS = ('v', 'y', 'ty', 'centery', 'posy')

FLIP_V = False             # True if blob v counts from the TOP of the image
PC_CHANNELS = (0, 1, 2)    # which point-cloud channels hold x, y, z
CAMERA_TO_TD = False       # True if the point cloud is y-down / z-forward (OpenCV style)
SAMPLE_RADIUS = 4          # px around each blob to sample depth (tape center is often invalid)
MAX_RMS = 0.015            # m; reject fits worse than this (was 0.012: rejected ~17% of frames)
MIN_DOTS = 3               # 4+ is far more stable; 3 is the bare minimum
RIM_COUNT = 3              # dots on the rim (step 1)
ALL_COUNT = 7              # total dots (step 2): 3 rim + armrest x2 + tailpiece + headstock
# ------------------------------

_cache = {'assign': {}, 'last': None}
_cal = {'mode': None, 'left': 0, 'total': 0, 'acc': {}, 'rim_d': None}


# ---------------- parameters / pulses ----------------

def onSetupParameters(scriptOp):
    page = scriptOp.appendCustomPage('Calibrate')
    page.appendPulse('Capturerim', label='1. Capture Rim (3 dots)')
    page.appendPulse('Captureall', label='2. Capture All (5 dots)')
    p = page.appendInt('Capframes', label='Frames to Average')
    p[0].default = 30
    p[0].val = 30
    p[0].normMin = 5
    p[0].normMax = 120
    return


def onPulse(par):
    frames = 30
    try:
        frames = max(1, int(par.owner.par.Capframes.eval()))
    except Exception:
        pass
    if par.name == 'Capturerim':
        _start_capture('rim', frames)
    elif par.name == 'Captureall':
        if _cal['rim_d'] is None:
            debug('banjo_pose: do step 1 (Capture Rim) first')
            return
        _start_capture('all', frames)
    return


def _start_capture(mode, frames):
    _cal['mode'] = mode
    _cal['left'] = frames
    _cal['total'] = frames
    _cal['acc'] = {}
    debug('banjo_pose: capturing', mode, 'for', frames, 'frames - hold still')


# ---------------- reading inputs ----------------

def _find_col(header, names):
    for n in names:
        if n in header:
            return header.index(n)
    return None


def _read_layout():
    t = op(LAYOUT_DAT)
    names, pts = [], []
    if t is None:
        return names, np.zeros((0, 3))
    for r in range(1, t.numRows):
        try:
            pts.append([float(t[r, 1]), float(t[r, 2]), float(t[r, 3])])
            names.append(t[r, 0].val)
        except Exception:
            pass
    return names, np.array(pts, dtype=np.float64).reshape(-1, 3)


def _read_blobs(w, h):
    t = op(BLOBS_DAT)
    if t is None or t.numRows < 2:
        return []
    header = [c.val.strip().lower() for c in t.row(0)]
    ci = _find_col(header, ID_COLS)
    cu = _find_col(header, U_COLS)
    cv = _find_col(header, V_COLS)
    if cu is None or cv is None:
        debug('banjo_pose: cannot find u/v columns in', header)
        return []
    blobs = []
    for r in range(1, t.numRows):
        bid = t[r, ci].val if ci is not None else str(r)
        blobs.append([bid, float(t[r, cu]), float(t[r, cv])])
    if blobs and max(max(b[1], b[2]) for b in blobs) <= 1.0:
        for b in blobs:
            b[1] *= w
            b[2] *= h
    return blobs


def _sample(pc, u, v):
    h, w = pc.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if FLIP_V:
        y = h - 1 - y
    r = SAMPLE_RADIUS
    patch = pc[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1][..., list(PC_CHANNELS)]
    patch = patch.reshape(-1, 3).astype(np.float64)
    ok = np.all(np.isfinite(patch), axis=1) & (np.abs(patch[:, 2]) > 1e-4)
    good = patch[ok]
    if len(good) < 3:
        return None
    p = np.median(good, axis=0)
    if CAMERA_TO_TD:
        p = p * np.array([1.0, -1.0, -1.0])
    return p


# ---------------- calibration math ----------------

def _pair_dists(P):
    return np.sort([np.linalg.norm(P[i] - P[j]) for i, j in combinations(range(len(P)), 2)])


def _circle_3d(a, b, c):
    """Center, radius and unit normal of the circle through 3 points."""
    ab, ac = b - a, c - a
    n = np.cross(ab, ac)
    nn = np.dot(n, n)
    if nn < 1e-12:
        return None
    center = a + (np.dot(ac, ac) * np.cross(n, ab) + np.dot(ab, ab) * np.cross(ac, n)) / (2.0 * nn)
    return center, np.linalg.norm(a - center), n / np.sqrt(nn)


def _averaged_points(min_fraction=0.5):
    need = max(1, int(_cal['total'] * min_fraction))
    return np.array([np.mean(v, axis=0) for v in _cal['acc'].values() if len(v) >= need])


def _finish_capture():
    mode = _cal['mode']
    _cal['mode'] = None
    P = _averaged_points()

    if mode == 'rim':
        if len(P) != RIM_COUNT:
            debug('banjo_pose: rim capture saw', len(P), 'steady dots, need', RIM_COUNT,
                  '- cover the arm-rest dots and try again')
            return
        _cal['rim_d'] = _pair_dists(P)
        circ = _circle_3d(*P)
        r = circ[1] if circ else 0
        debug('banjo_pose: rim captured, radius %.3f m. Now uncover all dots and press Capture All.' % r)
        return

    # mode == 'all'
    if len(P) != ALL_COUNT:
        debug('banjo_pose: full capture saw', len(P), 'steady dots, need', ALL_COUNT)
        return

    # Which 3 are the rim dots? The triple whose shape best matches step 1.
    best = None
    for tri in combinations(range(len(P)), RIM_COUNT):
        err = np.abs(_pair_dists(P[list(tri)]) - _cal['rim_d']).sum()
        if best is None or err < best[0]:
            best = (err, tri)
    err, tri = best
    if err > 0.03:
        debug('banjo_pose: warning - rim dots only matched within %.1f mm total' % (err * 1000))
    rim_idx = list(tri)
    arm_idx = [i for i in range(len(P)) if i not in tri]

    circ = _circle_3d(*P[rim_idx])
    if circ is None:
        debug('banjo_pose: rim dots are in a line - cannot fit a circle')
        return
    center, radius, n = circ
    if np.dot(n, center) > 0:   # make +z point toward the camera (out of the head)
        n = -n
    d = P[arm_idx].mean(axis=0) - center
    x = d - np.dot(d, n) * n
    if np.linalg.norm(x) < 1e-6:
        x = P[rim_idx[0]] - center
        x = x - np.dot(x, n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)

    def local(p):
        q = p - center
        return [float(np.dot(q, x)), float(np.dot(q, y)), float(np.dot(q, n))]

    rows = []
    rim_sorted = sorted(rim_idx, key=lambda i: np.arctan2(local(P[i])[1], local(P[i])[0]))
    for k, i in enumerate(rim_sorted):
        rows.append(['rim%d' % (k + 1)] + local(P[i]))
    arm_sorted = sorted(arm_idx, key=lambda i: local(P[i])[1])
    for k, i in enumerate(arm_sorted):
        rows.append(['arm%d' % (k + 1)] + local(P[i]))

    t = op(LAYOUT_DAT)
    if t is None:
        t = me.parent().create(tableDAT, LAYOUT_DAT)
    t.clear()
    t.appendRow(['name', 'x', 'y', 'z'])
    for row in rows:
        t.appendRow([row[0]] + ['%.5f' % v for v in row[1:]])

    _cache['assign'] = {}
    _cache['last'] = None
    debug('banjo_pose: layout written to %s (rim radius %.3f m)' % (LAYOUT_DAT, radius))


# ---------------- pose solving ----------------

def _kabsch_batch(A, B):
    """A: (m,k,3) model candidates, B: (k,3) measured. Returns R (m,3,3), t (m,3), rms (m,)."""
    ca = A.mean(axis=1)
    cb = B.mean(axis=0)
    A0 = A - ca[:, None, :]
    B0 = B - cb
    H = np.einsum('mki,kj->mij', A0, B0)
    U, S, Vt = np.linalg.svd(H)
    V = np.transpose(Vt, (0, 2, 1))
    Ut = np.transpose(U, (0, 2, 1))
    d = np.sign(np.linalg.det(V @ Ut))
    D = np.tile(np.eye(3), (len(A), 1, 1))
    D[:, 2, 2] = d
    R = V @ D @ Ut
    t = cb - np.einsum('mij,mj->mi', R, ca)
    pred = np.einsum('mij,mkj->mki', R, A) + t[:, None, :]
    rms = np.sqrt(((pred - B) ** 2).sum(-1).mean(-1))
    return R, t, rms


def _search(model, meas):
    n, k = len(model), len(meas)
    best = None
    det_sets = [tuple(range(k))] if k <= n else list(combinations(range(k), n))
    for ds in det_sets:
        B = meas[list(ds)]
        perms = np.array(list(permutations(range(n), len(ds))), dtype=int)
        if perms.size == 0:
            continue
        R, t, rms = _kabsch_batch(model[perms], B)
        i = int(np.argmin(rms))
        if best is None or rms[i] < best[3]:
            best = (ds, perms[i], (R[i], t[i]), rms[i])
    return best


def _euler_xyz(R):
    ry = np.degrees(np.arcsin(-np.clip(R[2, 0], -1.0, 1.0)))
    rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return rx, ry, rz


def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.numSamples = 1

    names, model = _read_layout()
    pct = op(POINTCLOUD_TOP)
    w, h = pct.width, pct.height
    blobs = _read_blobs(w, h)
    # The camera delivers 30 fps but TD cooks at 60: on a repeated camera frame the
    # blobs are identical, so re-send the last result instead of redoing the work.
    sig = tuple((b[0], round(b[1], 2), round(b[2], 2)) for b in blobs)
    if not _cal['mode'] and sig == _cache.get('sig') and _cache.get('out'):
        for name, val in _cache['out']:
            scriptOp.appendChan(name)[0] = val
        return
    _cache['sig'] = sig
    pc = pct.numpyArray(delayed=False)
    ids, meas, uvs = [], [], []
    for bid, u, v in blobs:
        p = _sample(pc, u, v)
        if p is not None:
            ids.append(bid)
            meas.append(p)
            uvs.append((u, v))
    meas = np.array(meas).reshape(-1, 3)

    # ---- calibration capture ----
    if _cal['mode']:
        for i, p in zip(ids, meas):
            _cal['acc'].setdefault(i, []).append(p)
        _cal['left'] -= 1
        if _cal['left'] <= 0:
            _finish_capture()
            names, model = _read_layout()

    # ---- pose ----
    dbg_fit = -1.0
    pose, rms_out, used = None, 0.0, len(meas)
    if len(model) >= MIN_DOTS and len(meas) >= MIN_DOTS:
        pose, rms_out, used = _track(model, ids, meas)
        # lost: full search on the 5 core dots (rim + armrest) and at most SEARCH_CAP
        # blobs; tail/peg are picked up by _track on the next frame
        _cache['lost'] = 0 if pose is not None else _cache.get('lost', 0) + 1
        # while lost, only search every 3rd frame (the search costs ~20-40 ms)
        lost = _cache['lost']
        best = _search(model[:5], _cap(meas)) if (pose is None and (lost < 3 or lost % 3 == 0)) else None
        if best is not None:
            sub = _cap_idx(meas)
            best = ([sub[d] for d in best[0]],) + tuple(best[1:])
        if best is not None:
            ds, perm, (R, t), rms = best
            dbg_fit = float(rms)
            if rms < MAX_RMS:
                pose, rms_out, used = (R, t), rms, len(ds)
                _cache['assign'] = {ids[d]: int(m) for d, m in zip(ds, perm)}

    rms2d = -1.0
    if pose is not None and REFINE_2D and len(meas) >= 4:
        try:
            pose, rms2d = _refine_pose(pose, model, meas, uvs, pc)
        except Exception as e:
            debug('banjo_pose: 2D refine skipped:', e)

    valid = 1.0 if pose is not None else 0.0
    if pose is not None:
        _cache['last'] = (pose, rms_out, used)
    elif _cache['last'] is not None:
        pose, rms_out, used = _cache['last']

    if pose is None:
        vals = [0.0] * 9
    else:
        R, t = pose
        rx, ry, rz = _euler_xyz(R)
        nrm = R[:, 2]
        vals = [t[0], t[1], t[2], rx, ry, rz, nrm[0], nrm[1], nrm[2]]

    out = list(zip(['tx', 'ty', 'tz', 'rx', 'ry', 'rz', 'nx', 'ny', 'nz'], vals))
    out += [
        ('rms', rms_out), ('ndots', used), ('valid', valid),
        ('cal_state', {None: 0, 'rim': 1, 'all': 2}[_cal['mode']]),
        ('cal_rim_ok', 1 if _cal['rim_d'] is not None else 0),
        ('dbg_layout_dots', len(model)),
        ('dbg_blobs', len(blobs)),
        ('dbg_points', len(meas)),
        ('dbg_fit', dbg_fit),
        ('rms2d_px', rms2d),
        ('dbg_pc_w', w),
        ('dbg_pc_h', h),
    ]
    _cache['out'] = [(name, float(val)) for name, val in out]
    for name, val in out:
        scriptOp.appendChan(name)[0] = float(val)
    return


# ---------------- frame-to-frame tracking (added) ----------------
# Avoids the full permutation search on every frame: known blob ids keep their
# dot; unknown/new blobs are matched to the nearest predicted dot from the last
# pose. The full search only runs when tracking is lost, on at most SEARCH_CAP blobs.

SEARCH_CAP = 7
MATCH_DIST = 0.04      # m, how close a blob must be to a predicted dot


def _cap_idx(meas):
    if len(meas) <= SEARCH_CAP:
        return list(range(len(meas)))
    c = np.median(meas, axis=0)
    return list(np.argsort(np.linalg.norm(meas - c, axis=1))[:SEARCH_CAP])


def _cap(meas):
    return meas[_cap_idx(meas)]


def _track(model, ids, meas):
    assign = _cache['assign']
    last = _cache['last']
    if last is None or len(meas) < MIN_DOTS:
        return None, 0.0, len(meas)
    (R0, t0) = last[0]
    pred = model @ R0.T + t0
    pairs = {}                         # meas index -> model index
    for j, bid in enumerate(ids):
        mi = assign.get(bid)
        if mi is not None and mi < len(model) and np.linalg.norm(meas[j] - pred[mi]) < 3 * MATCH_DIST:
            pairs[j] = mi
    used_m = set(pairs.values())
    for j in range(len(meas)):
        if j in pairs:
            continue
        dd = np.linalg.norm(pred - meas[j], axis=1)
        for mi in np.argsort(dd):
            if dd[mi] > MATCH_DIST:
                break
            if mi not in used_m:
                pairs[j] = int(mi)
                used_m.add(int(mi))
                break
    if len(pairs) < max(MIN_DOTS, 4):
        return None, 0.0, len(meas)
    J = sorted(pairs)
    M = np.array([pairs[j] for j in J])
    R, t, rms = _kabsch_batch(model[M][None], meas[J])
    if rms[0] >= MAX_RMS:
        return None, 0.0, len(meas)
    _cache['assign'] = {ids[j]: int(pairs[j]) for j in J}
    return (R[0], t[0]), float(rms[0]), len(J)


# ---------------- 2D refinement (added) ----------------
# The depth camera's 3D reading on retroreflective tape is ~1 cm noisy, but WHERE a
# dot appears in the IR image is precise to a fraction of a pixel. After the normal
# 3D fit, refine the pose so the dots line up with their IR-image positions, using
# the measured depths only as a weak hint. Pixel rays come from the depth camera's
# own point cloud (lens distortion included), fitted once.

REFINE_2D = True
IR_TOP = 'orbbec1'
SIG_RAY = 0.0012      # ray noise (normalized image units, ~0.6 px)
SIG_Z = 0.012         # depth noise on the tape, m
REFINE_MAX_MOVE = 0.03   # m; refused if the refinement wants to move further than this


def _poly_feats(u, v):
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3,
                     u ** 4, u ** 3 * v, u * u * v * v, u * v ** 3, v ** 4,
                     u ** 5, u ** 4 * v, u ** 3 * v * v, u * u * v ** 3, u * v ** 4, v ** 5], 1)


def _ray_model(pc):
    rm = _cache.get('ray')
    if rm is not None and rm[0] == pc.shape[:2]:
        return rm
    H, W = pc.shape[:2]
    P = pc[::3, ::3, :3].astype(np.float64)
    vv, uu = np.mgrid[0:H:3, 0:W:3]
    ok = np.isfinite(P[..., 2]) & (P[..., 2] > 0.2)
    if ok.sum() < 2000:
        raise ValueError('not enough depth to fit the ray model')
    u = (uu[ok] + 0.5) / W * 2 - 1
    v = (vv[ok] + 0.5) / H * 2 - 1
    F = _poly_feats(u, v)
    ca, *_ = np.linalg.lstsq(F, P[ok][:, 0] / P[ok][:, 2], rcond=None)
    cb, *_ = np.linalg.lstsq(F, P[ok][:, 1] / P[ok][:, 2], rcond=None)
    rm = (pc.shape[:2], ca, cb)
    _cache['ray'] = rm
    return rm


def _subpix(ir, u, v, r=4):
    H, W = ir.shape[:2]
    x, y = int(u), int(v)
    x0, x1, y0, y1 = max(0, x - r), min(W, x + r + 1), max(0, y - r), min(H, y + r + 1)
    p = ir[y0:y1, x0:x1].astype(np.float64)
    if p.size == 0:
        return u, v
    w = np.clip(p - 0.5 * p.max(), 0, None)
    if w.sum() <= 0:
        return u, v
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((xx * w).sum() / w.sum()) + 0.5, float((yy * w).sum() / w.sum()) + 0.5


def _rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def _refine_pose(pose, model, meas, uvs, pc):
    R0, t0 = pose
    H, W = pc.shape[:2]
    shape, ca, cb = _ray_model(pc)
    pred = model @ R0.T + t0
    # match measured dots to model dots (unique, nearest, within 4 cm)
    pairs, used = [], set()
    for j in range(len(meas)):
        dd = np.linalg.norm(pred - meas[j], axis=1)
        mi = int(np.argmin(dd))
        if dd[mi] < 0.04 and mi not in used:
            used.add(mi)
            pairs.append((j, mi))
    if len(pairs) < 4:
        return pose, -1.0
    ir = op(IR_TOP).numpyArray(delayed=False)[..., 0]
    obs, M, zm = [], [], []
    for j, mi in pairs:
        u, v = _subpix(ir, *uvs[j])
        f = _poly_feats(np.array([u / W * 2 - 1]), np.array([v / H * 2 - 1]))
        obs.append([float(f @ ca), float(f @ cb)])
        M.append(model[mi])
        zm.append(meas[j][2])
    obs, M, zm = np.array(obs), np.array(M), np.array(zm)
    base = M @ R0.T                     # model rotated by the initial pose

    def resid(x):
        X = base @ _rotvec(x[:3]).T + t0 + x[3:]
        r2 = (X[:, :2] / X[:, 2:3] - obs) / SIG_RAY
        rz = (X[:, 2] - zm) / SIG_Z
        return np.concatenate([r2.ravel(), rz])

    x = np.zeros(6)
    for _ in range(3):                  # starts close; 3 Gauss-Newton steps converge
        r = resid(x)
        J = np.empty((len(r), 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = (resid(x + dx) - r) / 1e-6
        step = np.linalg.solve(J.T @ J + 1e-6 * np.eye(6), -J.T @ r)
        x = x + step
        if np.linalg.norm(step) < 1e-7:
            break
    if np.linalg.norm(x[3:]) > REFINE_MAX_MOVE or np.linalg.norm(x[:3]) > np.radians(6):
        return pose, -1.0
    Rr = _rotvec(x[:3]) @ R0
    tr = t0 + x[3:]
    X = M @ Rr.T + tr
    e2 = np.linalg.norm(X[:, :2] / X[:, 2:3] - obs, axis=1)
    return (Rr, tr), float(np.sqrt(np.mean(e2 ** 2)) * W / 2 / max(abs(ca[1]), 1e-3))
