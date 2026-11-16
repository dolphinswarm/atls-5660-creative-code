# pose_filter_callbacks.py  (Script CHOP: show/pose_filter)
#
# Takes the raw head pose from /project1/banjo_info (via select_pose) and turns
# it into something that looks good on stage:
#   * One-Euro smoothing (low jitter when still, low lag when moving)
#   * velocity prediction to hide camera + projector latency
#   * hold-then-fade when tracking is lost
#   * a Simulate mode that fakes a swaying banjo for testing with no hardware
#
# Input 0: select_pose  (tx ty tz rx ry rz valid ... from banjo_info)
# Input 1: null_time    (channel 't', seconds; also forces a cook every frame)
#
# Controls live on the Show COMP's "Tracking" page.
#
# Outputs (1 sample):
#   m0..m15    head transform, column-major (feeds geo_head Xform Matrix CHOP).
#              Includes scale = head radius * Headscale, so a 1x1 unit
#              rectangle (-0.5..0.5) becomes the head. NOTE: rectangle is 1 unit,
#              so scale = diameter.
#   tx ty tz   predicted head center (world = depth camera space, meters)
#   vx vy vz   smoothed velocity, m/s
#   vlx vly    velocity in the head's own plane, in head-diameters / s
#   upx upy    world-up direction in the head's plane (unit 2D)
#   speed      |v| m/s
#   visible    0..1 fade (1 = tracking, ramps to 0 when lost)
#   tracking   1 when a fresh pose arrived within Holdtime
#   age        seconds since last fresh pose
#   diameter   head diameter used, meters
#   t, dt      time (s) and frame delta

import math
import numpy as np

JUMP_M = 0.015      # single-frame jumps bigger than this (beyond predicted motion) need confirming

_s = {
    'have': False,
    'p': None, 'v': np.zeros(3), 'q': None,
    'raw_last': None, 'last_fresh_t': -1e9, 'last_t': None,
    'sample_t': None, 'diam': None, 'layout_key': None,
}


# ---------------- small math helpers ----------------

def _rot_from_euler_xyz(rx, ry, rz):
    # Matches banjo_info's _euler_xyz: R = Rz @ Ry @ Rx  (TD rotate order xyz)
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_from_rot(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def _rot_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + t * (b - a)
        return r / np.linalg.norm(r)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / s


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * math.pi * max(cutoff, 1e-3))
    return 1.0 / (1.0 + tau / max(dt, 1e-4))


# ---------------- inputs ----------------

def _head_diameter():
    """Diameter from the rim dots in dot_layout (local coords, center = origin)."""
    t = parent.Project.op('dot_layout')
    if t is None:
        return 0.28
    key = t.text
    if key == _s['layout_key'] and _s['diam']:
        return _s['diam']
    rs = []
    for r in range(1, t.numRows):
        if t[r, 0].val.startswith('rim'):
            try:
                rs.append(math.hypot(float(t[r, 1]), float(t[r, 2])))
            except Exception:
                pass
    d = 2 * (sum(rs) / len(rs)) if rs else 0.28
    _s['diam'], _s['layout_key'] = d, key
    return d


def _sim_pose(t):
    """A gently swaying banjo ~1.3 m in front of the projector."""
    show = parent.Show
    cw = show.ProjectorWorldMatrix()          # 4x4 numpy, camera-to-world
    cpos = cw[:3, 3]
    right, up, back = cw[:3, 0], cw[:3, 1], cw[:3, 2]
    fwd = show.CenterRayWorld()      # middle of the projected image (projectors have lens shift)
    p = cpos + fwd * 1.3 + right * (0.18 * math.sin(0.7 * t)) + up * (0.07 * math.sin(1.3 * t + 1.0)) \
        + fwd * (0.10 * math.sin(0.45 * t))
    # head faces the projector, tilted a bit, neck pointing to stage-left
    base = np.stack([right, up, back], axis=1)
    tilt = _rot_from_euler_xyz(18 * math.sin(0.6 * t), 22 * math.sin(0.4 * t + 0.5), -30 + 10 * math.sin(0.3 * t))
    return p, base @ tilt


# ---------------- cook ----------------

def onSetupParameters(scriptOp):
    return


def onPulse(par):
    return


def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.numSamples = 1
    show = parent.Show
    tin = scriptOp.inputs[1] if len(scriptOp.inputs) > 1 else None
    now = float(tin['t'][0]) if tin is not None and tin.chan('t') is not None else 0.0
    dt = 1.0 / 60.0 if _s['last_t'] is None else max(1e-4, min(0.25, now - _s['last_t']))
    _s['last_t'] = now

    mincut = float(show.par.Smoothing)        # Hz, lower = smoother
    beta = float(show.par.Responsiveness)     # higher = less lag on fast moves
    predict = float(show.par.Predictms) / 1000.0
    hold = float(show.par.Holdtime)
    fade = max(0.01, float(show.par.Fadetime))
    simulate = bool(show.par.Simulate)

    # ---------- get a raw measurement (only when it's a NEW sample) ----------
    fresh, raw_p, raw_R = False, None, None
    if simulate:
        raw_p, raw_R = _sim_pose(now)
        fresh = True
    else:
        src = scriptOp.inputs[0] if scriptOp.inputs else None
        if src is not None and src.chan('valid') is not None and src['valid'][0] > 0.5:
            vals = tuple(round(float(src[c][0]), 6) for c in ('tx', 'ty', 'tz', 'rx', 'ry', 'rz'))
            # banjo_info cooks at 60 fps but the camera delivers 30 fps:
            # repeated identical poses are not new samples.
            if vals != _s['raw_last']:
                _s['raw_last'] = vals
                raw_p = np.array(vals[:3])
                raw_R = _rot_from_euler_xyz(*vals[3:])
                fresh = True

    # ---------- jump rejection ----------
    # The tracker occasionally jumps 2-3 cm for a single camera frame (a dot dropping
    # out, bad depth on the tape). A real move continues on the next frame; a glitch
    # doesn't. So hold a big jump back one frame and only accept it if confirmed.
    if fresh and _s['have'] and not simulate and (now - _s['last_fresh_t']) <= hold:
        sdt = max(1e-3, now - _s['sample_t'])
        pred = _s['p'] + _s['v'] * sdt
        jump = float(np.linalg.norm(raw_p - pred))
        limit = JUMP_M + 0.05 * float(np.linalg.norm(_s['v']))     # allow more when moving fast
        pend = _s.get('pending')
        if jump > limit:
            if pend is not None and np.linalg.norm(raw_p - pend) < limit:
                _s['pending'] = None            # confirmed by a second frame: accept
            else:
                _s['pending'] = raw_p.copy()    # wait for confirmation
                _s['rejected'] = _s.get('rejected', 0) + 1
                fresh = False
                _s['last_fresh_t'] = now        # still tracking, don't start fading
        else:
            _s['pending'] = None

    # ---------- filter ----------
    if fresh:
        if not _s['have'] or (now - _s['last_fresh_t']) > hold + fade:
            # (re)acquire: snap, no smoothing from a stale pose
            _s['p'], _s['v'], _s['q'] = raw_p.copy(), np.zeros(3), _quat_from_rot(raw_R)
            _s['sample_t'] = now
            _s['have'] = True
        else:
            sdt = max(1e-3, now - _s['sample_t'])
            _s['sample_t'] = now
            # position: one-euro
            dx = (raw_p - _s['p']) / sdt
            a_d = _alpha(1.0, sdt)
            _s['v'] = _s['v'] + a_d * (dx - _s['v'])
            cut = mincut + beta * float(np.linalg.norm(_s['v']))
            a = _alpha(cut, sdt)
            _s['p'] = _s['p'] + a * (raw_p - _s['p'])
            # rotation: adaptive slerp
            qn = _quat_from_rot(raw_R)
            ang = 2 * math.acos(min(1.0, abs(float(np.dot(qn, _s['q'])))))
            cut_r = mincut + beta * 0.15 * (ang / sdt)
            _s['q'] = _slerp(_s['q'], qn, _alpha(cut_r, sdt))
        _s['last_fresh_t'] = now

    age = now - _s['last_fresh_t']
    tracking = 1.0 if age <= hold else 0.0
    if not _s['have']:
        visible = 0.0
    elif age <= hold:
        visible = 1.0
    else:
        visible = max(0.0, 1.0 - (age - hold) / fade)

    # ---------- predict ----------
    if _s['have']:
        since = min(now - (_s['sample_t'] or now), 0.1)      # between camera frames
        lead = (since + predict) if tracking else 0.0
        p = _s['p'] + _s['v'] * lead
        R = _rot_from_quat(_s['q'])
    else:
        p = np.zeros(3)
        R = np.eye(3)

    diam = _head_diameter() * float(show.par.Headscale)
    M = np.eye(4)
    M[:3, :3] = R * diam
    M[:3, 3] = p

    v = _s['v'] if tracking else np.zeros(3)
    vl = R.T @ v / max(diam, 1e-3)
    # world "up" (+y, assumes the camera is mounted level) seen in the head's plane,
    # so content can stay upright and flames rise however the banjo is tilted
    upl = R.T @ np.array([0.0, 1.0, 0.0])
    un = math.hypot(upl[0], upl[1])
    up2 = (upl[0] / un, upl[1] / un) if un > 1e-3 else (0.0, 1.0)

    out = [('m%d' % i, float(M[i % 4, i // 4])) for i in range(16)]    # column-major
    out += [('tx', p[0]), ('ty', p[1]), ('tz', p[2]),
            ('vx', v[0]), ('vy', v[1]), ('vz', v[2]),
            ('vlx', vl[0]), ('vly', vl[1]),
            ('upx', up2[0]), ('upy', up2[1]),
            ('speed', float(np.linalg.norm(v))),
            ('visible', visible), ('tracking', tracking), ('age', min(age, 999.0)),
            ('diameter', diam), ('t', now), ('dt', dt),
            ('rejected', float(_s.get('rejected', 0)))]
    for name, val in out:
        scriptOp.appendChan(name)[0] = float(val)
    return
