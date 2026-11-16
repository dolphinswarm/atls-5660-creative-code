# ShowExt.py  (extension on /project1/show, shortcut parent.Show)
#
# Owns:
#   * projector calibration (capture 3D<->2D pairs with a single reflective
#     "wand" dot, solve the projector's lens + position, save/load JSON)
#   * the matrices that drive cam_projector (calib_view / calib_proj tables)
#   * the projector output window
#
# Coordinate conventions
#   world      = Femto Bolt point-cloud space (meters; x left, y up, z forward)
#   proj px    = projector output pixels, origin BOTTOM-left (TD texture space)
#   OpenCV     = pixels origin TOP-left, camera looks +z, y down

import json
import math
import os
import sys
import time

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

GRID_X = (0.2, 0.5, 0.8)
GRID_Y = (0.78, 0.5, 0.22)
# banjo mode: 5 targets per pass (center + 4 corners) -> 10 total
GRID_BANJO = ((0.5, 0.5), (0.27, 0.27), (0.73, 0.27), (0.73, 0.73), (0.27, 0.73))
PASSES =('NEAR - one step closer to the projector than where you play',
          'FAR - one step further away than where you play')
NEAR, FAR = 0.05, 30.0


class ShowExt:
    def __init__(self, ownerComp):
        self.owner = ownerComp
        self.Target = tdu.Dependency([0.0, 0.0, 0.0, 0.0])   # u px, v px, progress 0..1, active
        self.Status = tdu.Dependency('')
        self.Dot = tdu.Dependency([0.0, 0.0, 10.0, 0.0])      # autocal dot: u px, v px, radius, on
        self._ac = None
        self._ck = None
        p = os.path.join(project.folder, 'code', 'python', 'show')     # autocal_math.py
        if p not in sys.path:
            sys.path.insert(0, p)
        self.calib = None
        self.points = []          # [{'px':[u,v], 'world':[x,y,z]}]
        self.targets = []
        self.ti = 0
        self.calibrating = False
        self._buf = []            # [(t, xyz)]
        self._lock_pt = None
        self.Loadcalib(quiet=True)
        if self.calib is None:
            self.Apply()

    # ------------------------------------------------------------ utilities
    @property
    def W(self):
        return int(self.owner.par.Resw)

    @property
    def H(self):
        return int(self.owner.par.Resh)

    def _say(self, msg):
        self.Status.val = msg
        try:
            self.owner.par.Calibstatus = msg.replace('\n', '  ')
        except Exception:
            pass
        debug('show:', msg)

    def _calib_path(self):
        p = str(self.owner.par.Calibfile) or 'calibration/projector_calib.json'
        return p if os.path.isabs(p) else os.path.join(project.folder, p)

    # ------------------------------------------------------------ pulses
    def OnPulse(self, name):
        fn = {
            'Startcalib': self.Startcalib, 'Capture': self.Capture, 'Skip': self.Skip,
            'Undo': self.Undo, 'Solve': self.Solve, 'Savecalib': self.Savecalib,
            'Loadcalib': self.Loadcalib, 'Clearcalib': self.Clearcalib,
            'Openwindow': self.Openwindow, 'Closewindow': self.Closewindow,
            'Stopcalib': self.Stopcalib,
            'Autocal': self.Autocal, 'Stopautocal': self.Stopautocal,
            'Checkcalib': self.Checkcalib,
        }.get(name)
        if fn:
            fn()

    def OnValueChange(self, name):
        if name in ('Resw', 'Resh') and self.calib is None:
            self.Apply()

    # ------------------------------------------------------------ window
    def Openwindow(self):
        w = self.owner.op('window_projector')
        w.par.winopen.pulse()

    def Closewindow(self):
        self.owner.op('window_projector').par.winclose.pulse()

    # ------------------------------------------------------------ calibration flow
    def Startcalib(self):
        self.targets = []
        for pi in range(len(PASSES)):
            if self._banjo_mode():
                for gx, gy in GRID_BANJO:
                    self.targets.append((gx * self.W, gy * self.H, pi))
            else:
                for gy in GRID_Y:
                    for gx in GRID_X:
                        self.targets.append((gx * self.W, gy * self.H, pi))
        self.points = []
        self.ti = 0
        self._buf = []
        self._lock_pt = None
        self.calibrating = True
        self.owner.par.Mode = 'calibrate'
        self._update_target()

    def Stopcalib(self):
        self.calibrating = False
        self.Target.val = [0.0, 0.0, 0.0, 0.0]
        self._say('Calibration stopped (%d points kept).' % len(self.points))

    def _update_target(self):
        if self.ti >= len(self.targets):
            self.calibrating = False
            self.Target.val = [0.0, 0.0, 0.0, 0.0]
            self._say('All %d targets done - solving...' % len(self.targets))
            self.Solve()
            return
        u, v, pi = self.targets[self.ti]
        self.Target.val = [u, v, 0.0, 1.0]
        what = ('Put the CENTER of the banjo head on the crosshair (face it at the projector) and hold still.'
                if self._banjo_mode() else
                'Put ONE reflective dot on the crosshair and hold still.')
        self._say('Point %d/%d   %s\n%s' % (self.ti + 1, len(self.targets), PASSES[pi], what))

    def Capture(self, pt=None):
        if not self.calibrating:
            self._say('Press Start Calibration first.')
            return
        if pt is None:
            recent = [p for t, p in self._buf if time.perf_counter() - t < 0.4]
            if not recent:
                self._say('Capture: no single dot visible right now.')
                return
            pt = np.mean(recent, axis=0)
        u, v, _ = self.targets[self.ti]
        self.points.append({'px': [float(u), float(v)], 'world': [float(x) for x in pt]})
        self._lock_pt = np.array(pt)
        self._buf = []
        self.ti += 1
        self._update_target()

    def Skip(self):
        if self.calibrating:
            self.ti += 1
            self._update_target()

    def Undo(self):
        if self.points:
            self.points.pop()
            if self.calibrating:
                self.ti = max(0, self.ti - 1)
                self._update_target()
            else:
                self._say('Removed last point (%d left).' % len(self.points))

    def Clearcalib(self):
        self.calib = None
        self.points = []
        self.Apply()
        self._say('Calibration cleared (using default projector guess).')

    # ------------------------------------------------------------ automatic calibration
    # Projects one dot at a time; the Bolt's color camera finds it, the depth camera
    # (aligned to color) gives its 3D position, converted back into the depth frame the
    # tracker uses. Needs: dim room, projector on, a scene with some depth variety
    # (wall + furniture, or the banjo on a stand in front of a wall).

    AC_COLS, AC_ROWS, AC_MARGIN = 8, 5, 0.06
    AC_SETTLE = 0.9           # s between changing the pattern and grabbing the camera frame
                              # (projector + color camera latency measured well over 0.4 s)
    AC_BG_EVERY = 3           # fresh black reference every few dots (drift / motion)

    def _extr_path(self):
        return os.path.join(project.folder, 'calibration', 'bolt_depth_to_color.json')

    def _load_extr(self):
        try:
            with open(self._extr_path()) as f:
                d = json.load(f)
            return np.array(d['R']), np.array(d['t'])
        except Exception:
            return None

    def _cam(self):
        root = parent.Project
        return root.op('orbbec1'), root.op('orbbec_pointcloud'), root.op('orbbec_color')

    def _gray(self):
        c = self._cam()[2]
        c.cook(force=True)          # nothing downstream, so it doesn't cook by itself
        a = c.numpyArray(delayed=False)
        return a[..., :3].mean(axis=2).astype(np.float32)

    def Autocal(self):
        if self.calibrating or getattr(self, '_ac', None):
            self._say('Already calibrating.')
            return
        import importlib
        import autocal_math
        importlib.reload(autocal_math)      # pick up edits on disk
        if self._cam()[2] is None:
            self._say('Auto calibration needs /project1/orbbec_color (Orbbec Select TOP, Image = Color).')
            return
        if len(monitors) < 2:
            self._say('Projector is not connected as a display (only one monitor found).')
            return
        targets = []
        m = self.AC_MARGIN
        for j in range(self.AC_ROWS):
            gy = m + (1 - 2 * m) * j / (self.AC_ROWS - 1)
            row = []
            for i in range(self.AC_COLS):
                gx = m + (1 - 2 * m) * i / (self.AC_COLS - 1)
                row.append((gx * self.W, gy * self.H))
            targets += row if j % 2 == 0 else row[::-1]
        self._ac = {'state': 'unalign', 't0': time.perf_counter(), 'targets': targets, 'i': 0,
                    'found': [], 'missed': 0, 'extr': self._load_extr(), 'pc_d': None, 'pc_c': None}
        self.points = []
        self.Dot.val = [0.0, 0.0, 10.0, 0.0]
        self.owner.par.Mode = 'autocal'
        self._cam()[0].par.depthalignmode = 'disabled'
        self._say('Auto calibration: keep everything still and stay out of the camera view (~1 min)...')

    # ------------------------------------------------------------ calibration check
    # Flashes a few dots, finds where they land with the color camera, looks up the
    # 3D spot from the (unaligned) depth camera and asks the stored calibration where
    # it thinks that spot is in the projector image. If the rig has moved, the two
    # disagree. Works even if furniture moved, because it measures depth fresh.

    CK_DOTS = 6
    CK_OK_PX, CK_WARN_PX = 8.0, 20.0

    def Checkcalib(self):
        if self.calibrating or self._ac or getattr(self, '_ck', None):
            self._say('Busy - finish or stop the current calibration first.')
            return
        if self.calib is None or not self.calib.get('points'):
            self._say('Check: no saved calibration to check. Run Auto Calibrate.')
            return
        try:
            with open(self._extr_path()) as f:
                dd = json.load(f)
            R, t, intr = np.array(dd['R']), np.array(dd['t']), dd['color_intr']
        except Exception:
            self._say('Check: run Auto Calibrate once more first (it stores what the check needs).')
            return
        if len(monitors) < 2:
            self._say('Check: projector is not connected as a display.')
            return
        import importlib
        import autocal_math
        importlib.reload(autocal_math)
        # pick dots spread over the image from the ones that worked during calibration
        cand = [p for p in self.calib['points'] if 'cam_px' in p]
        cand = cand or self.calib['points']
        P = np.array([p['px'] for p in cand], dtype=float)
        sel = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
        while len(sel) < min(self.CK_DOTS, len(cand)):
            d = np.min(np.linalg.norm(P[:, None] - P[sel][None], axis=2), axis=1)
            sel.append(int(np.argmax(d)))
        self._ck = {'state': 'bg', 't0': time.perf_counter(), 'i': 0, 'targets': [cand[i] for i in sel],
                    'R': R, 't': t, 'intr': intr, 'errs': [], 'mode': str(self.owner.par.Mode),
                    'map': None}
        self.Dot.val = [0.0, 0.0, 10.0, 0.0]
        self.owner.par.Mode = 'autocal'
        self._say('Checking calibration (~10 s): keep the rig still, stand clear of the projector beam...')

    def _ck_depthmap(self, ck):
        """Depth-frame 3D point for every color pixel (nearest surface), from the live
        unaligned cloud projected through the stored depth->color transform."""
        orb, pct, col = self._cam()
        D = pct.numpyArray(delayed=False)[..., :3].astype(np.float64).reshape(-1, 3)
        ok = np.isfinite(D[:, 2]) & (D[:, 2] > 0.2)
        D = D[ok]
        Xc = D @ ck['R'].T + ck['t']
        ax, bx, ay, by = ck['intr']
        cc = np.floor(ax * Xc[:, 0] / Xc[:, 2] + bx).astype(int)
        rr = np.floor(ay * Xc[:, 1] / Xc[:, 2] + by).astype(int)
        W, H = col.width, col.height
        good = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
        return {'c': cc[good], 'r': rr[good], 'X': D[good], 'z': Xc[good, 2]}

    def _ck_lookup(self, mp, row, colx, r=4):
        sel = (np.abs(mp['c'] - colx) <= r) & (np.abs(mp['r'] - row) <= r)
        if sel.sum() < 3:
            return None
        z = mp['z'][sel]
        near = z <= np.min(z) + 0.05           # front-most surface only
        return np.median(mp['X'][sel][near], axis=0)

    def _ck_step(self):
        import autocal_math as am
        ck = self._ck
        now = time.perf_counter()
        el = now - ck['t0']

        def go(s):
            ck['state'], ck['t0'] = s, now

        st = ck['state']
        if st == 'bg':
            self.Dot.val = [0.0, 0.0, 10.0, 0.0]
            if el > self.AC_SETTLE + 0.15:
                ck['bg'] = self._gray()
                if ck['map'] is None:
                    ck['map'] = self._ck_depthmap(ck)
                go('show')
        elif st == 'show':
            if ck['i'] >= len(ck['targets']):
                self._ck_finish()
                return
            u, v = ck['targets'][ck['i']]['px']
            self.Dot.val = [float(u), float(v), float(self.owner.par.Autocaldot), 1.0]
            go('grab')
        elif st == 'grab':
            if el > self.AC_SETTLE:
                tgt = ck['targets'][ck['i']]
                diff = self._gray() - ck['bg']
                spots = am.detect_spots(diff, 3, min_peak=0.06)
                if len(spots) > 1:
                    s0 = spots[0]
                    spots = [s0] + [s for s in spots[1:] if math.hypot(s[0] - s0[0], s[1] - s0[1]) > 45]
                if spots and (len(spots) == 1 or spots[1][2] < 0.5 * spots[0][2]):
                    row, colx = spots[0][0], spots[0][1]
                    X = self._ck_lookup(ck['map'], row, colx)
                    if X is not None:
                        K, Rp, tp = (np.array(self.calib[k]) for k in ('K', 'R', 't'))
                        q = self._project(K, Rp, tp, X[None])[0]          # OpenCV px
                        u, v = tgt['px']
                        err = math.hypot(q[0] - u, q[1] - (self.H - v))
                        ck['errs'].append(err)
                ck['i'] += 1
                go('bg')

    def _ck_finish(self):
        ck = self._ck
        self._ck = None
        self.Dot.val = [0.0, 0.0, 10.0, 0.0]
        self.owner.par.Mode = ck['mode'] if ck['mode'] != 'autocal' else 'test'
        e = ck['errs']
        if len(e) < 3:
            self._say('Check: only saw %d of %d dots - room too bright, or someone in the beam? '
                      'Try again.' % (len(e), len(ck['targets'])))
            return
        med = float(np.median(e))
        deg = math.degrees(math.atan(med / float(self.calib['K'][0][0])))
        if med <= self.CK_OK_PX:
            verdict = 'OK - calibration still good'
        elif med <= self.CK_WARN_PX:
            verdict = 'slightly off - fine for a quick set, recalibrate if you have a minute'
        else:
            verdict = 'MOVED - run Auto Calibrate'
        self._say('Check: %s (off by %.0f px, about %.1f deg; %d dots).' % (verdict, med, deg, len(e)))

    def Stopautocal(self, msg='Auto calibration stopped.'):
        if getattr(self, '_ac', None):
            self._ac = None
            self._cam()[0].par.depthalignmode = 'disabled'
            self.Dot.val = [0.0, 0.0, 10.0, 0.0]
            self.owner.par.Mode = 'test'
        self._say(msg)

    def _ac_step(self):
        import autocal_math as am
        ac = self._ac
        now = time.perf_counter()
        el = now - ac['t0']
        orb, pct, col = self._cam()

        def go(state):
            ac['state'], ac['t0'] = state, now

        st = ac['state']
        if st == 'unalign':
            # unaligned cloud (depth frame) only needed to measure depth->color once
            if ac['extr'] is not None:
                orb.par.depthalignmode = 'software'
                go('align')
            elif el > 2.0 and pct.width == orb.width:
                ac['pc_d'] = pct.numpyArray(delayed=False)[..., :3].astype(np.float64)
                orb.par.depthalignmode = 'software'
                go('align')
            elif el > 15:
                self.Stopautocal('Auto calibration: depth camera did not respond.')
        elif st == 'align':
            if el > 2.5 and pct.width == col.width and pct.height == col.height:
                ac['pc_c'] = pct.numpyArray(delayed=False)[..., :3].astype(np.float64)
                if ac['extr'] is None:
                    intr, _ = am.fit_color_intrinsics(ac['pc_c'])
                    best = None
                    for a in (6, 3, 9, 0):
                        r = math.radians(a)
                        R0 = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
                        R, t, rms, frac = am.depth_to_color_icp(ac['pc_d'], ac['pc_c'], intr, R0=R0)
                        if best is None or rms < best[2]:
                            best = (R, t, rms, frac)
                    R, t, rms, frac = best
                    debug('show: depth->color rms %.4f frac %.2f t %s' % (rms, frac, t))
                    if rms > 0.012 or frac < 0.6 or np.linalg.norm(t) > 0.1:
                        self.Stopautocal('Auto calibration: could not line up depth and color '
                                         '(%.1f mm). Is something moving in view?' % (rms * 1000))
                        return
                    ac['extr'] = (R, t)
                    os.makedirs(os.path.dirname(self._extr_path()), exist_ok=True)
                    with open(self._extr_path(), 'w') as f:
                        json.dump({'R': R.tolist(), 't': t.tolist(), 'rms_m': rms,
                                   'time': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)
                # remember the color camera lens (needed by Checkcalib, which avoids the
                # slow align-mode switch)
                try:
                    intr, _ = am.fit_color_intrinsics(ac['pc_c'])
                    with open(self._extr_path()) as f:
                        dd = json.load(f)
                    dd['color_intr'] = list(intr)
                    dd['color_res'] = [int(col.width), int(col.height)]
                    with open(self._extr_path(), 'w') as f:
                        json.dump(dd, f, indent=1)
                except Exception as ex:
                    debug('show: could not store color intrinsics:', ex)
                # aligned cloud is kept; the color stream stalls in software-align mode,
                # so switch back before scanning (the scene must stay still from here)
                orb.par.depthalignmode = 'disabled'
                go('black0')
            elif el > 15:
                self.Stopautocal('Auto calibration: depth-to-color alignment did not start.')
        elif st == 'black0':
            # footprint: where can projector light land at all? (ignores people / TV etc.
            # changing elsewhere in the camera view)
            self.Dot.val = [0.0, 0.0, 10.0, 0.0]
            # streams restart after the align-mode switch: give the color camera time
            if el > 2.5 + ac.get('tries', 0):
                ac['black0'] = self._gray()
                self.Dot.val = [self.W / 2, self.H / 2, 1e5, 1.0]      # full white
                go('white')
        elif st == 'white':
            if el > self.AC_SETTLE + 0.2:
                d = self._gray() - ac['black0']
                # direct projector light is far brighter than light bounced around the room
                mask = (d > 0.1).astype(np.uint8)
                if cv2 is not None:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
                rr, cc = np.nonzero(d > 0.2)
                if len(rr) > 100:
                    # projector rectangle as seen by the camera (camera sits right next to
                    # the projector, so this is close to an affine map) -> predicts dots
                    ac['bbox'] = (np.percentile(rr, 1), np.percentile(rr, 99),
                                  np.percentile(cc, 1), np.percentile(cc, 99))
                    mask = cv2.dilate(mask, np.ones((25, 25), np.uint8))
                ac['mask'] = mask.astype(np.float32)
                cover = float(mask.mean())
                debug('show: projector footprint covers %.0f%% of the color image' % (cover * 100))
                if cover < 0.01 and ac.get('tries', 0) < 3:
                    ac['tries'] = ac.get('tries', 0) + 1
                    self.Dot.val = [0.0, 0.0, 10.0, 0.0]
                    go('black0')
                    return
                if cover < 0.01:
                    self.Stopautocal('Auto calibration: the camera cannot see the projector light. '
                                     'Is the projector on and showing TD, and the room dim?')
                    return
                self.Dot.val = [0.0, 0.0, 10.0, 0.0]
                go('bg')
        elif st == 'bg':
            self.Dot.val = [0.0, 0.0, 10.0, 0.0]
            if el > self.AC_SETTLE + 0.15:
                ac['bg'] = self._gray()
                ac['since_bg'] = 0
                go('show')
        elif st == 'show':
            if ac['i'] >= len(ac['targets']):
                self._ac_finish()
                return
            u, v = ac['targets'][ac['i']]
            self.Dot.val = [u, v, float(self.owner.par.Autocaldot), 1.0]
            self._say('Auto calibration: dot %d/%d (%d found)'
                      % (ac['i'] + 1, len(ac['targets']), len(ac['found'])))
            go('grab')
        elif st == 'grab':
            if el > self.AC_SETTLE:
                diff = (self._gray() - ac['bg']) * ac['mask']
                u, v = ac['targets'][ac['i']]
                if ac.get('bbox'):
                    r0, r1, c0, c1 = ac['bbox']
                    pr, pc_ = r0 + (v / self.H) * (r1 - r0), c0 + (u / self.W) * (c1 - c0)
                    rad = 0.15 * max(r1 - r0, c1 - c0)
                    Hh, Ww = diff.shape
                    win = np.zeros_like(diff)
                    win[max(0, int(pr - rad)):int(pr + rad) + 1, max(0, int(pc_ - rad)):int(pc_ + rad) + 1] = 1
                    diff = diff * win
                spots = am.detect_spots(diff, 3, min_peak=0.06)
                # a dot split by an edge shows up as two neighbouring spots: merge them
                if len(spots) > 1:
                    s0 = spots[0]
                    spots = [s0] + [s for s in spots[1:]
                                    if math.hypot(s[0] - s0[0], s[1] - s0[1]) > 45]
                ok = False
                if spots and spots[0][3] > 0.9 and (len(spots) == 1 or spots[1][2] < 0.5 * spots[0][2]):
                    row, colx, peak, _ = spots[0]
                    Xc = am.lookup_3d(ac['pc_c'], row, colx)
                    if Xc is not None:
                        R, t = ac['extr']
                        Xd = R.T @ (Xc - t)
                        ac['found'].append({'px': [float(u), float(v)], 'world': [float(x) for x in Xd],
                                            'cam_px': [colx, row], 'peak': peak})
                        ok = True
                ac.setdefault('log', []).append((ac['i'], [tuple(round(x, 3) for x in s) for s in spots], ok))
                if not ok:
                    ac['missed'] += 1
                ac['i'] += 1
                ac['since_bg'] += 1
                go('bg' if ac['since_bg'] >= self.AC_BG_EVERY else 'show')

    def _ac_finish(self):
        import autocal_math as am
        ac = self._ac
        pts = ac['found']
        self._ac = None
        self._cam()[0].par.depthalignmode = 'disabled'
        self.Dot.val = [0.0, 0.0, 10.0, 0.0]
        self.owner.par.Mode = 'test'
        if len(pts) < 12:
            self._say('Auto calibration: only %d of %d dots seen. Darken the room, check the '
                      'projector is on and showing the dots.' % (len(pts), len(ac['targets'])))
            return
        # drop outliers: refit until every point is within max(6 px, 3x median)
        dropped = 0
        for _ in range(5):
            try:
                K, R, t, rms, method, errs = self._fit(pts)
            except Exception as e:
                self._say('Auto calibration: solve failed (%s).' % e)
                return
            thr = max(6.0, 3.0 * float(np.median(errs)))
            keep = errs <= thr
            if keep.all() or keep.sum() < 12:
                break
            dropped += int((~keep).sum())
            pts = [p for p, k in zip(pts, keep) if k]
        flat = am.planarity(np.array([p['world'] for p in pts]))
        self.points = pts
        if not self.Solve():
            return
        note = ''
        if flat < 0.03:
            note = '  WARNING: nearly all dots hit one flat surface - put the banjo on a stand ' \
                   'or a box in front of the wall and run it again for a better result.'
        self._say(self.Status.val + '  [auto: %d dots used, %d missed, %d outliers dropped]%s'
                  % (len(pts), ac['missed'], dropped, note))
        self.owner.par.Mode = 'test'

    # called every frame by exec_frame
    def OnFrame(self):
        if getattr(self, '_ck', None):
            try:
                self._ck_step()
            except Exception as e:
                self._ck = None
                self.Dot.val = [0.0, 0.0, 10.0, 0.0]
                self.owner.par.Mode = 'test'
                self._say('Check error: %s' % e)
                raise
            return
        if getattr(self, '_ac', None):
            try:
                self._ac_step()
            except Exception as e:
                self.Stopautocal('Auto calibration error: %s' % e)
                raise
            return
        if not self.calibrating:
            return
        pt = self._banjo_center_3d() if self._banjo_mode() else self._single_dot_3d()
        now = time.perf_counter()
        if pt is None:
            self._buf = []
            self.Target.val = [self.Target.val[0], self.Target.val[1], 0.0, 1.0]
            return
        if self._lock_pt is not None:
            if np.linalg.norm(pt - self._lock_pt) < 0.04:
                return          # still sitting on the last captured spot
            self._lock_pt = None
        # every target is a different spot in the room: refuse anything within
        # 8 cm of an earlier capture (a banjo that hasn't moved to the new crosshair)
        for p in self.points:
            if np.linalg.norm(pt - np.array(p['world'])) < 0.08:
                self._buf = []
                self.Target.val = [self.Target.val[0], self.Target.val[1], 0.0, 1.0]
                return
        self._buf.append((now, pt))
        still = float(self.owner.par.Stilltime)
        self._buf = [(t, p) for t, p in self._buf if now - t <= still + 0.05]
        P = np.array([p for _, p in self._buf])
        spread = float(np.max(np.linalg.norm(P - P.mean(axis=0), axis=1))) if len(P) else 1.0
        if spread > 0.006:
            # moved: restart the stillness window from this sample
            self._buf = [(now, pt)]
            prog = 0.0
        else:
            prog = min(1.0, (now - self._buf[0][0]) / max(still, 0.1))
        self.Target.val = [self.Target.val[0], self.Target.val[1], prog, 1.0]
        if prog >= 1.0 and bool(self.owner.par.Autocapture):
            self.Capture(P.mean(axis=0))

    def _banjo_mode(self):
        try:
            return str(self.owner.par.Calibsource) == 'banjo'
        except Exception:
            return True

    def _banjo_center_3d(self):
        """Head center from the banjo tracker (banjo_info), or None if not tracked."""
        src = parent.Project.op('banjo_info')
        if src is None or src.chan('valid') is None or src['valid'][0] < 0.5:
            return None
        return np.array([float(src[c][0]) for c in ('tx', 'ty', 'tz')])

    def _single_dot_3d(self):
        """3D position of the dot when exactly one blob is visible, else None."""
        blobs = parent.Project.op('info_blobs')
        pc_top = parent.Project.op('orbbec_pointcloud')
        if blobs is None or pc_top is None or blobs.numRows != 2:
            return None
        hdr = [c.val.strip().lower() for c in blobs.row(0)]
        try:
            u = float(blobs[1, hdr.index('u')])
            v = float(blobs[1, hdr.index('v')])
        except Exception:
            return None
        pc = pc_top.numpyArray(delayed=False)
        h, w = pc.shape[:2]
        if u <= 1.0 and v <= 1.0:
            u, v = u * w, v * h
        x, y, r = int(round(u)), int(round(v)), 4
        patch = pc[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1, :3].reshape(-1, 3).astype(np.float64)
        ok = np.all(np.isfinite(patch), axis=1) & (np.abs(patch[:, 2]) > 1e-4)
        if ok.sum() < 3:
            return None
        return np.median(patch[ok], axis=0)

    # ------------------------------------------------------------ solving
    def _fit(self, points):
        """DLT + refine on a list of points. Returns (K, R, t, rms, method, errs)."""
        X = np.array([p['world'] for p in points], dtype=np.float64)
        uv = np.array([p['px'] for p in points], dtype=np.float64)
        img = np.stack([uv[:, 0], self.H - uv[:, 1]], axis=1)      # OpenCV pixels
        n = len(points)
        K, R, t = self._dlt(X, img)
        rms = self._reproj_rms(K, R, t, X, img)
        method = 'DLT'
        if cv2 is not None and n >= 8:
            try:
                K2, R2, t2 = self._refine(K, R, t, X, img)
                rms2 = self._reproj_rms(K2, R2, t2, X, img)
                if rms2 < rms * 1.5:
                    K, R, t, rms, method = K2, R2, t2, rms2, 'DLT+refine'
            except Exception as e:
                debug('show: refine skipped:', e)
        return K, R, t, rms, method, self._reproj_errs(K, R, t, X, img)

    def Solve(self):
        n = len(self.points)
        if n < 6:
            self._say('Need at least 6 points to solve (have %d).' % n)
            return False
        X = np.array([p['world'] for p in self.points], dtype=np.float64)
        uv = np.array([p['px'] for p in self.points], dtype=np.float64)
        img = np.stack([uv[:, 0], self.H - uv[:, 1]], axis=1)      # OpenCV pixels
        try:
            K, R, t = self._dlt(X, img)
        except Exception as e:
            self._say('Solve failed (DLT): %s' % e)
            return False
        rms = self._reproj_rms(K, R, t, X, img)
        method = 'DLT'
        if cv2 is not None and n >= 8:
            # Refine (zero skew, no lens distortion). Own LM because OpenCV's
            # calibrateCamera rejects principal points outside the image, which
            # is normal for projectors (lens shift).
            try:
                K2, R2, t2 = self._refine(K, R, t, X, img)
                rms2 = self._reproj_rms(K2, R2, t2, X, img)
                if rms2 < rms * 1.5:      # accept: removes skew, usually lower error
                    K, R, t, rms, method = K2, R2, t2, rms2, 'DLT+refine'
            except Exception as e:
                debug('show: refine skipped:', e)
        errs = self._reproj_errs(K, R, t, X, img)
        self.calib = {'W': self.W, 'H': self.H, 'K': K.tolist(), 'R': R.tolist(), 't': t.tolist(),
                      'rms_px': rms, 'n': n, 'method': method, 'points': self.points,
                      'errs_px': errs.tolist(), 'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        self.Apply()
        self.Savecalib(quiet=True)
        worst = int(np.argmax(errs))
        quality = 'great' if rms < 4 else ('ok' if rms < 10 else 'poor - redo or Undo bad points')
        self._say('Solved with %d points (%s): error %.1f px (%s). Worst point #%d = %.1f px. Saved.'
                  % (n, method, rms, quality, worst + 1, errs[worst]))
        self.owner.par.Mode = 'test'
        return True

    @staticmethod
    def _dlt(X, img):
        # Hartley normalisation
        def norm_mat(P):
            c = P.mean(axis=0)
            d = np.sqrt(((P - c) ** 2).sum(axis=1)).mean()
            s = math.sqrt(P.shape[1]) / max(d, 1e-9)
            T = np.eye(P.shape[1] + 1)
            T[:-1, :-1] *= s
            T[:-1, -1] = -s * c
            return T
        T3, T2 = norm_mat(X), norm_mat(img)
        Xh = (T3 @ np.c_[X, np.ones(len(X))].T).T
        xh = (T2 @ np.c_[img, np.ones(len(img))].T).T
        A = []
        for (x, y, z, w), (u, v, _) in zip(Xh, xh):
            A.append([x, y, z, w, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u * w])
            A.append([0, 0, 0, 0, x, y, z, w, -v * x, -v * y, -v * z, -v * w])
        _, _, Vt = np.linalg.svd(np.array(A))
        P = np.linalg.inv(T2) @ Vt[-1].reshape(3, 4) @ T3
        # sign: points must be in front of the projector
        if np.mean((P @ np.c_[X, np.ones(len(X))].T)[2]) < 0:
            P = -P
        M = P[:, :3]
        if cv2 is not None:
            Kr, Rq = cv2.RQDecomp3x3(M)[1:3]
        else:
            Kr, Rq = _rq(M)
        Kr, Rq = np.array(Kr, dtype=np.float64), np.array(Rq, dtype=np.float64)
        D = np.diag(np.sign(np.diag(Kr)))
        Kr, Rq = Kr @ D, D @ Rq
        t = np.linalg.solve(Kr, P[:, 3])
        if np.linalg.det(Rq) < 0:
            raise ValueError('mirrored solution - points may be mismatched; redo calibration')
        K = Kr / Kr[2, 2]
        return K, Rq, t

    def _refine(self, K, R, t, X, img, iters=60):
        """Levenberg-Marquardt over fx, fy, cx, cy, rvec, t."""
        rv, _ = cv2.Rodrigues(R)

        def unpack(p):
            Kp = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
            Rp, _ = cv2.Rodrigues(p[4:7].reshape(3, 1))
            return Kp, Rp, p[7:10]

        def resid(p):
            Kp, Rp, tp = unpack(p)
            return (self._project(Kp, Rp, tp, X) - img).ravel()

        p = np.r_[K[0, 0], K[1, 1], K[0, 2], K[1, 2], rv.ravel(), t]
        lam = 1e-3
        r = resid(p)
        cost = r @ r
        for _ in range(iters):
            J = np.empty((len(r), len(p)))
            for j in range(len(p)):
                h = 1e-6 * max(1.0, abs(p[j]))
                dp = np.zeros_like(p)
                dp[j] = h
                J[:, j] = (resid(p + dp) - r) / h
            A = J.T @ J
            g = J.T @ r
            improved = False
            for _ in range(10):
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
                pn = p + step
                rn = resid(pn)
                cn = rn @ rn
                if cn < cost:
                    p, r, cost, lam = pn, rn, cn, lam * 0.3
                    improved = True
                    break
                lam *= 10
            if not improved or np.linalg.norm(step) < 1e-10:
                break
        return unpack(p)

    @staticmethod
    def _project(K, R, t, X):
        c = (R @ X.T).T + t
        p = (K @ c.T).T
        return p[:, :2] / p[:, 2:3]

    def _reproj_errs(self, K, R, t, X, img):
        return np.linalg.norm(self._project(K, R, t, X) - img, axis=1)

    def _reproj_rms(self, K, R, t, X, img):
        e = self._reproj_errs(K, R, t, X, img)
        return float(np.sqrt(np.mean(e ** 2)))

    # ------------------------------------------------------------ matrices
    def ProjectorWorldMatrix(self):
        """Projector camera-to-world 4x4 (TD camera convention: looks down -Z, +Y up)."""
        if self.calib:
            R = np.array(self.calib['R'])
            t = np.array(self.calib['t'])
            S = np.diag([1.0, -1.0, -1.0])
            V = np.eye(4)
            V[:3, :3] = S @ R
            V[:3, 3] = S @ t
            return np.linalg.inv(V)
        # default guess: projector sits on the depth camera, looking the same way
        M = np.eye(4)
        M[0, 0] = -1.0
        M[2, 2] = -1.0     # rotate 180 about Y: camera -Z  ->  world +Z
        return M

    def CenterRayWorld(self):
        """Unit world direction through the middle of the projected image."""
        P = self.ProjectionMatrix()
        # GL camera space, z = -1:  ndc_x = P00*x - P02 = 0  ->  x = P02/P00
        d = np.array([P[0, 2] / P[0, 0], P[1, 2] / P[1, 1], -1.0])
        M = self.ProjectorWorldMatrix()
        w = M[:3, :3] @ d
        return w / np.linalg.norm(w)

    def ProjectionMatrix(self):
        W, H = float(self.W), float(self.H)
        if self.calib:
            K = np.array(self.calib['K'], dtype=np.float64)
            sx, sy = W / self.calib['W'], H / self.calib['H']
            fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, K[0, 2] * sx, K[1, 2] * sy
        else:
            # rough guess for a ~1.2 throw-ratio projector, centered
            fx = fy = 1.2 * W
            cx, cy = W / 2, H / 2
        n, f = NEAR, FAR
        return np.array([
            [2 * fx / W, 0, 1 - 2 * cx / W, 0],
            [0, 2 * fy / H, 2 * cy / H - 1, 0],
            [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
            [0, 0, -1, 0]])

    def Apply(self):
        for name, M in (('calib_view', self.ProjectorWorldMatrix()), ('calib_proj', self.ProjectionMatrix())):
            t = self.owner.op(name)
            t.clear()
            for row in M:
                t.appendRow(['%.9g' % v for v in row])
        pts = self.owner.op('calib_points')
        pts.clear()
        pts.appendRow(['n', 'u_px', 'v_px', 'x', 'y', 'z', 'err_px'])
        src = self.calib['points'] if self.calib else self.points
        errs = self.calib.get('errs_px', []) if self.calib else []
        for i, p in enumerate(src):
            e = '%.2f' % errs[i] if i < len(errs) else ''
            pts.appendRow([i + 1, '%.1f' % p['px'][0], '%.1f' % p['px'][1]] +
                          ['%.4f' % v for v in p['world']] + [e])

    # ------------------------------------------------------------ persistence
    def Savecalib(self, quiet=False):
        if not self.calib:
            if not quiet:
                self._say('Nothing to save - solve a calibration first.')
            return
        path = self._calib_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.calib, f, indent=1)
        if not quiet:
            self._say('Saved calibration to %s' % path)

    def Loadcalib(self, quiet=False):
        path = self._calib_path()
        if not os.path.exists(path):
            if not quiet:
                self._say('No calibration file at %s' % path)
            return
        try:
            with open(path) as f:
                self.calib = json.load(f)
            self.points = list(self.calib.get('points', []))
            self.Apply()
            self._say('Loaded calibration (%d points, %.1f px error, %s).'
                      % (self.calib.get('n', 0), self.calib.get('rms_px', -1), self.calib.get('time', '')))
        except Exception as e:
            self._say('Could not load calibration: %s' % e)


def _rq(M):
    """RQ decomposition fallback (if cv2 is missing)."""
    P = np.flipud(np.eye(3))
    Q, R = np.linalg.qr((P @ M).T)
    return P @ R.T @ P, P @ Q.T
