# Banjo projection - show system

Everything new lives in `/project1/show` (shortcut `parent.Show`). Your tracker
(`orbbec1 -> level1 -> blobtrack1`, `banjo_info`, `dot_layout`) is untouched and
feeds it.

## Controls (custom pages on /project1/show)
- **Show**: Mode (Show / Test Pattern / Calibrate / Black), Scene (0 video, 1 fire,
  2 nebula, 3 test; fractions crossfade), Intensity, Hue, Audio (manual),
  Audio Level CHOP + gain (defaults to TDAbleton abletonLevel1), Flame Lean,
  Afterglow trail, Trail Rise, Edge Feather, Video File.
- **Tracking**: Simulate (fake swaying banjo, no hardware), Smoothing, Responsiveness,
  Prediction ms, Hold When Lost, Fade Out, Head Size Trim.
- **Projector**: Output W/H (default 1280x720), Projector Display #, Nudge px,
  Open / Close Projector Window.
- **Calibrate**: Start, Auto-Capture When Still, Hold Still For, Capture / Skip / Undo,
  Stop, Solve, Save / Load / Clear, file path, Status.

## First test session
1. Projector: turn OFF auto-keystone, auto-focus-on-move, picture "enhancement". Mount it
   rigidly next to the Femto Bolt (one bar/plate). Moving the whole rig is fine; if one
   moves relative to the other, recalibrate.
2. Windows display settings: extend to the projector. Set Projector Display # (try 1),
   press Open Projector Window. Mode = Calibrate shows a grey frame at the image edge.
3. AUTO CALIBRATE (recommended, ~1 min, hands-off; 40 dots, expect ~20+ found):
   - Dim the room. Projector on and showing TD's window. Aim the rig where you'll play.
   - Make sure the projector hits more than one flat surface: put the banjo on a stand
     (or a box/chair) in front of the wall. Everyone out of the way, nothing moving.
   - Press Auto Calibrate (Calibrate page). Dots flash one at a time; the Bolt's color
     camera finds each one and the depth camera measures where it landed. It solves,
     saves and switches to Test Pattern. The status line reports dots used / missed.
   - Depth-to-color alignment for this Bolt is stored in
     `calibration/bolt_depth_to_color.json` (measured once; delete it to re-measure).
   Manual fallback (crosshair method):
   - Calibrate With = Banjo head (default). Hold the banjo facing the projector.
   - Press Start Calibration. Put the CENTER of the head on the crosshair and hold still;
     the ring fills and it auto-captures. 5 points NEAR (a step closer than where you
     play), 5 FAR.
   - (Alternative: Calibrate With = Single-dot wand - one dot on a card, banjo out of
     view, 9 + 9 points.)
   - It solves automatically, saves `calibration/projector_calib.json`, and switches to
     Test Pattern. Error under ~5 px is great, over ~10 px: Undo bad points or redo.
4. Pick up the banjo. Test Pattern should sit on the head: rings concentric with the rim,
   cyan arrow pointing at the ceiling, blinking yellow dot at the center.
   - Wrong size: Head Size Trim. Constant offset: Nudge. Lags on fast moves: raise
     Prediction (60 -> 100 ms) and/or Responsiveness. Jittery when still: lower Smoothing.
5. Mode = Show, Scene = 1 for the fire. Try Afterglow 0.85 for a rising trail.

## Notes
- New venue: no recalibration as long as camera + projector stay bolted together and the
  projector's zoom/keystone don't change. Just aim the rig.
- The projector is the weak link (brightness); keep it close and the room dark.
- Code is on disk and syncs live: `code/python/show/*.py`, `code/glsl/show/*.glsl`.
