"""
Live camera demo. Reads frames from a video device (USB capture or webcam),
runs the chip-segmentation model on each frame, draws the predicted seedable
mask on top of the input, and shows the seedable area in mm^2.

Workflow at the microscope:
  1. place a chip under the scope, frame it in the live window
  2. dial the zoom slider to match the microscope's current zoom
  3. read the area number off the corner of the frame
  4. press SPACE to freeze the reading while you write it down
  5. press SPACE again, place the next chip, repeat

Hotkeys:
  q       quit
  s       save current frame + overlay to outputs/predictions/
  d       toggle drop_defects (extra bright-speck removal) on/off
  c       toggle calibration source (lab vs back-derived)
  k       edit the lab px/mm value at runtime (type digits, ENTER to commit,
          ESC to cancel). Useful when re-calibrating the microscope without
          restarting the app.
  x       toggle crosshair lock (only count chip touching the center crosshair)
  SPACE   freeze / unfreeze the reading
  +/-     nudge zoom slider up / down by one quarter step

Mouse: drag the zoom slider in the band below the camera frame (snaps to
0.25 steps from 0.25x to 15.0x). Click the small triangle on the readout
panel header to collapse the panel down to a single area-number tab so
the chip view is unobstructed; click the triangle again to expand.

Center crosshair (default on): the app only reports the area of the chip
touching the crosshair. Aim the crosshair at the chip you care about; if a
hand, debris, or a neighbouring chip is also in frame they will be ignored.
Sliding the dish so a new chip lands on the crosshair resets the smoothing
window so the reading stabilises on the new chip.

Calibration sources:
  --calibration lab     (default) uses 68.6 pixels per mm linear at 1x zoom,
                        the same scale ImageJ used to compute the 24 ground
                        truth area values. This is the "official" lab scale.
  --calibration derived uses the back-derived 1.486e-4 mm^2 per pixel from
                        the labeled masks. See REPORT.txt section 4.4 / 5.1.

  --lab-px-per-mm 68.6  override the linear pixel-per-mm at 1x. Set this if
                        you re-calibrate the microscope.

Tested on Windows with a webcam. On the Raspberry Pi 4 with a Pi Camera you
may need cv2.VideoCapture(0, cv2.CAP_V4L2) and to adjust the resolution to
match what your USB capture device exposes.

Run from code/:
    python scripts/live_demo.py
    python scripts/live_demo.py --device 1 --calibration derived
    python scripts/live_demo.py --lab-px-per-mm 70.5
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from src.eval import load_model, predict_chip_mask
from src.stage2_classical import remove_defects


# the lab's measurement scale used in ImageJ to compute the ground truth areas
DEFAULT_LAB_PX_PER_MM = 68.6

# zoom slider config: float zoom in [ZOOM_MIN, ZOOM_MAX] snapped to ZOOM_STEP
ZOOM_MIN = 0.25
ZOOM_MAX = 15.0
ZOOM_STEP = 0.25
ZOOM_DEFAULT = 1.0

# slider band (drawn underneath the camera frame, full width)
SLIDER_BAND_H = 70
SLIDER_PAD_X = 50

# how many recent area readings to median for the displayed number
SMOOTH_WINDOW = 5

WINDOW_NAME = "microchip area"


def lab_mm2_per_pixel(px_per_mm):
    """Convert a linear pixels-per-mm value to area mm^2-per-pixel."""
    return 1.0 / (float(px_per_mm) ** 2)


def derived_mm2_per_pixel(metrics_path):
    """Read the back-derived mm^2-per-pixel from outputs/metrics.json."""
    metrics = json.loads(Path(metrics_path).read_text())
    val = metrics["calibration"]["mm2_per_pixel_mean"]
    if val is None:
        raise RuntimeError("derived calibration missing in metrics.json. "
                           "run scripts/evaluate.py first.")
    return float(val)


def overlay_mask(frame_bgr, mask, color=(0, 255, 0), alpha=0.4):
    out = frame_bgr.copy()
    color_layer = np.zeros_like(frame_bgr)
    color_layer[mask > 127] = color
    return cv2.addWeighted(out, 1.0, color_layer, alpha, 0)


def draw_contour_trace(frame, mask, color=(0, 255, 255), thickness=2,
                       calculating=False):
    """Trace the predicted chip boundary so the user sees what the model
    thinks the chip is. While the boundary is still settling
    (calculating == True) the line is drawn dashed so the user can tell
    the reading is not yet stable."""
    contours, _ = cv2.findContours((mask > 127).astype(np.uint8),
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame
    if calculating:
        # draw every other small stretch of contour to look dashed
        for c in contours:
            for i in range(0, len(c) - 1, 6):
                p1 = tuple(c[i][0])
                p2 = tuple(c[min(i + 3, len(c) - 1)][0])
                cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    else:
        cv2.drawContours(frame, contours, -1, color, thickness, cv2.LINE_AA)
    return frame


def is_settling(recent_areas, cv_threshold=0.05):
    """Return True while the readout has not stabilised (cv high or buffer
    not yet full). Used to show a 'calculating...' state."""
    if len(recent_areas) < recent_areas.maxlen:
        return True
    arr = np.asarray(recent_areas, dtype=float)
    mean = float(arr.mean())
    if mean <= 1e-6:
        return False
    return float(arr.std()) / mean > cv_threshold


def slider_x_to_zoom(x_px, frame_w):
    """Map a mouse-x in slider coordinates to a snapped zoom value."""
    span = max(1, frame_w - 2 * SLIDER_PAD_X)
    rel = (x_px - SLIDER_PAD_X) / span
    rel = max(0.0, min(1.0, rel))
    raw = ZOOM_MIN + rel * (ZOOM_MAX - ZOOM_MIN)
    snapped = round(raw / ZOOM_STEP) * ZOOM_STEP
    return max(ZOOM_MIN, min(ZOOM_MAX, snapped))


def zoom_to_slider_x(zoom, frame_w):
    """Inverse: zoom value -> pixel x in the slider band."""
    span = max(1, frame_w - 2 * SLIDER_PAD_X)
    rel = (zoom - ZOOM_MIN) / (ZOOM_MAX - ZOOM_MIN)
    rel = max(0.0, min(1.0, rel))
    return SLIDER_PAD_X + int(rel * span)


def _draw_triangle(frame, cx, cy, size, pointing="down", color=(220, 220, 220)):
    """Solid triangle used as the panel collapse / expand toggle."""
    if pointing == "down":
        pts = np.array([[cx - size, cy - size // 2],
                        [cx + size, cy - size // 2],
                        [cx,         cy + size // 2 + 1]], dtype=np.int32)
    else:  # up
        pts = np.array([[cx - size, cy + size // 2],
                        [cx + size, cy + size // 2],
                        [cx,         cy - size // 2 - 1]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], color, lineType=cv2.LINE_AA)


# panel layout - kept as constants so the mouse callback knows the hit zones
PANEL_W_FULL = 380
PANEL_H_FULL = 178
PANEL_W_TAB = 220
PANEL_H_TAB = 36
PANEL_TOGGLE_SIZE = 22  # square hit-zone for the triangle, top-right of panel


def panel_toggle_rect(collapsed):
    """Return (x0, y0, x1, y1) of the click region for the expand/collapse
    triangle. Stays consistent across both states so the user has a stable
    place to aim."""
    if collapsed:
        x1 = PANEL_W_TAB
        y1 = PANEL_H_TAB
    else:
        x1 = PANEL_W_FULL
        y1 = PANEL_H_FULL
    s = PANEL_TOGGLE_SIZE
    return (x1 - s, 0, x1, s)


def draw_edit_banner(frame, buffer):
    """Yellow banner along the bottom of the camera frame while the user
    is typing a new lab px/mm value."""
    h, w = frame.shape[:2]
    banner_h = 40
    y0 = h - banner_h
    cv2.rectangle(frame, (0, y0), (w, h), (0, 200, 230), -1)
    cv2.rectangle(frame, (0, y0), (w, h), (0, 130, 170), 1)
    text = f"new lab px/mm:  {buffer}|   ENTER to commit   ESC to cancel"
    cv2.putText(frame, text, (14, y0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    return frame


def render_slider_band(width, zoom):
    """Build a slider strip (BGR uint8 image) the user drags below the frame."""
    band = np.full((SLIDER_BAND_H, width, 3), 28, dtype=np.uint8)
    bar_y = SLIDER_BAND_H // 2 + 6
    bar_h = 6

    # bar background
    cv2.rectangle(band,
                  (SLIDER_PAD_X, bar_y - bar_h // 2),
                  (width - SLIDER_PAD_X, bar_y + bar_h // 2),
                  (90, 90, 90), -1)

    # ticks at every integer zoom
    for z in range(1, int(ZOOM_MAX) + 1):
        x = zoom_to_slider_x(z, width)
        cv2.line(band, (x, bar_y - 9), (x, bar_y + 9),
                 (170, 170, 170), 1, cv2.LINE_AA)
        label = f"{z}x"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(band, label, (x - tw // 2, bar_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1,
                    cv2.LINE_AA)

    # handle
    handle_x = zoom_to_slider_x(zoom, width)
    cv2.circle(band, (handle_x, bar_y), 11, (60, 240, 60), -1, cv2.LINE_AA)
    cv2.circle(band, (handle_x, bar_y), 11, (255, 255, 255), 1, cv2.LINE_AA)

    # current zoom label, top of band
    label = f"zoom: {zoom:.2f}x   (drag to change, +/- to nudge)"
    cv2.putText(band, label, (SLIDER_PAD_X, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
                cv2.LINE_AA)
    return band


def keep_centered_component(mask, search_radius=20):
    """
    Keep only the connected component of mask that touches the frame center.
    If the exact center pixel is background, search a small square around it
    so the user does not need pixel-perfect aim. Returns:
        kept_mask  - uint8 0/255, only the centered component
        had_chip   - True if a component was found at/near the center
    """
    h, w = mask.shape[:2]
    cy, cx = h // 2, w // 2
    binary = (mask > 127).astype(np.uint8)

    # find which label sits at the center, with a small fallback search radius
    n_labels, labels = cv2.connectedComponents(binary)
    label = labels[cy, cx]
    if label == 0:
        y0 = max(0, cy - search_radius)
        y1 = min(h, cy + search_radius + 1)
        x0 = max(0, cx - search_radius)
        x1 = min(w, cx + search_radius + 1)
        patch = labels[y0:y1, x0:x1]
        nonzero = patch[patch > 0]
        if nonzero.size:
            counts = np.bincount(nonzero.ravel())
            label = int(counts.argmax())

    if label == 0:
        return np.zeros_like(mask), False
    kept = ((labels == label).astype(np.uint8) * 255)
    return kept, True


def draw_crosshair(frame, size=22, color=(0, 220, 220)):
    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2
    cv2.line(frame, (cx - size, cy), (cx - 6, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx + 6, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy - 6), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy + 6), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 4, color, 1, cv2.LINE_AA)


def draw_readout(frame, area_mm2, zoom_x, cal_label, defects_on, fps,
                 frozen, center_lock, has_chip, calculating, collapsed):
    """Top-left readout. When `collapsed` is True, only a small tab with the
    area number and an expand triangle is drawn; otherwise the full panel
    with status lines, hotkey hints, and a collapse triangle is drawn."""
    if center_lock and not has_chip:
        area_text = "  --  mm^2"
        area_color = (140, 140, 140)
    elif calculating and not frozen:
        area_text = "calculating..."
        area_color = (0, 220, 220)
    else:
        area_color = (0, 200, 255) if frozen else (60, 240, 60)
        area_text = f"{area_mm2:6.2f} mm^2"

    if collapsed:
        cv2.rectangle(frame, (0, 0), (PANEL_W_TAB, PANEL_H_TAB),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (PANEL_W_TAB, PANEL_H_TAB),
                      (60, 60, 60), 1)
        cv2.putText(frame, area_text, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, area_color, 2,
                    cv2.LINE_AA)
        # expand triangle on the right edge
        x0, y0, x1, y1 = panel_toggle_rect(True)
        _draw_triangle(frame, (x0 + x1) // 2, (y0 + y1) // 2,
                       size=8, pointing="down")
        return frame

    cv2.rectangle(frame, (0, 0), (PANEL_W_FULL, PANEL_H_FULL),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (PANEL_W_FULL, PANEL_H_FULL),
                  (60, 60, 60), 1)
    cv2.putText(frame, area_text, (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, area_color, 3, cv2.LINE_AA)

    lock_label = "lock:on" if center_lock else "lock:off"

    # status: dynamic context line (prompt / FROZEN / hold steady). Empty
    # string when there's nothing to say so the slot stays reserved and the
    # hotkey line below it never moves.
    status = ""
    status_color = (220, 220, 220)
    if frozen:
        status = "FROZEN - SPACE to resume"
        status_color = (0, 200, 255)
    elif center_lock and not has_chip:
        status = "center the chip on the crosshair"
        status_color = (180, 180, 180)
    elif calculating:
        status = "hold steady..."
        status_color = (0, 220, 220)

    info_lines = [
        (f"zoom: {zoom_x:.2f}x   cal: {cal_label}",                    (220, 220, 220)),
        (f"defects: {'on' if defects_on else 'off'}   {lock_label}   fps: {fps:5.1f}",
                                                                       (220, 220, 220)),
        (status,                                                       status_color),
    ]
    for i, (line, color) in enumerate(info_lines):
        if line:
            cv2.putText(frame, line, (12, 80 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        1, cv2.LINE_AA)

    # always-visible hotkey hints, dimmer so they read as secondary
    cv2.putText(frame,
                "SPACE freeze  q quit  s save  k edit",
                (12, 80 + 3 * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160),
                1, cv2.LINE_AA)

    # collapse triangle on the right edge of the header row
    x0, y0, x1, y1 = panel_toggle_rect(False)
    _draw_triangle(frame, (x0 + x1) // 2, (y0 + y1) // 2,
                   size=8, pointing="up")
    return frame


class StillImageSource:
    """Mimics cv2.VideoCapture but returns the same image on every read.
    Useful for testing the demo on a machine with no webcam."""

    def __init__(self, path):
        self.frame = cv2.imread(path)
        if self.frame is None:
            raise RuntimeError(f"cannot read image: {path}")

    def isOpened(self):
        return self.frame is not None

    def read(self):
        return True, self.frame

    def set(self, *_):
        pass

    def release(self):
        pass


class PlaceholderSource:
    """Synthetic 'no webcam detected' frame so the UI still opens when no
    camera is attached. Lets the user interact with the slider, crosshair,
    calibration toggle, etc., and see the panel rendering."""

    is_placeholder = True

    def __init__(self, width=720, height=576):
        frame = np.full((height, width, 3), 38, dtype=np.uint8)
        # add a static low-amplitude noise so the window does not look dead
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 14, size=(height, width, 3)).astype(np.uint8)
        frame = cv2.add(frame, noise)

        msg = "No webcam detected"
        sub = "demo mode - pass --image <path> to feed a real chip image"
        # measure for centring
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # main message: 70% down so the centre stays clear for the crosshair
        cv2.putText(frame, msg,
                    ((width - mw) // 2, int(height * 0.72)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (60, 60, 235), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, sub,
                    ((width - sw) // 2, int(height * 0.72) + mh + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1,
                    cv2.LINE_AA)
        self.frame = frame

    def isOpened(self):
        return True

    def read(self):
        return True, self.frame

    def set(self, *_):
        pass

    def release(self):
        pass


def open_video_source(args):
    """Resolve --image / --video / --device into a VideoCapture-like object."""
    if args.image:
        print(f"reading still image: {args.image}")
        return StillImageSource(args.image)
    if args.video:
        print(f"reading video file: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open video file: {args.video}")
        return cap

    # webcam: try the windows-specific backends first, then ANY, before failing
    backends = [("DSHOW", cv2.CAP_DSHOW),
                ("MSMF",  cv2.CAP_MSMF),
                ("ANY",   cv2.CAP_ANY)]
    for name, backend in backends:
        cap = cv2.VideoCapture(args.device, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"opened device {args.device} via {name}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                return cap
            cap.release()

    # nothing opened. probe a few indices to leave a hint in the terminal,
    # then drop into a placeholder so the UI still opens for inspection.
    available = []
    for d in range(4):
        c = cv2.VideoCapture(d)
        if c.isOpened():
            available.append(d)
            c.release()

    avail_msg = (f"available device indices: {available}"
                 if available else "no webcam-style devices were detected")
    sample_img = "../Training_dataset_2/C03/C03_Bg2_z1.png"
    print(f"WARNING: cannot open video device {args.device} on any backend.")
    print(f"  {avail_msg}")
    print(f"  opening the demo in placeholder mode so the UI is still")
    print(f"  inspectable. for a real chip image run:")
    print(f"      python scripts/live_demo.py --image {sample_img}")
    return PlaceholderSource(width=args.width, height=args.height)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0, help="cv2 video device index")
    p.add_argument("--image", default=None,
                   help="run on a still image instead of a webcam (looped)")
    p.add_argument("--video", default=None,
                   help="run on a video file instead of a webcam")
    p.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    p.add_argument("--metrics", default="outputs/metrics.json")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--calibration", choices=["lab", "derived"], default="lab",
                   help="lab = ImageJ scale (default), derived = back-derived from masks")
    p.add_argument("--lab-px-per-mm", type=float, default=DEFAULT_LAB_PX_PER_MM,
                   help=f"linear pixels-per-mm at 1x for the lab scale (default {DEFAULT_LAB_PX_PER_MM})")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading model on {device} ...")
    model, _ = load_model(args.ckpt, device=device)

    # build calibration sources. user can toggle between them at runtime with c.
    lab_base = lab_mm2_per_pixel(args.lab_px_per_mm)
    try:
        derived_base = derived_mm2_per_pixel(args.metrics)
    except Exception as exc:
        derived_base = None
        print(f"warning: derived calibration unavailable ({exc})")

    cal_sources = {
        "lab":     {"label": f"lab {args.lab_px_per_mm:.1f}px/mm",
                    "base_mm2_per_pixel": lab_base},
    }
    if derived_base is not None:
        cal_sources["derived"] = {"label": "derived",
                                  "base_mm2_per_pixel": derived_base}

    cur_cal = args.calibration if args.calibration in cal_sources else "lab"
    print(f"using {cur_cal} calibration: {cal_sources[cur_cal]['label']}")

    # video source: webcam by default, or a still image / video file if
    # --image / --video was passed (no-camera fallback for desk testing)
    cap = open_video_source(args)

    # window + custom slider drawn into a band below the camera frame.
    # we do not use cv2.createTrackbar because its label only displays the
    # raw integer position which is unintuitive (user sees "21" for 5.25x).
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    zoom_state = {"value": ZOOM_DEFAULT}
    drag_state = {"dragging": False}
    panel_state = {"collapsed": False}

    def on_mouse(event, x, y, flags, _):
        # the displayed image is [frame on top | slider band on bottom]
        # so the slider band starts at y == args.height
        in_slider = y >= args.height
        if event == cv2.EVENT_LBUTTONDOWN:
            if in_slider:
                drag_state["dragging"] = True
                zoom_state["value"] = slider_x_to_zoom(x, args.width)
                return
            tx0, ty0, tx1, ty1 = panel_toggle_rect(panel_state["collapsed"])
            if tx0 <= x <= tx1 and ty0 <= y <= ty1:
                panel_state["collapsed"] = not panel_state["collapsed"]
        elif event == cv2.EVENT_MOUSEMOVE and drag_state["dragging"]:
            zoom_state["value"] = slider_x_to_zoom(x, args.width)
        elif event == cv2.EVENT_LBUTTONUP:
            drag_state["dragging"] = False

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    drop_defects = False
    frozen = False
    center_lock = True
    edit_mode = False
    edit_buffer = ""
    cur_lab_px_per_mm = args.lab_px_per_mm
    save_dir = Path("outputs/predictions")
    save_dir.mkdir(parents=True, exist_ok=True)

    recent_areas = collections.deque(maxlen=SMOOTH_WINDOW)
    last_annotated = None
    last_frame = None
    last_seedable = None
    last_has_chip = False

    last_t = time.time()
    fps = 0.0

    is_placeholder = getattr(cap, "is_placeholder", False)
    print("running. drag the zoom slider, press q to quit, SPACE to freeze.")
    while True:
        if not frozen:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break

            if is_placeholder:
                # do not run the model on the synthetic 'no webcam' frame; we
                # would just produce nonsense readings on the noise pattern
                seedable = np.zeros(frame.shape[:2], dtype=np.uint8)
                has_chip = False
            else:
                chip_mask = predict_chip_mask(model, frame, device=device)
                seedable, _ = remove_defects(frame, chip_mask,
                                             drop_defects=drop_defects)
                if center_lock:
                    seedable, has_chip = keep_centered_component(seedable)
                else:
                    has_chip = bool((seedable > 127).any())
            # if we just transitioned (no chip -> chip or vice versa),
            # treat it as a new chip and clear the smoothing buffer
            if has_chip != last_has_chip:
                recent_areas.clear()
            last_has_chip = has_chip
            last_frame = frame
            last_seedable = seedable
        else:
            frame = last_frame
            seedable = last_seedable
            has_chip = last_has_chip
            if frame is None:
                # space hit before the first frame loaded
                frozen = False
                continue

        # area calc - depends on current zoom slider value
        zoom_x = zoom_state["value"]
        base = cal_sources[cur_cal]["base_mm2_per_pixel"]
        # area scales as 1/zoom^2: zoom in -> each pixel is smaller in mm
        mm2_per_pixel = base / (zoom_x ** 2)
        seed_pix = int((seedable > 127).sum())
        seed_mm2 = seed_pix * mm2_per_pixel

        if not frozen and (has_chip or not center_lock):
            recent_areas.append(seed_mm2)
        if not frozen and center_lock and not has_chip:
            recent_areas.clear()

        if recent_areas:
            display_mm2 = float(np.median(recent_areas))
        else:
            display_mm2 = seed_mm2
        calculating = (has_chip or not center_lock) and is_settling(recent_areas)

        annotated = overlay_mask(frame, seedable, color=(0, 255, 0), alpha=0.20)
        if has_chip or not center_lock:
            annotated = draw_contour_trace(annotated, seedable,
                                           color=(0, 255, 255),
                                           thickness=2,
                                           calculating=calculating and not frozen)
        if center_lock:
            draw_crosshair(annotated)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - last_t))
        last_t = now
        annotated = draw_readout(annotated, display_mm2, zoom_x,
                                 cal_sources[cur_cal]["label"],
                                 drop_defects, fps, frozen, center_lock,
                                 has_chip, calculating,
                                 panel_state["collapsed"])
        last_annotated = annotated

        if edit_mode:
            annotated = draw_edit_banner(annotated, edit_buffer)

        # composite: camera frame on top, slider band below
        slider_band = render_slider_band(annotated.shape[1], zoom_state["value"])
        display = np.vstack([annotated, slider_band])
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if edit_mode:
            # consume keys for the inline px/mm editor; do not let other
            # shortcuts fire while the user is typing a number
            if key in (13, 10):  # enter
                try:
                    new_val = float(edit_buffer)
                except ValueError:
                    new_val = None
                if new_val is None or new_val <= 0:
                    print(f"invalid px/mm value: {edit_buffer!r}")
                else:
                    cur_lab_px_per_mm = new_val
                    cal_sources["lab"]["base_mm2_per_pixel"] = lab_mm2_per_pixel(new_val)
                    cal_sources["lab"]["label"] = f"lab {new_val:.1f}px/mm"
                    recent_areas.clear()
                    print(f"lab px/mm -> {new_val}")
                edit_mode = False
                edit_buffer = ""
            elif key == 27:  # escape
                edit_mode = False
                edit_buffer = ""
                print("edit cancelled")
            elif key in (8, 127):  # backspace / delete
                edit_buffer = edit_buffer[:-1]
            elif 32 <= key < 127:
                ch = chr(key)
                if ch.isdigit() or (ch == "." and "." not in edit_buffer):
                    edit_buffer += ch
            continue

        if key == ord("q"):
            break
        if key == ord(" "):
            frozen = not frozen
            print(f"frozen -> {frozen}")
        if key == ord("d"):
            drop_defects = not drop_defects
            recent_areas.clear()
            print(f"drop_defects -> {drop_defects}")
        if key == ord("c"):
            options = list(cal_sources.keys())
            cur_cal = options[(options.index(cur_cal) + 1) % len(options)]
            recent_areas.clear()
            print(f"calibration -> {cur_cal}: {cal_sources[cur_cal]['label']}")
        if key == ord("k"):
            edit_mode = True
            edit_buffer = f"{cur_lab_px_per_mm:.1f}"
            print(f"edit lab px/mm: type new value, ENTER to commit, ESC to cancel."
                  f" current = {cur_lab_px_per_mm}")
        if key == ord("x"):
            center_lock = not center_lock
            recent_areas.clear()
            print(f"crosshair lock -> {center_lock}")
        if key in (ord("+"), ord("=")):
            zoom_state["value"] = min(ZOOM_MAX, zoom_state["value"] + ZOOM_STEP)
            recent_areas.clear()
        if key == ord("-"):
            zoom_state["value"] = max(ZOOM_MIN, zoom_state["value"] - ZOOM_STEP)
            recent_areas.clear()
        if key == ord("s"):
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(save_dir / f"frame_{stamp}.png"), frame)
            if last_annotated is not None:
                cv2.imwrite(str(save_dir / f"frame_{stamp}_overlay.png"),
                            last_annotated)
            print(f"saved frame_{stamp}.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
