from maix import camera, display, image, nn, app
import time

# ---------------- SETTINGS ----------------
MODEL_PATH = "/root/models/model_10871.mud"
CONF_TH = 0.45      # candidate threshold: keep LOW so weak/small/side-view drones are seen
IOU_TH = 0.45
THICKNESS = 2

# --- confirmation tiers (applied AFTER detection) ---
HIGH_TH = 0.70      
MID_TH = 0.45      
MID_HITS = 2
LOW_HITS = 4        
WINDOW = 6

# --- tracking ---
MATCH_FACTOR = 1.5  # max centre distance, in multiples of box size
MATCH_MIN_PX = 12   # minimum matching distance in pixels
MAX_MISS = 4        # frames a track survives without a detection
COAST_FRAMES = 2    # confirmed drones stay drawn this many missed frames (anti-flicker)

# --- raw detection cleanup (not position based) ---
DEDUPE_IOU = 0.50
DEDUPE_CONTAIN = 0.85
MIN_BOX_PX = 3
MAX_AREA_FRAC = 0.50
MAX_ASPECT = 6.0

# --- static clutter learning (weak detections that never move) ---
STATIC_REJECT = True
STATIC_FRAMES = 150     # roughly 5 s at 30 FPS
STATIC_MOVE_PX = 6      # total movement allowed to count as "static"
STATIC_MEAN_TH = 0.50   # only if average score is below this
CLUTTER_TTL = 300       # frames a learned clutter zone stays active
CLUTTER_RADIUS = 14

SHOW_CANDIDATES = True  # thin yellow boxes for unconfirmed detections (tuning aid)

# ---------------- LOAD MODEL ----------------

detector = nn.YOLO26(model=MODEL_PATH, dual_buff=True)

cam = camera.Camera(
    detector.input_width(),
    detector.input_height(),
    detector.input_format()
)
disp = display.Display()

labels = detector.labels
fps = 0.0
prev = time.time()
FRAME_W = detector.input_width()
FRAME_H = detector.input_height()
tracks = []         # candidate/confirmed detections across frames
clutter = []        # learned static false-positive zones


def get_label(class_id):
    try:
        return labels[class_id]
    except Exception:
        return "drone"


def _overlap(a, b):
    ix = max(0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    area_a, area_b = a["w"] * a["h"], b["w"] * b["h"]
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    contain = inter / max(min(area_a, area_b), 1e-6)
    return iou, contain


def _prepare(raw):
    """Size sanity filter + duplicate removal on raw detections."""
    dets = []
    for o in raw:
        w, h = float(o.w), float(o.h)
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue
        if w * h > MAX_AREA_FRAC * FRAME_W * FRAME_H:
            continue
        if max(w / h, h / w) > MAX_ASPECT:
            continue
        dets.append(dict(x=float(o.x), y=float(o.y), w=w, h=h,
                         score=float(o.score), class_id=o.class_id))
    dets.sort(key=lambda d: d["score"], reverse=True)
    kept = []
    for d in dets:
        dup = False
        for k in kept:
            iou, contain = _overlap(d, k)
            if iou >= DEDUPE_IOU or contain >= DEDUPE_CONTAIN:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def _centre(d):
    return d["x"] + d["w"] / 2.0, d["y"] + d["h"] / 2.0


def _in_clutter(d):
    cx, cy = _centre(d)
    for z in clutter:
        if ((cx - z["cx"]) ** 2 + (cy - z["cy"]) ** 2) ** 0.5 <= CLUTTER_RADIUS:
            return True
    return False


def _confirmed(t):
    if t["confirmed"]:
        return True
    hits = sum(t["hist"])
    if t["peak"] >= HIGH_TH:
        ok = True
    elif t["peak"] >= MID_TH:
        ok = hits >= MID_HITS
    else:
        ok = hits >= LOW_HITS
    if ok:
        t["confirmed"] = True
    return ok


def update_tracks(raw):
    """Clean detections, match them to tracks, confirm real drones.
    Returns (confirmed_to_draw, unconfirmed_candidates)."""
    dets = _prepare(raw)

    for z in clutter:
        z["ttl"] -= 1
    clutter[:] = [z for z in clutter if z["ttl"] > 0]
    if STATIC_REJECT and clutter:
        dets = [d for d in dets if d["score"] >= HIGH_TH or not _in_clutter(d)]

    # best-first matching of detections to tracks (uses predicted position too)
    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(dets):
            dcx, dcy = _centre(d)
            d_last = ((t["cx"] - dcx) ** 2 + (t["cy"] - dcy) ** 2) ** 0.5
            d_pred = ((t["cx"] + t["vx"] - dcx) ** 2 + (t["cy"] + t["vy"] - dcy) ** 2) ** 0.5
            dist = min(d_last, d_pred)
            limit = max(MATCH_MIN_PX, MATCH_FACTOR * max(t["w"], t["h"], d["w"], d["h"]))
            if dist <= limit:
                pairs.append((dist, ti, di))
    pairs.sort()

    used_t, used_d = set(), set()
    for dist, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        t, d = tracks[ti], dets[di]
        ncx, ncy = _centre(d)
        t["vx"] = 0.5 * t["vx"] + 0.5 * (ncx - t["cx"])
        t["vy"] = 0.5 * t["vy"] + 0.5 * (ncy - t["cy"])
        t.update(x=d["x"], y=d["y"], w=d["w"], h=d["h"],
                 cx=ncx, cy=ncy, class_id=d["class_id"], miss=0)
        t["peak"] = max(t["peak"], d["score"])
        t["score"] = d["score"]
        t["sum"] += d["score"]
        t["n"] += 1
        t["hist"].append(1)
        t["move"] = max(t["move"], ((ncx - t["x0"]) ** 2 + (ncy - t["y0"]) ** 2) ** 0.5)

    for ti, t in enumerate(tracks):
        if ti not in used_t:
            t["miss"] += 1
            t["hist"].append(0)

    for di, d in enumerate(dets):
        if di not in used_d:
            cx, cy = _centre(d)
            tracks.append(dict(x=d["x"], y=d["y"], w=d["w"], h=d["h"],
                               cx=cx, cy=cy, x0=cx, y0=cy, vx=0.0, vy=0.0,
                               class_id=d["class_id"], score=d["score"],
                               peak=d["score"], sum=d["score"], n=1,
                               hist=[1], miss=0, age=0, move=0.0,
                               confirmed=False))

    alive = []
    for t in tracks:
        t["age"] += 1
        t["hist"] = t["hist"][-WINDOW:]
        _confirmed(t)
        if STATIC_REJECT and t["peak"] < HIGH_TH and t["age"] >= STATIC_FRAMES \
                and t["move"] <= STATIC_MOVE_PX and t["n"] > 0 \
                and (t["sum"] / t["n"]) < STATIC_MEAN_TH:
            clutter.append(dict(cx=t["cx"], cy=t["cy"], ttl=CLUTTER_TTL))
            continue            # drop this track
        if t["miss"] <= MAX_MISS:
            alive.append(t)
    tracks[:] = alive

    confirmed = [t for t in tracks if t["confirmed"] and t["miss"] <= COAST_FRAMES]
    candidates = [t for t in tracks if not t["confirmed"] and t["miss"] == 0]
    return confirmed, candidates


# ---------------- MAIN LOOP ----------------
while not app.need_exit():
    img = cam.read()
    raw = detector.detect(img, conf_th=CONF_TH, iou_th=IOU_TH)
    objs, cands = update_tracks(raw)

    # tuning aid: weak, not-yet-confirmed detections
    if SHOW_CANDIDATES:
        for c in cands:
            img.draw_rect(int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"]),
                          color=image.COLOR_YELLOW, thickness=1)

    # draw ALL confirmed drones
    for obj in objs:
        x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["w"]), int(obj["h"])
        img.draw_rect(x, y, w, h,
                      color=image.COLOR_RED, thickness=THICKNESS)
        msg = f"{get_label(obj['class_id'])} {obj['score']:.2f}"
        text_y = y - 12 if y > 14 else y + 2
        img.draw_string(x, text_y, msg, color=image.COLOR_RED)

    # FPS (smoothed) + drone count
    now = time.time()
    instant = 1.0 / max(now - prev, 1e-6)
    fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
    prev = now

    img.draw_string(5, 5, f"FPS:{fps:.1f}  Drones:{len(objs)}",
                    color=image.COLOR_GREEN)

    disp.show(img)