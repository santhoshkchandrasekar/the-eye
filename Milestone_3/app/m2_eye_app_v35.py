"""
The EYE – Explainable YOLO Explorer: the Milestone 2 detector demo (runs locally; nothing is uploaded anywhere).

The eye is the canvas: the test image sits in the pupil, the iris ring is the score (one segment per object,
coloured by its result), and Detect is a blink. The eyebrow holds the build settings (grid, mix, conditions, seed,
model, threshold); the dock on the left holds upload, random grid, stress test, about, files, snapshot and calm
mode; the review matrix below holds the score, every error with its reason, the inspect view and a threshold sweep.

Uses only the Python standard library for the web server, plus the packages the project already uses.
Run:   python m2_eye_app_v3.py     (opens http://127.0.0.1:7862)      Stop: Ctrl+C
The EYE v3.3: on DETECT the camera moves inside the pupil - the image fills the screen, score left, results right (port 7865).
The EYE v3.4: the COMPOUND view - the other strong models vote on every box of the selected model (and the Milestone 1
classifier on every grid cell). It explains; it never changes the answer or the score (port 7866).
The EYE v3.5: COMPOUND split - with COMPOUND on, the pupil divides into one pupil per model, each ringed by its score ring and
showing that model's own boxes; hover links an object across all pupils; click maximises (port 7867).
v3.2: camera move into the pupil on DETECT (port 7864).
v3.1: the v3 explorer with a calmer first screen.
v3: every improved model (M2 extra/boost, boost2, ...) in the model list; the model chosen on validation
(M2 extra/selected_model.json, written by m2_boost_val.py) is the default, with its own thresholds for grids and for
scenes; images can be dragged into the eye; the Mix panel scrolls when its options do not fit.
It only READS the project; the original EYE (m2_eye_app.py), The EYE v2 and the models are untouched.
Finds the project folder by itself (the nearest folder above this file that contains milestone2_outputs/);
override with M2_PROJECT_DIR. Optional: M2_PORT (default 7862), M2_BOOST_DIR (force the default improved model), M2_HOST=0.0.0.0 to share on a trusted network,
M2_STRESS_MAX (largest stress test, default 300; the hosted copy uses 60).
Keys:  N random grid · D detect · U upload · Esc close / zoom out
"""
import json, os, random, re, tempfile, time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

# ============================================================================================ settings
def find_project():
    """The project folder: M2_PROJECT_DIR if set, else the nearest folder (this file's folder or above, or the current
    folder or above) that contains milestone2_outputs/."""
    if os.environ.get("M2_PROJECT_DIR"): return Path(os.environ["M2_PROJECT_DIR"])
    for start in (Path(__file__).resolve().parent, Path.cwd()):
        for d in (start, *start.parents):
            if (d / "milestone2_outputs").is_dir(): return d
    raise SystemExit("Project folder not found. Put this file inside the project folder (the one containing "
                     "milestone2_outputs/), or set M2_PROJECT_DIR to it.")
PROJECT = find_project()
OUT, DATA = PROJECT / "milestone2_outputs", PROJECT / "Dataset - Updated"
RESULTS, MASTER = OUT / "detect" / "results.json", OUT / "boxes_master.csv"
PERCLASS, OBJLIST = OUT / "detect" / "per_class_ap_test.csv", PROJECT / "object_list.csv"
EXTRA = Path(os.environ["M2_EXTRA_DIR"]) if os.environ.get("M2_EXTRA_DIR") else PROJECT / "M2 extra"      # v2: the extension's outputs
def _read_json(f):
    try: return json.loads(Path(f).read_text())
    except Exception: return {}
SELECTED = _read_json(EXTRA / "selected_model.json")                  # v3: written by m2_boost_val.py (choice made on validation)
BOOST_DIRS = sorted((d for d in EXTRA.glob("boost*") if d.is_dir() and (d / "train.json").exists()),
                    key=lambda d: (len(d.name), d.name))                # boost, boost2, boost3 ...
BOOST = EXTRA / (os.environ.get("M2_BOOST_DIR") or SELECTED.get("dir") or "boost"); BOOST_NAME = "yolo26s_" + BOOST.name   # the default model
SCENES = next((d for d in reversed(BOOST_DIRS) if (d / "tests").is_dir()), BOOST)   # newest test scenes (latest corrected truth)
TILE, IMGSZ, CONF_FLOOR = 224, 1120, 0.01
GRIDS = {"Single photo (1x1)": (1, 1), "2x2 (4 objects)": (2, 2), "3x3 (9 objects)": (3, 3),
         "4x4 (16 objects)": (4, 4), "5x4 (20 objects)": (5, 4), "5x5 (25 objects)": (5, 5)}
DEGRADE = ["None", "Blur", "Low resolution", "Dark", "Heavy JPEG", "All (the report's low-quality test)"]
# Look-alike groups, defined from the Object List names (same as in training)
LOOKALIKE = {"watches": ["OBJ025", "OBJ046", "OBJ065"], "wallets": ["OBJ061", "OBJ067"],
             "power banks": ["OBJ048", "OBJ057"], "rackets": ["OBJ030", "OBJ034", "OBJ035", "OBJ049"],
             "sunglasses / goggles": ["OBJ014", "OBJ045", "OBJ043"], "brush / comb": ["OBJ050", "OBJ059"],
             "bottles": ["OBJ002", "OBJ015", "OBJ063"], "projectors": ["OBJ064", "OBJ070"],
             "plush toys": ["OBJ021", "OBJ024", "OBJ071"], "notebook / keyboard": ["OBJ003", "OBJ012"]}
GROUP_OF = {o: g for g, ids in LOOKALIKE.items() for o in ids}


# ============================================================================================ loading
def load_settings(path=RESULTS):
    r = json.loads(Path(path).read_text())
    best = r["selected_model"]
    a = r.get("any_shape") or {}
    mode = a.get("selected_inference", "standard")
    cfg = (a.get("candidates") or {}).get(mode) or {}
    models = {k: dict(weights=v["weights"], test_map50=v.get("test_map50"), test_map=v.get("test_map"),
                      val_map50=v.get("val_map50"), val_map=v.get("val_map"),
                      test_precision=v.get("test_precision"), test_recall=v.get("test_recall"))
              for k, v in r["models"].items()}
    model_conf, model_conf_scene, model_imgsz, scene_nms = {}, {}, {}, {}
    for d in BOOST_DIRS:                                              # v3: every trained improved model
        name, t, e = "yolo26s_" + d.name, _read_json(d / "train.json"), _read_json(d / "evaluation.json")
        n = (e.get("tests") or {}).get("normal", {}).get("after", {})
        models[name] = dict(weights=t["weights"], test_map50=n.get("mAP50"), test_map=n.get("mAP50_95"), val_map50=None, val_map=None,
                            test_precision=n.get("precision"), test_recall=n.get("recall"))
        model_imgsz[name] = int(t.get("imgsz", 1120)); model_conf[name] = float((e.get("chosen_on_validation") or {}).get("threshold", 0.10))
        model_conf_scene[name] = model_conf[name]
    if SELECTED.get("name") in models:                                # the validated pair of thresholds of the selected model
        model_conf[SELECTED["name"]] = float(SELECTED.get("threshold_grid", model_conf[SELECTED["name"]]))
        model_conf_scene[SELECTED["name"]] = float(SELECTED.get("threshold_scene", model_conf[SELECTED["name"]]))
        if SELECTED.get("scene_nms") is not None: scene_nms[SELECTED["name"]] = float(SELECTED["scene_nms"])   # class-agnostic duplicate removal on scenes
    ev = _read_json(SCENES / "evaluation.json") or None              # newest evaluation: before / after / previous on the same truth
    eval_names = dict(after="yolo26s_" + SCENES.name, previous="yolo26s_boost" if SCENES.name != "boost" else None)
    selection = _read_json(SELECTED["selection"]) if SELECTED.get("selection") else {}
    if ev and ev.get("tests", {}).get("normal"):                     # v3: every improved model scored on the same, newest truth
        nt = ev["tests"]["normal"]
        for col, name in (("after", eval_names["after"]), ("previous", eval_names["previous"])):
            if name in models and nt.get(col):
                models[name].update(test_map50=nt[col].get("mAP50"), test_map=nt[col].get("mAP50_95"),
                                    test_precision=nt[col].get("precision"), test_recall=nt[col].get("recall"))
    for k, v in ((selection.get("validation") or {}).get("models") or {}).items():   # extension validation set (mixed layouts)
        if "yolo26s_" + k in models: models["yolo26s_" + k].update(val_map50=v.get("mAP50"), val_map=v.get("mAP50_95"))
    sel_info = None
    if selection:
        v = selection.get("validation", {}); th = selection.get("thresholds", {})
        sel_info = dict(selected=SELECTED.get("name"), models={"yolo26s_" + k: m for k, m in (v.get("models") or {}).items()},
                        bootstrap=v.get("bootstrap"), images=v.get("images"), adopted=th.get("adopted"),
                        grid=SELECTED.get("threshold_grid"), scene=SELECTED.get("threshold_scene"),
                        tests_two=selection.get("tests_two_thresholds"))
    return dict(best=best, models=models, mode=mode, selection_rule=r.get("selection_rule", ""), model_conf=model_conf, model_imgsz=model_imgsz, boost_eval=ev,
                model_conf_scene=model_conf_scene, eval_names=eval_names, sel_info=sel_info, scene_nms=scene_nms,
                shapes=((a.get("test") or {}).get("by_shape") or {}), task=r.get("test_task_metrics", {}),
                conf=float(cfg.get("conf", r["test_task_metrics"]["conf_threshold"])),
                dup=cfg.get("dup", r["test_task_metrics"].get("duplicate_removal_iou")))

def load_objects(path=OBJLIST):
    """Object ID -> (name, section) from the course Object List; columns found by their content."""
    try:
        t = pd.read_csv(path, dtype=str).fillna("")
        idc = max(t.columns, key=lambda c: t[c].str.strip().str.match(r"OBJ\d+$").mean())
        namec = "OBJECT_NAME" if "OBJECT_NAME" in t.columns else next(c for c in t.columns if "name" in c.lower() and "student" not in c.lower() and c != idc)
        secc = next((c for c in t.columns if c.strip().upper().startswith("SECTION")), None)
        sec = lambda s: re.sub(r"\s+", " ", s.strip().title()) if s.strip() else "Unknown"
        return {i.strip(): (n.strip(), sec(t.at[k, secc]) if secc else "Unknown")
                for k, (i, n) in enumerate(zip(t[idc], t[namec])) if i.strip()}
    except Exception:
        return {}

def load_ap(path=PERCLASS):
    try: return pd.read_csv(path).set_index("object_id").ap50.to_dict()
    except Exception: return {}

def load_photos(master=MASTER, split="test", human_only=True):
    m = pd.read_csv(master)
    ok = m.status.isin(["verified", "fixed"]) if human_only else m.status.isin(["verified", "fixed", "unreviewed_auto"])
    m = m[(m.split == split) & ok & m.x1.notna()].copy()
    m["path"] = [str(DATA / f"images_{c}" / f) for c, f in zip(m.object_id, m.fname)]
    return m[m.path.map(os.path.exists)].reset_index(drop=True)


# ============================================================================================ geometry / post-processing
def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0: return 0.0
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / u if u > 0 else 0.0

def cell_of(b): return (int(((b[0] + b[2]) / 2) // TILE), int(((b[1] + b[3]) / 2) // TILE))

def postprocess(preds, dup, W, H, use_cell=True):
    """The evaluated post-processing: one box per 224-px cell (grids only), or overlap removal."""
    if dup == "cell":
        if not (use_cell and W % TILE == 0 and H % TILE == 0): return preds
        best = {}
        for p in sorted(preds, key=lambda q: -q[1]): best.setdefault(cell_of(p[2]), p)
        return list(best.values())
    if dup is None: return preds
    keep = []
    for p in sorted(preds, key=lambda q: -q[1]):
        if all(iou(p[2], k[2]) < float(dup) for k in keep): keep.append(p)
    return keep

def select(raw, conf, dup, W, H, use_cell):
    return postprocess([p for p in raw if p[1] >= conf], dup, W, H, use_cell)

def to_canvas(img, size=IMGSZ):
    W, H = img.size
    if W > size or H > size: return img, False
    c = Image.new("RGB", (size, size), (128, 128, 128)); c.paste(img, (0, 0)); return c, True

def run_model(model, img, mode, conf=CONF_FLOOR, imgsz=IMGSZ):
    """All predictions down to CONF_FLOOR (so missed objects can be explained), in input-image pixels, + ms.
    v2: a model trained at imgsz > 1120 gets the image scaled by imgsz/1120 on an imgsz canvas (as in training)."""
    W0, H0 = img.size; s = imgsz / IMGSZ
    if s != 1: img = img.resize((max(1, round(W0 * s)), max(1, round(H0 * s))), Image.Resampling.BICUBIC)
    W, H = img.size
    inp, padded = to_canvas(img, imgsz) if (mode == "native" or s != 1) else (img, False)
    t0 = time.perf_counter()
    r = model.predict(inp, imgsz=imgsz, conf=conf, max_det=100, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    out = []
    for c, p, b in zip(r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(float, b)
        if padded:
            if (x1 + x2) / 2 > W or (y1 + y2) / 2 > H: continue
            x2, y2 = min(x2, W), min(y2, H)
        out.append((int(c), float(p), (x1 / s, y1 / s, x2 / s, y2 / s)))           # back to the image's own pixels
    return out, ms


# ============================================================================================ degradation
def degrade(img, kind):
    if kind in (None, "None"): return img
    W, H = img.size
    def lowres(im, f=0.25): return im.resize((max(8, int(W * f)), max(8, int(H * f))), Image.Resampling.BILINEAR).resize((W, H), Image.Resampling.BILINEAR)
    def jpeg(im, q=20):
        import io; b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0); return Image.open(b).convert("RGB")
    if kind == "Blur": return img.filter(ImageFilter.GaussianBlur(2.0))
    if kind == "Low resolution": return lowres(img)
    if kind == "Dark": return ImageEnhance.Brightness(img).enhance(0.45)
    if kind == "Heavy JPEG": return jpeg(img, 12)
    return jpeg(ImageEnhance.Brightness(lowres(img)).enhance(0.6), 20)      # the report's low-quality test


# ============================================================================================ explaining every outcome
class Info:
    """Lookup of Object ID, name, section, class AP and look-alike group, for a model's class indices."""
    def __init__(self, ids, objects, ap):
        self.ids, self.objects, self.ap = ids, objects, ap
        self.idx = {v: k for k, v in ids.items()}
    def oid(self, c): return self.ids[c] if c is not None else ""
    def name(self, c): return self.objects.get(self.oid(c), ("", ""))[0] if c is not None else ""
    def section(self, c): return self.objects.get(self.oid(c), ("", "Unknown"))[1] if c is not None else ""
    def apv(self, c):
        v = self.ap.get(self.oid(c)); return round(float(v), 3) if v is not None else None
    def group(self, c): return GROUP_OF.get(self.oid(c), "")
    def label(self, c): return f"{self.oid(c)} ({self.name(c)})" if self.name(c) else self.oid(c)

def explain(gts, preds, raw, conf, info):
    """Every true object and every kept box gets one row: correct / box off / wrong ID / missed / extra + reason."""
    rows, used_g, used_p = [], set(), set()
    def row(result, reason, c, gc, p, v, k_gt=None, k_pred=None):
        rows.append(dict(result=result, reason=reason, predicted=info.oid(c), predicted_name=info.name(c),
                         true=info.oid(gc), true_name=info.name(gc),
                         section=info.section(gc if gc is not None else c),
                         class_AP=info.apv(gc if gc is not None else c),
                         confidence=round(p, 2) if p is not None else None, IoU=round(v, 2) if v is not None else None,
                         _gt=k_gt, _pred=k_pred))
    for i in sorted(range(len(preds)), key=lambda k: -preds[k][1]):          # 1. confident matches
        c, p, b = preds[i]; best, bj = 0.0, None
        for j, (gc, gb) in enumerate(gts):
            if j not in used_g and iou(b, gb) > best: best, bj = iou(b, gb), j
        if bj is not None and best >= 0.5:
            used_g.add(bj); used_p.add(i); gc = gts[bj][0]
            if gc == c: row("correct", "", c, gc, p, best, bj, i)
            else:
                why = (f"look-alike ({info.group(gc)}): called {info.label(c)}" if info.group(c) and info.group(c) == info.group(gc)
                       else f"confused with {info.label(c)}")
                if info.section(c) != info.section(gc): why += f"; different section ({info.section(c)})"
                row("wrong ID", why, c, gc, p, best, bj, i)
    for j, (gc, gb) in enumerate(gts):                                        # 2. right ID, box placed off
        if j in used_g: continue
        cand = [(iou(preds[i][2], gb), i) for i in range(len(preds)) if i not in used_p and preds[i][0] == gc]
        cand = [x for x in cand if x[0] >= 0.1]
        if cand:
            v, i = max(cand); used_g.add(j); used_p.add(i)
            row("box off", f"right ID, box placed off (overlap {v:.2f}, needs 0.50)", gc, gc, preds[i][1], v, j, i)
    for j, (gc, gb) in enumerate(gts):                                        # 3. missed: why?
        if j in used_g: continue
        low = [(q[1], q[0]) for q in raw if q[1] < conf and iou(q[2], gb) >= 0.5]
        same = [x for x in low if x[1] == gc]
        area = (gb[2] - gb[0]) * (gb[3] - gb[1]) / (TILE * TILE)
        if same:
            c = max(same)[0]; d = 2                                                  # never "0.10, below 0.10": add digits until they differ
            while d < 5 and f"{c:.{d}f}" == f"{conf:.{d}f}": d += 1
            why = f"seen at confidence {c:.{d}f}, below the threshold {conf:.{d}f}"
        elif low: x = max(low); why = f"seen weakly as {info.label(x[1])} ({x[0]:.2f}), below the threshold"
        elif area < 0.03: why = f"not found: very small object ({100 * area:.1f}% of its photo)"
        else: why = "not found" + (f" (look-alike group: {info.group(gc)})" if info.group(gc) else "")
        row("missed", why, None, gc, None, None, j, None)
    for i, (c, p, b) in enumerate(preds):                                     # 4. extra boxes: why?
        if i in used_p: continue
        near = [k for k in used_p if iou(preds[k][2], b) >= 0.3 or cell_of(preds[k][2]) == cell_of(b)]
        part = [g for g in gts if iou(g[1], b) > 0.05]
        if near: why = f"second box near {info.oid(preds[near[0]][0])}"
        elif part: why = f"box covers only part of {info.oid(part[0][0])}"
        else: why = "box on the background"
        row("extra", why, c, None, p, None, None, i)
    return rows

def tally(rows, n_gt):
    c = Counter(r["result"] for r in rows)
    return dict(n=n_gt, correct=c["correct"], box_off=c["box off"], wrong=c["wrong ID"], missed=c["missed"],
                extra=c["extra"], named=c["correct"] + c["box off"],
                fully=int(c["correct"] == n_gt and c["extra"] == 0 and c["box off"] == 0 and c["wrong ID"] == 0))

def bootstrap_ci(v, n=2000, seed=42):
    v = np.asarray(v, float)
    if not len(v): return (0.0, 0.0)
    rng = np.random.default_rng(seed); m = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ============================================================================================ grids
_cache = {}
def load_224(path):
    if path not in _cache:
        with Image.open(path) as im:
            _cache[path] = ImageOps.exif_transpose(im).convert("RGB").resize((TILE, TILE), Image.Resampling.BICUBIC)
    return _cache[path]

def pick_classes(pool_classes, n, mode, rng):
    """Distinct Object IDs for one grid: random, or whole look-alike groups first."""
    pool = sorted(pool_classes)
    if mode == "Look-alike challenge":
        groups = [[o for o in ids if o in pool] for ids in LOOKALIKE.values()]
        groups = [g for g in groups if len(g) >= 2]
        cand = sorted({o for g in groups for o in g})
        if len(cand) < n:
            raise ValueError(f"Only {len(cand)} look-alike objects available in this selection; choose a grid with at most {len(cand)} objects.")
        rng.shuffle(groups); out = []
        for g in groups:
            for o in g:
                if len(out) < n and o not in out: out.append(o)
        return out
    if len(pool) < n:
        raise ValueError(f"Only {len(pool)} different objects available in this selection; choose a grid with at most {len(pool)} objects.")
    return rng.sample(pool, n)

def make_grid(photos, cols, rows, sections=None, mode="Random", seed=None, objects=None):
    rng = random.Random(seed)
    p = photos
    if sections and objects:
        p = p[p.object_id.map(lambda o: objects.get(o, ("", "Unknown"))[1] in sections)]
    classes = pick_classes(set(p.object_id), cols * rows, mode, rng)
    canvas = Image.new("RGB", (cols * TILE, rows * TILE)); gts = []
    for k, oid in enumerate(classes):
        r = p[p.object_id == oid].sample(1, random_state=rng.randint(0, 10 ** 6)).iloc[0]
        x0, y0 = (k % cols) * TILE, (k // cols) * TILE
        canvas.paste(load_224(r.path), (x0, y0))
        gts.append((oid, (x0 + r.x1, y0 + r.y1, x0 + r.x2, y0 + r.y2)))
    return canvas, gts


# ============================================================================================ drawing
def font(size):
    for f in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try: return ImageFont.truetype(f, size)
        except Exception: pass
    return ImageFont.load_default()

COL = {"correct": (230, 30, 30), "box off": (255, 150, 0), "wrong ID": (170, 0, 220), "extra": (120, 120, 120)}
def draw(img, preds, info, gts=None, rows=None):
    im = img.convert("RGB").copy(); d = ImageDraw.Draw(im); f = font(max(11, im.width // 70))
    status = {r["_pred"]: r["result"] for r in (rows or []) if r["_pred"] is not None}
    for gc, gb in (gts or []): d.rectangle(gb, outline=(0, 200, 0), width=3)
    for i, (c, p, b) in enumerate(preds):
        col = COL.get(status.get(i, "correct"), (230, 30, 30)) if rows else (230, 30, 30)
        d.rectangle(b, outline=col, width=3)
        lab = f"{info.oid(c)} {p:.2f}"; tw = d.textlength(lab, font=f)
        d.rectangle([b[0], b[1], b[0] + tw + 6, b[1] + f.size + 4], fill=col); d.text((b[0] + 3, b[1] + 1), lab, fill="white", font=f)
    return im

_refs = {}
def reference(oid, train, size=150):
    """A training photo of the class, cropped to its box: 'what this class looks like'. Large sizes use the original photo."""
    key = (oid, size)
    if key in _refs: return _refs[key]
    t = train[train.object_id == oid].sort_values("fname")
    if not len(t): _refs[key] = None; return None
    r = t.iloc[0]; m = 0.1 * max(r.x2 - r.x1, r.y2 - r.y1)
    if size <= 150:
        im = load_224(r.path); c = im.crop((max(0, r.x1 - m), max(0, r.y1 - m), min(TILE, r.x2 + m), min(TILE, r.y2 + m)))
    else:                                                     # full resolution: boxes are in 224-px coordinates, scale them
        with Image.open(r.path) as im0: im = ImageOps.exif_transpose(im0).convert("RGB")
        sx, sy = im.width / TILE, im.height / TILE
        c = im.crop((max(0, (r.x1 - m) * sx), max(0, (r.y1 - m) * sy), min(im.width, (r.x2 + m) * sx), min(im.height, (r.y2 + m) * sy)))
    _refs[key] = ImageOps.contain(c, (size, size)); return _refs[key]

def one_tile(img, preds, info, train, gts, r, size=150):
    """The object in this image | a training photo of the predicted class (| of the true class), panels of `size` px."""
    pi, gi = r.get("_pred"), r.get("_gt")
    box = gts[gi][1] if gi is not None else preds[pi][2]
    m = 0.25 * max(box[2] - box[0], box[3] - box[1], 20)
    x1 = min(max(0, box[0] - m), img.width - 2); y1 = min(max(0, box[1] - m), img.height - 2)     # clamp: never crash
    x2 = max(min(img.width, box[2] + m), x1 + 2); y2 = max(min(img.height, box[3] + m), y1 + 2)
    crop = img.crop((x1, y1, x2, y2)); crop = ImageOps.contain(crop, (size, size), Image.Resampling.LANCZOS)
    panels = [("in this image", crop)]
    if pi is not None: panels.append((f"model: {info.oid(preds[pi][0])}", reference(info.oid(preds[pi][0]), train, size)))
    if gi is not None and (pi is None or preds[pi][0] != gts[gi][0]):            # gts hold class indices
        panels.append((f"true: {info.oid(gts[gi][0])}", reference(info.oid(gts[gi][0]), train, size)))
    cw, fs = size + 10, max(12, size // 12)
    tile = Image.new("RGB", (cw * len(panels), size + 14 + fs + 8), "white"); dd = ImageDraw.Draw(tile); f = font(fs)
    for k, (cap, im) in enumerate(panels):
        if im is not None: tile.paste(im, (cw * k + (cw - im.width) // 2, 4 + (size - im.height) // 2))
        dd.text((cw * k + 5, size + 10), cap, fill="black", font=f)
    title = (r["result"] + (f": {r['reason']}" if r.get("reason") else "")) if r.get("result") and r["result"] != "detected" else \
            f"{info.label(preds[pi][0])} {preds[pi][1]:.2f} | {info.section(preds[pi][0])} | class AP {info.apv(preds[pi][0])}"
    return tile, title

def gallery(img, preds, info, train, gts=None, rows=None, limit=30):
    """One tile per object, errors first."""
    entries = rows if rows is not None else [dict(result="", reason="", _pred=i, _gt=None) for i in range(len(preds))]
    return [one_tile(img, preds, info, train, gts, r) for r in sorted(entries, key=lambda r: r["result"] == "correct")[:limit]]

def highlight(img_drawn, box):
    """Dim everything except one region (the object's 224-px cell in grids, else the box) and frame it."""
    base = img_drawn.convert("RGBA"); W, H = base.size
    if W % TILE == 0 and H % TILE == 0:
        cx, cy = cell_of(box); reg = (cx * TILE, cy * TILE, cx * TILE + TILE - 1, cy * TILE + TILE - 1)
    else:
        m = 0.15 * max(box[2] - box[0], box[3] - box[1], 20)
        reg = (max(0, box[0] - m), max(0, box[1] - m), min(W - 1, box[2] + m), min(H - 1, box[3] + m))
    shade = Image.new("RGBA", base.size, (0, 0, 0, 150)); mask = Image.new("L", base.size, 255)
    ImageDraw.Draw(mask).rectangle(reg, fill=0)
    out = Image.alpha_composite(base, Image.composite(shade, Image.new("RGBA", base.size, (0, 0, 0, 0)), mask)).convert("RGB")
    ImageDraw.Draw(out).rectangle(reg, outline=(255, 215, 0), width=max(4, W // 180))
    return out

def row_box(rows, k, preds, gts):
    """The box behind table row k: the true box if the row has one, else the predicted box."""
    if rows is None: return preds[k][2]
    r = rows[k]
    return gts[r["_gt"]][1] if r.get("_gt") is not None else preds[r["_pred"]][2]

def det_table(preds, info):
    return pd.DataFrame([dict(object_id=info.oid(c), name=info.name(c), section=info.section(c), class_AP=info.apv(c),
                              look_alike_group=info.group(c), confidence=round(p, 2),
                              x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                         for c, p, b in sorted(preds, key=lambda t: (int(t[2][1] // TILE), t[2][0]))])

def save_downloads(img_drawn, table, stem="detections"):
    d = Path(tempfile.mkdtemp(prefix="m2demo_"))
    pi, pc, pj = d / f"{stem}.png", d / f"{stem}.csv", d / f"{stem}.json"
    img_drawn.save(pi); table.to_csv(pc, index=False); pj.write_text(table.to_json(orient="records", indent=2))
    return [str(pi), str(pc), str(pj)]

def sweep_plot(gts, raw, dup, W, H, use_cell, conf_now, info):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ts = np.round(np.arange(0.05, 0.91, 0.05), 2); series = {k: [] for k in ("correct", "missed", "extra", "wrong ID")}
    for t in ts:
        tl = tally(explain(gts, select(raw, t, dup, W, H, use_cell), raw, t, info), len(gts))
        series["correct"].append(tl["correct"]); series["missed"].append(tl["missed"] + tl["box_off"])
        series["extra"].append(tl["extra"] + tl["box_off"]); series["wrong ID"].append(tl["wrong"])
    fig, ax = plt.subplots(figsize=(6.5, 3.2), dpi=110)
    for k, st in zip(series, ("-o", "-s", "-^", "-x")): ax.plot(ts, series[k], st, markersize=3, label=k)
    ax.axvline(conf_now, color="black", linestyle="--", linewidth=1, label=f"current {conf_now:.2f}")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("objects"); ax.set_title("This grid at every threshold")
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3); fig.tight_layout(); return fig



# ============================================================================================ server (Python standard library only)
import base64, io, threading, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("M2_PORT", "7867"))
STRESS_MAX = max(10, int(os.environ.get("M2_STRESS_MAX", "300")))   # hosted copies set a lower cap (CPU, shared by everyone)
FAKE = os.environ.get("M2_FAKE_MODEL") == "1"          # tests only: a stand-in model, no GPU needed
# v3.4 COMPOUND view: the models that vote. The selected model answers; the others only vote on its boxes.
PANEL = ["yolo26s_boost", "yolo26s_boost2", "yolo26s_ft2", "yolov8s_ft"]
PANEL_LABEL = {"yolo26s_boost": "boost (1,280 px)", "yolo26s_boost2": "boost2 (1,600 px)", "yolo26s_ft2": "Milestone 2 model",
               "yolov8s_ft": "YOLOv8-s (Run 3)", "CLS": "Milestone 1 classifier"}
def _cls_path():
    """The Milestone 1 ConvNeXt-Tiny classifier (local v2, model E); optional. M2_CLS_WEIGHTS overrides the search."""
    e = os.environ.get("M2_CLS_WEIGHTS"); rel = Path("milestone1_outputs_v2") / "experiments" / "E_convnext_tiny_finetune" / "best.pt"
    cands = ([Path(e)] if e else []) + [PROJECT / rel, PROJECT.parent / "Milestone 1" / rel, PROJECT.parent.parent / "Milestone 1" / rel]
    return next((p for p in cands if p.exists()), None)
CLS_PATH = _cls_path()

def b64img(img, q=90):
    b = io.BytesIO(); img.convert("RGB").save(b, "JPEG", quality=q); return base64.b64encode(b.getvalue()).decode()

def num(v):
    if v is None: return None
    if isinstance(v, (np.floating, float)): return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)): return int(v)
    return v

class FakeModel:
    """Tests only: 'detects' the true objects with one miss, one box off, one wrong ID and one extra box."""
    def __init__(self, names, var=0): self.names = names; self.truth = []; self.var = var     # v3.4: var > 0 = a voter that differs a little
    def predict(self, inp, imgsz, conf, max_det, verbose):
        c, p, b = [], [], []
        for k, (ci, box) in enumerate(self.truth):
            if self.var:                                                                      # v3.4 tests: a voter sees the scene a little differently
                if (k + self.var) % 5 == 0: continue
                c.append((ci + 3) % len(self.names) if (k + 2 * self.var) % 7 == 0 else ci); p.append(0.9 - 0.01 * k); b.append(list(box)); continue
            if k == len(self.truth) - 1 and len(self.truth) > 1: continue                     # missed
            box = list(box)
            if k == 1: box = [box[0], box[1] + 0.45 * (box[3] - box[1]), box[2], box[3] + 0.45 * (box[3] - box[1])]   # box off
            ci2 = (ci + 1) % len(self.names) if k == 2 else ci                                  # wrong ID
            c.append(ci2); p.append(0.93 - 0.02 * k); b.append(box)
        if self.truth: c.append(self.truth[0][0]); p.append(0.31); b.append([5, 5, 40, 40])    # extra (same cell as #0)
        class _B:
            def __init__(s, a): s.a = np.array(a, float)
            def cpu(s): return s
            def numpy(s): return s.a
        class _R: pass
        r = _R(); r.boxes = type("x", (), dict(cls=_B(c), conf=_B(p), xyxy=_B(b if b else np.zeros((0, 4)))))()
        return [r]

class App:
    def __init__(self):
        self.S = load_settings(); self.OBJ = load_objects(); self.AP = load_ap()
        self.test = load_photos(split="test"); self.train = load_photos(split="train", human_only=False)
        self.sections = sorted({v[1] for v in self.OBJ.values()}) or ["Unknown"]
        self.models = {}; self.lock = threading.Lock(); self.warm = dict(state="idle", error=None); self.warmed = set()
        self.grids, self.dets, self.jobs = {}, {}, {}
        self.available = [m for m, v in self.S["models"].items() if FAKE or Path(v["weights"]).exists()] or [self.S["best"]]
        self.cls = None                                                     # v3.4: the classifier, loaded on first use

    def model(self, name):
        if name not in self.S["models"]: raise ValueError(f"unknown model {name}")
        if name not in self.models:
            if FAKE:
                ids = sorted(set(self.test.object_id) | set(self.OBJ))
                var = 0 if name == self.S["best"] else 1 + (PANEL.index(name) if name in PANEL else 3)
                self.models[name] = FakeModel({k: o for k, o in enumerate(ids)}, var)
            else:
                from ultralytics import YOLO
                self.models[name] = YOLO(self.S["models"][name]["weights"])
        m = self.models[name]; return m, Info(m.names, self.OBJ, self.AP)

    def warmup(self):
        """Load the selected model onto the GPU and run one dummy detection, so the first real one is fast."""
        self.warm["state"] = "loading"
        try:
            with self.lock:
                model, _ = self.model(self.S["best"])
                if FAKE: model.truth = []
                run_model(model, Image.new("RGB", (TILE * 3, TILE * 3), (90, 90, 90)), self.S["mode"])
                self.warmed.add(self.S["best"])
            self.warm["state"] = "ready"
        except Exception as e:
            self.warm.update(state="error", error=str(e))

    def status(self, _=None): return dict(warm=self.warm["state"], error=self.warm["error"], warmed=sorted(self.warmed))

    @staticmethod
    def _keep(d, n=40):
        while len(d) > n: d.pop(next(iter(d)))

    # ------------------------------------------------------------------ endpoints
    def config(self, _=None):
        S = self.S
        return dict(best=S["best"], mode=S["mode"], dup=S["dup"], conf=S["conf"], models=self.available,
                    model_info={k: {m: num(v.get(m)) for m in ("val_map50", "val_map", "test_map50", "test_map", "test_precision", "test_recall")}
                                for k, v in S["models"].items()},
                    selection_rule=S.get("selection_rule", ""),
                    shapes={k: {m: num(v.get(m)) for m in ("images", "fully_correct", "object_recall", "precision")} for k, v in S.get("shapes", {}).items()},
                    sections=self.sections, grids=list(GRIDS), degrade=DEGRADE, groups=LOOKALIKE,
                    n_test_photos=int(len(self.test)), n_objects=int(self.test.object_id.nunique()), **self.dataset_counts(),
                    model_conf=S["model_conf"], boost=BOOST_NAME if BOOST_NAME in self.available else None, boost_eval=S["boost_eval"],
                    layouts=["grid", "rotated"] + [n for n, f in (("scattered", "scattered"), ("cut-outs", "cutout")) if (SCENES / "tests" / f / "truth.json").exists()],
                    model_conf_scene=S["model_conf_scene"], eval_names=S["eval_names"], sel_info=S["sel_info"], scene_nms=S["scene_nms"], stress_max=STRESS_MAX,
                    panel=[dict(model=m, label=PANEL_LABEL.get(m, m)) for m in PANEL if m in self.available],
                    panel_cls=bool(FAKE or CLS_PATH))

    def dataset_counts(self):
        """Photos in the cleaned dataset (Milestone 1 file if present, else the box file) and photos with a reliable box."""
        try:
            m = pd.read_csv(MASTER)
            boxed = int((m.x1.notna() & m.status.isin(["verified", "fixed", "unreviewed_auto"])).sum())
            clean = PROJECT / "milestone1_outputs_v2" / "clean_dataset.csv"
            total = int(len(pd.read_csv(clean))) if clean.exists() else int(len(m))
            out = dict(n_photos=total, n_boxed=boxed, n_objects_all=int(m.object_id.nunique()), boxes_checked=False)
            sm = _read_json(EXTRA / "label_check" / "summary.json")          # v3: after the full human check of every box
            if sm.get("human_checked_boxes"):
                out.update(n_boxed=int(sm["human_checked_boxes"]), boxes_checked=bool(sm.get("every_usable_box_checked", True)),
                           n_excluded_person=sm.get("excluded_by_person"), n_dups_removed=sm.get("near_duplicates_removed"))
            return out
        except Exception:
            return dict(n_photos=None, n_boxed=None, n_objects_all=None)

    def scenes(self, name):
        """The evaluation's own test scenes (unseen test photos on unseen backgrounds), with their true boxes."""
        f = SCENES / "tests" / name / "truth.json"
        return json.loads(f.read_text()) if f.exists() else []

    def grid(self, q):
        layout = q.get("layout") or "grid"
        if layout in ("scattered", "cut-outs"):
            sc = self.scenes("scattered" if layout == "scattered" else "cutout")
            if not sc: raise ValueError(f"No {layout} scenes yet: run  python m2_boost.py build  in M2 extra.")
            seed = q.get("seed"); seed = int(seed) if str(seed or "").strip().isdigit() else random.randint(0, 10 ** 6)
            t = sc[seed % len(sc)]; img = Image.open(SCENES / "tests" / ("scattered" if layout == "scattered" else "cutout") / "images" / t["file"]).convert("RGB")
            gts_o = [(o, tuple(b)) for o, b in t["boxes"]]; gid = uuid.uuid4().hex[:10]
            self.grids[gid] = dict(img=img, gts_o=gts_o, cols=0, rows=0, seed=seed, kind="scene", size=layout, mix=layout, cells=False); self._keep(self.grids)
            return dict(id=gid, seed=seed, image=b64img(img), w=img.width, h=img.height, cols=0, rows=0, kind="scene", size=layout, mix=layout)
        size = q.get("size") or "3x3 (9 objects)"
        if size not in GRIDS: raise ValueError("unknown grid size")
        cols, rows = GRIDS[size]; mix = q.get("mix") or "Random"
        seed = q.get("seed"); seed = int(seed) if str(seed or "").strip().isdigit() else random.randint(0, 10 ** 6)
        img, gts_o = make_grid(self.test, cols, rows, q.get("sections") or self.sections, mix, seed, self.OBJ)
        if layout == "rotated":                                   # the same grid, turned 90/180/270 degrees: still cells, exact boxes
            k = 1 + seed % 3; W, H = img.size; img = img.rotate(90 * k, expand=True)
            def rb(b):
                x1, y1, x2, y2 = b; w, h = W, H
                for _ in range(k): x1, y1, x2, y2, w, h = y1, w - x2, y2, w - x1, h, w
                return (x1, y1, x2, y2)
            gts_o = [(o, rb(b)) for o, b in gts_o]
            if k % 2: cols, rows = rows, cols
            mix = f"{mix}, rotated {90 * k}"
        gid = uuid.uuid4().hex[:10]
        self.grids[gid] = dict(img=img, gts_o=gts_o, cols=cols, rows=rows, seed=seed, kind="grid", size=size, mix=mix); self._keep(self.grids)
        return dict(id=gid, seed=seed, image=b64img(img), w=img.width, h=img.height, cols=cols, rows=rows, kind="grid", size=size, mix=mix)

    def upload(self, q):
        raw = base64.b64decode(q["image"].split(",")[-1])
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
        if max(img.size) > 3000: img.thumbnail((3000, 3000))
        gid = uuid.uuid4().hex[:10]; W, H = img.size
        grid_like = W % TILE == 0 and H % TILE == 0
        self.grids[gid] = dict(img=img, gts_o=None, cols=W // TILE if grid_like else 0, rows=H // TILE if grid_like else 0,
                               seed=None, kind="upload", name=str(q.get("name", "upload"))); self._keep(self.grids)
        return dict(id=gid, image=b64img(img), w=W, h=H, cols=self.grids[gid]["cols"], rows=self.grids[gid]["rows"], kind="upload")

    def detect(self, q):
        g = self.grids.get(q.get("id"))
        if not g: raise ValueError("Make a grid or upload an image first.")
        name = q.get("model") or self.S["best"]
        scene = g["kind"] == "scene" or (g["kind"] == "upload" and not g["cols"])   # v3: scenes have their own validated threshold
        conf = float(q.get("conf", (self.S["model_conf_scene"] if scene else self.S["model_conf"]).get(name, self.S["conf"])))
        use_cell = bool(q.get("cell", True)) and g.get("cells", True); deg = q.get("degrade") or "None"
        imgsz = self.S["model_imgsz"].get(name, IMGSZ)
        if deg not in DEGRADE: raise ValueError("unknown degradation")
        with self.lock:
            model, info = self.model(name)
            img = degrade(g["img"], deg); W, H = img.size
            gts = [(info.idx[o], b) for o, b in g["gts_o"]] if g["gts_o"] else None
            if FAKE: model.truth = [(c, tuple(v * imgsz / IMGSZ for v in b)) for c, b in (gts or [])]   # the stand-in answers in model pixels
            first = name not in self.warmed
            raw, ms = run_model(model, img, self.S["mode"], imgsz=imgsz); self.warmed.add(name)
        dup = self.S["scene_nms"].get(name, self.S["dup"]) if scene else self.S["dup"]   # v3: the validated scene rule (grids keep one box per cell)
        preds = select(raw, conf, dup, W, H, use_cell)
        if gts is not None:
            rows = explain(gts, preds, raw, conf, info)
            rows = sorted(rows, key=lambda r: ["missed", "wrong ID", "box off", "extra", "correct"].index(r["result"]))
            t = tally(rows, len(gts))
            per = {}
            for r in rows:
                if r["true"]: per.setdefault(r["section"], [0, 0]); per[r["section"]][1] += 1; per[r["section"]][0] += r["result"] == "correct"
            sw = sweep_series(gts, raw, dup, W, H, use_cell, info)
        else:
            preds = sorted(preds, key=lambda t: (int(t[2][1] // TILE), t[2][0]))
            rows = [dict(result="detected", reason="", predicted=info.oid(c), predicted_name=info.name(c), true="", true_name="",
                         section=info.section(c), class_AP=info.apv(c), confidence=round(p, 2), IoU=None, _gt=None, _pred=k)
                    for k, (c, p, b) in enumerate(preds)]
            t, per, sw = None, {}, None
            ts = [round(float(x), 2) for x in np.arange(0.05, 0.91, 0.05)]                 # v3.3: boxes kept at each threshold
            usw = dict(t=ts, n=[len(select(raw, x, dup, W, H, use_cell)) for x in ts])
            ucand = sorted(([info.oid(c), info.name(c), round(float(p), 3)] for c, p, b in select(raw, 0.05, dup, W, H, use_cell)), key=lambda v: -v[2])[:40]
        status = {r["_pred"]: r["result"] for r in rows if r.get("_pred") is not None}
        boxes = [dict(k=k, id=info.oid(c), name=info.name(c), section=info.section(c), ap=info.apv(c), group=info.group(c),
                      conf=round(float(p), 3), x1=num(b[0]), y1=num(b[1]), x2=num(b[2]), y2=num(b[3]), result=status.get(k, "detected"))
                 for k, (c, p, b) in enumerate(preds)]
        truths = [dict(k=k, id=info.oid(c), name=info.name(c), x1=num(b[0]), y1=num(b[1]), x2=num(b[2]), y2=num(b[3]))
                  for k, (c, b) in enumerate(gts)] if gts else []
        if gts is not None: usw, ucand = None, None
        did = uuid.uuid4().hex[:10]
        self.dets[did] = dict(img=img, preds=preds, gts=gts, rows=rows, info=info, model=name, grid=g, deg=deg); self._keep(self.dets)
        out_rows = [{**{k: num(v) for k, v in r.items() if not k.startswith("_")}, "gt": r.get("_gt"), "pred": r.get("_pred")} for r in rows]
        return dict(det=did, model=name, conf=conf, degrade=deg, ms=round(ms, 1), first_call=first, w=W, h=H, cols=g["cols"], rows=g["rows"],
                    image=b64img(img) if deg != "None" else None, boxes=boxes, truths=truths, table=out_rows,
                    tally=t, per_section=per, sweep=sw, has_truth=gts is not None,
                    upsweep=usw, ucands=[dict(id=a, name=b, conf=c) for a, b, c in ucand] if ucand is not None else None)

    # ------------------------------------------------------------------ v3.4 COMPOUND view
    def classifier(self):
        """The Milestone 1 classifier, loaded once; False if its weights are not here."""
        if self.cls is None:
            if FAKE: self.cls = "fake"
            elif not CLS_PATH: self.cls = False
            else:
                import torch, torchvision
                from torchvision import transforms as T
                m = torchvision.models.convnext_tiny(weights=None); m.classifier[2] = torch.nn.Linear(768, 73)
                ck = torch.load(str(CLS_PATH), map_location="cpu", weights_only=False); sd = ck
                for key in ("model_state_dict", "state_dict", "model"):
                    if isinstance(ck, dict) and isinstance(ck.get(key), dict): sd = ck[key]; break
                m.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()}); dev = "cuda" if torch.cuda.is_available() else "cpu"
                classes = ck.get("classes") if isinstance(ck, dict) and ck.get("classes") else [f"OBJ{i:03d}" for i in range(1, 74)]
                tf = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                self.cls = dict(m=m.eval().to(dev), dev=dev, classes=list(classes), tf=tf, torch=torch)
        return self.cls

    def cls_vote(self, tile, truth_id=None, k=0):
        """Top-1 Object ID and probability for one 224-px cell, averaged over 4 rotations (it never learned rotated photos)."""
        c = self.classifier()
        if not c: return None
        if c == "fake": return (truth_id if k % 6 else "OBJ001"), 0.97
        x = c["torch"].stack([c["tf"](tile.convert("RGB").resize((TILE, TILE)).rotate(a)) for a in (0, 90, 180, 270)]).to(c["dev"])
        with c["torch"].no_grad(): p = c["torch"].softmax(c["m"](x), 1).mean(0).cpu().numpy()
        i = int(p.argmax()); return c["classes"][i], float(p[i])

    def compound(self, q):
        """v3.4: the other models' votes on each box of the selected model. Explains only: the answer and the score stay as they are."""
        d = self.dets.get(q.get("det"))
        if not d: raise ValueError("Detect first, then switch on COMPOUND.")
        g, img, info = d["grid"], d["img"], d["info"]; W, H = img.size
        scene = g["kind"] == "scene" or (g["kind"] == "upload" and not g["cols"]); use_cell = g.get("cells", True)
        voters = [m for m in PANEL if m in self.available and m != d["model"]]
        sel = [(info.oid(c), float(p), tuple(map(float, b))) for c, p, b in d["preds"]]
        gts = [(info.oid(c), tuple(map(float, b))) for c, b in d["gts"]] if d["gts"] else None
        name = lambda o: self.OBJ.get(o, (o,))[0] if o else None
        per, timing, judged = {}, {}, {}
        for v in voters:
            conf = float((self.S["model_conf_scene"] if scene else self.S["model_conf"]).get(v, self.S["conf"]))
            imgsz = self.S["model_imgsz"].get(v, IMGSZ)
            with self.lock:
                model, vinfo = self.model(v)
                if FAKE: model.truth = [(c, tuple(x * imgsz / IMGSZ for x in b)) for c, b in (d["gts"] or [])]
                raw, ms = run_model(model, img, self.S["mode"], imgsz=imgsz); self.warmed.add(v)
            dup = self.S["scene_nms"].get(d["model"], self.S["dup"]) if scene else self.S["dup"]   # the same rules as the answer
            kept = select(raw, conf, dup, W, H, use_cell)
            per[v] = [(vinfo.oid(c), float(p), tuple(map(float, b))) for c, p, b in kept]
            timing[v] = round(ms, 1); judged[v] = (raw, kept, conf, vinfo)
        use_cls = bool(g["cols"]) and not scene and bool(self.classifier())
        cls_ms, cells = 0.0, {}
        votes = []
        res_of = {r["_pred"]: r["result"] for r in d["rows"] if r.get("_pred") is not None}
        for k, (oid, p, b) in enumerate(sel):
            vv = []
            for v in voters:
                best, bi = 0.0, None
                for j, (_, _, b2) in enumerate(per[v]):
                    u = iou(b, b2)
                    if u > best: best, bi = u, j
                if bi is not None and best >= 0.5:
                    o2, p2, _ = per[v][bi]
                    vv.append(dict(model=v, label=PANEL_LABEL.get(v, v), vote="agree" if o2 == oid else "other", id=o2, name=name(o2), conf=round(p2, 3), iou=round(best, 2)))
                else:
                    vv.append(dict(model=v, label=PANEL_LABEL.get(v, v), vote="none", id=None, name=None, conf=None, iou=round(best, 2)))
            if use_cls:
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                i, j = min(int(cx // TILE), g["cols"] - 1), min(int(cy // TILE), g["rows"] - 1)
                if (i, j) not in cells:
                    t0 = time.perf_counter(); cells[(i, j)] = self.cls_vote(img.crop((i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE)), oid, k)
                    cls_ms += (time.perf_counter() - t0) * 1000
                o2, p2 = cells[(i, j)]
                vv.append(dict(model="CLS", label=PANEL_LABEL["CLS"], vote="agree" if o2 == oid else "other", id=o2, name=name(o2), conf=round(p2, 3), iou=None))
            n_ag = sum(x["vote"] == "agree" for x in vv)
            votes.append(dict(k=k, id=oid, name=name(oid), conf=round(p, 3), result=res_of.get(k, "detected"), n_agree=n_ag, n_total=len(vv), votes=vv))
        # hints: an object that at least two other models see, where the selected model kept no box (never counted in the score)
        cand = sorted(((v, o, p, b) for v in voters for (o, p, b) in per[v] if all(iou(b, s[2]) < 0.5 for s in sel)), key=lambda t: -t[2])
        clusters = []
        for v, o, p, b in cand:
            for cl in clusters:
                if iou(b, cl["box"]) >= 0.5: cl["m"].setdefault(v, (o, p)); break
            else: clusters.append(dict(box=b, m={v: (o, p)}))
        hints = []
        for cl in clusters:
            if len(cl["m"]) < 2: continue
            ids = [o for o, _ in cl["m"].values()]; top = max(set(ids), key=lambda o: (ids.count(o), max(p for oo, p in cl["m"].values() if oo == o)))
            gt = None
            if gts:
                for gi, (go, gb) in enumerate(gts):
                    if go == top and iou(cl["box"], gb) >= 0.5: gt = gi; break
            x1, y1, x2, y2 = cl["box"]
            hints.append(dict(id=top, name=name(top), n=len(cl["m"]), models=[PANEL_LABEL.get(v, v) for v in cl["m"]], x1=round(x1, 1), y1=round(y1, 1),
                              x2=round(x2, 1), y2=round(y2, 1), gt=gt, true=gt is not None))
        score, views, cls_cells = None, {}, []
        for v in voters:                                      # v3.5: each voter's own boxes, shaped like a detection, for the split pupils
            raw_v, kept_v, conf_v, vinfo = judged[v]
            rows_v = []
            if gts is not None:
                gts_v = [(vinfo.idx[o], b) for o, b in gts if o in vinfo.idx]
                if len(gts_v) == len(gts): rows_v = explain(gts_v, kept_v, raw_v, conf_v, vinfo)
            status = {r["_pred"]: r["result"] for r in rows_v if r.get("_pred") is not None}
            views[v] = dict(boxes=[dict(k=k, id=vinfo.oid(c), name=vinfo.name(c), conf=round(float(p), 3), x1=num(b[0]), y1=num(b[1]), x2=num(b[2]), y2=num(b[3]),
                                        result=status.get(k, "detected")) for k, (c, p, b) in enumerate(kept_v)],
                            table=[dict(result=r["result"], gt=r.get("_gt"), pred=r.get("_pred")) for r in rows_v])
        if gts is not None:                                   # v3.4: every model judged against the truth, exactly like the answer
            def verdicts(rows):
                out = {}
                for r in rows:
                    if r.get("_gt") is None: continue
                    seen = re.search(r"seen at confidence ([0-9.]+)", r.get("reason") or "")
                    out[r["_gt"]] = dict(result=r["result"], pred=r.get("predicted") or None, pred_name=r.get("predicted_name") or None,
                                         iou=num(r.get("IoU")), conf=num(r.get("confidence")), seen=float(seen.group(1)) if seen else None)
                return out
            cols = [dict(model=d["model"], label=PANEL_LABEL.get(d["model"], d["model"]), kind="answer", tally=tally(d["rows"], len(gts)), v=verdicts(d["rows"]))]
            for v in voters:
                raw_v, kept_v, conf_v, vinfo = judged[v]
                gts_v = [(vinfo.idx[o], b) for o, b in gts if o in vinfo.idx]
                if len(gts_v) != len(gts): continue
                rows_v = explain(gts_v, kept_v, raw_v, conf_v, vinfo)
                cols.append(dict(model=v, label=PANEL_LABEL.get(v, v), kind="detector", tally=tally(rows_v, len(gts)), v=verdicts(rows_v), ms=timing[v]))
            if bool(g["cols"]) and not scene and self.classifier():
                cv, named = {}, 0; t0 = time.perf_counter()
                for gi, (go, gb) in enumerate(gts):
                    cx, cy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                    i, j = min(int(cx // TILE), g["cols"] - 1), min(int(cy // TILE), g["rows"] - 1)
                    if (i, j) not in cells: cells[(i, j)] = self.cls_vote(img.crop((i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE)), go, gi)
                    o2, p2 = cells[(i, j)]; ok = o2 == go; named += ok
                    cv[gi] = dict(result="named" if ok else "misnamed", pred=o2, pred_name=name(o2), conf=round(p2, 3))
                cols.append(dict(model="CLS", label=PANEL_LABEL["CLS"], kind="classifier", tally=dict(n=len(gts), named=named), v=cv,
                                 ms=round(cls_ms + (time.perf_counter() - t0) * 1000, 1)))
            for c_ in cols:                                   # v3.5: the classifier's cell verdicts, for its split pupil
                if c_["kind"] == "classifier":
                    for gi, (go, gb) in enumerate(gts):
                        cx, cy = (gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2
                        i, j = min(int(cx // TILE), g["cols"] - 1), min(int(cy // TILE), g["rows"] - 1); x = c_["v"].get(gi)
                        if x: cls_cells.append(dict(gt=gi, x1=i * TILE, y1=j * TILE, x2=(i + 1) * TILE, y2=(j + 1) * TILE, ok=x["result"] == "named", pred=x["pred"]))
            objs = []
            for gi, (go, gb) in enumerate(gts):
                vs = {c["model"]: c["v"].get(gi) for c in cols}
                good = [m for m, x in vs.items() if x and x["result"] in ("correct", "named")]
                objs.append(dict(gt=gi, id=go, name=name(go), verdicts=vs, n_right=len(good), n_models=len(cols)))
            objs.sort(key=lambda o: (o["n_right"] == o["n_models"], o["n_right"], o["gt"]))
            extras = {c["model"]: c["tally"].get("extra", 0) for c in cols if c["kind"] != "classifier"}
            score = dict(columns=[{k: x for k, x in c.items() if k != "v"} for c in cols], objects=objs, extras=extras)
        n = len(votes); full = sum(x["n_total"] and x["n_agree"] == x["n_total"] for x in votes)
        alone = sum(x["n_total"] and x["n_agree"] == 0 for x in votes)
        by_level = {}
        if gts is not None:
            for x in votes:
                lv = "all agree" if x["n_agree"] == x["n_total"] else ("none agree" if x["n_agree"] == 0 else "some agree")
                by_level.setdefault(lv, [0, 0]); by_level[lv][1] += 1; by_level[lv][0] += x["result"] == "correct"
        return dict(det=q.get("det"), model=d["model"], label=PANEL_LABEL.get(d["model"], d["model"]),
                    voters=[dict(model=v, label=PANEL_LABEL.get(v, v), ms=timing[v]) for v in voters] + ([dict(model="CLS", label=PANEL_LABEL["CLS"], ms=round(cls_ms, 1))] if use_cls else []),
                    votes=votes, hints=hints, has_truth=gts is not None, score=score, views=views, cls_cells=cls_cells, summary=dict(objects=n, all_agree=full, none_agree=alone, split=n - full - alone,
                    hints=len(hints), hints_true=sum(h["true"] for h in hints), by_level=by_level))

    def tile(self, did, k, size=150):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        rows = d["rows"]
        if not 0 <= k < len(rows): raise ValueError("unknown object")
        img, cap = one_tile(d["img"], d["preds"], d["info"], self.train, d["gts"], rows[k], max(120, min(480, int(size))))
        return dict(image=b64img(img, 92), caption=cap)

    def download(self, did, kind):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        info = d["info"]
        if kind == "png":
            b = io.BytesIO(); draw(d["img"], d["preds"], info, d["gts"], d["rows"] if d["gts"] else None).save(b, "PNG")
            return b.getvalue(), "image/png", f"detections_{did}.png"
        tab = (pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in d["rows"]]) if d["gts"]
               else det_table(d["preds"], info))
        if kind == "csv": return tab.to_csv(index=False).encode(), "text/csv", f"detections_{did}.csv"
        if kind == "json": return tab.to_json(orient="records", indent=2).encode(), "application/json", f"detections_{did}.json"
        raise ValueError("unknown download type")

    def snapshot(self, did):
        d = self.dets.get(did)
        if not d: raise ValueError("unknown detection")
        info, g = d["info"], d["grid"]
        ann = draw(d["img"], d["preds"], info, d["gts"], d["rows"] if d["gts"] else None)
        ann = ImageOps.contain(ann, (900, 900)); P = 560; W = ann.width + P + 60; H = max(ann.height + 120, 760)
        out = Image.new("RGB", (W, H), (4, 7, 10)); dr = ImageDraw.Draw(out)
        ft, fb, fs = font(30), font(20), font(16)
        dr.text((30, 26), "THE EYE – Explainable YOLO Explorer", fill=(57, 255, 136), font=ft)
        sub = (f"{g['cols']}x{g['rows']} grid · seed {g['seed']} · {g.get('mix', '')}" if g["kind"] == "grid" else f"upload {d['img'].width}x{d['img'].height}")
        dr.text((30, 66), f"{sub} · model {d['model']}" + (f" · {d['deg']}" if d["deg"] != "None" else ""), fill=(127, 165, 148), font=fs)
        out.paste(ann, (30, 100)); x0, y = ann.width + 60, 100
        if d["gts"]:
            t = tally(d["rows"], len(d["gts"]))
            dr.text((x0, y), f"{t['correct']} / {t['n']} correct", fill=(220, 255, 240), font=font(40)); y += 56
            dr.text((x0, y), f"{t['named']} / {t['n']} named correctly · box off {t['box_off']} · wrong ID {t['wrong']} · missed {t['missed']} · extra {t['extra']}",
                    fill=(127, 165, 148), font=fs); y += 40
            errs = [r for r in d["rows"] if r["result"] != "correct"]
            dr.text((x0, y), "ERRORS" if errs else "NO ERRORS", fill=(57, 255, 136), font=fb); y += 32
            col = {"missed": (34, 228, 255), "wrong ID": (255, 79, 216), "box off": (255, 176, 32), "extra": (139, 151, 168)}
            for r in errs[:14]:
                dr.rectangle([x0, y + 5, x0 + 10, y + 15], fill=col.get(r["result"], (200, 200, 200)))
                dr.text((x0 + 20, y), f"{r['true'] or r['predicted']} · {r['result']}", fill=(220, 255, 240), font=fs); y += 22
                words, line = r["reason"].split(), ""
                for w_ in words:
                    if dr.textlength(line + " " + w_, font=fs) > P - 30: dr.text((x0 + 20, y), line.strip(), fill=(127, 165, 148), font=fs); y += 20; line = ""
                    line += " " + w_
                if line.strip(): dr.text((x0 + 20, y), line.strip(), fill=(127, 165, 148), font=fs); y += 20
                y += 8
        else:
            dr.text((x0, y), f"{len(d['preds'])} objects found", fill=(220, 255, 240), font=font(36)); y += 52
            for c, p_, b in d["preds"][:24]:
                dr.text((x0, y), f"{info.oid(c)}  {info.name(c)}  {p_:.2f}", fill=(127, 165, 148), font=fs); y += 22
        b = io.BytesIO(); out.save(b, "PNG"); return b.getvalue()

    def stress_start(self, q):
        jid = uuid.uuid4().hex[:8]; n = max(1, min(STRESS_MAX, int(q.get("n", 100))))
        self.jobs[jid] = dict(done=0, total=n, result=None, error=None); self._keep(self.jobs, 10)
        threading.Thread(target=self._stress, args=(jid, q, n), daemon=True).start()
        return dict(job=jid, total=n)

    def _stress(self, jid, q, n):
        J = self.jobs[jid]
        try:
            size = q.get("size") or "5x5 (25 objects)"; cols, rows_n = GRIDS[size]
            name = q.get("model") or self.S["best"]; conf = float(q.get("conf", self.S["model_conf"].get(name, self.S["conf"]))); use_cell = bool(q.get("cell", True))
            deg = q.get("degrade") or "None"; mix = q.get("mix") or "Random"; secs = q.get("sections") or self.sections
            tot, fully, ms_all, miss_by, pairs, reasons = Counter(), [], [], Counter(), Counter(), Counter()
            for g in range(n):
                img, gts_o = make_grid(self.test, cols, rows_n, secs, mix, 1000 + g, self.OBJ)
                with self.lock:
                    model, info = self.model(name)
                    img = degrade(img, deg); W, H = img.size; gts = [(info.idx[o], b) for o, b in gts_o]
                    imgsz = self.S["model_imgsz"].get(name, IMGSZ)
                    if FAKE: model.truth = [(c, tuple(v * imgsz / IMGSZ for v in b)) for c, b in gts]
                    raw, ms = run_model(model, img, self.S["mode"], imgsz=imgsz)
                ms_all.append(ms)
                rws = explain(gts, select(raw, conf, self.S["dup"], W, H, use_cell), raw, conf, info); t = tally(rws, len(gts))
                for k in ("n", "correct", "named", "box_off", "wrong", "missed", "extra"): tot[k] += t[k]
                fully.append(t["fully"])
                for r in rws:
                    if r["result"] in ("missed", "box off", "wrong ID") and r["true"]: miss_by[f"{r['true']} {r['true_name']}".strip()] += 1
                    if r["result"] == "wrong ID": pairs[f"{r['true']} -> {r['predicted']}"] += 1
                    if r["result"] != "correct": reasons[re.sub(r"[\d.]+", "#", r["reason"].split(";")[0]) or r["result"]] += 1
                J["done"] = g + 1
            lo, hi = bootstrap_ci(fully)
            J["result"] = dict(n_grids=n, size=size, model=name, mix=mix, degrade=deg, sections=secs, totals=dict(tot),
                               fully=float(np.mean(fully)), ci=[lo, hi], ms=float(np.median(ms_all)),
                               hardest=miss_by.most_common(8), confusions=pairs.most_common(8), reasons=reasons.most_common(6))
        except Exception as e:
            J["error"] = str(e)

def sweep_series(gts, raw, dup, W, H, use_cell, info):
    ts = [round(float(t), 2) for t in np.arange(0.05, 0.91, 0.05)]; out = dict(t=ts, correct=[], missed=[], extra=[], wrong=[])
    for t in ts:
        tl = tally(explain(gts, select(raw, t, dup, W, H, use_cell), raw, t, info), len(gts))
        out["correct"].append(tl["correct"]); out["missed"].append(tl["missed"] + tl["box_off"])
        out["extra"].append(tl["extra"] + tl["box_off"]); out["wrong"].append(tl["wrong"])
    return out


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items(): self.send_header(k, v)
            self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            try:
                p = self.path.split("?")[0].strip("/").split("/")
                if p == [""]: return self._send(200, PAGE, "text/html; charset=utf-8")
                if p == ["api", "config"]: return self._send(200, app.config())
                if p == ["api", "status"]: return self._send(200, app.status())
                if len(p) == 3 and p[:2] == ["api", "snapshot"]:
                    return self._send(200, app.snapshot(p[2]), "image/png", {"Content-Disposition": f'attachment; filename="snapshot_{p[2]}.png"'})
                if len(p) == 4 and p[:2] == ["api", "tile"]:
                    from urllib.parse import parse_qs
                    q = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
                    return self._send(200, app.tile(p[2], int(p[3]), int((q.get("size") or ["150"])[0])))
                if len(p) == 3 and p[:2] == ["api", "stress"]:
                    j = app.jobs.get(p[2]); return self._send(200 if j else 404, j or dict(error="unknown job"))
                if len(p) == 4 and p[:2] == ["api", "download"]:
                    data, ctype, fname = app.download(p[2], p[3])
                    return self._send(200, data, ctype, {"Content-Disposition": f'attachment; filename="{fname}"'})
                self._send(404, dict(error="not found"))
            except Exception as e:
                self._send(400, dict(error=str(e)))
        def do_POST(self):
            try:
                q = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                route = {"/api/grid": app.grid, "/api/upload": app.upload, "/api/detect": app.detect, "/api/stress": app.stress_start,
                         "/api/compound": app.compound}.get(self.path)
                if not route: return self._send(404, dict(error="not found"))
                self._send(200, route(q))
            except Exception as e:
                self._send(400, dict(error=str(e)))
    return H

def main():
    app = App()
    host = os.environ.get("M2_HOST", "127.0.0.1")          # "0.0.0.0" = reachable from other devices on your network
    srv = ThreadingHTTPServer((host, PORT), make_handler(app))
    threading.Thread(target=app.warmup, daemon=True).start()          # load the model while the page opens
    url = f"http://127.0.0.1:{PORT}"
    if host != "127.0.0.1":
        import socket
        try: ip = socket.gethostbyname(socket.gethostname())
        except Exception: ip = "<this computer's IP>"
        print(f"Shared on your network: http://{ip}:{PORT}  (only on a trusted network; the app shows dataset photos)")
    print(f"The EYE (Explainable YOLO Explorer) running at {url}  (model {app.S['best']}; press Ctrl+C to stop)")
    if os.environ.get("M2_NO_BROWSER") != "1":
        import webbrowser; threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The EYE v3.5 – Explainable YOLO Explorer</title>
<script>try { if (localStorage.getItem("m2theme") === "light") document.documentElement.classList.add("light"); } catch (e) { }</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&family=Saira+Stencil+One&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#04070a; --tile:rgba(8,16,13,.68); --tile2:rgba(10,22,17,.86);
  --line:rgba(57,255,136,.18); --line2:rgba(57,255,136,.5);
  --acc:#39ff88; --acc-d:#16c47a; --cyan:#22e4ff; --amber:#ffb020; --mag:#ff4fd8; --grey:#8b97a8; --gold:#ffd84a;
  --txt:#dcfff0; --dim:#7fa594; --dim2:#4d6b5e;
  --mono:"JetBrains Mono",ui-monospace,Consolas,Menlo,monospace; --sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;background:var(--bg);color:var(--txt);font-family:var(--sans)}
#bg{position:fixed;inset:0;z-index:0;display:block}
/* ---------------------------------------------------------------- dock */
.dock{position:fixed;left:18px;top:50%;transform:translateY(-50%);z-index:5;display:flex;flex-direction:column;gap:10px;
  padding:12px 9px;background:var(--tile);border:1px solid var(--line);border-radius:18px;backdrop-filter:blur(7px);box-shadow:0 18px 50px rgba(0,0,0,.55)}
.dbtn{width:62px;display:flex;flex-direction:column;align-items:center;gap:5px;padding:9px 4px;border-radius:12px;border:1px solid transparent;background:none;
  color:var(--dim);font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;cursor:pointer;transition:all .15s}
.dbtn svg{width:22px;height:22px;stroke:currentColor;fill:none;stroke-width:1.6}
.dbtn:hover{color:var(--acc);border-color:var(--line2);background:rgba(57,255,136,.06)}
.dbtn.on{color:var(--acc)}
.dsep{height:1px;background:var(--line);margin:2px 6px}
.dock.right{left:auto;right:18px}
/* ---------------------------------------------------------------- main column */
.main{position:relative;z-index:1;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:16px 110px 34px 110px}
.top{width:var(--ew);display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:18px;margin-bottom:6px;font-family:var(--mono)}
.top h1,.logo{margin:0;display:flex;flex-direction:column;align-items:center;gap:4px;white-space:nowrap;font-family:var(--mono)}
.logo .t1{font-size:22px;letter-spacing:.46em;font-weight:500;color:var(--acc);text-shadow:0 0 18px rgba(57,255,136,.5);text-transform:uppercase;margin-right:-.46em}
.logo .t1 b{color:var(--txt);font-weight:700}
.logo .t2{font-size:12px;letter-spacing:.34em;font-weight:500;color:var(--dim);text-transform:uppercase;margin-right:-.34em}
.logo .t2 i{font-style:normal;color:var(--acc);font-weight:700;text-shadow:0 0 10px rgba(57,255,136,.6)}
/* intro: the title appears in the pupil, letter by letter; the E's blink; then it flies to the top */
.intrologo{position:absolute;left:50%;top:50%;z-index:2;pointer-events:none;transform-origin:50% 50%}
.intrologo{--txt:#e6fff2;--dim:#9fd3b8;--acc:#39ff88}          /* the pupil is dark in both themes */
.intrologo .lt{display:inline-block;opacity:0;transform:translateY(14px) scale(.4)}
.intrologo.go .lt{animation:ltpop .55s cubic-bezier(.3,1.6,.5,1) forwards;animation-delay:var(--d)}
@keyframes ltpop{0%{opacity:0;transform:translateY(14px) scale(.4)}100%{opacity:1;transform:none}}
.intrologo.blink .lt.ee{animation:ltpop .01s forwards,eeblink .36s ease-in-out .02s 2}
@keyframes eeblink{50%{transform:scaleY(.12)}}
.intrologo .t2{opacity:0;transform:translateY(6px);transition:opacity .6s ease,transform .6s ease}
.intrologo.sub .t2{opacity:1;transform:none}
.flylogo{position:fixed;z-index:46;pointer-events:none;transform-origin:50% 50%}
body.intro .pempty{opacity:0} .pempty{transition:opacity .6s ease}
.top .meta{justify-self:end} .top .meta span[data-tip]{cursor:help;border-bottom:1px dotted var(--line2)}
.top .meta{font-size:12px;color:var(--dim);white-space:nowrap}
/* ---------------------------------------------------------------- eyebrow */
.brow{position:relative;width:calc(var(--ew) * 1.2);height:220px}
.brow svg.arc{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none}
.browband{fill:url(#browfill);stroke:rgba(57,255,136,.22);stroke-width:1}
.bf1{stop-color:rgba(57,255,136,.13)} .bf2{stop-color:rgba(57,255,136,.04)}
.hair{stroke:rgba(57,255,136,.26);stroke-width:1.1;stroke-linecap:round}
.browedge{fill:none;stroke:rgba(57,255,136,.55);stroke-width:1.6;filter:drop-shadow(0 0 6px rgba(57,255,136,.55))}
.seg{position:absolute;top:0;transform-origin:50% 50%;background:var(--tile);border:1px solid var(--line);border-radius:14px;padding:10px 11px;backdrop-filter:blur(7px);
  box-shadow:0 14px 40px rgba(0,0,0,.5);display:flex;flex-direction:column;gap:7px;transition:border-color .2s,box-shadow .2s}
.seg:hover{border-color:var(--line2);box-shadow:0 14px 44px rgba(0,0,0,.55),0 0 22px rgba(57,255,136,.12)}
.seg h3{margin:0;font:600 12px var(--mono);letter-spacing:.22em;color:var(--acc);text-transform:uppercase;display:flex;justify-content:space-between}
.seg h3 span{color:var(--dim2);font-weight:500;letter-spacing:.06em}
.lbl{font:11.5px var(--mono);color:var(--dim);letter-spacing:.05em}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{border:1px solid var(--line);background:rgba(0,0,0,.25);color:var(--dim);border-radius:999px;padding:4px 8px;font:500 12px var(--mono);cursor:pointer;
  transition:all .15s;user-select:none}
.chip:hover{color:var(--txt);border-color:var(--line2)}
.chip.on{color:var(--acc);border-color:var(--acc);background:rgba(57,255,136,.12);box-shadow:0 0 10px rgba(57,255,136,.18)}
select,input[type=text]{width:100%;background:rgba(0,0,0,.35);color:var(--txt);border:1px solid var(--line);border-radius:9px;padding:6px 8px;font:500 12.5px var(--mono);outline:none}
select:focus,input:focus{border-color:var(--acc)}
input[type=range]{width:100%;accent-color:var(--acc);margin:0}
.toggle{display:flex;align-items:center;gap:6px;font:11.5px var(--mono);color:var(--dim);cursor:pointer;user-select:none}
.toggle input{accent-color:var(--acc);margin:0}
.mini{display:flex;gap:5px}
.seg>.mbtn{flex:none;width:100%}
.seg .mbtn{white-space:normal;line-height:1.25;overflow-wrap:anywhere;letter-spacing:.05em}   /* v3: a label never runs past its pill */
.seg{overflow-y:auto;overflow-x:hidden;scrollbar-width:thin;scrollbar-color:var(--line2) transparent}
.seg::-webkit-scrollbar{width:5px}.seg::-webkit-scrollbar-thumb{background:var(--line2);border-radius:9px}
.seg.scrolls h3{position:sticky;top:-10px;z-index:1;background:#08130e;margin:-10px -11px 0;padding:10px 11px 5px;box-shadow:0 6px 8px -6px rgba(0,0,0,.6)}
html.light .seg.scrolls h3{background:#fbfdfc;box-shadow:0 6px 8px -6px rgba(20,70,45,.18)}
#model{font-size:11.5px;padding:6px 5px}
.thrkind{font:11px var(--mono);color:var(--dim);letter-spacing:.04em}
.eyewrap.dz-on .almond{stroke-width:4;stroke-dasharray:10 7;animation:dzspin 1.2s linear infinite}
@keyframes dzspin{to{stroke-dashoffset:-34}}
.dzhint{position:absolute;inset:0;z-index:6;display:none;align-items:center;justify-content:center;pointer-events:none;font:600 14px var(--mono);letter-spacing:.2em;text-transform:uppercase;color:var(--acc);text-shadow:0 0 12px rgba(57,255,136,.8)}
.eyewrap.dz-on .dzhint{display:flex}
html.light .eyewrap.dz-on .dzhint{text-shadow:none}
.mbtn{flex:1;white-space:nowrap;border:1px solid var(--line);background:rgba(0,0,0,.3);color:var(--dim);border-radius:8px;padding:5px 4px;font:500 11px var(--mono);letter-spacing:.08em;
  text-transform:uppercase;cursor:pointer}
.mbtn:hover{color:var(--acc);border-color:var(--acc)}
/* ---------------------------------------------------------------- the eye */
.eyewrap{position:relative;width:var(--ew);height:var(--eh);margin:0 auto}
.eyesvg,.lidsvg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.lidsvg{z-index:3;pointer-events:none}
.sclera{fill:url(#scl)}
.almond{fill:none;stroke:var(--acc);stroke-width:2;filter:drop-shadow(0 0 10px rgba(57,255,136,.55))}
.irisbase{fill:url(#irisg)}
.irisline{stroke:rgba(57,255,136,.06);stroke-width:1}
.gauge path{fill:none;stroke-width:var(--gw);stroke-linecap:butt;opacity:0;transition:opacity .25s}
.gauge path.show{opacity:1}
.gauge .tick{stroke:rgba(57,255,136,.18)}
.gauge path.plate{stroke:#d6fbe4;opacity:1}                 /* the DETECT button's own slot: pale, not a result colour */
.spin{transform-box:fill-box;transform-origin:center;animation:spin 1.6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.glint{fill:rgba(255,255,255,.7);filter:blur(1px)}
.lid .skin{fill:#071210}
.lid .edge{stroke:var(--acc);stroke-width:2.5;filter:drop-shadow(0 0 8px rgba(57,255,136,.8))}
.pupil{position:absolute;z-index:2;border-radius:50%;overflow:hidden;background:radial-gradient(circle at 50% 50%,#020504 60%,#000 100%);
  box-shadow:0 0 0 var(--pring,4px) rgba(57,255,136,.55),0 0 18px rgba(57,255,136,.35),0 0 40px rgba(57,255,136,.18) inset;cursor:zoom-in;transition:transform .35s cubic-bezier(.2,.9,.3,1.4)}
.pupil.narrow{transform:scale(.9)}
.lens{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);display:flex;align-items:center;justify-content:center}
.zoomer{position:relative;transition:transform .55s cubic-bezier(.2,.8,.2,1);transform-origin:0 0}
.zoomer img{display:block;width:100%;height:100%}
.zoomer svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.pempty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;text-align:center;font:12.5px/1.7 var(--mono);color:var(--dim2);padding:22%}
.scan{position:absolute;left:-5%;right:-5%;height:3px;top:0;background:linear-gradient(90deg,transparent,var(--acc),transparent);
  box-shadow:0 0 22px 4px rgba(57,255,136,.55);animation:scan .75s cubic-bezier(.5,0,.5,1) forwards;pointer-events:none;z-index:2}
@keyframes scan{from{top:0;opacity:1}to{top:100%;opacity:.2}}
.cover{position:absolute;background:#010302;animation:uncover .4s ease-out forwards;pointer-events:none}
@keyframes uncover{from{opacity:1}to{opacity:0}}
/* boxes (shared by pupil + stage) */
.box rect.f{fill:none;stroke-width:2.6;vector-effect:non-scaling-stroke}
.det{opacity:0;transform-box:fill-box;transform-origin:center;animation:boxin .38s cubic-bezier(.2,.9,.3,1.2) forwards;animation-delay:var(--d,0s)}
@keyframes boxin{from{opacity:0;transform:scale(.88)}to{opacity:1;transform:scale(1)}}
.det.err rect.f{animation:pulse .5s ease-in-out 2;animation-delay:calc(var(--d,0s) + .42s)}
@keyframes pulse{50%{stroke-width:6}}
.det.missed rect.f{animation:blink .45s steps(1) 4;animation-delay:calc(var(--d,0s) + .42s)}
@keyframes blink{50%{opacity:.15}}
.b-correct rect.f,.b-detected rect.f{stroke:var(--acc);filter:drop-shadow(0 0 4px rgba(57,255,136,.6))}
.b-box-off rect.f{stroke:var(--amber);stroke-dasharray:9 5}
.b-wrong-ID rect.f{stroke:var(--mag);filter:drop-shadow(0 0 5px rgba(255,79,216,.6))}
.b-extra rect.f{stroke:var(--grey);stroke-dasharray:2 4}
.b-missed rect.f{stroke:var(--cyan);stroke-dasharray:6 4}
.truth rect{fill:none;stroke:rgba(255,255,255,.55);stroke-width:1.4;stroke-dasharray:4 4;vector-effect:non-scaling-stroke}
.lab rect{fill:var(--c)}.lab text{font:700 12.5px var(--mono);fill:#02130a}
.b-correct,.b-detected{--c:var(--acc)}.b-box-off{--c:var(--amber)}.b-wrong-ID{--c:var(--mag)}.b-extra{--c:var(--grey)}.b-missed{--c:var(--cyan)}
.box{cursor:pointer}
.box rect.f{pointer-events:all}            /* the whole box area is clickable, not only its outline */
.hl{fill:rgba(0,0,0,.62);fill-rule:evenodd;pointer-events:none}
.hlf{fill:none;stroke:var(--gold);stroke-width:4;vector-effect:non-scaling-stroke;filter:drop-shadow(0 0 8px rgba(255,216,74,.8))}
/* ---------------------------------------------------------------- detect */
.detrow{display:flex;flex-direction:column;align-items:center;gap:8px;margin:8px 0 2px}
/* the iris ring is the DETECT button */
.detect{cursor:pointer;outline:none}
.detect .ringhit{fill:transparent;pointer-events:all}
.detect .ringglow{fill:none;stroke:rgba(57,255,136,0);pointer-events:none;transition:stroke .25s}
.detect:hover .ringglow,.detect:focus-visible .ringglow{stroke:rgba(57,255,136,.20)}
.detect:active .ringglow{stroke:rgba(57,255,136,.32)}
.detect .detlabel{font-family:var(--mono);font-weight:800;letter-spacing:.5em;fill:#04381f;pointer-events:none;user-select:none;
  paint-order:stroke;stroke:rgba(170,255,205,.75);stroke-width:3.2px;stroke-linejoin:round}
.detect:hover .detlabel{fill:#022814;stroke:rgba(200,255,225,.95)}
.detect.dis{pointer-events:none} .detect.dis .detlabel{opacity:.35}
.status{font:12px var(--mono);color:var(--dim);letter-spacing:.06em;position:relative;z-index:2}
.next{animation:nextglow 1.6s ease-in-out infinite}
@keyframes nextglow{50%{box-shadow:0 0 0 1px var(--acc),0 0 30px rgba(57,255,136,.45)}}
.detect.next .ringglow{animation:nextring 1.6s ease-in-out infinite}
@keyframes nextring{0%,100%{stroke:rgba(57,255,136,.04)}50%{stroke:rgba(57,255,136,.30)}}
/* ---------------------------------------------------------------- review matrix */
.review{width:var(--ew);display:grid;grid-template-columns:repeat(4,1fr);gap:14px;position:relative;z-index:2;margin-top:-6px}
.review .tile{transform-origin:50% 0}
.review{transition:grid-template-columns .4s cubic-bezier(.2,.8,.2,1)}
.review.wide{grid-template-columns:.95fr 2.1fr .95fr .62fr}       /* Results widens when a row is expanded; the sweep gives way */
body.calm .review{transition:none}
.beam{position:absolute;left:0;top:0;width:100%;pointer-events:none;z-index:1;overflow:visible}
.beam .cone{fill:url(#beamg)} .beam .cardbeam{fill:url(#beamg)} .beam .bg1{stop-color:rgba(57,255,136,.20)} .beam .bg2{stop-color:rgba(57,255,136,.06)} .beam .bg3{stop-color:rgba(57,255,136,0)}
.beam .ray{stroke:url(#rayg);stroke-width:1.4;opacity:.5;animation:shimmer 3.2s ease-in-out infinite}
.beam .rg1{stop-color:rgba(57,255,136,.55)} .beam .rg2{stop-color:rgba(57,255,136,0)}
@keyframes shimmer{0%,100%{opacity:.18}50%{opacity:.6}}
.beam.beam-pulse .cone{animation:beamflash 1.1s ease-out}
@keyframes beamflash{0%{opacity:1}25%{opacity:2.2;filter:brightness(1.8)}100%{opacity:1}}
.tile.fall{animation:fall .75s cubic-bezier(.25,1.25,.4,1) both;animation-delay:var(--fd,0s)}
@keyframes fall{0%{translate:0 -70px;opacity:0}60%{opacity:1}100%{translate:0 0;opacity:1}}
.rtitle{grid-column:1 / -1;font:500 12px var(--mono);letter-spacing:.3em;color:var(--dim2);display:flex;align-items:center;gap:10px}
.rtitle::after{content:"";flex:1;height:1px;background:var(--line)}
.tile{position:relative;height:236px;background:var(--tile);border:1px solid var(--line);border-radius:16px;padding:13px 14px;display:flex;flex-direction:column;
  overflow:hidden;backdrop-filter:blur(7px);box-shadow:0 18px 50px rgba(0,0,0,.55);transition:border-color .25s,box-shadow .25s}
.tile:hover{border-color:var(--line2)}
.th{display:flex;align-items:center;gap:9px;margin-bottom:9px}
.th .ix{font:500 11.5px var(--mono);color:var(--dim2)}
.th h3{margin:0;font:600 12.5px var(--mono);letter-spacing:.22em;text-transform:uppercase;color:var(--acc)}
.th .sp{flex:1}.th .hint{font:11.5px var(--mono);color:var(--dim2)}
.tb{flex:1;display:flex;flex-direction:column;gap:8px;min-height:0}
.big{font:700 34px var(--mono);line-height:1}
.score{display:flex;align-items:baseline;gap:6px}.score .of{font:500 14px var(--mono);color:var(--dim)}
.bar{height:6px;border-radius:9px;background:rgba(255,255,255,.06);overflow:hidden}
.bar i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));box-shadow:0 0 12px var(--acc);transition:width .9s cubic-bezier(.2,.8,.2,1)}
.verdict{font:600 12.5px var(--mono);letter-spacing:.1em}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:4px 10px;font:12.5px var(--mono);color:var(--dim)}.stats b{color:var(--txt)}
.list{flex:1;overflow:auto;display:flex;flex-direction:column;gap:5px;padding-right:3px;min-height:0}
.list::-webkit-scrollbar{width:5px}.list::-webkit-scrollbar-thumb{background:var(--line);border-radius:5px}
.item{display:flex;gap:8px;align-items:flex-start;padding:6px 8px;border-radius:9px;border:1px solid transparent;cursor:pointer;background:rgba(0,0,0,.2);font:12px/1.35 var(--mono)}
.item:hover{border-color:var(--line2)}.item.sel{border-color:var(--acc);background:rgba(57,255,136,.08)}
.dot{flex:0 0 8px;height:8px;border-radius:2px;margin-top:3px;background:var(--c)}
.r-correct,.r-detected{--c:var(--acc)}.r-box-off{--c:var(--amber)}.r-wrong-ID{--c:var(--mag)}.r-extra{--c:var(--grey)}.r-missed{--c:var(--cyan)}
.item .t{color:var(--txt)}.item .w{color:var(--dim)}
.insp{flex:1;display:flex;flex-direction:column;gap:6px;min-height:0}
.ftabs{display:flex;gap:4px} .chip.sm{padding:2px 7px;font-size:11px}
.enl{background:none;border:1px solid var(--line);color:var(--dim);border-radius:7px;padding:1px 7px;font:13px var(--mono);cursor:pointer;line-height:1.4}
.enl:hover{color:var(--acc);border-color:var(--acc)}
.item .ib{flex:1;min-width:0}
.item .sent{color:var(--txt);line-height:1.4}
.cmpb{display:inline-block;margin-left:6px;padding:0 6px;border-radius:9px;border:1px solid currentColor;font:600 11px/1.5 var(--mono);white-space:nowrap;vertical-align:1px}
/* v3.4: larger type everywhere except the curved eyebrow, whose fixed cells keep their original sizes */
#brow .seg h3{font-size:10.5px} #brow .chip{font-size:10.5px} #brow .lbl{font-size:10px} #brow select,#brow input[type=text]{font-size:11px}
#ov-stage .stagebar{font-size:11px;flex-wrap:nowrap;gap:9px} #ov-stage .stagebar>*{flex:none} #ov-stage .stagebar .sp{flex:0 0 4px}
#ov-stage #ssub{flex:1 1 auto;min-width:40px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;text-align:right} #ov-stage .stagebar h2{font-size:12px} #ov-stage .sctl{font-size:10px} #ov-stage .sctl select,#ov-stage .sctl input{font-size:10.5px} #ov-stage .stagebar .close{font-size:11px}
#brow .toggle{font-size:10px} #brow #model{font-size:10px} #brow .thrkind{font-size:9.5px} #brow .mbtn{font-size:9.5px} #brow .btn{font-size:10.5px}
/* v3.4: no frosted glass while the camera moves (it would re-blur the scaling eye behind every card, every frame) */
html.focusing .seg,html.focusing .tile,html.focusing .dock,html.focusing .howbtn,html.focusing #tip{backdrop-filter:none!important;-webkit-backdrop-filter:none!important}
/* v3.5: the COMPOUND split */
.multi{position:absolute;inset:0;z-index:4;pointer-events:none}
.multi .mp{position:absolute;pointer-events:auto;cursor:zoom-in;will-change:transform}
.multi .mring{position:absolute;inset:0} .multi .mring svg{width:100%;height:100%;display:block}
.multi .mpi{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border-radius:50%;overflow:hidden;display:flex;align-items:center;justify-content:center;
  background:radial-gradient(circle at 50% 50%,#020504 60%,#000 100%);box-shadow:0 0 0 4px rgba(57,255,136,.55),0 0 18px rgba(57,255,136,.35),0 0 40px rgba(57,255,136,.18) inset}
.multi .mring,.multi .mcap{transition:opacity .3s ease .38s} .multi.pre .mring,.multi.pre .mcap{opacity:0;transition:opacity .15s ease}
.multi .mzoom{position:relative;flex:none} .multi .mzoom img{display:block;width:100%;height:100%} .multi .mzoom svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.multi .lab{display:none}
.multi .mcap{position:absolute;left:50%;top:100%;transform:translateX(-50%);margin-top:7px;white-space:nowrap;font:13px var(--mono);color:#d9f7e6;text-align:center;background:rgba(4,12,8,.82);padding:2px 9px;border-radius:9px}
.multi .mcap b{color:#5dffa0} .multi .tagans{font:700 10.5px var(--mono);background:var(--acc);color:#04130b;border-radius:8px;padding:1px 6px;margin-left:5px;letter-spacing:.06em}
.multi.linking .box:not(.lk){opacity:.15} .multi .box{transition:opacity .15s}
.multi .mtip{position:absolute;left:50%;transform:translateX(-50%);font:13px var(--mono);color:#d9f7e6;background:rgba(4,12,8,.92);border:1px solid var(--line);border-radius:10px;
  padding:7px 14px;white-space:nowrap;opacity:0;transition:opacity .15s;pointer-events:none}
.multi .mtip.on{opacity:1}
.pupil.splitting{opacity:0!important;pointer-events:none}
#gauge,#irisgrp .glint,#irisgrp .irisbase,#irisgrp .ringglow{transition:opacity .35s ease}
body.splitview #gauge,body.splitview #irisgrp .glint,body.splitview #irisgrp .ringglow{opacity:0} body.splitview #irisgrp .irisbase{opacity:.35}
body.splitview #htitle{opacity:0!important;visibility:hidden}
.cmini{display:none;align-items:center;gap:6px;padding:6px 0 4px;overflow:hidden;cursor:pointer;min-height:132px}
.cmini .rc{flex:none;text-align:center} .cmini .rl{font:11.5px var(--mono);color:var(--dim);margin-top:2px;white-space:nowrap}
.cmini .rl.ans{color:var(--acc)} .cmini .more{flex:none;display:flex;flex-direction:column;align-items:center;gap:8px;font:13px var(--mono);color:var(--acc);white-space:nowrap}
.cmini .dots{display:flex;gap:4px} .cmini .dots i{width:6px;height:6px;border-radius:50%;background:var(--dim)} .cmini .msg{font:13px var(--mono);color:var(--dim)}
.cx-head{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.cx-head h2{margin:0} .cx-sub{color:var(--dim);font:13px var(--mono)}
.cx-i{margin-left:auto;width:30px;height:30px;border-radius:50%;border:1px solid var(--line);background:transparent;color:var(--txt);font:italic 700 15px Georgia,serif;cursor:pointer}
.cx-i:hover,.cx-i.on{border-color:var(--acc);color:var(--acc)}
.cx-info{margin:10px 0 4px;padding:14px 16px;border:1px solid var(--line);border-radius:12px;font:14px/1.6 var(--sans);color:var(--txt);background:rgba(127,127,127,.06)}
.cx-info p{margin:0 0 8px} .cx-info p:last-child{margin:0}
.cx-rings{display:flex;gap:14px;justify-content:center;flex-wrap:wrap;margin:18px 0 10px}
.cx-card{flex:1 1 0;min-width:130px;max-width:170px;text-align:center;padding:12px 8px 10px;border:1px solid var(--line);border-radius:16px;position:relative}
.cx-card.best{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc) inset}
.cx-card .tag{position:absolute;top:-9px;left:50%;transform:translateX(-50%);font:700 10.5px var(--mono);letter-spacing:.08em;padding:1px 8px;border-radius:9px;background:var(--acc);color:#04130b;white-space:nowrap}
.cx-card .nm{font:700 14px var(--mono);color:var(--txt);margin-top:4px} .cx-card .ms{font:12px var(--mono);color:var(--dim)}
.cx-card .ex{font:12px var(--mono);color:var(--dim);margin-top:2px}
.cx-tab{width:100%;table-layout:fixed;border-collapse:separate;border-spacing:0 6px;font:12.5px var(--mono)}
.cx-tab th:first-child{width:20%}
.cx-wrap{overflow-x:auto;scrollbar-width:thin;scrollbar-color:var(--line) transparent}
.cx-wrap::-webkit-scrollbar{height:6px} .cx-wrap::-webkit-scrollbar-thumb{background:var(--line);border-radius:3px} .cx-wrap::-webkit-scrollbar-track{background:transparent}
.cx-tab th{font:700 12px var(--mono);color:var(--dim);text-align:left;padding:0 8px 4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cx-tab td{padding:7px 8px;background:rgba(127,127,127,.06);vertical-align:middle}
.cx-tab td:first-child{border-radius:10px 0 0 10px;color:var(--txt);overflow-wrap:anywhere} .cx-tab td:last-child{border-radius:0 10px 10px 0}
.cx-tab tr.ans td:nth-child(2){box-shadow:inset 2px 0 0 var(--acc)}
.cx-chip{display:inline-flex;align-items:center;gap:6px;line-height:1.3}
.cx-chip span{overflow-wrap:anywhere}
.cx-chip b{flex:none;display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;font:700 12px var(--mono);color:#04130b}
.cx-more td{background:transparent;text-align:center;color:var(--dim);cursor:pointer;font:13px var(--mono)} .cx-more td:hover{color:var(--acc)}
.cx-legend{display:flex;gap:18px;flex-wrap:wrap;justify-content:center;margin-top:12px;font:12.5px var(--mono);color:var(--dim)}
.cmpnote{font:11.5px/1.4 var(--mono);color:var(--dim);margin:2px 0 6px}
.cmptab th{font:600 12px var(--mono);color:var(--dim);padding:4px 6px;border-bottom:1px solid var(--line)} .cmptab td{padding:3px 6px;font-size:12.5px;vertical-align:top}
.xbtn{flex:0 0 auto;align-self:flex-start;width:20px;height:20px;border-radius:6px;border:1px solid var(--line2);background:none;color:var(--acc);font:700 13px var(--mono);
  cursor:pointer;line-height:1;padding:0}
.xbtn:hover{background:rgba(57,255,136,.12)}
.xtab{width:100%;border-collapse:collapse;margin-top:6px;font:11px/1.45 var(--mono)}
.xtab td{padding:2px 4px;border-top:1px solid var(--line);vertical-align:top;word-break:break-word}
.xtab td:first-child{color:var(--dim);white-space:nowrap;width:1%}
.xtab td:last-child{color:var(--txt)}
.inav{display:flex;align-items:center;gap:6px;font:11.5px var(--mono);color:var(--dim)}
.inav .sp{flex:1}
.insp .cap{color:var(--txt)}
.sctl{display:flex;align-items:center;gap:5px;font:11.5px var(--mono);color:var(--dim);letter-spacing:.06em;text-transform:uppercase}
.sctl select,.sctl input{width:auto;padding:4px 6px;font-size:12px;border-radius:7px}
.stagebar{flex-wrap:wrap;row-gap:8px}
.legend span[data-tip]{cursor:help}
.bigview{width:min(1100px,95vw)}
.bigimg{width:100%;max-height:58vh;object-fit:contain;border-radius:10px;background:#fff}
.bignav{display:flex;align-items:center;gap:10px;margin:10px 0;font:12.5px var(--mono);color:var(--dim)}
.bignav .sp{flex:1}
.insp img{width:100%;flex:1;min-height:0;object-fit:contain;border-radius:9px;background:#fff}
.insp .cap{font:11.5px/1.45 var(--mono);color:var(--dim)}
.sweep{width:100%;flex:1}
.p4body{display:flex;flex-direction:column;gap:6px;min-height:0} .p4body #swtabs{align-self:flex-end} .p4body .sweep{min-height:0}
.glitch{animation:glitch .36s steps(1,end) 1}
@keyframes glitch{
  0%{clip-path:inset(0 0 0 0);translate:0}
  12%{clip-path:inset(8% 0 62% 0);translate:-4px 0;box-shadow:-3px 0 0 rgba(255,46,120,.65),3px 0 0 rgba(34,228,255,.65)}
  24%{clip-path:inset(58% 0 12% 0);translate:4px 0}
  36%{clip-path:inset(30% 0 44% 0);translate:-2px 1px;box-shadow:2px 0 0 rgba(255,46,120,.6),-2px 0 0 rgba(34,228,255,.6)}
  48%{clip-path:inset(72% 0 4% 0);translate:3px 0}
  60%{clip-path:inset(0 0 0 0);translate:-1px 0}
  100%{clip-path:inset(0 0 0 0);translate:0}}
/* ---------------------------------------------------------------- hints */
.hintdot{position:absolute;z-index:6;width:26px;height:26px;border-radius:50%;background:var(--acc);color:#02130a;font:700 13px var(--mono);
  display:flex;align-items:center;justify-content:center;box-shadow:0 0 18px rgba(57,255,136,.7);pointer-events:none;animation:bob 1.6s ease-in-out infinite;transition:opacity .6s}
.hintdot span{position:absolute;left:32px;white-space:nowrap;font:500 12px var(--mono);color:var(--acc);text-shadow:0 0 8px #000}
@keyframes bob{50%{transform:translateY(-4px)}}
.hidehints .hintdot{opacity:0}
/* ---------------------------------------------------------------- overlays (expanded pupil, stress, downloads, about) */
.ov{position:fixed;inset:0;z-index:30;background:rgba(1,3,4,.76);backdrop-filter:blur(5px);display:none;align-items:center;justify-content:center}
.ov.show{display:flex;animation:fade .2s ease-out}
@keyframes fade{from{opacity:0}to{opacity:1}}
.panel{background:var(--tile2);border:1px solid var(--line2);border-radius:18px;box-shadow:0 30px 90px rgba(0,0,0,.7),0 0 40px rgba(57,255,136,.08);animation:pop .28s cubic-bezier(.2,.9,.25,1.2)}
@keyframes pop{from{transform:scale(.94);opacity:0}to{transform:scale(1);opacity:1}}
.stage{width:min(1480px,96vw);height:min(920px,93vh);display:grid;grid-template-columns:1fr 390px;grid-template-rows:auto 1fr;gap:16px;padding:16px}
.stagebar{grid-column:1 / -1;display:flex;align-items:center;gap:12px;font:12.5px var(--mono);color:var(--dim)}
.stagebar h2{margin:0;font:600 13px var(--mono);letter-spacing:.24em;color:var(--acc)}.stagebar .sp{flex:1}
.figbox{position:relative;display:flex;align-items:center;justify-content:center;min-height:0;background:#010302;border-radius:12px;border:1px solid var(--line);overflow:hidden}
.fig{position:relative}.fig img{display:block;width:100%;height:100%}.fig svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.side{display:flex;flex-direction:column;gap:12px;min-height:0}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font:12px var(--mono);color:var(--dim)}
.legend span{display:flex;align-items:center;gap:6px}.legend i{width:16px;height:0;border-top:2px solid var(--c)}
.legend i.dash{border-top-style:dashed}.legend i.dots{border-top-style:dotted}
.ministats{font:12.5px/1.6 var(--mono);color:var(--dim)}.ministats b{color:var(--txt)}
.tabs{display:flex;gap:6px}
.btn{border:1px solid var(--line2);background:rgba(0,0,0,.3);color:var(--txt);border-radius:10px;padding:8px 12px;font:600 12px var(--mono);letter-spacing:.12em;
  text-transform:uppercase;cursor:pointer;text-decoration:none;text-align:center}
.btn:hover{border-color:var(--acc);box-shadow:0 0 14px rgba(57,255,136,.2)}
.btn.primary{background:linear-gradient(135deg,#39ff88,#16c47a);color:#02130a;border-color:transparent}
.btn:disabled{opacity:.45}
.dl{display:flex;gap:7px}.dl a{flex:1}
.close{background:none;border:1px solid var(--line);color:var(--dim);border-radius:9px;padding:6px 10px;font:12.5px var(--mono);cursor:pointer}
.close:hover{color:var(--txt);border-color:var(--acc)}
.small{width:min(900px,94vw);max-height:90vh;overflow:auto;padding:22px 24px}
.small h2{margin:0 0 12px;font:600 14px var(--mono);letter-spacing:.24em;color:var(--acc)}
.small p,.small li{font-size:14px;line-height:1.6;color:#bfe9d6}
.small table{width:100%;border-collapse:collapse;font:13px var(--mono);margin-top:6px}
.small td,.small th{border-bottom:1px solid var(--line);padding:6px 4px;text-align:left;vertical-align:top}.small th{color:var(--acc);font-weight:500}
.row{display:flex;align-items:center;gap:10px}
.prog{height:8px;border-radius:9px;background:rgba(255,255,255,.07);overflow:hidden;margin:12px 0}
.prog i{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--acc-d),var(--acc));transition:width .3s}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}
.kpi{border:1px solid var(--line);border-radius:12px;padding:12px;background:rgba(0,0,0,.25)}
.kpi .v{font:700 24px var(--mono)}.kpi .k{font:12px var(--mono);color:var(--dim);margin-top:4px}
kbd{font:600 11.5px var(--mono);border:1px solid var(--line2);border-radius:5px;padding:1px 5px;color:var(--acc)}
#tip{position:fixed;z-index:60;pointer-events:none;background:rgba(3,10,7,.95);border:1px solid var(--line2);border-radius:10px;padding:8px 10px;
  font:12.5px/1.5 var(--mono);color:var(--txt);max-width:300px;display:none;box-shadow:0 10px 30px rgba(0,0,0,.6)}
.htitle{position:fixed;z-index:40;pointer-events:none;font:500 15px var(--mono);letter-spacing:.24em;text-transform:uppercase;color:var(--acc);
  text-shadow:0 0 16px rgba(57,255,136,.7),0 0 2px #000;white-space:nowrap;opacity:0;transition:opacity .25s;padding:6px 12px;border-radius:10px;
  background:rgba(2,8,5,.55);border:1px solid rgba(57,255,136,.25);backdrop-filter:blur(4px)}
.htitle.show{opacity:1}
.top h1,.rtitle,#status{transition:opacity .2s}.top h1.away,.rtitle.away,#status.away{opacity:0}
.htitle::after{content:"_";margin-left:2px;animation:cur .8s steps(1) infinite}
@keyframes cur{50%{opacity:0}}
body.calm .htitle{transition:none}
#toast{position:fixed;left:50%;bottom:26px;transform:translateX(-50%);z-index:70;background:rgba(40,5,15,.95);border:1px solid #ff4f7a;color:#ffd9e2;
  padding:10px 16px;border-radius:10px;font:13px var(--mono);display:none}
.shock{position:fixed;z-index:50;width:20px;height:20px;margin:-10px 0 0 -10px;border-radius:50%;pointer-events:none;border:2px solid var(--acc);
  box-shadow:0 0 30px rgba(57,255,136,.8),inset 0 0 20px rgba(57,255,136,.5);animation:shock .7s ease-out forwards}
.shock.strong{border-width:3px;animation-duration:.9s}
@keyframes shock{from{transform:scale(.2);opacity:1}to{transform:scale(var(--s,22));opacity:0}}
.flash{position:fixed;inset:0;z-index:49;pointer-events:none;background:radial-gradient(circle at var(--x) var(--y),rgba(57,255,136,.18),transparent 45%);animation:fadeout .5s forwards}
@keyframes fadeout{to{opacity:0}}
/* wake-up intro: only the (closed) eye is visible until it has opened */
.top,.brow .seg,.brow .arc,.dock,.detrow,.review>*{transition:opacity .7s ease,translate .7s cubic-bezier(.2,.8,.2,1)}
body.intro .top,body.intro .brow .seg,body.intro .brow .arc,body.intro .dock,body.intro .detrow,body.intro .review>*,body.intro .hintdot{opacity:0;translate:0 16px;pointer-events:none}
.beam{transition:opacity 1.2s ease .5s} body.intro .beam{opacity:0}
body.intro .dock{translate:0 0}
body.intro #eyesvg,body.intro .pupil{animation:eyeglow 3.6s ease-in-out}   /* not on the lids' layer: a filter there hid the upper lid */
@keyframes eyeglow{0%{opacity:.35;filter:brightness(.6)}100%{opacity:1;filter:none}}
/* ================= light theme: only overrides; the dark styles above are unchanged ================= */
html.light{--bg:#d7ebdf;--tile:rgba(255,255,255,.93);--tile2:rgba(252,254,253,.98);--line:rgba(10,110,64,.22);--line2:rgba(10,130,74,.5);
  --acc:#0b9e56;--acc-d:#087a43;--cyan:#0094b5;--amber:#d48300;--mag:#c22ea1;--grey:#69768a;--gold:#c99a00;--txt:#10261c;--dim:#4b695b;--dim2:#8ba699}
html.light .zoomer,html.light .fig{--acc:#39ff88;--amber:#ffb020;--mag:#ff4fd8;--cyan:#22e4ff;--grey:#8b97a8;--gold:#ffd84a}   /* dark image areas keep the vivid colours */
html.light .dock,html.light .seg,html.light .tile{box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 2px 6px rgba(20,70,45,.08),0 12px 30px rgba(20,70,45,.14)}
html.light .seg:hover,html.light .tile:hover{box-shadow:inset 0 1px 0 rgba(255,255,255,.95),0 4px 10px rgba(20,70,45,.10),0 16px 38px rgba(20,70,45,.18),0 0 0 1px rgba(11,158,86,.22)}
html.light .tile::after,html.light .seg::after{content:none}
html.light .almond{filter:drop-shadow(0 6px 16px rgba(20,70,45,.22))}
html.light .irisline{stroke:rgba(225,255,238,.12)}
html.light .pupil{background:radial-gradient(circle at 50% 46%,#123323 0%,#081a12 62%,#030b07 100%);box-shadow:0 0 0 var(--pring,4px) rgba(150,240,190,.6),0 0 14px rgba(11,158,86,.3),inset 0 0 50px rgba(0,0,0,.55)}
html.light .pupil::before{content:"";position:absolute;inset:0;border-radius:50%;pointer-events:none;z-index:0;
  background:radial-gradient(ellipse 46% 26% at 34% 20%,rgba(255,255,255,.16),rgba(255,255,255,0) 70%)}
.lens{z-index:1}
html.light .pempty{color:#8fb3a1}
html.light .panel{box-shadow:0 24px 70px rgba(20,70,45,.18)}
html.light .chip,html.light .mbtn,html.light .btn,html.light .kpi{background:rgba(255,255,255,.75)}
html.light select,html.light input[type=text]{background:rgba(255,255,255,.9)}
html.light .chip.on{background:rgba(11,158,86,.12);box-shadow:none}
html.light .item{background:rgba(10,80,50,.045)} html.light .item.sel{background:rgba(11,158,86,.10)}
html.light .bar,html.light .prog{background:rgba(0,0,0,.08)}
html.light .logo .t1{text-shadow:0 0 14px rgba(11,158,86,.22)} html.light .logo .t2 i{text-shadow:none}
html.light .beam .bg1{stop-color:rgba(11,158,86,.16)} html.light .beam .bg2{stop-color:rgba(11,158,86,.05)} html.light .beam .rg1{stop-color:rgba(11,140,76,.45)}
html.light .browband{stroke:rgba(11,110,64,.22)} html.light .bf1{stop-color:rgba(11,120,70,.14)} html.light .bf2{stop-color:rgba(11,120,70,.05)}
html.light .hair{stroke:rgba(11,100,58,.30)} html.light .browedge{stroke:rgba(11,140,76,.55);filter:none}
html.light .lid .skin{fill:#e6eee9}
html.light .ov{background:rgba(232,240,235,.74)}
html.light .small p,html.light .small li{color:#2a4739}
html.light #tip{background:rgba(255,255,255,.97);color:var(--txt);box-shadow:0 10px 30px rgba(20,70,45,.16)}
html.light .htitle{background:rgba(255,255,255,.85);text-shadow:none}
html.light .hintdot span{text-shadow:0 0 6px #fff,0 0 2px #fff}
html.light .dbtn:hover{background:rgba(11,158,86,.07)}
body.calm *,body.calm *::before,body.calm *::after{animation:none!important;transition:none!important}
body.calm .det{opacity:1}
body.calm .scan,body.calm .cover,body.calm .shock,body.calm .flash{display:none}
@media (max-width:1100px){.review{grid-template-columns:repeat(2,1fr)} .main{padding-left:24px;padding-right:24px} .dock{position:static;transform:none;flex-direction:row;flex-wrap:wrap;justify-content:center;margin:10px auto}
  .dbtn{width:58px}}
/* ================= the journey: how the EYE was built (scoped to #jy) ================= */
#jy{    
  --bp:#173a63; --bp2:#1e4a7d; --bpline:rgba(255,255,255,.12); --chalk:#f4f7fb;
  --hivis:#f4c430; --hivis-ink:#3d2e00; --kraft:#efe2c4; --kraft2:#e4d3ac; --kraft-ink:#3b2f1c; --kraft-dim:#6f5d3d;
   --roadline:#f4c430; 
  --sans:"Barlow",system-ui,"Segoe UI",Roboto,sans-serif; --cond:"Barlow Condensed","Arial Narrow",system-ui,sans-serif; --stencil:"Saira Stencil One",Impact,"Arial Black",sans-serif;--page:#24272c;--page2:#1c1f23;--ink:#eef1f4;--ink2:#aab3bd;--line:rgba(238,241,244,.16);--road:#14171a;--shadow:0 18px 44px rgba(0,0,0,.45);color:var(--ink);font-family:var(--sans);font-size:16px;line-height:1.5;position:relative;z-index:2}
html.light #jy{--page:#e7e3da;--page2:#dcd7cb;--ink:#1d232b;--ink2:#4b5560;--line:rgba(29,35,43,.18);--road:#3a3f46;--shadow:0 18px 40px rgba(29,35,43,.18);}
#jy button{font:inherit;color:inherit}
#jy :focus-visible{outline:3px solid var(--hivis);outline-offset:3px}
#jy .wrap{max-width:1320px;margin:0 auto;padding:10px 0 36px}
#jy header{display:flex;align-items:flex-end;gap:24px;flex-wrap:wrap;margin-bottom:22px}
#jy header .titles{flex:1;min-width:280px}
#jy h1{font-family:var(--cond);font-weight:700;font-size:clamp(34px,5vw,58px);line-height:.95;margin:0;letter-spacing:-.01em}
#jy .sub{margin:10px 0 0;color:var(--ink2);max-width:62ch}
#jy .hdr-tools{display:flex;gap:8px}
#jy .tbtn{border:1.5px solid var(--line);background:transparent;border-radius:999px;padding:7px 14px;cursor:pointer;font-family:var(--cond);font-weight:600;letter-spacing:.02em}
#jy .tbtn:hover{border-color:var(--ink2)}
#jy /* station map */
.map{display:flex;align-items:center;gap:0;margin:6px 0 18px;overflow-x:auto;padding:6px 2px 10px;scrollbar-width:none}
#jy .map::-webkit-scrollbar{display:none}
#jy .mstop{display:flex;align-items:center;flex:1 0 auto}
#jy .mstop:last-child .mline{display:none}
#jy .mdot{width:38px;height:38px;border-radius:50%;border:2px solid var(--ink2);background:var(--page);display:grid;place-items:center;cursor:pointer;
  font-family:var(--stencil);font-size:15px;color:var(--ink2);transition:background .25s,color .25s,border-color .25s,transform .25s;flex:0 0 auto}
#jy .mdot:hover{border-color:var(--ink)}
#jy .mdot.on{background:var(--hivis);border-color:var(--hivis);color:var(--hivis-ink);transform:scale(1.12)}
#jy .mdot.done{border-color:var(--ink);color:var(--ink)}
#jy .mline{height:2px;flex:1;min-width:14px;background:repeating-linear-gradient(90deg,var(--ink2) 0 8px,transparent 8px 14px);margin:0 4px;opacity:.6}
#jy .mname{font-family:var(--cond);font-weight:600;font-size:14px;color:var(--ink2);margin-left:8px;white-space:nowrap;display:none}
#jy .mstop.on .mname{display:inline;color:var(--ink)}
#jy /* track */
.trackwrap{position:relative}
#jy .track{position:relative;display:flex;gap:0;overflow-x:auto;scroll-snap-type:x mandatory;scroll-behavior:smooth;padding:8px 0 0;scrollbar-width:thin}
#jy .stop{flex:0 0 auto;width:calc(min(380px,82vw) + 36px);padding-right:36px;scroll-snap-align:start;position:relative}
#jy .stop:last-child{width:min(380px,82vw);padding-right:0}
#jy .card{width:min(380px,82vw);height:550px;perspective:1600px;cursor:pointer;background:none;border:0;padding:0;text-align:left;display:block}
#jy .inner{position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform .7s cubic-bezier(.3,.7,.2,1)}
#jy .card.flipped .inner{transform:rotateY(180deg)}
#jy .face{position:absolute;inset:0;border-radius:14px;backface-visibility:hidden;-webkit-backface-visibility:hidden;overflow:hidden;box-shadow:var(--shadow)}
#jy /* front: blueprint */
.front{background-color:var(--bp);color:var(--chalk);display:flex;flex-direction:column;
  background-image:linear-gradient(var(--bpline) 1px,transparent 1px),linear-gradient(90deg,var(--bpline) 1px,transparent 1px);background-size:22px 22px}
#jy .front .tape{height:12px;background:repeating-linear-gradient(-45deg,var(--hivis) 0 12px,#1b1b1b 12px 24px)}
#jy .front .num{font-family:var(--stencil);font-size:64px;line-height:1;color:var(--hivis);margin:18px 22px 0}
#jy .front .scene{display:block;width:calc(100% - 28px);margin:6px 14px 0;height:auto}
#jy .front h2{font-family:var(--cond);font-weight:700;font-size:30px;line-height:1.05;margin:4px 22px 0;letter-spacing:.005em}
#jy .front .line{margin:8px 22px 0;font-size:17px;line-height:1.4;color:#d6e2f1;max-width:32ch}
#jy .front .hint{margin:auto 22px 18px;font-family:var(--cond);font-weight:600;font-size:14px;color:var(--hivis);letter-spacing:.03em;display:flex;align-items:center;gap:8px}
#jy .front .hint svg{width:16px;height:16px}
#jy .sc .w{stroke:var(--chalk);stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
#jy .sc .wf{fill:var(--chalk)}
#jy .sc .y{fill:var(--hivis)}
#jy .sc .ys{stroke:var(--hivis);stroke-width:2;fill:none;stroke-linecap:round}
#jy .sc .d{stroke:var(--chalk);stroke-width:1.5;fill:none;stroke-dasharray:4 4}
#jy .sc .t{fill:var(--chalk);font-family:var(--cond);font-weight:600;font-size:14px}
#jy .sc .ty{fill:var(--hivis-ink);font-family:var(--cond);font-weight:700;font-size:14px}
#jy .sc .faint{stroke:rgba(244,247,251,.35);stroke-width:1.5;fill:none}
#jy /* back: site report on kraft paper */
.back{transform:rotateY(180deg);background:var(--kraft);color:var(--kraft-ink);padding:20px 22px 18px;display:flex;flex-direction:column;
  background-image:repeating-linear-gradient(0deg,transparent 0 27px,rgba(59,47,28,.09) 27px 28px)}
#jy .back .bhead{display:flex;align-items:baseline;gap:12px;border-bottom:2px solid var(--kraft-ink);padding:18px 0 8px;margin-bottom:10px}
#jy .back .bnum{font-family:var(--stencil);font-size:30px;color:var(--kraft-ink)}
#jy .back h3{font-family:var(--cond);font-weight:700;font-size:23px;margin:0;line-height:1.1}
#jy .back ul{margin:0;padding:0 0 0 18px;font-size:14.5px;line-height:1.45}
#jy .back li{margin:0 0 6px}
#jy .back li::marker{color:var(--kraft-dim)}
#jy .back .figs{display:flex;gap:8px;margin-top:auto;padding-top:10px}
#jy .back .fig{flex:1;background:var(--kraft2);border-radius:8px;padding:8px 10px}
#jy .back .fig b{display:block;font-family:var(--cond);font-weight:700;font-size:22px;line-height:1.1}
#jy .back .fig span{font-size:13.5px;color:var(--kraft-dim);line-height:1.3;display:block}
#jy .back .why{margin:10px 0 0;font-size:13.5px;color:var(--kraft-dim);font-style:italic}
#jy .back .stamp{position:absolute;right:16px;top:14px;font-family:var(--cond);font-weight:700;font-size:13px;color:#8a2a1a;border:2px solid #8a2a1a;border-radius:6px;padding:2px 7px;transform:rotate(6deg);opacity:.75}
#jy /* road + truck */
.road{position:relative;height:58px;margin-top:22px;margin-right:-36px}
#jy .stop:last-child .road{margin-right:0}
#jy .road::before{content:"";position:absolute;left:0;right:0;top:16px;height:26px;background:var(--road);border-radius:4px}
#jy .road::after{content:"";position:absolute;left:0;right:0;top:28px;height:2px;background:repeating-linear-gradient(90deg,var(--roadline) 0 18px,transparent 18px 34px)}
#jy .post{position:absolute;left:calc(min(380px,82vw) / 2 - 14px);top:-6px;width:28px;height:28px;border-radius:50%;background:var(--hivis);color:var(--hivis-ink);
  font-family:var(--stencil);font-size:14px;display:grid;place-items:center;z-index:2;box-shadow:0 0 0 3px var(--page)}
#jy .truck{position:absolute;top:10px;left:0;width:64px;height:38px;z-index:3;filter:drop-shadow(0 0 8px var(--glow));transition:transform .9s cubic-bezier(.4,.1,.2,1);pointer-events:none}
#jy .nav{display:flex;align-items:center;gap:12px;margin-top:14px}
#jy .nbtn{width:46px;height:46px;border-radius:50%;border:2px solid var(--ink);background:transparent;cursor:pointer;display:grid;place-items:center}
#jy .nbtn:hover{background:var(--hivis);border-color:var(--hivis);color:var(--hivis-ink)}
#jy .nbtn svg{width:20px;height:20px}
#jy .count{font-family:var(--cond);font-weight:600;font-size:18px;min-width:80px}
#jy .keys{margin-left:auto;color:var(--ink2);font-size:14px}
#jy .keys kbd{font-family:var(--cond);font-weight:600;border:1.5px solid var(--line);border-radius:5px;padding:0 6px;margin:0 2px}
#jy footer{margin-top:26px;color:var(--ink2);font-size:14px;border-top:1.5px solid var(--line);padding-top:12px}
@media (prefers-reduced-motion: reduce){#jy .inner,#jy .truck,#jy .mdot{transition:none}
#jy .track{scroll-behavior:auto}}
@media (max-width:640px){#jy .card{height:640px}
#jy .keys{display:none}}
#jy{width:100%;max-width:1320px;margin-top:84px;scroll-margin-top:24px}
#jy header{flex-wrap:nowrap;align-items:center}
#jy .jyeye{flex:0 0 auto;width:clamp(120px,13vw,190px);opacity:0;transform:translateY(-70vh) scale(1.8)}
#jy .jyeye svg{display:block;width:100%;height:auto;overflow:visible}
#jy .jyeye.landed{animation:jyland 1.15s cubic-bezier(.3,1.3,.45,1) forwards}
@keyframes jyland{0%{opacity:0;transform:translateY(-70vh) scale(1.8)}55%{opacity:1}100%{opacity:1;transform:none}}
#jy .msc{fill:#0d2219} #jy .mout{fill:none;stroke:var(--acc);stroke-width:2.2;filter:drop-shadow(0 0 6px rgba(57,255,136,.55))}
#jy .ji1{stop-color:#062a18} #jy .ji2{stop-color:#0b4a2b} #jy .ji3{stop-color:#021a0e}
#jy .mring{fill:none;stroke:rgba(57,255,136,.55);stroke-width:5} #jy .mpup{fill:#030b07} #jy .mglint{fill:rgba(255,255,255,.75)}
#jy .mlid rect{fill:#071210} #jy .mlid.u{transform:translateY(-58px)} #jy .mlid.l{transform:translateY(58px)}
#jy .jyeye.blink .mlid.u{animation:jyblu .42s ease-in-out 2} #jy .jyeye.blink .mlid.l{animation:jybll .42s ease-in-out 2}
@keyframes jyblu{0%,100%{transform:translateY(-58px)}50%{transform:translateY(0)}} @keyframes jybll{0%,100%{transform:translateY(58px)}50%{transform:translateY(0)}}
#jy #jy-iris-g{transition:transform .25s ease-out}
html.light #jy .msc{fill:#f6faf7} html.light #jy .mout{stroke:#0b9e56;filter:drop-shadow(0 4px 10px rgba(20,70,45,.2))}
html.light #jy .ji1{stop-color:#0f5536} html.light #jy .ji2{stop-color:#1d7f4e} html.light #jy .ji3{stop-color:#0b452b}
html.light #jy .mring{stroke:rgba(150,240,190,.55)} html.light #jy .mlid rect{fill:#e6eee9}
#jy .more{flex:1 1 auto;min-height:0;overflow:auto;margin:4px 0 2px;padding-right:4px;font-size:14px;line-height:1.4;border-top:1.5px dashed rgba(59,47,28,.35);padding-top:6px}
#jy .more p{margin:6px 0 0}
#jy .mt{width:100%;border-collapse:collapse;font-size:13.5px}
#jy .mt th{text-align:left;font-family:var(--cond);font-weight:700;font-size:14px;border-bottom:1.5px solid var(--kraft-ink);padding:2px 4px}
#jy .mt td{padding:3px 4px;border-bottom:1px solid rgba(59,47,28,.18);vertical-align:top}
#jy .mt tr.win td{font-weight:600;background:rgba(244,196,48,.28)}
#jy .back a{color:var(--kraft-ink);text-decoration:underline;text-decoration-thickness:1.5px;text-underline-offset:2px}
#jy .back a:hover{color:#8a2a1a}
#jy .card{cursor:pointer}
@media (prefers-reduced-motion: reduce){#jy .jyeye{opacity:1;transform:none} #jy .jyeye.landed{animation:none} #jy .jyeye.blink .mlid{animation:none}}
body.calm #jy .jyeye{opacity:1;transform:none;animation:none}
#jy h1{font-size:clamp(26px,3.2vw,42px);line-height:1.12;max-width:34ch}
.howbtn{justify-self:start;border:1px solid var(--line2);background:var(--tile);color:var(--acc);border-radius:999px;padding:8px 14px;font:600 12.5px var(--mono);letter-spacing:.18em;text-transform:uppercase;cursor:pointer;backdrop-filter:blur(6px)}
.howbtn:hover{border-color:var(--acc);box-shadow:0 0 16px rgba(57,255,136,.25)}

/* ================= the journey, re-skinned in The EYE's design language (overrides) ================= */
#jy{--sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;--cond:Inter,system-ui,"Segoe UI",Roboto,sans-serif;--stencil:"JetBrains Mono",ui-monospace,Consolas,monospace;
  --ink:var(--txt);--ink2:var(--dim);--line:rgba(57,255,136,.18);--page:#04070a;
  --bp:#07130e;--bpline:rgba(57,255,136,.075);--chalk:#bff7d6;--hivis:#39ff88;--hivis-ink:#02130a;
  --kraft:#08140f;--kraft2:rgba(57,255,136,.07);--kraft-ink:#dcfff0;--kraft-dim:#7fa594;
  --road:#07110c;--roadline:#39ff88;--truckline:#02130a;--glow:rgba(57,255,136,.55);--shadow:0 18px 44px rgba(0,0,0,.5)}
html.light #jy{--ink:#10261c;--ink2:#4b695b;--line:rgba(10,110,64,.22);--page:#d7ebdf;
  --bp:#ffffff;--bpline:rgba(11,110,64,.07);--chalk:#0b6a3e;--hivis:#0b9e56;--hivis-ink:#ffffff;
  --kraft:#ffffff;--kraft2:rgba(11,158,86,.08);--kraft-ink:#10261c;--kraft-dim:#4b695b;
  --road:#cfe3d7;--roadline:#0b9e56;--truckline:#ffffff;--glow:rgba(11,158,86,.0);--shadow:inset 0 1px 0 rgba(255,255,255,.95),0 2px 6px rgba(20,70,45,.08),0 12px 30px rgba(20,70,45,.14)}
#jy h1{font:600 clamp(22px,2.6vw,34px)/1.18 var(--sans);letter-spacing:-.01em;color:var(--ink)}
#jy .sub{color:var(--ink2);font-size:14px}
#jy .tbtn{border:1px solid var(--line2);background:var(--tile);color:var(--acc);font:600 12px var(--mono);letter-spacing:.16em;text-transform:uppercase;padding:8px 14px;backdrop-filter:blur(6px)}
#jy .tbtn:hover{border-color:var(--acc);box-shadow:0 0 16px var(--glow)}
#jy .mdot{border:1px solid var(--line2);background:var(--tile2);color:var(--ink2);font:600 13px var(--mono)}
#jy .mdot:hover{border-color:var(--acc);color:var(--acc)}
#jy .mdot.on{background:var(--acc);border-color:var(--acc);color:var(--hivis-ink);box-shadow:0 0 16px var(--glow)}
#jy .mdot.done{border-color:var(--acc);color:var(--acc)}
#jy .mline{background:repeating-linear-gradient(90deg,var(--line2) 0 6px,transparent 6px 12px);opacity:1}
#jy .mname{font:600 11.5px var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--acc)}
#jy .face{border:1px solid var(--line)}
#jy .front{color:var(--chalk)}
#jy .front .tape{height:3px;background:linear-gradient(90deg,transparent,var(--hivis) 20%,var(--hivis) 80%,transparent);box-shadow:0 0 14px var(--glow)}
#jy .front .num{font:700 50px/1 var(--stencil);color:var(--hivis);text-shadow:0 0 18px var(--glow);margin:20px 22px 0}
#jy .front h2{font:600 24px/1.15 var(--sans);letter-spacing:-.01em;color:var(--kraft-ink)}
#jy .front .line{color:var(--kraft-dim);font-size:15px}
#jy .front .hint{font:600 12px var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--hivis)}
#jy .sc .t{font-family:var(--sans);font-weight:600;font-size:13px}
#jy .sc .ty{font-family:var(--sans);font-weight:700;font-size:13px}
#jy .sc .faint{stroke:var(--bpline);stroke-width:2}
#jy .back{background:var(--kraft);background-image:repeating-linear-gradient(0deg,transparent 0 27px,var(--bpline) 27px 28px);color:var(--kraft-ink)}
#jy .back .bhead{border-bottom:1px solid var(--line2)}
#jy .back .bnum{font:700 26px var(--stencil);color:var(--hivis);text-shadow:0 0 12px var(--glow)}
#jy .back h3{font:600 19px/1.2 var(--sans)}
#jy .back ul{font-size:13.5px}
#jy .back li::marker{color:var(--hivis)}
#jy .back .stamp{color:var(--hivis);border:1px solid var(--hivis);font:600 11px var(--mono);letter-spacing:.12em;text-transform:uppercase;opacity:.9}
#jy .back .fig{background:var(--kraft2);border:1px solid var(--line)}
#jy .back .fig b{font:700 18px var(--mono);color:var(--hivis)}
#jy .back .fig span{color:var(--kraft-dim)}
#jy .back .why{color:var(--kraft-dim)}
#jy .back a{color:var(--hivis)} #jy .back a:hover{color:var(--ink)}
#jy .more{border-top:1px dashed var(--line2)}
#jy .mt th{border-bottom:1px solid var(--line2);font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--hivis)}
#jy .mt td{border-bottom:1px solid var(--line)}
#jy .mt tr.win td{background:var(--kraft2);color:var(--hivis)}
#jy .road::before{background:var(--road);border:1px solid var(--line)}
#jy .road::after{background:repeating-linear-gradient(90deg,var(--roadline) 0 18px,transparent 18px 34px);filter:drop-shadow(0 0 4px var(--glow))}
#jy .post{background:var(--hivis);color:var(--hivis-ink);font:700 13px var(--mono);box-shadow:0 0 0 3px var(--page),0 0 14px var(--glow)}
#jy .nbtn{border:1px solid var(--line2);color:var(--acc);background:var(--tile)}
#jy .nbtn:hover{background:var(--acc);border-color:var(--acc);color:var(--hivis-ink);box-shadow:0 0 16px var(--glow)}
#jy .count{font:600 14px var(--mono);letter-spacing:.08em;color:var(--ink)}
#jy .keys{font-family:var(--mono);font-size:12.5px;letter-spacing:.04em}
#jy footer{border-top:1px solid var(--line);font-size:14px}
#jy .card:focus-visible{outline:2px solid var(--acc);outline-offset:4px;border-radius:14px}

body.intro .gauge path.plate,body.intro .detect .detlabel{opacity:0} body.intro .detect{pointer-events:none}
.gauge path.plate,.detect .detlabel{transition:opacity .6s ease .45s}
body.intro #lens{opacity:0} #lens{transition:opacity .5s ease}
body.intro #jy,body.intro .howbtn{opacity:0;pointer-events:none} #jy,.howbtn{transition:opacity .7s ease}


/* ===== v3.1: close buttons ===== */
.xclose{width:28px;height:28px;flex:0 0 28px;display:inline-flex;align-items:center;justify-content:center;border-radius:50%;border:1px solid var(--line2);
  background:var(--tile2);color:var(--dim);font:400 18px/1 var(--mono);cursor:pointer;padding:0;transition:color .15s,border-color .15s,background .15s}
.xclose:hover,.xclose:focus-visible{color:var(--acc);border-color:var(--acc);background:rgba(57,255,136,.08);outline:none}
.panel > .xclose{float:right;position:sticky;top:0;margin:-8px -10px 6px 14px;z-index:5}
.stagebar .xclose{margin-left:6px}
.panel > .xclose ~ h2{display:flow-root}

/* ===== v3.1: lean in on DETECT ===== */
/* v3.2 camera: the whole scene moves toward the pupil */
/* v3.3: the camera moves the pupil only (see focusIn) */
html{scrollbar-gutter:stable} html.focusing{overflow:hidden}
.brow,.review,.dock,.top,.detrow,.beam{transition:opacity .5s ease}
body.focus .brow,body.focus .dock,body.focus .top,body.focus .beam{opacity:.4} body.focus .review{opacity:.22} body.focus .detrow{opacity:0}
#fbanner{position:fixed;left:50%;bottom:20px;transform:translate(-50%,24px);z-index:25;width:min(760px,92vw);display:flex;align-items:center;gap:18px;
  padding:12px 14px 12px 22px;border-radius:16px;border:1px solid var(--acc);background:var(--tile2);box-shadow:0 18px 50px rgba(0,0,0,.55),0 0 24px rgba(57,255,136,.18);
  opacity:0;pointer-events:none;transition:opacity .35s ease,transform .45s cubic-bezier(.2,.8,.2,1)}
#fbanner.show{opacity:1;transform:translate(-50%,0);pointer-events:auto}
#fbanner .fb-score{font:700 34px/1 var(--mono);color:var(--acc);white-space:nowrap}
#fbanner .fb-text{flex:1;font:14px/1.45 var(--mono);color:var(--txt)}
#fbanner .fb-text b{color:var(--acc);font-weight:600}
#fbanner .fb-text .bad{color:var(--gold)}
html.light #fbanner{box-shadow:0 14px 40px rgba(20,70,45,.18)}
body.calm #main,body.calm .dock,body.calm #bg{transition:none}
body.focus #htitle{opacity:0 !important;visibility:hidden}
html,body{overflow-x:clip}


/* ===== v3.3: the verdict inside the eye, either side of the pupil ===== */
#fbanner{display:none !important}
#eyesvg{transition:opacity .5s ease} body.focus #eyesvg{opacity:0}
.pupil{transition:transform .35s cubic-bezier(.2,.9,.3,1.4),box-shadow .35s ease} body.focus .pupil{box-shadow:none;transition:transform .35s cubic-bezier(.2,.9,.3,1.4),box-shadow .35s ease .75s} html.light body.focus .pupil::before{opacity:0}
#fbL,#fbR{position:fixed;z-index:25;max-width:200px;opacity:0;pointer-events:none;transition:opacity .4s ease,transform .5s cubic-bezier(.2,.8,.2,1)}
#fbL{transform:translateX(-18px);text-align:center}
#fbR{transform:translateX(18px)}
#fbL.show,#fbR.show{opacity:1;transform:none;pointer-events:auto}
#fbL .fl-score{font:700 76px/1 var(--mono);color:var(--acc);letter-spacing:-.02em;text-shadow:0 0 22px rgba(57,255,136,.35)}
#fbL .fl-lbl{margin-top:8px;font:600 12.5px/1.4 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--dim)}
#fbL .fl-lbl b{color:var(--acc)} #fbL .fl-lbl .bad{color:var(--amber)}
#fbR .fr-h{font:600 12.5px/1 var(--mono);letter-spacing:.16em;text-transform:uppercase;color:var(--dim);margin:0 0 12px}
#fbR ul{list-style:none;margin:0;padding:0;font:17px/1.4 var(--mono);color:var(--txt)}
#fbR li{display:flex;align-items:flex-start;gap:10px;padding:4px 0;opacity:0;transform:translateX(10px);transition:opacity .3s ease,transform .35s ease}
#fbR.show li{opacity:1;transform:none}
#fbR li i{width:11px;height:11px;margin-top:7px;border-radius:50%;flex:0 0 11px;background:var(--acc);box-shadow:0 0 8px currentColor}
#fbR li b{font-weight:700;min-width:2.2ch;text-align:right}
#fbR li.missed i{background:var(--cyan)} #fbR li.wrong i{background:var(--mag)} #fbR li.off i{background:var(--amber)} #fbR li.extra i{background:var(--grey)}
#fbR li.missed b{color:var(--cyan)} #fbR li.wrong b{color:var(--mag)} #fbR li.off b{color:var(--amber)}
#fr-x{position:fixed;top:18px;right:18px;z-index:26;width:34px;height:34px;flex-basis:34px;font-size:20px;opacity:0;pointer-events:none;transition:opacity .3s ease}
#fr-x.show{opacity:1;pointer-events:auto}
html.light #fbL .fl-score{text-shadow:none}


/* ===== v3.3: a light camera move - heavy layers fade out first and are then skipped entirely ===== */
.brow,.review,.dock,.top,.beam,#jy{transition:opacity .2s ease,visibility 0s linear 0s}
body.focus .brow,body.focus .review,body.focus .dock,body.focus .top,body.focus .beam,body.focus #jy,body.focus .detrow{opacity:0 !important;visibility:hidden;transition:opacity .2s ease,visibility 0s linear .2s}
#eyesvg,#lidsvg{transition:opacity .3s ease,visibility 0s linear 0s}
body.focus #eyesvg,body.focus #lidsvg{opacity:0;visibility:hidden;transition:opacity .3s ease .6s,visibility 0s linear .9s}
body.focus .eyewrap{z-index:8}
#bg{transition:none}

#fpsbox{position:fixed;left:14px;bottom:12px;z-index:40;padding:6px 10px;border-radius:8px;background:rgba(0,0,0,.7);color:#39ff88;font:13px var(--mono)}
#d-sound .sx{display:none} #d-sound:not(.on) .sw1,#d-sound:not(.on) .sw2{display:none} #d-sound:not(.on) .sx{display:inline}
/* ===== v3.3: "How was I built?" reads as its own chapter ===== */
#jy{margin-top:110px;position:relative;padding:40px 30px 30px;border-radius:28px 28px 0 0;max-width:min(1320px,calc(100vw - 210px));
    background:linear-gradient(180deg,rgba(57,255,136,.07),rgba(57,255,136,.02) 140px,transparent 420px);border-top:1px solid rgba(57,255,136,.45);
    box-shadow:0 -1px 0 rgba(57,255,136,.25),0 -18px 40px -24px rgba(57,255,136,.45)}
html.light #jy{background:linear-gradient(180deg,rgba(11,158,86,.09),rgba(11,158,86,.03) 140px,transparent 420px);border-top-color:rgba(11,158,86,.5);box-shadow:0 -14px 34px -24px rgba(11,158,86,.45)}
.jytag{position:absolute;top:-13px;left:50%;transform:translateX(-50%);z-index:3;padding:4px 14px;border-radius:999px;font:600 12px var(--mono);letter-spacing:.18em;
       text-transform:uppercase;background:var(--bg);color:var(--acc);border:1px solid rgba(57,255,136,.45);white-space:nowrap;pointer-events:none}
html.light .jytag{border-color:rgba(11,158,86,.5)}
/* v3.3: the object's name under the score, for a single photo */
#fbL .fl-name{margin-top:14px;font:600 15px/1.3 var(--mono);color:var(--txt);display:none}
#fbL .fl-name.show{display:block} #fbL .fl-name small{display:block;margin-top:4px;font:13px/1.35 var(--mono);color:var(--dim)}
#fbL .fl-name .ok{color:var(--acc)} #fbL .fl-name .off{color:var(--amber)} #fbL .fl-name .wrong{color:var(--mag)} #fbL .fl-name .miss{color:var(--cyan)}
/* ===== v3.1: a calmer first glance (weights only; no element moves) ===== */
body.idle .review .tile .tb{opacity:.35}
.review .tile .tb{transition:opacity .45s ease}
.chip:not(.on){border-color:transparent;background:none;color:var(--dim2)}
.chip:not(.on):hover{color:var(--txt);border-color:var(--line2)}
html.light .chip:not(.on){background:none}
#bg{opacity:.45} .hair{opacity:.55} .beam{opacity:.55}
.almond{filter:drop-shadow(0 0 5px rgba(57,255,136,.35))} .browedge{filter:drop-shadow(0 0 3px rgba(57,255,136,.35))}
.lbl{color:var(--dim2)}
.dbtn{font-size:0;gap:0;min-height:58px;justify-content:center}
.dbtn:hover,.dbtn:focus-visible{font-size:11px;gap:5px}
</style>
</head>
<body class="intro idle">
<canvas id="bg"></canvas>

<nav class="dock" id="dock">
  <button class="dbtn" id="d-up" title="Upload an image [U]"><svg viewBox="0 0 24 24"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>upload</button>
  <button class="dbtn" id="d-rand" title="New random test grid [N]"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.4"/><circle cx="16" cy="16" r="1.4"/><circle cx="12" cy="12" r="1.4"/></svg>random</button>
  <button class="dbtn" id="d-cmp" title="COMPOUND view: the other strong models vote on every box (agrees / names another object / does not see it). It explains; it never changes the answer or the score."><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><circle cx="12" cy="5" r="2.2"/><circle cx="12" cy="19" r="2.2"/><circle cx="5.9" cy="8.5" r="2.2"/><circle cx="18.1" cy="8.5" r="2.2"/><circle cx="5.9" cy="15.5" r="2.2"/><circle cx="18.1" cy="15.5" r="2.2"/></svg>compound</button>
  <button class="dbtn" id="d-stress" title="Stress test"><svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>stress</button>
  <button class="dbtn" id="d-about" title="About"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>about</button>
  <input type="file" id="file" accept="image/*" hidden>
</nav>

<nav class="dock right" id="dock2">
  <button class="dbtn" id="d-dl" title="Downloads"><svg viewBox="0 0 24 24"><path d="M12 4v12M7 11l5 5 5-5M4 20h16"/></svg>files</button>
  <button class="dbtn" id="d-snap" title="Snapshot for reports"><svg viewBox="0 0 24 24"><rect x="3" y="6" width="18" height="14" rx="3"/><circle cx="12" cy="13" r="3.5"/><path d="M8 6l2-3h4l2 3"/></svg>snapshot</button>
  <button class="dbtn on" id="d-titles" title="Hover titles on / off"><svg viewBox="0 0 24 24"><path d="M5 6h14M12 6v13M8 19h8"/></svg>titles</button>
  <button class="dbtn" id="d-theme" title="Light / dark theme"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg><span id="themelbl">light</span></button>
  <button class="dbtn on" id="d-sound" title="Sound on or off"><svg viewBox="0 0 24 24"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path class="sw1" d="M16 9.5a3.5 3.5 0 0 1 0 5"/><path class="sw2" d="M18.5 7a7 7 0 0 1 0 10"/><path class="sx" d="M16 9l5 6M21 9l-5 6"/></svg>sound</button>
  <button class="dbtn" id="d-calm" title="Calm mode: no motion"><svg viewBox="0 0 24 24"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z"/></svg>calm</button>
</nav>

<main class="main" id="main">
  <svg class="beam" id="beam" aria-hidden="true"></svg>
  <div class="top"><button class="howbtn" id="howbtn" type="button">How was I built?</button><h1 class="logo"><span class="t1">The <b>EYE</b></span><span class="t2"><i>E</i>xplainable <i>Y</i>OLO <i>E</i>xplorer</span></h1><span class="meta" id="meta"><span id="m-obj" data-tip="">-</span> · <span id="m-photos" data-tip="">-</span> · <span id="m-boxed" data-tip="">-</span></span></div>

  <section class="brow" id="brow">
    <svg class="arc" id="arcsvg"><defs><linearGradient id="browgrad" x1="0" x2="1"><stop offset="0" stop-color="#39ff88" stop-opacity="0"/>
      <stop offset=".5" stop-color="#39ff88"/><stop offset="1" stop-color="#39ff88" stop-opacity="0"/></linearGradient></defs><path class="arcl" id="arcpath"/></svg>
    <div class="seg" data-i="0" id="s-grid"><h3>Grid <span>build</span></h3><div class="chips" id="sizes"></div><div class="lbl" id="sizeinfo"></div></div>
    <div class="seg" data-i="1" id="s-mix"><h3>Mix</h3><div class="chips" id="secs"></div><div class="chips" id="mixes"></div><div class="chips" id="layouts" title="layout of the test image"></div></div>
    <div class="seg" data-i="2" id="s-cond"><h3>Conditions</h3><div class="chips" id="degs"></div></div>
    <div class="seg" data-i="3" id="s-seed"><h3>Seed</h3><input type="text" id="seed" placeholder="empty = random">
      <div class="mini"><button class="mbtn" id="dice">random</button><button class="mbtn" id="keep">keep last</button></div><div class="lbl" id="lastseed"></div></div>
    <div class="seg" data-i="4" id="s-model"><h3>Model</h3><select id="model"></select><div class="lbl" id="mapv"></div></div>
    <div class="seg" data-i="5" id="s-thr"><h3>Threshold <span id="thrv">0.10</span></h3><input type="range" id="conf" min="0.05" max="0.9" step="0.05">
      <label class="toggle"><input type="checkbox" id="cell" checked>one box per cell</label><button class="mbtn" id="resetthr">reset to validated</button><div class="thrkind" id="thrkind"></div></div>
  </section>

  <div id="fbL" aria-hidden="true"><div class="fl-score" id="fl-score"></div><div class="fl-lbl" id="fl-lbl"></div><div class="fl-name" id="fl-name"></div></div>
  <div id="fbR" role="status" aria-live="polite"><div class="fr-h">results</div><ul id="fr-list"></ul></div>
  <button class="xclose inbar" id="fr-x" type="button" title="Close (Esc)" aria-label="Close">×</button>
  <div id="fbanner" role="status" aria-live="polite"><div class="fb-score" id="fb-score"></div><div class="fb-text" id="fb-text"></div>
    <button class="xclose inbar" id="fb-x" type="button" title="Close (Esc)" aria-label="Close">×</button></div>
  <section class="eyewrap" id="eye">
    <svg class="eyesvg" id="eyesvg"></svg>
    <div class="pupil" id="pupil"><div class="logo intrologo" id="intrologo"><span class="t1"><span class="lt" style="--d:0s">T</span><span class="lt" style="--d:.09s">h</span><span class="lt" style="--d:.18s">e</span> <b><span class="lt ee" style="--d:.36s">E</span><span class="lt" style="--d:.45s">Y</span><span class="lt ee" style="--d:.54s">E</span></b></span><span class="t2"><i>E</i>xplainable <i>Y</i>OLO <i>E</i>xplorer</span></div><div class="lens" id="lens"><div class="pempty" id="pempty"><div>press <b>random</b> or <kbd>N</kbd><br>to build a test grid</div></div></div></div>
    <svg class="lidsvg" id="lidsvg"></svg>
    <div class="dzhint" id="dzhint">drop the image into the eye</div>
  </section>

  <div class="detrow"><div class="status" id="status">warming up the model...</div></div>

  <section class="review" id="review">
    <div class="tile" id="t-score"><div class="th"><span class="ix">01</span><h3>Score</h3><span class="sp"></span><span class="hint" id="ms"></span></div>
      <div class="tb"><div class="score"><span class="big" id="sc">-</span><span class="of" id="scof"></span></div><div class="bar"><i id="scbar"></i></div>
        <div class="verdict" id="verdict"></div><div class="stats" id="stats"></div></div></div>
    <div class="tile" id="t-err"><div class="th"><span class="ix">02</span><h3>Results</h3><span class="sp"></span>
        <span class="ftabs" id="errtabs"><span class="chip sm on" data-f="err">errors</span><span class="chip sm" data-f="all">all</span></span></div>
      <div class="tb"><div class="list" id="errs"><div class="lbl">one line per object appears here</div></div></div></div>
    <div class="tile" id="t-insp"><div class="th"><span class="ix">03</span><h3>Inspect</h3><span class="sp"></span><button class="enl" id="insp-big" title="enlarge">⤢</button></div>
      <div class="tb"><div class="insp" id="insp"><div class="lbl">click an object to compare it here</div></div></div></div>
    <div class="tile" id="t-sweep"><div class="th"><span class="ix">04</span><h3 id="p4title">Threshold sweep</h3><span class="sp"></span><button class="enl" id="sweep-big" title="enlarge">⤢</button></div>
      <div class="tb p4body"><span class="ftabs" id="swtabs"><span class="chip sm" data-f="cmp" title="COMPOUND: how every model does on this image">compound</span><span class="chip sm on" data-f="sweep">sweep</span><span class="chip sm" data-f="tight" title="how tightly each box fits its object (IoU)">tightness</span></span><svg class="sweep" id="sweep" viewBox="0 0 360 150" preserveAspectRatio="none"></svg><div class="cmini" id="cmpmini"></div></div></div>
  </section>
  <section id="jy" aria-label="How the EYE was built"><div class="jytag" aria-hidden="true">How was I built?</div>
<div class="wrap">
  <header>
    <div class="jyeye" id="jy-eye" aria-hidden="true"><svg viewBox="0 0 200 110">
      <defs><clipPath id="jy-alm"><path d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/></clipPath>
        <radialGradient id="jy-iris" cx="50%" cy="50%" r="50%"><stop offset=".72" class="ji1"/><stop offset=".9" class="ji2"/><stop offset="1" class="ji3"/></radialGradient></defs>
      <path class="msc" d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/>
      <g clip-path="url(#jy-alm)"><g id="jy-iris-g"><circle cx="100" cy="55" r="42" fill="url(#jy-iris)"/>
        <circle class="mring" cx="100" cy="55" r="42"/><circle class="mpup" cx="100" cy="55" r="30"/><circle class="mglint" cx="113" cy="41" r="4"/></g>
        <g class="mlid u"><rect x="0" y="-60" width="200" height="115"/></g><g class="mlid l"><rect x="0" y="55" width="200" height="115"/></g></g>
      <path class="mout" d="M2 55 C46 -10,154 -10,198 55 C154 120,46 120,2 55 Z"/></svg></div>
    <div class="titles">
      <h1>&ldquo;For simplicity, and since you don&rsquo;t know much biology and human anatomy, I will tell you how I was built as if I were a building.&rdquo;</h1>
      <p class="sub">Milestones 2 and 3, Group 7: from 7,343 raw files to a detector for 73 everyday objects in any layout, explained object by object. Twelve stations, left to right. Flip any card for the full site report.</p>
    </div>
    <div class="hdr-tools">
      <button class="tbtn" id="jy-flipall" type="button">Flip all</button>
      <button class="tbtn" id="jy-up" type="button">Back to the eye</button>
      <button class="xclose inbar" id="jy-x" type="button" title="Close (back to the eye)" aria-label="Close">×</button>
    </div>
  </header>

  <nav class="map" id="jy-map" aria-label="Stations"></nav>

  <div class="trackwrap">
    <div class="track" id="jy-track" tabindex="0" aria-label="Journey stations"></div>
  </div>

  <div class="nav">
    <button class="nbtn" id="jy-prev" type="button" aria-label="Previous station"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg></button>
    <button class="nbtn" id="jy-next" type="button" aria-label="Next station"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg></button>
    <span class="count" id="jy-count">1 / 10</span>
    <span class="keys"><kbd>←</kbd><kbd>→</kbd> move · <kbd>Enter</kbd> flip</span>
  </div>

  <footer>IE7615 Deep Learning, Northeastern University. Detector: boost, a YOLO26-s fine-tuned from the Milestone 2 model at 1,280 px, chosen on validation by a rule sealed in advance and tested on boxes checked by a person.</footer>
</div>
  </section>
</main>

<div class="ov" id="ov-stage"><div class="panel stage">
  <div class="stagebar"><h2 id="stitle">PUPIL</h2>
    <label class="sctl">grid <select id="sc-size"></select></label>
    <label class="sctl">seed <input type="text" id="sc-seed" placeholder="random" size="7"></label><button class="close" id="sc-dice" title="empty seed = a new random grid">random seed</button>
    <label class="sctl">model <select id="sc-model"></select></label>
    <label class="sctl">threshold <select id="sc-thr"></select></label>
    <span class="sp"></span><span id="ssub"></span>
    <button class="btn" id="s-new">Random</button><button class="btn primary" id="s-det">Detect</button><button class="close" id="s-close">close [esc]</button></div>
  <div class="figbox" id="figbox"><div class="fig" id="fig"></div></div>
  <div class="side">
    <div class="legend" id="legend"><span class="b-correct" data-tip="Correct: right Object ID, and the box overlaps the true box by at least half (IoU ≥ 0.5)"><i></i>correct</span>
      <span class="b-box-off" data-tip="Box off: right Object ID, but the box overlaps the true box by less than half (IoU < 0.5)"><i class="dash"></i>box off</span>
      <span class="b-wrong-ID" data-tip="Wrong ID: the box is in the right place, but names a different object (look-alikes are flagged)"><i></i>wrong ID</span>
      <span class="b-extra" data-tip="Extra: a box that matches no true object (a second box, a partial box, background, or an unlabelled object in the photo)"><i class="dots"></i>extra</span>
      <span class="b-missed" data-tip="Missed: no box for this true object; the reason says whether it was seen below the threshold or is very small"><i class="dash"></i>missed</span>
      <span style="--c:rgba(255,255,255,.6)" data-tip="True box: the object's correct box, checked by a person"><i class="dash"></i>true box</span></div>
    <div class="ministats" id="sstats"></div>
    <div class="tabs"><span class="chip on" data-f="all">all</span><span class="chip" data-f="err">errors</span></div>
    <div class="list" id="srows"></div><div class="dl" id="sdl"></div></div></div></div>

<div class="ov" id="ov-big"><div class="panel small bigview" id="bigpanel"></div></div>

<div class="ov" id="ov-stress"><div class="panel small"><h2>STRESS TEST</h2>
  <p>Runs many random grids of held-out test photos with the settings on the eyebrow (grid, mix, conditions, model, threshold) and reports the totals. Grids use seeds 1000, 1001, ... so a run can be repeated exactly.</p>
  <div class="row"><span class="lbl">GRIDS</span><input type="range" id="sn" min="10" max="300" step="10" value="100"><span class="lbl" id="snv">100</span></div>
  <div class="row" style="margin-top:10px"><button class="btn primary" id="srun">Run</button><button class="close" id="sclose">close</button></div>
  <div class="prog"><i id="sprog"></i></div><div id="sres"></div></div></div>

<div class="ov" id="ov-dl"><div class="panel small"><h2>FILES</h2><p>The last detection, in the milestone's output format, plus a snapshot image for reports and slides.</p>
  <div class="dl" id="dls"></div><div class="row" style="margin-top:14px"><button class="close" id="dlclose">close</button></div></div></div>

<div class="ov" id="ov-about"><div class="panel small"><h2>ABOUT</h2>
  <p>The detector reports the <b>Object ID and bounding box</b> of every object. Random grids and the stress test use only <b>held-out test photos</b> whose true boxes were checked by a person.</p>
  <p><b>How to use:</b> ① set up the test on the eyebrow · ② press <b>RANDOM</b> (or upload) and <b>DETECT</b> · ③ review below · ④ press <b>COMPOUND</b> in the left dock to break the pupil into every model's pupil. Click the pupil for the large view; click an error to zoom onto it.</p>
  <table><tr><th>result</th><th>meaning</th></tr>
  <tr><td style="color:var(--acc)">correct</td><td>right Object ID and the box overlaps the true box by at least half (IoU >= 0.5)</td></tr>
  <tr><td style="color:var(--amber)">box off</td><td>right Object ID, but the box overlaps by less than half</td></tr>
  <tr><td style="color:var(--mag)">wrong ID</td><td>box in the right place, another object named (look-alikes flagged)</td></tr>
  <tr><td style="color:var(--cyan)">missed</td><td>no box; the reason says whether it was seen below the threshold or is very small</td></tr>
  <tr><td style="color:var(--grey)">extra</td><td>a box matching no object</td></tr></table>
  <h2 style="margin-top:18px">THE EXTENSION</h2>
  <p>After Milestone 2, <b>every usable box was checked by a person</b> (two passes, near-duplicate copies removed), and the detector was fine-tuned again on
  <b>rotated grids, photos scattered on real backgrounds and objects cut out with SAM</b>, at a higher input resolution. The <b>layout</b> chips in the Mix panel
  show these test scenes; their objects come from held-out test photos on unseen backgrounds. Each model keeps its own threshold chosen on validation,
  one for grids and one for scenes; drop any image onto the eye to try it.</p>
  <h2 style="margin-top:18px">BEFORE VS AFTER</h2><div id="aboutboost"></div>
  <h2 style="margin-top:18px">COMPOUND VIEW (v3.4)</h2>
  <p>The <b>compound</b> switch in the left dock shows how every model does on the same image: boost, boost2, the Milestone 2 model and a YOLOv8-s, plus the Milestone 1 classifier on each grid cell when its weights are present. On images with true boxes every model is judged against the truth, like the answer, and shown as a score ring and a row per object; on uploads the other models are compared with the answer's boxes. Press <b>compound</b> after a detection and the pupil breaks into one pupil per model, each ringed by its score, and panel 04 opens on its compound tab: hover a box to follow that object through every pupil, click a pupil to open it. Press compound again (or C, Esc, RANDOM, upload or DETECT) to return to the single pupil. It explains; it never changes the answer or the score. Combining the models into one answer was tested on validation under a sealed rule and did not pass (better grids, worse cut-out scenes and more boxes on empty backgrounds), so it stays a view.</p>
  <h2 style="margin-top:18px">MODELS</h2><div id="aboutmodels"></div>
  <h2 style="margin-top:18px">BY INPUT TYPE (MILESTONE 2 MODEL)</h2><div id="aboutshapes"></div>
  <p>The <b>iris</b> is the score: one segment per object, coloured by its result. After DETECT the view moves into the pupil to show the result; <kbd>Esc</kbd> or a click returns. Keys: <kbd>N</kbd> random grid · <kbd>D</kbd> detect · <kbd>U</kbd> upload · <kbd>R</kbd> replay the opening · <kbd>Esc</kbd> close or zoom out.</p>
  <div class="row"><button class="close" id="aclose">close</button></div></div></div>

<div id="tip"></div><div id="toast"></div><div class="htitle" id="htitle"></div>

<script>
"use strict";
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const st = { cfg: null, model: null, conf: .1, confAuto: true, confKind: "grid", cell: true, size: "3x3 (9 objects)", sections: [], mix: "Random", layout: "grid", degrade: "None",
             grid: null, det: null, sel: null, filter: "all", lastSeed: null, zoomed: null, busy: false, warm: "loading", hinted: false,
             cmpOn: false, cmp: null, cmpBusy: false };
const calm = () => document.body.classList.contains("calm");
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const cls = r => String(r || "detected").replace(/ /g, "-");
const isLight = () => document.documentElement.classList.contains("light");
const PAL = {
  dark:  { bg: "#04070a", px: ["57,255,136", "34,228,255", "220,255,240"], pxA: 1, pxS: 0, speed: 1, scl: ["#0d2219", "#06120d", "#030806"],
           iris: ["#062a18", "#0b4a2b", "#0b4a2b", "#021a0e"], lidShade: false,
           res: { "correct": "#39ff88", "detected": "#39ff88", "box off": "#ffb020", "wrong ID": "#ff4fd8", "missed": "#22e4ff", "extra": "#8b97a8" },
           sw: { bg: "rgba(0,0,0,.25)", text: "#7fa594", hint: "#4d6b5e", cur: "#ffd84a" } },
  light: { bg: "#d7ebdf", px: ["8,96,52", "0,98,118", "30,70,50"], pxA: 1.5, pxS: 0, speed: 2.2, scl: ["#ffffff", "#eef4f0", "#cfddd5"],
           iris: ["#0f5536", "#1d7f4e", "#43a873", "#0b452b"], lidShade: true,
           res: { "correct": "#0bb862", "detected": "#0bb862", "box off": "#e08a00", "wrong ID": "#d02fab", "missed": "#0096b8", "extra": "#69768a" },
           sw: { bg: "rgba(0,0,0,.04)", text: "#4b695b", hint: "#8ba699", cur: "#c99a00" } } };
const pal = () => PAL[isLight() ? "light" : "dark"];
const RCOL = new Proxy({}, { get: (_, k) => PAL.dark.res[k] });       // the iris is dark in both themes: vivid colours
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json(); if (!r.ok || j.error) throw new Error(j.error || r.statusText); return j;
}
function toast(m) { const t = $("#toast"); t.textContent = m; t.style.display = "block"; clearTimeout(t._h); t._h = setTimeout(() => t.style.display = "none", 4200); }
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ================================================================ pixel field */
const BG = (() => {
  const cv = $("#bg"), cx = cv.getContext("2d"); let W, H, px = [], waves = [], mouse = { x: -999, y: -999 }, running = true; const t0 = performance.now();
  function init() {
    const dpr = Math.min(2, devicePixelRatio || 1); W = innerWidth; H = innerHeight; cv.width = W * dpr; cv.height = H * dpr; cx.setTransform(dpr, 0, 0, dpr, 0, 0);
    px = []; for (let i = 0, n = Math.round(W * H / 460); i < n; i++) px.push({ x: Math.random() * W, y: Math.random() * H, s: Math.random() < .8 ? 2 : 3,
      a: .05 + Math.random() * .28, p: Math.random() * 6.28, f: .4 + Math.random() * 1.6, v: .02 + Math.random() * .08,
      ci: Math.random() < .9 ? 0 : (Math.random() < .5 ? 1 : 2) });
    draw(performance.now());
  }
  function draw(now) {
    const t = (now - t0) / 1000, P = pal(); cx.fillStyle = P.bg; cx.fillRect(0, 0, W, H); waves = waves.filter(w => t - w.t < 1.6);
    for (const q of px) {
      if (!calm()) { q.y += q.v * P.speed; if (q.y > H) { q.y = 0; q.x = Math.random() * W; } }
      let x = q.x, y = q.y, a = q.a * (calm() ? 1 : .65 + .35 * Math.sin(t * q.f + q.p));
      const dm = Math.hypot(x - mouse.x, y - mouse.y); if (dm < 110) a += (1 - dm / 110) * .55;
      for (const w of waves) { const r = (t - w.t) * w.v, d = Math.hypot(x - w.x, y - w.y), b = Math.abs(d - r);
        if (b < 48) { const k = (1 - b / 48) * (1 - (t - w.t) / 1.6) * w.amp; x += (x - w.x) / (d || 1) * k * 14; y += (y - w.y) / (d || 1) * k * 14; a += k * .7; } }
      cx.fillStyle = `rgba(${P.px[q.ci]},${Math.min(1, a * P.pxA).toFixed(3)})`; cx.fillRect(x, y, q.s + P.pxS, q.s + P.pxS);
    }
    if (running && !calm()) requestAnimationFrame(draw);
  }
  addEventListener("resize", init); addEventListener("mousemove", e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  document.addEventListener("visibilitychange", () => { running = !document.hidden; if (running) requestAnimationFrame(draw); });
  return { init, wave(x, y, amp = 1) { waves.push({ x, y, t: (performance.now() - t0) / 1000, v: 900, amp }); }, restart() { calm() ? draw(performance.now()) : requestAnimationFrame(draw); },
    pause(ms) { running = false; clearTimeout(this._t); this._t = setTimeout(() => { running = true; if (!calm()) requestAnimationFrame(draw); }, ms); } };
})();

/* ================================================================ glitch + shock */
function glitchable(el) { el.addEventListener("mouseenter", () => { if (calm() || el.classList.contains("glitch")) return; el.classList.add("glitch"); setTimeout(() => el.classList.remove("glitch"), 380); }); }
function shock(target, strong = false) {
  if (calm() || !target) return; const r = target.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
  const s = document.createElement("div"); s.className = "shock" + (strong ? " strong" : ""); Object.assign(s.style, { left: x + "px", top: y + "px" });
  s.style.setProperty("--s", strong ? 34 : 22); document.body.appendChild(s); setTimeout(() => s.remove(), 1000);
  const f = document.createElement("div"); f.className = "flash"; f.style.setProperty("--x", x + "px"); f.style.setProperty("--y", y + "px"); document.body.appendChild(f); setTimeout(() => f.remove(), 600);
  BG.wave(x, y, strong ? 1.3 : .8);
  $$(".seg,.tile,.dock").forEach(t => { const b = t.getBoundingClientRect(), dx = b.left + b.width / 2 - x, dy = b.top + b.height / 2 - y, d = Math.hypot(dx, dy) || 1;
    const k = Math.max(0, 1 - d / 950) * (strong ? 14 : 9);
    t.animate([{ translate: "0 0" }, { translate: `${dx / d * k}px ${dy / d * k}px` }, { translate: "0 0" }], { duration: 520, delay: d / 2.2, easing: "cubic-bezier(.2,.9,.3,1)" }); });
}

/* ================================================================ geometry: eyebrow + eye */
let G = {};
function layout() {
  const avail = Math.min(innerWidth - (innerWidth > 1100 ? 250 : 48), 1040), EW = Math.max(560, avail), EH = Math.round(EW * .47);
  document.documentElement.style.setProperty("--ew", EW + "px"); document.documentElement.style.setProperty("--eh", EH + "px");
  const cx = EW / 2, cy = EH / 2, RI = EH * .5, RP = RI * .76, GW = RI * .1, RG = (RI + RP) / 2 + GW * .1;
  G = { EW, EH, cx, cy, RI, RP, GW, RG, LENS: Math.floor(RP * Math.SQRT2 * .98) };
  // brow: one tapered band that follows an arch, hair strokes, the six controls set into it along the curve
  const brow = $("#brow"), segs = $$(".seg"), n = segs.length, gap = 9;
  const BW = Math.round(Math.min(EW * 1.2, innerWidth - (innerWidth > 1100 ? 240 : 24)));
  brow.style.width = BW + "px";
  const U = .78, sw = (U * BW - gap * (n - 1)) / n;                                    // the panels cover |u| <= U
  segs.forEach(sg => { sg.style.width = sw + "px"; sg.style.transform = "none"; sg.style.height = ""; });   // v3: natural height, re-measured every time
  /* full box, borders included (height is unset here) */ const nat = segs.map(sg => sg.offsetHeight).sort((a, b) => b - a), SEG_MAX = 176;
  const segH = Math.min(nat[0], Math.max(SEG_MAX, nat[1] || 0)), A = Math.round(EW * .09), half = BW / 2;
  const prof = u => Math.pow(Math.max(0, 1 - Math.pow(Math.abs(u), 3.4)), .5);          // thickness profile: full through the middle, tapering tails
  const PAD = 13, uOut = (U * BW / 2 - sw / 2) / half;
  const yTop = u => 6 + A * u * u;
  // panels hang from the top edge, so the band must reach PAD below every panel's lowest corner
  // even thickness along the body (just enough for the tilted panels), smooth taper in the tails
  const ext = segs.map((_, i) => { const xc = half - U * BW / 2 + sw / 2 + i * (sw + gap), u = (xc - half) / half, a = Math.min(.21, Math.abs(Math.atan(2 * A * u / half)));
    return Math.abs(Math.sin(a)) * sw + Math.cos(a) * segH; });
  const T0 = PAD * 2 + segH / 2 + Math.max(...ext) / 2 - 4, DROP = 18;           // the lower edge curves down a little more towards the tails
  const body = v => T0 + DROP * Math.min(1, v / U) ** 2;
  const th = u => { const v = Math.abs(u); if (v <= U) return body(v); const k = (v - U) / (1 - U); return Math.max(1.5, body(U) * Math.pow(.5 + .5 * Math.cos(Math.PI * Math.min(1, k)), .85)); };
  const Tc = T0, yMid = u => yTop(u) + PAD + segH / 2;
  const H = Math.ceil(Math.max(6 + Tc, A + 18)) + 4; brow.style.height = H + "px";
  brow.style.marginBottom = Math.round(6 + Tc + 14 - H) + "px";            // the eye's top sits 14 px below the brow's lower edge
  segs.forEach((sg, i) => {
    const xc = half - U * BW / 2 + sw / 2 + i * (sw + gap), u = (xc - half) / half, du = .01;
    const slope = (yTop(u + du) - yTop(u - du)) / (2 * du * half), ang = Math.max(-12, Math.min(12, Math.atan(slope) * 180 / Math.PI));
    Object.assign(sg.style, { left: (xc - sw / 2) + "px", top: (yMid(u) - segH / 2) + "px", height: segH + "px", transform: `rotate(${ang}deg)` });
    sg.classList.toggle("scrolls", sg.scrollHeight > sg.clientHeight + 1);   // a panel taller than the band scrolls inside
  });
  const N = 90, topPts = [], botPts = [];
  for (let k = 0; k <= N; k++) { const u = -1 + 2 * k / N, x = half + u * half; topPts.push(`${x.toFixed(1)},${yTop(u).toFixed(1)}`); botPts.unshift(`${x.toFixed(1)},${(yTop(u) + th(u)).toFixed(1)}`); }
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;     // same hairs on every resize
  let hairs = "";
  for (let k = 0; k < 420; k++) {
    const u = (rnd() * 2 - 1) * .985, t = th(u); if (t < 3) continue;
    const x = half + u * half, y = yTop(u) + (.12 + .76 * rnd()) * t, len = 7 + rnd() * 11 * (.4 + .6 * t / T0);
    const base = Math.atan(2 * A * u / half), lean = (u >= 0 ? -1 : 1) * (.42 + .25 * rnd());   // hair grows outwards from the centre
    const a2 = base + lean + (u >= 0 ? 0 : Math.PI), dx = Math.cos(a2) * len, dy = Math.sin(a2) * len;
    hairs += `<line class="hair" x1="${x.toFixed(1)}" y1="${y.toFixed(1)}" x2="${(x + dx).toFixed(1)}" y2="${(y + dy).toFixed(1)}"/>`;
  }
  $("#arcsvg").setAttribute("viewBox", `0 0 ${BW} ${H}`);
  $("#arcsvg").innerHTML = `<defs><linearGradient id="browfill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" class="bf1"/><stop offset="1" class="bf2"/></linearGradient></defs>
    <polygon class="browband" points="${topPts.join(" ")} ${botPts.join(" ")}"/>${hairs}
    <polyline class="browedge" points="${topPts.join(" ")}"/>`;
  // eye
  const almond = `M 0 ${cy} C ${EW * .22} ${-EH * .16}, ${EW * .78} ${-EH * .16}, ${EW} ${cy} C ${EW * .78} ${EH * 1.16}, ${EW * .22} ${EH * 1.16}, 0 ${cy} Z`;
  let lines = ""; for (let k = 0; k < 72; k++) { const a = k / 72 * Math.PI * 2; lines += `<line class="irisline" x1="${cx + Math.cos(a) * RP}" y1="${cy + Math.sin(a) * RP}" x2="${cx + Math.cos(a) * RI}" y2="${cy + Math.sin(a) * RI}"/>`; }
  $("#eyesvg").setAttribute("viewBox", `0 0 ${EW} ${EH}`);
  $("#eyesvg").innerHTML = `<defs><clipPath id="alm"><path d="${almond}"/></clipPath>
      <radialGradient id="scl" cx="50%" cy="50%" r="60%"><stop offset="0" stop-color="${pal().scl[0]}"/><stop offset=".7" stop-color="${pal().scl[1]}"/><stop offset="1" stop-color="${pal().scl[2]}"/></radialGradient>
      <radialGradient id="irisg" cx="50%" cy="50%" r="50%"><stop offset=".78" stop-color="${pal().iris[0]}"/><stop offset="${isLight() ? .88 : .93}" stop-color="${pal().iris[1]}"/>${isLight() ? `<stop offset=".96" stop-color="${pal().iris[2]}"/>` : ""}<stop offset="1" stop-color="${pal().iris[3]}"/></radialGradient>
      <linearGradient id="lidsh" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgba(20,60,40,.22)"/><stop offset=".32" stop-color="rgba(20,60,40,0)"/></linearGradient></defs>
    <path class="sclera" d="${almond}"/>${pal().lidShade ? `<path d="${almond}" fill="url(#lidsh)"/>` : ""}
    <g clip-path="url(#alm)"><g id="irisgrp"><circle class="irisbase" cx="${cx}" cy="${cy}" r="${RI}"/>${lines}
      <g class="gauge" id="gauge" style="--gw:${GW}px"></g>
      <g class="detect" id="detect" role="button" tabindex="0" aria-label="Detect">
        <path class="ringhit" fill-rule="evenodd" d="M ${cx - RI} ${cy} a ${RI} ${RI} 0 1 0 ${2 * RI} 0 a ${RI} ${RI} 0 1 0 ${-2 * RI} 0 Z M ${cx - RP} ${cy} a ${RP} ${RP} 0 1 1 ${2 * RP} 0 a ${RP} ${RP} 0 1 1 ${-2 * RP} 0 Z"/>
        <circle class="ringglow" cx="${cx}" cy="${cy}" r="${(RI + RP) / 2}" style="stroke-width:${RI - RP}px"/>
        <path id="dettext" d="M ${cx - (RI + RP) / 2 - 1} ${cy} A ${(RI + RP) / 2 + 1} ${(RI + RP) / 2 + 1} 0 0 0 ${cx + (RI + RP) / 2 + 1} ${cy}" fill="none"/>
        <text class="detlabel" style="font-size:${Math.max(10, (RI - RP) * .37).toFixed(1)}px" dy="${((RI - RP) * .13).toFixed(1)}"><textPath href="#dettext" startOffset="50%" text-anchor="middle">DETECT</textPath></text>
      </g>
      <circle cx="${cx}" cy="${cy}" r="${RI}" fill="none" stroke="rgba(57,255,136,.35)" stroke-width="1.5"/>
      <circle class="glint" cx="${cx + RP * .55}" cy="${cy - RP * .62}" r="${RP * .07}"/></g></g>`;
  $("#lidsvg").setAttribute("viewBox", `0 0 ${EW} ${EH}`);
  $("#lidsvg").innerHTML = `<defs><clipPath id="alm2"><path d="${almond}"/></clipPath></defs>
    <g clip-path="url(#alm2)"><g class="lid upper" id="lidtop" transform="translate(0 ${-(cy + 6) * lidF})"><rect class="skin" x="0" y="-4" width="${EW}" height="${cy + 4}"/><line class="edge" x1="0" y1="${cy}" x2="${EW}" y2="${cy}"/></g>
      <g class="lid lower" id="lidbot" transform="translate(0 ${(cy + 6) * lidF})"><rect class="skin" x="0" y="${cy}" width="${EW}" height="${cy + 4}"/><line class="edge" x1="0" y1="${cy}" x2="${EW}" y2="${cy}"/></g></g>
    <path class="almond" d="${almond}"/>`;
  const p = $("#pupil"); Object.assign(p.style, { width: 2 * RP + "px", height: 2 * RP + "px", left: cx - RP + "px", top: cy - RP + "px" });
  p.style.setProperty("--pring", Math.max(3, RI * .028).toFixed(1) + "px");
  Object.assign($("#lens").style, { width: G.LENS + "px", height: G.LENS + "px" });
  setStep();
  const btn = $("#detect"); btn.onclick = e => { e.stopPropagation(); detect(btn); };
  btn.onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); detect(btn); } };
  renderGauge(); renderPupil(false); layoutReview(); placeHints();
}
function layoutReview() {
  /* report cards on a gentle curve below the eye (middle lower, ends higher), tilted along it */
  const rev = $("#review"), tiles = $$("#review .tile"); if (!tiles.length) return;
  tiles.forEach(t => t.style.transform = "none");
  const R = rev.getBoundingClientRect(), half = R.width / 2, A = Math.round(G.EH * .12);
  let maxDy = 0;
  tiles.forEach(t => { const r = t.getBoundingClientRect(), u = (r.left + r.width / 2 - R.left - half) / half;
    const dy = A * (1 - u * u), ang = Math.max(-8, Math.min(8, Math.atan(-2 * A * u / half) * 180 / Math.PI));
    t.style.transform = `translateY(${dy.toFixed(1)}px) rotate(${ang.toFixed(2)}deg)`; maxDy = Math.max(maxDy, dy); });
  rev.style.paddingBottom = Math.ceil(maxDy + 10) + "px";
  /* the light: radiating from the pupil's lower half, one beam onto each card, rays spread along the pupil's edge */
  const M = $("#main").getBoundingClientRect(), Pp = $("#pupil").getBoundingClientRect(), beam = $("#beam");
  const pcx = Pp.left + Pp.width / 2 - M.left, pcy = Pp.top + Pp.height / 2 - M.top, pr = Pp.width / 2;
  const onPupil = deg => [pcx + pr * Math.cos(deg * Math.PI / 180), pcy + pr * Math.sin(deg * Math.PI / 180)];
  const cards = tiles.map(t => t.getBoundingClientRect()), bottom = R.bottom - M.top, Hb = bottom + 40;
  beam.setAttribute("viewBox", `0 0 ${M.width} ${Hb}`); beam.style.height = Hb + "px";
  const nC = cards.length, a0 = 150, a1 = 30;                          // the pupil's lower half, left to right (degrees, y down)
  let beams = "", rays = "";
  cards.forEach((c, i) => {
    const s0 = a0 + (a1 - a0) * (i / nC) - 4, s1 = a0 + (a1 - a0) * ((i + 1) / nC) + 4;  // this card's slice of the pupil edge
    const [x0, y0] = onPupil(s0), [x1, y1] = onPupil(s1), top = c.top - M.top + 6;
    beams += `<polygon class="cardbeam" points="${x0},${y0} ${x1},${y1} ${c.right - M.left - 6},${top} ${c.left - M.left + 6},${top}"/>`;
  });
  const NR = 17;
  for (let k = 0; k < NR; k++) { const f = k / (NR - 1), [x1, y1] = onPupil(a0 + (a1 - a0) * f);
    const L = cards[0].left - M.left, Rr = cards[nC - 1].right - M.left, x2 = L + (Rr - L) * f;
    const ci = Math.min(nC - 1, Math.floor(f * nC)), y2 = cards[ci].top - M.top + 4, delay = ((k * .37) % 1 * 3.2).toFixed(2);
    rays += `<line class="ray" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" style="animation-delay:-${delay}s"/>`; }
  const gy0 = pcy + G.RI * .55, gy1 = Math.max(...cards.map(c => c.top - M.top)) + 10;
  const irisR = G.RI + 2;                                                  // the light starts at the iris's outer edge
  beam.innerHTML = `<defs><mask id="beammask" maskUnits="userSpaceOnUse" x="0" y="0" width="${M.width}" height="${Hb}">
      <rect width="${M.width}" height="${Hb}" fill="#fff"/><circle cx="${pcx}" cy="${pcy}" r="${irisR}" fill="#000"/></mask>
      <linearGradient id="beamg" gradientUnits="userSpaceOnUse" x1="0" y1="${gy0}" x2="0" y2="${gy1}">
      <stop offset="0" class="bg1"/><stop offset=".6" class="bg2"/><stop offset="1" class="bg3"/></linearGradient>
      <linearGradient id="rayg" gradientUnits="userSpaceOnUse" x1="0" y1="${gy0}" x2="0" y2="${gy1}"><stop offset="0" class="rg1"/><stop offset="1" class="rg2"/></linearGradient></defs>
    <g mask="url(#beammask)"><g class="cone">${beams}</g>${rays}</g>`;
}
function followReview() {
  /* keep the cards' curve and the light beams aligned while the column widths animate */
  const t0 = performance.now(), step = now => { layoutReview(); if (now - t0 < 450 && !calm()) requestAnimationFrame(step); };
  calm() ? layoutReview() : requestAnimationFrame(step);
}
function dropCards() {
  /* the eye sheds light: the beam brightens and the cards fall into place, one after another */
  if (calm()) return; const beam = $("#beam"); beam.classList.remove("beam-pulse"); void beam.getBoundingClientRect(); beam.classList.add("beam-pulse");
  $$("#review .tile").forEach((t, i) => { t.classList.remove("fall"); void t.offsetWidth; t.style.setProperty("--fd", (.12 + i * .11) + "s"); t.classList.add("fall"); });
  setTimeout(() => $$("#review .tile").forEach(t => t.classList.remove("fall")), 1500);
}

/* ================================================================ iris = score gauge */
function arc(r, a0, a1) { const p = a => [G.cx + r * Math.cos(a), G.cy + r * Math.sin(a)], [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M ${x0} ${y0} A ${r} ${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1} ${y1}`; }
/* v3: DETECT gets its own slot at the bottom of the ring; the score segments share the rest, so no gap can cut the word */
function labelWin() {
  const t = document.querySelector("#detect .detlabel"), r = (G.RI + G.RP) / 2 + 1, fs = t ? parseFloat(t.style.fontSize) || 12 : 12;
  let L = 0; try { L = t ? t.getComputedTextLength() : 0; } catch (e) { }
  if (!(L > 0)) L = 6.6 * fs;                                                     // fallback: 6 letters with .5em spacing
  const w = Math.min(Math.PI * .6, (L + fs * 1.6) / r), c = Math.PI / 2 + (.25 * fs) / r;   // the trailing letter-space shifts the word left
  return { a0: c - w / 2, a1: c + w / 2, w };
}
const plate = (W, gap) => `<path class="show plate" d="${arc(G.RG, W.a0 + gap / 2, W.a1 - gap / 2)}"/>`;
function renderGauge(animate) {
  const g = $("#gauge"); if (!g) return; const det = st.det, W = labelWin();
  if (st.warm === "loading" || st.busy) { g.innerHTML = `<path class="show spin" d="${arc(G.RG, 0, Math.PI * 1.4)}" stroke="rgba(57,255,136,.55)" style="stroke-dasharray:6 8"/>`; return; }
  if (!det) { let t = plate(W, .035); for (let k = 0; k < 60; k++) { const a = k / 60 * Math.PI * 2;
      { const d = ((a - W.a0) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI); if (d < W.w) continue; }   // no ticks under the button
      t += `<line class="tick" x1="${G.cx + Math.cos(a) * (G.RG - G.GW * .3)}" y1="${G.cy + Math.sin(a) * (G.RG - G.GW * .3)}" x2="${G.cx + Math.cos(a) * (G.RG + G.GW * .3)}" y2="${G.cy + Math.sin(a) * (G.RG + G.GW * .3)}"/>`; }
    g.innerHTML = t; return; }
  const items = det.has_truth ? det.table.map(r => r.result) : det.boxes.map(() => "detected"), n = Math.max(1, items.length), gapA = .035;
  const start = W.a1, span = 2 * Math.PI - W.w;                                   // from the right end of the button, clockwise round to its left end
  g.innerHTML = plate(W, gapA) + items.map((r, i) => { const a0 = start + i / n * span + gapA / 2, a1 = start + (i + 1) / n * span - gapA / 2;
    return `<path d="${arc(G.RG, a0, Math.max(a0 + .001, a1))}" stroke="${RCOL[r] || "#39ff88"}" data-i="${i}"/>`; }).join("");
  const ps = $$("#gauge path"); if (!animate || calm()) ps.forEach(p => p.classList.add("show"));
  else ps.forEach((p, i) => setTimeout(() => p.classList.add("show"), 700 + i * (n > 20 ? 35 : 60)));
}

/* ================================================================ pupil content */
function svgFor(det, w, h, animate) {
  if (!det) return ""; const step = det.boxes.length > 30 ? .025 : .045, base = animate ? .78 : 0; let k = 0; const out = [];
  if (det.has_truth) out.push(...det.truths.map(t => `<g class="truth"><rect x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/></g>`));
  const fs = Math.max(11, w / 70), lab = (b, txt) => { const tw = txt.length * fs * .62 + 8;
    return `<g class="lab"><rect x="${b.x1}" y="${Math.max(0, b.y1 - fs - 4)}" width="${tw}" height="${fs + 4}" rx="2"/><text x="${b.x1 + 4}" y="${Math.max(fs, b.y1 - 4)}" style="font-size:${fs}px">${esc(txt)}</text></g>`; };
  for (const b of det.boxes) { const r = cls(b.result), err = ["box-off", "wrong-ID", "extra"].includes(r) ? " err" : "", d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det b-${r}${err}" data-b="${b.k}" style="--d:${d}"><rect class="f" x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}"/>${lab(b, b.id + " " + b.conf.toFixed(2))}</g>`); }
  if (det.has_truth) det.table.forEach((r, i) => { if (r.result !== "missed") return; const t = det.truths[r.gt], d = animate ? (base + step * k++).toFixed(3) + "s" : "0s";
    out.push(`<g class="box det missed b-missed" data-row="${i}" style="--d:${d}"><rect class="f" x="${t.x1}" y="${t.y1}" width="${t.x2 - t.x1}" height="${t.y2 - t.y1}"/>${lab(t, t.id + " missed")}</g>`); });
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${out.join("")}<g class="hlg"></g></svg>`;
}
const imgSrc = () => "data:image/jpeg;base64," + ((st.det && st.det.image) || st.grid.image);
function renderPupil(animate, cascade) {
  const lens = $("#lens"); if (!st.grid) return;
  const g = st.grid, s = Math.min(G.LENS / g.w, G.LENS / g.h), w = g.w * s, h = g.h * s;
  lens.innerHTML = `<div class="zoomer" id="zoomer" style="width:${w}px;height:${h}px"><img src="${imgSrc()}" alt="">${svgFor(st.det, g.w, g.h, animate && !calm())}</div>`;
  if (animate && st.det && !calm()) { const sc = document.createElement("div"); sc.className = "scan"; $("#zoomer").appendChild(sc); setTimeout(() => sc.remove(), 900); }
  if (cascade && g.kind === "grid" && !calm()) coverCells($("#zoomer"), g, .03);
  bindBoxes($("#zoomer")); st.zoomed = null; if (st.sel != null) highlightIn($("#zoomer"), st.sel);
}
function coverCells(el, g, step) { const cw = 100 / g.cols, ch = 100 / g.rows;
  for (let r = 0; r < g.rows; r++) for (let c = 0; c < g.cols; c++) { const d = document.createElement("div"); d.className = "cover";
    Object.assign(d.style, { left: c * cw + "%", top: r * ch + "%", width: cw + .3 + "%", height: ch + .3 + "%", animationDelay: (r + c) * step + "s" }); el.appendChild(d); setTimeout(() => d.remove(), 1600); } }
function bindBoxes(root) { $$(".box", root).forEach(el => { el.onmouseenter = e => showTip(e, el); el.onmousemove = moveTip; el.onmouseleave = hideTip;
  el.onclick = ev => { ev.stopPropagation(); const i = el.dataset.row != null ? +el.dataset.row : st.det.table.findIndex(r => r.pred === +el.dataset.b); if (i >= 0) select(i, true); }; }); }
function regionOf(i) {
  const det = st.det, g = st.grid, r = det.table[i]; if (!r) return null; const b = r.gt != null && det.truths[r.gt] ? det.truths[r.gt] : det.boxes[r.pred]; if (!b) return null;
  if (g.cols && g.rows) { const cx = Math.floor((b.x1 + b.x2) / 2 / 224), cy = Math.floor((b.y1 + b.y2) / 2 / 224); return [cx * 224, cy * 224, cx * 224 + 224, cy * 224 + 224]; }
  const m = .15 * Math.max(b.x2 - b.x1, b.y2 - b.y1, 20); return [b.x1 - m, b.y1 - m, b.x2 + m, b.y2 + m];
}
function highlightIn(root, i) { const hl = $(".hlg", root), g = st.grid, R = regionOf(i); if (!hl || !R) return; const [x1, y1, x2, y2] = R;
  hl.innerHTML = `<path class="hl" d="M0 0H${g.w}V${g.h}H0Z M${x1} ${y1}V${y2}H${x2}V${y1}Z"/><rect class="hlf" x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}"/>`; }
function zoomTo(i) {
  const z = $("#zoomer"); if (!z) return; const R = regionOf(i), g = st.grid; if (!R) return;
  const s = z.clientWidth / g.w, cw = (R[2] - R[0]) * s, ch = (R[3] - R[1]) * s, k = Math.min(3.2, G.LENS * .62 / Math.max(cw, ch));
  if (k <= 1.05) { z.style.transform = ""; st.zoomed = null; return; }
  const mx = (R[0] + R[2]) / 2 * s, my = (R[1] + R[3]) / 2 * s, W = z.clientWidth, H = z.clientHeight;
  z.style.transform = `translate(${W / 2 - mx * k}px,${H / 2 - my * k}px) scale(${k})`; st.zoomed = i;
}
function zoomOut() { const z = $("#zoomer"); if (z) z.style.transform = ""; st.zoomed = null; }

/* ================================================================ eye life: follow cursor, look around, blink, dilate */
let follow = { x: 0, y: 0 };
addEventListener("mousemove", e => {
  if (calm() || st.busy || st.zoomed != null) return; const r = $("#eye").getBoundingClientRect(), dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2), d = Math.hypot(dx, dy) || 1;
  const k = Math.min(1, d / 600) * 9; follow = { x: dx / d * k, y: dy / d * k * .7 }; applyFollow();
});
function applyFollow() { $("#irisgrp") && ($("#irisgrp").style.transform = `translate(${follow.x}px,${follow.y}px)`); $("#pupil").style.translate = `${follow.x}px ${follow.y}px`; }
function lookAround() { if (calm()) return; const seq = [[-14, 0], [14, -3], [-6, 3], [0, 0]], els = [$("#irisgrp"), $("#pupil")];
  els[0].animate(seq.map(([x, y]) => ({ transform: `translate(${x}px,${y}px)` })), { duration: 900, easing: "ease-in-out" });
  els[1].animate(seq.map(([x, y]) => ({ translate: `${x}px ${y}px` })), { duration: 900, easing: "ease-in-out" }); }
/* lids: positioned with the SVG transform attribute and animated frame by frame (requestAnimationFrame).
   The browser's own animation engine stopped painting the upper lid while it moved. */
let lidF = 0, lidRun = 0;
const EASE = { in: k => k * k * k, out: k => 1 - Math.pow(1 - k, 3), inout: k => k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2 };
function setLids(f) { const cy = G.cy + 6; lidF = f; $("#lidtop")?.setAttribute("transform", `translate(0 ${-cy * f})`); $("#lidbot")?.setAttribute("transform", `translate(0 ${cy * f})`); }
function lidsTo(f, ms, ease = "inout") {          // f = 0 closed ... 1 fully open
  const id = ++lidRun, f0 = lidF, t0 = performance.now(), E = EASE[ease] || EASE.inout;
  if (ms <= 1 || calm()) { setLids(f); return Promise.resolve(); }
  return new Promise(res => { const step = now => { if (id !== lidRun) return res();       // a newer movement took over
      const k = Math.min(1, (now - t0) / ms); setLids(f0 + (f - f0) * E(k)); k < 1 ? requestAnimationFrame(step) : res(); };
    requestAnimationFrame(step); });
}
function moveLids(closed, ms) { return lidsTo(closed ? 0 : 1, ms, closed ? "in" : "out"); }
function frames(n) { return new Promise(r => { const step = k => k ? requestAnimationFrame(() => step(k - 1)) : r(); step(n); }); }
async function predecode(b64) { if (!b64) return; try { const im = new Image(); im.src = "data:image/jpeg;base64," + b64; await im.decode(); } catch (e) {} }
async function blinkClose() { if (calm()) return; await moveLids(true, 190); }
async function blinkOpen() { if (calm()) return; await moveLids(false, 280); }

/* ================================================================ steps, guidance, hints */
function setStep() {
  const s = !st.grid ? 1 : !st.det ? 2 : 3;
  $("#d-rand").classList.toggle("next", s === 1); $("#detect")?.classList.toggle("next", s === 2);
  $$("#t-score,#t-err").forEach(t => t.classList.toggle("next", s === 3 && !st.reviewedOnce));
  if (s === 3 && !st.reviewedOnce) setTimeout(() => { st.reviewedOnce = true; $$("#t-score,#t-err").forEach(t => t.classList.remove("next")); }, 4500);
}
function placeHints() {
  $$(".hintdot").forEach(h => h.remove()); return;                       // numbered hints removed by request
  if (st.hinted) return; const main = $("#main").getBoundingClientRect();
  const add = (el, n, text, dx = -36, dy = 0) => { const r = el.getBoundingClientRect(), h = document.createElement("div"); h.className = "hintdot";
    h.innerHTML = n + `<span>${text}</span>`; Object.assign(h.style, { left: (r.left - main.left + dx) + "px", top: (r.top - main.top + r.height / 2 - 13 + dy) + "px" }); $("#main").appendChild(h); };
  add($("#s-grid"), 1, "set up the test here", -8, -64); add($("#detect"), 2, "press random, then detect", $("#detect").offsetWidth + 14); add($("#review"), 3, "review", -36, -100);
}
function retireHints() { if (st.hinted) return; st.hinted = true; $("#main").classList.add("hidehints"); setTimeout(() => $$(".hintdot").forEach(h => h.remove()), 700); }

/* ================================================================ controls on the eyebrow */
function chips(el, items, isOn, onClick) { el.innerHTML = ""; items.forEach(v => { const c = document.createElement("span"); c.className = "chip" + (isOn(v) ? " on" : ""); c.textContent = v.label || v; c.onclick = () => onClick(v); el.appendChild(c); }); }
function renderControls() {
  const c = st.cfg;
  $("#model").innerHTML = c.models.map(m => `<option ${m === st.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const mi = c.model_info[st.model] || {}; $("#mapv").textContent = `test mAP@0.5 ${mi.test_map50 != null ? mi.test_map50.toFixed(3) : "-"}` + (st.model === ((c.sel_info || {}).selected || c.boost) ? " · selected" : st.model === c.best ? " · M2 pick" : "");
  $("#conf").value = st.conf; $("#thrv").textContent = (+st.conf).toFixed(2); $("#cell").checked = st.cell;
  const vg = validatedConf("grid"), vs = validatedConf("scene"), kk = curKind();
  $("#thrkind").textContent = Math.abs(vg - vs) < 1e-9 ? `validated ${vg.toFixed(2)}` : `${vg.toFixed(2)} grids · ${vs.toFixed(2)} scenes` +
    (st.confAuto ? "" : " · set by hand");
  chips($("#sizes"), c.grids.map(g => ({ label: g.startsWith("Single") ? "1x1" : g.split(" ")[0], v: g })), x => x.v === st.size, x => { st.size = x.v; renderControls(); });
  $("#sizeinfo").textContent = st.size.replace("Single photo (1x1)", "single photo");
  chips($("#secs"), c.sections, s => st.sections.includes(s), s => { st.sections = st.sections.includes(s) ? st.sections.filter(x => x !== s) : [...st.sections, s]; if (!st.sections.length) st.sections = [s]; renderControls(); });
  chips($("#mixes"), [{ label: "random", v: "Random" }, { label: "look-alikes", v: "Look-alike challenge" }], x => x.v === st.mix, x => { st.mix = x.v; renderControls(); });
  chips($("#layouts"), (c.layouts || ["grid"]).map(v => ({ label: v, v })), x => x.v === st.layout, x => { st.layout = x.v; renderControls(); });
  chips($("#degs"), c.degrade.map(d => ({ label: d.startsWith("All") ? "all" : d.toLowerCase().replace("low resolution", "low-res").replace("heavy jpeg", "jpeg"), v: d })), x => x.v === st.degrade, x => { st.degrade = x.v; renderControls(); });
}
const pickModel = m => { st.model = m; st.confAuto = true; st.confKind = curKind(); st.conf = validatedConf(); renderControls();
  if ($("#ov-stage").classList.contains("show")) syncStageControls(); };
$("#model").onchange = e => pickModel(e.target.value);
$("#conf").oninput = e => { st.conf = +e.target.value; st.confAuto = false; $("#thrv").textContent = st.conf.toFixed(2); };
$("#conf").onchange = () => renderControls();
$("#cell").onchange = e => st.cell = e.target.checked;
// v3: the image on screen is a grid (one box per cell) or a scene (scattered, cut-outs, an upload that is not a grid)
const curKind = () => { const g = st.grid; if (!g) return ["scattered", "cut-outs"].includes(st.layout) ? "scene" : "grid";
  return g.kind === "scene" || (g.kind === "upload" && !g.cols) ? "scene" : "grid"; };
const validatedConf = (kind = curKind()) => { const c = st.cfg || {}, t = ((kind === "scene" ? c.model_conf_scene : c.model_conf) || {})[st.model] ?? (c.model_conf || {})[st.model];
  return t != null ? t : c.conf; };   // each model's own validated threshold, for grids or scenes
function syncConf() { const k = curKind(); if (st.confAuto) st.conf = validatedConf(k); st.confKind = k; renderControls();
  if ($("#ov-stage").classList.contains("show")) syncStageControls(); }
$("#resetthr").onclick = e => { shock(e.currentTarget); st.confAuto = true; st.conf = validatedConf(); renderControls(); };
$("#dice").onclick = e => { shock(e.currentTarget); $("#seed").value = ""; };
$("#keep").onclick = e => { shock(e.currentTarget); if (st.lastSeed != null) $("#seed").value = st.lastSeed; };

/* ================================================================ review tiles */
function countUp(el, to, ms = 900) { if (calm()) { el.textContent = to; return; } const t0 = performance.now();
  const f = n => { const k = Math.min(1, (n - t0) / ms); el.textContent = Math.round(to * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(f); }; requestAnimationFrame(f); }
function renderReview() {
  document.body.classList.toggle("idle", !st.det);
  const det = st.det;
  if (!det) { $("#sc").textContent = "-"; $("#scof").textContent = ""; $("#scbar").style.width = "0"; $("#verdict").textContent = ""; $("#stats").innerHTML = ""; $("#ms").textContent = "";
    $("#errs").innerHTML = `<div class="lbl">one line per object appears here</div>`; renderP4(); return; }
  $("#ms").textContent = det.ms + " ms" + (det.first_call ? " (incl. model loading)" : "");
  if (det.tally) { const t = det.tally; countUp($("#sc"), t.correct); $("#scof").textContent = "/ " + t.n; setTimeout(() => $("#scbar").style.width = (100 * t.correct / t.n) + "%", 30);
    $("#verdict").innerHTML = t.fully ? `<span style="color:var(--acc)">FULLY CORRECT</span>` : `<span style="color:var(--amber)">${t.named}/${t.n} NAMED CORRECTLY</span>`;
    $("#stats").innerHTML = `<span>box off <b>${t.box_off}</b></span><span>wrong ID <b>${t.wrong}</b></span><span>missed <b>${t.missed}</b></span><span>extra <b>${t.extra}</b></span>` +
      Object.entries(det.per_section).map(([s, v]) => `<span>${esc(s.replace("Section - ", "sec "))} <b>${v[0]}/${v[1]}</b></span>`).join(""); }
  else { countUp($("#sc"), det.boxes.length); $("#scof").textContent = "found"; $("#scbar").style.width = det.boxes.length ? "100%" : "0";
    $("#verdict").innerHTML = det.boxes.length ? "UPLOAD · NO TRUE BOXES" : `<span style="color:var(--amber)">NOTHING DETECTED</span>`; $("#stats").innerHTML = ""; }
  st.exp = new Set(); renderLists(); if ($("#review").classList.contains("wide")) { $("#review").classList.remove("wide"); followReview(); }
  $("#insp").innerHTML = det.table.length ? `<div class="lbl">click any object in the eye or the list, or use ‹ › to step through all ${det.table.length}</div>
      <div class="inav"><button class="enl" id="ip">‹</button><span>– / ${det.table.length}</span><button class="enl" id="in">›</button></div>` : `<div class="lbl">nothing to inspect</div>`;
  if (det.table.length) { $("#ip").onclick = () => select(det.table.length - 1, true); $("#in").onclick = () => select(0, true); }
  renderP4();
}
/* v3: panel 04 shows the threshold sweep or the box tightness of this image */
function renderP4() {
  const det = st.det, tight = st.p4 === "tight", up = det && !det.sweep && det.upsweep;
  if (st.p4 === "cmp") { $("#p4title").textContent = "Compound view";
    document.querySelectorAll("#swtabs .chip").forEach(c => c.classList.toggle("on", c.dataset.f === "cmp"));
    $("#sweep").style.display = "none"; const mini = $("#cmpmini"); mini.style.display = "flex"; mini.innerHTML = cmpMini();
    mini.onclick = () => { if (st.cmp && st.det && st.cmp.det === st.det.det) openCompoundBig(); }; return; }
  $("#sweep").style.display = ""; $("#cmpmini").style.display = "none";
  $("#sweep").style.cursor = ""; $("#sweep").onclick = null;
  $("#p4title").textContent = up ? (tight ? "Confidence" : "Boxes kept") : (tight ? "Box tightness" : "Threshold sweep");
  const ca = document.querySelector('#swtabs .chip[data-f="sweep"]'), cb = document.querySelector('#swtabs .chip[data-f="tight"]'); ca.textContent = up ? "kept" : "sweep"; cb.textContent = up ? "confidence" : "tightness";
  cb.title = up ? "how confident the model is in each box" : "how tightly each box fits its object (IoU)";
  document.querySelectorAll("#swtabs .chip").forEach(c => c.classList.toggle("on", c.dataset.f === (tight ? "tight" : "sweep")));
  if (!det) { $("#sweep").innerHTML = ""; return; }
  $("#sweep").innerHTML = up ? (tight ? uconfSvg(det, 360, 150) : ukeptSvg(det, 360, 150))
    : !det.sweep ? `<text x="10" y="30" font-size="11" fill="${pal().sw.hint}" font-family="monospace">needs true boxes (random grids and scenes)</text>`
    : tight ? tightSvg(det, 360, 150) : sweepSvg(det.sweep, det.conf, 360, 150);
}
/* v3.3: for uploads (no true boxes) - how many boxes survive each threshold, and how confident each candidate is */
function ukeptSvg(det, W, H) {
  const s = det.upsweep, P = pal(), big = W > 500, fs = big ? 13 : 10.5, mx = Math.max(1, ...s.n), x = t => 12 + (t - .05) / .85 * (W - 24), y = v => H - (big ? 40 : 24) - v / mx * (H - (big ? 70 : 40));
  const pts = s.t.map((t, i) => `${x(t).toFixed(1)},${y(s.n[i]).toFixed(1)}`).join(" "), now = s.n[Math.max(0, s.t.findIndex(t => t >= det.conf - 1e-9))];
  return `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>` +
    `<polyline fill="none" stroke="${P.res["correct"]}" stroke-width="${big ? 3 : 2}" points="${pts}"/>` +
    s.t.map((t, i) => `<circle cx="${x(t).toFixed(1)}" cy="${y(s.n[i]).toFixed(1)}" r="${big ? 3.5 : 2}" fill="${P.res["correct"]}"><title>threshold ${t.toFixed(2)}: ${s.n[i]} boxes</title></circle>`).join("") +
    `<line x1="${x(det.conf)}" x2="${x(det.conf)}" y1="8" y2="${H - (big ? 36 : 20)}" stroke="${P.sw.cur}" stroke-dasharray="3 3"/>` +
    `<text x="${(x(det.conf) + 5).toFixed(0)}" y="${big ? 26 : 16}" font-size="${fs}" fill="${P.sw.cur}" font-family="monospace">${now ?? det.boxes.length} at ${det.conf.toFixed(2)}</text>` +
    `<text x="10" y="${H - 6}" font-size="${fs}" fill="${P.sw.text}" font-family="monospace">${big ? "boxes kept at each threshold 0.05→0.90 · flat = sure, steep = fragile" : "boxes kept per threshold · flat = sure"}</text>`;
}
function uconfSvg(det, W, H) {
  const c = det.ucands || [], P = pal(), big = W > 500, fs = big ? 12 : 10.5, top = big ? 30 : 16, bot = H - (big ? 70 : 24), n = Math.max(1, c.length);
  const bw = (W - 24) / n, y = v => bot - v * (bot - top);
  const bars = c.map((d, i) => { const on = d.conf >= det.conf - 1e-9, xx = 12 + i * bw;
    return `<rect x="${(xx + bw * .14).toFixed(1)}" y="${y(d.conf).toFixed(1)}" width="${Math.max(1, bw * .72).toFixed(1)}" height="${(bot - y(d.conf)).toFixed(1)}" rx="2" fill="${on ? P.res["correct"] : P.sw.hint}" fill-opacity="${on ? .9 : .45}"><title>${esc(d.id)} ${esc(d.name || "")} · confidence ${d.conf.toFixed(2)}${on ? "" : " (below the threshold)"}</title></rect>` +
      (big && n <= 30 ? `<text transform="translate(${(xx + bw / 2 + 4).toFixed(1)},${bot + 8}) rotate(60)" font-size="10" fill="${P.sw.text}" font-family="monospace">${esc(d.id)}</text>` : ""); }).join("");
  const above = c.filter(d => d.conf >= det.conf - 1e-9).length;
  return `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>` + bars +
    `<line x1="10" x2="${W - 10}" y1="${y(det.conf).toFixed(1)}" y2="${y(det.conf).toFixed(1)}" stroke="${P.sw.cur}" stroke-dasharray="3 3"/>` +
    `<text x="${W - 12}" y="${(y(det.conf) - 4).toFixed(1)}" text-anchor="end" font-size="${fs}" fill="${P.sw.cur}" font-family="monospace">threshold ${det.conf.toFixed(2)}</text>` +
    `<text x="10" y="${H - 6}" font-size="${fs}" fill="${P.sw.text}" font-family="monospace">${c.length ? `${above} of ${c.length} candidates above the threshold · most confident first` : "no candidates above 0.05"}</text>`;
}
function tightStats(det) {
  const rows = (det.table || []).filter(r => ["correct", "box off", "wrong ID"].includes(r.result) && r.IoU != null), n = det.tally ? det.tally.n : rows.length;
  const right = rows.filter(r => r.result === "correct").map(r => +r.IoU).sort((a, b) => a - b);
  const at = t => right.filter(v => v >= t - 1e-9).length, med = right.length ? right[Math.floor((right.length - 1) / 2)] : null;
  return { rows, n, a50: at(.5), a75: at(.75), a95: at(.95), med };
}
function tightSvg(det, W, H) {
  const P = pal(), R = P.res, S = tightStats(det), big = W > 500, fs = big ? 13 : 10.5, x0 = big ? 46 : 26, y0 = H - (big ? 46 : 30);
  const top = big ? 40 : 30, x = v => x0 + (Math.max(.4, Math.min(1, v)) - .4) / .6 * (W - x0 - 12), y = c => y0 - Math.max(0, Math.min(1, c)) * (y0 - top);
  const cut = t => `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 2}" y2="${y0}" stroke="${P.sw.text}" stroke-opacity=".45" stroke-dasharray="3 3"/>` +
    `<text x="${x(t) + 3}" y="${y0 - 4}" font-size="${fs}" fill="${P.sw.text}" font-family="monospace">${t.toFixed(2)}</text>`;
  const dots = S.rows.map(r => `<circle cx="${x(+r.IoU)}" cy="${y(+r.confidence)}" r="${big ? 6 : 3.4}" fill="${R[r.result]}" fill-opacity=".85">` +
    `<title>${esc(r.true || "")} ${esc(r.result)} · IoU ${f2(r.IoU)} · confidence ${f2(r.confidence)}</title></circle>`).join("");
  const axis = `<line x1="${x0}" x2="${W - 12}" y1="${y0}" y2="${y0}" stroke="${P.sw.hint}"/><line x1="${x0}" x2="${x0}" y1="${top - 2}" y2="${y0}" stroke="${P.sw.hint}"/>` +
    `<text x="${x0 - 3}" y="${y(1) + 3}" text-anchor="end" font-size="${fs}" fill="${P.sw.hint}" font-family="monospace">1</text>` +
    `<text x="${W - 12}" y="${y0 + fs + 3}" text-anchor="end" font-size="${fs}" fill="${P.sw.hint}" font-family="monospace">IoU with the true box →</text>` +
    `<text x="${x0 - 4}" y="${y0 + fs + 3}" text-anchor="end" font-size="${fs}" fill="${P.sw.hint}" font-family="monospace">≤.4</text>` +
    `<text transform="translate(${big ? 16 : 10},${y0}) rotate(-90)" font-size="${fs}" fill="${P.sw.hint}" font-family="monospace">confidence →</text>`;
  const foot = S.n ? `IoU ≥0.50 ${S.a50}/${S.n} · ≥0.75 ${S.a75}/${S.n} · ≥0.95 ${S.a95}/${S.n}` + (S.med != null ? ` · median ${S.med.toFixed(2)}` : "") : "no matched objects";
  return `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>` + cut(.5) + cut(.75) + cut(.95) + axis + dots +
    `<text x="10" y="${H - 6}" font-size="${fs}" fill="${P.sw.text}" font-family="monospace">${foot}</text>`;
}
document.querySelectorAll("#swtabs .chip").forEach(c => c.onclick = e => { e.stopPropagation(); st.p4 = c.dataset.f; renderP4(); });
function sweepSvg(s, cur, W, H) {
  const mx = Math.max(1, ...s.correct, ...s.missed, ...s.extra, ...s.wrong), x = t => 10 + (t - .05) / .85 * (W - 20), y = v => H - 22 - v / mx * (H - 36);
  const P = pal(), R = P.res;
  const line = (k, c, d) => `<polyline fill="none" stroke="${c}" stroke-width="2" ${d ? 'stroke-dasharray="4 3"' : ""} points="${s.t.map((t, i) => x(t) + "," + y(s[k][i])).join(" ")}"/>`;
  return `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>` + line("correct", R["correct"]) + line("missed", R["missed"], 1) + line("extra", R["extra"], 1) + line("wrong", R["wrong ID"]) +
    `<line x1="${x(cur)}" x2="${x(cur)}" y1="8" y2="${H - 20}" stroke="${P.sw.cur}" stroke-dasharray="3 3"/>` +
    `<text x="10" y="${H - 6}" font-size="9" fill="${P.sw.text}" font-family="monospace">threshold 0.05→0.90 · green correct · cyan missed · grey extra · pink wrong</text>`;
}
/* ---------- one clear sentence per object, and every detail on demand ---------- */
let GROUPS = {};                                                   // Object ID -> look-alike group (from the server's config)
const f2 = v => v == null ? "-" : (+v).toFixed(2);
const lab = (id, name) => id ? `${id}${name ? " (" + name + ")" : ""}` : "";
function sentence(r) {
  const T = lab(r.true, r.true_name), P = lab(r.predicted, r.predicted_name);
  switch (r.result) {
    case "correct":  return `${T} found correctly, confidence ${f2(r.confidence)}.`;
    case "box off":  return `${T} was named correctly, but its box is placed off (overlap ${f2(r.IoU)}; at least 0.50 is needed).`;
    case "wrong ID": return `${T} was called ${P}` + (String(r.reason).startsWith("look-alike") ? ", a look-alike." : ".");
    case "missed": { const why = String(r.reason), g = why.match(/^not found \(look-alike group: (.+)\)$/);
      if (why === "not found") return `${T} was not found.`;
      if (g) return `${T} was not found (it belongs to the look-alike group ${g[1]}).`;
      if (why.startsWith("not found: ")) return `${T} was not found: ${why.slice(11)}.`;
      return `${T} was missed: ${why}.`; }
    case "extra":    return `Extra box: ${P} at confidence ${f2(r.confidence)}, ${String(r.reason)}.`;
    default:         return `${P} detected, confidence ${f2(r.confidence)}.`;
  }
}
function detailTable(r) {
  const det = st.det, pb = r.pred != null ? det.boxes[r.pred] : null, tb = r.gt != null ? det.truths[r.gt] : null;
  const bx = b => b ? `${Math.round(b.x1)}, ${Math.round(b.y1)} → ${Math.round(b.x2)}, ${Math.round(b.y2)}` : "-";
  const rows = [["result", r.result], ["reason", r.reason || "-"], ["true object", lab(r.true, r.true_name) || "-"], ["predicted", lab(r.predicted, r.predicted_name) || "-"],
    ["section", r.section || "-"], ["look-alike group", GROUPS[r.true || r.predicted] || "-"], ["class AP (test)", r.class_AP ?? "-"],
    ["confidence", f2(r.confidence)], ["IoU with true box", f2(r.IoU)], ["predicted box (px)", bx(pb)], ["true box (px)", bx(tb)]];
  return `<table class="xtab">${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</table>`;
}
st.exp = new Set(); st.tileFilter = "err";
function rowList(el, filter, zoom) {
  const det = st.det; if (!det) return;
  const f = det.has_truth ? filter : "all";      // an upload has no true boxes, so nothing in it can be an error: list every detection
  const rows = det.table.map((r, i) => [r, i]).filter(([r]) => f === "err" ? !["correct", "detected"].includes(r.result) : true);
  if (!rows.length) {
    el.innerHTML = det.has_truth ? `<div class="lbl" style="color:var(--acc)">no errors: every object found with the right ID${filter === "err" ? ". Choose “all” to see each object" : ""}</div>`
      : `<div class="lbl" style="color:var(--amber)">nothing detected above the threshold ${det.conf.toFixed(2)}. Uploads have no true boxes, so errors cannot be judged; try a lower threshold, or an upright photo</div>`;
    return; }
  el.innerHTML = rows.map(([r, i]) => `<div class="item r-${cls(r.result)}${i === st.sel ? " sel" : ""}" data-i="${i}"><span class="dot"></span>
      <div class="ib"><div class="sent">${esc(sentence(r))}${cmpBadge(r)}</div>${st.exp.has(i) ? detailTable(r) : ""}</div>
      <button class="xbtn" data-x="${i}" title="${st.exp.has(i) ? "hide details" : "show every detail"}">${st.exp.has(i) ? "−" : "+"}</button></div>`).join("");
  $$(".item", el).forEach(it => it.onclick = () => select(+it.dataset.i, zoom));
  $$(".xbtn", el).forEach(b => b.onclick = e => { e.stopPropagation(); const i = +b.dataset.x; st.exp.has(i) ? st.exp.delete(i) : st.exp.add(i); renderLists(); });
}
function renderLists() {
  if (!st.det) return;
  rowList($("#errs"), st.tileFilter, true);
  const shown = new Set($$("#errs .item").map(it => +it.dataset.i)), wide = [...st.exp].some(i => shown.has(i));
  if ($("#review").classList.contains("wide") !== wide) { $("#review").classList.toggle("wide", wide); followReview(); }
  if ($("#ov-stage").classList.contains("show")) rowList($("#srows"), st.filter, false);
  const nErr = st.det.table.filter(r => !["correct", "detected"].includes(r.result)).length;
  $$("#errtabs .chip").forEach(c => { c.classList.toggle("on", c.dataset.f === st.tileFilter); c.textContent = (c.dataset.f === "err" ? "errors " : "all ") + (c.dataset.f === "err" ? nErr : st.det.table.length); });
}
$$("#errtabs .chip").forEach(c => c.onclick = () => { st.tileFilter = c.dataset.f; renderLists(); });

/* ---------- inspect: every object, with ‹ › and an enlarged view ---------- */
async function showInspect(i) {
  const det = st.det; if (!det || !det.table.length) return; const n = det.table.length; st.inspI = i;
  $("#insp").innerHTML = `<div class="lbl">loading...</div>`;
  try { const t = await api(`/api/tile/${det.det}/${i}`); if (st.det !== det || st.inspI !== i) return;
    $("#insp").innerHTML = `<img src="data:image/jpeg;base64,${t.image}" alt="" title="click to enlarge" style="cursor:zoom-in"><div class="cap">${esc(sentence(det.table[i]))}</div>
      <div class="inav"><button class="enl" id="ip">‹</button><span>${i + 1} / ${n}</span><button class="enl" id="in">›</button><span class="sp"></span></div>`;
    $("#ip").onclick = () => select((i - 1 + n) % n, true); $("#in").onclick = () => select((i + 1) % n, true); $("#insp img").onclick = () => openInspectBig(i);
  } catch (e) { $("#insp").innerHTML = `<div class="lbl">${esc(e.message)}</div>`; }
}
async function openInspectBig(i) {
  const det = st.det; if (!det || !det.table.length) return toast("Detect first, then inspect.");
  i = i ?? st.inspI ?? st.sel ?? 0; const n = det.table.length, r = det.table[i];
  $("#bigpanel").innerHTML = `<h2>INSPECT · ${i + 1} / ${n}</h2><div class="lbl">loading...</div>`; $("#ov-big").classList.add("show");
  try { const t = await api(`/api/tile/${det.det}/${i}?size=360`);
    $("#bigpanel").innerHTML = `<h2>INSPECT · ${i + 1} / ${n}</h2><img class="bigimg" src="data:image/jpeg;base64,${t.image}" alt="">
      <div class="bignav"><button class="close" id="bp">‹ previous</button><span class="sp"></span><span>${esc(sentence(r))}</span><span class="sp"></span><button class="close" id="bn">next ›</button></div>
      ${detailTable(r)}<div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
    $("#bp").onclick = () => { const k = (i - 1 + n) % n; select(k, true); openInspectBig(k); };
    $("#bn").onclick = () => { const k = (i + 1) % n; select(k, true); openInspectBig(k); };
    $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
  } catch (e) { $("#bigpanel").innerHTML = `<p>${esc(e.message)}</p>`; }
}
function openUploadBig(det) {
  const tight = st.p4 === "tight";
  $("#bigpanel").innerHTML = tight ? `<h2>CONFIDENCE PER OBJECT</h2>
    <p>Every candidate box the model considered, most confident first. Bars above the dashed line are the boxes it reports at the current threshold (${det.conf.toFixed(2)}); the faint ones were seen, but not confidently enough. An upload has no true boxes, so this shows how sure the model is, not whether it is right.</p>
    <svg viewBox="0 0 900 420" style="width:100%;height:auto">${uconfSvg(det, 900, 420)}</svg>
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`
  : `<h2>BOXES KEPT VS THRESHOLD</h2>
    <p>How many boxes this image keeps at each confidence threshold; the dashed line is the current threshold (${det.conf.toFixed(2)}). A flat curve means the model is sure of what it found; a steep drop means many boxes sit close to the threshold, so the count is fragile. An upload has no true boxes, so this cannot say which boxes are right.</p>
    <svg viewBox="0 0 900 420" style="width:100%;height:auto">${ukeptSvg(det, 900, 420)}</svg>
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
  $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
}
function openSweepBig() {
  if (st.p4 === "cmp") return openCompoundBig();
  const det = st.det; if (det && !det.sweep && det.upsweep) return openUploadBig(det);
  if (!det || !det.sweep) return toast("This panel needs an image with true boxes: press RANDOM, then DETECT.");
  const R = pal().res;
  if (st.p4 === "tight") { const S = tightStats(det);
    $("#bigpanel").innerHTML = `<h2>BOX TIGHTNESS</h2>
    <p>One dot per true object the model boxed: across, how well the box overlaps the true box (IoU, 1.00 = perfect); up, the model's confidence. A box counts as correct from IoU 0.50; mAP@0.5:0.95 also asks for 0.55, 0.60 … 0.95, so dots left of a line lose credit at that line.</p>
    <svg viewBox="0 0 900 420" style="width:100%;height:auto">${tightSvg(det, 900, 420)}</svg>
    <div class="legend" style="margin-top:8px"><span style="--c:${R["correct"]}"><i></i>correct</span><span style="--c:${R["box off"]}"><i></i>box off</span><span style="--c:${R["wrong ID"]}"><i></i>wrong ID</span></div>
    <p class="lbl" style="margin-top:8px">Correct objects with IoU ≥ 0.50: ${S.a50} of ${S.n}; ≥ 0.75: ${S.a75}; ≥ 0.95: ${S.a95}${S.med != null ? `; median IoU ${S.med.toFixed(2)}` : ""}. The labels themselves match their objects at a median IoU of about 0.97, so a box near 0.95 is about as tight as the labels allow; credit lost at the strict cut-offs comes from the minority of looser boxes.</p>
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
    $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show"); return; }
  $("#bigpanel").innerHTML = `<h2>THRESHOLD SWEEP</h2>
    <p>How this grid's result changes with the confidence threshold. The dashed yellow line is the current threshold (${det.conf.toFixed(2)}).</p>
    <svg viewBox="0 0 900 420" style="width:100%;height:auto">${sweepSvg(det.sweep, det.conf, 900, 420)}</svg>
    <div class="legend" style="margin-top:8px"><span style="--c:${R["correct"]}"><i></i>correct objects</span><span style="--c:${R["missed"]}"><i class="dash"></i>missed (incl. box off)</span>
      <span style="--c:${R["extra"]}"><i class="dash"></i>extra boxes (incl. box off)</span><span style="--c:${R["wrong ID"]}"><i></i>wrong IDs</span></div>
    <p class="lbl" style="margin-top:8px">A low threshold finds more objects but allows more extra boxes; a high threshold is stricter but misses more. ${(+validatedConf()).toFixed(2)} was chosen on validation for this model and this kind of image.</p>
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
  $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
}
$("#insp-big").onclick = () => openInspectBig(); $("#sweep-big").onclick = openSweepBig;
$("#sweep").style.cursor = "zoom-in"; $("#sweep").onclick = openSweepBig;

async function select(i, zoom) {
  if (st.sel === i) { clearSel(); return; }
  st.sel = i; $$("#srows .item,#errs .item").forEach(el => el.classList.toggle("sel", +el.dataset.i === i));
  highlightIn($("#zoomer"), i); if ($("#ov-stage").classList.contains("show")) highlightIn($("#fig"), i);
  if (zoom && !$("#ov-stage").classList.contains("show")) zoomTo(i);
  showInspect(i);
}
function clearSel() { st.sel = null; zoomOut(); $$(".hlg").forEach(h => h.innerHTML = ""); $$(".item.sel").forEach(e => e.classList.remove("sel")); }

/* ================================================================ tooltips */
function showTip(e, el) { const det = st.det; if (!det) return; let html;
  if (el.dataset.row != null) { const r = det.table[+el.dataset.row]; html = `<b>${esc(r.true)}</b> ${esc(r.true_name)}<br>missed · ${esc(r.reason)}<br>${esc(r.section)}`; }
  else { const b = det.boxes[+el.dataset.b], r = det.table.find(x => x.pred === b.k) || {};
    html = `<b>${esc(b.id)}</b> ${esc(b.name)} · ${b.conf.toFixed(2)}<br>${esc(b.section)}${b.ap != null ? " · class AP " + b.ap : ""}${b.group ? " · look-alikes: " + esc(b.group) : ""}` +
      (r.result && !["correct", "detected"].includes(r.result) ? `<br><span style="color:var(--amber)">${esc(r.result)}: ${esc(r.reason)}</span>` : ""); }
  const t = $("#tip"); t.innerHTML = html; t.style.display = "block"; moveTip(e); }
function moveTip(e) { const t = $("#tip"); t.style.left = Math.min(innerWidth - 320, e.clientX + 14) + "px"; t.style.top = (e.clientY + 14) + "px"; }
function hideTip() { $("#tip").style.display = "none"; }
$$("[data-tip]").forEach(el => { el.onmouseenter = e => { const t = $("#tip"); t.textContent = el.dataset.tip; t.style.display = "block"; moveTip(e); };
  el.onmousemove = moveTip; el.onmouseleave = hideTip; });

/* ================================================================ expanded pupil (large view) */
function fit(fig, box, w, h) { const s = Math.min(box.clientWidth / w, box.clientHeight / h); fig.style.width = w * s + "px"; fig.style.height = h * s + "px"; }
function openStage() { if (!st.grid) return toast("Press RANDOM (or upload) first."); $("#ov-stage").classList.add("show"); renderStage(); }
function renderStage() {
  const g = st.grid, det = st.det, fig = $("#fig");
  $("#stitle").textContent = g.kind === "grid" ? `GRID ${g.cols}x${g.rows}` : g.kind === "scene" ? `SCENE · ${String(g.mix).toUpperCase()}` : "UPLOAD";
  $("#ssub").title = $("#ssub").textContent = (g.kind === "grid" ? `seed ${g.seed} · ${g.mix.toLowerCase()}` : g.kind === "scene" ? `scene ${g.seed} · unseen background` : `${g.w} x ${g.h}`) + (det ? ` · detected with ${det.model} at ${det.conf.toFixed(2)}${det.degrade !== "None" ? " · " + det.degrade.toLowerCase() : ""}` : " · not detected yet");
  syncStageControls();
  fig.innerHTML = `<img src="${imgSrc()}" alt="">${svgFor(det, g.w, g.h, false)}`; requestAnimationFrame(() => fit(fig, $("#figbox"), g.w, g.h)); bindBoxes(fig);
  const rows = $("#srows");
  if (!det) { $("#sstats").innerHTML = "Press <b>Detect</b> to run the model."; rows.innerHTML = ""; $("#sdl").innerHTML = ""; return; }
  const t = det.tally;
  $("#sstats").innerHTML = t ? `<b>${t.correct} of ${t.n}</b> correct · <b>${t.named} of ${t.n}</b> named correctly · <b>${det.ms} ms</b>` : `<b>${det.boxes.length}</b> objects found · <b>${det.ms} ms</b>`;
  rowList(rows, st.filter, false); $("#sdl").innerHTML = dlLinks(det.det); if (st.sel != null) highlightIn(fig, st.sel);
}
function syncStageControls() {
  const c = st.cfg; if (!c) return;
  $("#sc-size").innerHTML = c.grids.map(g => `<option value="${esc(g)}" ${g === st.size ? "selected" : ""}>${g.startsWith("Single") ? "1x1" : g.split(" ")[0]}</option>`).join("");
  $("#sc-model").innerHTML = c.models.map(m => `<option ${m === st.model ? "selected" : ""}>${esc(m)}</option>`).join("");
  const thr = []; for (let v = .05; v <= .901; v += .05) thr.push(+v.toFixed(2));
  if (!thr.includes(+(+st.conf).toFixed(2))) thr.push(+(+st.conf).toFixed(2));
  const vc = +(+validatedConf()).toFixed(2); if (!thr.includes(vc)) thr.push(vc);
  $("#sc-thr").innerHTML = thr.sort((a, b) => a - b).map(v => `<option value="${v}" ${Math.abs(v - st.conf) < 1e-9 ? "selected" : ""}>${v.toFixed(2)}${Math.abs(v - vc) < 1e-9 ? " (validated)" : ""}</option>`).join("");
  $("#sc-seed").value = $("#seed").value;
}
$("#sc-size").onchange = e => { st.size = e.target.value; renderControls(); };
$("#sc-model").onchange = e => pickModel(e.target.value);
$("#sc-thr").onchange = e => { st.conf = +e.target.value; st.confAuto = Math.abs(st.conf - validatedConf()) < 1e-9; renderControls(); };
$("#sc-seed").oninput = e => { $("#seed").value = e.target.value; };
$("#sc-dice").onclick = e => { shock(e.currentTarget); $("#sc-seed").value = ""; $("#seed").value = ""; };
const dlLinks = id => ["png", "csv", "json"].map(k => `<a class="btn" href="/api/download/${id}/${k}">${k}</a>`).join("") + `<a class="btn" href="/api/snapshot/${id}">snapshot</a>`;

/* ================================================================ actions */
async function randomGrid(src) {
  endFocus(); stopCompound(true); FZ.cancel = false; shock(src); if (st.busy) return; setBusy(true, "building a grid of held-out test photos...");
  try { st.grid = await api("/api/grid", { size: st.size, sections: st.sections, mix: st.mix, layout: st.layout, seed: $("#seed").value.trim() || null });
    st.det = null; st.sel = null; st.lastSeed = st.grid.seed; $("#lastseed").textContent = "last seed " + st.grid.seed; syncConf();
    $("#pempty") && $("#pempty").remove(); lookAround(); renderPupil(false, true); renderGauge(); renderReview(); $("#insp").innerHTML = `<div class="lbl">select an error after detecting</div>`;
    if ($("#ov-stage").classList.contains("show")) renderStage();
    $("#status").textContent = `grid ready · seed ${st.grid.seed} · press DETECT or D`;
  } catch (e) { toast(e.message); $("#status").textContent = "error"; } finally { setBusy(false); setStep(); }
}
async function detect(src) {
  shock(src || $("#detect"), true); if (st.busy) return; if (!st.grid) return toast("Press RANDOM (or upload) first.");
  stopCompound(true);
  FZ.cancel = false; FZ.awaiting = false; try { SND.scan(); } catch (e) {} setBusy(true, "detecting..."); $("#pupil").classList.add("narrow"); zoomOut(); renderGauge();
  try {
    const req = api("/api/detect", { id: st.grid.id, model: st.model, conf: st.conf, cell: st.cell, degrade: st.degrade });
    await blinkClose(); const [det] = await Promise.all([req, sleep(calm() ? 0 : 160)]);
    await predecode(det.image || st.grid.image);                     // v3.4: decode the picture before anything moves
    st.det = det; st.sel = null; st.busy = false; st.cmp = null;
    const zooming = !calm() && !ovOpen() && !FZ.cancel;
    renderPupil(true); renderGauge(true); renderReview(); retireHints();   // all the page work happens before the camera moves
    if ($("#ov-stage").classList.contains("show")) renderStage();
    $("#status").textContent = `${det.model} · ${det.ms} ms${det.first_call ? " (incl. model loading)" : ""}`;
    await frames(2);
    try { SND.ticks(det.boxes.length, calm() ? 0 : .78, det.boxes.length > 30 ? .025 : .045); } catch (e) {}
    focusIn(); await blinkOpen();
    $("#pupil").classList.remove("narrow"); if (!zooming) dropCards(); focusResult(det);
  } catch (e) { toast(e.message); $("#status").textContent = "error"; endFocus(); await blinkOpen(); $("#pupil").classList.remove("narrow"); }
  finally { setBusy(false); renderGauge(false); setStep(); }
}
function setBusy(b, msg) { st.busy = b; ["#s-det", "#s-new"].forEach(s => $(s).disabled = b); $("#detect")?.classList.toggle("dis", b); if (msg) $("#status").textContent = msg; if (!b && st.det) renderGauge(false); else if (b) renderGauge(); }
$("#d-rand").onclick = e => randomGrid(e.currentTarget); $("#s-new").onclick = e => randomGrid(e.currentTarget);
$("#s-det").onclick = e => detect(e.currentTarget);
$("#pupil").onclick = () => { if (st.zoomed != null) return zoomOut(); openStage(); };
$("#s-close").onclick = () => { $("#ov-stage").classList.remove("show"); hideTip(); };
$$(".tabs .chip").forEach(c => c.onclick = () => { $$(".tabs .chip").forEach(x => x.classList.remove("on")); c.classList.add("on"); st.filter = c.dataset.f; renderStage(); });
/* upload, stress, files, about, snapshot, calm */
$("#d-up").onclick = e => { shock(e.currentTarget); $("#file").click(); };
function uploadFile(f, done) {
  if (!f) return; if (st.busy) return toast("Busy, try again in a moment.");
  if (!/^image\//.test(f.type || "") && !/\.(jpe?g|png|webp|bmp|gif|tiff?)$/i.test(f.name || "")) return toast("That is not an image file.");
  const rd = new FileReader();
  rd.onload = async () => { stopCompound(true); setBusy(true, "uploading..."); try { st.grid = await api("/api/upload", { image: rd.result, name: f.name }); st.det = null; st.sel = null;
      $("#pempty") && $("#pempty").remove(); lookAround(); renderPupil(false); renderGauge(); renderReview(); syncConf();
      if ($("#ov-stage").classList.contains("show")) renderStage(); $("#status").textContent = "image ready · press DETECT or D"; }
    catch (err) { toast(err.message); } finally { setBusy(false); setStep(); done && done(); } };
  rd.onerror = () => toast("Could not read that file."); rd.readAsDataURL(f); }
$("#file").onchange = e => uploadFile(e.target.files[0], () => e.target.value = "");
/* v3: drag an image anywhere onto the page; the eye lights up and takes it */
let dzDepth = 0; const hasFiles = e => [...(e.dataTransfer?.types || [])].includes("Files");
const dzOff = () => { dzDepth = 0; $("#eye").classList.remove("dz-on"); };
addEventListener("dragenter", e => { if (!hasFiles(e)) return; e.preventDefault(); dzDepth++; $("#eye").classList.add("dz-on"); });
addEventListener("dragover", e => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = "copy"; });
addEventListener("dragleave", e => { if (!hasFiles(e)) return; if (--dzDepth <= 0) dzOff(); });
addEventListener("drop", e => { if (!hasFiles(e)) return; e.preventDefault(); dzOff(); const fs = [...e.dataTransfer.files];
  if (fs.length > 1) toast("One image at a time: using the first."); uploadFile(fs[0]); });
$("#d-dl").onclick = e => { shock(e.currentTarget); if (!st.det) return toast("Detect first, then download."); $("#dls").innerHTML = dlLinks(st.det.det); $("#ov-dl").classList.add("show"); };
$("#d-snap").onclick = e => { shock(e.currentTarget); if (!st.det) return toast("Detect first, then take a snapshot."); location.href = `/api/snapshot/${st.det.det}`; };
$("#dlclose").onclick = () => $("#ov-dl").classList.remove("show");
function renderAbout() {
  const c = st.cfg; if (!c) return; const f = v => v == null ? "-" : (+v).toFixed(3), pc = v => v == null ? "-" : (100 * v).toFixed(1) + "%";
  const desc = { yolov8s: "Run 1 · YOLOv8-s from COCO, 56 epochs (early stop)", yolo26s: "Run 2 · YOLO26-s from COCO, 60 epochs", yolov8s_ft: "Run 3 · Run 1 + 15 epochs (sneaker rule)",
                 yolo26s_ft: "Run 4 · Run 2 + 15 epochs (sneaker rule)", yolo26s_ft2: "Run 5 · Run 4 + 15 epochs (four more rules, single photos, look-alikes): the Milestone 2 model",
                 yolo26s_boost: "boost · Run 5 + 30 epochs at 1,280 px: checked labels, rotation, scattered and cut-out scenes",
                 yolo26s_boost2: "boost2 · Run 5 + 30 epochs at 1,600 px on the second label pass" };
  const E = c.boost_eval, tn = { normal: "normal grids", rotated: "rotated grids", scattered: "scattered scenes", cutout: "cut-outs, any background", singles: "single photos" };
  // v3: before (M2) -> first improved model -> newest, all on the same corrected truth; the model chosen on validation in bold
  const N = c.eval_names || {}, sel = (c.sel_info || {}).selected || c.boost, three = E && E.tests && Object.values(E.tests).some(v => v.previous);
  const cols = three ? [["before", "M2"], ["previous", (N.previous || "").replace("yolo26s_", "")], ["after", (N.after || "").replace("yolo26s_", "")]]
                     : [["before", "M2"], ["after", (N.after || "after").replace("yolo26s_", "")]];
  const cell = (v, key, p) => cols.map(([k, nm]) => { const x = v[k] ? v[k][key] : null, t = x == null ? "-" : p === 1 ? pc(x) : p === 0 ? x : f(x);
    return ("yolo26s_" + nm) === sel ? `<b>${t}</b>` : t; }).join(" → ");
  const SI = c.sel_info, fm = x => x == null ? "-" : (+x).toFixed(4);
  const selp = SI ? `<p class="lbl" style="margin-top:8px">Chosen on validation by a rule fixed in advance (higher mAP@0.5:0.95 on the ${SI.images} validation images): ` +
      Object.entries(SI.models || {}).map(([m, v]) => `${esc(m.replace("yolo26s_", ""))} ${fm(v.mAP50_95)}`).join(" vs ") + ` → <b>${esc(String(SI.selected).replace("yolo26s_", ""))}</b>` +
      (SI.bootstrap ? ` (difference ${SI.bootstrap.diff >= 0 ? "+" : ""}${(+SI.bootstrap.diff).toFixed(4)}, bootstrap 95% interval ${(+SI.bootstrap.ci95[0]).toFixed(4)} to ${(+SI.bootstrap.ci95[1]).toFixed(4)})` : "") +
      `. Thresholds: ${SI.adopted ? `grids ${(+SI.grid).toFixed(2)}, scenes ${(+SI.scene).toFixed(2)} (two thresholds, chosen on validation)` : `${(+SI.grid).toFixed(2)} for all images`}.` +
      (SI.adopted ? " The table above uses the single threshold of the evaluation; the final settings are below." : "") + `</p>` : "";
  const T2 = SI && SI.tests_two, fin2 = T2 && E && E.tests ? `<h2 style="margin-top:18px">FINAL SETTINGS (${esc(String(SI.selected).replace("yolo26s_", ""))})</h2>
    <table><tr><th>test set</th><th>threshold</th><th>recall</th><th>precision</th><th>wrong IDs</th><th>fully correct</th></tr>` +
    Object.entries(T2).filter(([k]) => E.tests[k] && E.tests[k].before).map(([k, v]) => { const b = E.tests[k].before;
      return `<tr><td>${esc(tn[k] || k)}</td><td>${(+v.threshold).toFixed(2)}</td><td>${pc(b.recall)} → <b>${pc(v.recall)}</b></td><td>${pc(b.precision)} → <b>${pc(v.precision)}</b></td>` +
             `<td>${b.wrong_ids} → <b>${v.wrong_ids}</b></td><td>${pc(b.fully_correct)} → <b>${pc(v.fully_correct)}</b></td></tr>`; }).join("") +
    `</table><p class="lbl" style="margin-top:8px">The Milestone 2 model at its own threshold (0.10, no scene rule) → the selected model with its final thresholds and the scene duplicate rule, on the same human-checked boxes. At the final settings the Milestone 2 model scores 0.368 on cut-outs and 0.730 on scattered scenes (final report); grids are the same either way.</p>` : "";
  $("#aboutboost").innerHTML = E && E.tests ? `<table><tr><th>test set</th><th>objects</th><th>recall</th><th>precision</th><th>wrong IDs</th><th>fully correct</th><th>mAP@0.5</th></tr>` +
    Object.entries(E.tests).map(([k, v]) => `<tr><td>${esc(tn[k] || k)}</td><td>${v.after.objects}</td><td>${cell(v, "recall", 1)}</td><td>${cell(v, "precision", 1)}</td>` +
      `<td>${cell(v, "wrong_ids", 0)}</td><td>${cell(v, "fully_correct", 1)}</td><td>${cell(v, "mAP50")}</td></tr>`).join("") +
    `</table><p class="lbl" style="margin-top:8px">Columns per cell: ${cols.map(x => esc(x[1])).join(" → ")}. M2 = the selected Milestone 2 model (threshold 0.10); each improved model at its own threshold chosen on validation; ` +
    `every test set used once per model, on the same human-checked boxes.</p>` + selp + fin2
    : `<p class="lbl">The improved model has not been trained and evaluated yet (M2 extra/boost).</p>`;
  const order = Object.keys(c.model_info).sort((a, b) => (c.model_info[a].val_map ?? 0) - (c.model_info[b].val_map ?? 0));
  $("#aboutmodels").innerHTML = `<table><tr><th>model</th><th>val mAP@0.5</th><th>val mAP@.5:.95</th><th>test mAP@0.5</th><th>test mAP@.5:.95</th><th>precision</th><th>recall</th></tr>` +
    order.map(m => { const v = c.model_info[m], selx = (c.sel_info || {}).selected || c.boost, sel = m === selx, m2 = m === c.best;
      return `<tr style="${sel ? "color:var(--acc);font-weight:700" : ""}"><td>${esc(m)}${sel ? " ✓" : m2 ? " (M2 pick)" : ""}<br><span style="color:var(--dim);font-weight:400">${esc(desc[m] || "")}</span></td>` +
        `<td>${f(v.val_map50)}</td><td>${f(v.val_map)}</td><td>${f(v.test_map50)}</td><td>${f(v.test_map)}</td><td>${f(v.test_precision)}</td><td>${f(v.test_recall)}</td></tr>`; }).join("") +
    `</table><p class="lbl" style="margin-top:8px">Milestone 2 models: selected on validation (${esc(c.selection_rule || "highest validation mAP50-95")}), then tested once on human-verified test boxes. ` +
    `Improved models: validation = the extension's mixed validation set (grids, rotated grids, scattered and cut-out scenes), test = normal grids on the corrected truth, so their rows compare with each other, not with the Milestone 2 rows. ✓ = chosen on validation.</p>`;
  const sh = c.shapes || {}, names = { "1x1": "single photo", "2x2": "2x2 grid", "3x3": "3x3 grid", "4x4": "4x4 grid", "5x4": "5x4 grid", "5x5": "5x5 grid" };
  $("#aboutshapes").innerHTML = Object.keys(sh).length ? `<table><tr><th>input</th><th>images</th><th>objects found (recall)</th><th>precision</th><th>fully correct images</th></tr>` +
    Object.keys(names).filter(k => sh[k]).map(k => `<tr><td>${names[k]}</td><td>${sh[k].images ?? "-"}</td><td>${pc(sh[k].object_recall)}</td><td>${pc(sh[k].precision)}</td><td>${pc(sh[k].fully_correct)}</td></tr>`).join("") +
    `</table><p class="lbl" style="margin-top:8px">The Milestone 2 model on a second test set built only from test photos (the improved models are compared in Before vs after above). "Fully correct" is strict: every object right and no extra box.</p>` : `<p class="lbl">not available</p>`;
}
$("#d-about").onclick = e => { shock(e.currentTarget); renderAbout(); $("#ov-about").classList.add("show"); }; $("#aclose").onclick = () => $("#ov-about").classList.remove("show");
/* ---------- v3.4 COMPOUND view: votes of the other models; it never changes the answer or the score ---------- */
async function loadCompound() {
  const det = st.det; if (!det || st.cmpBusy) return;
  st.cmpBusy = true; $("#status").textContent = "COMPOUND: asking the other models...";
  try { const c = await api("/api/compound", { det: det.det }); if (st.det !== det) return;
    c.byk = {}; c.votes.forEach(v => c.byk[v.k] = v); c.byGt = {}; c.hints.forEach(h => { if (h.gt != null) c.byGt[h.gt] = h; }); st.cmp = c;
    const S = c.summary; $("#status").textContent = c.score ? `COMPOUND · ${c.score.columns.map(x => (SHORT[x.model] || x.label) + " " + (x.kind === "classifier" ? x.tally.named : x.tally.correct) + "/" + x.tally.n).join(" · ")}` : `COMPOUND · ${c.voters.length} models · all see the same on ${S.all_agree}/${S.objects}`;
    st.cmpBusy = false; renderLists(); renderP4(); maybeSplit();
  } catch (e) { toast("COMPOUND: " + e.message); } finally { st.cmpBusy = false; renderP4(); }
}
const SHORT = { yolo26s_boost: "boost", yolo26s_boost2: "boost2", yolo26s_ft2: "M2 model", yolov8s_ft: "Run 3", CLS: "classifier" };
const SUB = { yolo26s_boost: "YOLO26-s · 1,280 px", yolo26s_boost2: "YOLO26-s · 1,600 px", yolo26s_ft2: "YOLO26-s · 1,120 px", yolov8s_ft: "YOLOv8-s · 1,120 px", CLS: "ConvNeXt-Tiny · names cells" };
const cxCol = r => { const R = pal().res; return { correct: R["correct"], named: R["correct"], "box off": R["box off"], "wrong ID": R["wrong ID"], misnamed: R["wrong ID"], missed: R["missed"] }[r] || R["extra"]; };
const cmpLevel = v => !v.n_total ? "none" : v.n_agree === v.n_total ? "all" : v.n_agree === 0 ? "zero" : "some";
function cmpBadge(r) {
  const c = st.cmp; if (!st.cmpOn || !c || !st.det || c.det !== st.det.det) return "";
  const R = pal().res;
  if (c.score && r.gt != null) {                      // with true boxes: how many of the other models got this object right
    const o = c.score.objects.find(x => x.gt === r.gt); if (!o) return "";
    const others = Object.entries(o.verdicts).filter(([m]) => m !== c.model), right = others.filter(([, x]) => x && ["correct", "named"].includes(x.result));
    const col = right.length === others.length ? R["correct"] : right.length === 0 ? R["wrong ID"] : R["box off"];
    const tip = others.map(([m, x]) => `${SHORT[m] || m}: ${!x ? "-" : x.result === "named" ? "named it right" : x.result === "misnamed" ? "named it " + x.pred : x.result === "wrong ID" ? "called it " + x.pred : x.result}`).join(" · ");
    return `<span class="cmpb" style="color:${col}" title="${esc(tip)}">${right.length} of ${others.length} others right</span>`;
  }
  if (r.pred != null && c.byk[r.pred]) { const v = c.byk[r.pred], lv = cmpLevel(v), col = lv === "all" ? R["correct"] : lv === "zero" ? R["wrong ID"] : R["box off"];
    return `<span class="cmpb" style="color:${col}" title="${esc(v.votes.map(x => `${SHORT[x.model] || x.label}: ${x.vote === "agree" ? "same place, same name" : x.vote === "other" ? "same place, called it " + x.id : "not seen"}`).join(" · "))}">${v.n_agree} of ${v.n_total} see the same</span>`; }
  return "";
}
/* a score ring per model, in the EYE's own language: correct / box off / wrong ID / missed around the score */
function ringSvg(col, size, big, mini, bare) {
  const R = pal().res, t = col.tally, n = t.n || 1, cx = size / 2, rad = size / 2 - (big ? 9 : 5), sw = big ? 9 : 5, circ = 2 * Math.PI * rad;
  const parts = col.kind === "classifier" ? [["named", t.named], ["misnamed", n - t.named]]
    : col.kind === "agree" ? [["correct", t.agree], ["extra", n - t.agree]]
    : [["correct", t.correct], ["box off", t.box_off], ["wrong ID", t.wrong], ["missed", t.missed]];
  let off = 0, arcs = `<circle cx="${cx}" cy="${cx}" r="${rad}" fill="none" stroke="${pal().sw.hint}" stroke-opacity=".25" stroke-width="${sw}"/>`;
  for (const [k, v] of parts) { if (!v) continue; const L = circ * v / n;
    arcs += `<circle cx="${cx}" cy="${cx}" r="${rad}" fill="none" stroke="${cxCol(k)}" stroke-width="${sw}" stroke-dasharray="${Math.max(0, L - (big ? 2 : 1)).toFixed(2)} ${circ.toFixed(2)}" stroke-dashoffset="${(-off).toFixed(2)}" transform="rotate(-90 ${cx} ${cx})"><title>${k}: ${v}</title></circle>`; off += L; }
  const main = col.kind === "classifier" ? t.named : col.kind === "agree" ? t.agree : t.correct;
  if (bare) return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${arcs}</svg>`;
  return `<svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${arcs}<text x="${cx}" y="${cx + (big ? 2 : 1)}" text-anchor="middle" font-family="monospace" font-weight="700" font-size="${big ? 26 : 13}" fill="${pal().sw.text === "#7fa594" ? "#e6fff1" : "#0f2a1c"}">${main}<tspan font-size="${big ? 15 : 9}" font-weight="400" fill="${pal().sw.text}">/${n}</tspan></text>` +
    (big && !mini ? `<text x="${cx}" y="${cx + 22}" text-anchor="middle" font-family="monospace" font-size="11.5" fill="${pal().sw.text}">${col.kind === "classifier" ? "named" : col.kind === "agree" ? "same" : "correct"}</text>` : "") + `</svg>`;
}
function cxColumns(c) {     // the same column shape for truth (score) and for uploads (agreement with the answer)
  if (c.score) return c.score.columns;
  const n = c.votes.length, cols = [{ model: c.model, label: c.label, kind: "agree", tally: { n, agree: n } }];
  for (const v of c.voters) cols.push({ model: v.model, label: v.label, kind: "agree", ms: v.ms, tally: { n, agree: c.votes.filter(x => x.votes.find(y => y.model === v.model && y.vote === "agree")).length } });
  return cols;
}
function cmpSvg(W, H) {
  const P = pal(), big = W > 500, c = st.cmp, box = `<rect width="${W}" height="${H}" fill="${P.sw.bg}" rx="8"/>`;
  const msg = t => box + `<text x="12" y="28" font-size="12" fill="${P.sw.hint}" font-family="monospace">${t}</text>`;
  if (!st.det) return msg("detect first");
  if (!st.cmpOn) return msg("press COMPOUND in the left dock");
  if (st.cmpBusy || !c || c.det !== st.det.det) return msg("asking the other models...");
  const cols = cxColumns(c), key = x => x.kind === "classifier" ? x.tally.named : x.kind === "agree" ? x.tally.agree : x.tally.correct;
  const others = cols.slice(1).filter(x => x.kind !== "classifier"), best = others.length ? others.reduce((m, x) => key(x) > key(m) ? x : m, others[0]) : null;
  const show = [cols[0], best].filter(Boolean), size = big ? 120 : 64, y = big ? 44 : 28;
  let out = box + `<text x="12" y="${big ? 24 : 17}" font-size="${big ? 13 : 11}" fill="${P.sw.text}" font-family="monospace">${c.score ? "correct objects, this image" : "boxes seen the same way"}</text>`;
  show.forEach((col, i) => { const x = (big ? 40 : 18) + i * (size + (big ? 60 : 30));
    out += `<g transform="translate(${x},${y})">${ringSvg(col, size, big)}</g>` +
      `<text x="${x + size / 2}" y="${y + size + (big ? 22 : 14)}" text-anchor="middle" font-size="${big ? 13 : 10.5}" font-family="monospace" fill="${i === 0 ? P.res["correct"] : P.sw.text}">${i === 0 ? "the answer" : "best other: " + esc(SHORT[col.model] || col.label)}</text>`; });
  const more = cols.length - show.length, dx = (big ? 40 : 18) + show.length * (size + (big ? 60 : 30)) - (big ? 20 : 8), cy = y + size / 2;
  if (more > 0) out += `<g class="cxmore" style="cursor:pointer">` + [0, 1, 2].map(i => `<circle cx="${dx + i * (big ? 16 : 10)}" cy="${cy}" r="${big ? 4 : 2.6}" fill="${P.sw.text}"/>`).join("") +
    `<text x="${dx - (big ? 4 : 2)}" y="${cy + (big ? 26 : 17)}" font-size="${big ? 13 : 10.5}" font-family="monospace" fill="${P.res["correct"]}">view all ${cols.length} ⤢</text></g>`;
  return out;
}
function cxCell(x, isCls) {
  if (!x) return `<span class="lbl">-</span>`;
  const col = cxCol(x.result), g = (t, txt, ttl) => `<span class="cx-chip" title="${esc(ttl || "")}"><b style="background:${col}">${t}</b><span style="color:${col}">${esc(txt)}</span></span>`;
  if (x.result === "correct") return g("✓", "correct", x.iou != null ? `IoU ${(+x.iou).toFixed(2)} · confidence ${(+x.conf).toFixed(2)}` : "");
  if (x.result === "named") return g("✓", "named it", `confidence ${(+x.conf).toFixed(2)}`);
  if (x.result === "box off") return g("◐", `box off · ${x.iou != null ? (+x.iou).toFixed(2) : ""}`, "right name, but the box overlaps the truth by less than 0.50");
  if (x.result === "wrong ID" || x.result === "misnamed") return g("✕", `${x.pred}`, `called it ${x.pred}${x.pred_name ? " (" + x.pred_name + ")" : ""}`);
  if (x.result === "missed") return g("○", x.seen != null ? `missed · seen ${x.seen.toFixed(2)}` : "missed", x.seen != null ? "seen, but below this model's threshold" : "not seen at all");
  return g("·", x.result);
}
function cmpMini() {            // two real rings (the answer and the best other model), three dots, view all
  const c = st.cmp;
  if (!st.det) return `<div class="msg">detect first</div>`;
  if (!st.cmpOn) return `<div class="msg">press COMPOUND in the left dock</div>`;
  if (st.cmpBusy || !c || c.det !== st.det.det) return `<div class="msg">asking the other models...</div>`;
  const cols = cxColumns(c), key = x => x.kind === "classifier" ? x.tally.named : x.kind === "agree" ? x.tally.agree : x.tally.correct;
  const others = cols.slice(1).filter(x => x.kind !== "classifier"), best = others.length ? others.reduce((m, x) => key(x) > key(m) ? x : m, others[0]) : null;
  const ring = (col, lab, ans) => `<div class="rc">${ringSvg(col, 76, true, true)}<div class="rl${ans ? " ans" : ""}">${esc(lab)}</div></div>`;
  return ring(cols[0], "the answer", true) + (best ? ring(best, "best: " + (SHORT[best.model] || best.label), false) : "") +
    (cols.length > 2 ? `<div class="more" title="open all ${cols.length} models"><span class="dots"><i></i><i></i><i></i></span><span>all ${cols.length}</span></div>` : "");
}
function openCompoundBig(showAll) {
  const c = st.cmp;
  if (!st.det) return toast("Detect first, then switch on COMPOUND.");
  if (!st.cmpOn || !c || c.det !== st.det.det) return toast("Press COMPOUND in the left dock and wait for the models.");
  const cols = cxColumns(c), score = !!c.score;
  const key = col => col.kind === "classifier" ? col.tally.named : col.kind === "agree" ? col.tally.agree : col.tally.correct;
  const best = Math.max(...cols.filter(x => x.kind !== "classifier").map(key));
  const cards = cols.map((col, i) => `<div class="cx-card${col.kind !== "classifier" && key(col) === best && score ? " best" : ""}">${i === 0 ? `<span class="tag">THE ANSWER</span>` : ""}
      ${ringSvg(col, 112, true)}<div class="nm">${esc(SHORT[col.model] || col.label)}</div>
      <div class="ms">${esc(SUB[col.model] || "")}</div>
      <div class="ex">${[col.ms >= 1 ? (col.ms >= 1000 ? (col.ms / 1000).toFixed(1) + " s" : Math.round(col.ms) + " ms") : (i === 0 ? "the answer" : ""),
        score && col.kind !== "classifier" ? (col.tally.extra ? col.tally.extra + " extra box" + (col.tally.extra > 1 ? "es" : "") : "no extra boxes") : ""].filter(Boolean).join(" · ")}</div></div>`).join("");
  let rows = "", hidden = 0;
  if (score) {
    const objs = c.score.objects, contested = objs.filter(o => o.n_right < o.n_models), calm = objs.filter(o => o.n_right === o.n_models);
    const row = o => `<tr class="ans"><td>${esc(o.id)} <span class="lbl">${esc(o.name || "")}</span></td>${cols.map(col => `<td>${cxCell(o.verdicts[col.model], col.kind === "classifier")}</td>`).join("")}</tr>`;
    rows = contested.map(row).join("");
    if (calm.length) rows += showAll ? calm.map(row).join("") + `<tr class="cx-more"><td colspan="${cols.length + 1}" id="cxless">hide the ${calm.length} objects every model got right</td></tr>`
      : `<tr class="cx-more"><td colspan="${cols.length + 1}" id="cxmore">+ ${calm.length} more object${calm.length > 1 ? "s" : ""} every model got right · show</td></tr>`;
    if (!contested.length && !showAll) rows = `<tr class="cx-more"><td colspan="${cols.length + 1}" id="cxmore">every model got every object right · show all ${calm.length}</td></tr>`;
  } else {
    rows = c.votes.map(v => `<tr class="ans"><td>${esc(v.id)} <span class="lbl">${esc(v.name || "")} ${v.conf.toFixed(2)}</span></td><td>${cxCell({ result: "correct" }).replace("correct", "the answer")}</td>${
      v.votes.map(x => `<td>${x.vote === "agree" ? cxCell({ result: "correct" }).replace(">correct<", ">same<") : x.vote === "other" ? cxCell({ result: "wrong ID", pred: x.id, pred_name: x.name }) : cxCell({ result: "missed" }).replace("missed", "not seen")}</td>`).join("")}</tr>`).join("");
  }
  const R = pal().res, legend = score ? `<span style="color:${R["correct"]}">✓ correct</span><span style="color:${R["box off"]}">◐ right name, box off</span><span style="color:${R["wrong ID"]}">✕ called it something else</span><span style="color:${R["missed"]}">○ missed</span>`
    : `<span style="color:${R["correct"]}">✓ same place, same name</span><span style="color:${R["wrong ID"]}">✕ same place, another name</span><span style="color:${R["missed"]}">○ not seen</span>`;
  $("#bigpanel").innerHTML = `<div class="cx-head"><h2>COMPOUND</h2><span class="cx-sub">${score ? "how every model does on this image" : "how the other models see the answer's boxes (no true boxes here)"}</span>
      <button class="cx-i" id="cxi" title="what is this?">i</button></div>
    <div class="cx-info" id="cxinfo" hidden>
      <p>Like an insect's compound eye, many small eyes look at the same image. <b>${esc(c.label)}</b> gives the answer; ${c.voters.map(v => esc(v.label)).join(", ")} look at the same image, each with its own validated settings and the same rules.</p>
      <p>${score ? "Because this image has true boxes, every model is judged against the truth exactly like the answer: correct (right name, IoU ≥ 0.5), box off, called it something else, or missed. The classifier draws no boxes; it names the 224-px cell under each object." : "This image has no true boxes, so the other models are compared with the answer instead: a box in the same place (IoU ≥ 0.5) with the same name, another name, or nothing seen."} Nothing here changes the answer or the score.</p>
      <p>Why only a view: combining the models into one answer (COMPOUND as a model) was tested on validation under a rule fixed in advance and did not pass: it raised fully correct grids but lost recall on cut-out scenes and added boxes on empty backgrounds.</p></div>
    <div class="cx-rings">${cards}</div>
    <div class="cx-wrap"><table class="cx-tab"><tr><th>${score ? "true object" : "answer's box"}</th>${cols.map(col => `<th>${esc(SHORT[col.model] || col.label)}</th>`).join("")}</tr>${rows}</table></div>
    <div class="cx-legend">${legend}</div>
    <div class="row" style="margin-top:14px"><button class="close" id="bclose">close [esc]</button></div>`;
  $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
  $("#cxi").onclick = e => { const i = $("#cxinfo"); i.hidden = !i.hidden; e.currentTarget.classList.toggle("on", !i.hidden); };
  const m = $("#cxmore"), l = $("#cxless"); if (m) m.onclick = () => openCompoundBig(true); if (l) l.onclick = () => openCompoundBig(false);
}
/* ---------- v3.5 COMPOUND split: the pupil divides into one pupil per model (it explains; it never changes the answer) ---------- */
const SPL = { on: false, det: null, want: false };
function splitReady() { const c = st.cmp; return !!(st.cmpOn && st.det && st.grid && c && c.det === st.det.det && !st.busy && !FZ.on && !$("#ov-stage").classList.contains("show")); }
function maybeSplit() { if (SPL.want && !SPL.on && splitReady()) openSplit(); }
function startCompound() {                       // the COMPOUND button: break the pupil into one pupil per model
  if (!st.det || !st.grid) return toast("Detect first, then press COMPOUND to see every model's pupil.");
  if (st.busy) return;
  SPL.want = true; st.cmpOn = true; $("#d-cmp").classList.add("on"); st.p4 = "cmp";
  if (FZ.on) { endFocus(); setTimeout(maybeSplit, 950); }                 // back from the verdict first, then break
  if (st.cmp && st.cmp.det === st.det.det) { renderLists(); renderP4(); maybeSplit(); }
  else { renderLists(); renderP4(); loadCompound(); }                    // the models' answers arrive, then it breaks
}
function stopCompound(instant) {                  // back to the original single pupil
  const was = SPL.want || SPL.on; SPL.want = false; closeSplit(instant);
  st.cmpOn = false; $("#d-cmp")?.classList.remove("on"); if (st.p4 === "cmp") st.p4 = "sweep";
  if (was) { renderLists(); renderP4(); }
}
function splitView(c, col) {                       // a detection-shaped object for svgFor: the answer is the real detection
  if (col.model === c.model) return st.det;
  const v = (c.views || {})[col.model]; if (!v) return null;
  return { boxes: v.boxes, table: v.table, truths: st.det.truths || [], has_truth: !!st.det.has_truth };
}
function clsSvg(c, w, h) {
  const R = pal().res;
  return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">${(c.cls_cells || []).map(x => `<g class="box cxcell" data-gt="${x.gt}"><rect class="f" x="${x.x1 + 8}" y="${x.y1 + 8}" width="${x.x2 - x.x1 - 16}" height="${x.y2 - x.y1 - 16}" fill="none" stroke="${x.ok ? R["correct"] : R["wrong ID"]}" stroke-width="3" vector-effect="non-scaling-stroke"${x.ok ? "" : ' stroke-dasharray="7 5"'}><title>${x.ok ? "named right" : "named it " + esc(x.pred)}</title></rect></g>`).join("")}</svg>`;
}
const splitKey = col => col.kind === "classifier" ? col.tally.named : col.kind === "agree" ? col.tally.agree : col.tally.correct;
function splitSrc() {                          // one shared image URL for every pupil: decoded once, not five times
  const b64 = (st.det && st.det.image) || (st.grid && st.grid.image); if (!b64) return imgSrc();
  if (SPL.src && SPL.srcFor === b64) return SPL.src;
  try { if (SPL.src) URL.revokeObjectURL(SPL.src); const bin = atob(b64), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    SPL.src = URL.createObjectURL(new Blob([u], { type: "image/jpeg" })); SPL.srcFor = b64; return SPL.src; } catch (e) { return imgSrc(); }
}
async function openSplit() {
  const c = st.cmp, g = st.grid, E = $("#eye"), P = $("#pupil"); if (!c || !g || !E || !P || $("#multi")) return;
  const cols = cxColumns(c); if (cols.length < 2) return;
  const er = E.getBoundingClientRect(), pr = P.getBoundingClientRect(), n = cols.length, gap = 16, ring = 9;
  const avail = Math.min(er.width * .94, innerWidth - 24);
  const O = Math.max(96, Math.min(pr.width * 1.02, (avail - (n - 1) * gap) / n)), D = O - 2 * (ring + 5);
  const cy = pr.top + pr.height / 2 - er.top, total = n * O + (n - 1) * gap, x0 = (er.width - total) / 2;
  const lensK = G.LENS / (P.offsetWidth || pr.width), s = Math.min(D * lensK / g.w, D * lensK / g.h), iw = g.w * s, ih = g.h * s;
  const box = document.createElement("div"); box.id = "multi"; box.className = "multi"; box.style.visibility = "hidden"; const src = splitSrc();
  box.innerHTML = cols.map((col, i) => {
    const ov = col.kind === "classifier" ? clsSvg(c, g.w, g.h) : svgFor(splitView(c, col), g.w, g.h, false);
    return `<div class="mp" data-m="${col.model}" title="${esc(col.label)}: click to open" style="left:${(x0 + i * (O + gap)).toFixed(1)}px;top:${(cy - O / 2).toFixed(1)}px;width:${O.toFixed(1)}px;height:${O.toFixed(1)}px">
      <div class="mring">${ringSvg(col, O, true, true, true)}</div>
      <div class="mpi" style="width:${D.toFixed(1)}px;height:${D.toFixed(1)}px"><div class="mzoom" style="width:${iw.toFixed(1)}px;height:${ih.toFixed(1)}px"><img src="${src}" alt="" decoding="sync">${ov}</div></div>
      <div class="mcap"><b>${esc(SHORT[col.model] || col.label)}</b> ${col.kind === "agree" ? (i === 0 ? `${col.tally.n} box${col.tally.n === 1 ? "" : "es"}` : `${splitKey(col)}/${col.tally.n} same`) : `${splitKey(col)}/${col.tally.n}`}${i === 0 ? `<span class="tagans">ANSWER</span>` : ""}</div></div>`; }).join("") +
    `<div class="mtip" id="mtip" style="top:${(cy + O / 2 + 40).toFixed(1)}px"></div>`;
  E.appendChild(box);
  cols.forEach((col, i) => { if (col.kind === "classifier") return; const v = splitView(c, col), el = box.children[i]; if (!v) return;
    $$(".box", el).forEach(b => { const r = b.dataset.b != null ? (v.table || []).find(t => t.pred === +b.dataset.b) : (v.table || [])[+b.dataset.row]; if (r && r.gt != null) b.dataset.gt = r.gt; }); });
  await Promise.all([...box.querySelectorAll("img")].map(im => im.decode().catch(() => {})));   // decoded before anything moves
  if (!SPL.want || box.parentNode !== E) { box.remove(); return; }                               // cancelled while decoding
  const mps = [...box.querySelectorAll(".mp")], pcx = pr.left + pr.width / 2 - er.left, pcy = pr.top + pr.height / 2 - er.top, sD = pr.width / D, mid = (n - 1) / 2;
  mps.forEach((m, i) => m.style.zIndex = String(10 + n - Math.abs(i)));          // the answer's copy on top: at first it looks exactly like the pupil
  P.style.transition = "none"; P.classList.add("splitting"); document.body.classList.add("splitview"); SPL.on = true; SPL.det = st.det.det;
  if (!calm()) {
    box.classList.add("pre");
    mps.forEach(m => { const mx = parseFloat(m.style.left) + O / 2, my = parseFloat(m.style.top) + O / 2; m.style.transition = "none";
      m.style.transform = `translate(${(pcx - mx).toFixed(1)}px,${(pcy - my).toFixed(1)}px) scale(${sD.toFixed(3)})`; });
    box.style.visibility = ""; box.getBoundingClientRect();
    requestAnimationFrame(() => requestAnimationFrame(() => { box.classList.remove("pre");
      mps.forEach((m, i) => { m.style.transition = `transform .7s cubic-bezier(.22,.82,.26,1) ${(Math.abs(i - mid) * .025).toFixed(3)}s`; m.style.transform = ""; }); }));
    setTimeout(() => { mps.forEach(m => { m.style.willChange = ""; }); P.style.transition = ""; }, 1100);
  } else { box.style.visibility = ""; P.style.transition = ""; }
  box.addEventListener("mouseover", e => { const b = e.target.closest(".box"); if (!b || b.dataset.gt == null) return; splitLink(+b.dataset.gt); });
  box.addEventListener("mouseout", e => { const b = e.target.closest(".box"); if (b && !(e.relatedTarget && e.relatedTarget.closest && e.relatedTarget.closest(".box"))) splitLink(null); });
  mps.forEach(m => m.onclick = ev => { ev.stopPropagation(); if (m.dataset.m === c.model) openStage(); else openSplitBig(m.dataset.m); });
}
function splitLink(gt) {
  const box = $("#multi"), tip = $("#mtip"); if (!box) return;
  $$(".box.lk", box).forEach(b => b.classList.remove("lk"));
  if (gt == null) { box.classList.remove("linking"); tip && tip.classList.remove("on"); return; }
  $$(`.box[data-gt="${gt}"]`, box).forEach(b => b.classList.add("lk")); box.classList.add("linking");
  const c = st.cmp, o = c && c.score && c.score.objects.find(x => x.gt === gt); if (!o || !tip) return;
  const say = x => !x ? "-" : x.result === "named" ? "named it" : x.result === "misnamed" ? "called it " + x.pred : x.result === "wrong ID" ? "called it " + x.pred : x.result === "box off" ? "box off " + (x.iou != null ? (+x.iou).toFixed(2) : "") : x.result;
  tip.innerHTML = `<b>${esc(o.id)}</b> ${esc(o.name || "")} · ` + cxColumns(c).map(col => `${esc(SHORT[col.model] || col.label)}: <span style="color:${cxCol((o.verdicts[col.model] || {}).result)}">${esc(say(o.verdicts[col.model]))}</span>`).join(" · ");
  tip.classList.add("on");
}
function closeSplit(instant) {
  const box = $("#multi"), P = $("#pupil"); SPL.on = false;
  if (!box || instant || calm() || box.style.visibility === "hidden") { P && P.classList.remove("splitting"); document.body.classList.remove("splitview"); if (box) box.remove(); return; }
  box.id = "";
  P.classList.add("splitting"); document.body.classList.add("splitview");               // keep the pupil hidden until the copies have merged into it
  const er = $("#eye").getBoundingClientRect(), pr = P.getBoundingClientRect(), pcx = pr.left + pr.width / 2 - er.left, pcy = pr.top + pr.height / 2 - er.top;
  box.classList.add("pre");                                                                  // rings and names fade first
  [...box.querySelectorAll(".mp")].forEach(m => { const O = m.offsetWidth, mx = parseFloat(m.style.left) + O / 2, my = parseFloat(m.style.top) + O / 2, D = m.querySelector(".mpi").offsetWidth;
    m.style.willChange = "transform"; m.style.transition = "transform .55s cubic-bezier(.45,0,.2,1) .1s";
    m.style.transform = `translate(${(pcx - mx).toFixed(1)}px,${(pcy - my).toFixed(1)}px) scale(${(pr.width / D).toFixed(3)})`; });
  setTimeout(() => { P.style.transition = "none"; P.classList.remove("splitting"); document.body.classList.remove("splitview"); box.remove(); setTimeout(() => { P.style.transition = ""; }, 60); }, 760);
}
function openSplitBig(model) {
  const c = st.cmp, g = st.grid, col = c && cxColumns(c).find(x => x.model === model); if (!col) return;
  const maxW = Math.min(innerWidth * .8, 1040), maxH = innerHeight * .62, s = Math.min(maxW / g.w, maxH / g.h);
  const ov = col.kind === "classifier" ? clsSvg(c, g.w, g.h) : svgFor(splitView(c, col), g.w, g.h, false);
  const wrong = c.score ? c.score.objects.filter(o => { const x = o.verdicts[model]; return x && !["correct", "named"].includes(x.result); }) : [];
  $("#bigpanel").innerHTML = `<div class="cx-head"><h2>${esc((SHORT[model] || col.label).toUpperCase())}</h2><span class="cx-sub">${esc(SUB[model] || "")} · ${splitKey(col)}/${col.tally.n} ${col.kind === "classifier" ? "named" : "correct"} · its own ${col.kind === "classifier" ? "names" : "boxes"} on this image</span></div>
    <div style="display:flex;justify-content:center;margin:14px 0 10px"><div class="zoomer" style="width:${(g.w * s).toFixed(1)}px;height:${(g.h * s).toFixed(1)}px"><img src="${imgSrc()}" alt="">${ov}</div></div>
    ${c.score ? (wrong.length ? `<table class="cx-tab"><tr><th>object</th><th>${esc(SHORT[model] || col.label)}</th></tr>${wrong.map(o => `<tr class="ans"><td>${esc(o.id)} <span class="lbl">${esc(o.name || "")}</span></td><td>${cxCell(o.verdicts[model])}</td></tr>`).join("")}</table>` : `<p class="lbl" style="text-align:center">every object right</p>`) : ""}
    <div class="row" style="margin-top:12px"><button class="close" id="bclose">close [esc]</button></div>`;
  $("#ov-big").classList.add("show"); $("#bclose").onclick = () => $("#ov-big").classList.remove("show");
}
$("#d-cmp").onclick = () => { if (SPL.want || SPL.on) stopCompound(); else startCompound(); };
$("#d-calm").onclick = e => { document.body.classList.toggle("calm"); e.currentTarget.classList.toggle("on", calm()); follow = { x: 0, y: 0 }; applyFollow(); BG.restart(); };
if (matchMedia("(prefers-reduced-motion: reduce)").matches) { document.body.classList.add("calm"); $("#d-calm").classList.add("on"); }
$("#d-stress").onclick = e => { shock(e.currentTarget); $("#ov-stress").classList.add("show"); }; $("#sclose").onclick = () => $("#ov-stress").classList.remove("show");
$("#sn").oninput = e => $("#snv").textContent = e.target.value;
$("#srun").onclick = async e => {
  shock(e.currentTarget, true); const n = +$("#sn").value; $("#srun").disabled = true; $("#sprog").style.width = "0"; $("#sres").innerHTML = `<p class="lbl">running ${n} grids...</p>`;
  try { const { job } = await api("/api/stress", { size: st.size, n, sections: st.sections, mix: st.mix, degrade: st.degrade, model: st.model,
      conf: st.confAuto ? validatedConf("grid") : st.conf, cell: st.cell });   // stress tests are grids: the grid threshold unless set by hand
    let j; do { await sleep(400); j = await api(`/api/stress/${job}`); $("#sprog").style.width = (100 * j.done / j.total) + "%"; } while (!j.result && !j.error);
    if (j.error) throw new Error(j.error); const R = j.result, T = R.totals, pct = v => (100 * v / Math.max(1, T.n)).toFixed(1) + "%";
    const tbl = (h, rows) => `<table><tr><th>${h}</th><th>times</th></tr>${rows.length ? rows.map(r => `<tr><td>${esc(r[0])}</td><td>${r[1]}</td></tr>`).join("") : `<tr><td>none</td><td></td></tr>`}</table>`;
    $("#sres").innerHTML = `<div class="kpis"><div class="kpi"><div class="v">${pct(T.correct)}</div><div class="k">objects correct (${T.correct}/${T.n})</div></div>` +
      `<div class="kpi"><div class="v">${(100 * R.fully).toFixed(1)}%</div><div class="k">fully correct grids · 95% CI ${(100 * R.ci[0]).toFixed(1)}-${(100 * R.ci[1]).toFixed(1)}%</div></div>` +
      `<div class="kpi"><div class="v">${pct(T.named)}</div><div class="k">named correctly · ${R.ms.toFixed(0)} ms per grid</div></div></div>` +
      `<p class="lbl">${R.n_grids} × ${esc(R.size)} · ${esc(R.mix)} · ${esc(R.degrade)} · ${esc(R.model)} · box off ${T.box_off} · wrong ID ${T.wrong} · missed ${T.missed} · extra ${T.extra}</p>` +
      tbl("most often missed / wrong", R.hardest) + tbl("most common confusions", R.confusions) + tbl("failure reasons", R.reasons);
  } catch (err) { $("#sres").innerHTML = `<p style="color:#ff8fa8">${esc(err.message)}</p>`; } finally { $("#srun").disabled = false; }
};
/* hover titles: what each region is for */
const TITLES = [["#dock", '"What you want to see?"', "right"], ["#brow", '"How you want to see?"', "heading"],
                ["#eye", '"Eye will be your eyes!"', "eye"], ["#review", '"My Report Card"', "cards"], ["#dock2", '"Want me to help more?"', "left"]];
let titlesOn = (() => { try { return localStorage.getItem("m2titles") !== "off"; } catch (e) { return true; } })();
$("#d-titles").classList.toggle("on", titlesOn);
let typing = null, hoverEl = null;
function showTitle(el, text, where) {
  if (!titlesOn || document.body.classList.contains("intro")) return; const h = $("#htitle"), r = el.getBoundingClientRect(); hoverEl = el; clearInterval(typing);
  $(".top h1").classList.toggle("away", where === "heading"); $("#status").classList.toggle("away", where === "cards");
  h.textContent = ""; h.classList.add("show");
  const place = () => { const w = h.offsetWidth, hh = h.offsetHeight; let x, y;
    if (where === "right") { x = r.right + 16; y = r.top + r.height / 2 - hh / 2; }
    else if (where === "left") { x = r.left - w - 16; y = r.top + r.height / 2 - hh / 2; }
    else if (where === "eye") {                         // on the white of the eye, between its left corner and the iris, never over the iris
      const ir = $("#irisgrp").getBoundingClientRect(), right = ir.left - 14, left = r.left + r.width * .05;
      x = Math.min((left + right) / 2 - w / 2, right - w); y = r.top + r.height / 2 - hh / 2; }
    else if (where === "heading") { const t = $(".top h1").getBoundingClientRect(); x = t.left + t.width / 2 - w / 2; y = t.top + t.height / 2 - hh / 2; }   // takes the heading's place
    else if (where === "cards") {                       // centred in the gap between the eye's bottom and the cards, in place of the status line
      const e = $("#eye").getBoundingClientRect(), cx0 = e.left + e.width / 2; x = cx0 - w / 2;
      const under = $$("#review .tile").map(t => t.getBoundingClientRect()).filter(t => t.right > x && t.left < x + w);
      const cardsTop = under.length ? Math.min(...under.map(t => t.top)) : e.bottom + 60; y = (e.bottom + cardsTop) / 2 - hh / 2; }
    else { x = r.left + r.width / 2 - w / 2; y = r.top - hh - 8; }
    h.style.left = Math.max(8, Math.min(innerWidth - w - 8, x)) + "px"; h.style.top = Math.max(8, y) + "px"; };
  if (calm()) { h.textContent = text; place(); return; }
  let k = 0; typing = setInterval(() => { h.textContent = text.slice(0, ++k); place(); if (k >= text.length) clearInterval(typing); }, 34);
}
function hideTitle(el) { if (hoverEl !== el) return; clearInterval(typing); $("#htitle").classList.remove("show"); hoverEl = null;
  $(".top h1").classList.remove("away"); $("#status").classList.remove("away"); }
TITLES.forEach(([sel, text, where]) => { const el = $(sel); el.addEventListener("mouseenter", () => showTitle(el, text, where)); el.addEventListener("mouseleave", () => hideTitle(el)); });
$("#d-titles").onclick = e => { titlesOn = !titlesOn; e.currentTarget.classList.toggle("on", titlesOn); try { localStorage.setItem("m2titles", titlesOn ? "on" : "off"); } catch (err) { }
  if (!titlesOn) { clearInterval(typing); $("#htitle").classList.remove("show"); hoverEl = null; $(".top h1").classList.remove("away"); $("#status").classList.remove("away"); } };
/* light / dark theme (the choice is remembered) */
function syncThemeBtn() { $("#themelbl").textContent = isLight() ? "dark" : "light"; }
$("#d-theme").onclick = e => {
  shock(e.currentTarget); document.documentElement.classList.toggle("light"); syncThemeBtn();
  try { localStorage.setItem("m2theme", isLight() ? "light" : "dark"); } catch (err) { }
  layout(); BG.restart();                                                     // eye colours + pixel field
  if (st.det) renderP4();
  if ($("#ov-stage").classList.contains("show")) renderStage();
};
syncThemeBtn();
$("#howbtn").onclick = e => { shock(e.currentTarget); window.jyRelayout?.(); $("#jy").scrollIntoView({ behavior: calm() ? "auto" : "smooth" }); };
/* keys */
addEventListener("keydown", e => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) { if (e.key === "Escape") document.activeElement.blur(); return; }
  const k = e.key.toLowerCase();
  if (k === "escape") { const o = $$(".ov.show"); if (o.length) o.forEach(x => x.classList.remove("show")); else if (SPL.on || SPL.want) stopCompound(); else clearSel(); }
  else if (k === "c") $("#d-cmp").click();
  else if (k === "n") randomGrid($("#d-rand")); else if (k === "d") detect($("#detect")); else if (k === "u") $("#file").click();
});
let rz; addEventListener("resize", () => { const was = SPL.on; if (was) closeSplit(true); clearTimeout(rz); rz = setTimeout(() => { layout(); if ($("#ov-stage").classList.contains("show")) renderStage(); if (was) maybeSplit(); }, 120); });

/* ================================================================ start */
async function pollWarm() {
  try { const s = await api("/api/status"); st.warm = s.warm;
    if (s.warm === "ready") { $("#status").textContent = st.grid ? $("#status").textContent : "model ready"; renderGauge(); return; }
    if (s.warm === "error") { st.warm = "ready"; $("#status").textContent = "model loads on first detection"; renderGauge(); return; }
  } catch (e) { }
  setTimeout(pollWarm, 700);
}
(async () => {
  $$(".seg,.tile,.dbtn").forEach(glitchable);
  try { st.cfg = await api("/api/config"); st.model = st.cfg.boost || st.cfg.best; st.conf = validatedConf("grid");
    { const sn = $("#sn"), mx = st.cfg.stress_max || 300; sn.max = mx; if (+sn.value > mx) sn.value = Math.min(100, mx); $("#snv").textContent = sn.value; } st.sections = [...st.cfg.sections];
    Object.entries(st.cfg.groups || {}).forEach(([g, ids]) => ids.forEach(o => GROUPS[o] = g));
    const c = st.cfg, fmtn = v => v == null ? "-" : Number(v).toLocaleString("en-US"), nObj = c.n_objects_all ?? c.n_objects;
    $("#m-obj").textContent = `${nObj} objects`;
    $("#m-obj").dataset.tip = `${nObj} Object IDs (classes): the everyday objects photographed by the students of the course`;
    $("#m-photos").textContent = `${fmtn(c.n_photos)} photos`;
    $("#m-photos").dataset.tip = `${fmtn(c.n_photos)} photos in the cleaned dataset, after background-only shots and exact duplicates were removed`;
    $("#m-boxed").textContent = `${fmtn(c.n_boxed)} with boxes`;
    $("#m-boxed").dataset.tip = c.boxes_checked
      ? `${fmtn(c.n_boxed)} photos with a box checked by a person: every usable box in the dataset` + (() => {
          const left = (c.n_photos || 0) - c.n_boxed, d = c.n_dups_removed || 0, x = c.n_excluded_person || 0, rest = left - d - x;
          if (left <= 0 || rest < 0) return "";
          return ` (${fmtn(left)} of the ${fmtn(c.n_photos)} photos left out: ` + [d ? `${fmtn(d)} near-duplicate copies removed` : "", x ? `${fmtn(x)} excluded by a person` : "",
            rest ? `${fmtn(rest)} deliberately excluded earlier (a screenshot, photos without the object)` : ""].filter(Boolean).join(", ") + ")"; })()
      : `${fmtn(c.n_boxed)} photos with a reliable bounding box: the ones used to train and test the model` +
      (c.n_photos && c.n_boxed ? ` (${fmtn(c.n_photos - c.n_boxed)} left out because no reliable box could be made)` : "");
    try { window.fillJourney && window.fillJourney(c); } catch (err) { console.warn(err); }
    renderControls(); }
  catch (e) { toast("Cannot reach the server: " + e.message); }
  layout(); BG.init(); setStep(); pollWarm(); document.fonts?.ready.then(() => { layout(); layoutReview(); }); await wake();
})();
async function wake() {
  /* the eye wakes up with its name in the pupil; the name flies to the top; then everything else appears */
  const logo = $("#intrologo"), h1 = $(".top h1");
  if (!(calm() || matchMedia("(prefers-reduced-motion: reduce)").matches)) { try { SND.wake(); } catch (e) {} }
  if (calm() || matchMedia("(prefers-reduced-motion: reduce)").matches) { logo.remove(); document.body.classList.remove("intro"); setLids(1); return; }
  const fitLogo = () => { const pw = $("#pupil").clientWidth, lw = logo.offsetWidth || 1; logo.style.transform = `translate(-50%,-50%) scale(${(pw * .66 / lw).toFixed(3)})`; };
  fitLogo();
  await sleep(600);
  const open1 = lidsTo(.32, 1200, "inout");
  setTimeout(() => logo.classList.add("go"), 380);                           // letters pop in as the lids part
  await open1; await sleep(250);
  await lidsTo(.14, 500, "inout"); await sleep(180);
  const open2 = lidsTo(1, 1400, "out");
  setTimeout(() => logo.classList.add("sub"), 500);
  await open2;
  logo.classList.add("blink"); await sleep(820);                              // the E's blink, like a pair of eyes
  /* fly: a copy of the title travels from the pupil to the heading's place and shrinks into it */
  const from = logo.getBoundingClientRect(), to = h1.getBoundingClientRect();
  const fly = document.createElement("div"); fly.className = "logo flylogo"; fly.innerHTML = h1.innerHTML;
  Object.assign(fly.style, { left: to.left + "px", top: to.top + "px", width: to.width + "px", height: to.height + "px" });
  document.body.appendChild(fly); logo.style.visibility = "hidden";
  const k = from.width / to.width, dx = (from.left + from.width / 2) - (to.left + to.width / 2), dy = (from.top + from.height / 2) - (to.top + to.height / 2);
  const anim = fly.animate([{ transform: `translate(${dx}px,${dy}px) scale(${k})` }, { transform: `translate(${dx * .35}px,${dy * .55 - 24}px) scale(${(k + 1) / 2})`, offset: .55 }, { transform: "none" }],
                           { duration: 1050, easing: "cubic-bezier(.45,0,.2,1)", fill: "forwards" });
  await anim.finished.catch(() => {});
  const top = $(".top"); top.style.transition = "none"; top.style.opacity = "1"; top.style.translate = "0 0";   // the real heading takes over in place
  fly.remove(); logo.remove(); void top.offsetWidth; top.style.transition = ""; await sleep(60);
  const order = [...$$(".brow .seg").map((_, i) => `.brow .seg[data-i="${i}"]`), ".brow .arc", "#dock", "#dock2", ".detrow", ...$$(".review > *").map((_, i) => `.review > :nth-child(${i + 1})`)];
  order.forEach((sel, n) => $$(sel).forEach(el => el.style.transitionDelay = (n * 70) + "ms"));
  document.body.classList.remove("intro");
  setTimeout(() => { order.forEach(sel => $$(sel).forEach(el => el.style.transitionDelay = "")); top.style.opacity = ""; top.style.translate = ""; }, order.length * 70 + 900);
  setTimeout(layoutReview, order.length * 70 + 950);
}

/* ===== v3.3: sound - synthesised in the browser (no files); on by default; the dock button or Calm mutes it ===== */
const SND = (() => {
  let ctx = null, out = null;
  let on = true; try { on = localStorage.getItem("m2sound") !== "off"; } catch (e) {}
  let OFF = null;                                             // an OfflineAudioContext while rendering a preview
  const live = () => {
    if (OFF) return OFF;
    if (!on || calm()) return null;
    try { if (!ctx) { const C = window.AudioContext || window.webkitAudioContext; if (!C) return null; ctx = new C(); out = ctx.createGain(); out.gain.value = 0.32;
            const comp = ctx.createDynamicsCompressor(); out.connect(comp); comp.connect(ctx.destination); }
          if (ctx.state === "suspended") ctx.resume(); } catch (e) { return null; }
    return ctx.state === "running" ? ctx : null;              // browsers keep audio off until the first click or key press
  };
  const env = (g, t, a, peak, d) => { g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(peak, t + a); g.gain.exponentialRampToValueAtTime(0.0001, t + a + d); };
  const dest = () => OFF ? OFF.__out : out;
  function tone(f, at, dur, o = {}) { const c = live(); if (!c) return; const t = c.currentTime + (at || 0);
    const osc = c.createOscillator(), g = c.createGain(); osc.type = o.type || "sine"; osc.frequency.setValueAtTime(f, t);
    if (o.to) osc.frequency.exponentialRampToValueAtTime(o.to, t + dur); if (o.detune) osc.detune.value = o.detune;
    env(g, t, o.a || 0.008, o.gain || 0.2, dur); osc.connect(g); g.connect(dest()); osc.start(t); osc.stop(t + (o.a || 0.008) + dur + 0.05); }
  function noise(at, dur, f0, f1, gain, q) { const c = live(); if (!c) return; const t = c.currentTime + (at || 0);
    const len = Math.ceil(c.sampleRate * (dur + 0.1)), buf = c.createBuffer(1, len, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    const src = c.createBufferSource(), bp = c.createBiquadFilter(), g = c.createGain(); src.buffer = buf; bp.type = "bandpass"; bp.Q.value = q || 0.9;
    bp.frequency.setValueAtTime(f0, t); bp.frequency.exponentialRampToValueAtTime(f1, t + dur);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + dur * 0.55); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    src.connect(bp); bp.connect(g); g.connect(dest()); src.start(t); src.stop(t + dur + 0.05); }
  /* electric and metallic building blocks */
  function metal(f, at, dur, gain) {                        // a struck metal bar: inharmonic partials, fast decay, tiny detune
    [[1, 1], [2.756, .5], [5.404, .28], [8.933, .14]].forEach(([r, a], i) => tone(f * r, at, dur / (1 + i * .55), { gain: gain * a, a: 0.002, detune: (i % 2 ? 6 : -4) })); }
  function burst(at, len, hp, gain) {                       // one crackle / relay contact: a few ms of high-passed noise
    const c = live(); if (!c) return; const t = c.currentTime + at, n = Math.ceil(c.sampleRate * len), buf = c.createBuffer(1, n, c.sampleRate), d = buf.getChannelData(0);
    for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / n, 2.2);
    const src = c.createBufferSource(), f = c.createBiquadFilter(), g = c.createGain(); src.buffer = buf; f.type = "highpass"; f.frequency.value = hp; g.gain.value = gain;
    src.connect(f); f.connect(g); g.connect(dest()); src.start(t); }
  function hum(at, dur, gain) {                             // mains hum that charges up as its filter opens
    const c = live(); if (!c) return; const t = c.currentTime + at, lp = c.createBiquadFilter(), g = c.createGain();
    lp.type = "lowpass"; lp.Q.value = 6; lp.frequency.setValueAtTime(140, t); lp.frequency.exponentialRampToValueAtTime(2600, t + dur * .9);
    g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + dur * .7); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    [[55, "sawtooth", 1], [110.4, "sawtooth", .5], [165, "square", .18]].forEach(([f, ty, a]) => { const o = c.createOscillator(), og = c.createGain(); o.type = ty; o.frequency.value = f; og.gain.value = a;
      o.connect(og); og.connect(lp); o.start(t); o.stop(t + dur + .05); });
    lp.connect(g); g.connect(dest()); }
  return {
    get on() { return on; },
    async render(name, seconds, ...args) {
      const sr = 44100, oc = new OfflineAudioContext(1, Math.ceil(sr * seconds), sr), g = oc.createGain(); g.gain.value = 0.32;
      const comp = oc.createDynamicsCompressor(); g.connect(comp); comp.connect(oc.destination); oc.__out = g; OFF = oc;
      try { this[name](...args); } finally { OFF = null; }
      const buf = await oc.startRendering(); return Array.from(buf.getChannelData(0)); },
    set(v) { on = !!v; try { localStorage.setItem("m2sound", on ? "on" : "off"); } catch (e) {} if (on) live(); },
    unlock() { live(); },
    wake() {                                   // the eye wakes: a low swell, then a glassy chime as the pupil appears
      tone(55, 0, 3.2, { a: 1.4, gain: 0.16 }); tone(82.4, 0.15, 3.0, { a: 1.3, gain: 0.1 }); noise(0.1, 2.6, 180, 600, 0.03, 0.6);
      tone(880, 2.25, 1.4, { gain: 0.09 }); tone(1318.5, 2.3, 1.3, { gain: 0.06, detune: 4 }); tone(1760, 2.36, 1.1, { gain: 0.03 }); },
    scan() {                                   // charge-up: hum + arcing crackle, ending in a short metallic ping
      hum(0, 0.5, 0.075);
      for (let i = 0; i < 9; i++) burst(0.06 + Math.random() * 0.36, 0.003 + Math.random() * 0.004, 2400 + Math.random() * 2400, 0.05 + Math.random() * 0.05);
      metal(1568, 0.44, 0.55, 0.045); },
    ticks(n, base, step) {                     // one dry relay click per box, same pitch each time, a faint metal edge
      for (let i = 0; i < Math.min(n, 40); i++) { const at = base + step * i + (Math.random() - .5) * 0.006;
        burst(at, 0.004, 1800, 0.09); metal(2793 * (1 + (Math.random() - .5) * 0.02), at, 0.06, 0.008); } },
    zoomIn() { noise(0.12, 0.95, 260, 2600, 0.09, 0.8); tone(110, 0.12, 0.9, { to: 220, gain: 0.05, a: 0.3 }); },
    zoomOut() { noise(0, 0.75, 2400, 260, 0.075, 0.8); tone(220, 0, 0.7, { to: 110, gain: 0.04, a: 0.2 }); },
    verdict(good) { if (good) { tone(659.3, 0, 0.5, { gain: 0.12 }); tone(987.8, 0.12, 0.7, { gain: 0.11 }); tone(1318.5, 0.24, 0.8, { gain: 0.05 }); }
                    else { tone(523.3, 0, 0.5, { type: "triangle", gain: 0.1 }); tone(440, 0.14, 0.7, { type: "triangle", gain: 0.09 }); } },
  };
})();
["pointerdown", "keydown"].forEach(ev => addEventListener(ev, () => SND.unlock(), { capture: true }));
(() => { const b = $("#d-sound"); if (!b) return; b.classList.toggle("on", SND.on);
  b.onclick = e => { shock && shock(e.currentTarget); SND.set(!SND.on); b.classList.toggle("on", SND.on); if (SND.on) SND.verdict(true); }; })();
/* R replays the opening animation (with sound, once the page has been clicked) */
const INTRO_HTML = (() => { const l = $("#intrologo"); return l ? l.outerHTML : ""; })();
async function replayIntro() {
  if (!INTRO_HTML || st.busy || document.body.classList.contains("intro") || ovOpen()) return;
  endFocus(); clearSel && clearSel();
  $("#pupil").insertAdjacentHTML("afterbegin", INTRO_HTML);
  document.body.classList.add("intro"); setLids(0); await sleep(400); await wake();
}
addEventListener("keydown", e => { if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key.toLowerCase() === "r" && !e.ctrlKey && !e.metaKey && !e.altKey) replayIntro(); });


/* ===== v3.1: lean in on DETECT, show the verdict, step back; and a close button on every window ===== */
const FZ = { on: false, t: null, hover: false, pending: false };
const ovOpen = () => !!document.querySelector(".ov.show");
function focusIn() {                  // the camera: the whole eye grows as one layer (no frosted glass in it); the panels fade first
  if (FZ.cancel) return;
  if (calm() || ovOpen() || FZ.on) { clearTimeout(FZ.t); return; }
  clearTimeout(FZ.back);
  const E = $("#eye"), P = $("#pupil"), er = E.getBoundingClientRect(), zr = ($("#zoomer") || P).getBoundingClientRect(), pr = P.getBoundingClientRect();
  const W = innerWidth, H = innerHeight, SIDE = 200, GAP = 40;
  const Z = $("#zoomer") || P, zh = Z.offsetHeight, zw = Z.offsetWidth;   // layout sizes: not affected by the blink's running animation
  const c = { x: pr.left + pr.width / 2, y: pr.top + pr.height / 2 };
  const pd = P.offsetWidth;                                             // the pupil's black disk must fit between the two sides of the verdict
  const s = Math.max(1.05, Math.min(4, (0.85 * H) / zh, (W - 2 * (SIDE + GAP)) / pd));
  E.style.transformOrigin = `${(c.x - er.left).toFixed(1)}px ${(c.y - er.top).toFixed(1)}px`;
  E.style.willChange = "transform"; const bg0 = $("#bg"); if (bg0) bg0.style.willChange = "transform";   // layers first, one frame before the move
  const move = `translate(${(W / 2 - c.x).toFixed(1)}px, ${(H / 2 - c.y).toFixed(1)}px) scale(${s.toFixed(3)})`;
  requestAnimationFrame(() => { if (!FZ.on) return; E.style.transition = "transform .95s cubic-bezier(.55,0,.25,1) .15s"; E.style.transform = move; });
  FZ.img = { left: W / 2 - pd * s / 2, right: W / 2 + pd * s / 2, top: H / 2 - zh * s / 2, height: zh * s };   // the verdict sits beside the disk
  const bg = $("#bg"); if (bg) { bg.style.transition = "transform 1.1s cubic-bezier(.55,0,.25,1) .15s"; bg.style.transformOrigin = `${c.x.toFixed(0)}px ${c.y.toFixed(0)}px`; bg.style.transform = "scale(1.12)"; }
  try { BG.pause(1350); } catch (e) {}
  document.documentElement.classList.add("focusing"); document.body.classList.add("focus"); FZ.on = true; FZ.awaiting = true; fpsMeasure("zoom in");
  try { SND.zoomIn(); } catch (e) {}
  clearTimeout(FZ.mv); FZ.mv = setTimeout(() => { E.style.willChange = ""; }, 1250);   // drop the layer on arrival: re-drawn sharp
}
function endFocus() {
  clearTimeout(FZ.t); FZ.t = null; FZ.pending = false; FZ.armed = false; FZ.hover = false;
  const was = FZ.on; FZ.on = false; document.body.classList.remove("focus");
  if (FZ.awaiting) { FZ.cancel = true; FZ.awaiting = false; }          // left before the verdict: it must not appear on its own later
  ["#fbanner", "#fbL", "#fbR", "#fr-x"].forEach(q => $(q) && $(q).classList.remove("show"));
  const E = $("#eye"), bg = $("#bg");
  if (was) { try { BG.pause(950); } catch (e) {} E.style.willChange = "transform"; E.style.transition = "transform .8s cubic-bezier(.45,0,.2,1)"; E.style.transform = ""; clearTimeout(FZ.mv); fpsMeasure("zoom out");
    try { SND.zoomOut(); } catch (e) {}
    if (bg) { bg.style.transition = "transform .8s cubic-bezier(.45,0,.2,1)"; bg.style.transform = ""; }
    FZ.mv = setTimeout(() => { E.style.willChange = ""; E.style.transition = ""; }, 900); }
  clearTimeout(FZ.back);
  if (was && !calm()) FZ.back = setTimeout(() => { if (!FZ.on) document.documentElement.classList.remove("focusing"); }, 900);
  else document.documentElement.classList.remove("focusing");
}
const FPSON = new URLSearchParams(location.search).has("fps");
function fpsMeasure(label) {
  if (!FPSON) return; const f = []; let go = true;
  requestAnimationFrame(function tick(t) { f.push(t); if (go) requestAnimationFrame(tick); });
  setTimeout(() => { go = false; const d = f.slice(1).map((t, i) => t - f[i]); if (!d.length) return;
    const avg = 1000 / (d.reduce((a, b) => a + b, 0) / d.length), worst = Math.max(...d);
    let el = $("#fpsbox"); if (!el) { el = document.createElement("div"); el.id = "fpsbox"; document.body.append(el); }
    el.textContent = `${label}: ${avg.toFixed(0)} fps · worst frame ${worst.toFixed(0)} ms`; }, 1050);
}
function verdict(det) {
  if (!det.has_truth) { const k = det.boxes.length; return [String(k), `<b>${k}</b> object${k === 1 ? "" : "s"} detected (an upload has no true boxes to score against)`, false]; }
  const t = det.tally, found = t.n - t.missed, bits = [`<b>${found}</b> detected`, `<b>${t.named}</b> named correctly`, `${t.missed ? '<span class="bad">' : ""}<b>${t.missed}</b> missed${t.missed ? "</span>" : ""}`];
  if (t.wrong) bits.push(`<span class="bad"><b>${t.wrong}</b> wrong ID</span>`); if (t.box_off) bits.push(`<span class="bad"><b>${t.box_off}</b> box off</span>`);
  if (t.extra) bits.push(`<span class="bad"><b>${t.extra}</b> extra box${t.extra === 1 ? "" : "es"}</span>`);
  const head = t.fully ? "fully correct · " : "";
  return [`${t.correct}/${t.n}`, `${head}${t.n} objects: ${bits.join(", ")}`, !t.fully];
}
function sideVerdict(det) {
  const L = [], R = [];
  if (!det.has_truth) { const k = det.boxes.length; L.push(String(k), k === 1 ? "object detected" : "objects detected"); R.push(["", k, `box${k === 1 ? "" : "es"} drawn`], ["extra", "", "no true boxes to score an upload"]); return [L, R]; }
  const t = det.tally;
  L.push(`${t.correct}/${t.n}`, t.fully ? "<b>fully correct</b>" : `correct · <span class="bad">${t.n - t.correct} to check</span>`);
  R.push(["", t.correct, "correct"]);
  if (t.box_off) R.push(["off", t.box_off, "box off"]); if (t.wrong) R.push(["wrong", t.wrong, "wrong ID"]);
  if (t.missed) R.push(["missed", t.missed, "missed"]); if (t.extra) R.push(["extra", t.extra, `extra box${t.extra === 1 ? "" : "es"}`]);
  return [L, R];
}
function singleName(det) {
  const nm = (id, name) => `${esc(id || "")}${name ? " · " + esc(name) : ""}`;
  if (!det.has_truth) {
    if (det.boxes.length === 0) return `<span class="miss">EYE cannot find anything, sorry.</span><small>try a lower threshold, or a photo with the object larger and upright</small>`;
    if (det.boxes.length === 1) { const b = det.boxes[0]; return `<span class="ok">${nm(b.id, b.name)}</span><small>confidence ${(+b.conf).toFixed(2)}</small>`; }
    return null;
  }
  if (!(det.cols === 1 && det.rows === 1)) return null;
  const r = (det.table || []).find(x => x.true) || null; if (!r) return null;
  const T = nm(r.true, r.true_name), P = nm(r.predicted, r.predicted_name), c = r.confidence != null ? ` · confidence ${(+r.confidence).toFixed(2)}` : "";
  switch (r.result) {
    case "correct":  return `<span class="ok">${T}</span><small>found correctly${c}</small>`;
    case "box off":  return `<span class="off">${T}</span><small>named right, box placed off${c}</small>`;
    case "wrong ID": return `<span class="wrong">seen as ${P}</span><small>it is ${T}</small>`;
    case "missed":   return `<span class="miss">EYE cannot find it, sorry.</span><small>it was ${T}</small>`;
  }
  return null;
}
function placeVerdict() {                       // each side centred in its own space, beside the image
  const im = FZ.on && FZ.img ? FZ.img : ($("#zoomer") || $("#pupil")).getBoundingClientRect(), cy = im.top + im.height / 2, L = $("#fbL"), R = $("#fbR");
  const lx = im.left / 2, rx = im.right + (innerWidth - im.right) / 2;
  L.style.right = "auto"; L.style.left = `${Math.max(10, lx - L.offsetWidth / 2).toFixed(0)}px`; L.style.top = `${(cy - L.offsetHeight / 2).toFixed(0)}px`;
  R.style.left = `${Math.min(innerWidth - R.offsetWidth - 10, rx - R.offsetWidth / 2).toFixed(0)}px`; R.style.top = `${(cy - R.offsetHeight / 2).toFixed(0)}px`;
}
function focusResult(det) {
  if (ovOpen() || FZ.cancel) { FZ.cancel = false; return; }
  const segs = $$("#gauge path").length, wait = calm() ? 0 : Math.max(1300, 700 + segs * (segs > 20 ? 35 : 60) + 250);
  clearTimeout(FZ.t);
  FZ.t = setTimeout(() => {
    FZ.awaiting = false; if (FZ.cancel) { FZ.cancel = false; return; }
    const [L, R] = sideVerdict(det);
    $("#fl-score").textContent = L[0]; $("#fl-lbl").innerHTML = L[1];
    const one = singleName(det), fn = $("#fl-name"); fn.innerHTML = one || ""; fn.classList.toggle("show", !!one);
    $("#fr-list").innerHTML = R.map(([k, n, txt], i) => `<li class="${k}" style="transition-delay:${(i * 70)}ms"><i></i><b>${n}</b><span>${txt}</span></li>`).join("");
    try { SND.verdict(!det.has_truth || det.tally.fully); } catch (e) {}
    placeVerdict(); $("#fbL").classList.add("show"); $("#fbR").classList.add("show"); $("#fr-x").classList.add("show"); $("#fbanner").classList.add("show");
    FZ.hover = false; FZ.armed = true;
    const errs = det.has_truth && !det.tally.fully;
    const leave = () => { if (FZ.hover) { FZ.pending = true; return; } endFocus(); };
    FZ.t = setTimeout(leave, errs ? 4200 : 3000);
  }, wait);
}
["#pupil", "#fbL", "#fbR"].forEach(q => { const el = $(q);   // only a mouse that MOVES over the eye after the verdict holds the view (a cursor resting on DETECT does not)
  el.addEventListener("mousemove", () => { if (FZ.armed) FZ.hover = true; });
  el.addEventListener("mouseleave", () => { FZ.hover = false; if (FZ.pending) { FZ.pending = false; FZ.t = setTimeout(endFocus, 900); } }); });
// "outside" means anywhere except the image and the verdict; during detection a click cancels the coming zoom
document.addEventListener("click", e => { if (st.busy && !FZ.on && !e.target.closest("#detect, #pupil")) FZ.cancel = true; }, true);
document.addEventListener("click", e => { if ((FZ.on || $("#fbR").classList.contains("show")) && !e.target.closest("#pupil, #fbanner, #fbL, #fbR, #fr-x")) endFocus(); }, true);
document.addEventListener("keydown", e => { if (e.key !== "Escape") return; if (st.busy && !FZ.on) FZ.cancel = true; if (FZ.on || $("#fbR").classList.contains("show")) endFocus(); });
addEventListener("resize", () => { if (FZ.on) endFocus(); });
document.body.append($("#fbanner")); document.body.append($("#fbL")); document.body.append($("#fbR")); document.body.append($("#fr-x"));
$("#fr-x").onclick = e => { e.stopPropagation(); endFocus(); };
$("#fb-x").onclick = e => { e.stopPropagation(); endFocus(); };
$("#jy-x").onclick = () => $("#jy-up").click();
function addX(panel) {
  if (!panel) return;
  const bar = panel.classList.contains("stage") ? panel.querySelector(".stagebar") : null, host = bar || panel;
  if (bar) {   // the large view: its own "close [esc]" button becomes the ×, so its control row keeps the same width
    const c = bar.querySelector("#s-close"); if (c && !c.classList.contains("xclose")) { c.classList.remove("close"); c.classList.add("xclose", "inbar"); c.textContent = "×"; c.title = "Close (Esc)"; c.setAttribute("aria-label", "Close"); }
    return; }
  if (host.querySelector(":scope > .xclose")) return;
  const b = document.createElement("button"); b.type = "button"; b.className = "xclose" + (bar ? " inbar" : ""); b.textContent = "×";
  b.title = "Close (Esc)"; b.setAttribute("aria-label", "Close");
  b.onclick = e => { e.stopPropagation();
    const c = [...panel.querySelectorAll("button.close")].find(x => /^close/i.test(x.textContent.trim()));
    if (c) c.click(); else panel.closest(".ov").classList.remove("show"); };
  if (bar) bar.append(b); else panel.prepend(b);
}
document.querySelectorAll(".ov").forEach(ov => {
  const panel = ov.querySelector(".panel");
  new MutationObserver(() => { if (ov.classList.contains("show")) { addX(panel); endFocus(); } }).observe(ov, { attributes: true, attributeFilter: ["class"] });
  new MutationObserver(() => addX(panel)).observe(panel, { childList: true });
  addX(panel);
});
</script>

<script>
(() => {
"use strict";


/* ---------- small drawing helpers (blueprint line art) ---------- */
const W = (d, c = "w") => `<path class="${c}" d="${d}"/>`;
const R = (x, y, w, h, c = "w", rx = 2) => `<rect class="${c}" x="${x}" y="${y}" width="${w}" height="${h}" rx="${rx}"/>`;
const C = (x, y, r, c = "w") => `<circle class="${c}" cx="${x}" cy="${y}" r="${r}"/>`;
const T = (x, y, s, c = "t", a = "middle") => `<text class="${c}" x="${x}" y="${y}" text-anchor="${a}">${s}</text>`;
function worker(x, y, o = {}) {           /* a builder in a hard hat; y = feet */
  const arm = o.arm || "M-10 -26 L-16 -14", arm2 = o.arm2 || "M10 -26 L16 -14";
  return `<g transform="translate(${x},${y})${o.flip ? " scale(-1,1)" : ""}">
    ${W("M0 -40 L0 -18")}${W("M0 -18 L-8 0")}${W("M0 -18 L8 0")}${W("M0 -34 L-10 -26")}${W(arm)}${W("M0 -34 L10 -26")}${W(arm2)}
    ${C(0, -47, 7)}<path class="y" d="M-9 -49 A9 9 0 0 1 9 -49 L11 -48 L-11 -48 Z"/>${o.extra || ""}</g>`;
}
const ground = (y = 168) => W(`M8 ${y} L292 ${y}`) + W(`M8 ${y + 6} L40 ${y + 6} M60 ${y + 6} L120 ${y + 6} M150 ${y + 6} L230 ${y + 6} M250 ${y + 6} L292 ${y + 6}`, "faint");
const photo = (x, y, s = 16, cross = false) => R(x, y, s, s * .78) + (cross ? W(`M${x + 3} ${y + 3} L${x + s - 3} ${y + s * .78 - 3} M${x + s - 3} ${y + 3} L${x + 3} ${y + s * .78 - 3}`) : C(x + s * .5, y + s * .4, s * .18));
const svg = inner => `<svg class="scene sc" viewBox="0 0 300 190" aria-hidden="true">${inner}</svg>`;

/* ---------- the ten scenes ---------- */
const scenes = [
  /* 1 clearing the site: broom, wheelbarrow of rejected photos, neat stack kept */
  svg(ground() + worker(70, 168, { arm2: "M10 -26 L26 -6", extra: W("M26 -6 L40 6") + `<path class="y" d="M34 2 L48 10 L44 14 L30 6 Z"/>` }) +
    photo(112, 156, 14, true) + photo(132, 160, 12, true) + photo(96, 160, 12, true) +
    W("M168 150 L222 150 L214 168 L176 168 Z") + C(190, 172, 5) + W("M222 150 L238 140") + photo(176, 136, 14, true) + photo(194, 134, 14, true) + photo(186, 124, 14, true) +
    photo(250, 150, 18) + photo(250, 135, 18) + photo(250, 120, 18) + photo(250, 105, 18) + T(259, 96, "6,942") +
    T(150, 40, "Clearing 385 empty shots and duplicates") ),
  /* 2 architects at the drafting table: plan with boxes, three drafters */
  svg(ground() + W("M90 120 L210 104 L218 130 L98 146 Z") + W("M120 140 L116 168 M196 128 L202 168") +
    `<g transform="rotate(-7 154 125)">` + R(112, 110, 22, 18, "d", 1) + C(123, 119, 4) + R(142, 108, 28, 20, "d", 1) + C(156, 118, 5) + R(178, 106, 20, 16, "d", 1) + C(188, 114, 3) + `</g>` +
    worker(56, 168, { arm2: "M10 -26 L34 -48" }) + worker(244, 168, { flip: true, arm2: "M10 -26 L30 -50" }) + worker(270, 168, { flip: true, arm2: "M10 -26 L22 -40" }) +
    T(150, 40, "Three drafters, one plan per photo") + `<path class="ys" d="M146 66 l6 6 l12 -14"/>` ),
  /* 3 surveyor: tripod, tape measure bracketing a pair of shoes, rule book */
  svg(ground() + W("M60 168 L72 120 L84 168 M72 120 L72 168") + R(62, 108, 22, 12) + C(73, 114, 3, "wf") + worker(42, 168, { arm2: "M10 -26 L22 -30" }) +
    W("M130 160 Q140 146 156 150 L160 160 Z") + W("M166 160 Q176 146 192 150 L196 160 Z") +
    R(122, 134, 82, 30, "d", 2) + W("M122 120 L204 120") + W("M122 116 L122 124 M204 116 L204 124 M143 118 L143 122 M163 118 L163 122 M183 118 L183 122", "w") +
    R(228, 126, 44, 38, "w", 2) + W("M236 136 L264 136 M236 144 L264 144 M236 152 L256 152") + `<rect class="y" x="228" y="126" width="6" height="38"/>` +
    T(163, 108, "one box, both shoes") + T(150, 40, "Every object gets one written rule") ),
  /* 4 bricklaying: a wall of photo bricks growing from 2x2 to 5x5 */
  svg(ground() + (() => { let s = ""; const sizes = [2, 3, 4, 5]; let x = 84;
      sizes.forEach((n, k) => { const cell = 46 / n, base = 168; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) s += R(x + c * cell, base - (r + 1) * cell, cell - 1.5, cell - 1.5, "w", 1);
        s += T(x + n * cell / 2, base - n * cell - 6, `${n}×${n}`); x += n * cell + 10; }); return s; })() +
    worker(54, 168, { arm2: "M10 -26 L28 -28", extra: `<rect class="y" x="26" y="-34" width="14" height="9" rx="1"/>` }) + T(150, 40, "Photos laid side by side like bricks") ),
  /* 5 foundation: chute pouring concrete onto a slab labelled COCO, rebar */
  svg(ground(170) + R(70, 140, 200, 30, "w", 2) + (() => { let s = ""; for (let x = 90; x < 262; x += 20) if (x < 118 || x > 222) s += W(`M${x} 140 L${x - 6} 170`, "faint"); return s; })() +
    T(170, 160, "COCO, 118k photos") + W("M210 60 L240 60 L250 82 L226 110") + W("M226 110 L214 136", "ys") + W("M222 110 L210 136", "ys") +
    `<path class="y" d="M196 136 Q210 126 226 136 Z"/>` + worker(44, 170, { arm2: "M10 -26 L24 -18", extra: W("M24 -18 L24 0") }) + T(150, 40, "A foundation of general vision") ),
  /* 6 tower crane lifting a floor onto a rising building: 60 floors */
  svg(ground() + W("M70 168 L70 46 M84 168 L84 46") + (() => { let s = ""; for (let y = 56; y < 168; y += 16) s += W(`M70 ${y} L84 ${y + 16} M84 ${y} L70 ${y + 16}`, "faint"); return s; })() +
    W("M40 46 L250 46 M70 46 L77 30 L84 46") + R(40, 46, 18, 12) + W("M200 46 L200 76") + `<rect class="y" x="174" y="76" width="52" height="7" rx="1"/>` +
    (() => { let s = ""; for (let k = 0; k < 6; k++) s += R(160, 160 - k * 13, 76, 12, "w", 1); return s; })() + T(198, 102, "floor 60") +
    T(150, 24, "Raising the detector, one epoch per floor") ),
  /* 7 finishing crew: ladder, paint roller, two coats */
  svg(ground() + R(150, 60, 110, 108, "w", 2) + `<rect class="y" x="150" y="60" width="54" height="108" opacity=".85"/>` + W("M204 60 L204 168", "d") +
    T(177, 118, "coat 1", "ty") + T(232, 118, "coat 2") + W("M96 168 L118 70 M122 168 L140 76") + W("M100 150 L124 150 M105 128 L128 128 M110 106 L133 106 M115 86 L136 86") +
    worker(126, 104, { arm2: "M10 -26 L22 -36", extra: W("M22 -36 L30 -40") + `<rect class="y" x="28" y="-48" width="8" height="16" rx="2"/>` }) + T(150, 40, "Two finishing coats: fine-tune 1 and 2") ),
  /* 8 judging five scale models: one wins the rosette */
  svg(ground() + W("M40 140 L262 140 M52 140 L52 168 M250 140 L250 168") +
    [[56, 30], [96, 38], [136, 37], [176, 46], [216, 58]].map(([x, h], k) => R(x, 140 - h, 26, h, "w", 1) + W(`M${x + 6} ${140 - h + 8} L${x + 6} ${140 - h + 14} M${x + 18} ${140 - h + 8} L${x + 18} ${140 - h + 14}`, "faint") +
      T(x + 13, 156, String(k + 1))).join("") + T(118, 60, "winner: Run 5, YOLO26-s") +
    C(229, 72, 9, "y") + W("M224 80 L220 96 M234 80 L238 96", "ys") + worker(276, 140, { flip: true, arm2: "M10 -26 L20 -38", extra: R(16, -46, 12, 14) }) +
    T(150, 36, "Five runs, judged on validation") ),
  /* 9 final inspection: certificate with stamp 0.947 */
  svg(ground() + R(92, 50, 144, 112, "w", 3) + W("M104 66 L224 66 M104 78 L210 78 M104 90 L218 90") + C(201, 124, 30, "y") +
    T(201, 121, "0.947", "ty") + T(201, 136, "mAP@.5", "ty") + T(133, 118, "1,041", "t") + T(133, 134, "objects") +
    worker(64, 168, { arm2: "M10 -26 L28 -32", extra: R(26, -42, 12, 15) }) + worker(250, 168, { flip: true }) + T(150, 40, "Tested once, on boxes checked by hand") ),
  /* 10 inspector with a magnifier checking every brick: ticks on a stack of photos */
  svg(ground() + (() => { let s = ""; for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) { const x = 120 + c * 24, y = 152 - r * 22;
      s += R(x, y, 20, 16, "w", 1) + `<path class="ys" d="M${x + 5} ${y + 8} l3 3 l6 -7"/>`; } return s; })() +
    C(104, 100, 14) + W("M94 110 L80 126") + worker(60, 168, { arm2: "M10 -26 L22 -40" }) + worker(278, 168, { flip: true, arm2: "M10 -26 L22 -22", extra: R(20, -30, 12, 15) }) +
    T(150, 40, "Every box checked by a person") ),
  /* 11 renovation for any weather: tilted and scattered photos lifted onto new backgrounds */
  svg(ground() + R(150, 92, 120, 76, "d", 3) + `<g transform="rotate(-18 186 128)">` + R(174, 116, 24, 18) + C(186, 125, 4) + `</g>` +
    `<g transform="rotate(24 238 140)">` + R(226, 130, 22, 17) + C(237, 138, 3) + `</g>` + `<g transform="rotate(90 206 110)">` + R(196, 102, 20, 16) + C(206, 110, 3) + `</g>` +
    `<path class="y" d="M240 104 q8 -10 16 0 q-8 10 -16 0 Z"/>` + W("M70 168 L70 50 M70 50 L176 50 M70 64 L84 50") + W("M150 50 L150 70", "ys") +
    `<g transform="rotate(-12 150 80)">` + R(138, 70, 24, 18, "y", 1) + `</g>` + worker(44, 168, { arm2: "M10 -26 L24 -40" }) +
    T(150, 30, "Rotated, scattered, cut out") ),
  /* 12 grand opening: building with the eye, ribbon, scissors */
  svg(ground() + R(108, 58, 120, 110, "w", 3) + W("M126 92 Q168 66 210 92 Q168 118 126 92 Z") + C(168, 92, 11) + C(168, 92, 4, "wf") + R(152, 128, 32, 40, "w", 1) +
    W("M60 146 L276 146", "ys") + W("M88 140 L100 152 M100 140 L88 152") + C(86, 138, 3) + C(86, 154, 3) +
    worker(52, 168, { arm2: "M10 -26 L30 -26" }) + worker(254, 168, { flip: true, arm2: "M10 -26 L20 -44" }) + worker(282, 168, { flip: true, arm2: "M10 -26 L18 -46" }) +
    T(150, 40, "Open to visitors: The EYE") )
];

/* ---------- content: front line + back report (real numbers from the milestone) ---------- */
const S = [
  { t: "Clear the site", f: "Cleaned 7,343 raw files down to 6,942 usable photos of 73 objects.",
    b: ["Removed 385 background-only shots and 16 exact duplicates before splitting, so no exact copy can sit in two splits.",
        "Kept the split from Milestone 1: 70 / 15 / 15, stratified by object, seed 42.",
        "Excluded a phone screenshot (OBJ054_017) and five OBJ004 photos that show no toy car."],
    g: [["6,942", "photos kept"], ["73", "objects, 89 to 106 photos each"], ["4,859 / 1,041 / 1,042", "train / val / test"]], w: "A clean site means nothing leaks between training and testing." },
  { t: "Draw the plans", f: "Three methods drew a box on every photo; a person checked the test set.",
    b: ["<a href=\"https://arxiv.org/abs/2306.09683\" target=\"_blank\" rel=\"noopener\">OWLv2</a> found each object from its name (17 minutes for 6,936 photos, 16× faster after a GPU memory fix).",
        "A visual search with example crops (hand-drawn for 7 objects) mostly boxed the floor, so it became one vote among three.",
        "A <a href=\"https://en.wikipedia.org/wiki/You_Only_Look_Once\" target=\"_blank\" rel=\"noopener\">YOLO</a> helper trained on training photos only cast the third vote: 3,172 strong, 2,935 auto, 829 to review.",
        "Every test photo was checked by hand; the accepted automatic boxes were 98.9% right."],
    g: [["1,968", "boxes drawn or confirmed by a person"], ["98.9%", "accepted automatic boxes correct (test)"], ["97.1%", "random training audit"]], w: "Trustworthy test boxes make every reported number trustworthy." },
  { t: "Survey and set the rules", f: "Where training showed weak objects, compared their boxes side by side and fixed five inconsistent definitions.",
    b: ["Found in two rounds, after Runs 1–2 and Runs 3–4. Sneakers: one box around both shoes, not one shoe.",
        "Tissue: the whole box, not just the paper (AP 0.30 → 0.995).",
        "Analog watch: case and strap (0.59 → 0.81). Power bank: the silver one only (0.71 → 0.95). Paddle: not the tennis racket (0.67 → 0.59: this one did not help until Milestone 3).",
        "314 photos (4.5%) were left out: 6 that are not photos of their object and 308 without a reliable box; no object lost more than 20%."],
    g: [["5", "label rules corrected"], ["0.30 → 0.995", "tissue AP after the fix"], ["6,628", "photos with boxes"]], w: "Consistent labels mattered more than the choice of model." },
  { t: "Lay the bricks", f: "Built 1,500 training images from photos placed side by side, 2×2 to 5×5.",
    b: ["Every photo is a 224-pixel brick; grids range from 448 to 1,120 pixels, and each box is shifted to its cell.",
        "In the final training set, 15% single photos, so single images are detected too; half of the grids put look-alikes side by side.",
        "Flips, colour changes and mosaics of four images on the fly; 20% of images baked with blur, low resolution, compression or darkness.",
        "Validation and test grids use every photo exactly once; a check confirms no photo crosses splits."],
    g: [["1,500", "training images"], ["5", "grid sizes, 4 to 25 objects"], ["10", "look-alike groups"]], w: "The grids are the milestone's input format, built exactly to spec." },
  { t: "Pour the foundation", f: "Started from models that already know how to see: pretrained on COCO.",
    b: ["<a href=\"https://arxiv.org/abs/1405.0312\" target=\"_blank\" rel=\"noopener\">COCO</a>: about 118,000 everyday photos in 80 categories, giving edges, shapes and 'objectness' for free.",
        "Two <a href=\"https://en.wikipedia.org/wiki/You_Only_Look_Once\" target=\"_blank\" rel=\"noopener\">YOLO</a> candidates: <a href=\"https://docs.ultralytics.com/models/yolov8/\" target=\"_blank\" rel=\"noopener\">YOLOv8-s</a>, the brief's model, and <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a>, the newest from Ultralytics.",
        "Both are <a href=\"https://en.wikipedia.org/wiki/Convolutional_neural_network\" target=\"_blank\" rel=\"noopener\">convolutional neural networks</a> built for <a href=\"https://en.wikipedia.org/wiki/Object_detection\" target=\"_blank\" rel=\"noopener\">object detection</a>: one look, every box at once.",
        "Input size 1,120 pixels, so photos in 5×5 grids keep their native 224 pixels."],
    g: [["118k", "COCO photos behind the start"], ["2", "architectures compared"], ["1,120 px", "input size"]], w: "Learning 73 new objects is far easier on a strong foundation." },
  { t: "Raise the floors", f: "Trained both detectors for up to 60 epochs on a laptop GPU.",
    b: ["RTX 5070 laptop GPU, 8 GB: batch size 4 at 1,120 pixels, cosine learning rate, early stopping after 12 flat epochs.",
        "YOLO26-s: 60 epochs in 104 minutes. YOLOv8-s: stopped after 56 epochs, 70 minutes.",
        "Chosen on validation: confidence 0.10, one box per 224-pixel cell, and small images placed at natural size on the canvas."],
    g: [["60", "epochs (YOLO26-s)"], ["104 min", "training time"], ["0.921", "first test mAP@0.5"]], w: "The structure stands; now it needs finishing." },
  { t: "Two finishing coats", f: "Fine-tuned twice, 15 epochs each, after improving the data.",
    b: ["Coat 1: both models, after the sneaker rule fix.",
        "Coat 2: YOLO26-s, after adding single photos, look-alike groups and four more label fixes.",
        "In total about 135,000 image passes for the final model."],
    g: [["15 + 15", "fine-tune epochs"], ["0.921 → 0.947", "test mAP@0.5"], ["90", "epochs in total"]], w: "Better data, not more epochs, produced most of the gain." },
  { t: "Judge the models", f: "Compared all five runs on validation and chose before looking at the test set.",
    b: ["Rule set in advance: highest validation mAP@0.5:0.95; threshold and post-processing chosen the same way.",
        "Winner: Run 5, <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a> fine-tuned twice. All five below; scroll for the table."],
    more: "<table class=\"mt\"><tr><th>model</th><th>val mAP@.5:.95</th><th>test mAP@0.5</th></tr>" +
      "<tr><td>Run 1 <a href=\"https://docs.ultralytics.com/models/yolov8/\" target=\"_blank\" rel=\"noopener\">YOLOv8-s</a></td><td>0.897</td><td>0.910</td></tr><tr><td>Run 2 <a href=\"https://docs.ultralytics.com/models/yolo26/\" target=\"_blank\" rel=\"noopener\">YOLO26-s</a></td><td>0.907</td><td>0.921</td></tr>" +
      "<tr><td>Run 3 YOLOv8-s, fine-tuned</td><td>0.906</td><td>0.932</td></tr><tr><td>Run 4 YOLO26-s, fine-tuned</td><td>0.919</td><td>0.931</td></tr>" +
      "<tr class=\"win\"><td>Run 5 YOLO26-s, fine-tuned twice</td><td>0.930</td><td>0.947</td></tr></table>",
    g: [["5", "models judged"], ["0.930", "winning validation score"], ["0", "peeks at the test set"]], w: "Choosing on validation keeps the final test an honest exam." },
  { t: "Final inspection", f: "Tested once on 1,041 objects whose boxes were checked by a person.",
    b: ["mAP@0.5 = 0.947, mAP@0.5:0.95 = 0.875, precision 0.931, recall 0.918.",
        "Single photos: 14 of 14 correct. Wrong Object IDs: only 16 of 1,041 objects.",
        "Known limits: tiny objects, look-alikes, co-occurring objects, degraded and rotated photos. Scroll for details."],
    more: "<table class=\"mt\"><tr><th>grid</th><th>objects found</th><th>fully correct</th></tr>" +
      "<tr><td>2×2</td><td>98.4%</td><td>87.5%</td></tr><tr><td>3×3</td><td>89.7%</td><td>35.7%</td></tr><tr><td>4×4</td><td>91.1%</td><td>14.3%</td></tr>" +
      "<tr><td>5×4</td><td>91.8%</td><td>21.4%</td></tr><tr><td>5×5</td><td>92.0%</td><td>7.1%</td></tr><tr class=\"win\"><td>all 72 grids</td><td>91.8%</td><td>34.7%</td></tr></table>" +
      "<p>Fully correct is strict: every object right and no extra box (95% CI 23.6 to 45.8%).</p>" +
      "<p>48 of 73 objects reach AP ≥ 0.95. Weakest: paddle 0.59, card wallet 0.62, mouth-guard case 0.77, sunglasses 0.79, all look-alikes or small.</p>" +
      "<p>Heavily degraded images (¼ resolution, dark, JPEG 20): mAP@0.5 drops to 0.428.</p>",
    g: [["0.947", "mAP@0.5"], ["14 / 14", "single photos"], ["16", "wrong IDs in 1,041"]], w: "A strong result, measured the honest way." },
  { t: "Check every brick", f: "A person checked every usable box in the dataset, twice where needed.",
    b: ["A label-check tool walked through all 6,936 photos: suspects, a sweep of every automatic box, box-size disagreements, the rules, left-out photos, second objects.",
        "One rule for all: box the <b>whole</b> object, every visible part, tight to its edges.",
        "A second pass used the improved detector as a second opinion: 200 decisions, 41 boxes corrected (17 of them in the test set).",
        "183 near-duplicate pairs reviewed; 41 training and validation copies removed, always keeping the test photo, so those pairs no longer cross training and test."],
    g: [["6,881", "boxes, all checked by a person"], ["423", "boxes corrected by hand"], ["41", "near-duplicate copies removed"]],
    w: "The answer key itself was checked, box by box." },
  { t: "Renovate for any weather", f: "Taught the detector rotated photos, scattered layouts and any background.",
    b: ["The diagnosis came first: turned 90° or 270°, recall fell to about 0.77 and wrong IDs rose from 5 to 70.",
        "3,000 new training images: rotated and tilted grids, photos scattered on real backgrounds (232 for training; test scenes use 100 unseen ones), and 6,752 objects cut out with <a href=\"https://arxiv.org/abs/2408.00714\" target=\"_blank\" rel=\"noopener\">SAM</a> and placed at any angle.",
        "Two improved models, both fine-tuned from the Milestone 2 model for 30 epochs: boost at 1,280 px and boost2 at 1,600 px.",
        "<span id=\"jy-x-sel\">The model is chosen on validation by a rule fixed in advance, with one threshold for grids and one for scenes.</span>",
        "Each model tested once on new test sets from held-out photos on unseen backgrounds (built like the training scenes, so somewhat optimistic). Scroll for the table.",
        "Before → after compares the Milestone 2 model at its own threshold with the chosen model at its final settings; at the final settings the Milestone 2 model finds 0.368 of cut-out objects, as in the final report."],
    more: "<div id=\"jy-x-table\"><p>The evaluation table appears here once the improved models are evaluated.</p></div>",
    g: [["-", "cut-out recall"], ["-", "rotated-grid recall"], ["-", "normal-grid recall"]],
    w: "Same photos, harder conditions: the detector now works beyond the grid." },
  { t: "Open to visitors", f: "Explained every result in The EYE, the Explainable YOLO Explorer.",
    b: ["Random test grids, rotated grids, scattered and cut-out scenes, look-alike challenges and degraded images on demand, with every error explained.",
        "Every model side by side in one list, each with its own validated thresholds; drop any photo onto the eye to try it.",
        "Stress tests over hundreds of grids, a threshold sweep and a box-tightness chart per image, per-object inspection and snapshots for the report.",
        "Shared with the course through a link from the laptop, with a user manual for every control; a lightweight replica runs on Hugging Face.",
        "COMPOUND: every model judged on the same image, as score rings and one row per object; it explains and never changes the answer.",
        "A report for every milestone and a final report that checks every number, and the code on GitHub: code only, no data."],
    g: [["12", "construction stations"], ["5", "models compared in COMPOUND"], ["4", "reports"]], w: "A model people can see into is a model people can trust." }
];

window.fillJourney = c => {
  const $j = id => document.getElementById(id), r3 = v => v == null ? "-" : (+v).toFixed(2), fmtn = v => Number(v).toLocaleString("en-US");
  const figs = t => { const k = S.findIndex(x => x.t === t); return k < 0 ? [] : [...document.querySelectorAll("#jy .stop")[k].querySelectorAll(".fig b")]; };
  if (c.boxes_checked && c.n_boxed) { const b = figs("Check every brick")[0]; if (b) b.textContent = fmtn(c.n_boxed); }
  const E = c.boost_eval, N = c.eval_names || {}, sel = (c.sel_info || {}).selected || c.boost;
  if (!E || !E.tests) return;
  const k = sel === N.after ? "after" : sel === N.previous ? "previous" : "after", nm = (k === "after" ? N.after : N.previous || N.after || "").replace("yolo26s_", "");
  const T2 = (c.sel_info || {}).tests_two || null, fin = t => (T2 && T2[t]) || (E.tests[t] && E.tests[t][k]);   // v3.3: the final settings, as in the report
  const pair = t => E.tests[t] && E.tests[t].before && fin(t) ? `${r3(E.tests[t].before.recall)}→${r3(fin(t).recall)}` : "-";
  const fb = figs("Renovate for any weather"); ["cutout", "rotated", "normal"].forEach((t, i) => { if (fb[i]) fb[i].textContent = pair(t); });
  const SI = c.sel_info;
  if (SI && SI.models) $j("jy-x-sel").innerHTML = `Chosen on validation by a rule fixed in advance: ` + Object.entries(SI.models).map(([m, v]) => `${m.replace("yolo26s_", "")} ${(+v.mAP50_95).toFixed(4)}`).join(" vs ") +
    ` (mAP@0.5:0.95) → <b>${String(SI.selected).replace("yolo26s_", "")}</b>; thresholds ${SI.adopted ? `${(+SI.grid).toFixed(2)} for grids, ${(+SI.scene).toFixed(2)} for scenes` : `${(+SI.grid).toFixed(2)} for all images`}.`;
  const tn = { normal: "normal grids", rotated: "rotated grids", scattered: "scattered", cutout: "cut-outs", singles: "single photos" };
  $j("jy-x-table").innerHTML = `<table class="mt"><tr><th>test set (M2 → ${nm})</th><th>recall</th><th>wrong IDs</th><th>mAP@0.5</th></tr>` +
    Object.entries(E.tests).filter(([t, v]) => v.before && fin(t)).map(([t, v]) => `<tr><td>${tn[t] || t}</td><td>${r3(v.before.recall)} → <b>${r3(fin(t).recall)}</b></td>` +
      `<td>${v.before.wrong_ids} → ${fin(t).wrong_ids}</td><td>${r3(v.before.mAP50)} → ${r3((v[k] || {}).mAP50)}</td></tr>`).join("") + `</table>` +
    (T2 ? `<p class="lbl" style="margin-top:6px">With the final settings: thresholds ${(+c.sel_info.grid).toFixed(2)} for grids and single photos, ${(+c.sel_info.scene).toFixed(2)} for scenes.</p>` : "");
};

/* ---------- build ---------- */
const track = document.getElementById("jy-track"), map = document.getElementById("jy-map");
const truckSVG = `<svg class="truck" id="jy-truck" viewBox="0 0 64 38" aria-hidden="true"><rect x="2" y="8" width="38" height="20" rx="2" style="fill:var(--hivis);stroke:var(--truckline)" stroke-width="1.5"/>
  <path d="M40 14 H52 L60 22 V28 H40 Z" style="fill:var(--hivis);stroke:var(--truckline)" stroke-width="1.5"/><rect x="44" y="16" width="8" height="6" style="fill:var(--bp)"/>
  <path d="M6 12 L36 12" style="stroke:var(--truckline)" stroke-width="1" stroke-dasharray="3 3"/><circle cx="14" cy="30" r="5" style="fill:var(--truckline)"/><circle cx="50" cy="30" r="5" style="fill:var(--truckline)"/>
  <circle cx="14" cy="30" r="2" style="fill:var(--hivis)"/><circle cx="50" cy="30" r="2" style="fill:var(--hivis)"/></svg>`;
S.forEach((s, i) => {
  const stop = document.createElement("div"); stop.className = "stop"; stop.dataset.i = i;
  stop.innerHTML = `<div class="card" role="button" tabindex="0" aria-label="Station ${i + 1}: ${s.t}. Press to flip for the full report." aria-pressed="false">
    <div class="inner">
      <div class="face front"><div class="tape"></div><div class="num">${String(i + 1).padStart(2, "0")}</div>${scenes[i]}
        <h2>${s.t}</h2><p class="line">${s.f}</p>
        <div class="hint"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M4 12a8 8 0 0 1 14-5.3M20 12a8 8 0 0 1-14 5.3"/><path d="M18 3v4h-4M6 21v-4h4"/></svg>Flip for the site report</div></div>
      <div class="face back"><span class="stamp">Station ${i + 1} of ${S.length}</span>
        <div class="bhead"><span class="bnum">${String(i + 1).padStart(2, "0")}</span><h3>${s.t}</h3></div>
        <ul>${s.b.map(x => `<li>${x}</li>`).join("")}</ul>${s.more ? `<div class="more">${s.more}</div>` : ""}
        <div class="figs">${s.g.map(([v, l]) => `<div class="fig"><b>${v}</b><span>${l}</span></div>`).join("")}</div>
        <p class="why">${s.w}</p></div>
    </div></div>
    <div class="road"><div class="post">${i + 1}</div></div>`;
  track.appendChild(stop);
  const m = document.createElement("div"); m.className = "mstop"; m.dataset.i = i;
  m.innerHTML = `<button class="mdot" type="button" aria-label="Go to station ${i + 1}: ${s.t}">${i + 1}</button><span class="mname">${s.t}</span><span class="mline"></span>`;
  map.appendChild(m);
});
track.insertAdjacentHTML("beforeend", truckSVG);
const stops = [...track.querySelectorAll(".stop")], cards = [...track.querySelectorAll(".card")], truck = document.getElementById("jy-truck");
let cur = 0, progT = 0;
function setCurrent(i, scroll = true) {
  cur = Math.max(0, Math.min(stops.length - 1, i));
  map.querySelectorAll(".mstop").forEach((m, k) => { m.classList.toggle("on", k === cur); m.querySelector(".mdot").classList.toggle("on", k === cur); m.querySelector(".mdot").classList.toggle("done", k < cur); });
  document.getElementById("jy-count").textContent = `${cur + 1} / ${stops.length}`;
  const st = stops[cur], post = st.querySelector(".post"), road = st.querySelector(".road");
  const x = st.offsetLeft + post.offsetLeft - 70, y = road.offsetTop + 8;          // parks just before the post
  truck.style.top = y + "px"; truck.style.transform = `translateX(${x}px)`;
  if (scroll) { progT = Date.now(); } if (scroll) track.scrollTo({ left: st.offsetLeft - 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  { const m = map.querySelectorAll(".mstop")[cur];                          // scroll only the station map (never the page)
    map.scrollTo({ left: m.offsetLeft - map.clientWidth / 2 + m.offsetWidth / 2, behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }); }
}
function flip(card) { const on = !card.classList.contains("flipped"); card.classList.toggle("flipped", on); card.setAttribute("aria-pressed", on); }
cards.forEach((c, i) => {
  c.addEventListener("click", e => { if (e.target.closest("a") || e.target.closest(".more") && e.target.closest(".more").scrollHeight > e.target.closest(".more").clientHeight && e.detail === 0) return;
    if (e.target.closest("a")) return; if (i !== cur) setCurrent(i); flip(c); });
  c.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); if (i !== cur) setCurrent(i); flip(c); } });
});
map.querySelectorAll(".mdot").forEach((d, i) => d.addEventListener("click", () => setCurrent(i)));
document.getElementById("jy-prev").onclick = () => setCurrent(cur - 1);
document.getElementById("jy-next").onclick = () => setCurrent(cur + 1);
document.getElementById("jy-flipall").onclick = () => { const anyFront = cards.some(c => !c.classList.contains("flipped")); cards.forEach(c => { c.classList.toggle("flipped", anyFront); c.setAttribute("aria-pressed", anyFront); });
  document.getElementById("jy-flipall").textContent = anyFront ? "Show fronts" : "Flip all"; };
let jyVisible = false;
const jeye = document.getElementById("jy-eye"), jiris = document.getElementById("jy-iris-g");
function landEye() {
  jeye.classList.remove("landed", "blink"); void jeye.offsetWidth; jeye.classList.add("landed");
  setTimeout(() => { jeye.classList.add("blink"); setTimeout(() => jeye.classList.remove("blink"), 900); }, 1250);
}
new IntersectionObserver(es => es.forEach(en => jyVisible = en.isIntersecting), { threshold: .25 }).observe(document.getElementById("jy"));
new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) landEye(); else if (en.boundingClientRect.top > 0) jeye.classList.remove("landed"); }),
                         { threshold: .6 }).observe(document.querySelector("#jy header"));
addEventListener("mousemove", e => { if (!jyVisible || document.body.classList.contains("calm")) return; const r = jeye.getBoundingClientRect();
  const dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2), d = Math.hypot(dx, dy) || 1, k = Math.min(1, d / 500) * 10;
  jiris.style.transform = `translate(${(dx / d * k).toFixed(1)}px,${(dy / d * k * .6).toFixed(1)}px)`; });
document.addEventListener("keydown", e => {
  if (!jyVisible || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) || document.querySelector(".ov.show")) return;
  if (e.key === "ArrowRight") { e.preventDefault(); setCurrent(cur + 1); }
  else if (e.key === "ArrowLeft") { e.preventDefault(); setCurrent(cur - 1); }
  else if ((e.key === "Enter" || e.key === " ") && document.activeElement && !document.activeElement.classList.contains("card") && document.activeElement.tagName !== "BUTTON") { e.preventDefault(); flip(cards[cur]); }
});
let st; track.addEventListener("scroll", () => { clearTimeout(st); st = setTimeout(() => {
  if (Date.now() - progT < 1100) return;                                   // scrolling started by the arrows, map or keys
  const l = track.scrollLeft, max = track.scrollWidth - track.clientWidth, c = stops[cur];
  const visible = c.offsetLeft >= l - 4 && c.offsetLeft + c.offsetWidth - 36 <= l + track.clientWidth + 4;
  if (l >= max - 4 && visible) return;                                     // end of the road: keep the chosen station
  let best = 0, bd = 1e9; stops.forEach((s, k) => { const d = Math.abs(s.offsetLeft - l); if (d < bd) { bd = d; best = k; } });
  if (best !== cur) setCurrent(best, false); }, 140); });
document.getElementById("jy-up").onclick = () => document.getElementById("main").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
addEventListener("resize", () => setCurrent(cur, false));
document.fonts && document.fonts.ready.then(() => setCurrent(cur, false));
window.jyRelayout = () => setCurrent(cur, false);
setCurrent(0, false);

})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
