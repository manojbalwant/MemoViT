"""
sam2_colab_annotator.py  (multi-object)
Ground-truth BOXES and BINARY MASKS for one or more anomalous objects in Drone-Anomaly videos
(Bike Roundabout test set)

Design
  * One SAM2 tracking session per anomalous object, stored in <OUT>/seg<k>/obj<j>/.
    SAM2's video predictor tracks objects independently anyway, so this costs no accuracy,
    and every object gets its own LOCATE step (objects that enter mid-segment are supported).
  * After an object is verified, a sheet of frames spanning the segment (finished objects drawn
    in colour) asks whether ANOTHER anomalous object still needs boxing. Answer with a frame id
    where it is visible (LOCATE starts there), or press Enter for "no more".
  * Ground truth exists only for anomalous (in-span) frames. Normal frames get no boxes and
    all-black masks. All objects' masks are OR-ed into one binary PNG per frame (0 / 255).
  * Every human input is saved immediately; re-running resumes. An object whose tracking finished
    is never re-tracked (re-tracking would overwrite its verification results).
  * Supported frame-folder layouts (detected automatically from the labels):
      full_video      - every frame of the video
      anomalous_only  - only the frames labelled 1 (in order)
      first_to_last   - first..last anomalous frame, including normal gaps between runs
    All exported ground truth uses ORIGINAL video frame numbers (0-based, aligned with the labels).

---------------------------------------------------------------- Colab setup (once)
!SAM2_BUILD_CUDA=0 pip -q install "git+https://github.com/facebookresearch/sam2.git"
!mkdir -p /content/ckpt && wget -q -P /content/ckpt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
from google.colab import drive; drive.mount('/content/drive')

---------------------------------------------------------------- Usage (one video)
from sam2_colab_annotator import *
SEQ    = "/content/data/.../b_s1_te_02"          # frame folder (any layout above) or video file
LABELS = "/content/data/.../b_s1_te_02.csv"      # frame-level labels of the FULL video
OUT    = "/content/drive/MyDrive/drone_gt/b_s1_te_02"

src, tl, obj_dirs = annotate_video(SEQ, LABELS, OUT)              # track + verify all objects
export_video_gt(obj_dirs, tl, f"{OUT}/gt_boxes.json")
mk = MaskMaker()
for d in obj_dirs:
    make_masks(d, src, mk)
    verify_masks(d, src, mk)
export_mask_gt(obj_dirs, src, tl, f"{OUT}/ground_truth_masks")     # one PNG per ORIGINAL frame
blind_agreement(obj_dirs, src, f"{OUT}/agreement.json", n=40)     # optional: 2nd annotator
res = evaluate_pixel(f"{OUT}/ground_truth_masks", range(tl["n_full"]), my_anomaly_map,
                     labels=load_frame_labels(LABELS), event_weighted=True)
"""
import base64
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import torch

OBJ = 1  # SAM2 object id inside each (single-object) session
FRAMES_ROOT = Path(os.environ.get("SAM2_FRAMES_DIR",
                                  "/content/_sam2_frames" if Path("/content").is_dir() else "/tmp/sam2_frames"))
SAM2_CFG = os.environ.get("SAM2_CFG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
SAM2_CKPT = os.environ.get("SAM2_CKPT", "/content/ckpt/sam2.1_hiera_base_plus.pt")
PALETTE = [(0, 255, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (0, 128, 255), (255, 255, 0)]


class Paused(Exception):
    """Raised when the annotator stops; all progress is saved - re-run the same call to resume."""


# ============================================================================ utils
def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def mask_to_box(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def box_iou(a, b):
    if a is None or b is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def pick_device():
    """bf16 autocast on Ampere+/Ada (A100, L4); plain fp32 on T4 (no native bf16)."""
    if not torch.cuda.is_available():
        return "cpu", contextlib.nullcontext
    if torch.cuda.get_device_capability()[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda", lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    return "cuda", contextlib.nullcontext


def _square_crop(img, cx, cy, half):
    """Fixed-size crop centred on (cx, cy); grey padding at image borders instead of stretching."""
    p = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    return p[cy:cy + 2 * half, cx:cx + 2 * half]


def _tile_sheet(tiles, cols):
    h, w = tiles[0].shape[:2]
    tiles = list(tiles)
    while len(tiles) % cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


# ============================================================================ labels, frames, timeline
def load_frame_labels(path):
    """0/1 per frame from .npy / .mat / .csv / .txt (header lines such as 'value' are skipped)."""
    p = Path(path)
    if p.suffix == ".npy":
        y = np.load(p)
    elif p.suffix == ".mat":
        from scipy.io import loadmat
        y = next(v for k, v in loadmat(p).items() if not k.startswith("__"))
    else:
        vals = []
        for line in open(p):
            tok = line.strip().replace(";", ",").split(",")[-1].strip()
            try:
                vals.append(float(tok))
            except ValueError:
                continue
        y = np.array(vals)
    return (np.asarray(y).ravel() > 0).astype(np.uint8)


def anomaly_segments(labels, min_len=1, merge_gap=0):
    """Contiguous anomalous runs [start, end] (inclusive); gaps <= merge_gap are bridged."""
    segs = []
    for f in np.flatnonzero(labels):
        if segs and f - segs[-1][1] <= merge_gap + 1:
            segs[-1][1] = int(f)
        else:
            segs.append([int(f), int(f)])
    return [(s, e) for s, e in segs if e - s + 1 >= min_len]


class FrameSource:
    """Random access to a sequence stored as a video file or a folder of frames."""
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, path):
        self.path = Path(path)
        if self.path.is_file():
            self.files = None
            cap = cv2.VideoCapture(str(self.path))
            self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            self.files = sorted((f for f in self.path.iterdir() if f.suffix.lower() in self.EXTS), key=natural_key)
            self.n = len(self.files)
        if self.n == 0:
            raise FileNotFoundError(path)

    def name(self, i):
        return self.files[i].name if self.files else f"{self.path.stem}_{i:05d}"

    def read(self, i):
        if self.files:
            return cv2.imread(str(self.files[i]))
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            raise IndexError(i)
        return fr

    def export_range(self, a, b, dst: Path):
        """Write positions a..b as <local>.jpg (SAM2 naming) to fast local disk (cached)."""
        dst.mkdir(parents=True, exist_ok=True)
        if len(list(dst.glob("*.jpg"))) == b - a + 1:
            return
        if self.files:
            for l, i in enumerate(range(a, b + 1)):
                cv2.imwrite(str(dst / f"{l:05d}.jpg"), cv2.imread(str(self.files[i])), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cap = cv2.VideoCapture(str(self.path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for l in range(b - a + 1):
                ok, fr = cap.read()
                if not ok:
                    raise IndexError(a + l)
                cv2.imwrite(str(dst / f"{l:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cap.release()


def build_timeline(src, full_labels, min_len=1, merge_gap=0):
    """Map folder positions -> original frame numbers and define segments (in folder positions)."""
    anom = np.flatnonzero(full_labels)
    if anom.size == 0:
        raise ValueError("No anomalous frames in the labels.")
    if src.n == len(full_labels):
        layout, orig, local = "full_video", np.arange(src.n), full_labels.copy()
    elif src.n == len(anom):
        layout, orig, local = "anomalous_only", anom, np.ones(src.n, np.uint8)
    elif src.n == anom[-1] - anom[0] + 1:
        layout = "first_to_last"
        orig, local = np.arange(anom[0], anom[-1] + 1), full_labels[anom[0]:anom[-1] + 1].copy()
    else:
        raise ValueError(f"Folder has {src.n} frames; labels imply {len(full_labels)} (full), "
                         f"{len(anom)} (anomalous only) or {anom[-1] - anom[0] + 1} (first..last).")
    if layout == "anomalous_only":  # split where ORIGINAL numbering jumps
        cuts = np.flatnonzero(np.diff(orig) > merge_gap + 1) + 1
        segs = [(int(r[0]), int(r[-1])) for r in np.split(np.arange(src.n), cuts) if len(r) >= min_len]
    else:
        segs = anomaly_segments(local, min_len, merge_gap)
    runs = anomaly_segments(full_labels, 1, 0)
    print(f"layout={layout} | video frames={len(full_labels)} | anomalous={len(anom)} | "
          f"label runs={runs} | segments (folder positions)={segs}")
    return dict(layout=layout, orig=[int(x) for x in orig], local_labels=[int(x) for x in local],
                segments=segs, n_full=int(len(full_labels)), min_len=min_len, merge_gap=merge_gap)


# ============================================================================ confidence
class ConfidenceScorer:
    """Weighted geometric mean of SAM2 object-presence, mask certainty, area stability and
    constant-velocity motion agreement. Direction-agnostic (works for backward passes)."""

    def __init__(self, weights=(0.3, 0.3, 0.2, 0.2), hist_len=8):
        w = np.asarray(weights, float)
        self.w, self.hist_len, self.h = w / w.sum(), hist_len, []

    def reset(self, frame=None, box=None):
        self.h = [(frame, box)] if box is not None else []

    def update(self, frame, box):
        if self.h and self.h[-1][0] == frame:
            self.h[-1] = (frame, box)
        else:
            self.h.append((frame, box))
        del self.h[:-self.hist_len]

    def score(self, frame, logits, box, obj_logit):
        if box is None:
            return 0.0, dict(obj=0.0, mask=0.0, area=0.0, motion=0.0)
        p = torch.sigmoid(logits.float())
        fg = p > 0.5
        s_mask = float(((p[fg] - 0.5) / 0.5).mean()) if fg.any() else 0.0
        s_obj = 1.0 if obj_logit is None else 1.0 / (1.0 + math.exp(-obj_logit))
        s_area = s_motion = 1.0
        if len(self.h) >= 2:
            area = (box[2] - box[0]) * (box[3] - box[1])
            ref = float(np.median([(b[2] - b[0]) * (b[3] - b[1]) for _, b in self.h]))
            s_area = math.exp(-abs(math.log((area + 1) / (ref + 1))))
        if self.h:
            f1, b1 = self.h[-1]
            pred = center(b1)
            if len(self.h) >= 2:
                f0, b0 = self.h[-2]
                if f1 != f0:
                    pred = center(b1) + (center(b1) - center(b0)) / (f1 - f0) * (frame - f1)
            diag = math.hypot(b1[2] - b1[0], b1[3] - b1[1]) + 1e-6
            s_motion = math.exp(-np.linalg.norm(center(box) - pred) / diag)
        parts = dict(obj=s_obj, mask=s_mask, area=s_area, motion=s_motion)
        v = np.clip(list(parts.values()), 1e-6, 1.0)
        return float(np.exp(np.sum(self.w * np.log(v)))), parts


def sam2_object_score(state, obj_id, frame_idx):
    try:
        store = state["output_dict_per_obj"][state["obj_id_to_idx"][obj_id]]
        out = store["non_cond_frame_outputs"].get(frame_idx) or store["cond_frame_outputs"].get(frame_idx)
        return float(out["object_score_logits"].flatten()[0])
    except Exception:
        return None


# ============================================================================ UI (Colab; remote_ui.py patches these)
_JS_BOX = r"""
window.sam2DrawBox = async function(b64, W, H, s, prop, title) {
  const root = document.createElement('div'); root.style.cssText = 'font-family:sans-serif;margin:6px 0';
  const hdr = document.createElement('div');
  hdr.textContent = title + '   |   drag = (re)draw, Enter = accept, x = absent';
  const wrap = document.createElement('div'); wrap.style.cssText = 'max-height:80vh;overflow:auto';
  const cv = document.createElement('canvas');
  cv.width = Math.round(W * s); cv.height = Math.round(H * s); cv.style.cursor = 'crosshair'; wrap.appendChild(cv);
  const bar = document.createElement('div');
  const mk = t => { const b = document.createElement('button'); b.textContent = t; b.style.margin = '4px'; bar.appendChild(b); return b; };
  const bOk = mk('Accept'), bAbs = mk('Absent'), bReset = mk('Reset'), bQuit = mk('Save & quit');
  root.append(hdr, wrap, bar); document.body.appendChild(root);
  const ctx = cv.getContext('2d'), img = new Image();
  await new Promise(r => { img.onload = r; img.src = 'data:image/jpeg;base64,' + b64; });
  const init = prop ? prop.map(v => v * s) : null;
  let box = init ? init.slice() : null, drag = null, edited = false;
  const paint = () => { ctx.drawImage(img, 0, 0, cv.width, cv.height);
    if (box) { ctx.lineWidth = 2; ctx.strokeStyle = edited ? '#00ff66' : '#ffd400';
               ctx.strokeRect(box[0], box[1], box[2] - box[0], box[3] - box[1]); } };
  paint();
  const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
  cv.onmousedown = e => { drag = pos(e); };
  cv.onmousemove = e => { if (!drag) return; const p = pos(e);
    box = [Math.min(drag[0], p[0]), Math.min(drag[1], p[1]), Math.max(drag[0], p[0]), Math.max(drag[1], p[1])];
    edited = true; paint(); };
  window.onmouseup = () => { drag = null; };
  bReset.onclick = () => { box = init ? init.slice() : null; edited = false; paint(); };
  const res = await new Promise(resolve => {
    bOk.onclick = () => { if (!box || box[2] - box[0] < 2 || box[3] - box[1] < 2) { hdr.style.color = 'red'; return; }
      resolve({action: edited ? 'redraw' : 'accept', box: box.map(v => v / s)}); };
    bAbs.onclick = () => resolve({action: 'absent', box: null});
    bQuit.onclick = () => resolve({action: 'quit', box: null});
    document.onkeydown = e => { if (e.key === 'Enter') bOk.click(); if (e.key === 'x') bAbs.click(); };
  });
  document.onkeydown = null; root.remove(); return res;
};
"""

_JS_POINTS = r"""
window.sam2Points = async function(b64, W, H, s, title) {
  const root = document.createElement('div'); root.style.cssText = 'font-family:sans-serif;margin:6px 0';
  const hdr = document.createElement('div');
  hdr.textContent = title + '   |   click = add object, shift/right-click = remove, then Re-segment';
  const cv = document.createElement('canvas'); cv.width = Math.round(W * s); cv.height = Math.round(H * s);
  const bar = document.createElement('div');
  const mk = t => { const b = document.createElement('button'); b.textContent = t; b.style.margin = '4px'; bar.appendChild(b); return b; };
  const bSeg = mk('Re-segment'), bUndo = mk('Undo point'), bOk = mk('Accept mask'),
        bBox = mk('Use box fill'), bQuit = mk('Save & quit');
  root.append(hdr, cv, bar); document.body.appendChild(root);
  const ctx = cv.getContext('2d'), img = new Image();
  await new Promise(r => { img.onload = r; img.src = 'data:image/jpeg;base64,' + b64; });
  let pts = [];
  const paint = () => { ctx.drawImage(img, 0, 0, cv.width, cv.height);
    for (const [x, y, l] of pts) { ctx.beginPath(); ctx.arc(x * s, y * s, 5, 0, 7);
      ctx.fillStyle = l ? '#00ff66' : '#ff3355'; ctx.fill(); ctx.strokeStyle = '#000'; ctx.stroke(); } };
  paint();
  cv.oncontextmenu = e => e.preventDefault();
  cv.onmousedown = e => { const r = cv.getBoundingClientRect();
    pts.push([(e.clientX - r.left) / s, (e.clientY - r.top) / s, (e.button === 2 || e.shiftKey) ? 0 : 1]); paint(); };
  bUndo.onclick = () => { pts.pop(); paint(); };
  const res = await new Promise(resolve => {
    bSeg.onclick = () => resolve({action: 'resegment', points: pts});
    bOk.onclick = () => resolve({action: 'accept', points: pts});
    bBox.onclick = () => resolve({action: 'boxfill', points: []});
    bQuit.onclick = () => resolve({action: 'quit', points: []});
  });
  root.remove(); return res;
};
"""


def _jpeg_b64(img, q=90):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


class ColabUI:
    """In-notebook canvas. ask() blocks until the annotator clicks; logs effort."""

    def __init__(self, display_width=1100):
        self.display_width, self.n, self.seconds = display_width, 0, 0.0

    def ask(self, img, proposal, title):
        from google.colab import output
        from IPython.display import display, Javascript, clear_output
        h, w = img.shape[:2]
        t0 = time.time()
        clear_output(wait=True)
        display(Javascript(_JS_BOX))
        res = output.eval_js(f"sam2DrawBox({json.dumps(_jpeg_b64(img))},{w},{h},{self.display_width / w},"
                             f"{json.dumps(proposal)},{json.dumps(title)})")
        self.n += 1
        self.seconds += time.time() - t0
        if res["action"] == "quit":
            raise KeyboardInterrupt
        box = [float(v) for v in res["box"]] if res["box"] else None
        return box, res["action"]


def _ask_points(crop_bgr, title, display_width):
    from google.colab import output
    from IPython.display import display, Javascript, clear_output
    h, w = crop_bgr.shape[:2]
    clear_output(wait=True)
    display(Javascript(_JS_POINTS))
    return output.eval_js(f"sam2Points({json.dumps(_jpeg_b64(crop_bgr, 92))},{w},{h},"
                          f"{display_width / w},{json.dumps(title)})")


def _ask_sheet(img, prompt):
    """Show an image, return typed text ('q' stops). remote_ui.py routes this to the browser."""
    from IPython.display import display, Image, clear_output
    clear_output(wait=True)
    display(Image(data=cv2.imencode(".png", img)[1].tobytes()))
    ans = input(prompt + "  ('q' = save & stop): ").strip()
    if ans.lower() == "q":
        raise KeyboardInterrupt
    return ans


# ============================================================================ csv helpers
FIELDS = ["frame", "file", "in_span", "x1", "y1", "x2", "y2", "conf", "source", "verified",
          "s_obj", "s_mask", "s_area", "s_motion"]


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_rows(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["frame"], r["in_span"], r["verified"] = int(r["frame"]), int(r["in_span"]), int(r["verified"])
        r["box"] = [float(r[k]) for k in ("x1", "y1", "x2", "y2")] if r["x1"] else None
    return rows


def set_box(r, box, source):
    r["box"], r["source"], r["verified"] = box, source, 1
    for k, v in zip(("x1", "y1", "x2", "y2"), box or [""] * 4):
        r[k] = v


def _frame_reader(obj_dir, src):
    meta = json.load(open(Path(obj_dir) / "meta.json"))
    fd = Path(meta["frames_dir"])

    def read(i):
        p = fd / f"{i - meta['g0']:05d}.jpg"
        return cv2.imread(str(p)) if p.exists() else src.read(i)
    return read


# ============================================================================ tracking session (ONE object)
@dataclass
class Config:
    seq: str
    out: str
    start: int                       # first anomalous position of the segment (folder position)
    end: int                         # last anomalous position of the segment (folder position)
    label: str = "anomalous object"  # shown in every prompt
    context: tuple = ()              # dirs of objects already done in this segment (drawn grey)
    locate_from: int = None          # LOCATE starts here (folder position) instead of `start`
    sam2_cfg: str = SAM2_CFG
    sam2_ckpt: str = SAM2_CKPT
    margin: int = 15                 # extra frames tracked outside the span (consistency check only)
    n_init: int = 5                  # manual frames incl. anchor (3-10)
    init_stride: int = 2
    locate_step: int = 5
    tau: float = 0.6                 # affects effort, not final quality (verification checks all)
    patience: int = 2
    audit_every: int = 60
    weights: tuple = (0.3, 0.3, 0.2, 0.2)
    display_width: int = 1100
    seed: int = 0


ACTION2SRC = {"accept": "manual_verified", "redraw": "manual_correction", "absent": "manual_absent"}


class SegmentSession:
    def __init__(self, cfg: Config, source: FrameSource = None):
        from sam2.build_sam import build_sam2_video_predictor
        seed_everything(cfg.seed)
        self.cfg, self.src = cfg, source or FrameSource(cfg.seq)
        self.out = Path(cfg.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.g0 = max(0, cfg.start - cfg.margin)
        self.g1 = min(self.src.n - 1, cfg.end + cfg.margin)
        tag = hashlib.md5(f"{Path(cfg.seq).resolve()}|{self.g0}|{self.g1}".encode()).hexdigest()[:10]
        self.frames_dir = FRAMES_ROOT / tag          # shared by all objects of the segment
        self.src.export_range(self.g0, self.g1, self.frames_dir)
        self.T = self.g1 - self.g0 + 1
        self.device, self.autocast = pick_device()
        self.predictor = build_sam2_video_predictor(cfg.sam2_cfg, cfg.sam2_ckpt, device=self.device)
        self.scorer, self.ui = ConfidenceScorer(cfg.weights), ColabUI(cfg.display_width)
        self.prompts, self.records = {}, {}          # local frame -> box|None / record
        self.suspended, self.n_corrections, self.completed = False, 0, False
        self._ctx = defaultdict(list)                # folder position -> [(name, box)] of finished objects
        for d in cfg.context:
            p = Path(d) / "annotations.csv"
            if p.exists():
                for r in read_rows(p):
                    if r["box"] and r["in_span"]:
                        self._ctx[r["frame"]].append((Path(d).name, r["box"]))

    def g(self, l):
        return l + self.g0

    def img(self, l):
        """Frame for DISPLAY only (finished objects in grey); SAM2 reads the clean frames."""
        im = cv2.imread(str(self.frames_dir / f"{l:05d}.jpg"))
        for name, b in self._ctx.get(self.g(l), []):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(im, (x1, y1), (x2, y2), (160, 160, 160), 1)
            cv2.putText(im, name, (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1)
        return im

    def _save_prompts(self):
        json.dump({str(self.g(l)): b for l, b in sorted(self.prompts.items())},
                  open(self.out / "prompts.json", "w"), indent=1)

    def add_prompt(self, l, box, source):
        self.prompts[l] = box
        self.records[l] = dict(box=box, conf=1.0, source=source, parts={})
        if box is not None:
            self.predictor.add_new_points_or_box(inference_state=self.state, frame_idx=l, obj_id=OBJ,
                                                 box=np.asarray(box, np.float32))
            self.scorer.reset(l, box)
        self.suspended = box is None
        self._save_prompts()

    def initialise(self):
        c = self.cfg
        l = (c.locate_from if c.locate_from is not None else c.start) - self.g0
        first = None
        while l < self.T and self.g(l) <= c.end:
            box, act = self.ui.ask(self.img(l), None, f"LOCATE {c.label}: frame {self.g(l)} "
                                   f"(span {c.start}-{c.end}) - draw it, or Absent if not visible yet")
            if act != "absent":
                first = box
                break
            l += c.locate_step
        if first is None:
            raise RuntimeError(f"{c.label} never marked visible in span {c.start}-{c.end}.")
        self.anchor = l
        self.add_prompt(l, first, "manual_init")
        prev = first
        for k in range(1, c.n_init):
            lk = self.anchor + k * c.init_stride
            if lk >= self.T:
                break
            box, act = self.ui.ask(self.img(lk), prev, f"INIT {c.label} {k + 1}/{c.n_init}: frame {self.g(lk)}")
            self.add_prompt(lk, box, "manual_init" if act != "absent" else "manual_absent")
            prev = box or prev

    def _resume(self):
        saved = json.load(open(self.out / "prompts.json"))
        for gs, b in saved.items():
            self.add_prompt(int(gs) - self.g0, b, "manual_resumed")
        visible = [l for l, b in self.prompts.items() if b is not None]
        if not visible:
            raise RuntimeError("Saved prompts contain no visible box; delete prompts.json to restart.")
        self.anchor = min(visible)
        print(f"[{self.cfg.label}] resumed {len(saved)} human inputs.")

    def _pass(self, reverse):
        c, sign = self.cfg, (-1 if reverse else 1)
        cursor = last_human = self.anchor
        self.scorer.reset()
        self.suspended = False
        while True:
            trigger, low_start, reappear = None, None, 0
            for l, _, masks in self.predictor.propagate_in_video(self.state, start_frame_idx=cursor, reverse=reverse):
                if l in self.prompts:
                    b = self.prompts[l]
                    self.suspended = b is None
                    if b is not None:
                        self.scorer.update(l, b)
                    last_human, low_start = l, None
                    continue
                logits = masks[0, 0]
                box = mask_to_box((logits > 0).cpu().numpy())
                conf, parts = self.scorer.score(l, logits, box, sam2_object_score(self.state, OBJ, l))
                if self.suspended:  # human said "absent": watch for re-entry
                    confident = box is not None and conf >= c.tau
                    reappear = reappear + 1 if confident else 0
                    self.records[l] = dict(box=None, conf=conf, source="absent_propagated", parts=parts)
                    if reappear >= c.patience:
                        trigger = (l - sign * (c.patience - 1), "object may be visible again")
                        break
                    continue
                ok = conf >= c.tau
                self.records[l] = dict(box=box, conf=conf, parts=parts, source="auto" if ok else "auto_lowconf")
                if ok:
                    self.scorer.update(l, box)
                    low_start = None
                else:
                    low_start = l if low_start is None else low_start
                    if abs(l - low_start) + 1 >= c.patience:
                        trigger = (low_start, "low confidence")
                        break
                if abs(l - last_human) >= c.audit_every:
                    trigger = (l, "scheduled audit")
                    break
            if trigger is None:
                return
            l_fix, reason = trigger
            box, act = self.ui.ask(self.img(l_fix), self.records.get(l_fix, {}).get("box"),
                                   f"{reason.upper()} [{c.label}]: frame {self.g(l_fix)} "
                                   f"({'backward' if reverse else 'forward'})")
            self.add_prompt(l_fix, box, ACTION2SRC[act])
            self.n_corrections += act == "redraw"
            cursor = l_fix

    def run(self):
        """Returns True if both passes finished; False if the annotator stopped (progress saved)."""
        try:
            with torch.inference_mode(), self.autocast():
                self.state = self.predictor.init_state(video_path=str(self.frames_dir), offload_video_to_cpu=True)
                if (self.out / "prompts.json").exists():
                    self._resume()
                else:
                    self.initialise()
                self._pass(reverse=False)
                self._pass(reverse=True)
                self.completed = True
        except KeyboardInterrupt:
            print(f"[{self.cfg.label}] stopped - progress saved.")
        finally:
            self.save()
            self.state = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.completed

    def save(self):
        c = self.cfg
        rows = []
        for l in range(self.T):
            r = self.records.get(l)
            if r is None:
                continue
            gi, p, b = self.g(l), r.get("parts") or {}, r["box"]
            rows.append(dict(frame=gi, file=self.src.name(gi), in_span=int(c.start <= gi <= c.end),
                             x1=b[0] if b else "", y1=b[1] if b else "", x2=b[2] if b else "", y2=b[3] if b else "",
                             conf=round(r["conf"], 4), source=r["source"],
                             verified=int(r["source"].startswith("manual")),
                             **{f"s_{k}": round(p[k], 4) for k in ("obj", "mask", "area", "motion") if k in p}))
        write_rows(self.out / "annotations.csv", rows)
        json.dump(dict(seq=c.seq, label=c.label, start=c.start, end=c.end, g0=self.g0, g1=self.g1,
                       frames_dir=str(self.frames_dir)), open(self.out / "meta.json", "w"), indent=1)
        span = [r for r in rows if r["in_span"]]
        stats = dict(config={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(c).items()},
                     tracking_done=self.completed, span_frames=c.end - c.start + 1,
                     human_inputs=len(self.prompts),
                     human_inputs_per_100=100 * len(self.prompts) / max(len(span), 1),
                     corrections=self.n_corrections, ui_seconds=round(self.ui.seconds, 1),
                     frac_auto_in_span=float(np.mean([r["source"] == "auto" for r in span])) if span else 0.0,
                     gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                     torch=torch.__version__)
        json.dump(stats, open(self.out / "stats_tracking.json", "w"), indent=1)


# ============================================================================ multi-object driver
def _stage(obj_dir):
    d = Path(obj_dir)
    if (d / "stats_verify.json").exists():
        return "done"
    st = d / "stats_tracking.json"
    if st.exists() and json.load(open(st)).get("tracking_done"):
        return "verifying"            # never re-track: it would overwrite verification results
    return "tracking"


def ask_more_objects(src, s, e, obj_dirs, n_tiles=16, cols=4, tile_w=360):
    """Sheet of frames across the segment with finished objects drawn; returns a folder position
    where a NOT-yet-boxed anomalous object is visible, or None."""
    boxes = defaultdict(list)
    for i, d in enumerate(obj_dirs):
        for r in read_rows(Path(d) / "annotations.csv"):
            if r["box"] and r["in_span"]:
                boxes[r["frame"]].append((Path(d).name, r["box"], i))
    read = _frame_reader(obj_dirs[0], src)
    tiles = []
    for f in np.unique(np.linspace(s, e, n_tiles).astype(int)):
        im = read(int(f)).copy()
        for name, b, i in boxes.get(int(f), []):
            col = PALETTE[i % len(PALETTE)]
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, 4)
            cv2.putText(im, name, (x1, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 3)
        t = cv2.resize(im, (tile_w, int(im.shape[0] * tile_w / im.shape[1])))
        cv2.putText(t, str(int(f)), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(t)
    while True:
        ans = _ask_sheet(_tile_sheet(tiles, cols),
                         f"Segment {s}-{e}: another anomalous object NOT yet boxed? "
                         f"Type a frame id where it is visible, Enter = no more")
        if not ans:
            return None
        try:
            f = int(ans.split()[0])
        except ValueError:
            continue
        if s <= f <= e:
            return f
        print(f"Frame must be within {s}-{e}.")


def annotate_video(seq, labels_path, out, min_len=1, merge_gap=0, src=None, **cfg_overrides):
    """Track + verify every anomalous object of every segment of one video (resumable).
    Returns (src, timeline, obj_dirs)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = src or FrameSource(seq)
    tl = build_timeline(src, load_frame_labels(labels_path), min_len, merge_gap)
    json.dump(tl, open(out / "timeline.json", "w"))
    margin = cfg_overrides.pop("margin", 0 if tl["layout"] == "anomalous_only" else 15)
    obj_dirs = []
    try:
        for k, (s, e) in enumerate(tl["segments"]):
            seg_dir = out / f"seg{k}"
            seg_objs = sorted((str(p) for p in seg_dir.glob("obj*") if p.is_dir()), key=natural_key)
            j = 0
            while True:
                if j < len(seg_objs):
                    d = seg_objs[j]
                elif j > 0 and (seg_dir / "no_more_objects").exists():
                    break
                else:
                    lf = None
                    if j > 0:
                        lf = ask_more_objects(src, s, e, seg_objs)
                        if lf is None:
                            (seg_dir / "no_more_objects").touch()
                            break
                    d = str(seg_dir / f"obj{j + 1}")
                    Path(d).mkdir(parents=True, exist_ok=True)
                    json.dump({"locate_from": lf}, open(Path(d) / "locate_from.json", "w"))
                    seg_objs.append(d)
                st = _stage(d)
                if st == "tracking":
                    lf_file = Path(d) / "locate_from.json"
                    lf = json.load(open(lf_file)).get("locate_from") if lf_file.exists() else None
                    cfg = Config(seq=str(seq), out=d, start=s, end=e, label=f"seg{k} object #{j + 1}",
                                 context=tuple(seg_objs[:j]), locate_from=lf, margin=margin, **cfg_overrides)
                    if not SegmentSession(cfg, src).run():
                        raise Paused(f"Stopped in {d}; re-run annotate_video() to resume.")
                if st != "done":
                    verify_segment(d, src)
                j += 1
            obj_dirs += seg_objs
    except KeyboardInterrupt:
        raise Paused("Stopped by annotator; all progress saved - re-run annotate_video() to resume.")
    print(f"{len(obj_dirs)} object(s) annotated: {obj_dirs}")
    return src, tl, obj_dirs


def collect_obj_dirs(out):
    """Rebuild obj_dirs from disk (e.g. in a fresh runtime)."""
    return sorted((str(p.parent) for p in Path(out).glob("seg*/obj*/annotations.csv")), key=natural_key)


# ============================================================================ box verification
def verify_segment(obj_dir, src, per_page=300, cols=20, tile=150, display_width=1100):
    """Contact sheet of every unverified in-span frame of ONE object; type wrong frame ids."""
    obj_dir = Path(obj_dir)
    rows, read, ui = read_rows(obj_dir / "annotations.csv"), _frame_reader(obj_dir, src), ColabUI(display_width)
    label = json.load(open(obj_dir / "meta.json")).get("label", obj_dir.name)
    todo = [r for r in rows if r["in_span"] and not r["verified"]]
    t0 = time.time()
    for p in range(0, len(todo), per_page):
        page, tiles = todo[p:p + per_page], []
        for r in page:
            im = read(r["frame"])
            if r["box"]:
                x1, y1, x2, y2 = map(int, r["box"])
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255), 2)
                #half = int(max(x2 - x1, y2 - y1, 48) * 1.5)
                #im = _square_crop(im, (x1 + x2) // 2, (y1 + y2) // 2, half)
                text = f"{r['frame']} {r['conf']}"
            else:
                text = f"{r['frame']} NO BOX"
            t = cv2.resize(im, (tile, tile))
            cv2.putText(t, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            tiles.append(t)
        ans = _ask_sheet(_tile_sheet(tiles, cols),
                         f"{label} page {p // per_page + 1}/{math.ceil(len(todo) / per_page)}: "
                         f"wrong frame ids (space-separated), Enter = all OK")
        bad = {int(x) for x in ans.split() if x.lstrip("-").isdigit()}
        for r in page:
            if r["frame"] in bad:
                box, act = ui.ask(read(r["frame"]), r["box"], f"QA FIX {label}: frame {r['frame']}")
                set_box(r, box, "manual_qa_fix" if act != "absent" else "manual_absent")
            else:
                r["verified"] = 1
                r["source"] = r["source"] + "+qa_ok"
        write_rows(obj_dir / "annotations.csv", rows)
    json.dump(dict(frames_checked=len(todo), seconds=round(time.time() - t0, 1)),
              open(obj_dir / "stats_verify.json", "w"), indent=1)


def blind_agreement(obj_dirs, src, out_json, n=40, seed=0, display_width=1100):
    """Annotator B redraws a random sample WITHOUT seeing the existing box; reports IoU."""
    rng, ui, samples = random.Random(seed), ColabUI(display_width), []
    for d in obj_dirs:
        read = _frame_reader(d, src)
        label = json.load(open(Path(d) / "meta.json")).get("label", Path(d).name)
        samples += [(r, read, label) for r in read_rows(Path(d) / "annotations.csv") if r["in_span"] and r["verified"]]
    ious = []
    for r, read, label in rng.sample(samples, min(n, len(samples))):
        box, _ = ui.ask(read(r["frame"]), None, f"BLIND re-annotation of {label}: frame {r['frame']}")
        ious.append(1.0 if (box is None and r["box"] is None) else box_iou(box, r["box"]))
    ious = np.array(ious)
    res = dict(n=len(ious), mean_iou=float(ious.mean()), p_iou_0_5=float((ious >= .5).mean()),
               p_iou_0_75=float((ious >= .75).mean()))
    json.dump(res, open(out_json, "w"), indent=1)
    print(res)
    return res


def export_video_gt(obj_dirs, tl, out_json, dup_iou=0.7):
    """Boxes for anomalous frames only, all objects, ORIGINAL frame numbering."""
    orig, local, n_full = tl["orig"], tl["local_labels"], tl["n_full"]
    boxes, ids, issues = defaultdict(list), defaultdict(list), defaultdict(list)
    for d in obj_dirs:
        tid = "/".join(Path(d).parts[-2:])                       # e.g. "seg0/obj2"
        for r in read_rows(Path(d) / "annotations.csv"):
            g = orig[r["frame"]]
            if r["in_span"] and r["box"]:
                if any(box_iou(b, r["box"]) > dup_iou for b in boxes[g]):
                    issues["possible_duplicate_object"].append(g)
                boxes[g].append(r["box"])
                ids[g].append(tid)
                if not r["verified"]:
                    issues["unverified_box"].append(g)
            elif r["box"] and not r["in_span"]:
                issues["object_visible_outside_span"].append(g)
    for f, y in enumerate(local):
        if y and not boxes[orig[f]]:
            issues["anomalous_frame_without_box"].append(orig[f])
    out = dict(format="xyxy pixels; boxes[f][i] belongs to track_ids[f][i]",
               frame_index="original video frame numbers (0-based, aligned with the labels)", n_frames=n_full,
               boxes={str(g): boxes.get(g, []) for g in range(n_full)},
               track_ids={str(g): ids.get(g, []) for g in range(n_full)},
               issues={k: sorted(set(v)) for k, v in issues.items()})
    json.dump(out, open(out_json, "w"))
    print({k: len(v) for k, v in out["issues"].items()} or "no issues")


# ============================================================================ masks
class MaskMaker:
    """SAM2 image predictor: verified box (+ optional clicks) -> mask."""

    def __init__(self, cfg=SAM2_CFG, ckpt=SAM2_CKPT):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.device, self.autocast = pick_device()
        self.pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=self.device))
        self._key = None

    def set(self, key, bgr):
        if key != self._key:
            with torch.inference_mode(), self.autocast():
                self.pred.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            self._key = key

    def predict(self, box, points=None, pad=0.1):
        kw = dict(box=np.asarray(box, np.float32), multimask_output=not points)
        if points:
            kw["point_coords"] = np.array([p[:2] for p in points], np.float32)
            kw["point_labels"] = np.array([p[2] for p in points], np.int32)
        with torch.inference_mode(), self.autocast():
            masks, scores, _ = self.pred.predict(**kw)
        best, best_q = None, -1.0
        for m, s in zip(masks, scores):
            m = clip_to_box(m > 0, box, pad)
            q = float(s) * box_iou(mask_to_box(m), box) if m.any() else -1.0
            if q > best_q:
                best, best_q = m, q
        return best, best_q


def clip_to_box(mask, box, pad):
    h, w = mask.shape
    bw, bh = box[2] - box[0], box[3] - box[1]
    x1, y1 = max(int(box[0] - pad * bw), 0), max(int(box[1] - pad * bh), 0)
    x2, y2 = min(int(box[2] + pad * bw) + 1, w), min(int(box[3] + pad * bh) + 1, h)
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def box_mask(shape, box):
    m = np.zeros(shape[:2], bool)
    x1, y1, x2, y2 = map(int, map(round, box))
    m[y1:y2, x1:x2] = True
    return m


def overlay(bgr, mask, box=None):
    vis = bgr.copy()
    if mask is not None and mask.any():
        vis[mask] = (0.55 * vis[mask] + 0.45 * np.array([0, 0, 255])).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (0, 0, 255), 1)
    if box is not None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    return vis


def _mask_dir(obj_dir):
    d = Path(obj_dir) / "masks"
    d.mkdir(exist_ok=True)
    return d


def _load_meta(obj_dir):
    p = Path(obj_dir) / "masks.json"
    return {int(k): v for k, v in json.load(open(p)).items()} if p.exists() else {}


def _save_meta(obj_dir, meta):
    json.dump({str(k): v for k, v in sorted(meta.items())}, open(Path(obj_dir) / "masks.json", "w"), indent=1)


def _write_mask(obj_dir, f, mask):
    cv2.imwrite(str(_mask_dir(obj_dir) / f"{f:06d}.png"), mask.astype(np.uint8) * 255)


def _read_mask(obj_dir, f):
    return cv2.imread(str(_mask_dir(obj_dir) / f"{f:06d}.png"), cv2.IMREAD_GRAYSCALE) > 127


def make_masks(obj_dir, src, mk: MaskMaker):
    """One mask per verified in-span box of ONE object (resumable)."""
    rows_all = read_rows(Path(obj_dir) / "annotations.csv")
    if any(r["in_span"] and not r["verified"] for r in rows_all):
        print(f"WARNING {obj_dir}: unverified boxes remain - run verify_segment() first.")
    read, meta = _frame_reader(obj_dir, src), _load_meta(obj_dir)
    for r in (r for r in rows_all if r["in_span"] and r["verified"] and r["box"]):
        f = r["frame"]
        if f in meta:
            continue
        img = read(f)
        mk.set((str(obj_dir), f), img)
        m, q = mk.predict(r["box"])
        source = "sam2_box"
        if m is None or not m.any():
            m, q, source = box_mask(img.shape, r["box"]), 0.0, "box_fallback"
        _write_mask(obj_dir, f, m)
        meta[f] = dict(box=r["box"], quality=round(q, 4), source=source, verified=0, area=int(m.sum()))
    _save_meta(obj_dir, meta)
    print(f"{obj_dir}: {len(meta)} masks, {sum(v['source'] == 'box_fallback' for v in meta.values())} box fallbacks")


def refine_mask(mk, img, key, box, mask, display_width=900):
    """Interactive click refinement on an enlarged crop around the object."""
    H, W = img.shape[:2]
    pad = int(max(box[2] - box[0], box[3] - box[1], 64))
    cx0, cy0 = int(max(0, box[0] - pad)), int(max(0, box[1] - pad))
    cx1, cy1 = int(min(W, box[2] + pad)), int(min(H, box[3] + pad))
    mk.set(key, img)
    pts = []
    while True:
        res = _ask_points(overlay(img, mask, box)[cy0:cy1, cx0:cx1],
                          f"MASK frame {key[1]} ({len(pts)} points so far)", display_width)
        pts += [(x + cx0, y + cy0, int(l)) for x, y, l in res["points"]]
        if res["action"] == "quit":
            raise KeyboardInterrupt
        if res["action"] == "boxfill":
            return box_mask(img.shape, box), "box_fallback"
        if res["action"] == "accept" and not res["points"]:
            return mask, "sam2_refined" if pts else "sam2_box"
        mask, _ = mk.predict(box, pts)


def verify_masks(obj_dir, src, mk, per_page=300, cols=20, tile=150, display_width=1100):
    """Overlay contact sheet for ONE object; '!' marks masks whose area jumps vs neighbours."""
    read, meta = _frame_reader(obj_dir, src), _load_meta(obj_dir)
    frames = sorted(meta)
    areas = {f: max(meta[f]["area"], 1) for f in frames}
    suspicious = set()
    for i, f in enumerate(frames):
        nb = [areas[g] for g in frames[max(0, i - 2):i + 3] if g != f]
        if nb and abs(np.log(areas[f] / np.median(nb))) > np.log(1.5):
            suspicious.add(f)
    todo = [f for f in frames if not meta[f]["verified"]]
    t0 = time.time()
    try:
        for p in range(0, len(todo), per_page):
            page, tiles = todo[p:p + per_page], []
            for f in page:
                box = meta[f]["box"]
                vis = overlay(read(f), _read_mask(obj_dir, f), box)
                #half = int(max(box[2] - box[0], box[3] - box[1], 48) * 1.2)
                #t = cv2.resize(_square_crop(vis, int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2), half),
                               #(tile, tile), interpolation=cv2.INTER_NEAREST)
                t = cv2.resize(vis, (tile, tile), interpolation=cv2.INTER_NEAREST)

                flag = "!" if f in suspicious else ""
                cv2.putText(t, f"{f} q{meta[f]['quality']:.2f}{flag}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255) if flag else (0, 255, 0), 2)
                tiles.append(t)
            ans = _ask_sheet(_tile_sheet(tiles, cols),
                             f"{Path(obj_dir).name} masks page {p // per_page + 1}: wrong frame ids, Enter = all OK")
            bad = {int(x) for x in ans.split() if x.lstrip("-").isdigit()}
            for f in page:
                if f in bad:
                    img = read(f)
                    m, source = refine_mask(mk, img, (str(obj_dir), f), meta[f]["box"],
                                            _read_mask(obj_dir, f), display_width)
                    _write_mask(obj_dir, f, m)
                    meta[f].update(source=source, area=int(m.sum()), quality=-1.0)
                meta[f]["verified"] = 1
            _save_meta(obj_dir, meta)
    except KeyboardInterrupt:
        print("Stopped - progress saved; re-run to continue.")
        raise Paused(f"Mask verification paused in {obj_dir}.")
    finally:
        _save_meta(obj_dir, meta)
        json.dump(dict(masks=len(meta), refined=sum(v["source"] == "sam2_refined" for v in meta.values()),
                       box_fallback=sum(v["source"] == "box_fallback" for v in meta.values()),
                       seconds=round(time.time() - t0, 1)),
                  open(Path(obj_dir) / "stats_masks.json", "w"), indent=1)


def export_mask_gt(obj_dirs, src, tl, out_dir, mvtec_style=True):
    """One binary mask per ORIGINAL video frame (OR over all objects; normal frames all black).
    mvtec_style=True writes only non-empty masks as <frame>_mask.png (missing file = normal)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    H, W = src.read(0).shape[:2]
    orig, local = tl["orig"], tl["local_labels"]
    pos = {g: f for f, g in enumerate(orig)}
    have, issues = defaultdict(list), defaultdict(list)
    for d in obj_dirs:
        for f, v in _load_meta(d).items():
            if not v["verified"]:
                issues["unverified_mask"].append(orig[f])
            have[f].append((d, v["source"]))
    zeros = np.zeros((H, W), np.uint8)
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "file", "frame_label", "mask_pixels", "n_objects", "mask_source"])
        for g in range(tl["n_full"]):
            f = pos.get(g)
            y = int(local[f]) if f is not None else 0      # frames outside the folder are normal
            m, srcs, n_obj = zeros.copy(), "", 0
            if f is not None and f in have:
                parts = [_read_mask(d, f) for d, _ in have[f]]
                for p in parts:
                    m[p] = 255
                nonempty = [p for p in parts if p.any()]
                if len(nonempty) > 1:
                    overlap = int((np.sum(nonempty, axis=0) > 1).sum())
                    if overlap > 0.5 * min(int(p.sum()) for p in nonempty):
                        issues["possible_duplicate_object"].append(g)
                srcs, n_obj = "+".join(sorted({s for _, s in have[f]})), len(parts)
            if m.shape != (H, W):
                raise ValueError(f"frame {g}: mask {m.shape} != frame {(H, W)}")
            px = int((m > 0).sum())
            if y and px == 0:
                issues["anomalous_frame_empty_mask"].append(g)
            if not y and px > 0:
                issues["normal_frame_nonempty_mask"].append(g)
            if mvtec_style:
                if px:
                    cv2.imwrite(str(out / f"{g:06d}_mask.png"), m)
            else:
                cv2.imwrite(str(out / f"{g:06d}.png"), m)
            w.writerow([g, src.name(f) if f is not None else "", y, px, n_obj, srcs])
    json.dump({k: sorted(set(v)) for k, v in issues.items()}, open(out / "issues.json", "w"), indent=1)
    print({k: len(set(v)) for k, v in issues.items()} or "no issues")


def load_gt_mask(gt_dir, f, shape):
    gt_dir = Path(gt_dir)
    for name in (f"{f:06d}.png", f"{f:06d}_mask.png"):
        p = gt_dir / name
        if p.exists():
            return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 127
    return np.zeros(shape, bool)


# ============================================================================ pixel metrics
def evaluate_pixel(gt_dir, frames, score_fn, labels=None, value_range=None, n_bins=4096,
                   fpr_limit=0.3, event_weighted=False):
    """Streaming pixel AUROC, pixel AP and AUPRO@fpr_limit (MVTec AD definition).
    score_fn(f) -> HxW float anomaly map at GT resolution, raw scores (no per-frame normalisation).
    value_range=(lo, hi) of scores, or None for an extra pass that finds it.
    event_weighted: also AUPRO with each anomalous event (label run) weighted equally."""
    frames = list(frames)
    if value_range is None:
        lo, hi = np.inf, -np.inf
        for f in frames:
            s = score_fn(f)
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
    else:
        lo, hi = value_range
    edges = np.linspace(lo, hi, n_bins + 1)
    pos, neg, pro_sum = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    n_regions = 0
    ev_sum, ev_n, seg_of = defaultdict(lambda: np.zeros(n_bins)), defaultdict(int), {}
    if event_weighted and labels is not None:
        for k, (s, e) in enumerate(anomaly_segments(labels)):
            for f in range(s, e + 1):
                seg_of[f] = k
    for f in frames:
        s = score_fn(f)
        gt = load_gt_mask(gt_dir, f, s.shape)
        if gt.shape != s.shape:
            raise ValueError(f"frame {f}: score map {s.shape} vs GT {gt.shape}")
        b = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, n_bins - 1)
        pos += np.bincount(b[gt], minlength=n_bins)
        neg += np.bincount(b[~gt], minlength=n_bins)
        n_cc, cc = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        for k in range(1, n_cc):
            region = cc == k
            h = np.bincount(b[region], minlength=n_bins) / region.sum()
            pro_sum += h
            n_regions += 1
            if f in seg_of:
                ev_sum[seg_of[f]] += h
                ev_n[seg_of[f]] += 1

    def rev_cum(x):
        return np.concatenate([[0.0], np.cumsum(x[::-1])])

    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        raise ValueError("Need both anomalous and normal pixels.")
    trapz = getattr(np, "trapezoid", None) or np.trapz
    tp, fp = rev_cum(pos), rev_cum(neg)
    tpr, fpr = tp / P, fp / N
    prec = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)

    def aupro(pro_curve):
        m = fpr <= fpr_limit
        x, y = fpr[m], pro_curve[m]
        i = int(m.sum())
        if i < len(fpr) and x[-1] < fpr_limit:
            t = (fpr_limit - fpr[i - 1]) / (fpr[i] - fpr[i - 1])
            x = np.append(x, fpr_limit)
            y = np.append(y, pro_curve[i - 1] + t * (pro_curve[i] - pro_curve[i - 1]))
        return float(trapz(y, x) / fpr_limit)

    res = dict(pixel_auroc=float(trapz(tpr, fpr)), pixel_ap=float(np.sum(np.diff(tpr) * prec[1:])),
               aupro=aupro(rev_cum(pro_sum) / max(n_regions, 1)), n_frames=len(frames), n_regions=n_regions,
               anomalous_pixel_fraction=float(P / (P + N)), n_bins=n_bins, fpr_limit=fpr_limit)
    if event_weighted and ev_n:
        res["aupro_event_weighted"] = aupro(np.mean([rev_cum(ev_sum[k]) / ev_n[k] for k in ev_n], axis=0))
    print(res)
    return res
