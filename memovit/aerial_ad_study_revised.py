"""
================================================================================
 aerial_ad_study.py — one-class aerial anomaly detection with frozen DINOv2
 CODE_VERSION 3.2-twodev
================================================================================

 Entry points, in the order used for the article
   cache        feature extraction only (no scoring; safe before selection)
   select       hyperparameter selection on the DEVELOPMENT datasets only
                (ShanghaiTech non-human subset + Drone-Anomaly Highway via
                --dev_extra)  -> selected_config.json (--config_out)
   bench        main comparison with the frozen configuration (--config)
                [--seeds ...] [--ablation_grid] [--sweeps] [--cache]
   sensitivity  one-at-a-time sensitivity around the frozen configuration
   failures     error events at the 5 % FPR operating point + figures (E7)
   overlays     heat-map overlay figures (best / worst frames)
   ablate       ablation axes A1-A7 of the Proposed method

 Setting
   Full-shot ONE-CLASS: the memory bank is built from normal TRAINING frames
   of the same dataset. This is NOT zero-shot.

 Hyperparameter protocol (development-set selection)
   1. `select` scores pre-declared grids on the test sets of the development
      datasets and picks, by MEAN frame AUROC over them (equal weight), the
      first configuration in the declared order within SELECT_TOL of the
      best, for every method.
   2. selected_config.json is committed BEFORE any held-out dataset is scored.
   3. Every held-out run loads it with --config; nothing is re-tuned.
   4. `sensitivity` shows how much results depend on the choice.
   ShanghaiTech subset: the six non-human test videos excluded from
   HR-ShanghaiTech (Morais et al., CVPR 2019, App. A; verify the IDs),
   training cameras 01/06/12, every SHT_TRAIN_STRIDE-th frame.
   Aerial datasets: every train_stride-th TRAINING frame per video
   (BenchConfig.train_stride / --train_stride); all test frames are scored.
   These choices are fixed a priori, not tuned.

 Pixel metrics (pixel AUROC, pixel AP, AUPRO@0.3) are computed for any
 dataset with {data_path}/test/test_pixel_mask/ (<video>.npy of shape
 (T,H,W) or <video>/*.png); otherwise only frame metrics.

 Pipeline split
   FeatureSpec  everything that changes extracted features; cached once per
                (dataset, split, FeatureSpec).
   MethodSpec   everything that changes how cached features are scored.
 Accuracy of the Proposed method is computed from cached (or identically
 batched streamed) features; latency on the live path, batch 1.
================================================================================
"""

# ── standard library ─────────────────────────────────────────────────────────
import os, sys, glob, json, math, time, random, argparse, hashlib, warnings, gc
import csv, platform, subprocess, datetime
from contextlib import contextmanager
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Optional, Tuple

# ── scientific stack ─────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader
import timm
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

warnings.filterwarnings('ignore', category=UserWarning)

CODE_VERSION    = '3.2-twodev-b_min'
FEATURE_VERSION = 'f1'          # bump ONLY when extraction code changes
IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)
SEEDS           = (42, 142, 242, 342, 442)
_DT = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
_CUDA_OOM = getattr(torch.cuda, 'OutOfMemoryError', RuntimeError)
_trapz = getattr(np, 'trapezoid', None) or np.trapz


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · SPECIFICATIONS AND CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

def _hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class FeatureSpec:
    backbone: str = 'vit_base_patch14_dinov2'
    image_size: int = 518
    precision: str = 'fp16'                      # backbone autocast: fp16 | fp32
    mean: Tuple[float, ...] = IMAGENET_MEAN      # frozen ImageNet-pretrained backbone
    std: Tuple[float, ...] = IMAGENET_STD
    batch_size: int = 16                         # batched GEMMs round differently
    patch_dtype: str = 'fp16'                    # storage of L2-normalised patches

    def key(self) -> str:
        return _hash({**asdict(self), 'v': FEATURE_VERSION})

    @property
    def grid(self) -> int:
        return self.image_size // 14


@dataclass(frozen=True)
class MethodSpec:
    """A-priori defaults. The frozen configuration comes from `select` (--config)."""
    selected_layers: Tuple[int, ...] = (11,)
    patch_layer: int = 11
    fps_ratio: Optional[float] = 0.05           # 5-10 None = keep every training frame
    fps_min_frames: int = 50      # declared bank floor (absolute frames); applies only when fps_ratio is set
    fps_seed: int = 42
    knn_k: int = 5                              # <= 0 : all retained frames
    agg: str = 'topk_mean'                       # topk_mean | max | mean | robust_mad
    topk_ratio: float = 0.01
    mad_sigma: float = 2.0
    # ── ablation-only knobs ────────────────────────────────────────────────
    key_source: str = 'concat_cls'               # concat_cls | last_cls | layer5_cls | gap | random
    key_metric: str = 'cosine'                   # cosine | l2 | l2_unit (== cosine ranking)
    key_proj_dim: Optional[int] = None           # uncentred SVD projection of retrieval keys
    position_aligned: bool = False               # patch p matched only to patch p

    def key(self) -> str:
        return _hash({**asdict(self), 'v': CODE_VERSION})


@dataclass(frozen=True)
class Agg:
    """Aggregation rule; duck-type compatible with MethodSpec for aggregate()."""
    agg: str = 'max'
    topk_ratio: float = 0.03
    mad_sigma: float = 2.0


def _agg_name(a) -> str:
    if a.agg == 'topk_mean':
        return f'top{a.topk_ratio:g}'
    if a.agg == 'robust_mad':
        return f'mad{a.mad_sigma:g}'
    return a.agg


def _same_rule(a, b) -> bool:
    if a.agg != b.agg:
        return False
    if a.agg == 'topk_mean':
        return abs(a.topk_ratio - b.topk_ratio) < 1e-12
    if a.agg == 'robust_mad':
        return abs(a.mad_sigma - b.mad_sigma) < 1e-12
    return True


@dataclass
class BenchConfig:
    data_path: str = 'dataset/shanghaitech/' #'dataset/Drone-Anomaly/Highway/' #'dataset/UIT-ADrone/'
    save_path: str = 'experiments/shanghaitech/' #'experiments/rough/Highway/' #'experiments/UIT-ADrone/'
    cache_dir: str = 'feature_cache/'
    no_cache: bool = True                        # bench: Proposed streams features (--cache to cache)
    num_workers: int = 8
    methods: Tuple[str, ...] = ('proposed',)
    train_stride: int = 10#                        # keep every n-th TRAINING frame per video (non-SHT)
    # pixel-level localisation (any dataset with test/<pixel_mask_dir>/)
    pixel_eval: bool = True
    pixel_mask_dir: str = 'test_pixel_mask'      # <video>.npy (T,H,W) or <video>/*.png
    pixel_eval_hw: Optional[Tuple[int, int]] = None   # None: native mask size / pixel_eval_div
    pixel_eval_div: int = 4                      # ShanghaiTech 480x856 -> 120x214
    pixel_max_fpr: float = 0.3
    pixel_smooth_sigma: float = 0.0              # Gaussian σ (px at eval size), all methods
    pixel_skip_inconsistent: bool = True         # drop frames whose mask contradicts the label
    pixel_eval_sweeps: bool = False
    # protocol and bookkeeping
    frozen_config: Optional[str] = None
    allow_label_truncation: bool = False
    save_runs: bool = True
    # shared exemplar banks + exact 1-NN search (all methods)
    bank_dtype: str = 'fp16'
    nn_compute_dtype: str = 'fp32'
    gpu_bank_budget_mb: float = 4096
    nn_chunk_vectors: int = 65536
    # visualisation (accuracy pass only; never timed)
    viz_dir: Optional[str] = None
    viz_every_n: int = 10
    viz_anomalous_only: bool = True
    # latency harness
    measure_latency: bool = True
    latency_frames: int = 200
    latency_warmup: int = 30
    profile_stages: bool = True
    # PaDiM
    padim_layers: Tuple[int, ...] = (11,)
    padim_pca_dim: int = 128
    padim_eps: float = 1e-4
    padim_pca_fit_rows: int = 200_000
    # PatchCore
    patchcore_layers: Tuple[int, ...] = (11,)
    patchcore_ratio: float = 0.05
    patchcore_kernel: int = 3
    patchcore_agg: str = 'max'
    patchcore_proj_dim: Optional[int] = 128#None
    patchcore_storage: str = 'auto'
    # SimpleNet (re-implementation; verify against the official code)
    simplenet_layers: Tuple[int, ...] = (11,)
    simplenet_noise_std: float = 0.015
    simplenet_lr: float = 2e-4
    simplenet_adaptor_lr: float = 1e-4
    simplenet_epochs: int = 30
    simplenet_hidden: int = 1024
    simplenet_th: float = 0.5
    # AnomalyDINO
    ad_layers: Tuple[int, ...] = (11,)
    ad_tail_q: float = 0.01
    ad_rotations: Tuple[int, ...] = ()
    ad_max_bank: Optional[int] = None            # None = full bank (published AnomalyDINO)
    ad_smooth_sigma: float = 4.0                 # visualisation only; pixel metrics use raw maps
    # SPADE
    spade_mode: str = 'patch'
    spade_k_image: int = 50
    spade_k_retrieve: int = 50
    spade_global: str = 'cls'
    spade_metric: str = 'l2'
    spade_patch_agg: str = 'max'
    # optional CNN backbone for PaDiM / PatchCore / SimpleNet
    cnn_backbone: bool = False
    cnn_name: str = 'wide_resnet50_2'
    cnn_out: Tuple[int, ...] = (1, 2, 3)
    cnn_grid: int = 37
    # studies
    ablation_grid: bool = False
    sweeps: bool = False
    sweep_N: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.0)   # fractions of the training frames
                                                                  # (values > 1: absolute counts)
    sweep_K: Tuple[int, ...] = (1, 3, 5, 10, 20)
    sweep_res: Tuple[int, ...] = (336, 448, 518)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _free(device=None):
    gc.collect()
    if device is not None and getattr(device, 'type', '') == 'cuda':
        torch.cuda.empty_cache()


def _safe(s: str) -> str:
    return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in s).strip('_')


def _f(x, nd: int = 4) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 'n/a'
    return f'{x:.{nd}f}' if np.isfinite(x) else 'n/a'


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _dataset_tag(cfg) -> str:
    return os.path.basename(cfg.data_path.rstrip('/\\'))


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _torch_load(path: str):
    """torch.load across versions; memory-mapped where supported."""
    for kw in ({'map_location': 'cpu', 'weights_only': False, 'mmap': True},
               {'map_location': 'cpu', 'weights_only': False},
               {'map_location': 'cpu'}):
        try:
            return torch.load(path, **kw)
        except TypeError:
            continue
        except RuntimeError as e:
            if 'mmap' in kw and 'mmap' in str(e).lower():
                continue
            raise
    raise RuntimeError(f'cannot load {path}')


def _atomic_save(obj, path: str):
    """Write to a temp file then rename, so a live mmap is never truncated."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · LATENCY PROFILER
# ══════════════════════════════════════════════════════════════════════════════

class StageProfiler:
    """
    Synchronised per-stage timing, active ONLY in pass 2 of the latency
    harness. Elsewhere every `with PROF.stage(...)` is a no-op.
    Stages: backbone | search | patch_score; 'other' = remainder.
    """
    STAGES = ('backbone', 'search', 'patch_score')

    def __init__(self, device=None, enabled: bool = False):
        self.device, self.enabled = device, enabled
        self.cuda = device is not None and device.type == 'cuda'
        self.frames: List[Dict[str, float]] = []
        self._cur, self._t0 = None, 0.0

    def _sync(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name: str):
        if not self.enabled or self._cur is None:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self._cur[name] = self._cur.get(name, 0.0) + (time.perf_counter() - t0) * 1e3

    def begin(self):
        if self.enabled:
            self._sync()
            self._cur, self._t0 = {}, time.perf_counter()

    def end(self):
        if self.enabled and self._cur is not None:
            self._sync()
            self._cur['frame_ms'] = (time.perf_counter() - self._t0) * 1e3
            self.frames.append(self._cur)
            self._cur = None

    def summary(self) -> Dict[str, float]:
        if not self.frames:
            return {}
        tot = np.array([f['frame_ms'] for f in self.frames])
        acc, out = np.zeros_like(tot), {}
        for s in self.STAGES:
            v = np.array([f.get(s, 0.0) for f in self.frames])
            acc += v
            out[f'{s}_ms'] = float(np.median(v))
        out['other_ms'] = float(np.median(np.maximum(tot - acc, 0.0)))
        out['profiled_frame_ms'] = float(np.median(tot))
        return out


PROF = StageProfiler()


def set_profiler(device, enabled: bool) -> StageProfiler:
    global PROF
    PROF = StageProfiler(device, enabled)
    return PROF


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · BACKBONE
# ══════════════════════════════════════════════════════════════════════════════

class DINOv2AllLayers(nn.Module):
    """
    Frozen DINOv2 returning (cls_list, patch_list) for EVERY block; the final
    LayerNorm is applied to the last block. Positional embeddings are
    bicubically interpolated for inputs other than the native 518 px.
    """
    def __init__(self, name: str = 'vit_base_patch14_dinov2', pretrained: bool = True):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=pretrained, num_classes=0)
        if getattr(self.vit, 'reg_token', None) is not None:
            raise NotImplementedError('register-token variants are not supported')
        self.embed_dim = int(self.vit.embed_dim)
        self.n_blocks = len(self.vit.blocks)
        L = self.vit.pos_embed.shape[1]
        for extra in (1, 0):
            g = math.isqrt(L - extra)
            if g * g == L - extra:
                self._g0, self._pe_extra = g, extra
                break
        else:
            raise RuntimeError('unrecognised pos_embed layout')
        self._pos = {}
        self.requires_grad_(False)
        self.eval()

    def _pos_embed(self, h: int, w: int):
        k = (h, w, str(self.vit.pos_embed.device))
        if k not in self._pos:
            pe = self.vit.pos_embed
            pre, grid = pe[:, :self._pe_extra], pe[:, self._pe_extra:]
            if (h, w) != (self._g0, self._g0):
                D = grid.shape[-1]
                grid = grid.reshape(1, self._g0, self._g0, D).permute(0, 3, 1, 2)
                grid = F.interpolate(grid.float(), size=(h, w), mode='bicubic',
                                     align_corners=False)
                grid = grid.permute(0, 2, 3, 1).reshape(1, h * w, D).to(pe.dtype)
            self._pos[k] = (pre, grid)
        return self._pos[k]

    def forward(self, img: torch.Tensor):
        x = self.vit.patch_embed.proj(img)                          # (B, D, h, w)
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = getattr(self.vit.patch_embed, 'norm', nn.Identity())(x)
        pre, grid = self._pos_embed(h, w)
        cls = self.vit.cls_token.expand(B, -1, -1)
        if pre.shape[1]:
            cls = cls + pre
        t = torch.cat([cls, x + grid], dim=1)
        t = getattr(self.vit, 'norm_pre', nn.Identity())(t)
        cls_out, pat_out = [], []
        for blk in self.vit.blocks:
            t = blk(t)
            cls_out.append(t[:, 0])
            pat_out.append(t[:, 1:])
        n = self.vit.norm(t)
        cls_out[-1], pat_out[-1] = n[:, 0], n[:, 1:]
        return cls_out, pat_out


def amp_ctx(fspec: FeatureSpec, device):
    return torch.autocast('cuda', dtype=torch.float16,
                          enabled=(fspec.precision == 'fp16' and device.type == 'cuda'))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · DATA, SCENES, FEATURE CACHE
# ══════════════════════════════════════════════════════════════════════════════

_IMG_EXT = ('.jpg', '.jpeg', '.png')


def _frame_key(p: str) -> int:
    return int(os.path.basename(p).split('.')[0].split('_')[-1])


def list_frames(folder: str) -> List[str]:
    entries = sorted(glob.glob(os.path.join(folder, '*')))
    if entries and os.path.isdir(entries[0]):
        out = []
        for v in entries:
            out += sorted([p for p in glob.glob(os.path.join(v, '*'))
                           if p.lower().endswith(_IMG_EXT)], key=_frame_key)
        return out
    return sorted([p for p in entries if p.lower().endswith(_IMG_EXT)], key=_frame_key)


# ── ShanghaiTech non-human subset (test IDs: Morais et al., CVPR 2019, App. A) ─
SHT_TAG          = 'shanghaitech'
SHT_CAMS         = {'01', '06', '12'}
SHT_EXCLUDE      = {'12_013', '12_014'}   # reported contaminated; verify visually
SHT_TEST         = ('01_0130', '01_0135', '01_0136', '06_0144', '06_0145', '12_0152')
SHT_TRAIN_STRIDE = 10                      # fixed a priori (not tuned)


def _is_sht(cfg) -> bool:
    return _dataset_tag(cfg).lower() == SHT_TAG


def train_frame_list(cfg) -> List[str]:
    tdir = os.path.join(cfg.data_path, 'train', 'frames')
    if not _is_sht(cfg):
        st = max(1, int(getattr(cfg, 'train_stride', 1)))
        vids = [v for v in sorted(glob.glob(os.path.join(tdir, '*'))) if os.path.isdir(v)]
        if vids:                                             # stride within each video
            return [p for v in vids for p in list_frames(v)[::st]]
        return list_frames(tdir)[::st]

    vids = [v for v in sorted(glob.glob(os.path.join(tdir, '*'))) if os.path.isdir(v)]
    if not vids:
        raise FileNotFoundError(f'{tdir}: expected one sub-folder per training video')
    keep = [v for v in vids if os.path.basename(v).split('_')[0] in SHT_CAMS
            and os.path.basename(v) not in SHT_EXCLUDE]
    print(f'[sht] {len(keep)} training videos kept: {[os.path.basename(v) for v in keep]}')
    return [p for v in keep for p in sorted(
        [q for q in glob.glob(os.path.join(v, '*')) if q.lower().endswith(_IMG_EXT)],
        key=_frame_key)[::SHT_TRAIN_STRIDE]]


def load_rgb(path: str, size: int) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))


class FrameDataset(tud.Dataset):
    def __init__(self, files: List[str], fspec: FeatureSpec):
        self.files, self.size = list(files), fspec.image_size
        self.tf = T.Compose([T.ToTensor(), T.Normalize(list(fspec.mean), list(fspec.std))])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return {'img': self.tf(Image.fromarray(load_rgb(self.files[i], self.size))), 'idx': i}


def compute_dataset_stats(files: List[str], size: int, num_workers: int = 4):
    fs = FeatureSpec(image_size=size, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    loader = DataLoader(FrameDataset(files, fs), batch_size=32, num_workers=num_workers)
    s1 = torch.zeros(3, dtype=torch.float64)
    s2 = torch.zeros(3, dtype=torch.float64)
    n = 0
    for b in tqdm(loader, desc='[stats] mean/std', leave=False):
        x = b['img'].double()
        s1 += x.sum((0, 2, 3))
        s2 += (x * x).sum((0, 2, 3))
        n += x.numel() // 3
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


@dataclass
class Scene:
    name: str
    files: List[str]
    labels: np.ndarray        # int; -1 = unlabelled (excluded from metrics)


class Context:
    """Test videos, loaders, pixel-GT location and the shared feature cache."""

    def __init__(self, cfg: BenchConfig, fspec: FeatureSpec, teacher, device,
                 patch_layers=(11,)):
        self.pix = None              # PixelEval during an accuracy pass, else None
        self.cfg, self.fspec, self.teacher, self.device = cfg, fspec, teacher, device
        self.patch_layers = tuple(sorted(set(patch_layers)))
        self.train_files = train_frame_list(cfg)
        if not self.train_files:
            raise FileNotFoundError(f'no training frames under {cfg.data_path}train/frames')
        troot = os.path.join(cfg.data_path, 'test')
        self.scenes: List[Scene] = []
        for d in sorted(glob.glob(os.path.join(troot, 'frames', '*'))):
            if not os.path.isdir(d):
                continue
            if _is_sht(cfg) and not (os.path.basename(d) in SHT_TEST
                                     and os.path.basename(d)[:2] in SHT_CAMS):
                continue
            name, files = os.path.basename(d), list_frames(d)
            lab = np.asarray(np.load(os.path.join(troot, 'test_frame_mask', f'{name}.npy'),
                                     allow_pickle=True)).ravel().astype(np.int64)
            if len(lab) < len(files):
                raise ValueError(f'{name}: {len(files)} frames but only {len(lab)} labels')
            if len(lab) > len(files):
                if not cfg.allow_label_truncation:
                    raise ValueError(f'{name}: {len(lab)} labels for {len(files)} frames; verify '
                                     f'the alignment or pass --allow_label_truncation')
                print(f'[WARN] {name}: {len(lab)} labels for {len(files)} frames; '
                      f'keeping the first {len(files)}')
                lab = lab[:len(files)]
            self.scenes.append(Scene(name, files, lab))
        if not self.scenes:
            raise FileNotFoundError(f'no test videos under {troot}/frames')

        scene_tag = _safe(_dataset_tag(cfg))
        if _is_sht(cfg):
            want = {v for v in SHT_TEST if v[:2] in SHT_CAMS}
            miss = want - {s.name for s in self.scenes}
            if miss:
                raise FileNotFoundError(f'missing test videos: {sorted(miss)}')
            scene_tag += f'_nonhuman_c{"".join(sorted(SHT_CAMS))}_s{SHT_TRAIN_STRIDE}'
        if not _is_sht(cfg) and cfg.train_stride > 1:
            scene_tag += f'_s{cfg.train_stride}'

        self.cache_root = os.path.join(cfg.cache_dir, scene_tag, fspec.key())
        self._mem: Dict[str, dict] = {}
        r = os.path.join(troot, cfg.pixel_mask_dir)
        self.mask_root = r if (cfg.pixel_eval and os.path.isdir(r)) else None
        print(f'[data] {scene_tag}: {len(self.train_files)} train frames, '
              f'{len(self.scenes)} test videos, {sum(len(s.files) for s in self.scenes)} '
              f'test frames, {int(self.labels.clip(0).sum())} anomalous; pixel GT: '
              f'{self.mask_root or "none (frame-level metrics only)"}')

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate([s.labels for s in self.scenes])

    def scene(self, name: str) -> Scene:
        return next(s for s in self.scenes if s.name == name)

    def loader(self, files: List[str]) -> DataLoader:
        return DataLoader(FrameDataset(files, self.fspec), batch_size=self.fspec.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=True, drop_last=False)

    def train_loader(self, idx=None) -> DataLoader:
        files = self.train_files if idx is None else [self.train_files[i] for i in sorted(idx)]
        return self.loader(files)

    # ── cache ─────────────────────────────────────────────────────────────
    def features(self, split: str, layers=()) -> dict:
        """split = 'train' or a test-video name. Returns {'cls', 'patches'}."""
        files = self.train_files if split == 'train' else self.scene(split).files
        base = os.path.join(self.cache_root, _safe(split))
        if split != 'train':                                   # NEW: keep at most one test video in RAM
            train_base = os.path.join(self.cache_root, _safe('train'))
            for k in list(self._mem):
                if not (k.startswith(train_base + '_') or k.startswith(base + '_')):
                    del self._mem[k]
            gc.collect()
        cls_path = base + '_cls.pth'
        ppath = lambda l: f'{base}_patchL{l}.pth'
        wanted = set(layers) | set(self.patch_layers)
        missing = sorted(l for l in wanted if not os.path.exists(ppath(l)))
        if missing or not os.path.exists(cls_path):
            self._extract(files, None if os.path.exists(cls_path) else cls_path,
                          {l: ppath(l) for l in missing})
        out = {'cls': self._load(cls_path, files)['cls'], 'patches': {}}
        for l in layers:
            out['patches'][l] = self._load(ppath(l), files)['patches']
        return out

    def _load(self, path: str, files: List[str]) -> dict:
        if path not in self._mem:
            c = _torch_load(path)
            if c.get('fkey') != self.fspec.key():
                raise RuntimeError(f'{path}: built with a different FeatureSpec; delete it')
            if list(c['files']) != list(files):
                raise RuntimeError(f'{path}: frame list changed since caching; delete it')
            self._mem[path] = c
        return self._mem[path]

    @torch.no_grad()
    def _extract(self, files, cls_path: Optional[str], patch_paths: Dict[int, str]):
        N, P = len(files), self.fspec.grid ** 2
        D, nb = self.teacher.embed_dim, self.teacher.n_blocks
        cls = torch.empty(nb, N, D, dtype=torch.float32) if cls_path else None
        pats = {l: torch.empty(N, P, D, dtype=_DT[self.fspec.patch_dtype]) for l in patch_paths}
        s = 0
        name = os.path.basename(cls_path or next(iter(patch_paths.values())))
        for b in tqdm(self.loader(files), desc=f'[cache] {name}', leave=False):
            x = b['img'].to(self.device, non_blocking=True)
            n = x.shape[0]
            with amp_ctx(self.fspec, self.device):
                c, p = self.teacher(x)
            if cls is not None:
                cls[:, s:s + n] = torch.stack([t.float() for t in c]).cpu()
            for l in pats:                                    # normalise in fp32, then store
                pats[l][s:s + n] = F.normalize(p[l].float(), dim=-1).to(pats[l].dtype).cpu()
            s += n
        meta = {'files': list(files), 'fkey': self.fspec.key(), 'fspec': asdict(self.fspec)}
        if cls_path:
            _atomic_save({'cls': cls, **meta}, cls_path)
        for l, path in patch_paths.items():
            _atomic_save({'patches': pats[l], **meta}, path)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · THE PROPOSED METHOD (single implementation)
# ══════════════════════════════════════════════════════════════════════════════

def cat_cls(cls_list) -> torch.Tensor:
    """Per-layer L2 normalisation, then concatenation (no layer dominates by norm)."""
    return torch.cat([F.normalize(c.float(), dim=-1) for c in cls_list], -1)


def fps_select(keys: torch.Tensor, n: int, seed: int) -> List[int]:
    """Greedy k-center on unit rows, float64 on CPU (identical on every machine)."""
    X = F.normalize(keys.double().cpu(), dim=1)
    N = X.shape[0]
    if n >= N:
        return list(range(N))
    start = int(np.random.default_rng(seed).integers(N))
    chosen = torch.zeros(N, dtype=torch.bool)
    sel = [start]
    chosen[start] = True
    d = 2.0 - 2.0 * (X @ X[start])
    for _ in range(1, n):
        d[chosen] = -float('inf')
        i = int(torch.argmax(d))
        sel.append(i)
        chosen[i] = True
        d = torch.minimum(d, 2.0 - 2.0 * (X @ X[i]))
    return sorted(sel)


def aggregate(d: torch.Tensor, m) -> torch.Tensor:
    """(P,) discrepancy -> 0-dim score. The ONLY aggregation code in the file."""
    d = d.float().reshape(-1)
    if m.agg == 'mean':
        return d.mean()
    if m.agg == 'max':
        return d.max()
    if m.agg == 'topk_mean':
        k = max(1, int(round(m.topk_ratio * d.numel())))
        return torch.topk(d, k).values.mean()
    if m.agg == 'robust_mad':
        med = d.median()
        sig = 1.4826 * (d - med).abs().median() + 1e-12
        mask = d > med + m.mad_sigma * sig
        k = max(1, int(0.05 * d.numel()))
        return torch.where(mask.any(), (d * mask).sum() / mask.sum().clamp(min=1),
                           torch.topk(d, k).values.mean())
    raise ValueError(f'unknown aggregator {m.agg!r}')


def _gap(pats: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    return torch.cat([pats[s:s + chunk].float().mean(1) for s in range(0, pats.shape[0], chunk)])


def query_keys(cls: torch.Tensor, pats: torch.Tensor, m: MethodSpec) -> torch.Tensor:
    """cls (n_blocks, N, D), pats (N, P, D) unit-norm -> (N, Dk) float32."""
    s = m.key_source
    if s in ('concat_cls', 'random'):
        return cat_cls([cls[l] for l in m.selected_layers])
    if s == 'last_cls':
        return cls[m.selected_layers[-1]].float()
    if s == 'layer5_cls':
        return cls[5].float()
    if s == 'gap':
        return _gap(pats)
    raise ValueError(f'unknown key_source {s!r}')


class ProposedScorer:
    """
    Stage 1: retrieve the K training frames whose keys are nearest the query.
    Stage 2: 1 - max cosine of every query patch against all patches of those
             K frames; reduce the (P,) map with the aggregation rule.
    """
    def __init__(self, m: MethodSpec, device, gpu_budget_mb: float = 4096):
        self.m, self.device, self.budget = m, device, gpu_budget_mb

    def fit(self, cls: torch.Tensor, pats: torch.Tensor, bank_idx=None) -> 'ProposedScorer':
        m, N = self.m, cls.shape[1]
        if bank_idx is None:
            sel_keys = cat_cls([cls[l] for l in m.selected_layers])
            if m.fps_ratio is None or m.fps_ratio >= 1.0:
                bank_idx = list(range(N))
            else:
                n_bank = min(N, max(int(m.fps_ratio * N), int(m.fps_min_frames), 1))
                bank_idx = fps_select(sel_keys, n_bank, m.fps_seed)
        self.idx = sorted(int(i) for i in bank_idx)
        it = torch.as_tensor(self.idx, dtype=torch.long)
        bank = pats[it]                                            # (n, P, D)
        keys = query_keys(cls[:, it], bank, m)
        self.proj = None
        if m.key_proj_dim:
            _, _, Vt = torch.linalg.svd(keys.double(), full_matrices=False)
            k = min(int(m.key_proj_dim), Vt.shape[0])
            self.proj = Vt[:k].float().to(self.device)
            keys = keys @ Vt[:k].float().T
        self.key_dim_eff = int(keys.shape[1])
        if m.key_metric in ('cosine', 'l2_unit'):
            keys = F.normalize(keys, dim=1)
        self.keys = keys.to(self.device).contiguous()
        mb = bank.numel() * bank.element_size() / 2 ** 20
        if self.device.type == 'cuda' and mb <= self.budget:
            self.bank, self.on_gpu = bank.to(self.device), True
        else:
            self.bank, self.on_gpu = bank, False
            if self.device.type == 'cuda':
                self.bank = bank.pin_memory()
                print(f'[Proposed] patch bank {mb:.0f} MB > budget {self.budget:.0f} MB: '
                      f'kept in pinned host memory (slower, identical scores)')
        return self

    def _k(self) -> int:
        n, k = len(self.idx), self.m.knn_k
        return n if (k <= 0 or k >= n) else k

    def _retrieve(self, qkey: torch.Tensor, qid: int):
        """-> (sorted indices on device or None = all frames, concat diagnostic)."""
        m, n, k = self.m, len(self.idx), self._k()
        nan = torch.tensor(float('nan'))
        if k == n:
            return None, nan
        if m.key_source == 'random':
            g = torch.Generator().manual_seed(m.fps_seed * 1_000_003 + int(qid))
            return torch.randperm(n, generator=g)[:k].sort().values.to(self.device), nan
        q = qkey.float().to(self.device)
        if self.proj is not None:
            q = q @ self.proj.T
        if m.key_metric == 'cosine':
            v, i = torch.topk(self.keys @ F.normalize(q, dim=0), k)
            return i.sort().values, 1.0 - v.mean()
        if m.key_metric == 'l2_unit':
            q = F.normalize(q, dim=0)
        d = torch.cdist(q[None], self.keys)[0]
        v, i = torch.topk(d, k, largest=False)
        return i.sort().values, v.mean()

    @torch.no_grad()
    def _discrepancy(self, q: torch.Tensor, inds, chunk: int = 8) -> torch.Tensor:
        q = q.to(self.device).float()
        n = len(self.idx)
        sel = torch.arange(n, device=self.device) if inds is None else inds
        best = None
        for s in range(0, sel.numel(), chunk):
            ii = sel[s:s + chunk]
            blk = self.bank[ii if self.on_gpu else ii.cpu()]
            blk = blk.to(self.device, non_blocking=True).float()   # (c, P, D)
            if self.m.position_aligned:
                mx = (blk * q[None]).sum(-1).amax(0)
            else:
                mx = (q @ blk.reshape(-1, q.shape[1]).T).amax(1)
            best = mx if best is None else torch.maximum(best, mx)
        return 1.0 - best

    @torch.no_grad()
    def score_split(self, cls, pats, rules: Optional[Dict[str, Agg]] = None,
                    keep_maps=(), qid_offset: int = 0) -> dict:
        keys = query_keys(cls, pats, self.m)
        M = keys.shape[0]
        scores, concat = np.empty(M), np.empty(M)
        rules = rules or {}
        extra = {n: np.empty(M) for n in rules}
        maps, keep = {}, set(keep_maps)
        for i in range(M):
            inds, cs = self._retrieve(keys[i], qid_offset + i)
            d = self._discrepancy(pats[i].float(), inds)
            scores[i] = float(aggregate(d, self.m))
            concat[i] = float(cs)
            for n, r in rules.items():
                extra[n][i] = float(aggregate(d, r))
            if i in keep:
                maps[i] = d.detach().float().cpu()          # host float32: no ties, no GPU growth
        return {'scores': scores, 'concat': concat, 'rules': extra, 'maps': maps}

    def _live_key(self, c, p, fspec) -> torch.Tensor:
        s = self.m.key_source
        if s in ('concat_cls', 'random'):
            return cat_cls([c[l][0] for l in self.m.selected_layers])
        if s == 'last_cls':
            return c[self.m.selected_layers[-1]][0].float()
        if s == 'layer5_cls':
            return c[5][0].float()
        q = F.normalize(p[self.m.patch_layer][0].float(), dim=-1)
        return q.to(_DT[fspec.patch_dtype]).float().mean(0)

    @torch.no_grad()
    def score_live(self, teacher, img: torch.Tensor, fspec: FeatureSpec) -> float:
        with PROF.stage('backbone'):
            with amp_ctx(fspec, img.device):
                c, p = teacher(img)
        with PROF.stage('search'):
            inds, _ = self._retrieve(self._live_key(c, p, fspec), 0)
        with PROF.stage('patch_score'):
            q = F.normalize(p[self.m.patch_layer][0].float(), dim=-1)
            q = q.to(_DT[fspec.patch_dtype]).float()        # same rounding as the cache
            s = aggregate(self._discrepancy(q, inds), self.m)
        return float(s)

    def bank_stats(self) -> dict:
        n, P = len(self.idx), int(self.bank.shape[1])
        pmb = self.bank.numel() * self.bank.element_size() / 2 ** 20
        gmb = self.keys.numel() * self.keys.element_size() / 2 ** 20
        return {'bank_kind': 'exemplar', 'bank_patch_vectors': n * P,
                'bank_global_vectors': n, 'bank_dim': int(self.bank.shape[-1]),
                'bank_mb': pmb + gmb, 'bank_on_gpu': bool(self.on_gpu)}


def dump_run(path: str, scorer: ProposedScorer, fkey: str, scores, labels):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.save(path, {'mkey': scorer.m.key(), 'fkey': fkey, 'spec': asdict(scorer.m),
                   'bank_idx': list(scorer.idx), 'scores': np.asarray(scores, np.float64),
                   'labels': np.asarray(labels)}, allow_pickle=True)


def assert_equivalent(path_a: str, path_b: str, atol: float = 1e-6):
    a = np.load(path_a, allow_pickle=True).item()
    b = np.load(path_b, allow_pickle=True).item()
    assert a['fkey'] == b['fkey'], 'different FeatureSpec'
    assert a['mkey'] == b['mkey'], 'different MethodSpec'
    assert a['bank_idx'] == b['bank_idx'], 'different bank frames'
    assert np.array_equal(a['labels'], b['labels']), 'different label alignment'
    diff = float(np.max(np.abs(a['scores'] - b['scores'])))
    assert diff <= atol, f'max score difference {diff:.2e} > {atol}'
    print(f'[equivalent] {len(a["scores"])} frames, bank {len(a["bank_idx"])} frames, '
          f'max |diff| = {diff:.2e}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · FRAME METRICS AND STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_eer(y, s) -> Tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
    fnr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    if 0 < i < len(fpr) - 1:
        f0, n0, f1, n1 = fpr[i - 1], fnr[i - 1], fpr[i], fnr[i]
        den = (f1 - f0) - (n1 - n0)
        if abs(den) > 1e-12:
            t = float(np.clip((n0 - f0) / den, 0, 1))
            return float(f0 + t * (f1 - f0)), float(thr[i - 1] + t * (thr[i] - thr[i - 1]))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def frame_metrics(labels, scores) -> dict:
    y, s = np.asarray(labels), np.asarray(scores, np.float64)
    v = (y >= 0) & np.isfinite(s)
    y, s = y[v], s[v]
    if len(np.unique(y)) < 2:
        return {}
    eer, thr = compute_eer(y, s)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    return {'auroc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)),
            'eer': eer, 'eer_threshold': thr, 'tpr_at_fpr05': float(np.interp(0.05, fpr, tpr)),
            'n_frames': int(len(y)), 'n_anomalous': int(y.sum())}


def per_video_auroc(scenes: List[Scene], scores) -> Dict[str, float]:
    out, off = {}, 0
    for sc in scenes:
        n = len(sc.files)
        y, s = sc.labels, np.asarray(scores[off:off + n])
        off += n
        v = y >= 0
        if len(np.unique(y[v])) == 2:
            out[sc.name] = float(roc_auc_score(y[v], s[v]))
    return out


def auto_block_len(scenes: List[Scene], scores, thr=0.2, lo=5, hi=200) -> int:
    """Block length = median lag at which the score ACF first drops below thr."""
    lags, off = [], 0
    for sc in scenes:
        x = np.asarray(scores[off:off + len(sc.files)], np.float64)
        off += len(sc.files)
        if len(x) < 2 * lo or x.std() == 0:
            continue
        x = (x - x.mean()) / x.std()
        L = next((l for l in range(1, min(hi, len(x) // 2))
                  if np.mean(x[:-l] * x[l:]) < thr), hi)
        lags.append(L)
    return int(np.clip(np.median(lags) if lags else lo, lo, hi))


def block_bootstrap(scenes: List[Scene], labels, ref, var, L: int,
                    n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired moving-block bootstrap within each test video; delta = AUROC(var) - AUROC(ref)."""
    y = np.asarray(labels)
    a, b = np.asarray(ref, np.float64), np.asarray(var, np.float64)
    valid = y >= 0
    obs = roc_auc_score(y[valid], b[valid]) - roc_auc_score(y[valid], a[valid])
    segs, off = [], 0
    for sc in scenes:
        segs.append((off, len(sc.files)))
        off += len(sc.files)
    rng = np.random.default_rng(seed)
    diffs, tries = [], 0
    while len(diffs) < n_boot and tries < 5 * n_boot:
        tries += 1
        parts = []
        for o, n in segs:
            if n <= L:
                parts.append(np.arange(o, o + n))
                continue
            starts = rng.integers(0, n - L + 1, math.ceil(n / L))
            parts.append((starts[:, None] + np.arange(L)[None]).ravel()[:n] + o)
        idx = np.concatenate(parts)
        idx = idx[valid[idx]]
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(yb, b[idx]) - roc_auc_score(yb, a[idx]))
    d = np.asarray(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = max(min(1.0, 2.0 * min((d <= 0).mean(), (d >= 0).mean())), 1.0 / len(d))
    return {'delta': float(obs), 'ci_low': float(lo), 'ci_high': float(hi),
            'p': float(p), 'n_boot': int(len(d)), 'block_len': int(L)}


def holm(ps: Dict[str, float]) -> Dict[str, float]:
    items = sorted((p, k) for k, p in ps.items() if p is not None)
    m, run, adj = len(items), 0.0, {}
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        adj[k] = run
    return adj


def print_table(rows: List[list], headers: List[str]):
    rows = [['' if c is None else str(c) for c in r] for r in rows]
    w = [max([len(h)] + [len(r[i]) for r in rows]) + 2 for i, h in enumerate(headers)]
    fmt = ''.join(f'{{:<{x}}}' for x in w)
    print('-' * sum(w))
    print(fmt.format(*headers))
    print('-' * sum(w))
    for r in rows:
        print(fmt.format(*r))
    print('-' * sum(w))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6b · PIXEL-LEVEL LOCALISATION (pixel AUROC, pixel AP, AUPRO@max_fpr)
# ══════════════════════════════════════════════════════════════════════════════
# Ground truth  {data_path}/test/{pixel_mask_dir}/<video>.npy  (T, H, W), value > 0
#               or {...}/<video>/<frame stem>[_mask|-mask|_gt].png (auto threshold)
# Frame policy  normal, no/empty mask -> all pixels normal (included)
#               anomalous, missing mask -> excluded
#               anomalous, empty mask / region lost in resizing -> excluded*
#               normal, non-empty mask -> excluded*      (* if pixel_skip_inconsistent)
#               unlabelled (-1) -> excluded
# Prediction    raw (P,) patch map -> bilinear to the GT aspect at eval size
#               (inverts the square input resize) -> Gaussian σ; same for all methods.
# Metrics       exact, from ONE sort: pixel AUROC (ties = 1/2), pixel AP (sklearn
#               definition), AUPRO (Bergmann et al., IJCV 2021) to max_fpr.

def _read_png_mask(path: str, thr: Optional[int] = None) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ('1', 'L', 'P', 'I', 'I;16', 'F'):
        im = im.convert('L')                       # RGB / RGBA / LA: luminance, alpha dropped
    a = np.asarray(im).astype(np.float64)
    t = thr if thr is not None else (0 if a.max() <= 1 else 127)
    return a > t


class PixelGT:
    _SUF = ('_mask', '-mask', '_gt')

    def __init__(self, root: str):
        self.root, self._name, self._arr, self._png = root, None, None, None

    def _open(self, sc: Scene):
        if sc.name == self._name:
            return
        self._name, self._arr, self._png = sc.name, None, None
        p, d = os.path.join(self.root, f'{sc.name}.npy'), os.path.join(self.root, sc.name)
        if os.path.exists(p):
            try:
                g = np.load(p, mmap_mode='r')
            except ValueError:                     # object arrays cannot be memory-mapped
                g = np.load(p, allow_pickle=True)
            if g.ndim != 3 or len(g) < len(sc.files):
                raise ValueError(f'{p}: shape {g.shape}, expected (>= {len(sc.files)}, H, W)')
            self._arr = g
        elif os.path.isdir(d):
            idx = {}
            for q in glob.glob(os.path.join(d, '**', '*'), recursive=True):
                if not q.lower().endswith('.png'):
                    continue
                stem = os.path.splitext(os.path.basename(q))[0]
                for suf in self._SUF:
                    if stem.endswith(suf):
                        stem = stem[:-len(suf)]
                        break
                idx.setdefault(stem, []).append(q)
            self._png = idx

    def get(self, sc: Scene, f: int) -> Tuple[str, Optional[np.ndarray]]:
        """-> ('mask', bool array at native resolution) or ('missing', None)."""
        self._open(sc)
        if self._arr is not None:
            return 'mask', np.asarray(self._arr[f]) > 0
        if self._png:
            c = self._png.get(os.path.splitext(os.path.basename(sc.files[f]))[0], [])
            if len(c) == 1:
                return 'mask', _read_png_mask(c[0])
        return 'missing', None

    def native_shape(self, scenes) -> Optional[Tuple[int, int]]:
        for sc in scenes:
            self._open(sc)
            if self._arr is not None:
                return tuple(int(x) for x in self._arr.shape[1:])
            if self._png:
                return _read_png_mask(next(iter(self._png.values()))[0]).shape
        return None


def pixel_metrics_exact(maps: np.ndarray, masks: np.ndarray, max_fpr: float = 0.3) -> dict:
    """Exact pixel AUROC, pixel AP and AUPRO@max_fpr from a single descending sort."""
    nan = float('nan')
    y = masks.reshape(-1).astype(bool, copy=False)
    n_pos, n_neg = int(y.sum()), int(y.size - y.sum())
    out = {'pixel_auroc': nan, 'pixel_ap': nan, 'aupro': nan, 'pixel_regions': 0,
           'pixel_anom_fraction': n_pos / max(y.size, 1)}
    if n_pos == 0 or n_neg == 0:
        return out
    w = np.zeros(y.size, np.float64)                 # per-pixel weight 1/|component|
    wv, n_comp = w.reshape(masks.shape), 0
    for i in range(masks.shape[0]):
        if not masks[i].any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(masks[i].astype(np.uint8), connectivity=8)
        area = st[:, cv2.CC_STAT_AREA].astype(np.float64)
        wv[i] = np.where(lab > 0, 1.0 / area[lab], 0.0)
        n_comp += n - 1
    s = maps.reshape(-1)
    order = np.argsort(-s, kind='stable')
    ss = s[order]
    last = np.r_[ss[1:] != ss[:-1], True]           # one point per distinct threshold
    del ss
    yo = y[order]
    tp = np.cumsum(yo, dtype=np.float64)[last]
    fp = np.cumsum(~yo, dtype=np.float64)[last]
    del yo
    pro = (np.cumsum(w[order]) / n_comp)[last]
    del w, wv, order
    tpr, fpr = np.r_[0.0, tp / n_pos], np.r_[0.0, fp / n_neg]
    out['pixel_auroc'] = float(_trapz(tpr, fpr))
    out['pixel_ap'] = float(np.sum(np.diff(np.r_[0.0, tp / n_pos]) * (tp / (tp + fp))))
    pro = np.r_[0.0, pro]
    j = int(np.searchsorted(fpr, max_fpr, side='right'))
    x = np.r_[fpr[:j], max_fpr]
    v = np.r_[pro[:j], np.interp(max_fpr, fpr, pro)]
    out['aupro'] = float(_trapz(v, x) / max_fpr)
    out['pixel_regions'] = int(n_comp)
    return out


def selftest_pixel_metrics(max_fpr: float = 0.3, tol: float = 0.02):
    """Known-answer AUPRO cases and exact agreement of AUROC / AP with scikit-learn."""
    rng = np.random.default_rng(0)
    G = np.zeros((20, 64, 64), bool)
    G[:, 20:30, 20:40] = True
    noise = 1e-3 * rng.random(G.shape, dtype=np.float32)
    cases = {'perfect':  (G.astype(np.float32) + noise, 1.0),
             'random':   (rng.random(G.shape, dtype=np.float32), max_fpr / 2),
             'inverted': ((~G).astype(np.float32) + noise, 0.0),
             'constant': (np.zeros(G.shape, np.float32), max_fpr / 2)}
    for name, (M, want) in cases.items():
        got = pixel_metrics_exact(M, G, max_fpr)['aupro']
        assert abs(got - want) <= tol, f'AUPRO self-test "{name}": got {got:.4f}, want {want:.4f}'
    y = rng.random((4, 16, 16)) < 0.2
    s = (np.round(rng.random(y.shape), 2) + 0.5 * y).astype(np.float32)   # many ties
    r = pixel_metrics_exact(s, y, max_fpr)
    assert abs(r['pixel_auroc'] - roc_auc_score(y.ravel(), s.ravel())) < 1e-7, 'pixel AUROC'
    assert abs(r['pixel_ap'] - average_precision_score(y.ravel(), s.ravel())) < 1e-7, 'pixel AP'
    print('[selftest] pixel metrics: AUPRO known answers and AUROC/AP vs scikit-learn passed')


class PixelEval:
    """Collects (anomaly map, GT mask) pairs at one evaluation resolution for one method."""

    @staticmethod
    def make(ctx) -> Optional['PixelEval']:
        if not (ctx.cfg.pixel_eval and getattr(ctx, 'mask_root', None)):
            return None
        pe = PixelEval(ctx)
        return pe if pe.hw else None

    def __init__(self, ctx):
        c = ctx.cfg
        self.gt = PixelGT(ctx.mask_root)
        if c.pixel_eval_hw:
            self.hw = tuple(int(v) for v in c.pixel_eval_hw)
        else:
            shp = self.gt.native_shape(ctx.scenes)
            self.hw = None if shp is None else (max(1, shp[0] // c.pixel_eval_div),
                                                max(1, shp[1] // c.pixel_eval_div))
        self.sigma, self.max_fpr = c.pixel_smooth_sigma, c.pixel_max_fpr
        self.skip = c.pixel_skip_inconsistent
        self.maps, self.masks = [], []
        self.n = dict.fromkeys(('used', 'no_map', 'unlabelled', 'normal_no_mask',
                                'anom_missing_mask', 'anom_empty_mask', 'anom_lost_resize',
                                'normal_with_region'), 0)
        self.agree = self.checked = 0

    def add(self, sc: Scene, f: int, amap, grid=None):
        y = int(sc.labels[f])
        if y < 0:
            self.n['unlabelled'] += 1
            return
        if amap is None:
            self.n['no_map'] += 1
            return
        state, g = self.gt.get(sc, f)
        if state == 'missing':
            if y == 1:
                self.n['anom_missing_mask'] += 1
                return
            self.n['normal_no_mask'] += 1
        has = bool(g is not None and g.any())
        if state == 'mask':
            self.checked += 1
            self.agree += int((y == 1) == has)
        if y == 0 and has:
            self.n['normal_with_region'] += 1
            if self.skip:
                return
        if y == 1 and not has:
            self.n['anom_empty_mask'] += 1
            if self.skip:
                return
        h, w = self.hw
        gm = (cv2.resize(g.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
              if has else np.zeros((h, w), bool))
        if y == 1 and has and not gm.any():
            self.n['anom_lost_resize'] += 1
            if self.skip:
                return
        a = (amap.detach().float().cpu().numpy() if torch.is_tensor(amap)
             else np.asarray(amap, np.float32))
        if grid is not None:
            a = a.reshape(grid)
        a = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        if self.sigma > 0:
            a = cv2.GaussianBlur(a, (0, 0), self.sigma, borderType=cv2.BORDER_REFLECT)
        self.maps.append(a)
        self.masks.append(gm)
        self.n['used'] += 1

    def metrics(self) -> dict:
        nan = float('nan')
        incons = ('anom_empty_mask', 'anom_lost_resize', 'normal_with_region')
        base = {'pixel_eval_hw': f'{self.hw[0]}x{self.hw[1]}', 'pixel_frames': self.n['used'],
                'pixel_label_mask_agreement': self.agree / max(self.checked, 1),
                'pixel_excluded': self.n['unlabelled'] + self.n['anom_missing_mask']
                + (sum(self.n[k] for k in incons) if self.skip else 0),
                'pixel_counts': dict(self.n), 'pixel_auroc': nan, 'pixel_ap': nan, 'aupro': nan}
        if self.n['no_map']:
            print(f'[pixel] {self.n["no_map"]} frames had no anomaly map; pixel metrics skipped')
            return base
        if not self.maps:
            return base
        M, G = np.stack(self.maps), np.stack(self.masks)
        self.maps = self.masks = None
        r = pixel_metrics_exact(M, G, self.max_fpr)
        del M, G
        print(f'[pixel] {base["pixel_eval_hw"]}: {self.n["used"]} frames, {r["pixel_regions"]} '
              f'regions, anomalous px {100 * r["pixel_anom_fraction"]:.2f}%  '
              f'pAUROC={_f(r["pixel_auroc"])} pAP={_f(r["pixel_ap"])} '
              f'AUPRO@{self.max_fpr:g}={_f(r["aupro"])}  excluded={base["pixel_excluded"]} {self.n}')
        if base['pixel_label_mask_agreement'] < 0.95:
            print(f'[pixel][WARN] label/mask agreement {base["pixel_label_mask_agreement"]:.3f} '
                  f'< 0.95: check for a frame-index offset between frames and masks')
        return {**base, **r}


def method_grid(method, ctx) -> Tuple[int, int]:
    ex = getattr(method, 'ex', None)
    return tuple(ex.grid_size) if ex is not None else (ctx.fspec.grid, ctx.fspec.grid)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · SHARED GPU BANKS AND EXACT 1-NN SEARCH (baselines)
# ══════════════════════════════════════════════════════════════════════════════

def _prep_rows(t, store: torch.dtype, normalise: bool, chunk: int = 262_144) -> torch.Tensor:
    t = t if isinstance(t, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(t))
    t = t.detach().cpu()
    flat = t.reshape(-1, t.shape[-1])
    out = torch.empty(flat.shape, dtype=store)
    for s in range(0, flat.shape[0], chunk):
        b = flat[s:s + chunk].float()
        out[s:s + b.shape[0]] = (F.normalize(b, dim=1) if normalise else b).to(store)
    return out.reshape(t.shape)


def _row_sq(bank: torch.Tensor, device, chunk: int = 262_144) -> torch.Tensor:
    flat = bank.reshape(-1, bank.shape[-1])
    out = torch.empty(flat.shape[0], dtype=torch.float32, device=device)
    for s in range(0, flat.shape[0], chunk):
        b = flat[s:s + chunk].to(device).float()
        out[s:s + b.shape[0]] = (b * b).sum(1)
    return out.reshape(bank.shape[:-1])


def _place_bank(t: torch.Tensor, device, budget_mb: float, name: str):
    mb = t.numel() * t.element_size() / 2 ** 20
    if device.type == 'cuda':
        if mb <= budget_mb:
            try:
                return t.to(device), True
            except _CUDA_OOM:
                _free(device)
        print(f'[{name}] {mb:.0f} MB kept in pinned host memory; latency includes PCIe transfer')
        return t.pin_memory(), False
    return t, False


@torch.no_grad()
def nn_over_frames(q, bank, inds, device, cdt, chunk_vectors, cosine=True,
                   bank_sq=None, bank_on_gpu=True) -> torch.Tensor:
    """Exact 1-NN distance of each query row to the (selected) bank frames."""
    q = q.to(device).float()
    if cosine:
        q = F.normalize(q, dim=1)
    qc = q.to(cdt)
    q2 = None if cosine else (q * q).sum(1, keepdim=True)
    N, Pb, D = bank.shape
    per = max(1, int(chunk_vectors) // max(Pb, 1))
    n_sel = N if inds is None else int(inds.numel())
    best = None
    for s in range(0, n_sel, per):
        if inds is None:
            blk = bank[s:s + per]
            sq = bank_sq[s:s + per] if bank_sq is not None else None
        else:
            sub = inds[s:s + per]
            blk = bank[sub.to(bank.device)]
            sq = bank_sq[sub.to(bank_sq.device)] if bank_sq is not None else None
        if not bank_on_gpu:
            blk = blk.to(device, non_blocking=True)
        prod = qc @ blk.reshape(-1, D).to(cdt).T
        if cosine:
            m = prod.amax(1).float()
            best = m if best is None else torch.maximum(best, m)
        else:
            m = (q2 - 2.0 * prod.float() + sq.reshape(1, -1)).amin(1)
            best = m if best is None else torch.minimum(best, m)
    return 1.0 - best if cosine else best.clamp_min(0).sqrt()


class GpuNNBank:
    def __init__(self, feats, metric: str, device, cfg: BenchConfig, name: str = 'bank'):
        self.cosine, self.device = metric == 'cosine', device
        self.cdt, self.chunk = _DT[cfg.nn_compute_dtype], cfg.nn_chunk_vectors
        rows = _prep_rows(feats, _DT[cfg.bank_dtype], self.cosine)
        rows = rows.reshape(-1, 1, rows.shape[-1])
        self.bank, self.on_gpu = _place_bank(rows, device, cfg.gpu_bank_budget_mb, name)
        self.sq = None if self.cosine else _row_sq(self.bank, device)

    @property
    def n(self):
        return int(self.bank.shape[0])

    @property
    def dim(self):
        return int(self.bank.shape[-1])

    @property
    def mb(self):
        t = self.bank.numel() * self.bank.element_size()
        if self.sq is not None:
            t += self.sq.numel() * 4
        return t / 2 ** 20

    def nn_distance(self, q):
        return nn_over_frames(q, self.bank, None, self.device, self.cdt, self.chunk,
                              self.cosine, self.sq, self.on_gpu)

    def stats(self) -> dict:
        return {'bank_kind': 'exemplar', 'bank_patch_vectors': self.n,
                'bank_global_vectors': 0, 'bank_dim': self.dim, 'bank_mb': self.mb,
                'bank_on_gpu': bool(self.on_gpu)}


@torch.no_grad()
def greedy_coreset(feats: torch.Tensor, n: int, device, proj_dim=None, normalise=True,
                   seed: int = 0, storage: str = 'auto', chunk: int = 131_072) -> torch.Tensor:
    """Greedy k-center (PatchCore Sec. 3.2); exact in the full space by default."""
    N, D = feats.shape
    if n >= N:
        return torch.arange(N)
    use_proj = proj_dim is not None and 0 < proj_dim < D
    d = proj_dim if use_proj else D
    unit = normalise and not use_proj
    cuda = device.type == 'cuda'
    order = ['fp32', 'fp16', 'cpu'] if cuda else ['cpu']
    if storage != 'auto':
        order = order[order.index(storage):]
    elif cuda:
        _free(device)
        try:
            free = torch.cuda.mem_get_info(device)[0]
        except Exception:
            free = 0
        need = lambda b: N * d * b + 8 * N * 4 + (768 << 20)
        order = [m for m, b in (('fp32', 4), ('fp16', 2)) if need(b) <= free] + ['cpu']
    proj = None
    if use_proj:
        g = torch.Generator().manual_seed(seed)
        proj = torch.randn(D, proj_dim, generator=g) / math.sqrt(proj_dim)
    Z = sq = None
    mode, work = 'cpu', torch.device('cpu')
    for mode in order:
        work = device if mode != 'cpu' else torch.device('cpu')
        zdt = torch.float16 if mode == 'fp16' else torch.float32
        try:
            Z = torch.empty(N, d, dtype=zdt, device=work)
            sq = None if unit else torch.empty(N, device=work)
            p = proj.to(work) if proj is not None else None
            for s in range(0, N, chunk):
                x = feats[s:s + chunk].to(work).float()
                if normalise:
                    x = F.normalize(x, dim=1)
                if p is not None:
                    x = x @ p
                x = x.to(zdt)
                Z[s:s + len(x)] = x
                if sq is not None:
                    sq[s:s + len(x)] = (x.float() ** 2).sum(1)
            break
        except _CUDA_OOM:
            Z = sq = None
            _free(device)
            print(f'[coreset] storage {mode!r} does not fit; trying the next option')
    if Z is None:
        raise RuntimeError('greedy_coreset: cannot allocate the working matrix')
    print(f'[coreset] {N:,} -> {n:,}  d={d}  storage={mode}')

    def dist(i):
        dots = torch.mv(Z, Z[i]).float()
        return (2.0 - 2.0 * dots).clamp_min_(0) if unit else (sq - 2.0 * dots + sq[i]).clamp_min_(0)

    sel = torch.empty(n, dtype=torch.long, device=work)
    sel[0] = 0
    dmin = dist(0)
    for k in tqdm(range(1, n), desc='  [coreset]', leave=False, mininterval=2.0):
        i = torch.argmax(dmin)
        sel[k] = i
        torch.minimum(dmin, dist(i), out=dmin)
    out = sel.cpu()
    del Z, sq, dmin, sel
    _free(device)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · METHOD INTERFACE, VISUALISATION, PATCH EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

def _upsample(d: torch.Tensor, grid: Tuple[int, int], size) -> np.ndarray:
    h, w = grid
    return F.interpolate(d.reshape(1, 1, h, w).float(), size=tuple(size), mode='bilinear',
                         align_corners=False)[0, 0].cpu().numpy()


class Viz:
    """Every n-th eligible frame per video; writes PNG overlays; never timed."""

    def __init__(self, cfg: BenchConfig, method: str):
        self.dir, self.n = cfg.viz_dir, max(1, cfg.viz_every_n)
        self.anom, self.method, self.cnt, self.saved = cfg.viz_anomalous_only, method, {}, 0

    def want(self, sc: Scene, f: int) -> bool:
        if not self.dir or (self.anom and int(sc.labels[f]) != 1):
            return False
        i = self.cnt.get(sc.name, 0)
        self.cnt[sc.name] = i + 1
        return i % self.n == 0

    def save(self, sc: Scene, f: int, hm: np.ndarray):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        img = load_rgb(sc.files[f], hm.shape[0])
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(img); ax[0].set_title('Input'); ax[0].axis('off')
        ax[1].imshow(img); ax[1].imshow(hm, cmap='jet', alpha=0.6)
        ax[1].set_title('Anomaly map'); ax[1].axis('off')
        fig.suptitle(f'{self.method} | {sc.name} frame {f} | label={sc.labels[f]}')
        fig.tight_layout()
        path = os.path.join(self.dir, _safe(self.method), f'{sc.name}_f{f:05d}_y{sc.labels[f]}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        self.saved += 1


class Method:
    name, backbone, training_free = '?', 'DINOv2 ViT-B/14', True

    def fit(self, ctx: Context, train_idx=None):
        raise NotImplementedError

    def score(self, img, return_heatmap: bool = False):
        raise NotImplementedError

    def score_one(self, img) -> float:
        return self.score(img, False)[0]

    def score_frames(self, ctx: Context) -> dict:
        """Live scoring of every test frame; raw patch maps go to ctx.pix if active."""
        viz, out, pix = Viz(ctx.cfg, self.name), [], getattr(ctx, 'pix', None)
        grid = method_grid(self, ctx)
        self._want_map = pix is not None           # read by SPADE (image mode) only
        try:
            for sc in ctx.scenes:
                f = 0
                for b in tqdm(ctx.loader(sc.files), desc=f'  {self.name} | {sc.name}', leave=False):
                    x = b['img'].to(ctx.device, non_blocking=True)
                    for j in range(x.shape[0]):
                        want = viz.want(sc, f)
                        self._last_map = None
                        s, hm = self.score(x[j:j + 1], return_heatmap=want)
                        out.append(s)
                        if want and hm is not None:
                            viz.save(sc, f, hm)
                        if pix is not None:
                            pix.add(sc, f, self._last_map, grid)
                        f += 1
        finally:
            self._want_map = False                 # latency pass must not compute extra maps
        return {'scores': np.asarray(out, np.float64)}

    def bank_stats(self) -> dict:
        return {}

    def trainable_params(self) -> int:
        return 0


class DinoPatches:
    def __init__(self, teacher, layers, fspec: FeatureSpec, device):
        self.t, self.layers, self.fspec, self.device = teacher, tuple(layers), fspec, device
        self.feat_dim = teacher.embed_dim * len(self.layers)
        self.grid_size = (fspec.grid, fspec.grid)

    @torch.no_grad()
    def __call__(self, imgs):
        with amp_ctx(self.fspec, self.device):
            _, p = self.t(imgs.to(self.device, non_blocking=True))
        return torch.cat([p[l].float() for l in self.layers], -1)      # (B, P, D)


class CNNPatches:
    def __init__(self, name: str, device, out_idx, grid: int):
        self.m = timm.create_model(name, pretrained=True, features_only=True,
                                   out_indices=tuple(out_idx)).to(device).eval().requires_grad_(False)
        self.device, self.grid_size = device, (grid, grid)
        with torch.no_grad():
            self.feat_dim = sum(o.shape[1] for o in self.m(torch.zeros(1, 3, 224, 224, device=device)))

    @torch.no_grad()
    def __call__(self, imgs):
        g = self.grid_size[0]
        fs = self.m(imgs.to(self.device))
        x = torch.cat([F.interpolate(f, size=(g, g), mode='bilinear', align_corners=False)
                       for f in fs], 1)
        return x.flatten(2).transpose(1, 2).float()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 · METHODS
# ══════════════════════════════════════════════════════════════════════════════

class ProposedMethod(Method):
    """Benchmark wrapper around ProposedScorer (cached accuracy, live latency)."""

    def __init__(self, spec: MethodSpec, ctx: Context, name='Proposed (DINOv2-MemKNN)',
                 bank_idx=None):
        self.spec, self.ctx, self.name, self.bank_idx = spec, ctx, name, bank_idx
        self.scorer = ProposedScorer(spec, ctx.device, ctx.cfg.gpu_bank_budget_mb)

    def fit(self, ctx: Context, train_idx=None):
        pl = self.spec.patch_layer
        tr = ctx.features('train', [pl])
        cls, pats = tr['cls'], tr['patches'][pl]
        if train_idx is not None:
            it = torch.as_tensor(sorted(train_idx), dtype=torch.long)
            cls, pats = cls[:, it], pats[it]
        self.scorer.fit(cls, pats, self.bank_idx)

    def score_frames(self, ctx: Context, rules=None) -> dict:
        pix = getattr(ctx, 'pix', None)
        pl, viz = self.spec.patch_layer, Viz(ctx.cfg, self.name)
        g, size = (ctx.fspec.grid,) * 2, (ctx.fspec.image_size,) * 2
        acc = {'scores': [], 'concat': []}
        racc = {n: [] for n in (rules or {})}
        off = 0
        for sc in ctx.scenes:
            te = ctx.features(sc.name, [pl])
            n = len(sc.files)
            want = [f for f in range(n) if viz.want(sc, f)]
            r = self.scorer.score_split(te['cls'], te['patches'][pl], rules=rules,
                                        keep_maps=range(n) if pix is not None else want,
                                        qid_offset=off)
            wset = set(want)
            for f, d in r['maps'].items():
                if f in wset:
                    viz.save(sc, f, _upsample(d, g, size))
                if pix is not None:
                    pix.add(sc, f, d, g)
            acc['scores'].append(r['scores'])
            acc['concat'].append(r['concat'])
            for k in racc:
                racc[k].append(r['rules'][k])
            off += n
        out = {k: np.concatenate(v) for k, v in acc.items()}
        out['rules'] = {k: np.concatenate(v) for k, v in racc.items()}
        return out

    def score_one(self, img) -> float:
        return self.scorer.score_live(self.ctx.teacher, img, self.ctx.fspec)

    def bank_stats(self) -> dict:
        return self.scorer.bank_stats()


class StreamingProposedMethod(ProposedMethod):
    """
    Proposed method without the feature cache. Extraction uses the SAME batches
    (frames, order, batch size) as the cache, so scores are bit-identical to the
    cached path and the ablate equivalence gate still holds.
    """
    @torch.no_grad()
    def _forward(self, ctx, x):
        with amp_ctx(ctx.fspec, ctx.device):
            c, p = ctx.teacher(x.to(ctx.device, non_blocking=True))
        cls = torch.stack([t.float() for t in c]).cpu()                       # (n_blocks, B, D)
        pat = F.normalize(p[self.spec.patch_layer].float(), dim=-1)
        return cls, pat.to(_DT[ctx.fspec.patch_dtype]).cpu()                  # same rounding as cache

    @torch.no_grad()
    def fit(self, ctx: Context, train_idx=None):
        m = self.spec
        files = ctx.train_files if train_idx is None else [ctx.train_files[i] for i in sorted(train_idx)]
        N, B = len(files), ctx.fspec.batch_size
        cls = torch.cat([self._forward(ctx, b['img'])[0]
                         for b in tqdm(ctx.loader(files), desc='[stream] train CLS', leave=False)], dim=1)
        idx = self.bank_idx
        if idx is None:
            keys = cat_cls([cls[l] for l in m.selected_layers])
            idx = (list(range(N)) if (m.fps_ratio is None or m.fps_ratio >= 1.0)
                   else fps_select(keys, min(N, max(int(m.fps_ratio * N), int(m.fps_min_frames), 1)), m.fps_seed))
        idx = sorted(int(i) for i in idx)
        want = set(idx)
        chunks = [list(range(s, min(s + B, N))) for s in range(0, N, B)]    # = the cache's batches
        chunks = [ch for ch in chunks if want.intersection(ch)]
        loader = DataLoader(ctx.loader(files).dataset, batch_sampler=chunks,
                            num_workers=ctx.cfg.num_workers, pin_memory=True)
        rows = {}
        for b in tqdm(loader, desc='[stream] bank patches', leave=False):
            _, pat = self._forward(ctx, b['img'])
            for j, i in enumerate(b['idx'].tolist()):
                if i in want:
                    rows[i] = pat[j]
        pats = torch.stack([rows[i] for i in idx])
        del rows
        self.scorer.fit(cls[:, torch.as_tensor(idx)], pats, bank_idx=list(range(len(idx))))
        self.scorer.idx = idx                     # global training-frame indices (dumps, gate)

    @torch.no_grad()
    def score_frames(self, ctx: Context, rules=None) -> dict:
        pix = getattr(ctx, 'pix', None)
        m, viz, rules = self.spec, Viz(ctx.cfg, self.name), rules or {}
        scores, concat, racc, qid = [], [], {n: [] for n in rules}, 0
        g, size = (ctx.fspec.grid,) * 2, (ctx.fspec.image_size,) * 2
        for sc in ctx.scenes:
            f = 0
            for b in tqdm(ctx.loader(sc.files), desc=f'  {self.name} | {sc.name}', leave=False):
                cls, pat = self._forward(ctx, b['img'])
                keys = query_keys(cls, pat, m)
                for j in range(pat.shape[0]):
                    inds, cs = self.scorer._retrieve(keys[j], qid)
                    d = self.scorer._discrepancy(pat[j].float(), inds)
                    scores.append(float(aggregate(d, m)))
                    concat.append(float(cs))
                    for n, r in rules.items():
                        racc[n].append(float(aggregate(d, r)))
                    if viz.want(sc, f):
                        viz.save(sc, f, _upsample(d, g, size))
                    if pix is not None:
                        pix.add(sc, f, d, g)
                    f += 1
                    qid += 1
        return {'scores': np.asarray(scores), 'concat': np.asarray(concat),
                'rules': {n: np.asarray(v) for n, v in racc.items()}}


class PaDiM(Method):
    """
    Defard et al., ICPR 2021, on frozen patch features. Per-position Gaussians
    need spatially aligned images; a moving camera violates this, so treat
    PaDiM as an alignment-dependent reference, not a strong baseline.
    """
    def __init__(self, ex, cfg: BenchConfig, name: str):
        self.ex, self.cfg, self.name = ex, cfg, name

    @torch.no_grad()
    def fit(self, ctx: Context, train_idx=None):
        loader, dev = ctx.train_loader(train_idx), ctx.device
        P = self.ex.grid_size[0] * self.ex.grid_size[1]
        per = max(1, self.cfg.padim_pca_fit_rows // len(loader.dataset))
        g = torch.Generator().manual_seed(0)
        rows = []
        for b in tqdm(loader, desc=f'[{self.name}] PCA', leave=False):
            f = self.ex(b['img'])
            for j in range(f.shape[0]):
                rows.append(f[j, torch.randperm(P, generator=g)[:per].to(f.device)].cpu())
        X = torch.cat(rows).double()
        mu = X.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(X - mu, full_matrices=False)
        d = min(self.cfg.padim_pca_dim, Vt.shape[0], X.shape[1] - 1)
        self.mu, self.W = mu.float().to(dev), Vt[:d].float().T.contiguous().to(dev)
        del X, rows
        s1 = torch.zeros(P, d, dtype=torch.float64, device=dev)
        s2 = torch.zeros(P, d, d, dtype=torch.float64, device=dev)
        n = 0
        for b in tqdm(loader, desc=f'[{self.name}] Gaussians', leave=False):
            z = ((self.ex(b['img']) - self.mu) @ self.W).double()        # (B, P, d)
            s1 += z.sum(0)
            s2 += torch.einsum('bpi,bpj->pij', z, z)
            n += z.shape[0]
        mean = s1 / n
        cov = (s2 - n * torch.einsum('pi,pj->pij', mean, mean)) / max(n - 1, 1)
        cov += self.cfg.padim_eps * torch.eye(d, dtype=cov.dtype, device=dev)
        self.mean, self.prec = mean.float(), torch.linalg.inv(cov).float()

    @torch.no_grad()
    def score(self, img, return_heatmap=False):
        with PROF.stage('backbone'):
            f = self.ex(img)[0]
        with PROF.stage('search'):
            z = (f - self.mu) @ self.W - self.mean
            dist = torch.einsum('pi,pij,pj->p', z, self.prec, z).clamp_min(0).sqrt()
            self._last_map = dist
        with PROF.stage('patch_score'):
            s = dist.max()
        return float(s), (_upsample(dist, self.ex.grid_size, img.shape[-2:]) if return_heatmap else None)

    def bank_stats(self):
        mb = sum(t.numel() * t.element_size() for t in (self.mu, self.W, self.mean, self.prec)) / 2 ** 20
        return {'bank_kind': 'gaussian', 'bank_patch_vectors': int(self.mean.shape[0]),
                'bank_global_vectors': 0, 'bank_dim': int(self.mean.shape[1]), 'bank_mb': mb}


class PatchCore(Method):
    """Roth et al., CVPR 2022: locally aware features, greedy coreset, image score = max."""

    def __init__(self, ex, cfg: BenchConfig, device, name: str, coreset_n=None, agg=None):
        self.ex, self.cfg, self.device, self.name = ex, cfg, device, name
        self.coreset_n, self.agg = coreset_n, agg or Agg(cfg.patchcore_agg)
        self.bank, self.total = None, 0

    def _features(self, imgs):
        p, k = self.ex(imgs), self.cfg.patchcore_kernel
        if k > 1:
            (h, w), (B, P, D) = self.ex.grid_size, p.shape
            m = p.reshape(B, h, w, D).permute(0, 3, 1, 2)
            p = F.avg_pool2d(m, k, 1, k // 2).permute(0, 2, 3, 1).reshape(B, P, D)
        return p

    @torch.no_grad()
    def fit(self, ctx: Context, train_idx=None):
        self.bank = None
        _free(self.device)
        chunks = [self._features(b['img']).reshape(-1, self.ex.feat_dim).half().cpu()
                  for b in tqdm(ctx.train_loader(train_idx), desc=f'[{self.name}] fit', leave=False)]
        flat = torch.cat(chunks)
        del chunks
        _free(self.device)
        self.total = int(flat.shape[0])
        n = self.coreset_n or max(1, int(self.total * self.cfg.patchcore_ratio))
        sel = greedy_coreset(flat, min(n, self.total), self.device,
                             proj_dim=self.cfg.patchcore_proj_dim, storage=self.cfg.patchcore_storage)
        self.bank = GpuNNBank(flat[sel], 'cosine', self.device, self.cfg, self.name)
        del flat

    @torch.no_grad()
    def score(self, img, return_heatmap=False):
        with PROF.stage('backbone'):
            p = self._features(img)[0]
        with PROF.stage('search'):
            d = self.bank.nn_distance(p)
            self._last_map = d
        with PROF.stage('patch_score'):
            s = aggregate(d, self.agg)
        return float(s), (_upsample(d, self.ex.grid_size, img.shape[-2:]) if return_heatmap else None)

    def bank_stats(self):
        st = self.bank.stats()
        st['coreset_fraction'] = self.bank.n / max(self.total, 1)
        return st


class SimpleNet(Method):
    """
    Liu et al., CVPR 2023 (re-implementation; verify against the official
    code). Linear adaptor; Gaussian noise on adapted features; 2-layer MLP
    discriminator; truncated L1 loss; anomaly = -D(adapted).
    """
    training_free = False

    def __init__(self, ex, cfg: BenchConfig, device, name: str, seed: int = 0):
        torch.manual_seed(seed)
        self.ex, self.cfg, self.device, self.name = ex, cfg, device, name
        D, H = ex.feat_dim, cfg.simplenet_hidden
        self.adaptor = nn.Linear(D, D).to(device)
        self.dsc = nn.Sequential(nn.Linear(D, H), nn.BatchNorm1d(H), nn.LeakyReLU(0.2),
                                 nn.Linear(H, 1)).to(device)

    def fit(self, ctx: Context, train_idx=None):
        cfg, D, loader = self.cfg, self.ex.feat_dim, ctx.train_loader(train_idx)
        s1 = torch.zeros(D, dtype=torch.float64, device=self.device)
        s2 = torch.zeros_like(s1)
        n = 0
        with torch.no_grad():
            for b in tqdm(loader, desc=f'[{self.name}] feature stats', leave=False):
                f = self.ex(b['img']).reshape(-1, D).double()
                s1 += f.sum(0)
                s2 += (f * f).sum(0)
                n += f.shape[0]
        mu = s1 / n
        self.fmu = mu.float()
        self.fsd = (s2 / n - mu ** 2).clamp_min(1e-8).sqrt().float()
        opt_a = torch.optim.Adam(self.adaptor.parameters(), lr=cfg.simplenet_adaptor_lr)
        opt_d = torch.optim.Adam(self.dsc.parameters(), lr=cfg.simplenet_lr, weight_decay=1e-5)
        th = cfg.simplenet_th
        self.adaptor.train()
        self.dsc.train()
        for ep in range(cfg.simplenet_epochs):
            tot, nb = 0.0, 0
            for b in tqdm(loader, desc=f'[{self.name}] epoch {ep + 1}', leave=False):
                with torch.no_grad():
                    f = (self.ex(b['img']).reshape(-1, D) - self.fmu) / self.fsd
                z = self.adaptor(f)
                fake = z + torch.randn_like(z) * cfg.simplenet_noise_std
                st = self.dsc(z).squeeze(-1)
                sf = self.dsc(fake).squeeze(-1)
                loss = F.relu(th - st).mean() + F.relu(th + sf).mean()
                opt_a.zero_grad(set_to_none=True)
                opt_d.zero_grad(set_to_none=True)
                loss.backward()
                opt_a.step()
                opt_d.step()
                tot += float(loss)
                nb += 1
            print(f'  [{self.name}] epoch {ep + 1:3d}/{cfg.simplenet_epochs}  loss={tot / max(nb, 1):.5f}')
        self.adaptor.eval()
        self.dsc.eval()

    @torch.no_grad()
    def score(self, img, return_heatmap=False):
        with PROF.stage('backbone'):
            f = (self.ex(img)[0] - self.fmu) / self.fsd
        with PROF.stage('search'):
            a = -self.dsc(self.adaptor(f)).squeeze(-1)
            self._last_map = a
        with PROF.stage('patch_score'):
            s = a.max()
        return float(s), (_upsample(a, self.ex.grid_size, img.shape[-2:]) if return_heatmap else None)

    def trainable_params(self):
        return sum(p.numel() for m in (self.adaptor, self.dsc) for p in m.parameters())

    def bank_stats(self):
        mb = sum(t.numel() * t.element_size() for m in (self.adaptor, self.dsc)
                 for t in list(m.parameters()) + list(m.buffers())) / 2 ** 20
        return {'bank_kind': 'parametric', 'bank_patch_vectors': 0,
                'bank_global_vectors': 0, 'bank_dim': 0, 'bank_mb': mb}


class AnomalyDINO(Method):
    """
    Damm et al., WACV 2025: patch memory bank, 1-NN cosine, image score = mean
    of the top-q patch distances. Zero-shot PCA masking omitted. Its Gaussian
    smoothing affects visualisations only; pixel metrics use the raw map.
    """
    def __init__(self, ex, cfg: BenchConfig, device, name: str):
        self.ex, self.cfg, self.device, self.name = ex, cfg, device, name
        self.bank = None

    @torch.no_grad()
    def fit(self, ctx: Context, train_idx=None):
        rots, chunks = (0,) + tuple(self.cfg.ad_rotations), []
        for b in tqdm(ctx.train_loader(train_idx), desc=f'[{self.name}] fit', leave=False):
            for r in rots:
                x = torch.rot90(b['img'], r, (-2, -1)) if r else b['img']
                chunks.append(self.ex(x).reshape(-1, self.ex.feat_dim).half().cpu())
        bank = torch.cat(chunks)
        del chunks
        cap = self.cfg.ad_max_bank
        if cap and bank.shape[0] > cap:
            sel = np.sort(np.random.default_rng(42).choice(bank.shape[0], cap, replace=False))
            bank = bank[torch.from_numpy(sel)]
        self.bank = GpuNNBank(bank, 'cosine', self.device, self.cfg, self.name)

    @torch.no_grad()
    def score(self, img, return_heatmap=False):
        with PROF.stage('backbone'):
            p = self.ex(img)[0]
        with PROF.stage('search'):
            d = self.bank.nn_distance(p)
            self._last_map = d
        with PROF.stage('patch_score'):
            q = self.cfg.ad_tail_q
            s = d.max() if q <= 0 else d.mean() if q >= 1 else aggregate(d, Agg('topk_mean', q))
        hm = None
        if return_heatmap:
            hm = _upsample(d, self.ex.grid_size, img.shape[-2:]).astype(np.float32)
            sg = float(self.cfg.ad_smooth_sigma)
            if sg > 0:
                hm = cv2.GaussianBlur(hm, (0, 0), sigmaX=sg, sigmaY=sg, borderType=cv2.BORDER_REFLECT)
        return float(s), hm

    def bank_stats(self):
        return self.bank.stats()


class SPADE(Method):
    """
    Cohen & Hoshen, 2020, on one DINOv2 level. 'image': mean distance to the
    k_image nearest global descriptors. 'patch': patch 1-NN over the
    k_retrieve nearest frames, reduced by `agg`.
    """
    def __init__(self, teacher, fspec, cfg: BenchConfig, device, name: str,
                 mode=None, store_patches=None):
        self.t, self.fspec, self.cfg, self.device, self.name = teacher, fspec, cfg, device, name
        self.mode = mode or cfg.spade_mode
        self.k_image, self.k_retrieve = cfg.spade_k_image, cfg.spade_k_retrieve
        self.cosine, self.agg = cfg.spade_metric == 'cosine', Agg(cfg.spade_patch_agg)
        self.store = (self.mode == 'patch') if store_patches is None else bool(store_patches)
        self.G = self.pb = self.psq = None
        self.pb_gpu, self.n = False, 0

    def _desc(self, x):
        with amp_ctx(self.fspec, self.device):
            c, p = self.t(x.to(self.device, non_blocking=True))
        pt = p[-1]
        g = c[-1] if self.cfg.spade_global == 'cls' else pt.mean(1)
        return g.float(), pt

    @torch.no_grad()
    def fit(self, ctx: Context, train_idx=None):
        gl, pl = [], []
        for b in tqdm(ctx.train_loader(train_idx), desc=f'[{self.name}] fit', leave=False):
            g, p = self._desc(b['img'])
            gl.append(g.cpu())
            if self.store:
                pl.append(p.half().cpu())
        G = torch.cat(gl)
        self.n = int(G.shape[0])
        self.G = (F.normalize(G, dim=1) if self.cosine else G).to(self.device)
        if self.store:
            rows = _prep_rows(torch.cat(pl), _DT[self.cfg.bank_dtype], self.cosine)
            self.pb, self.pb_gpu = _place_bank(rows, self.device, self.cfg.gpu_bank_budget_mb, self.name)
            self.psq = None if self.cosine else _row_sq(self.pb, self.device)

    @torch.no_grad()
    def score(self, img, return_heatmap=False):
        need_patch = self.mode == 'patch' or (
            (return_heatmap or getattr(self, '_want_map', False)) and self.pb is not None)
        if self.mode == 'patch' and self.pb is None:
            raise RuntimeError('SPADE patch mode needs store_patches=True')
        with PROF.stage('backbone'):
            g, p = self._desc(img)
        with PROF.stage('search'):
            q = g[0]
            dist = 1.0 - self.G @ F.normalize(q, dim=0) if self.cosine else torch.cdist(q[None], self.G)[0]
            k = self.k_image if self.mode == 'image' else 0
            if need_patch:
                k = max(k, self.k_retrieve)
            k = max(1, min(k, self.n))
            vals, inds = torch.topk(dist, k, largest=False)
            img_s = vals[:min(self.k_image, k)].mean() if self.mode == 'image' else None
        d, patch_s = None, None
        if need_patch:
            with PROF.stage('patch_score'):
                kr = min(self.k_retrieve, self.n)
                sel = None if kr >= self.n else inds[:kr]
                d = nn_over_frames(p[0], self.pb, sel, self.device, _DT[self.cfg.nn_compute_dtype],
                                   self.cfg.nn_chunk_vectors, self.cosine, self.psq, self.pb_gpu)
                patch_s = aggregate(d, self.agg) if self.mode == 'patch' else None
        s = img_s if self.mode == 'image' else patch_s
        hm = _upsample(d, (self.fspec.grid,) * 2, img.shape[-2:]) if (return_heatmap and d is not None) else None
        self._last_map = d
        return float(s), hm

    def bank_stats(self):
        P = int(self.pb.shape[1]) if self.pb is not None else 0
        mb = self.G.numel() * 4 / 2 ** 20
        if self.pb is not None:
            mb += self.pb.numel() * self.pb.element_size() / 2 ** 20
        return {'bank_kind': 'exemplar', 'bank_patch_vectors': self.n * P,
                'bank_global_vectors': self.n, 'bank_dim': self.t.embed_dim, 'bank_mb': mb}


def build_method(name: str, ctx: Context, cfg: BenchConfig, spec: MethodSpec, **kw) -> Method:
    t, fs, dev = ctx.teacher, ctx.fspec, ctx.device
    ex = lambda layers: DinoPatches(t, layers, fs, dev)
    if name == 'proposed':
        cls_ = StreamingProposedMethod if getattr(cfg, 'no_cache', False) else ProposedMethod
        return cls_(spec, ctx, **kw)
    if name == 'padim':
        return PaDiM(ex(cfg.padim_layers), cfg, 'PaDiM (DINOv2)')
    if name == 'patchcore':
        return PatchCore(ex(cfg.patchcore_layers), cfg, dev, 'PatchCore (DINOv2)', **kw)
    if name == 'simplenet':
        return SimpleNet(ex(cfg.simplenet_layers), cfg, dev, 'SimpleNet (DINOv2)', **kw)
    if name == 'anomalydino':
        return AnomalyDINO(ex(cfg.ad_layers), cfg, dev, 'AnomalyDINO (DINOv2)')
    if name == 'spade':
        return SPADE(t, fs, cfg, dev, 'SPADE (DINOv2)', **kw)
    raise ValueError(f'unknown method {name!r}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 · EVALUATION, LATENCY HARNESS, RESULT FILES
# ══════════════════════════════════════════════════════════════════════════════

_LAT: Dict[tuple, torch.Tensor] = {}


def latency_frames(ctx: Context, n_req: int) -> torch.Tensor:
    """Fixed, evenly spaced test frames, decoded once and held in host memory."""
    files = [f for sc in ctx.scenes for f in sc.files]
    key = (ctx.fspec.key(), len(files), files[0], n_req)
    if key not in _LAT:
        _LAT.clear()
        idx = np.unique(np.linspace(0, len(files) - 1, min(n_req, len(files))).round().astype(int))
        ds = FrameDataset([files[i] for i in idx], ctx.fspec)
        x = torch.stack([ds[i]['img'] for i in range(len(ds))])
        _LAT[key] = x.pin_memory() if torch.cuda.is_available() else x
    return _LAT[key]


@torch.no_grad()
def measure_latency(score_one, frames: torch.Tensor, device, cfg: BenchConfig) -> dict:
    """Batch 1, untimed warm-up; pass 1 end-to-end, pass 2 stage breakdown."""
    cuda = device.type == 'cuda'
    sync = (lambda: torch.cuda.synchronize(device)) if cuda else (lambda: None)
    n = int(frames.shape[0])

    def load(i):
        x = frames[i % n:i % n + 1].to(device, non_blocking=True)
        sync()
        return x

    set_profiler(device, False)
    for i in range(cfg.latency_warmup):
        score_one(load(i))
    sync()
    times = np.empty(n)
    for i in range(n):
        x = load(i)
        t0 = time.perf_counter()
        score_one(x)
        sync()
        times[i] = (time.perf_counter() - t0) * 1e3
    out = {'latency_median_ms': float(np.median(times)),
           'latency_p95_ms': float(np.percentile(times, 95)),
           'latency_mean_ms': float(times.mean()),
           'latency_std_ms': float(times.std(ddof=1)) if n > 1 else 0.0,
           'latency_n_frames': n, 'latency_warmup': int(cfg.latency_warmup)}
    if cfg.profile_stages:
        prof = set_profiler(device, True)
        for i in range(n):
            x = load(i)
            prof.begin()
            score_one(x)
            prof.end()
        out.update(prof.summary())
        set_profiler(device, False)
    return out


def evaluate_method(method: Method, ctx: Context, cfg: BenchConfig, label=None,
                    fit: bool = True, fit_idx=None) -> Tuple[dict, dict]:
    fit_s = float('nan')
    if fit:
        t0 = time.perf_counter()
        method.fit(ctx, fit_idx)
        fit_s = time.perf_counter() - t0
    set_profiler(ctx.device, False)
    ctx.pix = PixelEval.make(ctx) if cfg.pixel_eval else None
    try:
        res = method.score_frames(ctx)
    finally:
        pix, ctx.pix = ctx.pix, None
    pv = per_video_auroc(ctx.scenes, res['scores'])
    out = {'method': label or method.name, 'dataset': _dataset_tag(cfg),
           **frame_metrics(ctx.labels, res['scores']),
           'per_video_auroc_mean': float(np.mean(list(pv.values()))) if pv else float('nan'),
           'per_video': pv, 'fit_time_s': fit_s, 'backbone': method.backbone,
           'training_free': method.training_free, 'trainable_params': method.trainable_params(),
           'image_size': ctx.fspec.image_size, 'fkey': ctx.fspec.key(),
           'frozen_config': cfg.frozen_config, **method.bank_stats()}
    if pix is not None:
        out.update(pix.metrics())
        del pix
    if cfg.measure_latency:
        out.update(measure_latency(method.score_one, latency_frames(ctx, cfg.latency_frames),
                                   ctx.device, cfg))
    return out, res


def save_run(save_path: str, sub: str, row: dict, scores, ctx: Context):
    """runs/<sub>/<slug>.json (all metrics) + <slug>_scores.npz (per-frame scores)."""
    d = os.path.join(save_path, 'runs', sub)
    os.makedirs(d, exist_ok=True)
    slug = _safe(f"{row.get('ablation_id') or sub}_{row['method']}"
                 + (f"_s{row['seed']}" if row.get('seed') is not None else ''))
    with open(os.path.join(d, slug + '.json'), 'w') as fh:
        json.dump(row, fh, indent=2, default=str)
    np.savez_compressed(os.path.join(d, slug + '_scores.npz'), scores=np.asarray(scores),
                        labels=ctx.labels, videos=np.array([s.name for s in ctx.scenes]),
                        lengths=np.array([len(s.files) for s in ctx.scenes]))


def write_env(save_path: str, device, fspec: FeatureSpec, cfg: Optional[BenchConfig] = None):
    env = {'code_version': CODE_VERSION, 'git': _git_commit(), 'python': platform.python_version(),
           'torch': torch.__version__, 'timm': timm.__version__, 'cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
           'feature_spec': asdict(fspec), 'bench_config': asdict(cfg) if cfg is not None else None,
           'protocol': {'fail_fpr': FAIL_FPR, 'fail_gap': FAIL_GAP, 'a4_repeats': A4_REPEATS,
                        'train_stride': cfg.train_stride if cfg else None},
           'timestamp': datetime.datetime.now().isoformat()}
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'env.json'), 'w') as fh:
        json.dump(env, fh, indent=2, default=str)


def _print_row(r: dict):
    print(f'\n  ► {r.get("method")}  AUROC={_f(r.get("auroc"))}  AP={_f(r.get("ap"))}  '
          f'EER={_f(r.get("eer"))}  TPR@5%={_f(r.get("tpr_at_fpr05"))}  '
          f'pv-AUROC={_f(r.get("per_video_auroc_mean"))}  pAUROC={_f(r.get("pixel_auroc"))}  '
          f'pAP={_f(r.get("pixel_ap"))}  AUPRO={_f(r.get("aupro"))}  '
          f'median={_f(r.get("latency_median_ms"), 1)} ms  p95={_f(r.get("latency_p95_ms"), 1)} ms  '
          f'patch_vecs={int(r.get("bank_patch_vectors", 0)):,}  bank={_f(r.get("bank_mb"), 1)} MB')


_CSV_COLS = ['method', 'dataset', 'seed', 'ablation_id', 'backbone', 'training_free', 'auroc', 'ap',
             'eer', 'tpr_at_fpr05', 'per_video_auroc_mean', 'pixel_auroc', 'pixel_ap', 'aupro',
             'pixel_eval_hw', 'pixel_frames', 'pixel_regions', 'pixel_anom_fraction',
             'pixel_label_mask_agreement', 'pixel_excluded', 'n_frames', 'n_anomalous',
             'latency_median_ms', 'latency_p95_ms', 'latency_mean_ms', 'latency_std_ms',
             'latency_n_frames', 'backbone_ms', 'search_ms', 'patch_score_ms', 'other_ms',
             'profiled_frame_ms', 'bank_kind', 'bank_patch_vectors', 'bank_global_vectors',
             'bank_dim', 'bank_mb', 'bank_on_gpu', 'fit_time_s', 'trainable_params', 'image_size',
             'fkey', 'frozen_config', 'sweep', 'N_effective', 'K', 'resolution', 'P', 'method_base']


def print_results(rows: List[dict], title: str, csv_path: Optional[str] = None):
    if not rows:
        return
    marks = {'gaussian': '†', 'parametric': '‡'}
    table = [[r.get('method') + (f' s{r["seed"]}' if r.get('seed') is not None else ''),
              _f(r.get('auroc')), _f(r.get('ap')), _f(r.get('eer')), _f(r.get('tpr_at_fpr05')),
              _f(r.get('per_video_auroc_mean')), _f(r.get('pixel_auroc')), _f(r.get('pixel_ap')),
              _f(r.get('aupro')), _f(r.get('latency_median_ms'), 2), _f(r.get('latency_p95_ms'), 2),
              _f(r.get('backbone_ms'), 2), _f(r.get('search_ms'), 2), _f(r.get('patch_score_ms'), 2),
              _f(r.get('other_ms'), 2),
              f'{int(r.get("bank_patch_vectors", 0)):,}{marks.get(r.get("bank_kind"), "")}',
              _f(r.get('bank_mb'), 1)] for r in rows]
    print('\n' + '=' * 100 + f'\n{title}\n' + '=' * 100)
    print_table(table, ['Method', 'AUROC', 'AP', 'EER', 'TPR@5%', 'pv-AUROC', 'pAUROC', 'pAP',
                        'AUPRO', 'Median ms', 'p95 ms', 'Backbone', 'Search', 'Patch', 'Other',
                        'Patch vecs', 'MB'])
    print('Median/p95: end-to-end, batch 1, synchronised, fixed frames, warm-up excluded. '
          'Stage columns: separate synchronised pass.  † PaDiM: per-location Gaussians. '
          '‡ SimpleNet: MB = network parameters.  Pixel metrics: n/a without pixel GT.')
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=_CSV_COLS, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        print(f'[saved] {csv_path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 · ATTRIBUTION GRID (E2) AND SCALING SWEEPS (E6)
# ══════════════════════════════════════════════════════════════════════════════

# (id, a, b, what a - b isolates). Positive Δ favours row a. CHK is a code check.
GRID_CONTRASTS = [
    ('G1', 'A1', 'A2', 'Aggregation (Proposed): selected rule − max'),
    ('G2', 'B2', 'B1', 'Aggregation (PatchCore): selected rule − max'),
    ('G3', 'A1', 'A3', 'Retrieval: K nearest frames − all bank frames'),
    ('G4', 'A1', 'A6', 'Retrieval quality: K nearest − K random frames'),
    ('G5', 'A1', 'A5', 'Frame bank: FPS subset − all training frames'),
    ('G6', 'A5', 'D3', 'Similarity: cosine on unit tokens − L2 (SPADE-patch)'),
    ('G7', 'A1', 'C2', 'Proposed − PatchCore at matched memory, same aggregation'),
    ('G8', 'A3', 'C2', 'Same budget, no retrieval: frame FPS bank − patch coreset'),
    ('G9', 'C1', 'B1', 'PatchCore: memory-matched − configured coreset'),
    ('G10', 'A1', 'D1', 'Proposed − SPADE as published (image kNN, L2)'),
    ('CHK', 'A5', 'D4', 'Check: Proposed (all frames) − SPADE-patch cosine (≈ 0)'),
]


def _diff(ra: dict, rb: dict, k: str) -> Optional[float]:
    a, b = _num(ra.get(k)), _num(rb.get(k))
    return None if a is None or b is None else a - b


def run_bench_grid(ctx: Context, cfg: BenchConfig, spec: MethodSpec, n_boot: int = 1000):
    """
    Where does the gain come from? Same frozen DINOv2, same 1-NN kernel, same
    bank dtype and latency harness in every row. Residual confound: PatchCore's
    3x3 local aggregation. With selected_layers=(11,) row A5 and row D4 compute
    the same scores (CHK), so the method differs from SPADE-patch only through
    the FPS bank (G5), the similarity (G6) and the aggregation rule (G1).
    """
    rule = Agg(spec.agg, spec.topk_ratio, spec.mad_sigma)
    rn, K = _agg_name(rule), max(spec.knn_k, 1)
    rows, scores = [], {}

    def ev(aid, m, label, fit=True):
        r, res = evaluate_method(m, ctx, cfg, label=f'{aid} {label}', fit=fit)
        r['ablation_id'] = aid
        rows.append(r)
        scores[aid] = res['scores']
        _print_row(r)
        if cfg.save_runs:
            save_run(cfg.save_path, 'grid', r, res['scores'], ctx)

    for aid, kw, a, lbl in (('A1', {}, rule, f'Proposed K={K} · {rn}'),
                            ('A2', {}, Agg('max'), f'Proposed K={K} · max'),
                            ('A3', {'knn_k': 0}, rule, f'Proposed K=all · {rn}'),
                            ('A4', {'knn_k': 0}, Agg('max'), 'Proposed K=all · max'),
                            ('A5', {'fps_ratio': None}, rule, f'Proposed all frames K={K} · {rn}'),
                            ('A6', {'key_source': 'random'}, rule, f'Proposed random K={K} · {rn}')):
        s = replace(spec, agg=a.agg, topk_ratio=a.topk_ratio, mad_sigma=a.mad_sigma, **kw)
        ev(aid, ProposedMethod(s, ctx), lbl)
        _free(ctx.device)
    mm = int(rows[0]['bank_patch_vectors'])
    for fam, n in (('B', None), ('C', mm)):
        lbl = 'PatchCore' if n is None else 'PatchCore-MM'
        pc = build_method('patchcore', ctx, cfg, spec, coreset_n=n, agg=Agg('max'))
        ev(f'{fam}1', pc, f'{lbl} · max')
        pc.agg = rule
        ev(f'{fam}2', pc, f'{lbl} · {rn}', fit=False)
        del pc
        _free(ctx.device)
    sp = build_method('spade', ctx, cfg, spec, mode='image', store_patches=True)
    ev('D1', sp, f'SPADE image kNN · K={sp.k_image}')
    sp.mode, sp.k_retrieve, sp.agg = 'patch', K, Agg('max')
    ev('D2', sp, f'SPADE-patch L2 K={K} · max', fit=False)
    sp.agg = rule
    ev('D3', sp, f'SPADE-patch L2 K={K} · {rn}', fit=False)
    del sp
    _free(ctx.device)
    sp4 = SPADE(ctx.teacher, ctx.fspec, replace(cfg, spade_metric='cosine'), ctx.device,
                'SPADE-patch cosine', mode='patch', store_patches=True)
    sp4.k_retrieve, sp4.agg = K, rule
    ev('D4', sp4, f'SPADE-patch cosine K={K} · {rn}')
    del sp4
    _free(ctx.device)

    print_results(rows, f'TABLE — Attribution grid ({ctx.fspec.image_size}px, K={K}, rule={rn})',
                  os.path.join(cfg.save_path, 'ablation_grid.csv'))
    L = auto_block_len(ctx.scenes, scores['A1'])
    cons = []
    for cid, a, b, what in GRID_CONTRASTS:
        if a not in scores or b not in scores:
            continue
        bs = block_bootstrap(ctx.scenes, ctx.labels, scores[b], scores[a], L, n_boot)
        ra = next(r for r in rows if r['ablation_id'] == a)
        rb = next(r for r in rows if r['ablation_id'] == b)
        cons.append({'id': cid, 'a': a, 'b': b, 'a_label': ra['method'], 'b_label': rb['method'],
                     'isolates': what, **bs, 'd_ap': _diff(ra, rb, 'ap'),
                     'd_aupro': _diff(ra, rb, 'aupro'), 'd_ms': _diff(ra, rb, 'latency_median_ms')})
    adj = holm({c['id']: c['p'] for c in cons if c['id'] != 'CHK'})
    for c in cons:
        c['p_holm'] = adj.get(c['id'])
    table = [[c['id'], c['isolates'], f'{c["delta"]:+.4f} [{c["ci_low"]:+.4f},{c["ci_high"]:+.4f}]',
              _f(c['p_holm'], 3), _f(c['d_ap']), _f(c['d_aupro']), _f(c['d_ms'], 2)] for c in cons]
    print(f'\nCONTRASTS (Δ = a − b; paired moving-block bootstrap, block = {L} frames; Holm)')
    print_table(table, ['id', 'isolates', 'ΔAUROC [95% CI]', 'p (Holm)', 'ΔAP', 'ΔAUPRO', 'Δms'])
    chk_valid = tuple(spec.selected_layers) == (11,) and spec.key_source == 'concat_cls'
    with open(os.path.join(cfg.save_path, 'grid_contrasts.json'), 'w') as fh:
        json.dump({'dataset': _dataset_tag(cfg), 'code_version': CODE_VERSION, 'spec': asdict(spec),
                   'frozen_config': cfg.frozen_config, 'block_len': L, 'n_boot': n_boot,
                   'check_valid': chk_valid, 'contrasts': cons}, fh, indent=2, default=str)
    return rows


def run_sweeps(ctx: Context, cfg: BenchConfig, spec: MethodSpec):
    """N: bank size vs cost; K: stage-2 cost is linear in K; resolution: P = (res/14)^2."""
    out_dir = os.path.join(cfg.save_path, 'scaling')
    sweeps = {'N': [], 'K': [], 'resolution': []}
    viz, cfg.viz_dir = cfg.viz_dir, None
    pix_on, cfg.pixel_eval = cfg.pixel_eval, cfg.pixel_eval and cfg.pixel_eval_sweeps
    perm = np.random.default_rng(42).permutation(len(ctx.train_files))
    Ns = sorted({max(1, int(round(v * len(perm)))) if v <= 1 else min(int(v), len(perm))
                 for v in cfg.sweep_N})                     # no duplicate points after the stride
    for N in Ns:
        idx = sorted(int(i) for i in perm[:N])
        for name in cfg.methods:
            m = build_method(name, ctx, cfg, spec)
            r, _ = evaluate_method(m, ctx, cfg, label=f'{name} (N={len(idx)})', fit_idx=idx)
            r.update(sweep='N', N_effective=len(idx), method_base=name)
            sweeps['N'].append(r)
            _print_row(r)
            del m
            _free(ctx.device)
    for name in cfg.methods:
        if name == 'proposed':
            for K in cfg.sweep_K:
                m = ProposedMethod(replace(spec, knn_k=K), ctx)
                r, _ = evaluate_method(m, ctx, cfg, label=f'proposed (K={K})')
                r.update(sweep='K', K=K, method_base=name)
                sweeps['K'].append(r)
                _print_row(r)
        elif name == 'spade':
            m = build_method(name, ctx, cfg, spec)
            m.fit(ctx)
            for K in cfg.sweep_K:
                if m.mode == 'image':
                    m.k_image = K
                else:
                    m.k_retrieve = K
                r, _ = evaluate_method(m, ctx, cfg, label=f'spade (K={K})', fit=False)
                r.update(sweep='K', K=K, method_base=name)
                sweeps['K'].append(r)
                _print_row(r)
        else:
            print(f'  [note] {name} has no K; see the main table')
        _free(ctx.device)
    for res in cfg.sweep_res:
        if res % 14:
            print(f'  [skip] {res} px is not a multiple of the patch size 14')
            continue
        c2 = Context(cfg, replace(ctx.fspec, image_size=res), ctx.teacher, ctx.device, ctx.patch_layers)
        for name in cfg.methods:
            m = build_method(name, c2, cfg, spec)
            r, _ = evaluate_method(m, c2, cfg, label=f'{name} ({res}px)')
            r.update(sweep='resolution', resolution=res, P=(res // 14) ** 2, method_base=name)
            sweeps['resolution'].append(r)
            _print_row(r)
            del m
            _free(ctx.device)
    cfg.viz_dir, cfg.pixel_eval = viz, pix_on
    for key, rows in sweeps.items():
        print_results(rows, f'TABLE — {key} sweep', os.path.join(out_dir, f'sweep_{key}.csv'))
    return sweeps


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 · ABLATION STUDY OF THE PROPOSED METHOD (cached features only)
# ══════════════════════════════════════════════════════════════════════════════

A1_CONDITIONS = {'early_L0-2': (0, 1, 2), 'middle_L5-7': (5, 6, 7), 'last_L11': (11,),
                 'last_L10-11': (10, 11), 'last_L9-11': (9, 10, 11),
                 'last_L7-11': (7, 8, 9, 10, 11), 'all_L0-11': tuple(range(12))}
A2_DIMS = (None, 16, 32, 64, 128, 256, 512)
A4_STRATEGIES = ('fps', 'random', 'stride', 'kmeans')
A4_RATIOS = (0.05, 0.10, 0.25, 0.50, 1.00)
A5_METRICS = ('cosine', 'l2', 'l2_unit')
A5_K = (1, 3, 5, 10, 20)
A5_KEYS = ('concat_cls', 'last_cls', 'layer5_cls', 'gap')
A6_RULES = {'mean': Agg('mean'), 'max': Agg('max'),
            'mad_1σ': Agg('robust_mad', mad_sigma=1.0), 'mad_2σ': Agg('robust_mad', mad_sigma=2.0),
            'mad_3σ': Agg('robust_mad', mad_sigma=3.0),
            'top_0.005': Agg('topk_mean', 0.005), 'top_0.01': Agg('topk_mean', 0.01),
            'top_0.02': Agg('topk_mean', 0.02), 'top_0.03': Agg('topk_mean', 0.03),
            'top_0.05': Agg('topk_mean', 0.05), 'top_0.10': Agg('topk_mean', 0.10)}
ABLATION_PATCH_LAYERS = tuple(sorted({c[-1] for c in A1_CONDITIONS.values()}))


def subsample_idx(keys: torch.Tensor, strategy: str, n: int, seed: int) -> List[int]:
    N = keys.shape[0]
    if n >= N:
        return list(range(N))
    if strategy == 'fps':
        return fps_select(keys, n, seed)
    if strategy == 'random':
        return sorted(int(i) for i in np.random.default_rng(seed).choice(N, n, replace=False))
    if strategy == 'stride':
        return sorted({int(i) for i in np.linspace(0, N - 1, n).round()})
    if strategy == 'kmeans':                    # medoids: real frames nearest the centroids
        from sklearn.cluster import MiniBatchKMeans
        X = F.normalize(keys.float(), dim=1).numpy()
        C = MiniBatchKMeans(n_clusters=n, random_state=seed, n_init=3,
                            batch_size=max(1024, n)).fit(X).cluster_centers_
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        return sorted(int(i) for i in np.unique(np.argmax(X @ C.T, axis=0)))
    raise ValueError(strategy)


class Ablation:
    """
    Every condition is dataclasses.replace() of the frozen MethodSpec, scored by
    the same ProposedScorer on the same cached features. Inference: paired
    moving-block bootstrap vs the reference, Holm-corrected within each table.
    """
    def __init__(self, ctx: Context, cfg: BenchConfig, spec: MethodSpec, out_dir: str,
                 n_boot: int = 1000, block_len: int = 0, a4_repeats: int = 5):
        self.ctx, self.cfg, self.spec, self.out = ctx, cfg, spec, out_dir
        self.n_boot, self.a4_repeats = n_boot, a4_repeats
        self.labels = ctx.labels
        os.makedirs(out_dir, exist_ok=True)
        print(f'\n[ablate] reference = frozen configuration: {spec}')
        self.ref = self.run(spec)
        self.L = block_len or auto_block_len(ctx.scenes, self.ref['scores'])
        print(f'[ablate] reference AUROC={_f(self.ref["auroc"])}  bank={self.ref["n_bank"]} frames  '
              f'bootstrap block = {self.L} frames')
        self._scores: Dict[str, np.ndarray] = {}

    def run(self, spec: MethodSpec, bank_idx=None, rules=None) -> dict:
        m = ProposedMethod(spec, self.ctx, bank_idx=bank_idx)
        m.fit(self.ctx)
        self.ctx.pix = PixelEval.make(self.ctx)
        try:
            r = m.score_frames(self.ctx, rules=rules)
        finally:
            pix, self.ctx.pix = self.ctx.pix, None
        pm = pix.metrics() if pix is not None else {}
        met = frame_metrics(self.labels, r['scores'])
        c = r['concat']
        pv = per_video_auroc(self.ctx.scenes, r['scores'])
        out = {'scores': r['scores'], 'rules': r['rules'], 'bank_idx': list(m.scorer.idx),
               'n_bank': len(m.scorer.idx), 'key_dim': m.scorer.key_dim_eff, 'mkey': spec.key(),
               'auroc': met.get('auroc'), 'ap': met.get('ap'), 'eer': met.get('eer'),
               'per_video': float(np.mean(list(pv.values()))) if pv else None,
               'concat_auroc': frame_metrics(self.labels, c).get('auroc') if np.isfinite(c).all() else None,
               'pixel_auroc': pm.get('pixel_auroc'), 'aupro': pm.get('aupro')}
        del m
        _free(self.ctx.device)
        return out

    def _row(self, name: str, detail: str, r: dict, is_ref: bool) -> dict:
        self._scores[name] = r['scores']
        return {'name': name, 'detail': detail, 'auroc': r['auroc'], 'ap': r['ap'],
                'eer': r['eer'], 'per_video': r['per_video'], 'concat_auroc': r['concat_auroc'],
                'n_bank': r['n_bank'], 'key_dim': r['key_dim'], 'mkey': r['mkey'],
                'is_ref': is_ref, 'pixel_auroc': r.get('pixel_auroc'), 'aupro': r.get('aupro'),
                'boot': None if is_ref else block_bootstrap(self.ctx.scenes, self.labels,
                                                            self.ref['scores'], r['scores'],
                                                            self.L, self.n_boot)}

    def cond(self, name: str, spec: MethodSpec, detail='', bank='ref') -> dict:
        """bank='ref': reuse the reference bank frames; None: the condition's own FPS."""
        is_ref = spec == self.spec and bank == 'ref'
        r = self.ref if is_ref else self.run(spec, bank_idx=self.ref['bank_idx'] if bank == 'ref' else bank)
        return self._row(name, detail(r) if callable(detail) else detail, r, is_ref)

    def finish(self, axis: str, title: str, rows: List[dict], notes=()):
        adj = holm({r['name']: r['boot']['p'] for r in rows if r.get('boot')})
        table = []
        for r in rows:
            b = r.get('boot')
            table.append([r['name'], r.get('detail', ''), _f(r.get('auroc')), _f(r.get('ap')),
                          _f(r.get('per_video')), _f(r.get('pixel_auroc')), _f(r.get('aupro')),
                          _f(r.get('concat_auroc')),
                          'ref' if r.get('is_ref') else
                          (f'{b["delta"]:+.4f} [{b["ci_low"]:+.4f},{b["ci_high"]:+.4f}]' if b else '—'),
                          'ref' if r.get('is_ref') else (_f(adj.get(r['name']), 3) if b else '—')])
        headers = ['Condition', 'Detail', 'AUROC', 'AP', 'pv-AUROC', 'pixAUROC', 'AUPRO',
                   'Concat (diag)', 'ΔAUROC vs ref [95% CI]', 'p (Holm)']
        notes = list(notes) + [f'Paired moving-block bootstrap within videos (block {self.L} frames, '
                               f'{self.n_boot} resamples); Holm correction within this table.',
                               'pixAUROC / AUPRO: point estimates; n/a without pixel GT.']
        print(f'\n── {axis}: {title} ' + '─' * max(0, 70 - len(axis) - len(title)))
        print_table(table, headers)
        for n in notes:
            print('  ' + n)
        with open(os.path.join(self.out, f'{axis}.json'), 'w') as fh:
            json.dump({'code_version': CODE_VERSION, 'axis': axis, 'title': title,
                       'reference_spec': asdict(self.spec), 'fkey': self.ctx.fspec.key(),
                       'headers': headers, 'table': table, 'notes': notes,
                       'rows': [{k: v for k, v in r.items()} for r in rows]}, fh, indent=2, default=str)
        np.savez_compressed(os.path.join(self.out, f'{axis}_scores.npz'),
                            labels=self.labels, **{_safe(k): v for k, v in self._scores.items()})
        self._scores = {}

    def a1(self, own_fps: bool = False):
        rows = []
        for name, layers in A1_CONDITIONS.items():
            s = replace(self.spec, selected_layers=layers, patch_layer=layers[-1])
            rows.append(self.cond(name, s, lambda r, l=layers: f'layers={list(l)} patchL={l[-1]} '
                                                               f'key_dim={r["key_dim"]}',
                                  bank=None if own_fps else 'ref'))
        self.finish('A1', 'Layer selection (retrieval key and patch layer)', rows,
                    ['Bank frames fixed to the reference FPS selection; rerun with --a1_own_fps.'
                     if not own_fps else 'Each condition selects its own FPS bank.'])

    def a2(self):
        rows = []
        for d in A2_DIMS:
            s = replace(self.spec, key_proj_dim=d)
            rows.append(self.cond('none' if d is None else f'svd_{d}', s,
                                  lambda r, d=d: f'req={d} eff={r["key_dim"]}'
                                  + (' CLAMPED' if d and r['key_dim'] < d else '')))
        self.finish('A2', 'Retrieval-key compression (uncentred SVD)', rows,
                    ['Only the retrieval keys are projected; the patch bank is unchanged.'])

    def a4(self):
        tr = self.ctx.features('train', [self.spec.patch_layer])
        keys = cat_cls([tr['cls'][l] for l in self.spec.selected_layers])
        N, rows = keys.shape[0], []
        ref_ratio = self.spec.fps_ratio
        for ratio in A4_RATIOS:
            for strat in (('fps',) if ratio >= 1.0 else A4_STRATEGIES):
                n = max(1, int(ratio * N))
                reps = 1 if (strat == 'stride' or ratio >= 1.0) else self.a4_repeats
                au, nb = [], []
                for r in range(reps):
                    seed = self.spec.fps_seed if r == 0 else SEEDS[r % len(SEEDS)]
                    idx = subsample_idx(keys, strat, n, seed)
                    same = idx == self.ref['bank_idx']
                    res = self.ref if same else self.run(self.spec, bank_idx=idx)
                    if strat == 'fps' and r == 0 and ref_ratio and abs(ratio - ref_ratio) < 1e-9:
                        print(f'  [check {"ok" if same else "FAILED"}] fps@{ratio} run 1 '
                              f'{"selects" if same else "does NOT select"} the reference bank')
                    au.append(res['auroc'])
                    nb.append(res['n_bank'])
                name = 'all@1.00' if ratio >= 1.0 else f'{strat}@{ratio:.2f}'
                sd = float(np.std(au)) if len(au) > 1 else float('nan')
                rows.append({'name': name, 'detail': f'n={int(np.mean(nb))} runs={reps} σ={_f(sd)}',
                             'auroc': float(np.mean(au)), 'ap': None, 'eer': None,
                             'per_video': None, 'concat_auroc': None, 'boot': None,
                             'is_ref': False, 'auroc_runs': au})
        self.finish('A4', 'Bank subsampling strategy × budget', rows,
                    ['AUROC = mean over runs; σ over runs is the relevant variability.'])

    def a5(self):
        rows = [self.cond(mt, replace(self.spec, key_metric=mt), f'retrieval metric={mt}')
                for mt in A5_METRICS]
        by = {r['name']: r for r in rows}
        if 'cosine' in by and 'l2_unit' in by:
            ok = abs((by['cosine']['auroc'] or 0) - (by['l2_unit']['auroc'] or 0)) < 1e-3
            print(f'  [check {"ok" if ok else "FAILED"}] l2_unit must reproduce cosine')
        self.finish('A5a', 'Retrieval metric (patch metric fixed to cosine)', rows)
        rows = [self.cond(f'K={k}', replace(self.spec, knn_k=k), f'K={k}') for k in A5_K]
        self.finish('A5b', 'Number of retrieved frames K', rows)
        rows = [self.cond(src, replace(self.spec, key_source=src), f'key={src}') for src in A5_KEYS]
        self.finish('A5c', 'Retrieval key source', rows)

    def a6(self):
        rules = dict(A6_RULES)
        ref_name = next((n for n, a in rules.items() if _same_rule(a, self.spec)), None)
        if ref_name is None:
            ref_name = 'proposed'
            rules[ref_name] = Agg(self.spec.agg, self.spec.topk_ratio, self.spec.mad_sigma)
        r = self.run(self.spec, bank_idx=self.ref['bank_idx'], rules=rules)
        same = np.array_equal(r['rules'][ref_name], self.ref['scores'])
        print(f'  [check {"ok" if same else "FAILED"}] rule {ref_name} reproduces the reference scores')
        rows = []
        for name, sc in r['rules'].items():
            rr = dict(r, scores=sc, **{k: v for k, v in frame_metrics(self.labels, sc).items()
                                       if k in ('auroc', 'ap', 'eer')})
            pv = per_video_auroc(self.ctx.scenes, sc)
            rr['per_video'] = float(np.mean(list(pv.values()))) if pv else None
            rows.append(self._row(name, _agg_name(rules[name]), rr, name == ref_name))
        self.finish('A6', 'Patch-map aggregation rule', rows,
                    ['All rules reduce the SAME maps, so pixel metrics are identical across rows.'])

    def a7(self):
        s = self.spec
        conds = [('reference', s, f'K={s.knn_k}, free matching'),
                 ('K=all', replace(s, knn_k=0), 'no retrieval: match against every bank frame'),
                 ('random K', replace(s, key_source='random'), 'K random bank frames per query'),
                 ('aligned, K', replace(s, position_aligned=True), 'patch p vs patch p of K frames'),
                 ('aligned, K=all', replace(s, position_aligned=True, knn_k=0), 'positional prior only')]
        rows = [self.cond(n, sp, d) for n, sp, d in conds]
        self.finish('A7', 'Retrieval mechanism and matching geometry', rows)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 · HYPERPARAMETER SELECTION ON THE DEVELOPMENT DATASET
# ══════════════════════════════════════════════════════════════════════════════
# Declared BEFORE any held-out dataset is scored. Lists are in preference order
# (cheapest first); the first entry within SELECT_TOL of the best AUROC wins.

SELECT_TOL = 0.010        # "declared a priori"
SELECT_PATCH_LAYER = 11   # A1 on ShanghaiTech (patch layer 11 best)

SELECT_PROPOSED = {
    'selected_layers': ((11,),),   # ((11,), (9, 10, 11), (7, 8, 9, 10, 11)) retrieval key only
    'knn_k': (5, 10, 20),                                         # A5b flat; K=3 dropped
    'fps_ratio': (0.05,),                                         # A4 flat, memory-motivated
}
SELECT_RULES = {                                                  # one pass scores all rules
    'top_0.01':  Agg('topk_mean', 0.01),     # ~14 of 1369 patches
    'top_0.005': Agg('topk_mean', 0.005),    # ~7 patches: small objects seen from altitude
    'top_0.03':  Agg('topk_mean', 0.03),     # ~41 patches (previous default)
    'max':       Agg('max'),
}
SELECT_BASELINES = {
    'patchcore':   [{'patchcore_ratio': 0.05}],
    'anomalydino': [{'ad_tail_q': q} for q in (0.01, 0.005, 0.02, 0.0)],
    'spade':       [{'spade_k_retrieve': k} for k in (10, 20, 50)],
    'simplenet':   [{'simplenet_noise_std': s} for s in (0.015, 0.005, 0.05)],
    'padim':       [{'padim_pca_dim': d} for d in (128, 64, 256)],
}
_METRIC_KEYS = ('auroc', 'ap', 'aupro', 'pixel_auroc', 'per_dataset_auroc')


def _pick(recs: List[dict]) -> int:
    ok = [r['auroc'] for r in recs if _num(r.get('auroc')) is not None]
    best = max(ok)
    return next(i for i, r in enumerate(recs)
                if _num(r.get('auroc')) is not None and r['auroc'] >= best - SELECT_TOL)


def _proposed_eval(ctx: Context, spec: MethodSpec, rules=None) -> dict:
    """One cached-feature pass: frame AUROC (+ per rule) and pixel metrics if GT exists."""
    m = ProposedMethod(spec, ctx)
    m.fit(ctx)
    ctx.pix = PixelEval.make(ctx)
    try:
        r = m.score_frames(ctx, rules=rules)
    finally:
        pix, ctx.pix = ctx.pix, None
    pm = pix.metrics() if pix is not None else {}
    out = {'auroc': frame_metrics(ctx.labels, r['scores']).get('auroc'),
           'aupro': _num(pm.get('aupro')), 'pixel_auroc': _num(pm.get('pixel_auroc'))}
    if rules:
        out['rules'] = {n: frame_metrics(ctx.labels, v) for n, v in r['rules'].items()}
    del m
    _free(ctx.device)
    return out


def _apply_config(path: str, cfg: BenchConfig, fspec: FeatureSpec, spec: MethodSpec) -> MethodSpec:
    with open(path) as fh:
        C = json.load(fh)
    if C.get('proposed'):
        spec = MethodSpec(**{k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in C['proposed'].items()})
    for k, v in C.get('baselines', {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        else:
            print(f'[config][WARN] unknown BenchConfig field {k!r} ignored')
    fs = C.get('feature_spec')
    if fs and _hash(fs) != _hash(asdict(fspec)):
        print('[config][WARN] FeatureSpec differs from the one used for selection: '
              f'{fs} vs {asdict(fspec)}')
    cfg.frozen_config = os.path.abspath(path)
    print(f'[config] frozen configuration loaded from {path} '
          f'(selected on {C.get("dev_dataset")}, git {C.get("git")})')
    return spec


def _select_records(name: str, ctx: Context, cfg: BenchConfig, spec: MethodSpec, device) -> List[dict]:
    """Every grid point of one method on ONE development dataset, in the declared order."""
    recs = []
    if name == 'proposed':
        for lay in SELECT_PROPOSED['selected_layers']:
            for K in SELECT_PROPOSED['knn_k']:
                for fr in SELECT_PROPOSED['fps_ratio']:
                    s0 = replace(spec, selected_layers=tuple(lay), patch_layer=SELECT_PATCH_LAYER,
                                 knn_k=K, fps_ratio=fr)
                    r = _proposed_eval(ctx, s0, SELECT_RULES)
                    for rn in SELECT_RULES:
                        recs.append({'selected_layers': list(lay), 'knn_k': K, 'fps_ratio': fr,
                                     'rule': rn, 'auroc': r['rules'][rn].get('auroc'),
                                     'ap': r['rules'][rn].get('ap'),
                                     'aupro': r['aupro'], 'pixel_auroc': r['pixel_auroc']})
                    print(f'  [select] {_dataset_tag(cfg)} layers={list(lay)} K={K} ρ={fr}: ' +
                          ', '.join(f'{rn}={_f(r["rules"][rn].get("auroc"))}' for rn in SELECT_RULES))
        return recs
    for ov in SELECT_BASELINES.get(name, [{}]):
        c2 = replace(cfg, **ov)
        m = build_method(name, ctx, c2, spec)
        r, _ = evaluate_method(m, ctx, c2, label=f'{name} {ov}')
        recs.append({**ov, 'auroc': r.get('auroc'), 'ap': r.get('ap'),
                     'aupro': _num(r.get('aupro')), 'pixel_auroc': _num(r.get('pixel_auroc'))})
        print(f'  [select] {_dataset_tag(cfg)} {name} {ov}: AUROC={_f(r.get("auroc"))}')
        del m
        _free(device)
    return recs


def cmd_select(args):
    cfg, spec, fspec, device, teacher = setup(args, load_config=False)
    cfg.measure_latency, cfg.viz_dir = False, None
    paths = [args.data_path] + list(args.dev_extra)
    cfgs = [replace(cfg, data_path=p) for p in paths]
    ctxs = [Context(c, fspec, teacher, device, (SELECT_PATCH_LAYER,)) for c in cfgs]
    if any(c.mask_root for c in ctxs):
        selftest_pixel_metrics(cfg.pixel_max_fpr)
    path = args.config_out or os.path.join(cfg.save_path, 'selected_config.json')
    C = {}
    if os.path.exists(path):
        with open(path) as fh:
            C = json.load(fh)
    for k in ('results', 'selected_index', 'baselines'):
        C.setdefault(k, {})
    budget = {'proposed': len(SELECT_PROPOSED['selected_layers']) * len(SELECT_PROPOSED['knn_k'])
              * len(SELECT_PROPOSED['fps_ratio']) * len(SELECT_RULES),
              **{k: len(v) for k, v in SELECT_BASELINES.items()}}
    C.update({'code_version': CODE_VERSION, 'git': _git_commit(),
              'timestamp': datetime.datetime.now().isoformat(),
              'dev_dataset': ' + '.join(paths), 'dev_datasets': paths,
              'feature_spec': asdict(fspec), 'train_stride': cfg.train_stride,
              'protocol': {'criterion': 'mean frame AUROC over the development datasets (equal weight)',
                           'tolerance': SELECT_TOL,
                           'tie_break': 'first configuration in the declared order within the '
                                        'tolerance (K, then FPS ratio, ascending; rules and '
                                        'baseline grids in the listed order)',
                           'budget': budget,
                           'proposed_grid': {'selected_layers': [list(l) for l in SELECT_PROPOSED['selected_layers']],
                                             'patch_layer': SELECT_PATCH_LAYER,
                                             'knn_k': list(SELECT_PROPOSED['knn_k']),
                                             'fps_ratio': list(SELECT_PROPOSED['fps_ratio']),
                                             'rules': list(SELECT_RULES)},
                           'baseline_grids': SELECT_BASELINES,
                           'sht_subset': {'test': list(SHT_TEST), 'cams': sorted(SHT_CAMS),
                                          'exclude_train': sorted(SHT_EXCLUDE),
                                          'train_stride': SHT_TRAIN_STRIDE}}})
    for name in args.method:
        per = [_select_records(name, ctx, c, spec, device) for c, ctx in zip(cfgs, ctxs)]
        recs = []
        for i, base in enumerate(per[0]):                       # identical order on every dataset
            rec = {k: v for k, v in base.items() if k not in _METRIC_KEYS}
            for k in ('auroc', 'ap', 'aupro', 'pixel_auroc'):
                v = [x for x in (_num(p[i].get(k)) for p in per) if x is not None]
                rec[k] = float(np.mean(v)) if v else None
            rec['per_dataset_auroc'] = [p[i].get('auroc') for p in per]
            recs.append(rec)
        i = _pick(recs)
        if name == 'proposed':
            s, a = recs[i], SELECT_RULES[recs[i]['rule']]
            C['proposed'] = asdict(replace(spec, selected_layers=tuple(s['selected_layers']),
                                           patch_layer=SELECT_PATCH_LAYER, knn_k=s['knn_k'],
                                           fps_ratio=s['fps_ratio'], agg=a.agg,
                                           topk_ratio=a.topk_ratio, mad_sigma=a.mad_sigma))
        else:
            C['baselines'].update({k: v for k, v in recs[i].items() if k not in _METRIC_KEYS})
        C['results'][name], C['selected_index'][name] = recs, i
        print(f'[select] {name}: selected {recs[i]}')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(C, fh, indent=2, default=str)
    print(f'[saved] {path}\n[select] commit and tag this file BEFORE scoring any held-out dataset')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 · SENSITIVITY AROUND THE FROZEN CONFIGURATION (E3)
# ══════════════════════════════════════════════════════════════════════════════

SENS_K = (1, 3, 5, 10, 20, 40)
SENS_FPS = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0)
SENS_LAYERS = ((11,), (10, 11), (9, 10, 11), (7, 8, 9, 10, 11))
SENS_RULES = {'max': Agg('max'), 'mean': Agg('mean'),
              **{f'top_{q:g}': Agg('topk_mean', q) for q in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)}}


def cmd_sensitivity(args):
    cfg, spec, fspec, device, teacher = setup(args)
    cfg.viz_dir = None
    layers = set(SENS_LAYERS) | {tuple(spec.selected_layers)}
    ctx = Context(cfg, fspec, teacher, device, {spec.patch_layer} | {l[-1] for l in layers})
    rules = dict(SENS_RULES)
    agg_ref = next((n for n, a in rules.items() if _same_rule(a, spec)), None)
    if agg_ref is None:
        agg_ref = 'selected'
        rules[agg_ref] = Agg(spec.agg, spec.topk_ratio, spec.mad_sigma)
    ref = _proposed_eval(ctx, spec, rules)
    fps0 = 1.0 if spec.fps_ratio is None else float(spec.fps_ratio)
    rec = lambda v, r: {'value': v, 'auroc': r['auroc'], 'aupro': r['aupro']}
    axes = {'knn_k': [], 'fps_ratio': [], 'layers': [], 'aggregation': []}
    for v in sorted(set(SENS_K) | {spec.knn_k}):
        axes['knn_k'].append(rec(v, ref if v == spec.knn_k else
                                 _proposed_eval(ctx, replace(spec, knn_k=v))))
    for v in sorted(set(SENS_FPS) | {fps0}):
        same = abs(v - fps0) < 1e-12
        axes['fps_ratio'].append(rec(v, ref if same else _proposed_eval(
            ctx, replace(spec, fps_ratio=None if v >= 1.0 else v))))
    for v in sorted(layers, key=lambda t: (len(t), t)):
        same = tuple(v) == tuple(spec.selected_layers)
        axes['layers'].append(rec(list(v), ref if same else _proposed_eval(
            ctx, replace(spec, selected_layers=tuple(v), patch_layer=v[-1]))))
    for n in rules:
        axes['aggregation'].append({'value': n, 'auroc': ref['rules'][n].get('auroc'),
                                    'aupro': ref['aupro']})
    out = {'dataset': _dataset_tag(cfg), 'code_version': CODE_VERSION,
           'frozen_config': cfg.frozen_config, 'frozen': asdict(spec),
           'frozen_value': {'knn_k': spec.knn_k, 'fps_ratio': fps0,
                            'layers': list(spec.selected_layers), 'aggregation': agg_ref},
           'notes': 'one parameter at a time; each condition selects its own FPS bank',
           'axes': axes}
    os.makedirs(cfg.save_path, exist_ok=True)
    p = os.path.join(cfg.save_path, 'sensitivity.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f'[saved] {p}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 · OPERATING-POINT ERRORS AND QUALITATIVE FIGURES (E7)
# ══════════════════════════════════════════════════════════════════════════════

FAIL_CATEGORIES = {
    'BEH':  'Behavioural / motion anomaly without distinctive appearance',
    'SML':  'Anomalous object smaller than about one patch (14 px at input resolution)',
    'VIEW': 'Viewpoint, altitude or scene content absent from the memory bank',
    'PHOT': 'Illumination, shadow, weather or exposure change',
    'BLUR': 'Motion blur, compression or sensor artefact',
    'OCC':  'Occlusion or truncation at the frame border',
    'NOVN': 'Novel but normal object or activity (false positives only)',
    'LAB':  'Label boundary or annotation ambiguity',
    'OTH':  'Other (describe in notes)',
}
_FAIL_COLS = ['event_id', 'type', 'dataset', 'video', 'start', 'end', 'n_frames', 'rep_frame',
              'rep_score', 'threshold', 'margin', 'gt_area_frac', 'file', 'category', 'notes']


def _events(mask: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    ev, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        if i - prev > gap + 1:
            ev.append((start, prev))
            start = int(i)
        prev = int(i)
    ev.append((start, prev))
    return ev


@torch.no_grad()
def _probe(m: ProposedMethod, ctx: Context, sc: Scene, f: int, k_show: int = 3):
    """Discrepancy map and the k_show nearest retrieved training frames of one test frame."""
    pl, sco = m.spec.patch_layer, m.scorer
    te = ctx.features(sc.name, [pl])
    pats = te['patches'][pl]
    key = query_keys(te['cls'][:, f:f + 1], pats[f:f + 1], m.spec)[0]
    inds, _ = sco._retrieve(key, 0)
    d = sco._discrepancy(pats[f].float(), inds).cpu()
    cand = torch.arange(len(sco.idx), device=sco.device) if inds is None else inds
    q = key.float().to(sco.device)
    if sco.proj is not None:
        q = q @ sco.proj.T
    sims = sco.keys[cand] @ F.normalize(q, dim=0)
    top = cand[torch.argsort(sims, descending=True)[:k_show]].tolist()
    return d, [ctx.train_files[sco.idx[int(i)]] for i in top]


def _disp(path: str, h: int = 240) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    s = h / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), h), interpolation=cv2.INTER_AREA)


def _qual_figure(exs, ctx: Context, gt: Optional[PixelGT], vmin, vmax, k_show, stem,
                 width: Optional[float] = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g, ncol, nrow = ctx.fspec.grid, 2 + k_show, len(exs)
    compact = width is not None
    if compact:                                   # final printed size, e.g. \columnwidth
        tile = width / ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(width, tile * nrow + 0.12), squeeze=False)
        fs = 6
    else:
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.2 * nrow), squeeze=False)
        fs = 8
    for i, e in enumerate(exs):
        sc, f = e['scene'], e['rep_frame']
        img = _disp(sc.files[f])
        H, W = img.shape[:2]
        ax = axes[i]
        ax[0].imshow(img)
        if gt is not None:
            st, mk = gt.get(sc, f)
            if st == 'mask' and mk.any():
                mm = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                ax[0].contour(mm, levels=[0.5], colors='lime', linewidths=0.6 if compact else 1.0)
        ax[1].imshow(img)
        ax[1].imshow(_upsample(e['map'], (g, g), (H, W)), cmap='jet', alpha=0.5, vmin=vmin, vmax=vmax)
        for j in range(k_show):
            if j < len(e['nn']):
                ax[2 + j].imshow(_disp(e['nn'][j], H))
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.3 if compact else 0.8)
        if compact:                               # score inside the image, no side label
            ax[0].text(0.03, 0.97, f"s={e['rep_score']:.3f}", transform=ax[0].transAxes,
                       fontsize=fs, color='white', va='top', ha='left',
                       bbox=dict(facecolor='black', alpha=0.55, pad=1.0, lw=0))
        else:
            ax[0].set_ylabel(f"{e['type']} · {e['video']}\nframe {f} · s={e['rep_score']:.3f}",
                             fontsize=fs)
        if i == 0:
            heads = ['Input (+GT)', 'Anomaly map'] + [f'NN {j + 1}' for j in range(k_show)]
            for a, t in zip(ax, heads):
                a.set_title(t, fontsize=fs, pad=1.5 if compact else 4)
    if compact:
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 0.12 / (tile * nrow + 0.12),
                            wspace=0.02, hspace=0.02)
    else:
        fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{stem}.{ext}', dpi=300 if compact else 200,
                    bbox_inches='tight', pad_inches=0.01 if compact else 0.1)
    plt.close(fig)


def cmd_failures(args):
    cfg, spec, fspec, device, teacher = setup(args)
    cfg.viz_dir = None
    ctx = Context(cfg, fspec, teacher, device, (spec.patch_layer,))
    out = os.path.join(cfg.save_path, 'failures')
    os.makedirs(out, exist_ok=True)
    m = ProposedMethod(spec, ctx)
    m.fit(ctx)
    s, y = m.score_frames(ctx)['scores'], ctx.labels
    thr = float(np.quantile(s[y == 0], 1.0 - FAIL_FPR))
    gt = PixelGT(ctx.mask_root) if ctx.mask_root else None
    tag, rows, off = _dataset_tag(cfg), [], 0
    for sc in ctx.scenes:
        n = len(sc.files)
        ss, yy = s[off:off + n], sc.labels
        pred = ss >= thr
        for typ, mask, pick in (('FN', (yy == 1) & ~pred, np.argmin), ('FP', (yy == 0) & pred, np.argmax),
                                ('TP', (yy == 1) & pred, np.argmax)):
            for a, b in _events(mask, FAIL_GAP):
                fr = np.arange(a, b + 1)
                fr = fr[mask[a:b + 1]]
                rf = int(fr[pick(ss[fr])])
                area = None
                if gt is not None:
                    st, gm = gt.get(sc, rf)
                    area = float(gm.mean()) if st == 'mask' else None
                rows.append({'type': typ, 'dataset': tag, 'video': sc.name, 'start': int(a),
                             'end': int(b), 'n_frames': int(len(fr)), 'rep_frame': rf,
                             'rep_score': float(ss[rf]), 'threshold': thr,
                             'margin': float(ss[rf] - thr), 'gt_area_frac': area,
                             'file': sc.files[rf], 'category': '', 'notes': ''})
        off += n
    order = {'FN': 0, 'FP': 1, 'TP': 2}
    rows.sort(key=lambda r: (order[r['type']], r['rep_score'] if r['type'] == 'FN' else -r['rep_score']))
    for i, r in enumerate(rows):
        r['event_id'] = f"{r['type']}{i:04d}"
    p = os.path.join(out, 'failure_events.csv')
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_FAIL_COLS)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, 'categories.md'), 'w') as fh:
        fh.write(f'# Failure categories ({tag}; threshold at {100 * FAIL_FPR:g}% FPR = {thr:.4f})\n\n'
                 'Fill the `category` column of failure_events.csv for every FN and FP event with ONE '
                 'code. Two annotators should label independently; report Cohen\'s kappa on a '
                 'shared subset. TP events are for figures only.\n\n')
        fh.write(''.join(f'- **{k}**: {v}\n' for k, v in FAIL_CATEGORIES.items()))
    cnt = {t: sum(r['type'] == t for r in rows) for t in order}
    print(f'[failures] threshold {thr:.4f} (FPR {FAIL_FPR:g}); events {cnt}; saved {p}')

    by = {sc.name: sc for sc in ctx.scenes}
    exs = []
    for t, k in (('TP', args.n_tp), ('FN', args.n_fig), ('FP', args.n_fig)):
        for r in [r for r in rows if r['type'] == t][:k]:
            sc = by[r['video']]
            d, nn = _probe(m, ctx, sc, r['rep_frame'], args.k_show)
            exs.append({**r, 'map': d, 'nn': nn, 'scene': sc})
    if exs:
        allm = torch.cat([e['map'].reshape(-1) for e in exs])
        vmin, vmax = float(allm.min()), float(allm.max())
        for t in ('TP', 'FN', 'FP'):
            sel = [e for e in exs if e['type'] == t]
            if sel:
                _qual_figure(sel, ctx, gt, vmin, vmax, args.k_show, os.path.join(out, f'fig_examples_{t}'), args.fig_width)
        print(f'[failures] figures in {out} (shared colour scale [{vmin:.3f}, {vmax:.3f}])')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15b · HEAT-MAP OVERLAY FIGURES (`overlays`)
# ══════════════════════════════════════════════════════════════════════════════
# Frames are chosen AUTOMATICALLY by score (no hand-picking), de-duplicated in
# time (>= min_gap frames apart within a video) and capped per video, so a
# figure is not ten near-identical frames of one event. All panels of a
# dataset share ONE colour scale (percentiles of the shown maps); per-image
# normalisation would make confidently normal frames look anomalous.
#   best_anomalous   highest-scoring anomalous frames   (confident detections)
#   best_normal      lowest-scoring normal frames       (confidently normal)
#   worst_anomalous  lowest-scoring anomalous frames    (missed-detection candidates)
#   worst_normal     highest-scoring normal frames      (false-alarm candidates)

OVERLAY_GROUPS = {
    'best_anomalous':  (1, True,  'Highest-scoring anomalous frames (confident detections)'),
    'best_normal':     (0, False, 'Lowest-scoring normal frames (confidently normal)'),
    'worst_anomalous': (1, False, 'Lowest-scoring anomalous frames (missed-detection candidates)'),
    'worst_normal':    (0, True,  'Highest-scoring normal frames (false-alarm candidates)'),
}


def _select_frames(ctx: Context, scores, label: int, highest: bool, n: int,
                   min_gap: int, per_video: int) -> List[dict]:
    flat, off = [], 0
    for sc in ctx.scenes:
        for f in range(len(sc.files)):
            if int(sc.labels[f]) == label:
                flat.append((float(scores[off + f]), sc, f))
        off += len(sc.files)
    flat.sort(key=lambda t: -t[0] if highest else t[0])
    out, per = [], {}
    for s, sc, f in flat:
        if per.get(sc.name, 0) >= per_video:
            continue
        if any(o['scene'] is sc and abs(o['frame'] - f) < min_gap for o in out):
            continue
        out.append({'scene': sc, 'frame': f, 'score': s, 'label': label})
        per[sc.name] = per.get(sc.name, 0) + 1
        if len(out) == n:
            break
    if len(out) < n:
        print(f'[overlays] only {len(out)} of {n} frames satisfy min_gap={min_gap}, '
              f'per_video={per_video} (label {label})')
    return out


def _tile(e: dict, d, grid, gt: Optional[PixelGT], sigma: float, h: int = 240):
    img = _disp(e['scene'].files[e['frame']], h)
    H, W = img.shape[:2]
    hm = _upsample(d, grid, (H, W)).astype(np.float32)
    if sigma > 0:
        hm = cv2.GaussianBlur(hm, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    mk = None
    if gt is not None:
        st, g = gt.get(e['scene'], e['frame'])
        if st == 'mask' and g.any():
            mk = cv2.resize(g.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    return img, hm, mk


def _overlay_grid(tiles: List[dict], title: str, out: str, stem: str, vmin, vmax, cmap: str,
                  ncol: int, show_input: bool, cbar_label: str, alpha: float = 0.5):
    plt = _plt()
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    nr, rp = math.ceil(len(tiles) / ncol), (2 if show_input else 1)
    fig, axes = plt.subplots(nr * rp, ncol, figsize=(2.6 * ncol, 2.0 * nr * rp + 0.8), squeeze=False)
    for a in axes.ravel():
        a.axis('off')
    for i, t in enumerate(tiles):
        r, c = divmod(i, ncol)
        r *= rp
        if show_input:
            a = axes[r][c]
            a.imshow(t['img'])
            if t['mask'] is not None:
                a.contour(t['mask'], levels=[0.5], colors='lime', linewidths=0.8)
            a.set_title(t['caption'], fontsize=7)
        a = axes[r + rp - 1][c]
        a.imshow(t['img'])
        a.imshow(t['hm'], cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        if t['mask'] is not None:
            a.contour(t['mask'], levels=[0.5], colors='lime', linewidths=0.8)
        if not show_input:
            a.set_title(t['caption'], fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.subplots_adjust(wspace=0.03, hspace=0.18, right=0.92)
    cax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap=cmap), cax=cax, label=cbar_label)
    _savefig(fig, out, stem)
    plt.close(fig)


def _overlay_compare(args, ctx: Context, cfg: BenchConfig, spec: MethodSpec, picks: List[dict],
                     gt: Optional[PixelGT], out: str):
    """Same frames, one column per method; colour scale shared within each method column."""
    cols = [('Proposed', [e['map'] for e in picks], (ctx.fspec.grid,) * 2)]
    for name in args.compare:
        bm = build_method(name, ctx, cfg, spec)
        bm.fit(ctx)
        bm._want_map = True
        ds = FrameDataset([e['scene'].files[e['frame']] for e in picks], ctx.fspec)
        maps = []
        with torch.no_grad():
            for i in range(len(ds)):
                bm._last_map = None
                bm.score(ds[i]['img'][None].to(ctx.device))
                maps.append(None if bm._last_map is None else bm._last_map.detach().float().cpu())
        cols.append((bm.name, maps, method_grid(bm, ctx)))
        del bm
        _free(ctx.device)
    plt = _plt()
    fig, axes = plt.subplots(len(picks), 1 + len(cols), figsize=(2.6 * (1 + len(cols)), 2.0 * len(picks)),
                             squeeze=False)
    rng = []
    for _, maps, _ in cols:
        v = np.concatenate([m.numpy().ravel() for m in maps if m is not None]) if any(
            m is not None for m in maps) else np.zeros(1)
        rng.append(np.percentile(v, args.vrange))
    for i, e in enumerate(picks):
        img, _, mk = _tile(e, e['map'], (ctx.fspec.grid,) * 2, gt, 0.0)
        H, W = img.shape[:2]
        axes[i][0].imshow(img)
        if mk is not None:
            axes[i][0].contour(mk, levels=[0.5], colors='lime', linewidths=0.8)
        axes[i][0].set_ylabel(f"{'anom.' if e['label'] else 'normal'} · {e['scene'].name}\n#{e['frame']}",
                              fontsize=7)
        for j, (cname, maps, grid) in enumerate(cols, start=1):
            a = axes[i][j]
            a.imshow(img)
            if maps[i] is not None:
                hm = _upsample(maps[i], grid, (H, W)).astype(np.float32)
                if args.sigma > 0:
                    hm = cv2.GaussianBlur(hm, (0, 0), args.sigma, borderType=cv2.BORDER_REFLECT)
                a.imshow(hm, cmap=args.cmap, alpha=0.5, vmin=rng[j - 1][0], vmax=rng[j - 1][1])
            if i == 0:
                a.set_title(cname, fontsize=8)
        if i == 0:
            axes[i][0].set_title('Input (+GT)', fontsize=8)
    for a in axes.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    _savefig(fig, out, 'fig_overlay_compare')
    plt.close(fig)


def cmd_overlays(args):
    cfg, spec, fspec, device, teacher = setup(args)
    cfg.viz_dir, cfg.measure_latency = None, False
    ctx = Context(cfg, fspec, teacher, device, (spec.patch_layer,))
    out = os.path.join(cfg.save_path, 'overlays')
    os.makedirs(out, exist_ok=True)
    m = ProposedMethod(spec, ctx)
    m.fit(ctx)
    scores = m.score_frames(ctx)['scores']
    gt = PixelGT(ctx.mask_root) if ctx.mask_root else None
    groups = [g for g in OVERLAY_GROUPS if args.mode == 'both' or g.startswith(args.mode)]
    want_cmp = {}
    for tok in (args.compare_groups.split(',') if args.compare else []):
        g, k = tok.split(':')
        want_cmp[g] = int(k)
    sel = {}
    for g in dict.fromkeys(groups + list(want_cmp)):
        lab, hi, _ = OVERLAY_GROUPS[g]
        sel[g] = _select_frames(ctx, scores, lab, hi, args.n, args.min_gap, args.per_video)
        for e in sel[g]:
            e['map'], _ = _probe(m, ctx, e['scene'], e['frame'], 1)
    shown = [e for g in groups for e in sel[g]]
    if not shown:
        print('[overlays] nothing to plot')
        return
    vmin, vmax = np.percentile(np.concatenate([e['map'].numpy().ravel() for e in shown]), args.vrange)
    grid, tag, rows = (ctx.fspec.grid,) * 2, _dataset_tag(cfg), []
    for g in groups:
        tiles = []
        for rank, e in enumerate(sel[g], 1):
            img, hm, mk = _tile(e, e['map'], grid, gt, args.sigma)
            tiles.append({'img': img, 'hm': hm, 'mask': mk,
                          'caption': f"{e['scene'].name} #{e['frame']}  s={e['score']:.3f}"})
            area = None
            if gt is not None:
                st, gm = gt.get(e['scene'], e['frame'])
                area = float(gm.mean()) if st == 'mask' else None
            rows.append({'group': g, 'rank': rank, 'video': e['scene'].name, 'frame': e['frame'],
                         'label': e['label'], 'score': e['score'], 'gt_area_frac': area,
                         'file': e['scene'].files[e['frame']]})
        if tiles:
            _overlay_grid(tiles, f'{OVERLAY_GROUPS[g][2]} — {tag}', out, f'fig_overlay_{g}',
                          vmin, vmax, args.cmap, args.ncol, not args.no_input,
                          'patch discrepancy (1 − max cosine)')
    with open(os.path.join(out, 'overlay_frames.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {'dataset': tag, 'n': args.n, 'groups': groups, 'min_gap': args.min_gap,
            'per_video': args.per_video, 'vrange_percentiles': list(args.vrange),
            'vmin': float(vmin), 'vmax': float(vmax), 'display_sigma': args.sigma, 'cmap': args.cmap,
            'frozen_config': cfg.frozen_config, 'compare': args.compare or []}
    with open(os.path.join(out, 'overlay_meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    if args.compare:
        picks = [e for g, k in want_cmp.items() for e in sel[g][:k]]
        _overlay_compare(args, ctx, cfg, spec, picks, gt, out)
    print(f'[overlays] {len(rows)} frames in {len(groups)} figure(s); shared scale '
          f'[{vmin:.3f}, {vmax:.3f}]; saved in {out}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 · ARTICLE TABLES, FIGURES AND CAPTIONS (`paper`)
# ══════════════════════════════════════════════════════════════════════════════

PAPER_TEMPLATE = {
    'out_dir': 'paper',
    'proposed': 'Proposed (DINOv2-MemKNN)',
    'selected_config': 'experiments/development/selected_config.json',
    'dev': [{'name': 'ShanghaiTech', 'path': 'experiments/shanghaitech/'},
            {'name': 'DA-Highway', 'path': 'experiments/Drone-Anomaly/Highway/'}],
    'heldout': [{'name': 'UIT-ADrone', 'path': 'experiments/UIT-ADrone/'}],
    'cost_dataset': 'UIT-ADrone',
    'methods': ['Proposed (DINOv2-MemKNN)', 'PatchCore (DINOv2)', 'AnomalyDINO (DINOv2)',
                'SPADE (DINOv2)', 'PaDiM (DINOv2)', 'SimpleNet (DINOv2)'],
}


def _devs(M) -> List[dict]:
    """Development datasets from the manifest ('dev' may be one dict or a list)."""
    d = M.get('dev') or []
    return d if isinstance(d, list) else [d]

# Draft wording per contrast; the verdict sentence is chosen from the CIs.
CONTRAST_TEXT = {
    'G1': ('Aggregation rule', 'Does the selected aggregation of the patch map detect anomalous frames better than its maximum?',
           'Averaging the most discrepant patches is more robust than a single maximum, which reacts to isolated noisy patches.',
           'The maximum is preferable, suggesting that anomalies occupy very few patches.',
           'The aggregation rule has no measurable effect, so it does not explain the gains of the method.'),
    'G2': ('Aggregation rule applied to PatchCore', 'Does the same rule also improve PatchCore?',
           'The rule is a generic improvement and must not be credited to retrieval.',
           'The rule helps only in combination with retrieval.',
           'The rule does not transfer to PatchCore.'),
    'G3': ('Retrieval versus the whole bank', 'Does restricting patch matching to the K retrieved frames help compared with matching against every bank frame?',
           'Retrieval removes spurious matches from unrelated viewpoints, the central claim of the method.',
           'Matching against the whole bank is more accurate; retrieval mainly saves computation.',
           'Retrieval does not change accuracy; its benefit is computational (Table 6).'),
    'G4': ('Retrieval quality', 'Do the K nearest frames matter, or would any K frames do?',
           'Choosing similar frames, not merely using fewer reference frames, produces the gain.',
           'Random frames are better, which would contradict the retrieval hypothesis.',
           'Any K frames work equally well; the global key carries little useful information.'),
    'G5': ('FPS frame bank', 'Does subsampling the training frames with farthest-point sampling cost accuracy?',
           'FPS removes redundant frames and even improves detection.',
           'Subsampling costs accuracy, to be weighed against its memory saving (Table 6).',
           'FPS reduces memory without measurable loss of accuracy.'),
    'G6': ('Cosine versus L2 matching', 'Does cosine matching of unit-normalised tokens outperform L2 matching of raw tokens (SPADE-patch)?',
           'Normalising DINOv2 tokens is an important design choice.',
           'Raw-token L2 matching is preferable on these data.',
           'The similarity measure does not matter here.'),
    'G7': ('Proposed versus PatchCore at equal memory', 'At the same number of stored vectors and the same aggregation, is a frame bank with retrieval better than a patch coreset?',
           'Organising memory by frames and retrieving viewpoint-matched frames is more effective than a global patch coreset.',
           'A patch coreset uses the same memory more effectively.',
           'At equal memory the two designs are equivalent.'),
    'G8': ('Bank construction without retrieval', 'Without retrieval and at the same vector budget, is a frame-level FPS bank better than a patch coreset?',
           'Part of the gain comes from the frame-level bank itself.',
           'A patch coreset is the better bank; any gain in G7 must come from retrieval.',
           'Bank construction alone explains nothing; the gain in G7 comes from retrieval.'),
    'G9': ('PatchCore memory budget', "Does shrinking PatchCore's coreset to the proposed method's budget hurt it?",
           'A smaller coreset unexpectedly helps PatchCore.',
           'PatchCore needs its larger memory, so G7 is the fair comparison.',
           'PatchCore is insensitive to memory in this range.'),
    'G10': ('Proposed versus SPADE as published', "How does the method compare with SPADE's image-level kNN score?",
            'Patch-level matching after retrieval improves on image-level kNN.',
            'Image-level kNN is better on these data.',
            'Both scores perform similarly.'),
    'CHK': ('Implementation check', 'With last-layer CLS keys, Proposed without FPS and SPADE-patch with cosine similarity must give the same scores.',
            'The two code paths disagree; investigate before reporting.',
            'The two code paths disagree; investigate before reporting.',
            'The scores agree, so the method differs from SPADE-patch only through the FPS bank (G5), the similarity (G6) and the aggregation rule (G1).'),
}


def _fmt(x, nd: int = 3) -> str:
    v = _num(x)
    return 'n/a' if v is None else f'{v:.{nd}f}'


def _ms(vals, nd: int = 3) -> str:
    v = [x for x in map(_num, vals) if x is not None]
    if not v:
        return 'n/a'
    return f'{np.mean(v):.{nd}f}' + (f'±{np.std(v, ddof=1):.{nd}f}' if len(v) > 1 else '')


def _load_runs(path: str, sub: str = 'main') -> List[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(path, 'runs', sub, '*.json'))):
        with open(p) as fh:
            rows.append(json.load(fh))
    return rows


def _group(rows: List[dict]) -> Dict[str, List[dict]]:
    g = {}
    for r in rows:
        g.setdefault(r['method'], []).append(r)
    return g


def _mean(rs: List[dict], k: str) -> Optional[float]:
    v = [x for x in (_num(r.get(k)) for r in rs) if x is not None]
    return float(np.mean(v)) if v else None


def _lead(c) -> Optional[float]:
    try:
        return float(str(c).replace('**', '').split('±')[0].split(' ')[0])
    except ValueError:
        return None


def _bold_best(rows: List[list], cols):
    for j in cols:
        v = [_lead(r[j]) for r in rows]
        ok = [x for x in v if x is not None]
        if ok:
            best = max(ok)
            for r, x in zip(rows, v):
                if x is not None and abs(x - best) < 1e-12:
                    r[j] = f'**{r[j]}**'


def _tex_cell(s) -> str:
    s = str(s)
    bold = s.startswith('**') and s.endswith('**') and len(s) > 4
    if bold:
        s = s[2:-2]
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('_', r'\_'), ('#', r'\#')):
        s = s.replace(a, b)
    for a, b in (('±', r'$\pm$'), ('†', r'$^\dagger$'), ('Δ', r'$\Delta$'), ('−', '-'),
                 ('≈', r'$\approx$'), ('✓', r'\checkmark'), ('ρ', r'$\rho$'), ('σ', r'$\sigma$')):
        s = s.replace(a, b)
    return r'\textbf{' + s + '}' if bold else s


def _write(out: str, stem: str, headers, rows, caption: str, label: str, colspec=None):
    os.makedirs(out, exist_ok=True)
    md = [f'**{caption}**\n', '| ' + ' | '.join(headers) + ' |',
          '|' + '|'.join(['---'] * len(headers)) + '|']
    md += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    with open(os.path.join(out, stem + '.md'), 'w') as fh:
        fh.write('\n'.join(md) + '\n')
    colspec = colspec or 'l' + 'c' * (len(headers) - 1)
    tex = [r'\begin{table*}[t]', r'\centering', r'\small', rf'\caption{{{_tex_cell(caption)}}}',
           rf'\label{{{label}}}', rf'\begin{{tabular}}{{{colspec}}}', r'\toprule',
           ' & '.join(_tex_cell(h) for h in headers) + r' \\', r'\midrule']
    tex += [' & '.join(_tex_cell(c) for c in r) + r' \\' for r in rows]
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    with open(os.path.join(out, stem + '.tex'), 'w') as fh:
        fh.write('\n'.join(tex) + '\n')
    with open(os.path.join(out, stem + '.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows([[str(c).replace('**', '') for c in r] for r in rows])
    print(f'[paper] {stem}')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _savefig(fig, out: str, stem: str):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{stem}.{ext}'), dpi=300, bbox_inches='tight')
    print(f'[paper] {stem}')


def _paper_table2(M, out, caps):
    prop, held, devs = M['proposed'], M['heldout'], _devs(M)
    D = {s['name']: _group(_load_runs(s['path'])) for s in held}
    DV = {d['name']: _group(_load_runs(d['path'])) for d in devs}
    methods = M.get('methods') or sorted({m for g in D.values() for m in g})
    px = any(_mean(rs, 'aupro') is not None for g in D.values() for rs in g.values())
    hdr = (['Method'] + [s['name'] for s in held] + ['Held-out mean', 'AP', 'TPR@5%FPR']
           + (['AUPRO'] if px else []))
    nb = len(hdr)
    for d in devs:
        hdr += [f"{d['name']}† AUROC", f"{d['name']}† AUPRO"]
    rows, seeds = [], 1
    for m in methods:
        r, sm, ap, tp, au = [m], [], [], [], []
        for s in held:
            rs = D[s['name']].get(m, [])
            seeds = max(seeds, len(rs))
            r.append(_ms([x.get('auroc') for x in rs]))
            for lst, k in ((sm, 'auroc'), (ap, 'ap'), (tp, 'tpr_at_fpr05'), (au, 'aupro')):
                v = _mean(rs, k)
                if v is not None:
                    lst.append(v)
        cov = '' if len(sm) == len(held) else f' ({len(sm)}/{len(held)})'
        r += [(_fmt(np.mean(sm)) + cov) if sm else 'n/a', _fmt(np.mean(ap)) if ap else 'n/a',
              _fmt(np.mean(tp)) if tp else 'n/a'] + ([_fmt(np.mean(au)) if au else 'n/a'] if px else [])
        for d in devs:
            rs = DV[d['name']].get(m, [])
            r += [_ms([x.get('auroc') for x in rs]), _ms([x.get('aupro') for x in rs])]
        rows.append(r)
    _bold_best(rows, range(1, nb))
    names = ' and '.join(d['name'] for d in devs)
    cap = (f'Frame-level anomaly detection on {len(held)} held-out datasets: AUROC per dataset '
           f'(mean±s.d. over up to {seeds} seeds for stochastic methods), held-out mean, and held-out '
           f'mean AP and TPR at 5% FPR. † {names}: development datasets, used only to select the '
           f'hyperparameters of every method; excluded from all held-out means. AUPRO only where '
           f'pixel annotations exist. Best held-out value in bold.')
    _write(out, 'table2_main', hdr, rows, cap, 'tab:main',
           'l' + 'c' * (nb - 1) + ''.join('|cc' for _ in devs))
    caps.append(('Table 2', cap))
    st = []
    from scipy.stats import wilcoxon
    for m in methods:
        if m == prop:
            continue
        d = [_mean(D[s['name']].get(prop, []), 'auroc') - _mean(D[s['name']].get(m, []), 'auroc')
             for s in held if _mean(D[s['name']].get(prop, []), 'auroc') is not None
             and _mean(D[s['name']].get(m, []), 'auroc') is not None]
        if not d:
            continue
        d = np.asarray(d)
        ci, p = 'n/a', 'n<5'
        if len(d) >= 3:
            bs = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
            ci = f'[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]'
        if len(d) >= 5 and np.any(d != 0):
            try:
                p = _fmt(wilcoxon(d).pvalue, 3)
            except ValueError:
                p = 'n/a'
        st.append([m, len(d), f'{int((d > 0).sum())}/{len(d)}', f'{d.mean():+.3f}', ci, p])
    if st:
        _write(out, 'table2_stats', ['Baseline', 'Datasets', 'Proposed wins', 'Mean ΔAUROC',
                                     'Dataset-bootstrap 95% CI', 'Wilcoxon p'], st,
               'Proposed minus each baseline across held-out datasets (dataset = unit of analysis).',
               'tab:main_stats')

def _paper_pervideo(M, out):
    sets = [(s['name'], s['path'], '') for s in M['heldout']]
    sets += [(d['name'], d['path'], '†') for d in _devs(M)]
    for name, path, mark in sets:
        G = _group(_load_runs(path))
        if not G:
            continue
        vids = sorted({v for rs in G.values() for r in rs for v in (r.get('per_video') or {})})
        rows = [[m] + [_ms([(r.get('per_video') or {}).get(v) for r in G.get(m, [])]) for v in vids]
                for m in (M.get('methods') or sorted(G)) if m in G]
        _bold_best(rows, range(1, len(vids) + 1))
        _write(out, f'tableS_pervideo_{_safe(name)}', ['Method'] + vids, rows,
               f'Per-video frame AUROC on {name}{mark}.', f'tab:pv_{_safe(name)}')


def _paper_selection(M, out, caps):
    devs = _devs(M)
    p = M.get('selected_config') or (os.path.join(devs[0]['path'], 'selected_config.json')
                                     if devs else '')
    if not (p and os.path.exists(p)):
        return
    with open(p) as fh:
        C = json.load(fh)
    rows = []
    for meth, recs in C.get('results', {}).items():
        sel = C.get('selected_index', {}).get(meth)
        for i, r in enumerate(recs):
            par = ', '.join(f'{k}={v}' for k, v in r.items() if k not in _METRIC_KEYS) or 'published'
            per = ' / '.join(_fmt(x) for x in (r.get('per_dataset_auroc') or [])) or 'n/a'
            rows.append([meth, par, _fmt(r.get('auroc')), per, _fmt(r.get('ap')),
                         _fmt(r.get('aupro')), '✓' if i == sel else ''])
    names = ' + '.join(d['name'] for d in devs) or C.get('dev_dataset')
    cap = (f'Hyperparameter selection on the development datasets ({names}): every grid point; '
           f'criterion mean frame AUROC over the development datasets (per-dataset values in the '
           f'same order); first configuration in the declared order within '
           f'{C["protocol"]["tolerance"]} of the best (✓). Frozen at git {C.get("git")}.')
    _write(out, 'tableS1_selection', ['Method', 'Configuration', 'Mean AUROC', 'AUROC per dataset',
                                      'AP', 'AUPRO', 'Selected'], rows, cap, 'tab:selection',
           'llccccc')
    caps.append(('Table S1', cap))

def _contrast_paragraph(cid: str, ents: List[dict]) -> str:
    title, question, pos, neg, null = CONTRAST_TEXT[cid]
    n = len(ents)
    d = np.array([e['delta'] for e in ents])
    sp = sum(e['ci_low'] > 0 for e in ents)
    sn = sum(e['ci_high'] < 0 for e in ents)
    verdict = pos if sp > n / 2 else neg if sn > n / 2 else null if sp + sn == 0 else \
        'The effect depends on the dataset; discuss the datasets individually.'
    txt = (f'**{cid} — {title}** ({ents[0]["a_label"]} vs {ents[0]["b_label"]}). {question} '
           f'Across {n} held-out dataset(s), the change shifted frame AUROC by {d.mean():+.3f} on '
           f'average (range {d.min():+.3f} to {d.max():+.3f}); the 95% block-bootstrap interval '
           f'excluded zero in favour of the first condition in {sp} and against it in {sn}.')
    da = [e['d_aupro'] for e in ents if _num(e.get('d_aupro')) is not None]
    if da:
        txt += f' AUPRO changed by {np.mean(da):+.3f}.'
    dm = [e['d_ms'] for e in ents if _num(e.get('d_ms')) is not None]
    if dm:
        txt += f' Median latency changed by {np.mean(dm):+.1f} ms.'
    return txt + f' {verdict} [Author: add dataset-specific interpretation.]\n'


def _paper_table3(M, out, caps):
    C = {}
    for s in M['heldout']:
        p = os.path.join(s['path'], 'grid_contrasts.json')
        if os.path.exists(p):
            with open(p) as fh:
                J = json.load(fh)
            C[s['name']] = {c['id']: dict(c, check_valid=J.get('check_valid')) for c in J['contrasts']}
    if not C:
        print('[paper] no grid_contrasts.json in held-out datasets; Table 3 skipped')
        return
    names = list(C)
    rows, paras = [], []
    for cid, *_ in GRID_CONTRASTS:
        ents = [C[n][cid] for n in names if cid in C[n]]
        if not ents:
            continue
        cells = []
        for n in names:
            e = C[n].get(cid)
            cells.append('n/a' if e is None else
                         f'{e["delta"]:+.3f} [{e["ci_low"]:+.3f}, {e["ci_high"]:+.3f}]'
                         + ('*' if _num(e.get('p_holm')) is not None and e['p_holm'] < 0.05 else ''))
        da = [e['d_aupro'] for e in ents if _num(e.get('d_aupro')) is not None]
        dm = [e['d_ms'] for e in ents if _num(e.get('d_ms')) is not None]
        rows.append([cid, ents[0]['isolates']] + cells +
                    [f'{np.mean([e["delta"] for e in ents]):+.3f}',
                     f'{np.mean(da):+.3f}' if da else 'n/a', f'{np.mean(dm):+.1f}' if dm else 'n/a'])
        if cid != 'CHK' or all(e.get('check_valid') for e in ents):
            paras.append(_contrast_paragraph(cid, ents))
    cap = ('Attribution contrasts on held-out datasets. ΔAUROC = AUROC(a) − AUROC(b) with 95% paired '
           'moving-block bootstrap intervals within videos; * Holm-adjusted p < 0.05 within each '
           'dataset (CHK excluded). ΔAUPRO is a point estimate; Δms is the change in median '
           'per-frame latency. All rows use the frozen configuration and the same backbone.')
    _write(out, 'table3_contrasts', ['ID', 'Isolates'] + [f'{n} ΔAUROC' for n in names]
           + ['Mean ΔAUROC', 'Mean ΔAUPRO', 'Mean Δms'], rows, cap, 'tab:contrasts',
           'll' + 'c' * (len(names) + 3))
    with open(os.path.join(out, 'contrasts_paragraphs.md'), 'w') as fh:
        fh.write('\n'.join(paras))
    caps.append(('Table 3', cap))


def _flat(pts, frozen) -> Optional[float]:
    vals = [e['value'] for e in pts]
    i = next((j for j, v in enumerate(vals) if v == frozen or
              (_num(v) is not None and _num(frozen) is not None and abs(float(v) - float(frozen)) < 1e-9)), None)
    if i is None:
        return None
    a = [_num(e['auroc']) for e in pts[max(0, i - 1):i + 2] if _num(e['auroc']) is not None]
    return max(a) - min(a) if a else None


def _paper_sensitivity(M, out, caps):
    sets = [(s['name'], s['path'], False) for s in M['heldout']]
    sets += [(d['name'] + '†', d['path'], True) for d in _devs(M)]
    L = []
    def bank_sizes(path, d):
        """Bank size in frames for each FPS ratio of the sensitivity axis (the method's own rule)."""
        try:
            with open(os.path.join(path, 'env.json')) as fh:
                bc = json.load(fh)['bench_config']
            c = BenchConfig(data_path=bc['data_path'], save_path=path)
            c.train_stride = int(bc.get('train_stride') or 1)
            n = len(train_frame_list(c))
        except Exception as e:
            print(f'[paper] bank sizes unavailable for {path}: {e}')
            return None
        floor = int(d.get('frozen', {}).get('fps_min_frames', 0) or 0)
        return [n if e['value'] >= 1.0 else min(n, max(1, int(e['value'] * n), floor))
                for e in d['axes'].get('fps_ratio', [])]

    L, banks = [], {}
    for name, path, isdev in sets:
        p = os.path.join(path, 'sensitivity.json')
        if os.path.exists(p):
            with open(p) as fh:
                L.append((name, json.load(fh), isdev))
            banks[name] = bank_sizes(path, L[-1][1])
    if not L:
        print('[paper] no sensitivity.json found; sensitivity figure skipped')
        return
    plt = _plt()
    panels = (('knn_k', 'K (retrieved frames)', True), ('fps_ratio', 'Bank size (frames)', True),
              ('layers', 'CLS key layers', False), ('aggregation', 'Aggregation rule', False))
    has_px = any(_num(e.get('aupro')) is not None for _, d, _ in L for ax in d['axes'].values() for e in ax)
    metrics = ['auroc', 'aupro'] if has_px else ['auroc']
    fig, axes = plt.subplots(len(metrics), 4, figsize=(16, 3.6 * len(metrics)), squeeze=False,
                         sharey='row')
    fv = L[0][1]['frozen_value']
    lab = lambda v: (f'{v[0]}–{v[-1]}' if len(v) > 1 else str(v[0])) if isinstance(v, list) else str(v)
    dev_styles = iter([dict(ls='--', color='0.45', marker='x', ms=4),
                   dict(ls=':',  color='0.25', marker='+', ms=5)])
    dev_style = {name: next(dev_styles) for name, _, isdev in L if isdev}
    for c, (k, xl, num) in enumerate(panels):
        for r, met in enumerate(metrics):
            a = axes[r][c]
            if k == 'aggregation' and met == 'aupro':      # pixel maps don't depend on the rule
                a.axis('off')
                continue
            ticks = None
            for name, d, isdev in L:
                pts = d['axes'].get(k, [])
                if not pts:
                    continue
                if k == 'fps_ratio' and banks.get(name):
                    xs = banks[name]                       # bank size in frames, not rho
                else:
                    xs = [e['value'] for e in pts] if num else list(range(len(pts)))
                ys = [np.nan if _num(e.get(met)) is None else e[met] for e in pts]
                if isdev:
                    a.plot(xs, ys, label=name, lw=1.2, **dev_style[name])
                else:
                    a.plot(xs, ys, '-o', ms=4, label=name)
                if k == 'fps_ratio' and banks.get(name):   # ring at this dataset's frozen setting
                    fz = d['frozen_value']['fps_ratio']
                    i = next((j for j, e in enumerate(pts) if abs(e['value'] - fz) < 1e-9), None)
                    if i is not None and np.isfinite(ys[i]):
                        a.plot(xs[i], ys[i], 'o', mfc='none', mec='k', ms=9, mew=1.2, zorder=5)
                if not num:
                    ticks = [lab(e['value']) for e in pts]
            if num:
                a.set_xscale('log')
                if k != 'fps_ratio':               # bank panel: frozen points are ringed instead
                    a.axvline(fv[k], ls=':', c='k')
            elif ticks:
                a.set_xticks(range(len(ticks)))
                a.set_xticklabels(ticks, rotation=45, ha='right', fontsize=8)
                fl = lab(fv[k]) if k == 'layers' else fv[k]
                if fl in ticks:
                    a.axvline(ticks.index(fl), ls=':', c='k')
            a.set_xlabel(xl)
            a.set_ylabel('Frame AUROC' if met == 'auroc' else 'AUPRO')
            a.grid(alpha=.3)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out, 'fig_sensitivity')
    plt.close(fig)
    rows, worst = [], 0.0
    for name, d, isdev in L:
        if isdev:
            continue
        v = [_flat(d['axes'].get(k, []), d['frozen_value'][k]) for k, *_ in panels]
        worst = max([worst] + [x for x in v if x is not None])
        rows.append([name] + [_fmt(x) for x in v])
    if rows:
        _write(out, 'table_sensitivity_flatness', ['Dataset', 'K', 'FPS ratio', 'Layers', 'Aggregation'],
               rows, 'Range of held-out frame AUROC over the frozen value and its immediate '
               'neighbours on each axis.', 'tab:flatness')
    caps.append(('Figure (sensitivity)',
                 'Sensitivity of frame AUROC' + (' (top) and AUPRO (bottom)' if has_px else '')
                 + ' to one hyperparameter at a time, others fixed at the frozen configuration '
                 + ' to one hyperparameter at a time, others fixed at the frozen configuration '
                 '(dotted line). Solid: held-out datasets; dashed grey: development dataset†. '
                 'AUPRO is not shown for the aggregation rule, which does not change the pixel maps. '
                 '(dotted line). Solid: held-out datasets; dashed grey: development dataset†. '
                 'For the FPS ratio, the x-axis shows the resulting bank size in frames '
                 '(min(N, max(ρN, 50)); N training frames kept); black rings mark each '
                 'dataset\'s frozen setting (ρ = 0.05). '
                 f'Largest AUROC range within the neighbourhood of the frozen value: {worst:.3f}.'))


def _paper_cost(M, out, caps):
    sets = M['heldout'] + ([M['dev']] if M.get('dev') else [])
    name = M.get('cost_dataset') or M['heldout'][0]['name']
    s = next((x for x in sets if x['name'] == name), None)
    if s is None:
        return
    G = _group(_load_runs(s['path']))
    env = {}
    if os.path.exists(os.path.join(s['path'], 'env.json')):
        with open(os.path.join(s['path'], 'env.json')) as fh:
            env = json.load(fh)
    rows, nl, nw = [], None, None
    for m in (M.get('methods') or sorted(G)):
        rs = [r for r in G.get(m, []) if _num(r.get('latency_median_ms')) is not None]
        if not rs:
            continue
        r = rs[0]
        nl, nw = r.get('latency_n_frames'), r.get('latency_warmup')
        rows.append([m, 'yes' if r.get('training_free') else 'no', f"{int(r.get('trainable_params') or 0):,}",
                     _fmt(r.get('fit_time_s'), 1), _fmt(r.get('latency_median_ms'), 1),
                     _fmt(r.get('latency_p95_ms'), 1), _fmt(r.get('backbone_ms'), 1),
                     _fmt(r.get('search_ms'), 1), _fmt(r.get('patch_score_ms'), 1),
                     _fmt(r.get('bank_mb'), 1), {True: 'yes', False: 'no'}.get(r.get('bank_on_gpu'), '—')])
    if rows:
        prec = (env.get('feature_spec') or {}).get('precision', '?')
        cap = (f'Per-frame cost on {name} ({env.get("gpu", "GPU")}, PyTorch {env.get("torch", "?")}, '
               f'batch 1, {prec} backbone, synchronised wall clock over {nl} fixed frames after {nw} '
               f'warm-up frames). Stage times come from a separate profiled pass. Memory: exemplar '
               f'bank, Gaussian parameters (PaDiM) or network parameters (SimpleNet).')
        _write(out, 'table6_cost', ['Method', 'Training-free', 'Trainable params', 'Fit (s)',
                                    'Median (ms)', 'p95 (ms)', 'Backbone (ms)', 'Search (ms)',
                                    'Patch (ms)', 'Memory (MB)', 'Bank on GPU'], rows, cap, 'tab:cost')
        caps.append(('Table 6', cap))
    data = {}
    for key in ('N', 'K', 'resolution'):
        p = os.path.join(s['path'], 'scaling', f'sweep_{key}.csv')
        if os.path.exists(p):
            with open(p, newline='') as fh:
                data[key] = list(csv.DictReader(fh))
    if not data:
        return
    plt = _plt()
    panels = [('N', 'N_effective', 'latency_median_ms', 'Training frames N', True),
              ('K', 'K', 'patch_score_ms', 'Retrieved frames K', False),
              ('resolution', 'P', 'latency_median_ms', 'Patches per frame P', True)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), squeeze=False)
    for c, (key, xk, yk, xl, logx) in enumerate(panels):
        rows_ = data.get(key, [])
        for mb in sorted({r['method_base'] for r in rows_}):
            pts = sorted((_num(r[xk]), _num(r.get(yk)), _num(r.get('auroc'))) for r in rows_
                         if r['method_base'] == mb and _num(r.get(xk)) is not None)
            for row, j in ((0, 1), (1, 2)):
                q = [(p[0], p[j]) for p in pts if p[j] is not None]
                if q:
                    axes[row][c].plot(*zip(*q), 'o-', label=mb)
        for row, yl in ((0, yk.replace('_', ' ')), (1, 'Frame AUROC')):
            a = axes[row][c]
            a.set_xlabel(xl)
            a.set_ylabel(yl)
            a.grid(alpha=.3)
            if logx:
                a.set_xscale('log')
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out, 'fig_scaling')
    plt.close(fig)
    caps.append(('Figure (scaling)', f'Cost (top) and frame AUROC (bottom) on {name} as functions '
                 'of the number of training frames N, retrieved frames K and patches per frame P.'))


def _paper_failures(M, out, caps):
    rows, gap = [], None
    for s in M['heldout']:
        p = os.path.join(s['path'], 'failures', 'failure_events.csv')
        if os.path.exists(p):
            with open(p, newline='') as fh:
                rows += [dict(r, dataset=s['name']) for r in csv.DictReader(fh)]
    rows = [r for r in rows if r['type'] in ('FN', 'FP')]
    if not rows:
        return
    unc = sum(not r['category'].strip() for r in rows)
    tot = {t: [r for r in rows if r['type'] == t] for t in ('FN', 'FP')}
    tab = []
    for code, desc in list(FAIL_CATEGORIES.items()) + [('', 'Uncategorised')]:
        line = [code or '—', desc]
        for t in ('FN', 'FP'):
            ev = [r for r in tot[t] if r['category'].strip().upper() == code]
            fr = sum(int(r['n_frames']) for r in ev)
            line += [f'{len(ev)} ({100 * len(ev) / max(len(tot[t]), 1):.0f}%)', str(fr)]
        tab.append(line)
    cap = ('Categorised error events of the Proposed method at the 5% FPR operating point on held-out '
           'datasets. An event is a maximal run of misclassified frames of one type (short gaps '
           f'merged); categories assigned by manual inspection ({unc} event(s) uncategorised).')
    _write(out, 'table_failures', ['Code', 'Category', 'FN events', 'FN frames', 'FP events', 'FP frames'],
           tab, cap, 'tab:failures', 'llcccc')
    caps.append(('Table (failures)', cap))
    caps.append(('Figures (examples)', 'Qualitative examples (failures/fig_examples_{TP,FN,FP}): input '
                 'with ground-truth contour (green) where available, anomaly map on a shared colour '
                 'scale, and the three nearest training frames retrieved by the CLS key.'))


def cmd_paper(args):
    if args.init or not os.path.exists(args.manifest):
        if os.path.exists(args.manifest):
            print(f'[paper] {args.manifest} exists; not overwritten')
        else:
            with open(args.manifest, 'w') as fh:
                json.dump(PAPER_TEMPLATE, fh, indent=2)
            print(f'[paper] template written to {args.manifest}; edit it and rerun `paper`')
        return
    with open(args.manifest) as fh:
        M = json.load(fh)
    out = M.get('out_dir', 'paper')
    os.makedirs(out, exist_ok=True)
    caps = []
    _paper_table2(M, out, caps)
    _paper_pervideo(M, out)
    _paper_selection(M, out, caps)
    _paper_table3(M, out, caps)
    _paper_sensitivity(M, out, caps)
    _paper_cost(M, out, caps)
    _paper_failures(M, out, caps)
    for s in M['heldout'] + _devs(M):
        p = os.path.join(s['path'], 'overlays', 'overlay_meta.json')
        if os.path.exists(p):
            with open(p) as fh:
                o = json.load(fh)
            caps.append((f'Figure (overlays, {s["name"]})',
                         f'Anomaly-map overlays of the Proposed method on {s["name"]} '
                         f'({", ".join(g.replace("_", " ") for g in o["groups"])}; {o["n"]} frames each). '
                         f'Frames were selected automatically by frame score, at most {o["per_video"]} per '
                         f'video and at least {o["min_gap"]} frames apart. Green: ground-truth regions. '
                         f'One colour scale for all panels ({o["vrange_percentiles"][0]:g}th–'
                         f'{o["vrange_percentiles"][1]:g}th percentile of the shown maps)'
                         + (f'; display smoothing σ = {o["display_sigma"]:g} px' if o['display_sigma'] else '')
                         + '. Files: overlays/fig_overlay_*.'))
    with open(os.path.join(out, 'captions.md'), 'w') as fh:
        fh.write('\n'.join(f'**{k}.** {v}\n' for k, v in caps))
    print(f'[paper] done: {out}/ (tables .md/.tex/.csv, figures .pdf/.png, captions.md)')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 · ENTRY POINTS
# ══════════════════════════════════════════════════════════════════════════════

def setup(args, load_config: bool = True):
    set_seed(42)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
        print(f'[main] device {device} ({torch.cuda.get_device_name(device)})')
    else:
        device = torch.device('cpu')
        print('[WARN] CUDA not available; running on CPU')
    cfg = BenchConfig(data_path=args.data_path, save_path=args.save_path)   # all else: class defaults
    cfg.train_stride = max(1, int(args.train_stride))
    fspec = FeatureSpec()                        # 518 px, fp16, batch 16, ImageNet normalisation
    spec = MethodSpec()                          # a-priori defaults; replaced by --config
    if load_config and args.config:
        spec = _apply_config(args.config, cfg, fspec, spec)
    teacher = DINOv2AllLayers(fspec.backbone).to(device)
    print(f'[main] FeatureSpec {fspec.key()}: {fspec}\n[main] MethodSpec  {spec.key()}: {spec}')
    print(f'[main] train_stride={cfg.train_stride} (aerial datasets; ShanghaiTech uses '
          f'SHT_TRAIN_STRIDE={SHT_TRAIN_STRIDE})')
    print(f'[main] {"FROZEN configuration: " + cfg.frozen_config if cfg.frozen_config else "NOTE: no --config; a-priori defaults, NOT the frozen selection"}')
    return cfg, spec, fspec, device, teacher


def cmd_cache(args):
    """Extract and cache features (train + every test video) without scoring anything."""
    cfg, spec, fspec, device, teacher = setup(args, load_config=False)
    ctx = Context(cfg, fspec, teacher, device, (spec.patch_layer,))
    ctx.features('train', [spec.patch_layer])
    for sc in ctx.scenes:
        ctx.features(sc.name, [spec.patch_layer])
    print(f'[cache] done: {ctx.cache_root}')


_SEEDED = {'proposed', 'simplenet'}


def cmd_bench(args):
    cfg, spec, fspec, device, teacher = setup(args)
    cfg.methods = tuple(args.method)
    cfg.ablation_grid, cfg.sweeps, cfg.no_cache = args.ablation_grid, args.sweeps, args.no_cache
    cfg.gpu_bank_budget_mb = args.bank_budget_mb
    os.makedirs(cfg.save_path, exist_ok=True)
    write_env(cfg.save_path, device, fspec, cfg)
    ctx = Context(cfg, fspec, teacher, device, (spec.patch_layer,))
    if ctx.mask_root:
        selftest_pixel_metrics(cfg.pixel_max_fpr)
    seeds = args.seeds or [spec.fps_seed]
    lat0, results = cfg.measure_latency, []
    for name in cfg.methods:
        for si, sd in enumerate(seeds):
            if si > 0 and name not in _SEEDED:
                continue
            print('\n' + '─' * 60 + f'\n  METHOD: {name}  seed {sd}\n' + '─' * 60)
            s_spec = replace(spec, fps_seed=sd) if name == 'proposed' else spec
            m = build_method(name, ctx, cfg, s_spec, **({'seed': sd} if name == 'simplenet' else {}))
            cfg.measure_latency = lat0 and si == 0
            r, res = evaluate_method(m, ctx, cfg)
            cfg.measure_latency = lat0
            r['seed'] = sd if name in _SEEDED else None
            results.append(r)
            _print_row(r)
            if cfg.save_runs:
                save_run(cfg.save_path, 'main', r, res['scores'], ctx)
            if name == 'proposed' and sd == spec.fps_seed:
                p = os.path.join(cfg.save_path, 'proposed_scores.npy')
                dump_run(p, m.scorer, fspec.key(), res['scores'], ctx.labels)
                print(f'[saved] {p}  (equivalence reference for `ablate`)')
            del m
            _free(device)
    if cfg.cnn_backbone:
        ex = CNNPatches(cfg.cnn_name, device, cfg.cnn_out, cfg.cnn_grid)
        for name, make in (('padim', lambda: PaDiM(ex, cfg, 'PaDiM (WRN-50)')),
                           ('patchcore', lambda: PatchCore(ex, cfg, device, 'PatchCore (WRN-50)')),
                           ('simplenet', lambda: SimpleNet(ex, cfg, device, 'SimpleNet (WRN-50)'))):
            if name in cfg.methods:
                m = make()
                m.backbone = 'WideResNet-50'
                r, res = evaluate_method(m, ctx, cfg)
                results.append(r)
                _print_row(r)
                if cfg.save_runs:
                    save_run(cfg.save_path, 'main', r, res['scores'], ctx)
                del m
                _free(device)
    print_results(results, f'TABLE — Accuracy and per-frame cost ({_dataset_tag(cfg)}, {fspec.image_size}px)',
                  os.path.join(cfg.save_path, 'benchmark_results.csv'))
    if cfg.ablation_grid:
        run_bench_grid(ctx, cfg, spec, N_BOOT)
    if cfg.sweeps:
        run_sweeps(ctx, cfg, spec)


def cmd_ablate(args):
    cfg, spec, fspec, device, teacher = setup(args)
    ctx = Context(cfg, fspec, teacher, device, set(ABLATION_PATCH_LAYERS) | {spec.patch_layer})
    out = os.path.join(cfg.save_path, 'ablation')
    ab = Ablation(ctx, cfg, spec, out, N_BOOT, BLOCK_LEN, A4_REPEATS)
    mine = os.path.join(out, 'reference_scores.npy')
    ref_scorer = ProposedScorer(spec, device)
    ref_scorer.idx = ab.ref['bank_idx']
    dump_run(mine, ref_scorer, fspec.key(), ab.ref['scores'], ctx.labels)
    bench = os.path.join(cfg.save_path, 'proposed_scores.npy')
    if os.path.exists(bench):
        b = np.load(bench, allow_pickle=True).item()
        if b['mkey'] == spec.key() and b['fkey'] == fspec.key():
            assert_equivalent(bench, mine)
        else:
            msg = ('bench scores were produced with a different MethodSpec / FeatureSpec; '
                   're-run `bench` with the same arguments')
            raise RuntimeError(msg)
    else:
        print('[gate] no benchmark scores yet; run `bench --method proposed` with the same arguments')
    axes = {'A1': lambda: ab.a1(args.a1_own_fps), 'A2': ab.a2, 'A4': ab.a4,
            'A5': ab.a5, 'A6': ab.a6, 'A7': ab.a7}
    for a in args.axes:
        axes[a]()
    print(f'\nDone. Results in {out}')


def cmd_verify(args):
    assert_equivalent(args.a, args.b, VERIFY_ATOL)


def cmd_report(args):
    d = os.path.join(args.save_path, 'ablation')
    files = sorted(glob.glob(os.path.join(d, 'A*.json')))
    if not files:
        print(f'no ablation results in {d}')
        return
    for p in files:
        with open(p) as fh:
            r = json.load(fh)
        if r.get('code_version') != CODE_VERSION and not args.force:
            print(f'[STALE] {os.path.basename(p)}: written by {r.get("code_version")}; re-run it')
            continue
        print(f'\n── {r["axis"]}: {r["title"]}')
        print_table(r['table'], r['headers'])
        for n in r.get('notes', []):
            print('  ' + n)


def _add_bool(p: argparse.ArgumentParser, name: str, default: bool, help: str = None):
    """--name / --no-name pair (argparse.BooleanOptionalAction needs Python >= 3.9)."""
    flag = name.lstrip('-')
    dest = flag.replace('-', '_')
    p.add_argument(f'--{flag}', dest=dest, action='store_true', default=default,
                   help=f'{help or ""} (default: {default})')
    p.add_argument(f'--no-{flag}', dest=dest, action='store_false', default=default,
                   help=argparse.SUPPRESS)


# ── Fixed analysis protocol (deliberately NOT command-line options) ──────────
FAIL_FPR    = 0.05     # operating point of the failure analysis (5% FPR)
FAIL_GAP    = 5        # error frames <= 5 apart are merged into one event
N_BOOT      = 1000     # bootstrap resamples (delete if statistics were removed)
BLOCK_LEN   = 0        # 0 = automatic block length from the score ACF (idem)
A4_REPEATS  = 5        # runs per stochastic subsampling strategy in A4
VERIFY_ATOL = 1e-6     # equivalence tolerance between dumped runs

def main():
    methods = ['proposed', 'padim', 'patchcore', 'simplenet', 'anomalydino', 'spade']
    axes = ['A1', 'A2', 'A4', 'A5', 'A6', 'A7']

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--data_path', default=BenchConfig.data_path)
    common.add_argument('--save_path', default=BenchConfig.save_path)
    common.add_argument('--config', default=None, help='frozen selected_config.json from `select`')
    common.add_argument('--gpu', type=int, default=7)
    common.add_argument('--train_stride', type=int, default=BenchConfig.train_stride,
                        help='keep every n-th TRAINING frame per video (aerial datasets)')

    ap = argparse.ArgumentParser(description='One-class aerial anomaly detection: selection, '
                                             'benchmark, ablation and article material')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('cache', parents=[common],
                   help='feature extraction only (no scoring)').set_defaults(fn=cmd_cache)

    s = sub.add_parser('select', parents=[common], help='hyperparameter selection (development data)')
    s.add_argument('--method', nargs='+', default=methods, choices=methods)
    s.add_argument('--dev_extra', nargs='+', default=[], help='additional development datasets')
    s.add_argument('--config_out', default=None,
                   help='output file (default: {save_path}/selected_config.json)')
    s.set_defaults(fn=cmd_select)

    b = sub.add_parser('bench', parents=[common], help='method comparison + latency')
    b.add_argument('--method', nargs='+', default=['proposed'], choices=methods)
    b.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='seeds for stochastic methods (Proposed FPS start, SimpleNet)')
    b.add_argument('--cache', dest='no_cache', action='store_false', default=True,
                   help='use the on-disk feature cache for the Proposed method')
    b.add_argument('--ablation_grid', action='store_true')
    b.add_argument('--sweeps', action='store_true')
    b.add_argument('--bank_budget_mb', type=float, default=BenchConfig.gpu_bank_budget_mb,
                   help='GPU memory for exemplar banks (raise for latency runs, one job per GPU)')
    b.set_defaults(fn=cmd_bench)

    sub.add_parser('sensitivity', parents=[common],
                   help='one-at-a-time sensitivity').set_defaults(fn=cmd_sensitivity)

    fa = sub.add_parser('failures', parents=[common], help='error events and figures')
    fa.add_argument('--n_fig', type=int, default=4, help='FN and FP examples per figure')
    fa.add_argument('--n_tp', type=int, default=2, help='TP examples per figure')
    fa.add_argument('--k_show', type=int, default=3, help='retrieved training frames shown')
    fa.add_argument('--fig_width', type=float, default=None,
                help='compact figure at this printed width in inches (e.g. the journal column width)')
    fa.set_defaults(fn=cmd_failures)

    a = sub.add_parser('ablate', parents=[common], help='ablation study of the Proposed method')
    a.add_argument('--axes', nargs='+', default=axes, choices=axes)
    a.add_argument('--a1_own_fps', action='store_true',
                   help='supplementary check: each A1 condition selects its own bank')
    a.set_defaults(fn=cmd_ablate)

    ov = sub.add_parser('overlays', parents=[common], help='heat-map overlay figures')
    ov.add_argument('--n', type=int, default=10, help='frames per group')
    ov.add_argument('--mode', choices=['best', 'worst', 'both'], default='both')
    ov.add_argument('--min_gap', type=int, default=30, help='min. frame distance within a video')
    ov.add_argument('--per_video', type=int, default=3, help='max. frames per video and group')
    ov.add_argument('--ncol', type=int, default=5)
    ov.add_argument('--cmap', default='jet')
    ov.add_argument('--sigma', type=float, default=4.0, help='display smoothing (px); state it')
    ov.add_argument('--vrange', type=float, nargs=2, default=[1.0, 99.5])
    ov.add_argument('--no_input', action='store_true')
    ov.add_argument('--compare', nargs='+', default=None,
                    choices=['patchcore', 'anomalydino', 'spade', 'padim', 'simplenet'])
    ov.add_argument('--compare_groups', default='best_anomalous:3,worst_anomalous:1,'
                                                'best_normal:1,worst_normal:1')
    ov.set_defaults(fn=cmd_overlays)

    v = sub.add_parser('verify', help='assert two dumped Proposed runs are identical')
    v.add_argument('a')
    v.add_argument('b')
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser('report', help='reprint saved ablation tables')
    r.add_argument('--save_path', default=BenchConfig.save_path)
    r.add_argument('--force', action='store_true')
    r.set_defaults(fn=cmd_report)

    pa = sub.add_parser('paper', help='build article tables, figures and captions')
    pa.add_argument('--manifest', default='paper_manifest.json')
    pa.add_argument('--init', action='store_true')
    pa.set_defaults(fn=cmd_paper)

    args = ap.parse_args()
    args.fn(args)

if __name__ == '__main__':
    main()
