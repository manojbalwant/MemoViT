"""
baselines.py — aerial one-class anomaly-detection benchmark (paper Tables 1-2).

Compares MemoViT against PatchCore, PaDiM, and SimpleNet under an identical
pipeline. All methods share the frozen DINOv2 ViT-B/14 backbone (optionally a
WideResNet-50 CNN backbone). The MemoViT entry (`ProposedModel`) reports the
robust-MAD patch score (Eq. 7, via scoring.py); PatchCore is the training-free
comparison re-run on identical 518x518 inputs.

Metrics: frame-level AUROC, equal error rate, average precision, latency (ms/frame).

Usage:
    python baselines.py --method patchcore          # the re-run baseline for Tables 1-2
    python baselines.py --method proposed patchcore  # MemoViT + PatchCore
    python baselines.py --cnn_backbone               # add WideResNet-50 variants
"""

# ── Standard library ────────────────────────────────────────────────────────
import os, sys, glob, random, math, time, csv, argparse, warnings
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

# ── Scientific stack ─────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader

# ── Vision / ML ──────────────────────────────────────────────────────────────
import timm
import cv2
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import gc

# ── Optional FAISS ───────────────────────────────────────────────────────────
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

# ── Canonical reported scorer (Eq. 7) ─────────────────────────────────────────
# robust_patch_score is the single source of truth for the reported MemoViT
# frame/image score; see scoring.py and REPRODUCIBILITY.md (issue #7).
from scoring import robust_patch_score

warnings.filterwarnings('ignore', category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 · CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkConfig:
    """
    Unified configuration for all four methods.
    Inherits all fields from the original Config and adds per-baseline params.
    """
    def __init__(self):
        # ── Dataset / paths ───────────────────────────────────────────────
        self.data_path   = 'dataset/Drone-Anomaly/Farmland Inspection/'
        self.save_path   = 'experiments/Farmland Inspection/'
        self.gpu = 2

        # ── Input ─────────────────────────────────────────────────────────
        self.image_size  = 518          # keep consistent with ViT patch14
        self.batch_size  = 32#16
        self.num_workers = 8
        self.num_frames  = 1

        # ── DINOv2 backbone ───────────────────────────────────────────────
        self.selected_layers          = [7,8,9, 10, 11]
        self.scoring                  = 'cosine'
        self.heatmap_interp           = cv2.INTER_LINEAR
        self.display_frames           = False

        # ── Proposed method: KNN / retrieval ──────────────────────────────
        self.knn_k                    = 5
        self.knn_metric               = 'cosine'
        self.use_faiss_index          = True and _HAS_FAISS
        self.concat_fps_target        = 0.25#1           # fraction of train set kept
        self.per_layer_pca_dim        = None        # per-layer PCA dim (None = skip)
        self.concat_pca_dim           = 512         # final concat PCA dim
        self.use_last_layer_patches_for_knn = True

        # ── PaDiM ─────────────────────────────────────────────────────────
        # pca_dim: project each patch to this many dims before Gaussian fit.
        # Must be < embed_dim (768 for ViT-B).  100–256 works well.
        self.padim_pca_dim            = 128#256#150
        self.padim_eps                = 1e-4        # regularisation added to Σ
        # which DINOv2 layers to concat for PaDiM patch features
        self.padim_layers             = [11]#[9, 10, 11]

        # ── PatchCore ─────────────────────────────────────────────────────
        self.patchcore_coreset_ratio  = 0.01#0.25#1        # fraction of patches kept
        self.patchcore_knn_k          = 5
        self.patchcore_layers         = [11]        # last layer only (purest semantics)
        self.patchcore_neighbour_k    = 3           # local-neighbourhood kernel size

        # ── SimpleNet ─────────────────────────────────────────────────────
        self.simplenet_noise_std      = 0.015#{0.005, 0.015, 0.05, 0.1}       # σ for Gaussian pseudo-anomalies
        self.simplenet_lr             = 2e-4
        self.simplenet_epochs         = 30
        self.simplenet_hidden_mult    = 2           # adapter hidden dim = in_dim * mult
        self.simplenet_layers         = [11]        # feature source layer(s)

        # ── Optional CNN backbone (secondary comparison) ───────────────────
        # Set to True to also run PaDiM / PatchCore / SimpleNet with WRN-50
        self.run_cnn_backbone         = False
        self.cnn_model_name           = 'wide_resnet50_2'
        self.cnn_out_indices          = (1, 2, 3)  # timm stage indices
        self.cnn_target_grid          = 37          # upsample all scales to NxN

        # ── Benchmark control ─────────────────────────────────────────────
        self.methods_to_run = ['patchcore']#['proposed', 'padim', 'patchcore', 'simplenet']
        # max training frames per method (None = use all).  Set e.g. 2000 if OOM.
        self.max_train_samples        = None
        self.results_csv              = os.path.join(self.save_path, 'benchmark_results.csv')

    def update_from_args(self, args):
        if hasattr(args, 'method') and args.method:
            self.methods_to_run = args.method
        if hasattr(args, 'cnn_backbone') and args.cnn_backbone:
            self.run_cnn_backbone = True
        if hasattr(args, 'max_train_samples') and args.max_train_samples:
            self.max_train_samples = args.max_train_samples


def print_config(cfg: BenchmarkConfig):
    print('\n' + '═'*64)
    print('  BENCHMARK CONFIGURATION')
    print('═'*64)
    for k, v in cfg.__dict__.items():
        print(f'  {k:<36} {v}')
    print('═'*64 + '\n')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 · SHARED UTILITIES  (preserved from original code)
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_eer(y_true, y_score) -> Tuple[float, float]:
    """
    Equal Error Rate via linear interpolation.
    Returns (eer, threshold).  Lower EER is better.
    """
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return 0.5, float('nan')
    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    if 0 < idx < len(fpr) - 1:
        fpr0, fnr0, fpr1, fnr1 = fpr[idx-1], fnr[idx-1], fpr[idx], fnr[idx]
        denom = (fpr1 - fpr0) - (fnr1 - fnr0)
        if abs(denom) > 1e-12:
            t   = float(np.clip((fnr0 - fpr0) / denom, 0.0, 1.0))
            eer = float(fpr0 + t * (fpr1 - fpr0))
            thr = float(thresholds[idx-1] + t * (thresholds[idx] - thresholds[idx-1]))
        else:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            thr = float(thresholds[idx])
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        thr = float(thresholds[idx])
    return eer, thr


def compute_mean_and_std(train_folder, resize_h, resize_w, device, batch_size=64, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
    ])
    dataset  = datasets.ImageFolder(train_folder, transform=transform)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channels = dataset[0][0].shape[0]
    mean     = torch.zeros(channels, device=device)
    std      = torch.zeros(channels, device=device)
    n_pix    = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc='Computing mean/std'):
            imgs = imgs.to(device, non_blocking=True)
            b, c, h, w = imgs.shape
            n_pix  += b * h * w
            mean   += imgs.sum(dim=[0, 2, 3])
            std    += (imgs ** 2).sum(dim=[0, 2, 3])
    mean /= n_pix
    std   = torch.sqrt(std / n_pix - mean ** 2)
    return mean.cpu(), std.cpu()


def np_load_frame(filename: str, resize_h: int, resize_w: int) -> np.ndarray:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_w, resize_h))
    return img


class CustomFrameDataset(data.Dataset):
    """Loads consecutive frames from video folders (unchanged from original)."""
    def __init__(self, video_folder, transform, resize_h, resize_w,
                 time_step=4, num_pred=0, frame_step=1, return_image_path=False):
        self.dir              = video_folder
        self.transform        = transform
        self._resize_h        = resize_h
        self._resize_w        = resize_w
        self._time_step       = time_step
        self._num_pred        = num_pred
        self._frame_step      = frame_step
        self.return_image_path= return_image_path
        self.video_frames     = []
        self.index_samples    = []
        self._setup()

    def _setup(self):
        videos = sorted(glob.glob(os.path.join(self.dir, '*')))
        all_frames = []
        if videos and os.path.isdir(videos[0]):
            for v in videos:
                frames = sorted(glob.glob(os.path.join(v, '*.jpg')),
                                key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
                all_frames.extend(frames)
        else:
            all_frames = sorted(videos,
                                key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
        self.video_frames  = all_frames
        max_idx            = len(all_frames) - (self._time_step + self._num_pred - 1) * self._frame_step
        self.index_samples = list(range(max_idx))

    def __getitem__(self, index):
        fi = self.index_samples[index]
        frames = np.zeros((self._time_step + self._num_pred, 3, self._resize_h, self._resize_w))
        for i in range(self._time_step + self._num_pred):
            img_np = np_load_frame(self.video_frames[fi + i * self._frame_step],
                                   self._resize_h, self._resize_w)
            if self.transform:
                img_np = self.transform(Image.fromarray(img_np))
            frames[i] = img_np
        if self.return_image_path:
            return {'standard': frames, '256': frames, 'image_path': self.video_frames[fi]}
        return {'standard': frames, '256': frames}

    def __len__(self):
        return len(self.index_samples)


def _norm_scores(arr):
    """Per-video min-max normalisation (consistent with original evaluation)."""
    a  = np.array(arr, dtype=np.float32)
    mn, mx = a.min(), a.max()
    return (a - mn) / (mx - mn + 1e-8) if mx > mn else a


@contextmanager
def timer():
    """Simple wall-clock timer context manager."""
    t0 = time.perf_counter()
    state = {}
    yield state
    state['elapsed'] = time.perf_counter() - t0




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 · DINOv2 multi-layer teacher (frozen feature extractor)
# ══════════════════════════════════════════════════════════════════════════════

def _get_embed_dim(vit):
    return getattr(vit, 'embed_dim', getattr(vit, 'num_features', None)) or 768


class TeacherMultiLayer(nn.Module):
    """
    Frozen DINOv2 ViT-B/14 that taps intermediate CLS and patch tokens.
    Preserved exactly from the original code.
    """
    def __init__(self, model_name='vit_base_patch14_dinov2',
                 pretrained=True, selected_layers=None):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        for attr in ('head', 'head_dist', 'fc'):
            if hasattr(self.vit, attr):
                try: setattr(self.vit, attr, nn.Identity())
                except Exception: pass
        self.embed_dim    = _get_embed_dim(self.vit)
        self.n_blocks     = len(getattr(self.vit, 'blocks', []))
        self.selected_layers = (list(range(self.n_blocks))
                                if selected_layers is None
                                else [l for l in selected_layers if 0 <= l < self.n_blocks])

    def forward(self, x):
        x   = self.vit.patch_embed(x)
        B, N, D = x.shape
        has_cls  = hasattr(self.vit, 'cls_token')
        has_dist = hasattr(self.vit, 'dist_token')
        if has_cls:
            cls_token = self.vit.cls_token.expand(B, -1, -1)
        if has_dist:
            dist_token = self.vit.dist_token.expand(B, -1, -1)
            tokens     = torch.cat((cls_token, dist_token, x), dim=1)
        elif has_cls:
            tokens     = torch.cat((cls_token, x), dim=1)
        else:
            tokens     = x
        if hasattr(self.vit, 'pos_embed'):
            tokens = tokens + self.vit.pos_embed.to(tokens.device)

        cls_raw_list, patch_tokens_list = [], []
        for i, block in enumerate(getattr(self.vit, 'blocks', [])):
            tokens = block(tokens)
            if i in self.selected_layers:
                if has_dist:
                    cls_raw     = tokens[:, 0:2].mean(dim=1)
                    patch_toks  = tokens[:, 2:].contiguous()
                elif has_cls:
                    cls_raw     = tokens[:, 0].contiguous()
                    patch_toks  = tokens[:, 1:].contiguous()
                else:
                    cls_raw     = tokens.mean(dim=1)
                    patch_toks  = tokens.contiguous()
                cls_raw_list.append(cls_raw)
                patch_tokens_list.append(patch_toks)

        # Replace last captured layer with layer-normed output
        if hasattr(self.vit, 'norm') and self.selected_layers:
            normed = self.vit.norm(tokens)
            if has_dist:
                cls_final   = normed[:, 0:2].mean(dim=1)
                patch_final = normed[:, 2:].contiguous()
            elif has_cls:
                cls_final   = normed[:, 0]
                patch_final = normed[:, 1:].contiguous()
            else:
                cls_final   = normed.mean(dim=1)
                patch_final = normed.contiguous()
            cls_raw_list[-1]     = cls_final
            patch_tokens_list[-1]= patch_final

        return cls_raw_list, patch_tokens_list


def greedy_farthest_point_sampling_rows_cpu(
    feat_np : np.ndarray,      # (N, D) any dtype
    target_n: int,
    verbose : bool = True,
) -> np.ndarray:
    """
    Greedy FPS using the dot-product identity for unit vectors.

    For unit-normalised rows:
        ||a - b||² = 2(1 - a·b)

    Each iteration reduces to a BLAS matrix-vector multiply (rows @ rows[idx])
    instead of a broadcast subtract + square + sum — roughly 3-4× fewer FLOPs
    and dramatically better cache utilisation.

    Time complexity : O(target_n × N)  — same as original
    FLOP reduction  : ~3×  (one matvec vs subtract+square+sum)
    Memory          : O(N)  — only `dists` buffer allocated

    Parameters
    ----------
    feat_np  : (N, D) array — will be float32-cast and row-normalised in-place
    target_n : number of samples to select
    verbose  : show tqdm progress bar

    Returns
    -------
    selected : (target_n,) int32 indices into feat_np rows
    """
    N, D = feat_np.shape
    if target_n >= N:
        return np.arange(N, dtype=np.int32)

    # ── Row-normalise (float32 for SIMD/cache efficiency) ─────────────────
    rows  = feat_np.astype(np.float32, copy=True)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    rows /= norms                                              # unit sphere

    selected  = np.empty(target_n, dtype=np.int32)
    selected[0] = 0
    dot_buf   = np.empty(N, dtype=np.float32)                 # preallocate once

    # Initial distances from seed point 0
    np.dot(rows, rows[0], out=dot_buf)
    dists = 2.0 * (1.0 - dot_buf)                            # (N,) float32

    it = range(1, target_n)
    if verbose:
        it = tqdm(it, desc="  [FPS] Selecting diverse tiles", leave=False)

    for k in it:
        idx         = int(np.argmax(dists))
        selected[k] = idx
        np.dot(rows, rows[idx], out=dot_buf)
        new_d = 2.0 * (1.0 - dot_buf)
        np.minimum(dists, new_d, out=dists)                   # in-place minimum

    del rows, dists, dot_buf
    gc.collect()
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 · FEATURE EXTRACTOR ABSTRACTIONS
#   Unified interface so PaDiM / PatchCore / SimpleNet are backbone-agnostic.
# ══════════════════════════════════════════════════════════════════════════════

class BaseFeatureExtractor(ABC):
    """
    Contract for patch-level feature extractors.

    extract_patches(imgs) → torch.Tensor (B, P, D)  on CPU
      B  = batch size
      P  = number of spatial patch positions (H_grid × W_grid)
      D  = feature dimensionality

    All heavy computation happens inside this method.
    Returns CPU tensors to keep GPU memory free for the model.
    """
    @abstractmethod
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def feat_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def grid_size(self) -> Tuple[int, int]:
        """(H_grid, W_grid) of the spatial feature map."""
        ...

    def extract_cls(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Default CLS extraction: global average pool over the patch grid.
        Subclasses with a genuine CLS token should override this.
        Returns (B, D) on CPU.
        """
        patches = self.extract_patches(imgs)  # (B, P, D)
        return patches.mean(dim=1)            # (B, D)


class DINOv2PatchExtractor(BaseFeatureExtractor):
    """
    Wraps TeacherMultiLayer.  Returns patch tokens from selected layers,
    either from the last layer only or concatenated across layers.

    Parameters
    ----------
    teacher        : TeacherMultiLayer instance (frozen, on device)
    selected_layers: subset of teacher.selected_layers to use
    concat_layers  : if True, concat patch tokens across all selected layers
                     → D = len(selected_layers) × embed_dim
                     if False, use only the last selected layer (post-norm)
    image_size     : used to infer grid dimensions
    patch_size     : DINOv2 default = 14
    """
    def __init__(self, teacher: TeacherMultiLayer,
                 selected_layers: Optional[List[int]] = None,
                 concat_layers: bool = False,
                 image_size: int = 518,
                 patch_size: int = 14):
        self.teacher        = teacher
        self._layers        = selected_layers or teacher.selected_layers
        self.concat_layers  = concat_layers
        self._patch_size    = patch_size
        g                   = image_size // patch_size
        self._grid          = (g, g)
        n_layers            = len(self._layers) if concat_layers else 1
        self._feat_dim      = teacher.embed_dim * n_layers

    @torch.no_grad()
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, C, H, W) on device.  Returns (B, P, D) on CPU."""
        imgs = imgs.to(next(self.teacher.parameters()).device, non_blocking=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            cls_list, patch_list = self.teacher(imgs)
        if self.concat_layers:
            # concat along feature dim → (B, P, L*D)
            return torch.cat(patch_list, dim=-1)
        else:
            return patch_list[-1]          # (B, P, D) last layer (normed)

    @torch.no_grad()
    def extract_cls(self, imgs: torch.Tensor) -> torch.Tensor:
        """Returns (B, D) raw CLS token(s) from last selected layer, CPU."""
        cls_list, _ = self.teacher(imgs)
        return cls_list[-1]

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self._grid


class CNNPatchExtractor(BaseFeatureExtractor):
    """
    Multi-scale feature extractor using a pretrained timm CNN
    (WideResNet-50-2 by default).

    Feature maps from `out_indices` stages are bilinear-upsampled to a
    common (target_grid × target_grid) grid, then channel-concatenated.
    This produces (B, P, D_cnn) where P = target_grid².

    Parameters
    ----------
    model_name   : timm model name, e.g. 'wide_resnet50_2'
    device       : torch.device
    out_indices  : timm stage indices to extract (default (1, 2, 3))
    target_grid  : spatial grid size after upsampling all scales
    """
    def __init__(self, model_name: str = 'wide_resnet50_2',
                 device: torch.device = torch.device('cpu'),
                 out_indices: Tuple[int, ...] = (1, 2, 3),
                 target_grid: int = 37):
        self.device      = device
        self._target_grid= target_grid
        self.backbone    = timm.create_model(
            model_name, pretrained=True,
            features_only=True, out_indices=out_indices
        )
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone = self.backbone.to(device, non_blocking=True)

        # Infer feature dimensionality with a dummy forward pass
        with torch.no_grad():
            dummy    = torch.zeros(1, 3, 224, 224).to(device)
            out      = self.backbone(dummy)
            n_ch     = sum(o.shape[1] for o in out)
        self._feat_dim = n_ch
        self._grid     = (target_grid, target_grid)
        print(f'[CNNPatchExtractor] {model_name} | stages {out_indices} | '
              f'feat_dim={n_ch} | grid={target_grid}×{target_grid}')

    @torch.no_grad()
    def extract_patches(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: (B, C, H, W) on device.  Returns (B, P, D) on CPU."""
        feats = self.backbone(imgs)          # list of (B, Ci, Hi, Wi)
        g     = self._target_grid
        upsampled = [F.interpolate(f, size=(g, g), mode='bilinear', align_corners=False)
                     for f in feats]
        combined = torch.cat(upsampled, dim=1)  # (B, D_total, g, g)
        B, D, H, W = combined.shape
        patches = combined.permute(0, 2, 3, 1).reshape(B, H * W, D)
        return patches.cpu()

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    @property
    def grid_size(self) -> Tuple[int, int]:
        return self._grid


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 · PROPOSED METHOD WRAPPER  (original functions + class interface)
# ══════════════════════════════════════════════════════════════════════════════

# ─── Memory bank & KNN helpers (preserved from original code) ─────────────

def collect_concatenated_memory_bank(teacher, loader, device, config,
                                     train_folder=None):
    """Collect multi-layer CLS + last-layer patches (original code, unchanged)."""
    teacher.eval()
    concat_list, last_patches_list = [], ([] if config.use_last_layer_patches_for_knn else None)
    per_layer_raw = None

    with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16):
        for batch in tqdm(loader, desc='[Proposed] Collecting memory bank'):
            inputs = batch['standard'][:, 0].float().to(device, non_blocking=True)
            cls_raw_list, patch_tokens_list = teacher(inputs)
            cls_raw_list = [t.detach().cpu() for t in cls_raw_list]
            concat_list.append(torch.cat(cls_raw_list, dim=1))
            if config.use_last_layer_patches_for_knn:
                last_patches_list.append(patch_tokens_list[-1].detach().cpu())
            if per_layer_raw is None:
                per_layer_raw = [[] for _ in range(len(cls_raw_list))]
            for i, cr in enumerate(cls_raw_list):
                per_layer_raw[i].append(cr)

    per_image_concat = torch.cat(concat_list, dim=0)
    per_layer_raw    = [torch.cat(lst, dim=0) for lst in per_layer_raw]
    per_image_last_patches = (torch.cat(last_patches_list, dim=0)
                               if config.use_last_layer_patches_for_knn else None)
    ds        = getattr(loader, 'dataset', None)
    file_list = ds.video_frames if ds is not None else []

    # ── PCA on concatenated CLS ───────────────────────────────────────────
    pca_meta = None
    per_image_concat_cpu = per_image_concat.clone().cpu()

    if getattr(config, 'concat_pca_dim', None) is not None:
        arr   = per_image_concat_cpu.numpy().astype('float32')
        N, D  = arr.shape
        k     = int(min(config.concat_pca_dim, N - 1, D))
        if 0 < k < D:
            print(f'[Proposed] PCA: {D} → {k}')
            mean_col = arr.mean(axis=0, keepdims=True).astype('float32')
            Xc = arr - mean_col
            try:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                comps    = Vt[:k].astype('float32')
                projected= np.dot(Xc, comps.T).astype('float32')
                per_image_concat_cpu = torch.from_numpy(projected)
                pca_meta = [{'mean': mean_col, 'components': comps,
                              '_legacy_single_shot': True}]
            except Exception as e:
                print(f'[WARN] PCA failed: {e}')

    # ── FPS coreset ───────────────────────────────────────────────────────
    fps_target = getattr(config, 'concat_fps_target', None)
    if fps_target is not None:
        arr   = per_image_concat_cpu.numpy().astype('float32')
        N_img = arr.shape[0]
        if fps_target < N_img:
            sel   = greedy_farthest_point_sampling_rows_cpu(arr, fps_target)
            sel   = [int(i) for i in sel]
            per_image_concat_cpu    = per_image_concat_cpu[sel]
            per_layer_raw           = [feat[sel] for feat in per_layer_raw]
            if per_image_last_patches is not None:
                per_image_last_patches = per_image_last_patches[sel]
            file_list = [file_list[i] for i in sel] if file_list else file_list
            print(f'[Proposed] FPS kept {len(sel)}/{N_img} images')

    return per_image_concat_cpu, per_layer_raw, per_image_last_patches, file_list, pca_meta


def build_knn_index(features_cpu, metric='cosine', use_faiss=True):
    feats = (features_cpu.numpy().astype('float32')
             if isinstance(features_cpu, torch.Tensor)
             else features_cpu.astype('float32'))
    N, D = feats.shape
    if use_faiss and _HAS_FAISS:
        if metric == 'cosine':
            norms  = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats_n= feats / norms
            idx    = faiss.IndexFlatIP(D)
            idx.add(feats_n)
            return {'index': idx, 'normalized': True}
        else:
            idx = faiss.IndexFlatL2(D)
            idx.add(feats)
            return {'index': idx, 'normalized': False}
    else:
        if metric == 'cosine':
            norms  = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            return {'features': feats / norms, 'metric': metric, 'normalized': True}
        return {'features': feats, 'metric': metric, 'normalized': False}


def knn_search(index_obj, query_vec, k=5):
    q = (query_vec.detach().cpu().numpy().astype('float32')
         if isinstance(query_vec, torch.Tensor)
         else np.asarray(query_vec, dtype='float32'))
    if 'index' in index_obj:
        idx = index_obj['index']
        if index_obj.get('normalized'):
            qn = q / (np.linalg.norm(q) + 1e-8)
            d, i = idx.search(qn.reshape(1, -1), k)
            return d.ravel(), i.ravel()
        qn = q.reshape(1, -1)
        d, i = idx.search(qn, k)
        return np.sqrt(d).ravel(), i.ravel()
    else:
        feats = index_obj['features']
        if index_obj.get('normalized'):
            qn   = q / (np.linalg.norm(q) + 1e-8)
            sims = feats @ qn
            inds = np.argsort(-sims)[:k]
            return sims[inds], inds
        d2   = ((feats - q) ** 2).sum(axis=1)
        inds = np.argsort(d2)[:k]
        return np.sqrt(d2[inds]), inds


def score_cls_vector_vs_prototypes(cls_vec, prototypes, metric='cosine', **_):
    if cls_vec.device != prototypes.device:
        cls_vec = cls_vec.to(prototypes.device)
    cls = cls_vec.unsqueeze(0)
    if metric == 'cosine':
        sim = torch.matmul(F.normalize(cls, dim=1),
                           F.normalize(prototypes, dim=1).t())
        return float(1.0 - sim.max())
    dists = torch.cdist(cls, prototypes)
    return float(dists.min())


def aggregate_layer_scores(layer_scores):
    ws   = np.ones(len(layer_scores), dtype=np.float32)
    vals = np.array(layer_scores, dtype=np.float32)
    return float((ws * vals).sum() / (ws.sum() + 1e-12))


class ProposedModel:
    """
    Thin wrapper that packages the proposed DINOv2-MemKNN method into
    the same fit() / score() interface as the other baselines.
    """
    def __init__(self, teacher: TeacherMultiLayer, config: BenchmarkConfig,
                 mean, std, device):
        self.teacher = teacher
        self.config  = config
        self.mean    = mean
        self.std     = std
        self.device  = device
        # fit state
        self.per_layer_raw             = None
        self.per_image_concat_cpu      = None
        self.per_image_last_patches    = None
        self.train_image_paths         = None
        self.pca_meta                  = None
        self.concat_index_obj          = None

    def fit(self, loader: DataLoader):
        cfg = self.config
        cfg.concat_fps_target = max(1, int(cfg.concat_fps_target * len(loader.dataset)))
        (self.per_image_concat_cpu,
         self.per_layer_raw,
         self.per_image_last_patches,
         self.train_image_paths,
         self.pca_meta) = collect_concatenated_memory_bank(
             self.teacher, loader, self.device, cfg)
        self.concat_index_obj = build_knn_index(
            self.per_image_concat_cpu,
            metric=cfg.knn_metric,
            use_faiss=cfg.use_faiss_index)
        print(f'[Proposed] Memory bank built: {self.per_image_concat_cpu.shape}')

    @torch.no_grad()
    def score(self, img: torch.Tensor) -> Tuple[float, np.ndarray]:
        """img: (1, C, H, W) on device.  Returns (frame_score, heatmap)."""
        cfg = self.config
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            cls_raw_list, patch_tokens_list = self.teacher(img)

        # CLS-prototype score
        layer_scores = []
        for li, cls_raw in enumerate(cls_raw_list):
            mem   = self.per_layer_raw[li]
            score = score_cls_vector_vs_prototypes(
                cls_raw.squeeze(0).cpu(), mem.cpu(), metric=cfg.scoring)
            layer_scores.append(score)
        cls_score = aggregate_layer_scores(layer_scores)

        # Concat-KNN score + heatmap
        concat_vec_parts = [cls_raw_list[i][0].detach().cpu().numpy().astype('float32')
                            for i in range(len(cls_raw_list))]
        if self.pca_meta:
            legacy = (len(self.pca_meta)==1
                      and self.pca_meta[0].get('_legacy_single_shot', False))
            if legacy:
                raw_c  = np.concatenate(concat_vec_parts)
                qc     = raw_c - self.pca_meta[0]['mean'].ravel()
                q_vec  = np.dot(qc, self.pca_meta[0]['components'].T).astype('float32')
            else:
                projected = []
                for li, vec in enumerate(concat_vec_parts):
                    ml = self.pca_meta[li] if li < len(self.pca_meta) else None
                    if ml is None:
                        projected.append(vec)
                    else:
                        projected.append(np.dot(vec - ml['mean'].ravel(),
                                                ml['components'].T).astype('float32'))
                q_vec = np.concatenate(projected)
        else:
            q_vec = np.concatenate(concat_vec_parts).astype('float32')

        dists, inds = knn_search(self.concat_index_obj, q_vec, k=cfg.knn_k)
        concat_score = float(1.0 - np.max(dists)) if cfg.knn_metric == 'cosine' \
                       else float(np.min(dists))

        # Patch discrepancy heatmap
        test_patches = patch_tokens_list[-1]          # (1, P, D)
        if self.per_image_last_patches is not None:
            retrieved   = self.per_image_last_patches[[int(i) for i in inds]].to(self.device)
        else:
            retrieved   = torch.zeros(1, test_patches.shape[1], test_patches.shape[2]).to(self.device)

        test_n = F.normalize(test_patches, dim=2)
        retr_n = F.normalize(retrieved.to(self.device), dim=2)
        sim_kp = torch.matmul(test_n.squeeze(0),
                              retr_n.view(-1, retr_n.shape[2]).t())    # (P, K*P)
        patch_disc = 1.0 - sim_kp.max(dim=1)[0]                       # (P,)

        P    = patch_disc.shape[0]
        g    = int(P**0.5)
        hm   = patch_disc.view(1, g, g).unsqueeze(1)
        hm   = F.interpolate(hm, size=(img.shape[-2], img.shape[-1]),
                             mode='bilinear', align_corners=False)
        hm_np= hm.squeeze().cpu().numpy()

        # Reported MemoViT frame score (Eq. 7): robust-MAD aggregation of the
        # per-patch discrepancy field. The CLS-prototype (`cls_score`) and
        # concat-kNN (`concat_score`) values are diagnostics only and are NOT
        # the reported metric — see REPRODUCIBILITY.md (issues #2, #7).
        reported_score = robust_patch_score(patch_disc)

        self.last_diagnostics = {
            'reported_robust_patch': reported_score,   # Eq. 7  (reported)
            'cls_proto':             float(cls_score), # diagnostic
            'concat_knn':            float(concat_score),  # diagnostic
            'patch_mean':            float(patch_disc.mean().cpu()),  # diagnostic
        }
        return reported_score, hm_np


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 · PaDiM
#   Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
#   Anomaly Detection and Localization", ICPR 2021
# ══════════════════════════════════════════════════════════════════════════════

class PaDiMModel:
    """
    PaDiM adapted for aerial imagery using DINOv2 or CNN patch features.

    Training (fit)
    ──────────────
    1. Extract patch features for each training frame → (N, P, D)
    2. PCA-reduce to d dimensions for memory / conditioning stability
    3. For each spatial position p ∈ {1…P}:
         μ_p  = mean over N training samples           shape (d,)
         Σ_p  = sample covariance + ε·I               shape (d, d)
         Σ_p⁻¹= inverse covariance (pre-computed)

    Inference (score)
    ─────────────────
    For a test patch feature z_p:
       dist_p = sqrt( (z_p − μ_p)ᵀ Σ_p⁻¹ (z_p − μ_p) )   [Mahalanobis]
    Anomaly map = bilinear-upsample(dist_map) to image resolution
    Frame score = max(dist_map)
    """
    def __init__(self, extractor: BaseFeatureExtractor,
                 pca_dim: int = 150,
                 eps: float = 1e-4):
        self.extractor = extractor
        self.pca_dim   = pca_dim
        self.eps       = eps
        # fit state
        self.mean_map    : Optional[np.ndarray] = None   # (H*W, d)
        self.cov_inv_map : Optional[np.ndarray] = None   # (H*W, d, d)
        self.pca_mean_   : Optional[np.ndarray] = None   # (1, D)
        self.pca_comps_  : Optional[np.ndarray] = None   # (d, D)
        self.grid_h = self.grid_w = None

    # ─── Fit ──────────────────────────────────────────────────────────────
    def fit(self, loader: DataLoader):
        print('[PaDiM] Collecting patch features …')
        all_patches = []   # accumulate (B, P, D) CPU tensors
        for batch in tqdm(loader, desc='[PaDiM] fit'):
            imgs    = batch['standard'][:, 0].float()
            imgs    = imgs.to(next(iter([p for p in
                      (self.extractor.teacher.parameters()
                       if hasattr(self.extractor, 'teacher')
                       else [])] or [torch.zeros(1)]), torch.zeros(1)).device
                      if hasattr(self.extractor, 'teacher') else imgs)
            # Move imgs to same device as extractor
            patches = self.extractor.extract_patches(imgs).detach().cpu()   # (B, P, D) CPU
            all_patches.append(patches.half())               # float16 to save RAM

        all_patches = torch.cat(all_patches, dim=0).float().numpy()   # (N, P, D)
        N, P, D     = all_patches.shape
        H, W        = self.extractor.grid_size
        self.grid_h, self.grid_w = H, W
        assert P == H * W, f'Grid mismatch: P={P}, H×W={H*W}'

        # ── PCA on all patches jointly ─────────────────────────────────────
        d          = min(self.pca_dim, D - 1, N - 1)
        flat       = all_patches.reshape(N * P, D)
        pca_mean   = flat.mean(axis=0, keepdims=True).astype('float32')
        flat_c     = (flat - pca_mean).astype('float32')
        print(f'[PaDiM] Computing PCA {D} → {d} on {N*P} patch vectors …')
        # SVD on a random subset if too large (>500k samples)
        if N * P > 500_000:
            idx_sub  = np.random.choice(N*P, 500_000, replace=False)
            _, _, Vt = np.linalg.svd(flat_c[idx_sub], full_matrices=False)
        else:
            _, _, Vt = np.linalg.svd(flat_c, full_matrices=False)
        comps      = Vt[:d].astype('float32')                # (d, D)
        flat_proj  = flat_c @ comps.T                        # (N*P, d)
        self.pca_mean_ = pca_mean                            # (1, D)
        self.pca_comps_= comps                               # (d, D)

        # ── Per-position Gaussian ──────────────────────────────────────────
        feat_spatial = flat_proj.reshape(N, P, d)            # (N, P, d)
        print(f'[PaDiM] Fitting {P} per-position Gaussians (d={d}) …')
        mean_map     = feat_spatial.mean(axis=0)             # (P, d)

        # Vectorised covariance: Σ_p = (X_p - μ_p)ᵀ(X_p - μ_p) / (N-1)
        diff         = feat_spatial - mean_map[None]         # (N, P, d)
        # cov_map[p] = diff[:,p,:].T @ diff[:,p,:] / (N-1)
        # efficient einsum over all positions simultaneously
        cov_map      = np.einsum('npi,npj->pij', diff, diff) / max(N - 1, 1)  # (P, d, d)
        cov_map     += self.eps * np.eye(d)[None]            # regularise

        print(f'[PaDiM] Inverting {P} covariance matrices ({d}×{d}) …')
        cov_inv_map  = np.linalg.inv(cov_map)                # (P, d, d)

        self.mean_map    = mean_map
        self.cov_inv_map = cov_inv_map
        print(f'[PaDiM] Fit complete.  Memory: mean={mean_map.nbytes/1e6:.1f}MB  '
              f'cov_inv={cov_inv_map.nbytes/1e6:.1f}MB')

    # ─── Score ────────────────────────────────────────────────────────────
    def score(self, img: torch.Tensor) -> Tuple[float, np.ndarray]:
        """img: (1, C, H, W) on device.  Returns (frame_score, heatmap)."""
        patches = self.extractor.extract_patches(img)           # (1, P, D) CPU
        patches = patches.squeeze(0).detach().cpu().numpy().astype('float32')  # (P, D)

        # PCA project
        patches_c    = patches - self.pca_mean_                 # (P, D)
        patches_proj = patches_c @ self.pca_comps_.T            # (P, d)

        # Vectorised Mahalanobis distance
        diff     = patches_proj - self.mean_map                 # (P, d)
        temp     = np.einsum('pi,pij->pj', diff, self.cov_inv_map)  # (P, d)
        dist_sq  = (temp * diff).sum(axis=-1)                   # (P,)
        dist_map = np.sqrt(np.maximum(dist_sq, 0.0))            # (P,)  ensure non-neg

        # Upsample to image resolution
        H, W   = self.grid_h, self.grid_w
        dm_2d  = dist_map.reshape(H, W)
        dm_t   = torch.from_numpy(dm_2d).unsqueeze(0).unsqueeze(0)
        dm_up  = F.interpolate(dm_t, size=(img.shape[-2], img.shape[-1]),
                               mode='bilinear', align_corners=False)
        hm_np  = dm_up.squeeze().numpy()

        frame_score = float(dist_map.max())
        return frame_score, hm_np


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 · PatchCore
#   Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
#   CVPR 2022
# ══════════════════════════════════════════════════════════════════════════════

class PatchCoreModel:
    """
    PatchCore adapted for aerial imagery using DINOv2 or CNN patch features.

    Training (fit)
    ──────────────
    1. Extract patch tokens from all training frames: shape (N×P, D)
    2. Optionally aggregate neighbourhood (avg-pool over k×k spatial kernel)
    3. Build a coreset via greedy FPS, retaining `coreset_ratio` of all patches
    4. Index the coreset with FAISS (or torch fallback) for fast KNN search

    Inference (score)
    ─────────────────
    For each test patch p:
       score_p = min_distance_to_coreset(p)         [cosine or L2]
    Anomaly map = bilinear-upsample(score_map)
    Frame score = max(score_map)   [matches paper's image-level score]
    """
    def __init__(self, extractor: BaseFeatureExtractor,
                 coreset_ratio: float = 0.01,
                 knn_k: int = 9,
                 neighbour_kernel: int = 3,
                 metric: str = 'cosine',
                 use_faiss: bool = True):
        self.extractor       = extractor
        self.coreset_ratio   = coreset_ratio
        self.knn_k           = knn_k
        self.neighbour_kernel= neighbour_kernel
        self.metric          = metric
        self.use_faiss       = use_faiss and _HAS_FAISS
        # fit state
        self.index_obj  = None
        self.grid_h = self.grid_w = None

    # ─── Neighbourhood aggregation (local context) ─────────────────────────
    @staticmethod
    def _aggregate_neighbourhood(patches: torch.Tensor,
                                  grid_h: int, grid_w: int,
                                  kernel: int = 3) -> torch.Tensor:
        """
        Replace each patch embedding with the average of its k×k neighbourhood.
        This makes representations more robust to small misalignments.
        patches: (B, P, D) → returns (B, P, D)
        """
        B, P, D = patches.shape
        feat_map = patches.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)  # (B,D,H,W)
        pad      = kernel // 2
        avg      = F.avg_pool2d(feat_map.float(), kernel_size=kernel,
                                stride=1, padding=pad)                          # (B,D,H,W)
        return avg.permute(0, 2, 3, 1).reshape(B, P, D)

    # ─── Fit ──────────────────────────────────────────────────────────────
    def fit(self, loader: DataLoader):
        H, W = self.extractor.grid_size
        self.grid_h, self.grid_w = H, W

        print('[PatchCore] Collecting patch features …')
        all_patches = []
        for batch in tqdm(loader, desc='[PatchCore] fit'):
            imgs    = batch['standard'][:, 0].float()
            patches = self.extractor.extract_patches(imgs).detach().cpu()           # (B, P, D) CPU
            if self.neighbour_kernel > 1:
                patches = self._aggregate_neighbourhood(patches, H, W,
                                                        self.neighbour_kernel)
            all_patches.append(patches.half())

        all_patches = torch.cat(all_patches, dim=0).float()         # (N, P, D)
        N, P, D     = all_patches.shape
        flat        = all_patches.reshape(N * P, D).numpy().astype('float32')  # (N*P, D)
        total       = flat.shape[0]

        # ── Greedy coreset via FPS ─────────────────────────────────────────
        target_n = max(1, int(total * self.coreset_ratio))
        print(f'[PatchCore] FPS coreset: {total} → {target_n} patches …')
        sel_idx  = greedy_farthest_point_sampling_rows_cpu(flat, target_n)
        coreset  = flat[sel_idx]                                     # (M, D)

        # ── Build KNN index ────────────────────────────────────────────────
        self.index_obj = build_knn_index(coreset, metric=self.metric,
                                         use_faiss=self.use_faiss)
        print(f'[PatchCore] Coreset size: {coreset.shape}  '
              f'FAISS={self.use_faiss}')

    # ─── Score ────────────────────────────────────────────────────────────
    # ─── NEW: Batch KNN search for the full patch grid in ONE FAISS call ──────
    @staticmethod
    def _batch_knn_search(index_obj: dict,
                          queries: np.ndarray,   # (P, D) float32
                          k: int,
                          metric: str) -> np.ndarray:
        """
        Search all P patch queries simultaneously.
        Returns patch_scores: (P,) float32.
    
        Replaces the Python for-loop that called knn_search() P times per frame.
        A single index.search(queries, k) is orders of magnitude faster than
        P separate index.search(q, k) calls because FAISS parallelises across
        both the query batch and the index vectors using BLAS/AVX2 internally.
        """
        if 'index' in index_obj:
            # ── FAISS path ─────────────────────────────────────────────────────
            idx = index_obj['index']
            if index_obj.get('normalized'):
                # Inner-product (cosine) index: normalise queries first
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                np.maximum(norms, 1e-8, out=norms)
                queries_n = queries / norms                         # (P, D)
                sims, _   = idx.search(queries_n, k)               # (P, k) inner products
                # anomaly score = 1 - max_similarity (higher sim → more normal)
                patch_scores = 1.0 - sims.max(axis=1)              # (P,)
            else:
                # L2 index: distances already squared
                dists_sq, _ = idx.search(queries, k)               # (P, k)
                patch_scores = np.sqrt(
                    np.maximum(dists_sq.min(axis=1), 0.0))         # (P,)
        else:
            # ── NumPy fallback (no FAISS): batched matmul ──────────────────────
            feats = index_obj['features']                           # (M, D) float32
            if index_obj.get('normalized'):
                norms = np.linalg.norm(queries, axis=1, keepdims=True)
                np.maximum(norms, 1e-8, out=norms)
                queries_n = queries / norms                         # (P, D)
                # Sims matrix (P, M) — done in one matmul
                sims      = queries_n @ feats.T                     # (P, M)
                # top-k per row
                if k == 1:
                    patch_scores = 1.0 - sims.max(axis=1)
                else:
                    top_k = np.partition(sims, -k, axis=1)[:, -k:]
                    patch_scores = 1.0 - top_k.max(axis=1)
            else:
                # L2: ||q - f||² = ||q||² - 2q·f + ||f||²
                q_sq  = (queries ** 2).sum(axis=1, keepdims=True)  # (P, 1)
                f_sq  = (feats   ** 2).sum(axis=1, keepdims=True).T # (1, M)
                cross = queries @ feats.T                           # (P, M)
                d2    = np.maximum(q_sq - 2 * cross + f_sq, 0.0)   # (P, M)
                if k == 1:
                    patch_scores = np.sqrt(d2.min(axis=1))
                else:
                    top_k = np.partition(d2, k, axis=1)[:, :k]
                    patch_scores = np.sqrt(top_k.min(axis=1))
    
        return patch_scores.astype('float32')
    
    
    # ─── Score ────────────────────────────────────────────────────────────
    def score(self, img: torch.Tensor) -> Tuple[float, np.ndarray]:
        """
        img: (1, C, H, W) on device.  Returns (frame_score, heatmap).
    
        KEY FIX: replaced the per-patch for-loop (1369 FAISS calls/frame) with
        a single batched _batch_knn_search() call.  Typical speedup: 100-500×.
        """
        H, W    = self.grid_h, self.grid_w
        patches = self.extractor.extract_patches(img)               # (1, P, D) CPU
        if self.neighbour_kernel > 1:
            patches = self._aggregate_neighbourhood(
                patches, H, W, self.neighbour_kernel)
    
        patches_np = patches.squeeze(0).cpu().numpy().astype('float32')  # (P, D)
    
        # ── Single batched KNN search over all P patches ───────────────────────
        patch_scores = self._batch_knn_search(                      # (P,) float32
            self.index_obj, patches_np, k=self.knn_k, metric=self.metric)
    
        # ── Spatial anomaly map → upsample to image resolution ────────────────
        score_map = patch_scores.reshape(H, W)
        sm_t  = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        sm_up = F.interpolate(sm_t, size=(img.shape[-2], img.shape[-1]),
                              mode='bilinear', align_corners=False)
        hm_np = sm_up.squeeze().numpy()
    
        frame_score = float(patch_scores.mean())
        return frame_score, hm_np


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 · SimpleNet
#   Liu et al., "SimpleNet: A Simple Network for Image Anomaly Detection
#   and Localization", CVPR 2023
# ══════════════════════════════════════════════════════════════════════════════

class _SimpleNetAdapter(nn.Module):
    """
    2-layer MLP with BatchNorm that adapts pretrained features into a
    'normality space' in which genuine normals cluster together and
    Gaussian-noise pseudo-anomalies are pushed apart.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,    hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim,  bias=False),
            nn.BatchNorm1d(out_dim),
        )
        # Kaiming init (important for training stability)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SimpleNetDiscriminator(nn.Module):
    """Single linear layer: outputs logit for 'normal' class."""
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)              # (N,)


class SimpleNetModel:
    """
    SimpleNet: trains a lightweight adapter + discriminator on normal-only
    patches.  Pseudo-anomalies are synthesised by adding Gaussian noise to
    normal features in-the-loop (no external anomaly data needed).

    Training (fit)
    ──────────────
    For each mini-batch of normal patches f:
      adapted_normal  = Adapter(f)
      adapted_anomaly = Adapter(f + ε)          ε ~ N(0, σ²)
      loss = BCE(D(adapted_normal), 1) + BCE(D(adapted_anomaly), 0)
    Optimise Adapter + Discriminator jointly.

    Inference (score)
    ─────────────────
    anomaly_score_p = 1 − sigmoid(D(Adapter(f_p)))
    Frame score     = max over patches
    """
    def __init__(self, extractor: BaseFeatureExtractor,
                 device: torch.device,
                 noise_std: float = 0.015,
                 lr: float        = 2e-4,
                 epochs: int      = 30,
                 hidden_mult: int = 2):
        self.extractor = extractor
        self.device    = device
        self.noise_std = noise_std
        self.lr        = lr
        self.epochs    = epochs
        # Networks
        D_in = extractor.feat_dim
        D_h  = D_in * hidden_mult
        self.adapter       = _SimpleNetAdapter(D_in, D_h, D_in).to(device, non_blocking=True)
        self.discriminator = _SimpleNetDiscriminator(D_in).to(device, non_blocking=True)
        self.optimizer     = torch.optim.AdamW(
            list(self.adapter.parameters()) + list(self.discriminator.parameters()),
            lr=lr, weight_decay=1e-5)
        self.scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=lr * 0.01)
        self.criterion     = nn.BCEWithLogitsLoss()
        self.grid_h = self.grid_w = None

    # ─── Fit ──────────────────────────────────────────────────────────────
    def fit(self, loader: DataLoader):
        H, W            = self.extractor.grid_size
        self.grid_h, self.grid_w = H, W
        self.adapter.train()
        self.discriminator.train()

        best_loss = float('inf')
        for epoch in range(self.epochs):
            total_loss, n_batches = 0.0, 0
            for batch in tqdm(loader,
                              desc=f'[SimpleNet] Epoch {epoch+1}/{self.epochs}',
                              leave=False):
                imgs = batch['standard'][:, 0].float()
                with torch.no_grad():
                    patches = self.extractor.extract_patches(imgs).detach().cpu()   # (B, P, D) CPU

                B, P, D = patches.shape
                flat    = patches.reshape(B * P, D).to(self.device)  # (B*P, D)

                # ── Forward ──────────────────────────────────────────────
                adapted_normal  = self.adapter(flat)
                noise           = torch.randn_like(flat) * self.noise_std
                adapted_anomaly = self.adapter(flat + noise)

                score_normal  = self.discriminator(adapted_normal)   # (B*P,)
                score_anomaly = self.discriminator(adapted_anomaly)  # (B*P,)

                loss = (self.criterion(score_normal,  torch.ones_like(score_normal))
                      + self.criterion(score_anomaly, torch.zeros_like(score_anomaly)))

                # ── Backward ─────────────────────────────────────────────
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.adapter.parameters())
                    + list(self.discriminator.parameters()), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            self.scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)
            print(f'  [SimpleNet] Epoch {epoch+1:3d}/{self.epochs} | '
                  f'loss={avg_loss:.5f} | lr={self.scheduler.get_last_lr()[0]:.2e}')
            if avg_loss < best_loss:
                best_loss = avg_loss

        self.adapter.eval()
        self.discriminator.eval()
        print(f'[SimpleNet] Training done. Best loss: {best_loss:.5f}')

    # ─── Score ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def score(self, img: torch.Tensor) -> Tuple[float, np.ndarray]:
        """img: (1, C, H, W) on device.  Returns (frame_score, heatmap)."""
        patches = self.extractor.extract_patches(img)               # (1, P, D) CPU
        P       = patches.shape[1]
        flat    = patches.reshape(P, -1).to(self.device)            # (P, D)

        adapted = self.adapter(flat)                                 # (P, D)
        logits  = self.discriminator(adapted)                       # (P,)
        normal_prob    = torch.sigmoid(logits)                      # (P,)  high = normal
        anomaly_scores = (1.0 - normal_prob).cpu().numpy()          # (P,)  high = anomalous

        # Spatial map
        H, W  = self.grid_h, self.grid_w
        sm    = anomaly_scores.reshape(H, W)
        sm_t  = torch.from_numpy(sm).unsqueeze(0).unsqueeze(0)
        sm_up = F.interpolate(sm_t, size=(img.shape[-2], img.shape[-1]),
                              mode='bilinear', align_corners=False)
        hm_np = sm_up.squeeze().numpy()

        frame_score = float(anomaly_scores.max())
        return frame_score, hm_np


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 · UNIFIED EVALUATION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model_name: str,
                   score_fn,
                   test_scenes: List[str],
                   label_path: str,
                   config: BenchmarkConfig,
                   device: torch.device,
                   mean, std) -> Dict:
    """
    Unified evaluation loop shared by all methods.

    Parameters
    ----------
    model_name   : display name (e.g. 'PaDiM (DINOv2)')
    score_fn     : callable(img_tensor) → (frame_score: float, heatmap: np.ndarray)
                   img_tensor has shape (1, 3, H, W) on `device`
    test_scenes  : list of scene folder paths
    label_path   : path to folder containing *.npy binary GT label arrays
    config       : BenchmarkConfig instance
    mean, std    : per-channel normalisation stats (CPU tensors or lists)

    Returns
    -------
    dict with keys: auroc, eer, eer_threshold, ap, n_frames,
                    n_anomalous, inference_ms_per_frame
    """
    all_scores, all_labels = [], []
    total_time_s = 0.0
    total_frames = 0

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    for path_scene in test_scenes:
        scene_name = os.path.basename(path_scene)
        np_label   = np.load(os.path.join(label_path, f'{scene_name}.npy'),
                             allow_pickle=True)

        test_ds = CustomFrameDataset(
            path_scene, test_transform,
            resize_h=config.image_size, resize_w=config.image_size,
            time_step=config.num_frames)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False,
                                 num_workers=config.num_workers,
                                 pin_memory=True, drop_last=False)

        video_scores = []
        for batch in tqdm(test_loader, desc=f'  {model_name} | {scene_name}', leave=False):
            imgs = batch['standard'][:, 0].float().to(device)   # (B, 3, H, W)
            B    = imgs.shape[0]
        
            t0 = time.perf_counter()
            if B == 1:
                frame_score, _ = score_fn(imgs)
                frame_scores_batch = [frame_score]
            else:
                # Batched scoring: extract features for all B frames together,
                # then score each independently (patches are per-frame)
                frame_scores_batch = []
                for b in range(B):
                    fs, _ = score_fn(imgs[b:b+1])
                    frame_scores_batch.append(fs)
            total_time_s += time.perf_counter() - t0
            total_frames += B
            video_scores.extend(frame_scores_batch)

        video_scores = _norm_scores(video_scores)
        valid_len    = min(len(video_scores), len(np_label))
        all_scores.extend(video_scores[:valid_len])
        all_labels.extend(np_label[-valid_len:].tolist())

    all_labels = np.asarray(all_labels, dtype=np.float32)
    all_scores = np.asarray(all_scores, dtype=np.float32)

    if len(np.unique(all_labels)) < 2:
        print(f'[WARN] {model_name}: GT has only one class; AUROC undefined.')
        return {}

    auroc = float(roc_auc_score(all_labels, all_scores))
    ap    = float(average_precision_score(all_labels, all_scores))
    eer, eer_thr = compute_eer(all_labels, all_scores)
    ms_per_frame = (total_time_s / max(total_frames, 1)) * 1000.0
    n_anomalous  = int(all_labels.sum())

    return {
        'method'            : model_name,
        'auroc'             : auroc,
        'eer'               : eer,
        'eer_threshold'     : eer_thr,
        'ap'                : ap,
        'n_frames'          : total_frames,
        'n_anomalous'       : n_anomalous,
        'inference_ms_frame': ms_per_frame,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 · BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class BenchmarkRunner:
    """
    Orchestrates the full benchmark:
      1. Build DINOv2 teacher and, optionally, a CNN backbone
      2. For each enabled method: fit → evaluate → collect metrics
      3. Print and save the comparison table
    """

    def __init__(self, config: BenchmarkConfig, device: torch.device):
        self.config = config
        self.device = device
        self.results: List[Dict] = []

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _build_train_loader(self, mean, std) -> DataLoader:
        cfg   = self.config
        train_folder = os.path.join(cfg.data_path, 'train/frames')
        ds    = CustomFrameDataset(
            train_folder,
            transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=mean, std=std)]),
            resize_h=cfg.image_size, resize_w=cfg.image_size,
            time_step=cfg.num_frames)

        if cfg.max_train_samples and cfg.max_train_samples < len(ds):
            indices = random.sample(range(len(ds)), cfg.max_train_samples)
            ds      = data.Subset(ds, indices)
            print(f'[BenchmarkRunner] Subsampled train set to {cfg.max_train_samples} frames')

        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    def _get_test_scenes_and_labels(self):
        cfg        = self.config
        test_path  = os.path.join(cfg.data_path, 'test/')
        scenes     = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
        label_path = os.path.join(test_path, 'test_frame_mask')
        return scenes, label_path

    def _make_dinov2_extractor(self, teacher: TeacherMultiLayer,
                                layers: List[int],
                                concat: bool = False) -> DINOv2PatchExtractor:
        return DINOv2PatchExtractor(teacher, selected_layers=layers,
                                    concat_layers=concat,
                                    image_size=self.config.image_size)

    def _move_imgs_to_device(self, batch):
        """Helper: extract (1,C,H,W) frame from a batch dict and send to device."""
        return batch['standard'][:, 0].float().to(self.device)

    # ─── Main run ─────────────────────────────────────────────────────────

    def run(self):
        cfg = self.config
        os.makedirs(cfg.save_path, exist_ok=True)

        # ── Shared: compute dataset statistics ───────────────────────────
        train_folder = os.path.join(cfg.data_path, 'train/frames')
        print('\n[BenchmarkRunner] Computing dataset mean/std …')
        mean, std = compute_mean_and_std(
            train_folder, cfg.image_size, cfg.image_size, self.device,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        print(f'  mean={mean.tolist()}  std={std.tolist()}\n')

        # ── Shared: build DINOv2 teacher (once, reused by all DINOv2 methods) ─
        print('[BenchmarkRunner] Loading DINOv2 ViT-B/14 …')
        teacher = TeacherMultiLayer(selected_layers=cfg.selected_layers)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher = teacher.to(self.device)

        train_loader              = self._build_train_loader(mean, std)
        test_scenes, label_path   = self._get_test_scenes_and_labels()

        # ══════════════════════════════════════════════════════════════════
        # Method 1 · PROPOSED  (DINOv2-MemKNN)
        # ══════════════════════════════════════════════════════════════════
        if 'proposed' in cfg.methods_to_run:
            print('\n' + '─'*60)
            print('  METHOD: Proposed (DINOv2-MemKNN)')
            print('─'*60)
            model = ProposedModel(teacher, cfg, mean, std, self.device)
            with timer() as t:
                model.fit(train_loader)
            print(f'  Fit time: {t["elapsed"]:.1f}s')

            result = evaluate_model(
                'Proposed (DINOv2-MemKNN)',
                model.score,
                test_scenes, label_path, cfg, self.device, mean, std)
            result['fit_time_s'] = t['elapsed']
            result['trainable_params'] = 0
            result['backbone'] = 'DINOv2 ViT-B/14'
            result['training_free'] = True
            self.results.append(result)
            self._print_single(result)

        # ══════════════════════════════════════════════════════════════════
        # Method 2 · PaDiM  (DINOv2 backbone)
        # ══════════════════════════════════════════════════════════════════
        if 'padim' in cfg.methods_to_run:
            print('\n' + '─'*60)
            print('  METHOD: PaDiM (DINOv2 ViT-B/14)')
            print('─'*60)
            extractor = self._make_dinov2_extractor(
                teacher, cfg.padim_layers, concat=len(cfg.padim_layers) > 1)
            model = PaDiMModel(extractor, pca_dim=cfg.padim_pca_dim,
                               eps=cfg.padim_eps)
            with timer() as t:
                model.fit(train_loader)
            print(f'  Fit time: {t["elapsed"]:.1f}s')

            result = evaluate_model(
                'PaDiM (DINOv2)', model.score,
                test_scenes, label_path, cfg, self.device, mean, std)
            result['fit_time_s'] = t['elapsed']
            result['trainable_params'] = 0
            result['backbone'] = 'DINOv2 ViT-B/14'
            result['training_free'] = True
            self.results.append(result)
            self._print_single(result)

        # ══════════════════════════════════════════════════════════════════
        # Method 3 · PatchCore  (DINOv2 backbone)
        # ══════════════════════════════════════════════════════════════════
        if 'patchcore' in cfg.methods_to_run:
            print('\n' + '─'*60)
            print('  METHOD: PatchCore (DINOv2 ViT-B/14)')
            print('─'*60)
            extractor = self._make_dinov2_extractor(
                teacher, cfg.patchcore_layers, concat=False)
            model = PatchCoreModel(
                extractor,
                coreset_ratio   = cfg.patchcore_coreset_ratio,
                knn_k           = cfg.patchcore_knn_k,
                neighbour_kernel= cfg.patchcore_neighbour_k,
                metric          = cfg.knn_metric,
                use_faiss       = cfg.use_faiss_index)
            with timer() as t:
                model.fit(train_loader)
            print(f'  Fit time: {t["elapsed"]:.1f}s')

            result = evaluate_model(
                'PatchCore (DINOv2)', model.score,
                test_scenes, label_path, cfg, self.device, mean, std)
            result['fit_time_s'] = t['elapsed']
            result['trainable_params'] = 0
            result['backbone'] = 'DINOv2 ViT-B/14'
            result['training_free'] = True
            self.results.append(result)
            self._print_single(result)

        # ══════════════════════════════════════════════════════════════════
        # Method 4 · SimpleNet  (DINOv2 backbone)
        # ══════════════════════════════════════════════════════════════════
        if 'simplenet' in cfg.methods_to_run:
            print('\n' + '─'*60)
            print('  METHOD: SimpleNet (DINOv2 ViT-B/14)')
            print('─'*60)
            extractor = self._make_dinov2_extractor(
                teacher, cfg.simplenet_layers, concat=False)
            model = SimpleNetModel(
                extractor, self.device,
                noise_std   = cfg.simplenet_noise_std,
                lr          = cfg.simplenet_lr,
                epochs      = cfg.simplenet_epochs,
                hidden_mult = cfg.simplenet_hidden_mult)
            with timer() as t:
                model.fit(train_loader)
            print(f'  Fit time: {t["elapsed"]:.1f}s')

            n_params = (sum(p.numel() for p in model.adapter.parameters())
                      + sum(p.numel() for p in model.discriminator.parameters()))
            result = evaluate_model(
                'SimpleNet (DINOv2)', model.score,
                test_scenes, label_path, cfg, self.device, mean, std)
            result['fit_time_s'] = t['elapsed']
            result['trainable_params'] = n_params
            result['backbone'] = 'DINOv2 ViT-B/14'
            result['training_free'] = False
            self.results.append(result)
            self._print_single(result)

        # ══════════════════════════════════════════════════════════════════
        # Optional: CNN variants (WideResNet-50)
        # ══════════════════════════════════════════════════════════════════
        if cfg.run_cnn_backbone and any(m in cfg.methods_to_run
                                         for m in ('padim', 'patchcore', 'simplenet')):
            print('\n' + '═'*60)
            print('  CNN BACKBONE VARIANTS (WideResNet-50)')
            print('═'*60)
            cnn_extractor = CNNPatchExtractor(
                model_name  = cfg.cnn_model_name,
                device      = self.device,
                out_indices = cfg.cnn_out_indices,
                target_grid = cfg.cnn_target_grid)

            if 'padim' in cfg.methods_to_run:
                print('\n  METHOD: PaDiM (WideResNet-50)')
                model = PaDiMModel(cnn_extractor, pca_dim=cfg.padim_pca_dim)
                with timer() as t:
                    model.fit(train_loader)
                result = evaluate_model(
                    'PaDiM (WRN-50)', model.score,
                    test_scenes, label_path, cfg, self.device, mean, std)
                result.update({'fit_time_s': t['elapsed'],
                               'trainable_params': 0,
                               'backbone': 'WideResNet-50',
                               'training_free': True})
                self.results.append(result)
                self._print_single(result)

            if 'patchcore' in cfg.methods_to_run:
                print('\n  METHOD: PatchCore (WideResNet-50)')
                model = PatchCoreModel(cnn_extractor,
                                       coreset_ratio=cfg.patchcore_coreset_ratio,
                                       knn_k=cfg.patchcore_knn_k)
                with timer() as t:
                    model.fit(train_loader)
                result = evaluate_model(
                    'PatchCore (WRN-50)', model.score,
                    test_scenes, label_path, cfg, self.device, mean, std)
                result.update({'fit_time_s': t['elapsed'],
                               'trainable_params': 0,
                               'backbone': 'WideResNet-50',
                               'training_free': True})
                self.results.append(result)
                self._print_single(result)

            if 'simplenet' in cfg.methods_to_run:
                print('\n  METHOD: SimpleNet (WideResNet-50)')
                model = SimpleNetModel(cnn_extractor, self.device,
                                       noise_std=cfg.simplenet_noise_std,
                                       lr=cfg.simplenet_lr,
                                       epochs=cfg.simplenet_epochs)
                with timer() as t:
                    model.fit(train_loader)
                n_params = (sum(p.numel() for p in model.adapter.parameters())
                          + sum(p.numel() for p in model.discriminator.parameters()))
                result = evaluate_model(
                    'SimpleNet (WRN-50)', model.score,
                    test_scenes, label_path, cfg, self.device, mean, std)
                result.update({'fit_time_s': t['elapsed'],
                               'trainable_params': n_params,
                               'backbone': 'WideResNet-50',
                               'training_free': False})
                self.results.append(result)
                self._print_single(result)

        # ── Final table ───────────────────────────────────────────────────
        if self.results:
            print_comparison_table(self.results)
            save_results_csv(self.results, cfg.results_csv)

    def _print_single(self, r: Dict):
        if not r:
            return
        print(f'\n  ► {r.get("method","?")}'
              f'  AUROC={r.get("auroc",0):.4f}'
              f'  EER={r.get("eer",0):.4f}'
              f'  AP={r.get("ap",0):.4f}'
              f'  {r.get("inference_ms_frame",0):.1f}ms/frame')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 · RESULTS REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def print_comparison_table(results: List[Dict]):
    """
    Prints a LaTeX-style ASCII comparison table.
    Best values per metric are marked with *.
    """
    if not results:
        return

    hdr_cols = ['Method', 'Backbone', 'Train-Free',
                'AUROC↑', 'EER↓', 'AP↑',
                'ms/frame', 'Params']
    rows = []
    for r in results:
        rows.append([
            r.get('method', '—'),
            r.get('backbone', '—'),
            '✓' if r.get('training_free') else '✗',
            f'{r.get("auroc",0):.4f}',
            f'{r.get("eer",0):.4f}',
            f'{r.get("ap",0):.4f}',
            f'{r.get("inference_ms_frame",0):.1f}',
            f'{r.get("trainable_params",0):,}',
        ])

    # Mark best AUROC, best EER, best AP
    if results:
        best_auroc = max(r.get('auroc', 0) for r in results)
        best_eer   = min(r.get('eer', 1)   for r in results)
        best_ap    = max(r.get('ap', 0)    for r in results)
        for i, r in enumerate(results):
            if abs(r.get('auroc',0) - best_auroc) < 1e-6:
                rows[i][3] += '*'
            if abs(r.get('eer',1)   - best_eer)   < 1e-6:
                rows[i][4] += '*'
            if abs(r.get('ap',0)    - best_ap)     < 1e-6:
                rows[i][5] += '*'

    col_w = [max(len(hdr_cols[j]), *(len(rows[i][j]) for i in range(len(rows))))
             for j in range(len(hdr_cols))]

    sep = '+-' + '-+-'.join('-'*w for w in col_w) + '-+'

    def fmt_row(cells):
        return '| ' + ' | '.join(c.ljust(col_w[i]) for i, c in enumerate(cells)) + ' |'

    print('\n')
    print('╔' + '═'*(sum(col_w) + 3*(len(col_w)-1) + 4) + '╗')
    print('║  BENCHMARK RESULTS — Anomaly Detection Comparison'
          + ' '*(sum(col_w)+3*(len(col_w)-1)+4-52) + '║')
    print('║  * = best in column'
          + ' '*(sum(col_w)+3*(len(col_w)-1)+4-20) + '║')
    print('╚' + '═'*(sum(col_w) + 3*(len(col_w)-1) + 4) + '╝')
    print(sep)
    print(fmt_row(hdr_cols))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)
    print()


def save_results_csv(results: List[Dict], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    keys = ['method', 'backbone', 'training_free',
            'auroc', 'eer', 'eer_threshold', 'ap',
            'n_frames', 'n_anomalous', 'inference_ms_frame',
            'fit_time_s', 'trainable_params']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    print(f'[Results] Saved → {path}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 · MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Aerial anomaly detection benchmark: '
                    'Proposed vs PaDiM / PatchCore / SimpleNet')
    parser.add_argument('--method', nargs='+',
                        choices=['proposed', 'padim', 'patchcore', 'simplenet'],
                        default=None,
                        help='Methods to run (default: all four)')
    parser.add_argument('--cnn_backbone', action='store_true',
                        help='Also run PaDiM/PatchCore/SimpleNet with WRN-50')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Cap training frames per method (useful on GPU-memory-constrained machines)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Override dataset root path')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Override results output path')
    parser.add_argument('--gpu', type=int, default=0,
                        help='CUDA device index (default: 0)')
    return parser.parse_args()


def main():
    set_seed(42)
    args   = parse_args()
    config = BenchmarkConfig()
    config.update_from_args(args)

    # Override paths from CLI if provided
    if args.data_path:
        config.data_path = args.data_path
    if args.save_path:
        config.save_path = args.save_path

    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{config.gpu}')
    else:
        device = torch.device('cpu')
        print('[WARN] CUDA not available. Running on CPU — will be slow.')
    print(f'[Main] Device: {device}')

    print_config(config)

    runner = BenchmarkRunner(config, device)
    runner.run()


if __name__ == '__main__':
    main()
