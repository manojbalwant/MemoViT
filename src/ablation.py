"""
ablation.py — MemoViT ablation study (paper Table 6).

Four axes, one variable changed at a time with all else at the proposed config:
  A1 layer selection       (which ViT blocks form the CLS retrieval key)
  A2 CLS-PCA dimension     (compression of the multi-layer CLS key)
  A4 subsampling strategy  (FPS vs random vs stride vs k-means; retention ratio)
  A5 retrieval design      (metric, k, feature source)

Notes:
* A2/A4/A5 reuse one pre-built uncompressed bank from disk; A1 re-forwards once
  with all layers and slices in Python (layer choice changes captured activations).
* Stochastic conditions use seeds [42, 123, 7]; Wilcoxon signed-rank tests compare
  each variant to the proposed config on per-video AUROCs.
* The reported column is `patch_auroc` (robust-MAD, Eq. 7); `cls_proto_auroc` and
  `concat_auroc` are diagnostics. Results are checkpointed to JSON.

Usage:
    python ablation.py --build-bank          # build/save the base bank (A2/A4/A5)
    python ablation.py --run-all             # all axes
    python ablation.py --run A1              # a single axis
    python ablation.py --report              # summary tables + plots
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import glob
import copy
import json
import time
import random
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import timm
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.cluster import MiniBatchKMeans
from scoring import robust_patch_score   # canonical reported scorer (Eq. 7)
import cv2
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image

try:
    from scipy.stats import wilcoxon
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not found — Wilcoxon tests will be skipped.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    warnings.warn("matplotlib not found — plots will be skipped.")

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — BASE CONFIGURATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class BaseConfig:
    """
    Proposed (best) configuration.
    Ablation runners override individual fields via copy.deepcopy().
    """
    # ── paths ────────────────────────────────────────────────────────────────
    data_path   = "dataset/Drone-Anomaly/Highway/"
    save_path   = "experiments/Highway/"

    # ── image / model ────────────────────────────────────────────────────────
    image_size  = 518
    batch_size  = 16
    num_workers = 8
    num_frames  = 1

    # ── proposed layer config ────────────────────────────────────────────────
    selected_layers = [7,8,9,10,11]#[9, 10, 11]#list(range(12))

    # ── memory bank compression ──────────────────────────────────────────────
    per_layer_pca_dim = None      # None = disabled
    concat_pca_dim    = None      # None = disabled (set to 1280 in original)
    concat_fps_target = None      # None = keep all (will be set in ablation)

    # ── KNN ──────────────────────────────────────────────────────────────────
    knn_k              = 5
    knn_metric         = "cosine"   # "cosine" | "Euclidean"
    use_faiss_index    = True and _HAS_FAISS
    scoring            = "cosine"   # used by score_cls_vector_vs_prototypes

    # ── patch retrieval ──────────────────────────────────────────────────────
    use_last_layer_patches_for_knn = True

    # ── misc ─────────────────────────────────────────────────────────────────
    display_frames   = False
    heatmap_interp   = cv2.INTER_LINEAR


def cfg_from_overrides(overrides: dict) -> BaseConfig:
    """Return a deep-copied BaseConfig with overrides applied."""
    cfg = copy.copy(BaseConfig())
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise ValueError(f"Unknown config key: {k!r}")
        setattr(cfg, k, v)
    return cfg


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — ABLATION CONDITION REGISTRIES                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── A1: layer selection ──────────────────────────────────────────────────────
# concat_pca_dim=512 is pinned so all conditions retrieve from same-dim space.
A1_CONDITIONS: Dict[str, dict] = {
    "early_L012":      {"selected_layers": [0, 1, 2],          "concat_pca_dim": 512},
    "middle_L567":     {"selected_layers": [5, 6, 7],          "concat_pca_dim": 512},
    "last_L11":        {"selected_layers": [11],               "concat_pca_dim": 512},
    "last_L10_11":     {"selected_layers": [10, 11],           "concat_pca_dim": 512},
    "last_L9_11":      {"selected_layers": [9, 10, 11],        "concat_pca_dim": 512},  # proposed
    "last_L7_11":      {"selected_layers": [7, 8, 9, 10, 11],  "concat_pca_dim": 512},
    "all_L0_11":       {"selected_layers": list(range(12)),    "concat_pca_dim": 512},
}

# ── A2: compression strategy ─────────────────────────────────────────────────
# selected_layers and knn settings are pinned to proposed values.
A2_CONDITIONS: Dict[str, dict] = {
    "no_compress":        {"per_layer_pca_dim": None, "concat_pca_dim": None},
    "single_pca_128":     {"per_layer_pca_dim": None, "concat_pca_dim": 128},
    "single_pca_256":     {"per_layer_pca_dim": None, "concat_pca_dim": 256},
    "single_pca_512":     {"per_layer_pca_dim": None, "concat_pca_dim": 512},
    "single_pca_1024":    {"per_layer_pca_dim": None, "concat_pca_dim": 1024},
    "single_pca_2048":    {"per_layer_pca_dim": None, "concat_pca_dim": 2048},
    "per_layer_pca_32":   {"per_layer_pca_dim": 32,   "concat_pca_dim": None},
    "per_layer_pca_64":   {"per_layer_pca_dim": 64,   "concat_pca_dim": None},
    "per_layer_pca_128":  {"per_layer_pca_dim": 128,  "concat_pca_dim": None},
    "per_layer_pca_256":  {"per_layer_pca_dim": 256,  "concat_pca_dim": None},
    "per_layer_pca_512":  {"per_layer_pca_dim": 512,  "concat_pca_dim": None},
}

# ── A4: subsampling strategy ─────────────────────────────────────────────────
# Evaluated at multiple budget fractions.  Strategy name + ratio → condition.
A4_STRATEGIES  = ["fps", "random", "stride", "kmeans"]
A4_RATIOS      = [0.05, 0.10, 0.25, 0.50, 0.75, 1.00]
A4_RANDOM_RUNS = 5   # independent random subsamples per budget level

# ── A5: KNN design ───────────────────────────────────────────────────────────
A5_METRIC_CONDITIONS: Dict[str, dict] = {
    "cosine":            {"knn_metric": "cosine",    "scoring": "cosine"},
    "l2":                {"knn_metric": "Euclidean", "scoring": "l2"},
    "l2_normalized":     {"knn_metric": "Euclidean", "scoring": "l2",
                          "_normalize_before_l2": True},
}

A5_K_CONDITIONS: Dict[str, dict] = {
    f"k_{k}": {"knn_k": k} for k in [1, 3, 5, 10, 20]
}

# Feature source: controls what vector is fed into the KNN index.
# Values are handled specially in the runner (not direct config keys).
A5_FEAT_CONDITIONS = [
    "concat_cls",          # proposed — concat of selected layer CLS tokens
    "last_cls_only",       # only the last selected layer's CLS
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — MODEL AND DATA UTILITIES (reproduced for self-containment) ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resize_w, resize_h))
    return img


class CustomFrameDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width,
                 time_step=1, num_pred=0, frame_step=1):
        self.dir           = video_folder
        self.transform     = transform
        self._resize_h     = resize_height
        self._resize_w     = resize_width
        self._time_step    = time_step
        self._num_pred     = num_pred
        self._frame_step   = frame_step
        self.video_frames  = []
        self.index_samples = []
        self._setup()

    def _setup(self):
        videos = sorted(glob.glob(os.path.join(self.dir, "*")))
        all_frames = []
        if videos and os.path.isdir(videos[0]):
            for v in videos:
                frames = sorted(
                    glob.glob(os.path.join(v, "*.jpg")),
                    key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1])
                )
                all_frames.extend(frames)
        else:
            videos.sort(
                key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1])
            )
            all_frames = videos
        self.video_frames  = all_frames
        max_idx            = len(all_frames) - (self._time_step + self._num_pred - 1) * self._frame_step
        self.index_samples = list(range(max_idx))

    def __getitem__(self, index):
        fi   = self.index_samples[index]
        path = self.video_frames[fi]
        img  = np_load_frame(path, self._resize_h, self._resize_w)
        img  = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        # return dict matching original script's key convention
        img_np = img.unsqueeze(0)          # (1, C, H, W)
        return {"standard": img_np, "256": img_np}

    def __len__(self):
        return len(self.index_samples)


def compute_mean_std(train_folder, image_size, device, batch_size=64, num_workers=4):
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    ds     = datasets.ImageFolder(train_folder, transform=tf)
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    C      = ds[0][0].shape[0]
    mean   = torch.zeros(C, device=device)
    std    = torch.zeros(C, device=device)
    npix   = 0
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="mean/std"):
            imgs  = imgs.to(device)
            B,c,h,w = imgs.shape
            npix += B * h * w
            mean += imgs.sum(dim=[0, 2, 3])
            std  += (imgs ** 2).sum(dim=[0, 2, 3])
    mean /= npix
    std   = (std / npix - mean ** 2).sqrt()
    return mean.cpu(), std.cpu()


def compute_eer(y_true, y_score):
    from sklearn.metrics import roc_curve
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score, drop_intermediate=False)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    if 0 < idx < len(fpr) - 1:
        fpr0, fnr0, fpr1, fnr1 = fpr[idx-1], fnr[idx-1], fpr[idx], fnr[idx]
        denom = (fpr1 - fpr0) - (fnr1 - fnr0)
        if abs(denom) > 1e-12:
            t   = float(np.clip((fnr0 - fpr0) / denom, 0, 1))
            eer = float(fpr0 + t * (fpr1 - fpr0))
            thr_val = float(thr[idx-1] + t * (thr[idx] - thr[idx-1]))
        else:
            eer, thr_val = float((fpr[idx]+fnr[idx])/2), float(thr[idx])
    else:
        eer, thr_val = float((fpr[idx]+fnr[idx])/2), float(thr[idx])
    return eer, thr_val


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — TEACHER MODEL (collect ALL 12 layers always)               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _get_embed_dim(vit):
    return getattr(vit, "embed_dim", None) or getattr(vit, "num_features", 768)


class TeacherAllLayers(nn.Module):
    """
    Runs DINOv2 ViT and always captures ALL transformer blocks.
    Callers then slice to their desired selected_layers in Python,
    avoiding redundant GPU forward passes across A1 conditions.
    """
    def __init__(self, model_name="vit_base_patch14_dinov2", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        for attr in ["head", "head_dist", "fc"]:
            if hasattr(self.vit, attr):
                try:
                    setattr(self.vit, attr, nn.Identity())
                except Exception:
                    pass
        self.embed_dim = _get_embed_dim(self.vit)
        self.n_blocks  = len(getattr(self.vit, "blocks", []))

    def forward(self, x):
        x = self.vit.patch_embed(x)
        B, N, D = x.shape
        has_cls  = hasattr(self.vit, "cls_token")
        has_dist = hasattr(self.vit, "dist_token")

        if has_cls:
            cls_tok = self.vit.cls_token.expand(B, -1, -1)
        if has_dist:
            dist_tok = self.vit.dist_token.expand(B, -1, -1)
            tokens   = torch.cat((cls_tok, dist_tok, x), dim=1)
        elif has_cls:
            tokens = torch.cat((cls_tok, x), dim=1)
        else:
            tokens = x

        if hasattr(self.vit, "pos_embed"):
            tokens = tokens + self.vit.pos_embed.to(tokens.device)

        all_cls    = []   # list[n_blocks] of (B, D)
        all_patches = []  # list[n_blocks] of (B, P, D)

        for i, block in enumerate(self.vit.blocks):
            tokens = block(tokens)
            if has_dist:
                c = tokens[:, 0:2].mean(dim=1)
                p = tokens[:, 2:].contiguous()
            elif has_cls:
                c = tokens[:, 0].contiguous()
                p = tokens[:, 1:].contiguous()
            else:
                c = tokens.mean(dim=1)
                p = tokens.contiguous()
            all_cls.append(c)
            all_patches.append(p)

        # Apply final norm to the last block outputs (in-place replacement)
        if hasattr(self.vit, "norm"):
            normed = self.vit.norm(tokens)
            if has_dist:
                all_cls[-1]     = normed[:, 0:2].mean(dim=1)
                all_patches[-1] = normed[:, 2:].contiguous()
            elif has_cls:
                all_cls[-1]     = normed[:, 0].contiguous()
                all_patches[-1] = normed[:, 1:].contiguous()
            else:
                all_cls[-1]     = normed.mean(dim=1)
                all_patches[-1] = normed.contiguous()

        return all_cls, all_patches   # both are lists of length n_blocks


def slice_layers(all_cls: list, all_patches: list, selected_layers: list):
    """Slice full-layer output to only the requested layer indices."""
    cls_list     = [all_cls[i]     for i in selected_layers]
    patches_list = [all_patches[i] for i in selected_layers]
    return cls_list, patches_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — MEMORY BANK CONSTRUCTION                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def collect_full_bank(teacher: TeacherAllLayers, loader, device, image_size,
                      all_n_blocks: int, patch_layers_to_keep: Optional[List[int]] = None):
    """
    Collect ALL layers' CLS tokens and ALL layers' patch tokens for every
    training image.  This is the one expensive forward pass; everything else
    slices from this in memory.

    Returns
    -------
    all_cls_cpu     : list[n_blocks] of (N, D) float32 tensors
    all_patches_cpu : list[n_blocks] of (N, P, D) float32 tensors
    file_list       : list[str]  — same order as loader
    """
    teacher.eval()

    cls_accum = [[] for _ in range(all_n_blocks)]

    keep_set = set(patch_layers_to_keep) if patch_layers_to_keep is not None else None
    patches_accum = {}
    if keep_set is not None:
        for li in keep_set:
            if li < 0 or li >= all_n_blocks:
                raise ValueError(f"patch layer index out of range: {li}")
            patches_accum[li] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting bank"):
            imgs = batch["standard"][:, 0].float().to(device)
            all_cls, all_patches = teacher(imgs)

            for i in range(all_n_blocks):
                cls_accum[i].append(all_cls[i].detach().cpu())

                if keep_set is not None and i in keep_set:
                    patches_accum[i].append(all_patches[i].detach().cpu())

    all_cls_cpu = [torch.cat(cls_accum[i], dim=0) for i in range(all_n_blocks)]

    if keep_set is not None:
        all_patches_cpu = {
            li: torch.cat(patches_accum[li], dim=0)
            for li in sorted(keep_set)
        }
    else:
        all_patches_cpu = None

    file_list = loader.dataset.video_frames
    return all_cls_cpu, all_patches_cpu, file_list


def apply_pca_per_layer(per_layer_raw: list, per_layer_pca_dim: int):
    """
    Independent PCA per layer.
    Returns projected list[L] of (N, k) and pca_meta list[L] of dicts.
    """
    projected, meta = [], []
    for li, arr_t in enumerate(per_layer_raw):
        arr = arr_t.numpy().astype("float32")
        N, C = arr.shape
        k    = int(min(per_layer_pca_dim, N - 1, C))
        if k <= 0 or k >= C:
            projected.append(arr_t)
            meta.append(None)
            continue
        mu  = arr.mean(axis=0, keepdims=True).astype("float32")
        Xc  = arr - mu
        try:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        except np.linalg.LinAlgError:
            projected.append(arr_t)
            meta.append(None)
            continue
        comps = Vt[:k].astype("float32")
        Z     = np.dot(Xc, comps.T).astype("float32")
        projected.append(torch.from_numpy(Z))
        meta.append({"mean": mu, "components": comps})
        print(f"  [PCA] layer {li}: {C} → {k}  ({N} samples, "
              f"top-{k} SVs explain "
              f"{((np.linalg.svd(Xc, compute_uv=False)[:k]**2).sum() / (Xc**2).sum() * 100):.1f}% var)")
    return projected, meta


def apply_pca_single_shot(concat_mat: torch.Tensor, target_dim: int):
    """Single-shot PCA on the full concatenated matrix."""
    arr   = concat_mat.numpy().astype("float32")
    N, D  = arr.shape
    k     = int(min(target_dim, N - 1, D))
    if k <= 0 or k >= D:
        return concat_mat, None
    mu    = arr.mean(axis=0, keepdims=True).astype("float32")
    Xc    = arr - mu
    try:
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return concat_mat, None
    comps = Vt[:k].astype("float32")
    Z     = np.dot(Xc, comps.T).astype("float32")
    print(f"  [PCA single-shot] {D} → {k}  ({N} samples)")
    return torch.from_numpy(Z), {"mean": mu, "components": comps,
                                 "_legacy_single_shot": True}


def greedy_fps(feature_matrix_np: np.ndarray, target_n: int) -> np.ndarray:
    """Greedy farthest-point sampling — returns selected row indices."""
    N, D = feature_matrix_np.shape
    if target_n >= N:
        return np.arange(N, dtype=np.int32)
    rows  = feature_matrix_np.copy()
    norms = np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
    rows /= norms
    sel   = [0]
    dists = np.sum((rows - rows[0:1]) ** 2, axis=1)
    for _ in range(1, target_n):
        idx = int(np.argmax(dists))
        sel.append(idx)
        newd  = np.sum((rows - rows[idx:idx+1]) ** 2, axis=1)
        dists = np.minimum(dists, newd)
    return np.array(sel, dtype=np.int32)


def subsample_bank(per_layer_raw_concat: torch.Tensor,
                   all_cls_cpu: list,
                   all_patches_cpu: list,
                   file_list: list,
                   strategy: str,
                   target_n: int,
                   selected_layers: list,
                   seed: int = 42) -> Tuple[torch.Tensor, list, list, list]:
    """
    Apply one of {fps, random, stride, kmeans} to produce a subsampled bank.

    Returns
    -------
    concat_sub   : (M, D) subsampled feature matrix (or centroids for kmeans)
    cls_sub      : list[L] of (M, C) — None for kmeans (no original rows)
    patches_sub  : list[L] of (M, P, C) — None for kmeans
    files_sub    : list[str] — None for kmeans
    """
    set_seed(seed)
    N = len(file_list)
    arr = per_layer_raw_concat.numpy().astype("float32")

    if strategy == "fps":
        idx = greedy_fps(arr, target_n)
    elif strategy == "random":
        idx = np.random.choice(N, size=min(target_n, N), replace=False)
    elif strategy == "stride":
        idx = np.linspace(0, N - 1, min(target_n, N)).astype(int)
    elif strategy == "kmeans":
        n_clust = min(target_n, N)
        km      = MiniBatchKMeans(n_clusters=n_clust, random_state=seed,
                                  n_init=3, batch_size=max(256, n_clust))
        km.fit(arr)
        centroids = km.cluster_centers_.astype("float32")
        return (torch.from_numpy(centroids),
                None, None, None)          # no patch tokens for centroids
    else:
        raise ValueError(f"Unknown subsampling strategy: {strategy!r}")

    idx_list    = [int(i) for i in idx]
    concat_sub  = per_layer_raw_concat[idx_list]
    cls_sub     = [all_cls_cpu[li][idx_list]     for li in selected_layers]
    patches_sub = [all_patches_cpu[li][idx_list] for li in selected_layers[-1:]] #[all_patches_cpu[li][idx_list] for li in selected_layers]
    files_sub   = [file_list[i] for i in idx_list]
    return concat_sub, cls_sub, patches_sub, files_sub


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6 — KNN INDEX AND SEARCH                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_index(features_cpu: torch.Tensor, metric: str = "cosine",
                use_faiss: bool = True, normalize_l2: bool = False):
    """
    Build a KNN index.  normalize_l2=True applies unit-norm before L2 indexing
    (used in A5 l2_normalized condition to separate geometry from normalization).
    """
    feats = features_cpu.numpy().astype("float32")
    N, D  = feats.shape

    if normalize_l2 or metric == "cosine":
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        feats = feats / norms
        normalized = True
    else:
        normalized = False

    if use_faiss and _HAS_FAISS:
        if metric == "cosine" or normalize_l2:
            idx = faiss.IndexFlatIP(D)
        else:
            idx = faiss.IndexFlatL2(D)
        idx.add(feats)
        return {"index": idx, "normalized": normalized,
                "metric": metric, "feats_shape": (N, D)}
    else:
        return {"features": feats, "metric": metric,
                "normalized": normalized, "feats_shape": (N, D)}


def knn_search(index_obj: dict, query: np.ndarray, k: int):
    """
    Returns (distances, indices) for top-k neighbours of query vector.
    For cosine / normalized L2: distances are inner products (higher = more similar).
    For raw L2: distances are Euclidean distances.
    """
    q = query.astype("float32").ravel()

    if "index" in index_obj:
        fidx = index_obj["index"]
        if index_obj["normalized"]:
            q = q / (np.linalg.norm(q) + 1e-8)
        dists, inds = fidx.search(q.reshape(1, -1), k)
        if not index_obj["normalized"] and index_obj["metric"] != "cosine":
            dists = np.sqrt(np.maximum(dists, 0))   # squared → Euclidean
        return dists.ravel(), inds.ravel()
    else:
        feats = index_obj["features"]
        if index_obj["normalized"]:
            q = q / (np.linalg.norm(q) + 1e-8)
            sims = feats @ q
            inds = np.argsort(-sims)[:k]
            return sims[inds], inds
        else:
            d2   = np.sum((feats - q) ** 2, axis=1)
            d    = np.sqrt(d2)
            inds = np.argsort(d)[:k]
            return d[inds], inds


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7 — SCORING HELPERS                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def score_cls_vs_bank(cls_vec: torch.Tensor, bank: torch.Tensor,
                      metric: str = "cosine") -> float:
    """Frame-level anomaly score: distance of test CLS to nearest prototype."""
    v = cls_vec.unsqueeze(0).to(bank.device)
    if metric == "cosine":
        vn = F.normalize(v, dim=1)
        bn = F.normalize(bank, dim=1)
        sim = torch.matmul(vn, bn.t()).max().item()
        return 1.0 - sim
    elif metric == "l2":
        d = torch.cdist(v, bank).min().item()
        return float(d)
    return 0.0

def patch_discrepancy_score(test_patches: torch.Tensor,
                             retrieved_patches: torch.Tensor,
                             metric: str = "cosine") -> float:
    """
    Mean per-patch discrepancy between test patches and k retrieved sets.

    test_patches       : (P, C)
    retrieved_patches  : (K, P, C)
    """
    if metric == "cosine":
        tn = F.normalize(test_patches, dim=1)          # (P, C)
        rn = F.normalize(retrieved_patches, dim=2)     # (K, P, C)
        rn_flat = rn.view(-1, rn.shape[2])             # (K*P, C)
        sim  = torch.matmul(tn, rn_flat.t())           # (P, K*P)
        best = sim.max(dim=1)[0]                       # (P,)
        return robust_patch_score(1.0 - best) #float((1.0 - best).mean().item())
    else:
        te  = test_patches.unsqueeze(0).expand(retrieved_patches.shape[0], -1, -1)
        d2  = ((retrieved_patches - te) ** 2).sum(dim=2)
        md  = d2.min(dim=0)[0].sqrt()
        return robust_patch_score(md) #float(md.mean().item())    


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 8 — EVALUATION ENGINE                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
    
def evaluate(teacher: TeacherAllLayers,
             cfg: BaseConfig,
             all_cls_cpu: list,
             all_patches_cpu: list,
             file_list: list,
             mean, std,
             device,
             pca_meta=None,
             feature_src: str = "concat_cls",
             normalize_l2: bool = False) -> Dict[str, Any]:
    """
    Run zero-shot evaluation given a pre-built memory bank.

    Parameters
    ----------
    all_cls_cpu      : list[n_blocks] full-layer CLS from training set
    all_patches_cpu  : list[n_blocks] full-layer patches from training set
    feature_src      : one of {"concat_cls", "last_cls_only",
                                "last_patch_mean", "early_cls"}
    pca_meta         : output of apply_pca_per_layer or apply_pca_single_shot
    normalize_l2     : whether query/bank were L2-normalized before indexing

    Returns
    -------
    dict with keys: cls_proto_auroc, frame_eer, patch_auroc, concat_auroc,
                    per_video_aurocs (list), build_time_s, infer_ms_per_frame
    """
    selected = cfg.selected_layers

    # ── slice bank to selected layers ────────────────────────────────────────
    per_layer_raw    = [all_cls_cpu[li]     for li in selected]
    patch_layer = 11#selected[-1]
    last_patches_cpu = all_patches_cpu[patch_layer] # (N, P, C)

    # ── build the retrieval feature matrix ───────────────────────────────────
    t_build_start = time.time()

    if feature_src == "concat_cls":
        # Stack selected CLS layers → (N, L*C)
        raw_concat = torch.cat(per_layer_raw, dim=1)

        # Apply compression if requested
        if pca_meta is not None:
            legacy = (len(pca_meta) == 1 and isinstance(pca_meta[0], dict)
                      and pca_meta[0].get("_legacy_single_shot", False))
            if legacy:
                arr = raw_concat.numpy().astype("float32")
                arr = arr - pca_meta[0]["mean"]
                arr = np.dot(arr, pca_meta[0]["components"].T).astype("float32")
                index_feats = torch.from_numpy(arr)
            else:
                parts = []
                for li, feat in enumerate(per_layer_raw):
                    m = pca_meta[li] if li < len(pca_meta) else None
                    if m is None:
                        parts.append(feat.numpy().astype("float32"))
                    else:
                        arr = feat.numpy().astype("float32") - m["mean"]
                        arr = np.dot(arr, m["components"].T).astype("float32")
                        parts.append(arr)
                index_feats = torch.from_numpy(np.concatenate(parts, axis=1))
        else:
            index_feats = raw_concat

    elif feature_src == "last_cls_only":
        index_feats = per_layer_raw[-1]     # (N, C)
        pca_meta    = None

    elif feature_src == "last_patch_mean":
        # Mean-pool last-layer patch tokens → global descriptor (N, C)
        index_feats = last_patches_cpu.mean(dim=1)
        pca_meta    = None

    elif feature_src == "early_cls":
        # Use layer 5 (index 5 in all_cls_cpu)
        early_layer = min(5, len(all_cls_cpu) - 1)
        index_feats = all_cls_cpu[early_layer]
        pca_meta    = None

    else:
        raise ValueError(f"Unknown feature_src: {feature_src!r}")

    index_obj   = build_index(index_feats, metric=cfg.knn_metric,
                              use_faiss=cfg.use_faiss_index,
                              normalize_l2=normalize_l2)
    build_time  = time.time() - t_build_start

    # ── test loop ─────────────────────────────────────────────────────────────
    test_base   = os.path.join(cfg.data_path, "test")
    scenes      = sorted(glob.glob(os.path.join(test_base, "frames/*")))
    label_base  = os.path.join(test_base, "test_frame_mask")
    tf          = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean.tolist() if isinstance(mean, torch.Tensor) else mean,
            std =std.tolist()  if isinstance(std,  torch.Tensor) else std,
        )
    ])

    all_frame_scores  = []
    all_patch_scores  = []
    all_concat_scores = []
    all_labels        = []
    per_video_aurocs  = []
    total_frames      = 0
    infer_total_ms    = 0.0

    teacher.eval()
    with torch.no_grad():
        for scene in scenes:
            scene_name = os.path.basename(scene)
            label_path = os.path.join(label_base, f"{scene_name}.npy")
            if not os.path.exists(label_path):
                print(f"  [WARN] label not found: {label_path} — skipping")
                continue

            np_label = np.load(label_path, allow_pickle=True)
            ds       = CustomFrameDataset(
                scene, tf,
                resize_height=cfg.image_size,
                resize_width =cfg.image_size,
                time_step    =cfg.num_frames,
            )
            loader = data.DataLoader(
                ds, batch_size=1, shuffle=False,
                num_workers=cfg.num_workers, pin_memory=True, drop_last=False
            )

            vid_frame_scores  = []
            vid_patch_scores  = []
            vid_concat_scores = []

            for batch in loader:
                img = batch["standard"][:, 0].float().to(device)

                t0 = time.time()
                all_cls_out, all_patches_out = teacher(img)
                infer_total_ms += (time.time() - t0) * 1000
                total_frames   += 1

                # Slice to selected layers
                cls_sel, patches_sel = slice_layers(
                    all_cls_out, all_patches_out, selected
                )

                # ── CLS prototype score (per-layer mean) ──────────────────
                layer_scores = []
                for li, cls_t in enumerate(cls_sel):
                    s = score_cls_vs_bank(
                        cls_t.squeeze(0).cpu(),
                        per_layer_raw[li].cpu(),
                        metric=cfg.scoring
                    )
                    layer_scores.append(s)
                frame_score = float(np.mean(layer_scores))
                vid_frame_scores.append(frame_score)

                # ── Concat KNN score ──────────────────────────────────────
                if feature_src == "concat_cls":
                    cls_parts = [cls_sel[i][0].detach().cpu().numpy()
                                 for i in range(len(cls_sel))]
                    if pca_meta is not None:
                        legacy = (len(pca_meta) == 1 and isinstance(pca_meta[0], dict)
                                  and pca_meta[0].get("_legacy_single_shot", False))
                        if legacy:
                            raw = np.concatenate(cls_parts) - pca_meta[0]["mean"].ravel()
                            qvec = np.dot(raw, pca_meta[0]["components"].T).astype("float32")
                        else:
                            qparts = []
                            for li, vec in enumerate(cls_parts):
                                m = pca_meta[li] if li < len(pca_meta) else None
                                if m is None:
                                    qparts.append(vec)
                                else:
                                    qparts.append(
                                        np.dot(vec - m["mean"].ravel(),
                                               m["components"].T).astype("float32")
                                    )
                            qvec = np.concatenate(qparts)
                    else:
                        qvec = np.concatenate(cls_parts).astype("float32")
                elif feature_src == "last_cls_only":
                    qvec = cls_sel[-1][0].detach().cpu().numpy().astype("float32")
                elif feature_src == "last_patch_mean":
                    qvec = patches_sel[-1][0].mean(dim=0).detach().cpu().numpy().astype("float32")
                elif feature_src == "early_cls":
                    early_layer = min(5, len(all_cls_out) - 1)
                    qvec = all_cls_out[early_layer][0].detach().cpu().numpy().astype("float32")
                else:
                    qvec = np.zeros(index_feats.shape[1], dtype="float32")

                dists, inds = knn_search(index_obj, qvec, k=cfg.knn_k)
                if cfg.knn_metric == "cosine":
                    concat_score = float(1.0 - np.max(dists))
                else:
                    concat_score = float(np.min(dists))
                vid_concat_scores.append(concat_score)

                # ── Patch discrepancy score ───────────────────────────────
                if cfg.use_last_layer_patches_for_knn and last_patches_cpu is not None:
                    retr_patches = last_patches_cpu[[int(i) for i in inds]].to(device)
                    test_patches = patches_sel[-1][0]   # (P, C)
                    ps = patch_discrepancy_score(test_patches, retr_patches,
                                                 metric=cfg.knn_metric)
                else:
                    ps = 0.0
                vid_patch_scores.append(ps)

            # ── per-video normalisation and AUROC ─────────────────────────
            labels_vid = np_label[-len(vid_frame_scores):]

            nfs = np.array(vid_frame_scores, dtype=np.float32)
            ncs = np.array(vid_concat_scores, dtype=np.float32)
            nps = np.array(vid_patch_scores, dtype=np.float32)

            all_frame_scores.extend(nfs.tolist())
            all_concat_scores.extend(ncs.tolist())
            all_patch_scores.extend(nps.tolist())
            all_labels.extend(labels_vid.tolist())

            if len(np.unique(labels_vid)) >= 2:
                va = roc_auc_score(labels_vid, ncs)
                per_video_aurocs.append(float(va))

    labels_arr  = np.array(all_labels, dtype=np.float32)
    cls_proto_auroc = patch_auroc = concat_auroc = None
    frame_eer   = None

    if len(labels_arr) > 0 and len(np.unique(labels_arr)) >= 2:
        cls_proto_auroc = float(roc_auc_score(labels_arr, np.array(all_frame_scores)))
        frame_eer, _= compute_eer(labels_arr, np.array(all_frame_scores))
        concat_auroc= float(roc_auc_score(labels_arr, np.array(all_concat_scores)))
        if any(s > 0 for s in all_patch_scores):
            patch_auroc = float(roc_auc_score(labels_arr, np.array(all_patch_scores)))

    infer_ms = infer_total_ms / total_frames if total_frames > 0 else 0.0

    return {
        "cls_proto_auroc":      cls_proto_auroc,
        "frame_eer":        frame_eer,
        "patch_auroc":      patch_auroc,
        "concat_auroc":     concat_auroc,
        "per_video_aurocs": per_video_aurocs,
        "build_time_s":     build_time,
        "infer_ms_frame":   infer_ms,
        "n_bank":           int(index_feats.shape[0]),
        "bank_dim":         int(index_feats.shape[1]),
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9 — STATISTICAL TESTS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def wilcoxon_vs_proposed(proposed_aurocs: list,
                          variant_aurocs: list,
                          alpha: float = 0.05) -> dict:
    """
    Wilcoxon signed-rank test: are variant per-video AUROCs significantly
    different from the proposed config?

    Returns dict with statistic, p_value, significant flag.
    """
    if not _HAS_SCIPY:
        return {"statistic": None, "p_value": None, "significant": None}
    if len(proposed_aurocs) != len(variant_aurocs) or len(proposed_aurocs) < 5:
        return {"statistic": None, "p_value": None,
                "significant": None, "note": "insufficient data"}
    try:
        stat, p = wilcoxon(proposed_aurocs, variant_aurocs,
                           alternative="two-sided", zero_method="wilcox")
        return {"statistic": float(stat), "p_value": float(p),
                "significant": bool(p < alpha)}
    except Exception as e:
        return {"statistic": None, "p_value": None,
                "significant": None, "note": str(e)}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 10 — ABLATION RUNNERS                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

SEEDS = [42, 123, 7]


def _save(results: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [saved] {path}")


def run_a1_layer_selection(teacher, all_cls_cpu, all_patches_cpu, file_list,
                            mean, std, device, save_dir):
    """
    A1: vary selected_layers.  Re-use the full-layer bank by slicing.
    PCA dim fixed at 512 to hold dimensionality constant.
    """
    print("\n" + "="*60)
    print("ABLATION A1 — Layer Selection")
    print("="*60)
    results = {}

    for cond_name, overrides in A1_CONDITIONS.items():
        print(f"\n  Condition: {cond_name}  overrides={overrides}")
        cfg = cfg_from_overrides(overrides)
        cfg.use_last_layer_patches_for_knn = True

        # Build per-layer CLS concat for this layer selection
        selected    = cfg.selected_layers
        per_layer_raw = [all_cls_cpu[li] for li in selected]
        raw_concat    = torch.cat(per_layer_raw, dim=1)

        # Apply single-shot PCA at fixed 512-D to hold dim constant
        pca_meta = None
        if cfg.concat_pca_dim is not None:
            raw_concat, single_meta = apply_pca_single_shot(raw_concat, cfg.concat_pca_dim)
            pca_meta = [single_meta] if single_meta is not None else None

        # Rebuild all_cls_cpu and all_patches_cpu as full lists for evaluate()
        metrics = evaluate(
            teacher, cfg,
            all_cls_cpu, all_patches_cpu, file_list,
            mean, std, device,
            pca_meta=pca_meta,
            feature_src="concat_cls",
        )
        metrics["condition"]  = cond_name
        metrics["overrides"]  = overrides
        results[cond_name]    = metrics

        print(f"    cls_proto_auroc={metrics['cls_proto_auroc']:.4f}  "
              f"concat_auroc={metrics['concat_auroc']:.4f}  "
              f"eer={metrics['frame_eer']:.4f}")

    # Wilcoxon test against proposed (last_L9_11)
    proposed_name = "last_L9_11"
    if proposed_name in results:
        prop_aurocs = results[proposed_name]["per_video_aurocs"]
        for cname, r in results.items():
            if cname == proposed_name:
                continue
            r["wilcoxon_vs_proposed"] = wilcoxon_vs_proposed(
                prop_aurocs, r["per_video_aurocs"]
            )

    _save(results, os.path.join(save_dir, "A1_layer_selection.json"))
    return results


def run_a2_compression(teacher, all_cls_cpu, all_patches_cpu, file_list,
                        mean, std, device, save_dir):
    """
    A2: vary PCA strategy and target dimension.
    selected_layers, knn settings fixed to proposed.
    """
    print("\n" + "="*60)
    print("ABLATION A2 — Memory Bank Compression")
    print("="*60)
    results = {}
    proposed_layers = BaseConfig.selected_layers

    for cond_name, overrides in A2_CONDITIONS.items():
        print(f"\n  Condition: {cond_name}  overrides={overrides}")
        cfg = cfg_from_overrides(overrides)

        per_layer_raw = [all_cls_cpu[li] for li in proposed_layers]
        raw_concat    = torch.cat(per_layer_raw, dim=1)

        pca_meta = None

        if overrides.get("per_layer_pca_dim") is not None:
            projected, pca_meta = apply_pca_per_layer(
                per_layer_raw, overrides["per_layer_pca_dim"]
            )
            # Replace per_layer_raw with projected versions in all_cls_cpu
            # for evaluate() — we pass via pca_meta, not direct replacement
            pca_type = "per_layer"

        elif overrides.get("concat_pca_dim") is not None:
            _, single_meta = apply_pca_single_shot(
                raw_concat, overrides["concat_pca_dim"]
            )
            pca_meta = [single_meta] if single_meta is not None else None
            pca_type = "single_shot"
        else:
            pca_type = "none"

        metrics = evaluate(
            teacher, cfg,
            all_cls_cpu, all_patches_cpu, file_list,
            mean, std, device,
            pca_meta=pca_meta,
            feature_src="concat_cls",
        )
        metrics["condition"] = cond_name
        metrics["overrides"] = overrides
        metrics["pca_type"]  = pca_type
        results[cond_name]   = metrics

        print(f"    concat_auroc={metrics['concat_auroc']:.4f}  "
              f"bank_dim={metrics['bank_dim']}  "
              f"build_time={metrics['build_time_s']:.2f}s")

    # Wilcoxon vs. no_compress baseline
    baseline = "no_compress"
    if baseline in results:
        base_aurocs = results[baseline]["per_video_aurocs"]
        for cname, r in results.items():
            if cname == baseline:
                continue
            r["wilcoxon_vs_baseline"] = wilcoxon_vs_proposed(
                base_aurocs, r["per_video_aurocs"]
            )

    _save(results, os.path.join(save_dir, "A2_compression.json"))
    return results


def run_a4_subsampling(teacher, all_cls_cpu, all_patches_cpu, file_list,
                        mean, std, device, save_dir):
    """
    A4: compare FPS, random, stride, K-means at several budget fractions.
    Patch scoring disabled for K-means (no original rows).
    """
    print("\n" + "="*60)
    print("ABLATION A4 — FPS vs. Random Subsampling")
    print("="*60)
    results = {}

    proposed_layers = BaseConfig.selected_layers
    per_layer_raw   = [all_cls_cpu[li] for li in proposed_layers]
    raw_concat      = torch.cat(per_layer_raw, dim=1)

    # Apply fixed PCA 512 so all strategies operate on same space
    concat_for_fps, single_meta = apply_pca_single_shot(raw_concat, 512)
    pca_meta = [single_meta] if single_meta is not None else None

    N_total = concat_for_fps.shape[0]

    for strategy in A4_STRATEGIES:
        for ratio in A4_RATIOS:
            target_n = max(1, int(ratio * N_total))
            n_runs   = A4_RANDOM_RUNS if strategy == "random" else 1

            run_aurocs = []
            run_patch_aurocs = []
            run_concat_aurocs = []

            for run_idx in range(n_runs):
                seed = SEEDS[0] + run_idx * 100
                print(f"\n  {strategy} | ratio={ratio:.2f} | "
                      f"target_n={target_n} | run={run_idx+1}/{n_runs}")

                concat_sub, cls_sub, patches_sub, files_sub = subsample_bank(
                    concat_for_fps, all_cls_cpu, all_patches_cpu,
                    file_list, strategy, target_n,
                    selected_layers=proposed_layers, seed=seed,
                )

                cfg = cfg_from_overrides({
                    "selected_layers": proposed_layers,
                    "knn_metric": "cosine",
                    "knn_k": BaseConfig.knn_k,
                    "use_last_layer_patches_for_knn": (strategy != "kmeans"),
                })

                # For K-means we have no original patches — build a synthetic
                # index from centroids (patches disabled).
                # IMPORTANT: pass the full 12-layer all_cls_cpu / all_patches_cpu
                # lists, NOT pre-sliced versions.  _evaluate_with_prebuilt_index
                # slices internally using cfg.selected_layers (absolute indices).
                if strategy == "kmeans":
                    index_obj = build_index(concat_sub, metric="cosine",
                                            use_faiss=_HAS_FAISS)
                    metrics = _evaluate_with_prebuilt_index(
                        teacher, cfg, index_obj,
                        all_cls_cpu, all_patches_cpu,   # full 12-layer lists
                        per_layer_raw, None,
                        pca_meta, mean, std, device,
                        feature_src="concat_cls",
                    )
                else:
                    # Rebuild all_cls / all_patches dicts pointing to subsampled rows
                    # We pass the subsampled versions via a synthetic all_cls_cpu
                    sub_all_cls     = list(all_cls_cpu)     # full list
                    sub_all_patches = list(all_patches_cpu) # full list
                    # Replace the selected layer slots with subsampled tensors
                    sub_all_cls_copy     = list(all_cls_cpu)
                    sub_all_patches_copy = {}#list(all_patches_cpu)
                    for rank, li in enumerate(proposed_layers):
                        sub_all_cls_copy[li]     = cls_sub[rank]
                        sub_all_patches_copy[proposed_layers[-1]] = patches_sub[0]#sub_all_patches_copy[li] = patches_sub[rank]

                    metrics = evaluate(
                        teacher, cfg,
                        sub_all_cls_copy, sub_all_patches_copy, files_sub,
                        mean, std, device,
                        pca_meta=pca_meta,
                        feature_src="concat_cls",
                    )

                run_aurocs.append(metrics.get("cls_proto_auroc") or 0.0)
                run_patch_aurocs.append(metrics.get("patch_auroc") or 0.0)
                run_concat_aurocs.append(metrics.get("concat_auroc") or 0.0)

                print(f"    concat_auroc={metrics.get('concat_auroc'):.4f}  "
                      f"n_bank={metrics['n_bank']}")

            key = f"{strategy}_r{int(ratio*100):03d}"
            results[key] = {
                "strategy": strategy,
                "ratio": ratio,
                "target_n": target_n,
                "cls_proto_auroc_mean": float(np.mean(run_aurocs)),
                "cls_proto_auroc_std":  float(np.std(run_aurocs)),
                "patch_auroc_mean": float(np.mean(run_patch_aurocs)),
                "patch_auroc_std":  float(np.std(run_patch_aurocs)),
                "concat_auroc_mean": float(np.mean(run_concat_aurocs)),
                "concat_auroc_std":  float(np.std(run_concat_aurocs)),
                "runs": run_concat_aurocs,
            }

    _save(results, os.path.join(save_dir, "A4_subsampling.json"))
    return results



def _evaluate_with_prebuilt_index(teacher, cfg, index_obj,
                                   all_cls_cpu, all_patches_cpu,
                                   per_layer_raw_orig, files_sub,
                                   pca_meta, mean, std, device,
                                   feature_src="concat_cls"):
    """
    Thin wrapper: runs the test loop using an already-built index_obj.
    Used for K-means centroids which have no original patch tokens.
    Falls back to per_layer_raw_orig for CLS prototype scoring.
    """
    # Temporarily monkey-patch evaluate to use provided index
    # (simpler than duplicating the test loop)
    selected = cfg.selected_layers
    per_layer_raw  = [all_cls_cpu[li] for li in selected]
    last_patches   = all_patches_cpu[selected[-1]]

    test_base  = os.path.join(cfg.data_path, "test")
    scenes     = sorted(glob.glob(os.path.join(test_base, "frames/*")))
    label_base = os.path.join(test_base, "test_frame_mask")
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean.tolist() if isinstance(mean, torch.Tensor) else mean,
            std =std.tolist()  if isinstance(std,  torch.Tensor) else std,
        )
    ])

    all_concat_scores = []
    all_labels        = []
    per_video_aurocs  = []

    teacher.eval()
    with torch.no_grad():
        for scene in scenes:
            scene_name = os.path.basename(scene)
            label_path = os.path.join(label_base, f"{scene_name}.npy")
            if not os.path.exists(label_path):
                continue
            np_label = np.load(label_path, allow_pickle=True)
            ds = CustomFrameDataset(
                scene, tf,
                resize_height=cfg.image_size, resize_width=cfg.image_size,
                time_step=cfg.num_frames,
            )
            loader = data.DataLoader(ds, batch_size=1, shuffle=False,
                                     num_workers=cfg.num_workers,
                                     pin_memory=True, drop_last=False)
            vid_scores = []
            for batch in loader:
                img = batch["standard"][:, 0].float().to(device)
                all_cls_out, _ = teacher(img)
                cls_sel, _ = slice_layers(all_cls_out, all_cls_out, selected)

                # Build query vector
                cls_parts = [cls_sel[i][0].detach().cpu().numpy()
                             for i in range(len(cls_sel))]
                if pca_meta is not None and pca_meta[0] is not None:
                    legacy = pca_meta[0].get("_legacy_single_shot", False)
                    if legacy:
                        raw = np.concatenate(cls_parts) - pca_meta[0]["mean"].ravel()
                        qvec = np.dot(raw, pca_meta[0]["components"].T).astype("float32")
                    else:
                        parts = []
                        for li, vec in enumerate(cls_parts):
                            m = pca_meta[li] if li < len(pca_meta) else None
                            if m is None:
                                parts.append(vec)
                            else:
                                parts.append(
                                    np.dot(vec - m["mean"].ravel(),
                                           m["components"].T).astype("float32")
                                )
                        qvec = np.concatenate(parts)
                else:
                    qvec = np.concatenate(cls_parts).astype("float32")

                dists, _ = knn_search(index_obj, qvec, k=cfg.knn_k)
                score = float(1.0 - np.max(dists)) if cfg.knn_metric == "cosine" \
                        else float(np.min(dists))
                vid_scores.append(score)

            labels_vid = np_label[-len(vid_scores):]
            ncs = np.array(vid_scores, dtype=np.float32)
            mn, mx = ncs.min(), ncs.max()
            if mx > mn:
                ncs = (ncs - mn) / (mx - mn + 1e-8)
            all_concat_scores.extend(ncs.tolist())
            all_labels.extend(labels_vid.tolist())
            if len(np.unique(labels_vid)) >= 2:
                per_video_aurocs.append(float(roc_auc_score(labels_vid, ncs)))

    labels_arr   = np.array(all_labels, dtype=np.float32)
    concat_auroc = None
    if len(labels_arr) > 0 and len(np.unique(labels_arr)) >= 2:
        concat_auroc = float(roc_auc_score(labels_arr, np.array(all_concat_scores)))
    return {
        "concat_auroc":     concat_auroc,
        "cls_proto_auroc":      concat_auroc,
        "per_video_aurocs": per_video_aurocs,
        "build_time_s":     0.0,
        "infer_ms_frame":   0.0,
        "n_bank":           int(index_obj.get("feats_shape", [0])[0]),
        "bank_dim":         int(index_obj.get("feats_shape", [0, 0])[1]),
    }


def run_a5_knn_design(teacher, all_cls_cpu, all_patches_cpu, file_list,
                       mean, std, device, save_dir):
    """
    A5: vary KNN metric, K, and feature source independently.
    Three sub-axes — metric, K sweep, feature source.
    """
    print("\n" + "="*60)
    print("ABLATION A5 — KNN Retrieval Design")
    print("="*60)
    results = {"metric": {}, "k_sweep": {}, "feature_src": {}}

    # ── Sub-axis 5a: distance metric ─────────────────────────────────────────
    print("\n  Sub-axis 5a: distance metric")
    for cond_name, overrides in A5_METRIC_CONDITIONS.items():
        print(f"    Condition: {cond_name}")
        normalize_l2 = overrides.pop("_normalize_before_l2", False)
        cfg = cfg_from_overrides({k: v for k, v in overrides.items()
                                  if k in BaseConfig.__dict__})
        metrics = evaluate(
            teacher, cfg,
            all_cls_cpu, all_patches_cpu, file_list,
            mean, std, device,
            pca_meta=None,
            feature_src="concat_cls",
            normalize_l2=normalize_l2,
        )
        metrics["condition"]      = cond_name
        metrics["normalize_l2"]   = normalize_l2
        results["metric"][cond_name] = metrics
        print(f"      concat_auroc={metrics['concat_auroc']:.4f}  "
              f"cls_proto_auroc={metrics['cls_proto_auroc']:.4f}")

    # Wilcoxon: cosine vs. others
    if "cosine" in results["metric"]:
        prop = results["metric"]["cosine"]["per_video_aurocs"]
        for cname, r in results["metric"].items():
            if cname == "cosine":
                continue
            r["wilcoxon_vs_cosine"] = wilcoxon_vs_proposed(
                prop, r["per_video_aurocs"]
            )

    # ── Sub-axis 5b: K sweep ─────────────────────────────────────────────────
    print("\n  Sub-axis 5b: K sweep")
    for cond_name, overrides in A5_K_CONDITIONS.items():
        print(f"    Condition: {cond_name}")
        cfg = cfg_from_overrides(overrides)
        metrics = evaluate(
            teacher, cfg,
            all_cls_cpu, all_patches_cpu, file_list,
            mean, std, device,
            pca_meta=None,
            feature_src="concat_cls",
        )
        metrics["condition"]     = cond_name
        results["k_sweep"][cond_name] = metrics
        print(f"      k={overrides['knn_k']}  "
              f"concat_auroc={metrics['concat_auroc']:.4f}")

    # ── Sub-axis 5c: feature source ───────────────────────────────────────────
    print("\n  Sub-axis 5c: feature source")
    for feat_src in A5_FEAT_CONDITIONS:
        print(f"    Feature source: {feat_src}")
        cfg = cfg_from_overrides({
            "knn_metric": "cosine",
            "knn_k": BaseConfig.knn_k,
        })
        metrics = evaluate(
            teacher, cfg,
            all_cls_cpu, all_patches_cpu, file_list,
            mean, std, device,
            pca_meta=None,
            feature_src=feat_src,
        )
        metrics["condition"] = feat_src
        results["feature_src"][feat_src] = metrics
        print(f"      concat_auroc={metrics['concat_auroc']:.4f}  "
              f"bank_dim={metrics['bank_dim']}")

    # Wilcoxon: concat_cls vs. others
    if "concat_cls" in results["feature_src"]:
        prop = results["feature_src"]["concat_cls"]["per_video_aurocs"]
        for cname, r in results["feature_src"].items():
            if cname == "concat_cls":
                continue
            r["wilcoxon_vs_concat_cls"] = wilcoxon_vs_proposed(
                prop, r["per_video_aurocs"]
            )

    _save(results, os.path.join(save_dir, "A5_knn_design.json"))
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 11 — REPORTING AND PLOTTING                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _load_json(path):
    with open(path) as f:
        return json.load(f)


def print_table(rows: list, headers: list, col_widths: list = None):
    if col_widths is None:
        col_widths = [max(len(str(r[i])) for r in rows + [headers]) + 2
                      for i in range(len(headers))]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    sep = "-" * sum(col_widths)
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))
    print(sep)


def report_a1(save_dir):
    path = os.path.join(save_dir, "A1_layer_selection.json")
    if not os.path.exists(path):
        print("A1 results not found.")
        return

    data_r = _load_json(path)
    print("\n── A1: Layer Selection ──────────────────────────────────────")

    rows = []
    for cname, r in data_r.items():
        layers = r["overrides"].get("selected_layers", "?")
        fa     = f"{r['cls_proto_auroc']:.4f}"  if r.get("cls_proto_auroc")  is not None else "N/A"
        pa     = f"{r['patch_auroc']:.4f}"  if r.get("patch_auroc")  is not None else "N/A"
        ca     = f"{r['concat_auroc']:.4f}" if r.get("concat_auroc") is not None else "N/A"
        wil    = r.get("wilcoxon_vs_proposed", {})
        p_val  = f"{wil.get('p_value'):.3f}" if wil.get("p_value") is not None else "—"
        sig    = "*" if wil.get("significant") else ""
        rows.append([cname, str(layers), fa, pa, ca, p_val + sig])

    print_table(
        rows,
        ["Condition", "Layers", "CLS-proto AUROC (diag.)", "Patch AUROC (reported, Eq.7)", "Concat AUROC (diag.)", "Wilcoxon p"],
        [22, 26, 24, 28, 21, 12]
    )

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 4))
        cnames = list(data_r.keys())
        aurocs = [data_r[c].get("cls_proto_auroc") or 0 for c in cnames]
        ax.bar(range(len(cnames)), aurocs, color="#378ADD", alpha=0.85)
        ax.axhline(y=max(aurocs), color="#D85A30", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(cnames)))
        ax.set_xticklabels(cnames, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Frame AUROC")
        ax.set_title("A1: Layer selection — Frame AUROC", fontsize=11)
        ax.set_ylim(max(0, min(aurocs) - 0.05), 1.02)
        plt.tight_layout()
        out = os.path.join(save_dir, "A1_layer_selection.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [plot saved] {out}")


def report_a2(save_dir):
    path = os.path.join(save_dir, "A2_compression.json")
    if not os.path.exists(path):
        print("A2 results not found.")
        return

    data_r = _load_json(path)
    print("\n── A2: Compression ──────────────────────────────────────────")

    rows = []
    for cname, r in data_r.items():
        fa   = f"{r['cls_proto_auroc']:.4f}"  if r.get("cls_proto_auroc")  is not None else "N/A"
        pa   = f"{r['patch_auroc']:.4f}"  if r.get("patch_auroc")  is not None else "N/A"
        ca   = f"{r['concat_auroc']:.4f}" if r.get("concat_auroc") is not None else "N/A"
        dim  = str(r.get("bank_dim", "?"))
        bt   = f"{r.get('build_time_s', 0):.2f}s"
        ptype = r.get("pca_type", "?")
        rows.append([cname, ptype, dim, fa, pa, ca, bt])

    print_table(
        rows,
        ["Condition", "CLS-PCA type", "CLS-key dim", "CLS-proto AUROC (diag.)", "Patch AUROC (reported, Eq.7)", "Concat AUROC (diag.)", "Build time"],
        [22, 14, 12, 24, 28, 21, 12]
    )

    if _HAS_MPL:
        # Separate curves for per-layer PCA and single-shot PCA
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, ptype, label in zip(
            axes, ["per_layer", "single_shot"],
            ["Per-layer PCA", "Single-shot PCA"]
        ):
            dims, aurocs = [], []
            for cname, r in data_r.items():
                if r.get("pca_type") == ptype and r.get("concat_auroc") is not None:
                    dims.append(r["bank_dim"])
                    aurocs.append(r["concat_auroc"])
            if dims:
                order = np.argsort(dims)
                dims   = [dims[i] for i in order]
                aurocs = [aurocs[i] for i in order]
                ax.plot(dims, aurocs, "o-", color="#378ADD")
                # Baseline
                if "no_compress" in data_r:
                    base = data_r["no_compress"].get("concat_auroc", 0)
                    ax.axhline(y=base, color="#D85A30", linestyle="--",
                               linewidth=1, label="no compress")
                    ax.legend(fontsize=8)
            ax.set_xlabel("Compressed dimension")
            ax.set_ylabel("Concat AUROC")
            ax.set_title(label, fontsize=10)

        plt.suptitle("A2: Memory bank compression", fontsize=11)
        plt.tight_layout()
        out = os.path.join(save_dir, "A2_compression.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [plot saved] {out}")


def report_a4(save_dir):
    path = os.path.join(save_dir, "A4_subsampling.json")
    if not os.path.exists(path):
        print("A4 results not found.")
        return
    data_r = _load_json(path)

    print("\n── A4: Subsampling Strategy ─────────────────────────────────")
    rows = []
    for key, r in data_r.items():
        strat = r["strategy"]
        ratio = r["ratio"]
        frame_mu = f"{r['cls_proto_auroc_mean']:.4f}" if r.get("cls_proto_auroc_mean") is not None else "N/A"
        patch_mu = f"{r['patch_auroc_mean']:.4f}" if r.get("patch_auroc_mean") is not None else "N/A"
        mu = f"{r['concat_auroc_mean']:.4f}" if r.get("concat_auroc_mean") is not None else "N/A"
        sd = f"{r['concat_auroc_std']:.4f}" if r.get("concat_auroc_std") is not None else "N/A"
        rows.append([strat, f"{ratio:.2f}", str(r["target_n"]), frame_mu, patch_mu, mu, sd])
    print_table(
        rows,
        ["Strategy", "Ratio", "N_bank", "Frame AUROC μ", "Patch AUROC μ", "Concat AUROC μ", "AUROC std"],
        [12, 8, 10, 14, 14, 14, 12]
    )

    if _HAS_MPL:
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"fps": "#378ADD", "random": "#D85A30",
                  "stride": "#1D9E75", "kmeans": "#BA7517"}
        for strat in A4_STRATEGIES:
            ratios, means, stds = [], [], []
            for key, r in data_r.items():
                if r["strategy"] == strat:
                    ratios.append(r["ratio"])
                    means.append(r["cls_proto_auroc_mean"])
                    stds.append(r["cls_proto_auroc_std"])
            if ratios:
                order = np.argsort(ratios)
                ratios = [ratios[i] for i in order]
                means = [means[i] for i in order]
                stds = [stds[i] for i in order]
                ax.plot(ratios, means, "o-", label=strat,
                        color=colors.get(strat, "gray"))
                ax.fill_between(ratios,
                                [m - s for m, s in zip(means, stds)],
                                [m + s for m, s in zip(means, stds)],
                                alpha=0.15, color=colors.get(strat, "gray"))
        ax.set_xlabel("N_sub / N_train")
        ax.set_ylabel("Frame AUROC")
        ax.set_title("A4: Subsampling strategy vs. budget", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out = os.path.join(save_dir, "A4_subsampling.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [plot saved] {out}")


def report_a5(save_dir):
    path = os.path.join(save_dir, "A5_knn_design.json")
    if not os.path.exists(path):
        print("A5 results not found.")
        return
    data_r = _load_json(path)

    print("\n── A5a: Metric ───────────────────────────────────────────────")
    rows = []
    for cname, r in data_r["metric"].items():
        ca = f"{r['concat_auroc']:.4f}" if r.get("concat_auroc") is not None else "N/A"
        pa = f"{r['patch_auroc']:.4f}" if r.get("patch_auroc") is not None else "N/A"
        wil = r.get("wilcoxon_vs_cosine", {})
        p = f"{wil.get('p_value'):.3f}" if wil.get("p_value") is not None else "—"
        sig = "*" if wil.get("significant") else ""
        rows.append([cname, ca, pa, p + sig])
    print_table(rows, ["Metric", "Concat AUROC", "Patch AUROC", "p vs. cosine"], [20, 14, 14, 14])

    print("\n── A5b: K sweep ──────────────────────────────────────────────")
    rows = []
    for cname, r in sorted(
        data_r["k_sweep"].items(),
        key=lambda x: x[1].get("knn_k", 0) if hasattr(x[1], "get") else 0
    ):
        k = cname.replace("k_", "")
        ca = f"{r['concat_auroc']:.4f}" if r.get("concat_auroc") is not None else "N/A"
        pa = f"{r['patch_auroc']:.4f}" if r.get("patch_auroc") is not None else "N/A"
        rows.append([k, ca, pa])
    print_table(rows, ["K", "Concat AUROC", "Patch AUROC"], [8, 14, 14])

    print("\n── A5c: Feature source ───────────────────────────────────────")
    rows = []
    for cname, r in data_r["feature_src"].items():
        ca = f"{r['concat_auroc']:.4f}" if r.get("concat_auroc") is not None else "N/A"
        pa = f"{r['patch_auroc']:.4f}" if r.get("patch_auroc") is not None else "N/A"
        dim = str(r.get("bank_dim", "?"))
        wil = r.get("wilcoxon_vs_concat_cls", {})
        p = f"{wil.get('p_value'):.3f}" if wil.get("p_value") is not None else "—"
        sig = "*" if wil.get("significant") else ""
        rows.append([cname, dim, ca, pa, p + sig])
    print_table(
        rows,
        ["Feature src", "Dim", "Concat AUROC", "Patch AUROC", "p vs. concat_cls"],
        [22, 8, 14, 14, 18]
    )

    if _HAS_MPL:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 5a
        ax = axes[0]
        names = list(data_r["metric"].keys())
        aurocs = [data_r["metric"][n].get("concat_auroc") or 0 for n in names]
        ax.bar(names, aurocs, color="#378ADD", alpha=0.85)
        ax.set_title("A5a: Metric", fontsize=10)
        ax.set_ylabel("Concat AUROC")
        ax.set_ylim(max(0, min(aurocs) - 0.05), 1.02)

        # 5b
        ax = axes[1]
        ks, aurocs_k = [], []
        for cname, r in sorted(data_r["k_sweep"].items(),
                               key=lambda x: int(x[0].replace("k_", ""))):
            ks.append(int(cname.replace("k_", "")))
            aurocs_k.append(r.get("concat_auroc") or 0)
        ax.plot(ks, aurocs_k, "o-", color="#378ADD")
        ax.set_xlabel("K")
        ax.set_title("A5b: K sweep", fontsize=10)
        ax.set_ylabel("Concat AUROC")

        # 5c
        ax = axes[2]
        srcs = list(data_r["feature_src"].keys())
        aurocs_s = [data_r["feature_src"][s].get("concat_auroc") or 0 for s in srcs]
        ax.bar(srcs, aurocs_s, color="#1D9E75", alpha=0.85)
        ax.set_title("A5c: Feature source", fontsize=10)
        ax.set_ylabel("Concat AUROC")
        ax.tick_params(axis="x", labelrotation=15)
        ax.set_ylim(max(0, min(aurocs_s) - 0.05), 1.02)

        plt.suptitle("A5: KNN retrieval design", fontsize=11)
        plt.tight_layout()
        out = os.path.join(save_dir, "A5_knn_design.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"  [plot saved] {out}")


def generate_report(save_dir):
    print("\n" + "="*60)
    print("ABLATION STUDY — SUMMARY REPORT")
    print("="*60)
    report_a1(save_dir)
    report_a2(save_dir)
    report_a4(save_dir)
    report_a5(save_dir)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 12 — MAIN ENTRY POINT                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_and_save_base_bank(cfg: BaseConfig, device):
    """
    Run one forward pass over all training images to collect all 12 layers.
    Saves to disk as a .pth file for reuse across A2/A4/A5.
    """
    bank_path = os.path.join(cfg.save_path, "ablation_base_bank.pth")
    if os.path.exists(bank_path):
        print(f"Base bank already exists at {bank_path} — loading.")
        ck = torch.load(bank_path, map_location="cpu")
        return (ck["all_cls_cpu"], ck["all_patches_cpu"],
                ck["file_list"], ck["mean"], ck["std"])

    train_folder = os.path.join(cfg.data_path, "train/frames")
    mean, std    = compute_mean_std(train_folder, cfg.image_size, device,
                                    batch_size=cfg.batch_size,
                                    num_workers=cfg.num_workers)
    print(f"Dataset mean={mean}  std={std}")

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])
    ds = CustomFrameDataset(
        train_folder, tf,
        resize_height=cfg.image_size, resize_width=cfg.image_size,
        time_step=cfg.num_frames,
    )
    loader = data.DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
    )

    teacher = TeacherAllLayers(pretrained=True).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    all_cls_cpu, all_patches_cpu, file_list = collect_full_bank(
        teacher, loader, device, cfg.image_size, teacher.n_blocks, patch_layers_to_keep=[cfg.selected_layers[-1]],
    )

    Path(cfg.save_path).mkdir(parents=True, exist_ok=True)
    torch.save({
        "all_cls_cpu":     all_cls_cpu,
        "all_patches_cpu": all_patches_cpu,
        "file_list":       file_list,
        "mean":            mean,
        "std":             std,
    }, bank_path)
    print(f"Base bank saved → {bank_path}")
    return all_cls_cpu, all_patches_cpu, file_list, mean, std


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study runner for DINOv2 anomaly detection"
    )
    parser.add_argument("--build-bank", action="store_true",
                        help="Build and save the full base memory bank")
    parser.add_argument("--run-all", action="store_true",
                        help="Run all four ablations (A1, A2, A4, A5)")
    parser.add_argument("--run", choices=["A1", "A2", "A4", "A5"],
                        help="Run a single ablation")
    parser.add_argument("--report", action="store_true",
                        help="Generate summary tables and plots from saved results")
    parser.add_argument("--data-path", default=None,
                        help="Override BaseConfig.data_path")
    parser.add_argument("--save-path", default=None,
                        help="Override BaseConfig.save_path")
    parser.add_argument("--device", default=None,
                        help="e.g. cuda:0 or cpu")
    args = parser.parse_args()

    # ── apply CLI overrides ──────────────────────────────────────────────────
    if args.data_path:
        BaseConfig.data_path = args.data_path
    if args.save_path:
        BaseConfig.save_path = args.save_path

    cfg       = BaseConfig()
    save_dir  = os.path.join(cfg.save_path, "ablation_results")
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)

    # ── report only (no model needed) ───────────────────────────────────────
    if args.report:
        generate_report(save_dir)
        return

    # ── build / load base bank ───────────────────────────────────────────────
    all_cls_cpu, all_patches_cpu, file_list, mean, std = \
        build_and_save_base_bank(cfg, device)

    if args.build_bank:
        print("Base bank ready. Exiting (--build-bank only mode).")
        return

    # ── load teacher once (reused across all ablations) ─────────────────────
    teacher = TeacherAllLayers(pretrained=True).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── dispatch ─────────────────────────────────────────────────────────────
    to_run = []
    if args.run_all:
        to_run = ["A1", "A2", "A4", "A5"]
    elif args.run:
        to_run = [args.run]
    else:
        parser.print_help()
        return

    for ablation in to_run:
        if ablation == "A1":
            run_a1_layer_selection(
                teacher, all_cls_cpu, all_patches_cpu, file_list,
                mean, std, device, save_dir
            )
        elif ablation == "A2":
            run_a2_compression(
                teacher, all_cls_cpu, all_patches_cpu, file_list,
                mean, std, device, save_dir
            )
        elif ablation == "A4":
            run_a4_subsampling(
                teacher, all_cls_cpu, all_patches_cpu, file_list,
                mean, std, device, save_dir
            )
        elif ablation == "A5":
            run_a5_knn_design(
                teacher, all_cls_cpu, all_patches_cpu, file_list,
                mean, std, device, save_dir
            )

    generate_report(save_dir)
    print("\nDone. All results saved to:", save_dir)


if __name__ == "__main__":
    main()
