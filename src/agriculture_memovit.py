"""
agrivision_dinov2_anomaly.py
============================
Zero-shot, one-class anomaly detection on Agriculture-Vision-2021
using a frozen DINOv2 teacher and a multi-layer memory-bank KNN.

Dataset  : Agriculture-Vision-2021 (RGB + optional NIR, 512×512 tiles)
Backbone : DINOv2 ViT-B/14  (pretrained, fully frozen)

Pipeline
--------
1. Scan train + val folders → construct one-class splits
       train  = pure-normal tiles   (memory bank)
       val    = small normal holdout (threshold calibration, optional)
       test   = balanced mix of held-out normal + ALL anomalous tiles
2. Build memory bank
       • Run frozen DINOv2 over all training tiles
       • Store per-tile concatenated CLS tokens (selected layers)
         + last-layer patch tokens  (optionally compressed via FPS / PCA)
3. Per test tile
       a) Tile-level score  = 1 – max-cosine-sim(CLS_query, memory_bank)
                              aggregated over selected transformer layers
       b) Patch heatmap     = per-patch cosine discrepancy vs K nearest
                              training tile patch tokens, bilinear-upsampled
                              to 512×512
4. Evaluate
       • Tile-level AUROC  (3 scoring variants: CLS, concat-KNN, patch-mean)
       • Pixel-level AUROC (heatmap scores restricted to valid pixels)

Integration notes  (doc1 → doc2)
---------------------------------
• CustomFrameDataset  →  AgriVisionAnomalyDataset  (tile-based, mask-aware)
• batch['standard'][:, 0]  →  image_tensor directly (no temporal dimension)
• tile_id strings replace video file_list in the memory bank
• Anomaly masks (pixel-level GT) replace frame-level binary labels
• Heatmaps are evaluated against pixel GT masked by the valid_mask
• Optional NIRChannelAdapter maps RGBN (4-ch) → pseudo-RGB (3-ch) for DINOv2
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import json
import random
import math
from glob import glob
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import timm
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scoring import robust_patch_score   # canonical reported scorer (Eq. 7)

from scipy.ndimage import label as scipy_label   
import gc
from memory_bank_fixes import collect_agri_memory_bank_fixed
import heapq, psutil, torchvision.transforms as T
from scipy.ndimage import gaussian_filter

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
#  §1  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the AgriVision-DINOv2 anomaly detection pipeline.
    Edit paths and hyper-parameters directly here before running.
    """
    # ── Dataset / Paths ────────────────────────────────────────────────────
    data_root: str          = "dataset/Agriculture-Vision-2021"
    save_path: str          = "experiments/agrivision_dinov2/"
    #  Set to a JSON path to skip re-scanning (auto-derived when None)
    splits_json: str        = None

    # ── Anomaly classes to evaluate (subset of 8 defined in the challenge) ─
    anomaly_classes: list   = ["water"]   # e.g. ['double_plant', 'drydown', 'endrow', 'nutrient_deficiency', 'planter_skip', 'storm_damage', 'water',  'waterway', 'weed_cluster']

    # ── Image ──────────────────────────────────────────────────────────────
    #  518 = 37 × 14 → perfectly divisible by DINOv2 ViT-B/14 patch size
    image_size: int         = 518
    original_tile_size: int = 512           # native AgriVision tile resolution
    use_nir: bool           = False         # True → RGBN 4-channel input
    apply_boundary: bool    = False         # restrict pixel eval to field boundaries

    # ── DINOv2 backbone ────────────────────────────────────────────────────
    model_name: str         = "vit_base_patch14_dinov2"
    selected_layers: list   = [7, 8, 9, 10, 11] #[9, 10, 11]    # ViT-B has 12 blocks (0-11)

    # ── Memory bank / KNN ─────────────────────────────────────────────────
    batch_size: int         = 32#64#128#16
    test_batch_size : int = 1#16    # tune up if GPU memory allows
    num_workers: int        = 8#4
    knn_k: int              = 5
    knn_metric: str         = "cosine"      # "cosine" | "l2"
    scoring: str            = "cosine"      # per-layer CLS distance metric
    use_faiss_index: bool   = True and _HAS_FAISS
    #  None  → no FPS row-compression on memory bank
    concat_fps_target: int  = 0.25#2500#5000#10000
    #  None  → no PCA on concatenated CLS features
    concat_pca_dim: int     = None#1024#512
    use_last_layer_patches_for_knn: bool = True#False

    # ── Evaluation ─────────────────────────────────────────────────────────
    pixel_eval: bool        = True          # pixel-level AUROC on valid pixels
    frame_eval: bool        = True          # tile-level AUROC

    # ── Misc ───────────────────────────────────────────────────────────────
    seed: int               = 42
    #  True = build memory bank and evaluate;  False = load checkpoint + eval
    train: bool             = True

    #Anomaly overlay display
    display_tiles: bool = False#True
    display_anomalous_only: bool = False#True
    display_every_n: int = 10


def print_config(cfg: Config):
    print("─── Configuration ───")
    for k, v in vars(cfg.__class__).items():
        if not k.startswith("__"):
            print(f"  {k:<35} = {v}")
    print("─────────────────────")


# ═══════════════════════════════════════════════════════════════════════════
#  §2  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_resize(size: int) -> T.Resize:
    """Construct T.Resize, handling older torchvision that lacks antialias kwarg."""
    try:
        return T.Resize((size, size), antialias=True)
    except TypeError:
        return T.Resize((size, size))

def verify_optimal_config(teacher, config, device):
    """
    Probe exact GPU usage at recommended batch sizes before
    committing to full 59480-tile Pass 1.
    Runs in under 30 seconds.
    """
    import psutil
    import torch.nn.functional as F

    def _report(tag, expected_gb):
        alloc = torch.cuda.memory_allocated(device) / 1e9
        peak  = torch.cuda.max_memory_allocated(device) / 1e9
        ram   = psutil.Process().memory_info().rss / 1e9
        gap   = expected_gb - peak
        flag  = "✓" if gap > 2.0 else "⚠ tight" if gap > 0 else "✗ OOM risk"
        print(f"  {tag:<35} alloc={alloc:.2f} peak={peak:.2f} "
              f"expected≈{expected_gb:.2f} margin={gap:.2f} GB  {flag}  "
              f"RAM={ram:.2f} GB")

    teacher.eval()
    P, D, K = 1369, 768, config.knn_k

    # ── Pass 1 probe ──────────────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        x = torch.randn(
            config.batch_size, 3,
            config.image_size, config.image_size,
            device=device
        )
        cls_list, patch_list = teacher(x)
        _report(f"Pass1 batch={config.batch_size}", expected_gb=15.93)
        del x, cls_list, patch_list
        torch.cuda.empty_cache()

    # ── Test inference probe ──────────────────────────────────────────
    torch.cuda.reset_peak_memory_stats(device)
    B = config.test_batch_size
    with torch.no_grad():
        x = torch.randn(
            B, 3, config.image_size, config.image_size,
            device=device
        )
        cls_list, patch_list = teacher(x)

        # Simulate patch similarity (the dominant test allocation)
        retrieved = torch.randn(B, K, P, D, dtype=torch.float16, device=device)
        t_n  = F.normalize(patch_list[-1].half(), dim=2)         # (B, P, D)
        r_n  = F.normalize(retrieved.view(B, K*P, D), dim=2)     # (B, K*P, D)
        sims = torch.bmm(t_n, r_n.transpose(1, 2))               # (B, P, K*P)
        disc = (1.0 - sims.max(dim=2).values).clamp(0.0, 2.0)
        _report(f"Test  batch={B} incl sims", expected_gb=18.61)

        del x, cls_list, patch_list, retrieved, t_n, r_n, sims, disc
        torch.cuda.empty_cache()

    print("\n  Config verified — safe to launch full experiment ✓")

# ═══════════════════════════════════════════════════════════════════════════
#  §3  DATASET COMPONENTS   (doc1 AgriVisionAnomalyDataset, extended)
# ═══════════════════════════════════════════════════════════════════════════

# ── Low-level mask helpers ─────────────────────────────────────────────────

def get_tile_ids(split_dir: str) -> List[str]:
    """Return sorted unique tile IDs from <split_dir>/images/rgb/."""
    rgb_dir = os.path.join(split_dir, "images", "rgb")
    files: List[str] = []
    for pat in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(glob(os.path.join(rgb_dir, pat))))
    return sorted({os.path.splitext(os.path.basename(f))[0] for f in files})


def read_mask(path: str) -> Optional[np.ndarray]:
    """Load a PNG mask and binarise to uint8 {0, 1}.  Returns None if absent."""
    if not os.path.exists(path):
        return None
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)


def load_valid_mask(tile_id: str, split_dir: str) -> np.ndarray:
    """Load the official valid-pixel mask.  Raises if not found."""
    p = os.path.join(split_dir, "masks", f"{tile_id}.png")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Valid mask missing: {p}")
    return read_mask(p)


def load_boundary_mask(tile_id: str, split_dir: str) -> Optional[np.ndarray]:
    """Load the field-boundary mask (optional; may be absent)."""
    p = os.path.join(split_dir, "boundaries", f"{tile_id}.png")
    return read_mask(p) if os.path.exists(p) else None


def combined_anomaly_mask(
    tile_id: str,
    split_dir: str,
    anomaly_classes: List[str],
    apply_boundary: bool = False,
) -> np.ndarray:
    """
    Union of all per-class anomaly masks restricted to valid (and optionally
    boundary) pixels.  Returns uint8 (512, 512) array; all-zeros for normal tiles.
    """
    combined = np.zeros((512, 512), dtype=np.uint8)
    labels_dir = os.path.join(split_dir, "labels")
    for cls in anomaly_classes:
        mask = read_mask(os.path.join(labels_dir, cls, f"{tile_id}.png"))
        if mask is not None:
            combined = np.maximum(combined, mask)
    # Always restrict to valid pixels (required by benchmark protocol)
    combined &= load_valid_mask(tile_id, split_dir)
    if apply_boundary:
        bnd = load_boundary_mask(tile_id, split_dir)
        if bnd is not None:
            combined &= bnd
    return combined


# ── One-class split construction ───────────────────────────────────────────

def make_anomaly_detection_splits(
    data_root: str,
    anomaly_classes: List[str],
    normal_test_ratio: float = 1.0,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[dict, str]:
    """
    Partition Agriculture-Vision-2021 train + val folders into a rigorous
    one-class anomaly detection benchmark:

        train → pure-normal tiles           (memory bank construction)
        val   → small normal holdout        (threshold / hyper-param tuning)
        test  → balanced: normal + anomalous (evaluation)

    Returns (splits_dict, path_to_saved_json).
    """
    random.seed(seed)
    np.random.seed(seed)

    TRAIN_DIR = os.path.join(data_root, "train")
    VAL_DIR   = os.path.join(data_root, "val")

    normal: List[dict] = []
    anomalous: List[dict] = []
    for sname, sdir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
        for tid in tqdm(get_tile_ids(sdir), desc=f"Scanning {sname}"):
            try:
                amask = combined_anomaly_mask(tid, sdir, anomaly_classes)
                rec   = {"original_split": sname, "tile_id": tid}
                (anomalous if amask.sum() > 0 else normal).append(rec)
            except Exception as exc:
                print(f"  [WARN] Skipping {tid}: {exc}")

    random.shuffle(normal)
    n_anom  = len(anomalous)
    n_test  = int(n_anom * normal_test_ratio)
    n_val   = max(1, int((len(normal) - n_test) * val_fraction))

    test_normal  = normal[:n_test]
    val_normal   = normal[n_test : n_test + n_val]
    train_normal = normal[n_test + n_val :]

    test_ids = test_normal + anomalous
    random.shuffle(test_ids)

    splits = {
        "train": train_normal,
        "val":   val_normal,
        "test":  test_ids,
        "metadata": {
            "seed": seed,
            "anomaly_classes": anomaly_classes,
            "train_count":          len(train_normal),
            "val_count":            len(val_normal),
            "test_normal_count":    len(test_normal),
            "test_anomalous_count": n_anom,
            "test_total":           len(test_ids),
        },
    }
    suffix   = "_".join(anomaly_classes)
    out_path = os.path.join(data_root, f"agri_anomaly_splits_{suffix}.json")
    with open(out_path, "w") as fh:
        json.dump(splits, fh, indent=2)

    print(f"\n Split Statistics:")
    print(f"   Train (normal)   : {len(train_normal):>5}")
    print(f"   Val   (normal)   : {len(val_normal):>5}  ← threshold calibration")
    print(f"   Test  (normal)   : {len(test_normal):>5}")
    print(f"   Test  (anomalous): {n_anom:>5}")
    print(f"   Saved → {out_path}")
    return splits, out_path


# ── PyTorch Dataset ───────────────────────────────────────────────────────

class AgriVisionAnomalyDataset(data.Dataset):
    """
    Tile-based PyTorch Dataset for Agriculture-Vision-2021 anomaly detection.

    Each record is a dict {"original_split": "train"|"val", "tile_id": str}
    pointing to a tile in the Agriculture-Vision-2021 folder structure.

    Returns (per __getitem__)
    -------------------------
    image_tensor : Tensor (C, image_size, image_size)  float32 ∈ [0, 1]
                   then normalised by `transform`.
                   C = 3 (RGB) or 4 (RGBN) depending on use_nir.
    anomaly_mask : Tensor (512, 512) int64
                   1 on anomalous valid pixels; all-zeros for normal tiles.
                   Skipped (returned as zeros) when return_masks=False.
    valid_mask   : Tensor (512, 512) int64
                   1 on valid agricultural pixels.
    tile_id      : str

    Notes
    -----
    • image_tensor is resized to (image_size, image_size) by `transform`.
    • anomaly_mask and valid_mask are always at native 512×512 so they can
      be directly compared with upsampled heatmaps.
    • Set return_masks=False during memory-bank collection to skip anomaly-mask
      I/O (no anomaly labels needed for normal training tiles).
    """

    _SPLIT_TO_SUBDIR: Dict[str, str] = {"train": "train", "val": "val"}

    def __init__(
        self,
        tile_records: List[dict],
        data_root: str,
        anomaly_classes: List[str],
        *,
        use_nir: bool = False,
        transform: Optional[T.Compose] = None,
        apply_boundary: bool = False,
        return_masks: bool = True,
    ):
        self.records         = tile_records
        self.data_root       = data_root
        self.anomaly_classes = anomaly_classes
        self.use_nir         = use_nir
        self.transform       = transform
        self.apply_boundary  = apply_boundary
        self.return_masks    = return_masks

    def _split_dir(self, split: str) -> str:
        return os.path.join(self.data_root, self._SPLIT_TO_SUBDIR[split])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec   = self.records[idx]
        split = rec["original_split"]
        tid   = rec["tile_id"]
        base  = self._split_dir(split)

        # ── RGB image ──────────────────────────────────────────────────
        rgb_path = os.path.join(base, "images", "rgb", f"{tid}.jpg")
        rgb_img  = Image.open(rgb_path).convert("RGB")

        if self.use_nir:
            # Concatenate NIR as a 4th channel
            nir_path = os.path.join(base, "images", "nir", f"{tid}.png")
            if not os.path.exists(nir_path):
                raise FileNotFoundError(f"NIR image missing: {nir_path}")
            rgb_arr  = np.array(rgb_img, dtype=np.uint8)                   # (H,W,3)
            nir_arr  = np.array(Image.open(nir_path).convert("L"))[..., None]  # (H,W,1)
            image_tensor = (
                torch.from_numpy(np.concatenate([rgb_arr, nir_arr], axis=2))
                .permute(2, 0, 1).float() / 255.0
            )  # (4, H, W)
        else:
            image_tensor = TF.to_tensor(rgb_img)   # (3, H, W)  ∈ [0, 1]

        # ── Optional resize + normalise transform ──────────────────────
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # ── Masks  (always at native 512×512) ──────────────────────────
        valid_np     = load_valid_mask(tid, base)
        valid_tensor = torch.from_numpy(valid_np).long()

        if self.return_masks:
            anom_np = combined_anomaly_mask(
                tid, base, self.anomaly_classes, self.apply_boundary
            )
        else:
            anom_np = np.zeros((512, 512), dtype=np.uint8)   # dummy
        anom_tensor = torch.from_numpy(anom_np).long()

        return image_tensor, anom_tensor, valid_tensor, tid


# ── Per-channel mean / std ────────────────────────────────────────────────

def compute_agri_mean_std(
    train_records: List[dict],
    data_root: str,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
    use_nir: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Online computation of per-channel mean and std over all normal training tiles.
    Images are resized to (image_size × image_size) before accumulation.
    """
    resize_tf = _make_resize(image_size)
    ds = AgriVisionAnomalyDataset(
        train_records, data_root, anomaly_classes=[],
        use_nir=use_nir, transform=resize_tf, return_masks=False,
    )
    loader = data.DataLoader(ds, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    C    = 4 if use_nir else 3
    mean = torch.zeros(C, device=device)
    sq   = torch.zeros(C, device=device)
    npx  = 0
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(loader, desc="Computing mean/std"):
            imgs = imgs.float().to(device)
            B, c, h, w = imgs.shape
            npx  += B * h * w
            mean += imgs.sum(dim=[0, 2, 3])
            sq   += (imgs ** 2).sum(dim=[0, 2, 3])
    mean = mean / npx
    std  = (sq / npx - mean ** 2).clamp(min=1e-8).sqrt()
    return mean.cpu(), std.cpu()


# ═══════════════════════════════════════════════════════════════════════════
#  §4  NIR CHANNEL ADAPTER
#      Projects 4-channel RGBN input → 3-channel pseudo-RGB for DINOv2
# ═══════════════════════════════════════════════════════════════════════════

class NIRChannelAdapter(nn.Module):
    """
    Lightweight 1×1 convolution that maps RGBN (4-ch) to pseudo-RGB (3-ch).

    Initialisation strategy (zero-shot compatible):
        R → R,  G → G,  B → B  (identity pass-through for RGB channels)
        NIR → 0               (NIR contributes nothing until fine-tuned)

    This ensures the adapted output is initially identical to the plain RGB
    image, preserving all DINOv2 pretrained representations out of the box.

    To enable NIR signal in a supervised fine-tuning phase, set freeze=False
    and attach an optimiser to this module only.
    """

    def __init__(self, freeze: bool = True):
        super().__init__()
        self.proj = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        # Identity for RGB; zero weight for NIR
        w = torch.zeros(3, 4)
        w[0, 0] = w[1, 1] = w[2, 2] = 1.0
        self.proj.weight.data.copy_(w.view(3, 4, 1, 1))
        if freeze:
            for p in self.proj.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 4, H, W)  →  (B, 3, H, W)"""
        return self.proj(x)


# ═══════════════════════════════════════════════════════════════════════════
#  §5  DINOV2 MULTI-LAYER TEACHER
#      Adapted from doc2 TeacherMultiLayer; extended with NIR support
# ═══════════════════════════════════════════════════════════════════════════

class TeacherMultiLayer(nn.Module):
    """
    Frozen DINOv2 ViT that exposes intermediate CLS tokens and patch tokens
    from a configurable subset of transformer blocks.

    Parameters
    ----------
    model_name      : timm identifier, e.g. 'vit_base_patch14_dinov2'
    selected_layers : block indices (0-based) to capture; None = all blocks
    use_nir         : if True, prepend NIRChannelAdapter before the ViT

    Returns (via forward)
    ----------------------
    cls_raw_list      : list[L]  each (B, D)   – raw CLS token per captured block
    patch_tokens_list : list[L]  each (B, P, D) – patch tokens per captured block
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        selected_layers: Optional[List[int]] = None,
        use_nir: bool = False,
    ):
        super().__init__()
        self.use_nir = use_nir
        if use_nir:
            self.nir_adapter = NIRChannelAdapter(freeze=True)

        self.vit = timm.create_model(model_name, pretrained=pretrained)
        for attr in ("head", "head_dist", "fc"):
            if hasattr(self.vit, attr):
                try:
                    setattr(self.vit, attr, nn.Identity())
                except Exception:
                    pass

        self.embed_dim = getattr(
            self.vit, "embed_dim", getattr(self.vit, "num_features", 768)
        )
        self.n_blocks = len(getattr(self.vit, "blocks", []))
        self.selected_layers = (
            list(range(self.n_blocks))
            if selected_layers is None
            else [l for l in selected_layers if 0 <= l < self.n_blocks]
        )

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        x : (B, C, H, W)   C = 3 (RGB) or 4 (RGBN; requires use_nir=True)
        """
        if self.use_nir and x.shape[1] == 4:
            x = self.nir_adapter(x)                     # (B, 3, H, W)

        x = self.vit.patch_embed(x)                     # (B, N_patches, D)
        B, N, _ = x.shape
        has_cls  = hasattr(self.vit, "cls_token")
        has_dist = hasattr(self.vit, "dist_token")

        if has_dist:
            tokens = torch.cat([
                self.vit.cls_token.expand(B, -1, -1),
                self.vit.dist_token.expand(B, -1, -1),
                x,
            ], dim=1)
        elif has_cls:
            tokens = torch.cat([self.vit.cls_token.expand(B, -1, -1), x], dim=1)
        else:
            tokens = x

        if hasattr(self.vit, "pos_embed"):
            tokens = tokens + self.vit.pos_embed.to(tokens.device)

        cls_raw_list: List[torch.Tensor]   = []
        patch_tokens_list: List[torch.Tensor] = []

        for i, block in enumerate(getattr(self.vit, "blocks", [])):
            tokens = block(tokens)
            if i in self.selected_layers:
                if has_dist:
                    c, p = tokens[:, :2].mean(1), tokens[:, 2:]
                elif has_cls:
                    c, p = tokens[:, 0], tokens[:, 1:]
                else:
                    c, p = tokens.mean(1), tokens
                cls_raw_list.append(c.contiguous())
                patch_tokens_list.append(p.contiguous())

        # Replace the LAST captured layer with the layer-normed output
        # for improved numerical stability (mirrors doc2 behaviour)
        if hasattr(self.vit, "norm") and cls_raw_list:
            normed = self.vit.norm(tokens)
            if has_dist:
                c_n, p_n = normed[:, :2].mean(1), normed[:, 2:]
            elif has_cls:
                c_n, p_n = normed[:, 0], normed[:, 1:]
            else:
                c_n, p_n = normed.mean(1), normed
            cls_raw_list[-1]      = c_n.contiguous()
            patch_tokens_list[-1] = p_n.contiguous()

        return cls_raw_list, patch_tokens_list


# ═══════════════════════════════════════════════════════════════════════════
#  §6  MEMORY BANK CONSTRUCTION
#      Adapted from doc2 collect_concatenated_memory_bank for tile datasets
# ═══════════════════════════════════════════════════════════════════════════

def _greedy_fps_rows(feat_np: np.ndarray, target_n: int) -> np.ndarray:
    """
    Greedy farthest-point sampling over ROWS (tiles).
    Selects `target_n` maximally diverse rows in the D-dimensional feature space.
    Operates entirely on CPU; O(target_n × N) time complexity.
    """
    N, _ = feat_np.shape
    if target_n >= N:
        return np.arange(N, dtype=np.int32)
    # Row-normalise to make the FPS metric cosine-distance equivalent
    rows  = feat_np / (np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-12)
    sel   = [0]
    dists = ((rows - rows[0:1]) ** 2).sum(axis=1)
    for _ in range(1, target_n):
        idx   = int(np.argmax(dists))
        sel.append(idx)
        dists = np.minimum(dists, ((rows - rows[idx : idx + 1]) ** 2).sum(axis=1))
    return np.array(sel, dtype=np.int32)

def collect_agri_memory_bank(
    teacher: TeacherMultiLayer,
    loader: data.DataLoader,
    device: torch.device,
    config: Config,
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], List[str], Optional[dict]]:
    """
    TWO-PASS memory bank construction to avoid OOM on patch tokens.

    Pass 1 : Collect CLS/concat features for ALL normal tiles  (~953 MB for 62 k tiles)
             Apply PCA + FPS → select `concat_fps_target` (e.g. 1024) tile indices
    Pass 2 : Re-run ONLY the selected tiles → collect their patch tokens
             (~4.3 GB for 1024 × 1369 × 768 × float32)

    Key change vs original:  patch_parts is never accumulated over the full
    training set; it is built only for the FPS-selected subset.
    """
    teacher.eval()

    # ─────────────────────────────────────────────────────────────────────
    # PASS 1 ─ Collect CLS / concat features only (no patch tokens yet)
    # ─────────────────────────────────────────────────────────────────────
    concat_parts: List[torch.Tensor]             = []
    layer_buckets: Optional[List[List[torch.Tensor]]] = None
    tid_list: List[str]                          = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Pass 1] CLS features"):
            imgs = batch[0].float().to(device)
            tids = batch[3]

            cls_list, _ = teacher(imgs)   # ← patch tokens discarded in Pass 1
            cls_cpu = [c.detach().cpu() for c in cls_list]

            concat_parts.append(torch.cat(cls_cpu, dim=1))   # (B, L*D)

            if layer_buckets is None:
                layer_buckets = [[] for _ in range(len(cls_cpu))]
            for i, c in enumerate(cls_cpu):
                layer_buckets[i].append(c)

            tid_list.extend(list(tids) if not isinstance(tids, list) else tids)

    per_image_concat = torch.cat(concat_parts, dim=0)           # (N, L*D)
    per_layer_raw    = [torch.cat(b, dim=0) for b in layer_buckets]
    N_full           = per_image_concat.shape[0]
    
    del concat_parts, layer_buckets
    gc.collect()
    print(f"[Pass 1] Collected {N_full} tiles,  "
          f"concat_dim = {per_image_concat.shape[1]}")

    # ── Optional PCA ─────────────────────────────────────────────────────
    pca_meta   = None
    concat_cpu = per_image_concat.clone()
    del per_image_concat
    gc.collect()

    k_pca = getattr(config, "concat_pca_dim", None)
    if k_pca is not None:
        arr  = concat_cpu.numpy().astype("float32")
        N, D = arr.shape
        k    = int(min(k_pca, N - 1, D))
        if 0 < k < D:
            try:
                mu   = arr.mean(axis=0, keepdims=True).astype("float32")
                _, _, Vt = np.linalg.svd(arr - mu, full_matrices=False)
                comps    = Vt[:k].astype("float32")
                concat_cpu = torch.from_numpy((arr - mu) @ comps.T)
                pca_meta   = {"mean": mu, "components": comps}
                print(f"[PCA] {D} → {k}")
                del arr, Xc, Vt
                gc.collect()
            except Exception as exc:
                print(f"[WARN] PCA failed ({exc}); skipping.")
                del arr, Xc, Vt
                gc.collect()

    # ── FPS: determine which tile indices to KEEP ─────────────────────────
    fps_target = getattr(config, "concat_fps_target", None)
    selected_indices: List[int] = list(range(N_full))          # default: keep all

    if fps_target is not None and fps_target < N_full:
        arr = concat_cpu.numpy().astype("float32")
        sel = _greedy_fps_rows(arr, fps_target).tolist()
        del arr
        gc.collect()
        selected_indices = sel
        concat_cpu    = concat_cpu[sel]
        per_layer_raw = [feat[sel] for feat in per_layer_raw]
        tid_list      = [tid_list[i] for i in sel]
        print(f"[FPS] {N_full} → {len(sel)} tiles selected for memory bank")
    else:
        print(f"[FPS] Skipped (target={fps_target}, N={N_full})")

    gc.collect()

    # ─────────────────────────────────────────────────────────────────────
    # PASS 2 ─ Collect patch tokens ONLY for the FPS-selected tiles
    # Memory cost: 1024 × 1369 × 768 × 4 bytes ≈ 4.3 GB  ✓
    # ─────────────────────────────────────────────────────────────────────
    per_image_last_patches = None

    if config.use_last_layer_patches_for_knn:
        M = len(selected_indices)
    
        # ── Probe patch shape with one sample (no full batch needed) ──────────
        probe_ds     = torch.utils.data.Subset(loader.dataset, [selected_indices[0]])
        probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=1)
        with torch.no_grad():
            probe_imgs = next(iter(probe_loader))[0].float().to(device)
            _, _probe_patches = teacher(probe_imgs)
            P_last = _probe_patches[-1].shape[1]   # spatial patches per image
            D_last = _probe_patches[-1].shape[2]   # feature dim
            del _probe_patches, probe_imgs

        print(f"[Pass 2] patch shape per image: ({P_last}, {D_last})")
    
        # ── FIX 1+3: Pre-allocate fp16 bank — write-in-place, zero torch.cat ─
        #   fp16 cost: M × P × D × 2 B
        #   e.g. 35 k × 1369 × 768 × 2 B ≈ 71 GB  (vs 278 GB peak before fix)
        #   e.g.  1 k × 1369 × 768 × 2 B ≈  2 GB  (typical agri fps_target)
        patch_bank = torch.empty(
            (M, P_last, D_last),
            dtype  = torch.float16,   # FIX 3: fp16
            device = "cpu",
        )
        write_ptr = 0
    
        # ── FIX 2: Subset DataLoader — only the FPS-selected images run through
        #   teacher; zero wasted forward passes on discarded samples ────────────
        subset_ds     = torch.utils.data.Subset(loader.dataset, selected_indices)
        subset_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size  = loader.batch_size or 16,
            shuffle     = False,
            num_workers = getattr(loader, "num_workers", 0),
            pin_memory  = False,   # already targeting CPU bank; pin_memory adds pressure
        )
    
        with torch.no_grad():
            for batch in tqdm(subset_loader, desc="[Pass 2] Patch tokens (FPS subset)"):
                imgs    = batch[0].float().to(device)
                B_cur   = imgs.shape[0]
                _, patch_list = teacher(imgs)
                # FIX 3: .half() before moving to CPU
                patches_cpu = patch_list[-1].detach().half().cpu()   # (B_cur, P, D) fp16
                patch_bank[write_ptr : write_ptr + B_cur] = patches_cpu
                write_ptr += B_cur
                del patches_cpu, patch_list   # release GPU tensor immediately
    
        per_image_last_patches = patch_bank   # (M, P_last, D_last)  fp16, no copy
        print(
            f"[Pass 2] Patch bank: {per_image_last_patches.shape}  "
            f"dtype={per_image_last_patches.dtype}  "
            f"≈ {per_image_last_patches.element_size() * per_image_last_patches.numel() / 1e9:.2f} GB"
        )

    print(f"[Memory bank] tiles={concat_cpu.shape[0]}, "
          f"concat_dim={concat_cpu.shape[1]}, "
          f"patch_cache={'yes' if per_image_last_patches is not None else 'no'}")

    return concat_cpu, per_layer_raw, per_image_last_patches, tid_list, pca_meta


# ═══════════════════════════════════════════════════════════════════════════
#  §7  KNN INDEX  (retained from doc2 with minor robustness additions)
# ═══════════════════════════════════════════════════════════════════════════

def build_knn_index(
    features_cpu: torch.Tensor,
    metric: str = "cosine",
    use_faiss: bool = True,
) -> dict:
    feats = (
        features_cpu.numpy().astype("float32")
        if isinstance(features_cpu, torch.Tensor)
        else np.asarray(features_cpu, dtype="float32")
    )
    N, D = feats.shape
    if use_faiss and _HAS_FAISS:
        if metric == "cosine":
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            idx   = faiss.IndexFlatIP(D)
            idx.add(feats / norms)
            return {"index": idx, "normalized": True, "metric": metric}
        else:
            idx = faiss.IndexFlatL2(D)
            idx.add(feats)
            return {"index": idx, "normalized": False, "metric": metric}
    else:
        if metric == "cosine":
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            return {"features": feats / norms, "normalized": True, "metric": metric}
        return {"features": feats, "normalized": False, "metric": metric}


def knn_search(index_obj: dict, query, k: int = 5):
    """
    Top-k nearest neighbour search.
    Returns (scores, indices) where:
        cosine → scores = inner-product similarities (higher = more similar)
        l2     → scores = Euclidean distances       (lower  = more similar)
    """
    q = (
        query.detach().cpu().numpy()
        if isinstance(query, torch.Tensor)
        else np.asarray(query)
    ).astype("float32").ravel()

    if "index" in index_obj:                            # FAISS path
        idx = index_obj["index"]
        if index_obj["normalized"]:
            q_n = q / (np.linalg.norm(q) + 1e-8)
            dists, inds = idx.search(q_n.reshape(1, -1), k)
            return dists.ravel(), inds.ravel()
        else:
            dists, inds = idx.search(q.reshape(1, -1), k)
            return np.sqrt(np.maximum(dists, 0)).ravel(), inds.ravel()
    else:                                               # Torch / NumPy fallback
        feats = index_obj["features"]
        if index_obj["normalized"]:
            q_n  = q / (np.linalg.norm(q) + 1e-8)
            sims = feats @ q_n
            inds = np.argsort(-sims)[:k]
            return sims[inds], inds
        else:
            d2   = ((feats - q) ** 2).sum(1)
            inds = np.argsort(d2)[:k]
            return np.sqrt(d2[inds]), inds


# ═══════════════════════════════════════════════════════════════════════════
#  §8  SCORING HELPERS  (adapted from doc2)
# ═══════════════════════════════════════════════════════════════════════════

def score_cls_vs_memory(cls_vec, prototypes, metric='cosine',
                                   mu=None, cov_inv=None, k=5):
    if cls_vec.device != prototypes.device:
        cls_vec = cls_vec.to(prototypes.device)

    cls = cls_vec.unsqueeze(0)  # (1, D)

    if metric == 'cosine':
        cls_n = F.normalize(cls, dim=1)
        prot_n = F.normalize(prototypes, dim=1)
        sim = torch.matmul(cls_n, prot_n.t())  # (1, K)
        # take top-k highest similarities
        topk_sim, _ = torch.topk(sim, k=min(k, sim.shape[1]), dim=1)
        # convert similarity to distance
        return float(1.0 - topk_sim.mean().item())

    elif metric == 'l2':
        dists = torch.cdist(cls, prototypes)  # (1, K)
        # take k smallest distances
        topk_dists, _ = torch.topk(dists, k=min(k, dists.shape[1]),
                                  largest=False, dim=1)
        return float(topk_dists.mean().item())

    elif metric == 'mahalanobis':
        # NOTE: Mahalanobis here is against a single distribution,
        # so k-NN doesn't directly apply unless you have multiple mus.
        d = (cls.squeeze(0).cpu() - mu.cpu()).unsqueeze(0)
        m = (d @ cov_inv.cpu() @ d.t()).squeeze().item()
        return float(m)

    else:
        raise ValueError('Unknown metric')


def aggregate_layer_scores(scores: List[float], reduce: str = "mean") -> float:
    vals = np.array(scores, dtype=np.float32)
    return float(vals.max() if reduce == "max" else vals.mean())

def save_topk_overlay_plots(
    records: List[dict],
    mean,
    std,
    out_dir: str,
    k: int = 5,
):
    """
    Save top-k highest CLS-scoring normal and anomalous tiles.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _save_subset(subset: List[dict], subset_name: str):
        subset = sorted(subset, key=lambda r: r["cls_score"], reverse=True)[:k]
        for rank, rec in enumerate(subset, start=1):
            fname = (
                f"{subset_name}_rank{rank:02d}_"
                f"{rec['tile_id']}_cls{rec['cls_score']:.4f}.png"
            )
            title = (
                f"{subset_name.upper()} | rank={rank} | "
                f"cls={rec['cls_score']:.4f} | tile={rec['tile_id']}"
            )
            plot_tile_heatmap_overlay(
                rec["img"],
                rec["heatmap"],
                anomaly_mask=rec["mask"],
                mean=mean,
                std=std,
                title=title,
                save_path=os.path.join(out_dir, fname),
            )

    normals = [r for r in records if r["label"] == 0]
    anoms   = [r for r in records if r["label"] == 1]

    _save_subset(normals, "normal")
    _save_subset(anoms, "anomalous")

def plot_tile_heatmap_overlay(img_tensor, heatmap, anomaly_mask=None,
                             mean=None, std=None,
                             alpha=0.6, title=None, save_path=None):
    """
    Clean visualization: input + heatmap (+ GT if available)
    """
    img = img_tensor.detach().cpu()

    if mean is not None and std is not None:
        mean = torch.tensor(mean).view(-1,1,1)
        std  = torch.tensor(std).view(-1,1,1)
        img = img * std + mean

    img = img.clamp(0,1).permute(1,2,0).numpy()

    plt.figure(figsize=(12,4))

    # Input
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input")
    plt.axis("off")

    # Heatmap
    plt.subplot(1,3,2)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.title("Anomaly Map")
    plt.axis("off")

    # GT (optional)
    plt.subplot(1,3,3)
    if anomaly_mask is not None:
        plt.imshow(anomaly_mask, cmap='gray')
        plt.title("GT Mask")
    else:
        plt.text(0.5, 0.5, "No GT", ha='center')
    plt.axis("off")

    if title:
        plt.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
# ═══════════════════════════════════════════════════════════════════════════
#  §9  PER-TILE PATCH HEATMAP
#      Core change vs doc2: works on tile tensors instead of video batches;
#      always upsamples to 512×512 for comparison with ground-truth masks.
# ═══════════════════════════════════════════════════════════════════════════
# robust_patch_score (Eq. 7) imported from scoring.py

def compute_tile_heatmap(
    cls_raw_list: List[torch.Tensor],
    patch_tokens_list: List[torch.Tensor],
    per_image_concat_cpu: torch.Tensor,
    concat_index_obj: dict,
    per_image_last_patches_cpu: torch.Tensor,
    config: Config,
    device: torch.device,
    pca_meta: Optional[dict] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute anomaly heatmap for a single test tile.

    Steps
    -----
    1. Build a CLS query vector by concatenating raw CLS tokens from all
       selected layers.  Apply PCA projection if pca_meta is provided.
    2. Retrieve K nearest tiles from the memory bank using the KNN index.
    3. Compare the test tile's last-layer patch tokens with the retrieved
       tiles' patch tokens using cosine discrepancy (per patch).
    4. Reshape the patch-level discrepancy map and bilinear-upsample to
       (original_tile_size × original_tile_size) = 512×512.

    Returns
    -------
    heatmap_512   : (512, 512) float32  – higher = more anomalous
    concat_score  : float               – tile-level score from KNN
    patch_score   : float               – mean patch discrepancy
    """
    # ── 1. Build concat CLS query ────────────────────────────────────────
    query = (
        torch.cat([c[0].detach().cpu() for c in cls_raw_list])
        .numpy()
        .astype("float32")
    )
    if pca_meta is not None:
        query = np.dot(
            query - pca_meta["mean"].ravel(), pca_meta["components"].T
        ).astype("float32")

    dists, inds = knn_search(concat_index_obj, query, k=config.knn_k)
    k_inds = [int(i) for i in inds]

    # ── 2. Tile-level score ──────────────────────────────────────────────
    concat_score = (
        float(1.0 - np.mean(dists))
        if config.knn_metric == "cosine"
        else float(np.mean(dists))
    )

    # ── 3. Retrieve nearest patch tokens from cache ──────────────────────
    assert per_image_last_patches_cpu is not None, (
        "per_image_last_patches_cpu is required.  "
        "Set config.use_last_layer_patches_for_knn = True."
    )
    retrieved = per_image_last_patches_cpu[k_inds].to(device).float()   # (K, P, D)

    # ── 4. Per-patch cosine discrepancy ──────────────────────────────────
    test_patches = patch_tokens_list[-1][[0]].to(device)          # (1, P, D)
    P_last = test_patches.shape[1]

    if config.knn_metric == "cosine":
        t_n   = F.normalize(test_patches, dim=2).squeeze(0)        # (P, D)
        r_n   = F.normalize(retrieved, dim=2).reshape(-1, retrieved.shape[2])  # (K*P, D)
        sims  = t_n @ r_n.t()                                       # (P, K*P)
        # Per-patch anomaly score: 1 - max similarity over all K*P neighbours
        disc  = (1.0 - sims.max(dim=1).values).clamp(0.0, 2.0)    # (P,)
    else:
        test_exp = test_patches.expand(retrieved.shape[0], -1, -1)
        disc = ((retrieved - test_exp) ** 2).sum(2).min(0).values.sqrt()   # (P,)

    #patch_score = float(disc.mean().cpu().item())
    patch_score = robust_patch_score(disc)

    # ── 5. Reshape & upsample to native tile resolution (512×512) ────────
    grid = int(math.isqrt(P_last))
    assert grid * grid == P_last, (
        f"Patch count P={P_last} is not a perfect square.  "
        f"Ensure image_size ({config.image_size}) is divisible by patch_size (14)."
    )

    heatmap_512 = (
        F.interpolate(
            disc.view(1, 1, grid, grid).float(),
            size=(config.original_tile_size, config.original_tile_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )  # (512, 512) float32
    heatmap_512 = gaussian_filter(heatmap_512, sigma=4).astype(np.float32)

    return heatmap_512, concat_score, patch_score


# ═══════════════════════════════════════════════════════════════════════════
#  §10  EVALUATION
#       Tile-level AUROC (3 scoring variants) + Pixel-level AUROC
#       Key addition vs doc2: pixel-level evaluation using AgriVision masks
# ═══════════════════════════════════════════════════════════════════════════
def _safe_auroc_(
    scores: List[float],
    labels: np.ndarray,
    name: str,
) -> Optional[float]:
    """
    Memory-safe, numerically stable AUROC computation.

    Matches sklearn.metrics.roc_auc_score for binary labels
    using a rank-based (Mann–Whitney U) formulation.

    - O(N log N) time (sorting)
    - O(N) memory
    - No label binarization or dense allocations
    """

    s = np.asarray(scores, dtype=np.float32)#np.asarray(scores, dtype=np.float64)
    l = np.asarray(labels)

    # --- Robust label handling (match sklearn expectations) ---
    unique = np.unique(l)
    if unique.size != 2:
        print(f"  [WARN] {name}: requires exactly 2 classes, got {unique}.")
        return None

    # Map labels to {0,1} with positive = max label (sklearn convention)
    pos_label = unique.max()
    l_bin = (l == pos_label).astype(np.int8)

    n_pos = int(l_bin.sum())
    n = len(l_bin)
    n_neg = n - n_pos

    if n_pos == 0 or n_neg == 0:
        print(f"  [WARN] {name}: trivial labels (pos={n_pos}/{n}), AUC undefined.")
        return None

    # --- Rank computation with tie handling (vectorized) ---
    order = np.argsort(s, kind="mergesort")  # stable sort
    s_sorted = s[order]

    # Find tie groups
    diff = np.diff(s_sorted)
    tie_starts = np.concatenate(([0], np.where(diff != 0)[0] + 1))
    tie_ends = np.concatenate((tie_starts[1:], [n]))

    ranks = np.empty(n, dtype=np.float64)

    # Assign average rank per tie group (1-based ranks)
    for start, end in zip(tie_starts, tie_ends):
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank

    # --- Mann–Whitney U statistic ---
    rank_sum_pos = ranks[l_bin == 1].sum()

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(np.clip(auc, 0.0, 1.0))

def _safe_auroc(
    scores: List[float],
    labels: np.ndarray,
    name: str,
) -> Optional[float]:
    """
    Compute AUROC safely.

    For large arrays (>~10M elements), sklearn's roc_auc_score internally
    calls label_binarize which allocates an (N, 2) int64 dense matrix —
    fatal at pixel scale (~billions of elements). We bypass this entirely
    by computing the ROC curve manually via a sorted-rank approach, which
    is O(N log N) time and O(N) memory with no binarization overhead.
    """
    s = np.asarray(scores, dtype=np.float32)
    l = np.asarray(labels, dtype=np.int32)

    n_pos = int(l.sum())
    n_neg = len(l) - n_pos
    print('new auroc **')

    if n_pos == 0 or n_neg == 0:
        print(f"  [WARN] {name}: trivial labels "
              f"(pos={n_pos}/{len(l)}), AUC undefined.")
        return None

    # Normalise scores to [0, 1]  (not strictly needed but keeps parity)
    rng = float(s.max()) - float(s.min())
    if rng > 0:
        s = (s - s.min()) / rng

    # ── Memory-safe AUC via Wilcoxon-Mann-Whitney rank-sum ─────────
    # AUC = P(score of random positive > score of random negative)
    # Equivalent to (rank_sum_of_positives - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    # Uses only argsort + cumsum — no (N,2) allocation, no sparse matrix.
    # This is numerically identical to sklearn's result for binary labels.
    #
    # Reference: Hanley & McNeil (1982); Mason & Graham (2002)

    # Rank all scores (1-based); average ranks for ties
    order      = np.argsort(s, kind="mergesort")   # stable sort
    ranks      = np.empty(len(s), dtype=np.float64)
    
    # Assign averaged ranks to handle ties correctly
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0           # 1-based average rank
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = float(ranks[l == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(np.clip(auc, 0.0, 1.0))

def compute_aupro(
    gt_masks: List[np.ndarray],
    pred_maps: List[np.ndarray],
    fpr_limit: float = 0.3,
    num_steps: int = 300,
    min_region_area: int = 100,          # NEW: filter annotation noise
) -> float:

    # [FIX 1] Separate normal vs anomaly pixels for correct FPR
    normal_pred_parts  = []
    regions_pred_sorted = []

    for gt, pred in zip(gt_masks, pred_maps):
        # [FIX 2] Input validation
        assert gt.shape == pred.shape
        
        normal_mask = (gt < 0.5)
        normal_pred_parts.append(pred[normal_mask].ravel())  # ONLY normal pixels

        if gt.max() < 0.5:
            continue

        labeled_gt, num_r = scipy_label(gt > 0.5)
        for r_id in range(1, num_r + 1):
            region_mask = (labeled_gt == r_id)
            region_size = int(region_mask.sum())
            if region_size < min_region_area:   # [FIX 3] filter tiny regions
                continue
            sorted_vals = np.sort(
                pred[region_mask].ravel().astype(np.float32)
            )
            regions_pred_sorted.append(sorted_vals)

    if not regions_pred_sorted:
        warnings.warn("[AUPRO] No anomaly regions — returning NaN", RuntimeWarning)
        return float('nan')                     # [FIX 4] NaN not 0.0

    # [FIX 1 continued] Thresholds from NORMAL pixels only
    all_normal_flat = np.concatenate(normal_pred_parts).astype(np.float32)
    fpr_arr    = np.linspace(0.0, fpr_limit, num=num_steps, dtype=np.float64)
    pcts       = np.clip((1.0 - fpr_arr) * 100.0, 0.0, 100.0)
    thresh_arr = np.percentile(all_normal_flat, pcts).astype(np.float32)
    del all_normal_flat

    # [FIX 5] Enforce monotone decreasing thresholds
    thresh_arr = np.flip(np.minimum.accumulate(np.flip(thresh_arr)))

    R       = len(regions_pred_sorted)
    pro_sum = np.zeros(num_steps, dtype=np.float64)

    for sorted_r in regions_pred_sorted:
        S_r    = len(sorted_r)
        counts = S_r - np.searchsorted(sorted_r, thresh_arr, side="left")
        pro_sum += counts.astype(np.float64) / S_r

    pro_curve = (pro_sum / R).astype(np.float32)
    aupro     = float(np.trapz(pro_curve, fpr_arr) / fpr_limit)  # [FIX 6] no epsilon
    return aupro

def evaluate_agrivision(
    teacher: TeacherMultiLayer,
    splits: dict,
    config: Config,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    memory_bank: Optional[tuple] = None,
) -> Tuple[dict, tuple]:
    """
    End-to-end AgriVision evaluation pipeline.

    Steps
    -----
    1. Construct (or reuse) the memory bank from ``splits['train']``.
    2. Run inference on every tile in ``splits['test']``.
    3. Compute and log:
         - Tile-level AUROC  (CLS, concat-KNN, patch-mean scoring variants)
         - Pixel-level AUROC (heatmap scores restricted to valid_mask pixels)

    Parameters
    ----------
    memory_bank : optional pre-built tuple
        (per_image_concat, per_layer_raw, per_image_last_patches,
         tile_ids, pca_meta)
        Pass this to skip memory-bank construction (e.g. checkpoint eval).

    Returns
    -------
    results      : dict  {'tile_auroc_cls', 'tile_auroc_concat',
                          'tile_auroc_patch', 'pixel_auroc'}
    memory_bank  : tuple (same 5-element format as above)
    """
    # ── Shared image transform: resize → normalise ────────────────────────
    img_tf = T.Compose([
        _make_resize(config.image_size),
        T.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # ── 1. Build / restore memory bank ───────────────────────────────────
    if memory_bank is None:
        print("\n[1/2] Building memory bank from normal training tiles …")
        train_ds = AgriVisionAnomalyDataset(
            splits["train"],
            config.data_root,
            config.anomaly_classes,
            use_nir=config.use_nir,
            transform=img_tf,
            return_masks=False,                 # skip anomaly-mask I/O for speed
        )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        config.concat_fps_target = int(config.concat_fps_target * len(train_ds))
        #memory_bank = collect_agri_memory_bank(teacher, train_loader, device, config)
        memory_bank = collect_agri_memory_bank_fixed(teacher, train_loader, device, config)

    (per_image_concat, per_layer_raw,
     per_image_last_patches, tile_ids, pca_meta) = memory_bank

    concat_index = build_knn_index(
        per_image_concat, metric=config.knn_metric,
        use_faiss=config.use_faiss_index,
    )

    # ── 2. Test inference ────────────────────────────────────────────────
    print("\n[2/2] Evaluating on test split …")
    test_ds = AgriVisionAnomalyDataset(
        splits["test"],
        config.data_root,
        config.anomaly_classes,
        use_nir=config.use_nir,
        transform=img_tf,
        apply_boundary=config.apply_boundary,
        return_masks=True,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,                           # heatmap computation assumes B=1
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Pre-allocated pixel buffers  (FIX 4+5) ───────────────────────
    n_test        = len(splits["test"])
    max_px        = n_test * 512 * 512
    pixel_scores_buf = np.empty(max_px, dtype=np.float32)
    pixel_labels_buf = np.empty(max_px, dtype=np.uint8)
    pixel_ptr        = 0
    TILE_H = getattr(config, "original_tile_size", 512)             # ← defined after use
    TILE_W = TILE_H
    gt_buf   = np.empty((n_test, TILE_H, TILE_W), dtype=np.float32)
    pred_buf = np.empty((n_test, TILE_H, TILE_W), dtype=np.float32)
    tile_ptr = 0

    # ── Bounded viz heaps  (FIX 1) ───────────────────────────────────
    TOP_K         = 5#getattr(config, "topk_save", 5)
    normal_heap   = []   # min-heap by cls_score: evicts lowest normal score
    anomalous_heap= []   # min-heap by cls_score: evicts lowest anom score

    tile_labels   : List[int]   = []
    cls_scores    : List[float] = []
    concat_scores : List[float] = []
    patch_scores  : List[float] = []
    

    teacher.eval()
    with torch.no_grad():
        for imgs, anom_masks, valid_masks, tids in tqdm(test_loader, desc="Test"):
            imgs = imgs.float().to(device)              # (1, C, 518, 518)
            cls_list, patch_list = teacher(imgs)

            # ── Tile-level CLS score (mean over selected layers) ─────────
            layer_sc = [
                score_cls_vs_memory(
                    cls_list[li][0].cpu(), per_layer_raw[li], config.scoring
                )
                for li in range(len(cls_list))
            ]
            cls_score = aggregate_layer_scores(layer_sc, reduce="mean")
            cls_scores.append(cls_score)

            # ── Patch heatmap + concat/patch tile scores ─────────────────
            hm512, c_score, p_score = compute_tile_heatmap(
                cls_list, patch_list,
                per_image_concat, concat_index,
                per_image_last_patches,
                config, device, pca_meta=pca_meta,
            )
            # ── Visualization hook (NEW, clean separation) ──
            if config.display_tiles:
                tile_label = int((anom_masks[0] & valid_masks[0]).any())
            
                should_plot = True
            
                if config.display_anomalous_only:
                    should_plot = tile_label == 1
            
                if config.display_every_n > 1:
                    idx_global = len(cls_scores)
                    should_plot = should_plot and (idx_global % config.display_every_n == 0)
            
                if should_plot:
                    save_path = None
            
                    plot_tile_heatmap_overlay(
                        imgs[0],
                        hm512,
                        anomaly_mask=anom_masks[0].numpy(),
                        mean=mean.tolist(),
                        std=std.tolist(),
                        title=f"Tile: {tids[0]} | Label: {tile_label}",
                        save_path=save_path
                    )
                    
            concat_scores.append(c_score)
            patch_scores.append(p_score)

            # ── Ground-truth tile label ───────────────────────────────────
            # A tile is "anomalous" if it contains ≥1 anomalous valid pixel
            anom_np  = anom_masks[0].numpy()            # (512, 512)
            valid_np = valid_masks[0].numpy()           # (512, 512)
            tile_labels.append(int((anom_np & valid_np).any()))

            # ── NEW: collect visualization data ──
            # FIX 1: always collect viz record (no display_tiles gate)
            label = int((anom_np & valid_np).any())
            # FIX 1: bounded heap instead of unbounded list
            record = {
                "img"      : imgs[0].detach().cpu(),
                "heatmap"  : hm512.copy(),
                "mask"     : anom_np.copy(),
                "label"    : label,
                "cls_score": float(cls_score),
                "tile_id"  : tids[0],
            }
            score = -float(cls_score) if label == 0 else float(cls_score)
            heap = normal_heap if label == 0 else anomalous_heap
            heapq.heappush(heap, (score, len(tile_labels), record))
            if len(heap) > TOP_K:
                heapq.heappop(heap)

            # ── Pixel-level accumulation (valid pixels only) ──────────────
            # This is the core AgriVision-specific evaluation step:
            # heatmap pixel scores are compared against the pixel-level GT
            # annotations, restricted to agriculturally-valid pixels.
            if config.pixel_eval:
                valid_flat = valid_np.astype(bool).ravel()
                n_valid    = int(valid_flat.sum())
                pixel_scores_buf[pixel_ptr:pixel_ptr+n_valid] = \
                    hm512.ravel()[valid_flat]
                pixel_labels_buf[pixel_ptr:pixel_ptr+n_valid] = \
                    anom_np.ravel()[valid_flat]
                pixel_ptr += n_valid
                
                valid_f32 = valid_np.astype(np.float32)
                np.multiply(anom_np,  valid_f32, out=gt_buf[tile_ptr])           # in-place
                np.multiply(hm512,    valid_f32, out=pred_buf[tile_ptr])         # in-place
                tile_ptr += 1

            del imgs, cls_list, patch_list, hm512

    # ── Metrics ───────────────────────────────────────────────────────
    tl      = np.array(tile_labels)
    results = {
        "tile_auroc_cls"   : _safe_auroc(cls_scores,    tl, "CLS"),
        "tile_auroc_concat": _safe_auroc(concat_scores, tl, "concat"),
        "tile_auroc_patch" : _safe_auroc(patch_scores,  tl, "patch"),
    }

    if config.pixel_eval and pixel_ptr > 0:
        ps = pixel_scores_buf[:pixel_ptr]
        pl = pixel_labels_buf[:pixel_ptr].astype(np.int32)
        results["pixel_auroc"] = _safe_auroc(ps, pl, "pixel")
        print(f"  {'pixel_auroc':<30}: {results['pixel_auroc']:.4f}")

        n_valid_px = pixel_ptr  # Total valid pixels accumulated
        n_anom_px = int(pl.sum())  # Sum of binary labels
        print(f"\n  Valid pixels evaluated : {n_valid_px:>10,}")
        print(f"  Anomalous pixels       : {n_anom_px:>10,}  "
        f"({100.0 * n_anom_px / max(n_valid_px, 1):.2f}%)")

        del ps, pl, pixel_scores_buf, pixel_labels_buf
        gc.collect()
        gt_view   = gt_buf[:tile_ptr]              # (T, H, W) float32 view
        pred_view = pred_buf[:tile_ptr]            # (T, H, W) float32 view
        results["aupro"] = compute_aupro(gt_view, pred_view)

    # ── Top-k plots from bounded heaps ────────────────────────────────
    viz_records = (
        [r for _, _, r in normal_heap] +
        [r for _, _, r in anomalous_heap]
    )
    suffix   = "_".join(config.anomaly_classes)
    save_dir = os.path.join(config.save_path, "topk_plots",
                            f"agri_anomaly_splits_{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    save_topk_overlay_plots(viz_records, mean.tolist(),
                            std.tolist(), save_dir, k=TOP_K)

    return results, memory_bank


# ═══════════════════════════════════════════════════════════════════════════
#  §11  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = Config()
    set_seed(cfg.seed)
    print_config(cfg)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # ── Resolve / build splits ───────────────────────────────────────────
    suffix      = "_".join(cfg.anomaly_classes)
    splits_path = cfg.splits_json or os.path.join(
        cfg.data_root, f"agri_anomaly_splits_{suffix}.json"
    )
    if os.path.exists(splits_path):
        with open(splits_path) as fh:
            splits = json.load(fh)
        print(f"Loaded splits from  {splits_path}")
    else:
        print("Generating new splits …")
        splits, splits_path = make_anomaly_detection_splits(
            cfg.data_root, cfg.anomaly_classes, seed=cfg.seed
        )
    print("Metadata:", splits["metadata"])

    # ── Build teacher (fully frozen) ─────────────────────────────────────
    teacher = TeacherMultiLayer(
        model_name=cfg.model_name,
        pretrained=True,
        selected_layers=cfg.selected_layers,
        use_nir=cfg.use_nir,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)

    os.makedirs(os.path.join(cfg.save_path, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(cfg.save_path, "checkpoints", "memory_bank.pth")

    # Call once in main() before the full run:
    verify_optimal_config(teacher, cfg, device)

    if cfg.train:
        # ── Compute per-channel normalisation statistics ───────────────
        print("\nComputing normalisation statistics over training tiles …")
        mean, std = compute_agri_mean_std(
            splits["train"], cfg.data_root, cfg.image_size, device,
            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
            use_nir=cfg.use_nir,
        )
        print(f"  mean = {[round(v, 4) for v in mean.tolist()]}")
        print(f"  std  = {[round(v, 4) for v in std.tolist()]}")

        # ── Build memory bank + evaluate ──────────────────────────────
        results, memory_bank = evaluate_agrivision(
            teacher, splits, cfg, mean, std, device, memory_bank=None)

        # ── Save checkpoint ────────────────────────────────────────────
        (per_image_concat, per_layer_raw,
         per_image_last_patches, tile_ids, pca_meta) = memory_bank
        #torch.save(
         #   {
          #      "mean": mean,
           #     "std":  std,
            #    "per_image_concat":       per_image_concat,
             #   "per_layer_raw":          per_layer_raw,
              #  "per_image_last_patches": per_image_last_patches,
               # "tile_ids":               tile_ids,
                #"pca_meta":               pca_meta,
                #"config":                 cfg.__dict__,
                #"results":                results,
            #},
            #ckpt_path,
        #)
        print(f"\nCheckpoint saved → {ckpt_path}")

    else:
        # ── Inference from saved checkpoint ───────────────────────────
        print(f"Loading checkpoint → {ckpt_path}")
        ck   = torch.load(ckpt_path, map_location="cpu")
        mean = ck["mean"]
        std  = ck["std"]
        mb   = (
            ck["per_image_concat"],
            ck["per_layer_raw"],
            ck.get("per_image_last_patches"),
            ck.get("tile_ids"),
            ck.get("pca_meta"),
        )
        results, _ = evaluate_agrivision(
            teacher, splits, cfg, mean, std, device, memory_bank=mb
        )

    print("\n═══ Final Results ═══")
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<30}: {v:.4f}")


if __name__ == "__main__":
    main()