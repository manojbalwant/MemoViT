"""
mvtec_dinov2_anomaly.py
=======================
Zero-shot, one-class anomaly detection on MVTec Anomaly Detection (AD)
using a frozen DINOv2 teacher and a multi-layer memory-bank KNN.

Dataset  : MVTec AD (15 categories, RGB, variable resolution)
Backbone : DINOv2 ViT-B/14  (pretrained, fully frozen)

MVTec AD folder layout (per category)
--------------------------------------
mvtec_ad/
  <category>/
    train/
      good/          ← ALL training images are defect-free
        *.png
    test/
      good/          ← normal test images  (label = 0)
        *.png
      <defect_A>/    ← anomalous test images (label = 1)
        *.png
      <defect_B>/
        *.png
    ground_truth/
      <defect_A>/    ← binary GT masks
        <stem>_mask.png
      <defect_B>/
        <stem>_mask.png

Pipeline
--------
1. Scan train/good/ → pure-normal training images (memory bank)
   Scan test/*/     → normal (good/) + anomalous (defect folders) test images
2. Build memory bank
   • Run frozen DINOv2 over all training images
   • Store per-image concatenated CLS tokens (selected layers)
     + last-layer patch tokens (optionally compressed via FPS / PCA)
3. Per test image
   a) Image-level score = 1 – max-cosine-sim(CLS_query, memory_bank)
                          aggregated over selected transformer layers
   b) Pixel heatmap    = per-patch cosine discrepancy vs K nearest
                          training image patch tokens, bilinear-upsampled
                          to (eval_mask_size × eval_mask_size)
4. Evaluate (per category, then optionally aggregated)
   • Image-level AUROC  (3 scoring variants: CLS, concat-KNN, patch-mean)
   • Pixel-level AUROC  (heatmap vs binary GT mask; ALL pixels are valid)
   • AUPRO              (per-region overlap, FPR ≤ 0.3)

Key differences from AgriVision version
-----------------------------------------
• No valid_mask / boundary_mask  — every pixel is evaluated
• No NIR channel / NIRChannelAdapter
• GT masks loaded from ground_truth/<defect>/ with `_mask` suffix
• Normal images carry an all-zero GT mask (no mask file on disk)
• split construction is trivial (MVTec pre-defines train/test)
• 15 independent categories; run one or all via Config.run_all_categories
• Image-level label derived from subfolder name ("good" → 0, else → 1)
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import json
import random
import math
import warnings
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

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False
import matplotlib.pyplot as plt
import heapq

# ═══════════════════════════════════════════════════════════════════════════
#  §0  DATASET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

# All 15 categories in the MVTec Anomaly Detection benchmark.
MVTEC_CATEGORIES: List[str] = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

# All 12 categories in the VisA benchmark.
# VisA (Visual Anomaly) — Zou et al., CVPR 2022
# "SPot-the-Difference Self-Supervised Pre-Training for Anomaly Detection
# and Segmentation"
#
# Folder layout (per category)
# ─────────────────────────────
# visa/
#   <category>/
#     Data/
#       Images/
#         Normal/          ← all normal images (train + test-normal share this pool)
#           *.JPG
#         Anomaly/
#           <defect_type>/ ← anomalous test images
#             *.JPG
#       Masks/
#         Anomaly/
#           <defect_type>/ ← binary pixel-level GT masks
#             *.png        ← same stem as the anomalous image
#
# Split strategy
# ──────────────
# VisA ships an official CSV (split_csv/1cls.csv) that assigns each
# Normal image to either "train" or "test".  When the CSV is present it
# is used; otherwise an 80 / 20 random split of the Normal folder is
# applied and the CSV is written for reproducibility.
VISA_CATEGORIES: List[str] = [
    "candle", "capsules", "cashew", "chewinggum",
    "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4",
    "pipe_fryum",
]

# ═══════════════════════════════════════════════════════════════════════════
#  §1  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class Config:
    """
    Central configuration for the MVTec / VisA DINOv2 anomaly detection pipeline.
    Edit paths and hyper-parameters directly here before running.

    dataset selector
    ────────────────
    Set  dataset = "mvtec"  to run on MVTec AD (15 categories).
    Set  dataset = "visa"   to run on VisA     (12 categories).
    The two datasets share every backbone, memory-bank, scoring, and
    evaluation component.  Only split construction and GT-mask loading
    differ between them.
    """
    # ── Dataset selector ──────────────────────────────────────────────────
    #  "mvtec" | "visa"
    dataset: str = "visa"

    # ── Dataset / Paths ───────────────────────────────────────────────────
    data_root: str   = "dataset/MVtech"       # MVTec root
    visa_data_root: str = "dataset/VisA"      # VisA root
    save_path: str   = "experiments/mvtec_dinov2/"
    #  Single category to evaluate (ignored when run_all_categories=True)
    category: str    = "bottle"
    #  Set to a JSON path to skip re-scanning (auto-derived when None)
    splits_json: str = None

    # ── Image ─────────────────────────────────────────────────────────────
    #  224 = 16 × 14 → perfectly divisible by DINOv2 ViT-B/14 patch size
    #  518 = 37 × 14 → higher-resolution alternative (more patches, slower)
    image_size: int     = 518
    #  Resolution at which GT masks and heatmaps are compared.
    #  Set equal to image_size so no secondary resize is needed.
    eval_mask_size: int = 518

    # ── DINOv2 backbone ───────────────────────────────────────────────────
    model_name: str      = "vit_base_patch14_dinov2"
    selected_layers: list = [7, 8, 9, 10, 11]    # ViT-B has 12 blocks (0–11)

    # ── Memory bank / KNN ─────────────────────────────────────────────────
    batch_size: int      = 16
    num_workers: int     = 4
    knn_k: int           = 5
    knn_metric: str      = "cosine"          # "cosine" | "l2"
    scoring: str         = "cosine"          # per-layer CLS distance metric
    use_faiss_index: bool = True and _HAS_FAISS
    #  Fraction (≤1.0) or absolute count of training images kept after FPS.
    #  0.25 → keep 25 % of normal training images in the memory bank.
    concat_fps_target: float = 0.25#0.1
    concat_pca_dim: int      = None          # None → no PCA
    use_last_layer_patches_for_knn: bool = True

    # ── Evaluation ────────────────────────────────────────────────────────
    pixel_eval: bool = True    # pixel-level AUROC + AUPRO
    image_eval: bool = True    # image-level AUROC

    # ── Multi-category run ────────────────────────────────────────────────
    #  True  → evaluate ALL categories of the selected dataset in sequence;
    #           a summary table is printed at the end.
    #  False → evaluate Config.category only.
    run_all_categories: bool = True

    # ── VisA-specific ─────────────────────────────────────────────────────
    #  Fraction of Normal images used for TEST (the rest go to train).
    #  Only applied when no official split CSV is found.
    visa_test_fraction: float = 0.2
    #  Path to the official VisA split CSV (split_csv/1cls.csv inside the
    #  visa_data_root).  None → auto-detect at visa_data_root/split_csv/1cls.csv
    visa_split_csv: str = None

    # ── Misc ──────────────────────────────────────────────────────────────
    seed: int        = 42
    train: bool      = True    # True = build bank; False = load checkpoint

    # ── Visualization ─────────────────────────────────────────────────────
    display_images: bool         = False
    display_anomalous_only: bool = False
    display_every_n: int         = 10


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


# ═══════════════════════════════════════════════════════════════════════════
#  §3  DATASET COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Low-level mask helpers ─────────────────────────────────────────────────

def read_mask(
    path: str,
    size: Optional[int] = None,
    nonzero_binarize: bool = False,
) -> Optional[np.ndarray]:
    """
    Load a PNG mask and binarise to uint8 {0, 1}.

    Parameters
    ----------
    path             : file path; returns None if absent
    size             : if given, nearest-neighbour resize to (size × size)
                       applied AFTER binarisation to preserve hard edges
    nonzero_binarize : if True  → any non-zero pixel is anomalous (VisA)
                       if False → pixel > 127 is anomalous (MVTec default)

    Binarisation strategies
    -----------------------
    MVTec (nonzero_binarize=False):
        Masks are stored as 0 / 255.  The > 127 threshold is robust to
        JPEG artefacts and is the standard MVTec convention.

    VisA (nonzero_binarize=True):
        Masks may use palette indices, anti-aliased edges, or low-value
        annotation markers (e.g. value 1 or 2) that encode genuine anomaly
        pixels.  These fall below the 127 threshold and would be silently
        dropped by the MVTec strategy.  Setting all non-zero values to 255
        before thresholding catches every annotated pixel regardless of its
        original numeric value:

            mask_array = np.array(mask)
            mask_array[mask_array != 0] = 255
            mask = Image.fromarray(mask_array)

        The > 127 step that follows is then effectively equivalent to != 0
        on the original values, but goes through the same binarisation code
        path so resize and dtype handling are consistent.
    """
    if not os.path.exists(path):
        return None

    img = Image.open(path).convert("L")     # grayscale, values in [0, 255]

    if nonzero_binarize:
        # VisA binarisation: any non-zero pixel → 255, zero stays 0.
        # This matches the reference snippet:
        #   mask_array = np.array(mask)
        #   mask_array[mask_array != 0] = 255
        #   mask = Image.fromarray(mask_array)
        mask_array = np.array(img, dtype=np.uint8)
        mask_array[mask_array != 0] = 255
        img = Image.fromarray(mask_array)

    # Common final step: threshold at 127 → binary {0, 1}
    mask = (np.array(img) > 127).astype(np.uint8)

    if size is not None and (mask.shape[0] != size or mask.shape[1] != size):
        mask = np.array(
            Image.fromarray(mask * 255).resize((size, size), Image.NEAREST)
        )
        mask = (mask > 127).astype(np.uint8)

    return mask


def load_mvtec_gt_mask(
    img_path: str,
    category_dir: str,
    defect_type: str,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Load the ground-truth binary mask for a single MVTec test image.

    MVTec GT layout
    ---------------
    {category_dir}/ground_truth/{defect_type}/{stem}_mask.png

    For normal images (defect_type == "good") there is no mask file;
    an all-zeros array of shape (size, size) is returned instead.

    Parameters
    ----------
    img_path     : absolute path to the test image
    category_dir : absolute path to the category root, e.g. .../mvtec_ad/bottle
    defect_type  : subfolder name, e.g. "broken_large" or "good"
    size         : target spatial resolution of the returned mask

    Returns
    -------
    (size, size) uint8 array — 1 on defect pixels, 0 elsewhere
    """
    h = size if size is not None else 224
    if defect_type == "good":
        return np.zeros((h, h), dtype=np.uint8)

    stem      = os.path.splitext(os.path.basename(img_path))[0]
    mask_path = os.path.join(
        category_dir, "ground_truth", defect_type, f"{stem}_mask.png"
    )
    mask = read_mask(mask_path, size=size)
    if mask is None:
        warnings.warn(
            f"GT mask not found: {mask_path} — treating as all-zeros.",
            RuntimeWarning,
        )
        return np.zeros((h, h), dtype=np.uint8)
    return mask


def get_mvtec_image_paths(
    category_dir: str,
    split: str,             # "train" | "test"
) -> List[Tuple[str, str, int]]:
    """
    Scan one MVTec split folder and return image records.

    Returns
    -------
    List of (img_path, defect_type, label):
        label = 0  →  "good"  (normal)
        label = 1  →  any other defect subfolder (anomalous)
    """
    split_dir = os.path.join(category_dir, split)
    records: List[Tuple[str, str, int]] = []
    for defect_type in sorted(os.listdir(split_dir)):
        defect_dir = os.path.join(split_dir, defect_type)
        if not os.path.isdir(defect_dir):
            continue
        label = 0 if defect_type == "good" else 1
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            for p in sorted(glob(os.path.join(defect_dir, ext))):
                records.append((p, defect_type, label))
    return records


# ── Split construction ─────────────────────────────────────────────────────

def make_mvtec_splits(
    data_root: str,
    category: str,
    seed: int = 42,
) -> Tuple[dict, str]:
    """
    Build train / test splits for one MVTec AD category.

    The MVTec benchmark pre-defines its split:
        train → train/good/    (all normal training images)
        test  → test/good/     (normal, label = 0)
                test/<defect>/ (anomalous, label = 1)

    No random partitioning is performed.  A JSON summary is saved for
    reproducibility and to allow fast reload on subsequent runs.

    Returns
    -------
    (splits_dict, path_to_saved_json)
    """
    category_dir = os.path.join(data_root, category)

    train_recs = get_mvtec_image_paths(category_dir, split="train")
    test_recs  = get_mvtec_image_paths(category_dir, split="test")

    def _to_dict(img_path: str, defect_type: str, label: int) -> dict:
        return {"img_path": img_path, "defect_type": defect_type, "label": label}

    train_list = [_to_dict(*r) for r in train_recs]
    test_list  = [_to_dict(*r) for r in test_recs]

    n_train          = len(train_list)
    n_test_normal    = sum(1 for r in test_list if r["label"] == 0)
    n_test_anomalous = sum(1 for r in test_list if r["label"] == 1)
    defect_types     = sorted({r["defect_type"] for r in test_list if r["label"] == 1})

    splits = {
        "train": train_list,
        "test":  test_list,
        "metadata": {
            "category":             category,
            "seed":                 seed,
            "train_count":          n_train,
            "test_normal_count":    n_test_normal,
            "test_anomalous_count": n_test_anomalous,
            "test_total":           len(test_list),
            "defect_types":         defect_types,
        },
    }
    out_path = os.path.join(data_root, f"{category}_mvtec_splits.json")
    #with open(out_path, "w") as fh:
        #json.dump(splits, fh, indent=2)

    print(f"\n MVTec Split Statistics ({category}):")
    print(f"   Train (normal)   : {n_train:>5}")
    print(f"   Test  (normal)   : {n_test_normal:>5}")
    print(f"   Test  (anomalous): {n_test_anomalous:>5}  [{', '.join(defect_types)}]")
    print(f"   Saved → {out_path}")
    return splits, out_path


# ═══════════════════════════════════════════════════════════════════════════
#  §3b  VISA DATASET COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

def load_visa_gt_mask(
    img_path: str,
    category_dir: str,
    defect_type: str,
    size: Optional[int] = None,
) -> np.ndarray:
    """
    Load the ground-truth binary mask for a single VisA test image.

    VisA actual GT layout (flat — no per-defect subfolder)
    -------------------------------------------------------
    {category_dir}/Data/Masks/Anomaly/{stem}.png

    The original implementation assumed a subfolder per defect type
    (mirroring MVTec's ground_truth/<defect>/ layout), which produced the
    doubled path  .../Masks/Anomaly/Anomaly/096.png  and the RuntimeWarning.

    Fix: always try the flat path first.  If missing, fall back to the
    subfolder path for any future VisA variants that do use subfolders.

    Parameters
    ----------
    img_path     : absolute path to the test image
    category_dir : absolute path to the category root, e.g. .../visa/cashew
    defect_type  : "normal" for normal images; any other value for anomalous
    size         : target spatial resolution of the returned mask

    Returns
    -------
    (size, size) uint8 array — 1 on defect pixels, 0 elsewhere
    """
    h = size if size is not None else 224
    if defect_type == "normal":
        return np.zeros((h, h), dtype=np.uint8)

    stem = os.path.splitext(os.path.basename(img_path))[0]

    # ── Priority 1: flat layout (standard VisA) ───────────────────────────
    # Data/Masks/Anomaly/<stem>.png  — no per-defect subfolder.
    # nonzero_binarize=True: any non-zero pixel is anomalous (VisA convention).
    flat_path = os.path.join(
        category_dir, "Data", "Masks", "Anomaly", f"{stem}.png"
    )
    mask = read_mask(flat_path, size=size, nonzero_binarize=True)
    if mask is not None:
        return mask

    # ── Priority 2: subfolder layout (non-standard / future variants) ─────
    # Data/Masks/Anomaly/<defect_type>/<stem>.png
    # Only attempted when defect_type is a meaningful directory name, not
    # the sentinel values "anomaly" / "Anomaly" assigned by the flat scanner.
    if defect_type.lower() not in ("anomaly", "normal"):
        sub_path = os.path.join(
            category_dir, "Data", "Masks", "Anomaly", defect_type, f"{stem}.png"
        )
        mask = read_mask(sub_path, size=size, nonzero_binarize=True)
        if mask is not None:
            return mask

    warnings.warn(
        f"VisA GT mask not found for '{stem}' "
        f"(tried flat and subfolder paths in "
        f"{os.path.join(category_dir, 'Data', 'Masks', 'Anomaly')}) "
        f"— treating as all-zeros.",
        RuntimeWarning,
    )
    return np.zeros((h, h), dtype=np.uint8)


def _scan_visa_dir(
    category_dir: str,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Scan the VisA directory tree for a single category and return image records
    without relying on a CSV file.

    VisA uses a flat anomaly layout — images sit directly inside
    Data/Images/Anomaly/ with no per-defect subfolder.  The original
    implementation only looked for subdirectories and therefore missed all
    anomalous images in the flat layout.

    This function handles both layouts:
      Flat     : Data/Images/Anomaly/*.JPG  → defect_type = "anomaly"
      Subfolder: Data/Images/Anomaly/<defect>/*.JPG → defect_type = <defect>

    Returns
    -------
    normal_recs   : list of (img_path, "normal", 0)   — all Normal images
    anomalous_recs: list of (img_path, defect_type, 1) — all Anomaly images
    """
    _EXTS = ("*.JPG", "*.jpg", "*.png", "*.jpeg", "*.bmp")

    normal_dir  = os.path.join(category_dir, "Data", "Images", "Normal")
    anomaly_dir = os.path.join(category_dir, "Data", "Images", "Anomaly")

    normal_recs: List[Tuple[str, str, int]] = []
    if os.path.isdir(normal_dir):
        for ext in _EXTS:
            for p in sorted(glob(os.path.join(normal_dir, ext))):
                normal_recs.append((p, "normal", 0))

    anomalous_recs: List[Tuple[str, str, int]] = []
    if os.path.isdir(anomaly_dir):
        # ── Flat layout: images directly in Anomaly/ ─────────────────────
        # Standard VisA layout — collect these first.
        flat_found = False
        for ext in _EXTS:
            for p in sorted(glob(os.path.join(anomaly_dir, ext))):
                anomalous_recs.append((p, "anomaly", 1))
                flat_found = True

        # ── Subfolder layout: images in Anomaly/<defect>/ ─────────────────
        # Only scan subdirectories when NO flat images were found, preventing
        # double-counting on datasets that mix both conventions.
        if not flat_found:
            for defect_type in sorted(os.listdir(anomaly_dir)):
                defect_dir = os.path.join(anomaly_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                for ext in _EXTS:
                    for p in sorted(glob(os.path.join(defect_dir, ext))):
                        anomalous_recs.append((p, defect_type, 1))

    return normal_recs, anomalous_recs


def get_visa_image_paths_from_csv(
    visa_root: str,
    category: str,
    csv_path: str,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Load VisA image records from the official split CSV.

    The official CSV (split_csv/1cls.csv) has five columns:
        split     : "train" | "test"
        label     : "normal" | "anomaly"
        image     : relative path from visa_root, e.g. candle/Data/Images/...
        mask      : relative mask path (empty string for normal images)
        (category): not always present; filtered by matching the image path

    Returns
    -------
    train_recs : list of (abs_img_path, "normal", 0)  — normal training images
    test_recs  : list of (abs_img_path, defect_type, label)
    """
    import csv

    train_recs: List[Tuple[str, str, int]] = []
    test_recs:  List[Tuple[str, str, int]] = []

    cat_prefix = category + os.sep   # e.g. "candle/"

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            img_rel = row.get("image", "").strip()
            # Filter to this category by checking whether the relative path
            # begins with the category directory name.
            if not (img_rel.startswith(cat_prefix) or
                    img_rel.startswith(category + "/")):
                continue

            abs_img = os.path.join(visa_root, img_rel)
            split   = row.get("split", "").strip().lower()
            label_s = row.get("label", "").strip().lower()
            mask_rel= row.get("mask", "").strip()
            label   = 0 if label_s == "normal" else 1

            # Derive defect_type from the mask path.
            # The official CSV mask column looks like:
            #   Flat layout    : cashew/Data/Masks/Anomaly/096.png
            #                    → parent dir is "Anomaly" → flat, use "anomaly"
            #   Subfolder layout: cashew/Data/Masks/Anomaly/<defect>/096.png
            #                    → parent dir is <defect> → use it directly
            if label == 1 and mask_rel:
                mask_parts  = mask_rel.replace("\\", "/").split("/")
                parent_dir  = mask_parts[-2] if len(mask_parts) >= 2 else ""
                if parent_dir.lower() == "anomaly":
                    # Flat layout — parent is the Anomaly/ dir itself.
                    # Assign the sentinel "anomaly" so load_visa_gt_mask
                    # knows to use the flat path.
                    defect_type = "anomaly"
                else:
                    # Subfolder layout — parent is a real defect folder name.
                    defect_type = parent_dir
            else:
                defect_type = "normal"

            rec = (abs_img, defect_type, label)
            if split == "train":
                train_recs.append(rec)
            else:
                test_recs.append(rec)

    return train_recs, test_recs


def make_visa_splits(
    visa_root: str,
    category: str,
    seed: int = 42,
    test_fraction: float = 0.2,
    csv_path: Optional[str] = None,
) -> Tuple[dict, str]:
    """
    Build train / test splits for one VisA category.

    Split strategy (in priority order)
    -----------------------------------
    1. Official CSV   (csv_path or auto-detected at visa_root/split_csv/1cls.csv)
       The CSV pre-assigns every image to "train" or "test" with exact labels.
    2. Directory scan + random 80/20 split of Normal images (fallback).
       All Anomaly images go to the test set unconditionally.

    VisA train set: normal images only (one-class setting).
    VisA test  set: held-out normal images (label=0) + all anomalous (label=1).

    Returns
    -------
    (splits_dict, path_to_saved_json)
    """
    category_dir = os.path.join(visa_root, category)

    # ── Try CSV first ─────────────────────────────────────────────────────
    auto_csv = csv_path or os.path.join(visa_root, "split_csv", "1cls.csv")
    used_csv = False

    if os.path.exists(auto_csv):
        print(f"[VisA] Using official split CSV: {auto_csv}")
        train_recs, test_recs = get_visa_image_paths_from_csv(
            visa_root, category, auto_csv
        )
        used_csv = True
    else:
        # ── Fallback: directory scan + random split ────────────────────────
        print(f"[VisA] CSV not found at {auto_csv}; using directory scan + "
              f"{int(test_fraction*100)}/{int((1-test_fraction)*100)} split.")
        normal_recs, anomalous_recs = _scan_visa_dir(category_dir)

        rng = random.Random(seed)
        rng.shuffle(normal_recs)
        n_test_normal = max(1, int(len(normal_recs) * test_fraction))
        test_normal   = normal_recs[:n_test_normal]
        train_recs    = normal_recs[n_test_normal:]
        test_recs     = test_normal + anomalous_recs

    def _to_dict(img_path: str, defect_type: str, label: int) -> dict:
        return {
            "img_path"   : img_path,
            "defect_type": defect_type,
            "label"      : label,
            "dataset"    : "visa",      # tag for mask-loader dispatch
        }

    train_list = [_to_dict(*r) for r in train_recs]
    test_list  = [_to_dict(*r) for r in test_recs]

    n_train          = len(train_list)
    n_test_normal    = sum(1 for r in test_list if r["label"] == 0)
    n_test_anomalous = sum(1 for r in test_list if r["label"] == 1)
    defect_types     = sorted({r["defect_type"] for r in test_list if r["label"] == 1})

    splits = {
        "train": train_list,
        "test" : test_list,
        "metadata": {
            "dataset"              : "visa",
            "category"             : category,
            "seed"                 : seed,
            "used_official_csv"    : used_csv,
            "train_count"          : n_train,
            "test_normal_count"    : n_test_normal,
            "test_anomalous_count" : n_test_anomalous,
            "test_total"           : len(test_list),
            "defect_types"         : defect_types,
        },
    }

    out_path = os.path.join(visa_root, f"{category}_visa_splits.json")
    #with open(out_path, "w") as fh:
        #json.dump(splits, fh, indent=2)

    print(f"\n VisA Split Statistics ({category}):")
    print(f"   Train (normal)   : {n_train:>5}")
    print(f"   Test  (normal)   : {n_test_normal:>5}")
    print(f"   Test  (anomalous): {n_test_anomalous:>5}  [{', '.join(defect_types)}]")
    print(f"   Saved → {out_path}")
    return splits, out_path


# ── PyTorch Dataset (unified: MVTec + VisA) ───────────────────────────────

class AnomalyDataset(data.Dataset):
    """
    Unified image-level PyTorch Dataset for MVTec AD and VisA.

    Each record is a dict with keys:
        "img_path"   : absolute path to the image file
        "defect_type": "good" (MVTec) / "normal" (VisA) for normal;
                       defect subfolder name for anomalous
        "label"      : int  0 (normal) | 1 (anomalous)
        "dataset"    : "mvtec" | "visa"  (controls GT mask loading)

    Returns (per __getitem__)
    -------------------------
    image_tensor : Tensor (3, image_size, image_size)  float32, normalised
    gt_mask      : Tensor (eval_mask_size, eval_mask_size) int64
                   Binary GT mask; all-zeros for normal images.
    label        : int  0 | 1
    img_path     : str  (for identification / visualisation)

    GT mask loading dispatch
    ─────────────────────────
    MVTec: {category_dir}/ground_truth/{defect_type}/{stem}_mask.png
    VisA : {category_dir}/Data/Masks/Anomaly/{defect_type}/{stem}.png

    Notes
    ─────
    • All images are resized to (image_size × image_size) by `transform`.
    • GT masks are resized to (eval_mask_size × eval_mask_size) using
      nearest-neighbour interpolation to preserve binary values.
    • return_masks=False skips GT mask I/O during memory-bank collection.
    • Every pixel is valid for both MVTec and VisA — no valid_mask needed.
    """

    def __init__(
        self,
        records: List[dict],
        category_dir: str,
        *,
        image_size: int = 224,
        eval_mask_size: int = 224,
        transform: Optional[T.Compose] = None,
        return_masks: bool = True,
        dataset: str = "mvtec",     # "mvtec" | "visa"
    ):
        self.records        = records
        self.category_dir   = category_dir
        self.image_size     = image_size
        self.eval_mask_size = eval_mask_size
        self.transform      = transform
        self.return_masks   = return_masks
        self.dataset        = dataset

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec         = self.records[idx]
        img_path    = rec["img_path"]
        defect_type = rec["defect_type"]
        label       = rec["label"]
        # Per-record dataset tag takes priority over the class-level default.
        ds          = rec.get("dataset", self.dataset)

        # ── RGB image ──────────────────────────────────────────────────────
        image_tensor = TF.to_tensor(
            Image.open(img_path).convert("RGB")
        )   # (3, H, W)  ∈ [0, 1]

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # ── Ground-truth mask ──────────────────────────────────────────────
        if self.return_masks:
            if ds == "visa":
                gt_np = load_visa_gt_mask(
                    img_path, self.category_dir, defect_type,
                    size=self.eval_mask_size,
                )
            else:   # mvtec (default)
                gt_np = load_mvtec_gt_mask(
                    img_path, self.category_dir, defect_type,
                    size=self.eval_mask_size,
                )
        else:
            gt_np = np.zeros(
                (self.eval_mask_size, self.eval_mask_size), dtype=np.uint8
            )
        gt_tensor = torch.from_numpy(gt_np).long()

        return image_tensor, gt_tensor, label, img_path


# Backward-compatible alias so any remaining MVTecDataset references still work.
MVTecDataset = AnomalyDataset


# ── Per-channel mean / std ────────────────────────────────────────────────

def compute_mvtec_mean_std(
    train_records: List[dict],
    category_dir: str,
    image_size: int,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Online per-channel mean and std over all normal training images.
    Images are resized to (image_size × image_size) before accumulation.
    """
    resize_tf = _make_resize(image_size)
    ds = MVTecDataset(
        train_records, category_dir,
        image_size=image_size, eval_mask_size=image_size,
        transform=resize_tf, return_masks=False,
    )
    loader = data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    mean = torch.zeros(3, device=device)
    sq   = torch.zeros(3, device=device)
    npx  = 0
    with torch.no_grad():
        for imgs, _, _, _ in tqdm(loader, desc="Computing mean/std"):
            imgs  = imgs.float().to(device)
            B, c, h, w = imgs.shape
            npx  += B * h * w
            mean += imgs.sum(dim=[0, 2, 3])
            sq   += (imgs ** 2).sum(dim=[0, 2, 3])
    mean = mean / npx
    std  = (sq / npx - mean ** 2).clamp(min=1e-8).sqrt()
    return mean.cpu(), std.cpu()


# ═══════════════════════════════════════════════════════════════════════════
#  §4  DINOV2 MULTI-LAYER TEACHER
#      NIRChannelAdapter removed — MVTec is RGB-only.
# ═══════════════════════════════════════════════════════════════════════════

class TeacherMultiLayer(nn.Module):
    """
    Frozen DINOv2 ViT that exposes intermediate CLS tokens and patch tokens
    from a configurable subset of transformer blocks.

    Parameters
    ----------
    model_name      : timm identifier, e.g. 'vit_base_patch14_dinov2'
    selected_layers : block indices (0-based) to capture; None = all blocks

    Returns (via forward)
    ----------------------
    cls_raw_list      : list[L]  each (B, D)    – raw CLS token per block
    patch_tokens_list : list[L]  each (B, P, D) – patch tokens per block
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2",
        pretrained: bool = True,
        selected_layers: Optional[List[int]] = None,
    ):
        super().__init__()
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
        """x : (B, 3, H, W)"""
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

        cls_raw_list: List[torch.Tensor]      = []
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

        # Replace last captured layer with layer-normed output for
        # improved numerical stability.
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
#  §5  MEMORY BANK CONSTRUCTION  (dataset-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def _greedy_fps_rows(feat_np: np.ndarray, target_n: int) -> np.ndarray:
    """
    Greedy farthest-point sampling over rows.
    Selects `target_n` maximally diverse rows in the D-dimensional feature space.
    CPU-only; O(target_n × N) time complexity.
    """
    N, _ = feat_np.shape
    if target_n >= N:
        return np.arange(N, dtype=np.int32)
    rows  = feat_np / (np.linalg.norm(feat_np, axis=1, keepdims=True) + 1e-12)
    sel   = [0]
    dists = ((rows - rows[0:1]) ** 2).sum(axis=1)
    for _ in range(1, target_n):
        idx   = int(np.argmax(dists))
        sel.append(idx)
        dists = np.minimum(dists, ((rows - rows[idx : idx + 1]) ** 2).sum(axis=1))
    return np.array(sel, dtype=np.int32)


def collect_memory_bank(
    teacher: TeacherMultiLayer,
    loader: data.DataLoader,
    device: torch.device,
    config: Config,
) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor], List[str], Optional[dict]]:
    """
    TWO-PASS memory bank construction.

    Pass 1 : CLS / concat features for ALL normal training images.
             Apply optional PCA + FPS → select a diverse subset of images.
    Pass 2 : Patch tokens ONLY for the FPS-selected subset (fp16,
             pre-allocated; avoids OOM from accumulating all patches).

    Returns
    -------
    (concat_cpu, per_layer_raw, patch_bank, img_id_list, pca_meta)
        concat_cpu      : (M, L*D)  float32  – FPS-selected CLS concat
        per_layer_raw   : list[L]  each (M, D) – per-layer CLS features
        patch_bank      : (M, P, D) float16  – last-layer patch tokens
        img_id_list     : list[M] img_path strings for the selected images
        pca_meta        : dict {"mean", "components"} or None
    """
    teacher.eval()

    # ── Pass 1: CLS features ─────────────────────────────────────────────
    concat_parts: List[torch.Tensor]                   = []
    layer_buckets: Optional[List[List[torch.Tensor]]]  = None
    id_list: List[str]                                  = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="[Pass 1] CLS features"):
            imgs = batch[0].float().to(device)
            ids  = batch[3]                             # img_path strings

            cls_list, _ = teacher(imgs)                 # patch tokens discarded
            cls_cpu = [c.detach().cpu() for c in cls_list]
            concat_parts.append(torch.cat(cls_cpu, dim=1))   # (B, L*D)

            if layer_buckets is None:
                layer_buckets = [[] for _ in range(len(cls_cpu))]
            for i, c in enumerate(cls_cpu):
                layer_buckets[i].append(c)

            id_list.extend(list(ids) if not isinstance(ids, list) else ids)

    per_image_concat = torch.cat(concat_parts, dim=0)          # (N, L*D)
    per_layer_raw    = [torch.cat(b, dim=0) for b in layer_buckets]
    N_full           = per_image_concat.shape[0]
    del concat_parts, layer_buckets
    gc.collect()
    print(f"[Pass 1] Collected {N_full} images, concat_dim={per_image_concat.shape[1]}")

    # ── Optional PCA ──────────────────────────────────────────────────────
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
                mu       = arr.mean(axis=0, keepdims=True).astype("float32")
                _, _, Vt = np.linalg.svd(arr - mu, full_matrices=False)
                comps    = Vt[:k].astype("float32")
                concat_cpu = torch.from_numpy((arr - mu) @ comps.T)
                pca_meta   = {"mean": mu, "components": comps}
                print(f"[PCA] {D} → {k}")
                del arr, Vt
                gc.collect()
            except Exception as exc:
                print(f"[WARN] PCA failed ({exc}); skipping.")

    # ── FPS: select a diverse subset of images ────────────────────────────
    fps_target       = getattr(config, "concat_fps_target", None)
    selected_indices = list(range(N_full))               # default: keep all

    if fps_target is not None and fps_target < N_full:
        arr = concat_cpu.numpy().astype("float32")
        sel = _greedy_fps_rows(arr, fps_target).tolist()
        del arr
        gc.collect()
        selected_indices = sel
        concat_cpu    = concat_cpu[sel]
        per_layer_raw = [feat[sel] for feat in per_layer_raw]
        id_list       = [id_list[i] for i in sel]
        print(f"[FPS] {N_full} → {len(sel)} images selected for memory bank")
    else:
        print(f"[FPS] Skipped (target={fps_target}, N={N_full})")

    gc.collect()

    # ── Pass 2: Patch tokens for FPS-selected images (fp16) ───────────────
    per_image_last_patches = None

    if config.use_last_layer_patches_for_knn:
        M = len(selected_indices)

        # Probe patch shape with one sample
        probe_ds     = torch.utils.data.Subset(loader.dataset, [selected_indices[0]])
        probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=1)
        with torch.no_grad():
            probe_imgs = next(iter(probe_loader))[0].float().to(device)
            _, _probe  = teacher(probe_imgs)
            P_last = _probe[-1].shape[1]   # spatial patches per image
            D_last = _probe[-1].shape[2]   # feature dimension
            del _probe, probe_imgs

        print(f"[Pass 2] patch shape per image: ({P_last}, {D_last})")

        # Pre-allocate fp16 bank: avoids repeated torch.cat peaks
        patch_bank = torch.empty(
            (M, P_last, D_last), dtype=torch.float16, device="cpu"
        )
        write_ptr = 0

        subset_ds     = torch.utils.data.Subset(loader.dataset, selected_indices)
        subset_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size  = loader.batch_size or 16,
            shuffle     = False,
            num_workers = getattr(loader, "num_workers", 0),
            pin_memory  = False,
        )

        with torch.no_grad():
            for batch in tqdm(subset_loader, desc="[Pass 2] Patch tokens (FPS subset)"):
                imgs    = batch[0].float().to(device)
                B_cur   = imgs.shape[0]
                _, patch_list = teacher(imgs)
                patches_cpu = patch_list[-1].detach().half().cpu()   # (B, P, D) fp16
                patch_bank[write_ptr : write_ptr + B_cur] = patches_cpu
                write_ptr += B_cur
                del patches_cpu, patch_list

        per_image_last_patches = patch_bank
        print(
            f"[Pass 2] Patch bank: {per_image_last_patches.shape}  "
            f"≈ {per_image_last_patches.element_size() * per_image_last_patches.numel() / 1e9:.2f} GB"
        )

    print(
        f"[Memory bank] images={concat_cpu.shape[0]}, "
        f"concat_dim={concat_cpu.shape[1]}, "
        f"patches={'yes' if per_image_last_patches is not None else 'no'}"
    )
    return concat_cpu, per_layer_raw, per_image_last_patches, id_list, pca_meta


# ═══════════════════════════════════════════════════════════════════════════
#  §6  KNN INDEX
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
    Returns (scores, indices):
        cosine → scores = inner-product similarities (higher = more similar)
        l2     → scores = Euclidean distances        (lower  = more similar)
    """
    q = (
        query.detach().cpu().numpy()
        if isinstance(query, torch.Tensor)
        else np.asarray(query)
    ).astype("float32").ravel()

    if "index" in index_obj:                                # FAISS path
        idx = index_obj["index"]
        if index_obj["normalized"]:
            q_n = q / (np.linalg.norm(q) + 1e-8)
            dists, inds = idx.search(q_n.reshape(1, -1), k)
            return dists.ravel(), inds.ravel()
        else:
            dists, inds = idx.search(q.reshape(1, -1), k)
            return np.sqrt(np.maximum(dists, 0)).ravel(), inds.ravel()
    else:                                                   # NumPy fallback
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
#  §7  SCORING HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def score_cls_vs_memory(
    cls_vec: torch.Tensor,
    memory_cpu: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """
    Scalar anomaly score for a single CLS token vs the full memory bank.
    Returns a value in [0, 2] for cosine, [0, ∞) for l2.
    Higher = more anomalous.
    """
    c = cls_vec.detach().cpu().float().unsqueeze(0)    # (1, D)
    m = memory_cpu.float()                              # (N, D)
    if metric == "cosine":
        return float(
            1.0 - (F.normalize(c, dim=1) @ F.normalize(m, dim=1).t()).max().item()
        )
    elif metric == "l2":
        return float(torch.cdist(c, m).min().item())
    else:
        raise ValueError(f"Unknown scoring metric: {metric}")


def aggregate_layer_scores(scores: List[float], reduce: str = "mean") -> float:
    vals = np.array(scores, dtype=np.float32)
    return float(vals.max() if reduce == "max" else vals.mean())


# ═══════════════════════════════════════════════════════════════════════════
#  §8  PER-IMAGE PATCH HEATMAP
#      Upsamples to eval_mask_size × eval_mask_size (not 512 × 512).
# ═══════════════════════════════════════════════════════════════════════════
# robust_patch_score (Eq. 7) imported from scoring.py

def compute_image_heatmap(
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
    Compute the anomaly heatmap for a single test image.

    Steps
    -----
    1. Build a CLS query by concatenating CLS tokens from all selected layers.
       Apply PCA projection if pca_meta is provided.
    2. Retrieve K nearest training images from the memory bank (KNN search).
    3. Compute per-patch cosine discrepancy against the retrieved patch tokens.
    4. Bilinear-upsample the patch grid to (eval_mask_size × eval_mask_size)
       for direct comparison with the GT mask.

    Returns
    -------
    heatmap      : (eval_mask_size, eval_mask_size) float32  (higher = anomalous)
    concat_score : float  – image-level score from KNN
    patch_score  : float  – mean patch discrepancy
    """
    # ── 1. Build concat CLS query ─────────────────────────────────────────
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

    # ── 2. Image-level score ──────────────────────────────────────────────
    concat_score = (
        float(1.0 - np.mean(dists))  #float(1.0 - np.max(dists))
        if config.knn_metric == "cosine"
        else float(np.max(dists))
    )

    # ── 3. Patch discrepancy ──────────────────────────────────────────────
    assert per_image_last_patches_cpu is not None, (
        "per_image_last_patches_cpu is required. "
        "Set config.use_last_layer_patches_for_knn = True."
    )
    retrieved    = per_image_last_patches_cpu[k_inds].to(device).float()   # (K, P, D)
    test_patches = patch_tokens_list[-1][[0]].to(device)                   # (1, P, D)
    P_last       = test_patches.shape[1]

    if config.knn_metric == "cosine":
        t_n  = F.normalize(test_patches, dim=2).squeeze(0)                  # (P, D)
        r_n  = F.normalize(retrieved, dim=2).reshape(-1, retrieved.shape[2])# (K*P, D)
        sims = t_n @ r_n.t()                                                # (P, K*P)
        disc = (1.0 - sims.max(dim=1).values).clamp(0.0, 2.0)             # (P,)
    else:
        test_exp = test_patches.expand(retrieved.shape[0], -1, -1)
        disc = ((retrieved - test_exp) ** 2).sum(2).min(0).values.sqrt()

    #patch_score = float(disc.mean().cpu().item())
    patch_score = robust_patch_score(disc)

    # ── 4. Upsample to eval_mask_size × eval_mask_size ───────────────────
    grid = int(math.isqrt(P_last))
    assert grid * grid == P_last, (
        f"Patch count P={P_last} is not a perfect square. "
        f"Ensure image_size ({config.image_size}) is divisible by patch_size (14)."
    )

    heatmap = (
        F.interpolate(
            disc.view(1, 1, grid, grid).float(),
            size=(config.eval_mask_size, config.eval_mask_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )   # (eval_mask_size, eval_mask_size) float32
    from scipy.ndimage import gaussian_filter
    heatmap = gaussian_filter(heatmap, sigma=4).astype(np.float32)

    return heatmap, concat_score, patch_score


# ═══════════════════════════════════════════════════════════════════════════
#  §9  AUROC / AUPRO METRICS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_auroc(
    scores: List[float],
    labels: np.ndarray,
    name: str,
) -> Optional[float]:
    """Compute AUROC via Wilcoxon–Mann–Whitney rank-sum (O(N log N), O(N) memory).
    Returns None and warns if labels are trivial."""
    s = np.asarray(scores, dtype=np.float32)
    l = np.asarray(labels, dtype=np.int32)
    n_pos = int(l.sum())
    n     = len(l)
    if n_pos == 0 or n_pos == n:
        print(f"  [WARN] {name}: trivial labels (pos={n_pos}/{n}), AUC undefined.")
        return None
    rng = float(s.max()) - float(s.min())
    if rng > 0:
        s = (s - s.min()) / rng

    # Rank-sum formulation — avoids sklearn's dense label_binarize at pixel scale
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    n_neg        = n - n_pos
    rank_sum_pos = float(ranks[l == 1].sum())
    auc          = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


def compute_aupro(
    gt_masks: List[np.ndarray],
    pred_maps: List[np.ndarray],
    fpr_limit: float = 0.3,
    num_steps: int = 300,
    min_region_area: int = 10,
) -> float:
    """
    Per-Region Overlap (PRO) curve integrated up to `fpr_limit`.

    Each connected anomaly region is treated as an equal-weight unit,
    removing the bias toward large defects present in plain pixel-AUROC.
    FPR thresholds are derived exclusively from normal pixels.

    Parameters
    ----------
    min_region_area : connected components smaller than this (in pixels)
                      are discarded (annotation noise filter).
    """
    normal_pred_parts:   List[np.ndarray] = []
    regions_pred_sorted: List[np.ndarray] = []

    for gt, pred in zip(gt_masks, pred_maps):
        assert gt.shape == pred.shape
        normal_pred_parts.append(pred[gt < 0.5].ravel())
        if gt.max() < 0.5:
            continue
        labeled_gt, num_r = scipy_label(gt > 0.5)
        for r_id in range(1, num_r + 1):
            region_mask = (labeled_gt == r_id)
            if int(region_mask.sum()) < min_region_area:
                continue
            regions_pred_sorted.append(
                np.sort(pred[region_mask].ravel().astype(np.float32))
            )

    if not regions_pred_sorted:
        warnings.warn("[AUPRO] No anomaly regions — returning NaN", RuntimeWarning)
        return float("nan")

    all_normal_flat = np.concatenate(normal_pred_parts).astype(np.float32)
    fpr_arr    = np.linspace(0.0, fpr_limit, num=num_steps, dtype=np.float64)
    pcts       = np.clip((1.0 - fpr_arr) * 100.0, 0.0, 100.0)
    thresh_arr = np.percentile(all_normal_flat, pcts).astype(np.float32)
    del all_normal_flat
    # Enforce monotone-decreasing thresholds (numerical safety)
    thresh_arr = np.flip(np.minimum.accumulate(np.flip(thresh_arr)))

    R       = len(regions_pred_sorted)
    pro_sum = np.zeros(num_steps, dtype=np.float64)
    for sorted_r in regions_pred_sorted:
        S_r    = len(sorted_r)
        counts = S_r - np.searchsorted(sorted_r, thresh_arr, side="left")
        pro_sum += counts.astype(np.float64) / S_r

    pro_curve = (pro_sum / R).astype(np.float32)
    return float(np.trapz(pro_curve, fpr_arr) / fpr_limit)


# ═══════════════════════════════════════════════════════════════════════════
#  §10  VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_image_heatmap_overlay(
    img_tensor: torch.Tensor,
    heatmap: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    mean=None,
    std=None,
    alpha: float = 0.6,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Visualise input image, heatmap overlay, and GT mask side-by-side."""
    img = img_tensor.detach().cpu()
    if mean is not None and std is not None:
        img = img * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img);  plt.title("Input");  plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(img);  plt.imshow(heatmap, cmap="jet", alpha=alpha)
    plt.title("Anomaly Map");  plt.axis("off")
    plt.subplot(1, 3, 3)
    if gt_mask is not None:
        plt.imshow(gt_mask, cmap="gray");  plt.title("GT Mask")
    else:
        plt.text(0.5, 0.5, "No GT", ha="center");  plt.title("GT Mask")
    plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight");  plt.close()
    else:
        plt.show()


def save_topk_overlay_plots(
    records: List[dict],
    mean,
    std,
    out_dir: str,
    k: int = 5,
):
    """Save top-k highest-scoring normal and anomalous images to disk."""
    os.makedirs(out_dir, exist_ok=True)

    def _save_subset(subset: List[dict], subset_name: str):
        subset = sorted(subset, key=lambda r: r["cls_score"], reverse=True)[:k]
        for rank, rec in enumerate(subset, start=1):
            img_id = os.path.splitext(os.path.basename(rec["img_path"]))[0]
            fname  = (
                f"{subset_name}_rank{rank:02d}_{img_id}"
                f"_cls{rec['cls_score']:.4f}.png"
            )
            plot_image_heatmap_overlay(
                rec["img"], rec["heatmap"],
                gt_mask=rec["mask"],
                mean=mean, std=std,
                title=(
                    f"{subset_name.upper()} | rank={rank} | "
                    f"cls={rec['cls_score']:.4f} | {img_id}"
                ),
                save_path=os.path.join(out_dir, fname),
            )

    _save_subset([r for r in records if r["label"] == 0], "normal")
    _save_subset([r for r in records if r["label"] == 1], "anomalous")


# ═══════════════════════════════════════════════════════════════════════════
#  §11  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_mvtec(
    teacher: TeacherMultiLayer,
    splits: dict,
    config: Config,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    category_dir: str,
    memory_bank: Optional[tuple] = None,
    dataset: str = "mvtec",
) -> Tuple[dict, tuple]:
    """
    End-to-end evaluation pipeline for one category of MVTec or VisA.

    The function is dataset-agnostic: `dataset` controls only which GT mask
    loader is used inside AnomalyDataset.  All memory-bank, KNN, scoring,
    heatmap, and metric components are shared between MVTec and VisA.

    Key differences between MVTec and VisA handled here
    ----------------------------------------------------
    MVTec: GT mask at  {category_dir}/ground_truth/{defect}/{stem}_mask.png
           Normal training images in   train/good/
    VisA:  GT mask at  {category_dir}/Data/Masks/Anomaly/{defect}/{stem}.png
           Normal training images in   Data/Images/Normal/ (split via CSV)

    Parameters
    ----------
    dataset      : "mvtec" | "visa"  — forwarded to AnomalyDataset
    memory_bank  : optional pre-built 5-tuple (skips bank construction)

    Returns
    -------
    results      : dict of metric names → float
    memory_bank  : 5-tuple (per_image_concat, per_layer_raw,
                   per_image_last_patches, img_id_list, pca_meta)
    """
    img_tf = T.Compose([
        _make_resize(config.image_size),
        T.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # ── 1. Build / restore memory bank ────────────────────────────────────
    if memory_bank is None:
        print("\n[1/2] Building memory bank from normal training images …")
        train_ds = AnomalyDataset(
            splits["train"], category_dir,
            image_size=config.image_size,
            eval_mask_size=config.eval_mask_size,
            transform=img_tf,
            return_masks=False,
            dataset=dataset,
        )
        train_loader = data.DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        # Resolve fractional FPS target to an absolute image count
        fps_frac = config.concat_fps_target
        config.concat_fps_target = (
            int(fps_frac * len(train_ds))
            if isinstance(fps_frac, float) and fps_frac <= 1.0
            else int(fps_frac)
        )
        memory_bank = collect_memory_bank(teacher, train_loader, device, config)
        config.concat_fps_target = fps_frac   # restore fractional value

    (per_image_concat, per_layer_raw,
     per_image_last_patches, img_ids, pca_meta) = memory_bank

    concat_index = build_knn_index(
        per_image_concat,
        metric=config.knn_metric,
        use_faiss=config.use_faiss_index,
    )

    # ── 2. Test inference ─────────────────────────────────────────────────
    print("\n[2/2] Evaluating on test split …")
    test_ds = AnomalyDataset(
        splits["test"], category_dir,
        image_size=config.image_size,
        eval_mask_size=config.eval_mask_size,
        transform=img_tf,
        return_masks=True,
        dataset=dataset,
    )
    test_loader = data.DataLoader(
        test_ds,
        batch_size=1,               # heatmap computation assumes B = 1
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ── Pre-allocated pixel buffers ───────────────────────────────────────
    n_test = len(splits["test"])
    H = W  = config.eval_mask_size
    max_px = n_test * H * W         # upper bound

    # All pixels are valid in MVTec — no masking needed, buffer all pixels
    px_scores_buf = np.empty(max_px, dtype=np.float32)
    px_labels_buf = np.empty(max_px, dtype=np.uint8)
    px_ptr        = 0

    gt_buf   = np.empty((n_test, H, W), dtype=np.float32)
    pred_buf = np.empty((n_test, H, W), dtype=np.float32)
    tile_ptr = 0

    img_labels:    List[int]   = []
    cls_scores:    List[float] = []
    concat_scores: List[float] = []
    patch_scores:  List[float] = []

    TOP_K          = 5
    normal_heap    = []
    anomalous_heap = []

    teacher.eval()
    with torch.no_grad():
        for imgs, gt_masks, labels, img_paths in tqdm(test_loader, desc="Test"):
            imgs = imgs.float().to(device)
            cls_list, patch_list = teacher(imgs)

            # ── Image-level CLS score ─────────────────────────────────────
            layer_sc = [
                score_cls_vs_memory(
                    cls_list[li][0].cpu(), per_layer_raw[li], config.scoring
                )
                for li in range(len(cls_list))
            ]
            cls_score = aggregate_layer_scores(layer_sc, reduce="mean")
            cls_scores.append(cls_score)

            # ── Heatmap ───────────────────────────────────────────────────
            heatmap, c_score, p_score = compute_image_heatmap(
                cls_list, patch_list,
                per_image_concat, concat_index,
                per_image_last_patches,
                config, device, pca_meta=pca_meta,
            )
            concat_scores.append(c_score)
            patch_scores.append(p_score)

            # ── Ground-truth ──────────────────────────────────────────────
            # Image-level label comes directly from the record (folder name),
            # not from pixel aggregation — unlike the AgriVision version.
            label    = int(labels[0].item())
            gt_np    = gt_masks[0].numpy().astype(np.uint8)    # (H, W)
            img_path = img_paths[0]
            img_labels.append(label)

            # ── Visualization hook ────────────────────────────────────────
            if config.display_images:
                should_plot = True
                if config.display_anomalous_only:
                    should_plot = (label == 1)
                if config.display_every_n > 1:
                    should_plot = should_plot and (
                        len(img_labels) % config.display_every_n == 0
                    )
                if should_plot:
                    plot_image_heatmap_overlay(
                        imgs[0], heatmap,
                        gt_mask=gt_np,
                        mean=mean.tolist(), std=std.tolist(),
                        title=f"{os.path.basename(img_path)} | label={label}",
                    )

            # ── Bounded viz heap ──────────────────────────────────────────
            record = {
                "img":       imgs[0].detach().cpu(),
                "heatmap":   heatmap.copy(),
                "mask":      gt_np.copy(),
                "label":     label,
                "cls_score": float(cls_score),
                "img_path":  img_path,
            }
            score = -float(cls_score) if label == 0 else float(cls_score)
            heap  = normal_heap if label == 0 else anomalous_heap
            heapq.heappush(heap, (score, len(img_labels), record))
            if len(heap) > TOP_K:
                heapq.heappop(heap)

            # ── Pixel-level accumulation (ALL pixels — no valid_mask) ─────
            # MVTec does not use a valid-pixel mask.  Every pixel in the
            # resized image participates in pixel AUROC and AUPRO.
            if config.pixel_eval:
                n_px = H * W
                px_scores_buf[px_ptr : px_ptr + n_px] = heatmap.ravel()
                px_labels_buf[px_ptr : px_ptr + n_px] = gt_np.ravel()
                px_ptr += n_px

                gt_buf[tile_ptr]   = gt_np.astype(np.float32)
                pred_buf[tile_ptr] = heatmap.astype(np.float32)
                tile_ptr += 1

            del imgs, cls_list, patch_list, heatmap

    # ── 3. Metrics ────────────────────────────────────────────────────────
    print("\n═══ Evaluation Results ═══")
    il = np.array(img_labels)
    results: dict = {
        "image_auroc_cls":    _safe_auroc(cls_scores,    il, "Image AUROC (CLS)"),
        "image_auroc_concat": _safe_auroc(concat_scores, il, "Image AUROC (concat)"),
        "image_auroc_patch":  _safe_auroc(patch_scores,  il, "Image AUROC (patch)"),
    }
    for k, v in results.items():
        print(f"  {k:<30}: {v:.4f}" if v is not None else f"  {k:<30}: N/A")

    if config.pixel_eval and px_ptr > 0:
        ps = px_scores_buf[:px_ptr]
        pl = px_labels_buf[:px_ptr].astype(np.int32)
        results["pixel_auroc"] = _safe_auroc(ps, pl, "Pixel AUROC")
        px_auc = results["pixel_auroc"]
        if px_auc is not None:
            print(f"  {'pixel_auroc':<30}: {px_auc:.4f}")

        n_anom_px = int(pl.sum())
        print(f"\n  Pixels evaluated : {px_ptr:>10,}")
        print(
            f"  Anomalous pixels : {n_anom_px:>10,}  "
            f"({100.0 * n_anom_px / max(px_ptr, 1):.2f}%)"
        )
        del ps, pl, px_scores_buf, px_labels_buf
        gc.collect()

        gt_view   = gt_buf[:tile_ptr]
        pred_view = pred_buf[:tile_ptr]
        results["aupro"] = compute_aupro(gt_view, pred_view)
        print(f"  {'aupro (FPR ≤ 0.3)':<30}: {results['aupro']:.4f}")

    # ── Top-k visualisations ──────────────────────────────────────────────
    viz_records = (
        [r for _, _, r in normal_heap] +
        [r for _, _, r in anomalous_heap]
    )
    save_dir = os.path.join(config.save_path, "topk_plots", config.category)
    save_topk_overlay_plots(
        viz_records, mean.tolist(), std.tolist(), save_dir, k=TOP_K
    )
    print(f"Saved top-{TOP_K} overlay plots → {save_dir}")

    return results, memory_bank


# ═══════════════════════════════════════════════════════════════════════════
#  §12  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_category(cfg: Config, category: str, device: torch.device) -> dict:
    """
    Full pipeline for a single MVTec category.

    Encapsulates split loading, teacher construction, normalisation,
    memory-bank building, evaluation, and checkpointing.

    Returns
    -------
    results dict for this category.
    """
    cfg.category = category
    category_dir = os.path.join(cfg.data_root, category)

    print(f"\n{'═' * 60}")
    print(f"  Category : {category.upper()}  [MVTec]")
    print(f"{'═' * 60}")

    # ── Resolve / build splits ─────────────────────────────────────────────
    splits_path = cfg.splits_json or os.path.join(
        cfg.data_root, f"{category}_mvtec_splits.json"
    )
    if os.path.exists(splits_path):
        with open(splits_path) as fh:
            splits = json.load(fh)
        print(f"Loaded splits from  {splits_path}")
    else:
        print("Generating new splits …")
        splits, splits_path = make_mvtec_splits(
            cfg.data_root, category, seed=cfg.seed
        )
    print("Metadata:", splits["metadata"])

    # ── Build teacher (fully frozen) ───────────────────────────────────────
    teacher = TeacherMultiLayer(
        model_name=cfg.model_name,
        pretrained=True,
        selected_layers=cfg.selected_layers,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)

    os.makedirs(os.path.join(cfg.save_path, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(
        cfg.save_path, "checkpoints", f"{category}_memory_bank.pth"
    )

    if cfg.train:
        # ── Compute per-channel normalisation statistics ─────────────────
        print("\nComputing normalisation statistics over training images …")
        #mean, std = compute_mvtec_mean_std(
        #    splits["train"], category_dir, cfg.image_size, device,
        #    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        #)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        print(f"  mean = {[round(v, 4) for v in mean.tolist()]}")
        print(f"  std  = {[round(v, 4) for v in std.tolist()]}")

        # ── Build memory bank + evaluate ─────────────────────────────────
        results, memory_bank = evaluate_mvtec(
            teacher, splits, cfg, mean, std, device,
            category_dir=category_dir, memory_bank=None,
            dataset="mvtec",
        )

        # ── Save checkpoint ───────────────────────────────────────────────
        (per_image_concat, per_layer_raw,
         per_image_last_patches, img_ids, pca_meta) = memory_bank
        torch.save(
            {
                "mean":                   mean,
                "std":                    std,
                "per_image_concat":       per_image_concat,
                "per_layer_raw":          per_layer_raw,
                "per_image_last_patches": per_image_last_patches,
                "img_ids":                img_ids,
                "pca_meta":               pca_meta,
                "category":               category,
                "dataset":                "mvtec",
                "results":                results,
            },
            ckpt_path,
        )
        print(f"\nCheckpoint saved → {ckpt_path}")

    else:
        # ── Load checkpoint + evaluate ────────────────────────────────────
        print(f"Loading checkpoint → {ckpt_path}")
        ck   = torch.load(ckpt_path, map_location="cpu")
        mean = ck["mean"]
        std  = ck["std"]
        mb   = (
            ck["per_image_concat"],
            ck["per_layer_raw"],
            ck.get("per_image_last_patches"),
            ck.get("img_ids"),
            ck.get("pca_meta"),
        )
        results, _ = evaluate_mvtec(
            teacher, splits, cfg, mean, std, device,
            category_dir=category_dir, memory_bank=mb,
            dataset="mvtec",
        )

    print(f"\n  [{category.upper()}] Final Results:")
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<30}: {v:.4f}")

    # Release teacher GPU memory before the next category
    teacher.cpu()
    del teacher
    torch.cuda.empty_cache()
    gc.collect()

    return results


def run_visa_category(cfg: Config, category: str, device: torch.device) -> dict:
    """
    Full pipeline for a single VisA category.

    Mirrors run_category() but uses:
      • visa_data_root instead of data_root
      • make_visa_splits() instead of make_mvtec_splits()
      • dataset="visa" forwarded to AnomalyDataset (→ load_visa_gt_mask)
      • Per-category checkpoint saved with a "visa_" prefix

    All backbone, memory-bank, KNN, scoring, and metric components are
    shared with the MVTec pipeline — they are dataset-agnostic.

    Returns
    -------
    results dict for this category.
    """
    cfg.category = category
    category_dir = os.path.join(cfg.visa_data_root, category)

    print(f"\n{'═' * 60}")
    print(f"  Category : {category.upper()}  [VisA]")
    print(f"{'═' * 60}")

    if not os.path.isdir(category_dir):
        raise FileNotFoundError(
            f"VisA category directory not found: {category_dir}\n"
            f"Check cfg.visa_data_root = {cfg.visa_data_root!r}"
        )

    # ── Resolve / build splits ─────────────────────────────────────────────
    splits_path = cfg.splits_json or os.path.join(
        cfg.visa_data_root, f"{category}_visa_splits.json"
    )
    if os.path.exists(splits_path):
        with open(splits_path) as fh:
            splits = json.load(fh)
        # Validate that the cached JSON is a VisA split (not an accidental MVTec one)
        if splits.get("metadata", {}).get("dataset") != "visa":
            print("[WARN] Cached splits JSON has wrong dataset tag. Regenerating …")
            splits, splits_path = make_visa_splits(
                cfg.visa_data_root, category,
                seed=cfg.seed,
                test_fraction=cfg.visa_test_fraction,
                csv_path=cfg.visa_split_csv,
            )
        else:
            print(f"Loaded VisA splits from  {splits_path}")
    else:
        print("Generating new VisA splits …")
        splits, splits_path = make_visa_splits(
            cfg.visa_data_root, category,
            seed=cfg.seed,
            test_fraction=cfg.visa_test_fraction,
            csv_path=cfg.visa_split_csv,
        )
    print("Metadata:", splits["metadata"])

    # ── Build teacher (fully frozen) ───────────────────────────────────────
    teacher = TeacherMultiLayer(
        model_name=cfg.model_name,
        pretrained=True,
        selected_layers=cfg.selected_layers,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(device)

    visa_save = os.path.join(cfg.save_path, "visa")
    os.makedirs(os.path.join(visa_save, "checkpoints"), exist_ok=True)
    ckpt_path = os.path.join(
        visa_save, "checkpoints", f"visa_{category}_memory_bank.pth"
    )

    # ImageNet statistics — correct choice for a frozen DINOv2 backbone.
    #mean, std = compute_mvtec_mean_std(
    #    splits["train"], category_dir, cfg.image_size, device,
    #    batch_size=cfg.batch_size, num_workers=cfg.num_workers,
    #)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std  = torch.tensor([0.229, 0.224, 0.225])
    print(f"  Normalisation: ImageNet  mean={mean.tolist()}  std={std.tolist()}")

    if cfg.train:
        results, memory_bank = evaluate_mvtec(
            teacher, splits, cfg, mean, std, device,
            category_dir=category_dir, memory_bank=None,
            dataset="visa",
        )

        (per_image_concat, per_layer_raw,
         per_image_last_patches, img_ids, pca_meta) = memory_bank
        torch.save(
            {
                "mean":                   mean,
                "std":                    std,
                "per_image_concat":       per_image_concat,
                "per_layer_raw":          per_layer_raw,
                "per_image_last_patches": per_image_last_patches,
                "img_ids":                img_ids,
                "pca_meta":               pca_meta,
                "category":               category,
                "dataset":                "visa",
                "results":                results,
            },
            ckpt_path,
        )
        print(f"\nVisA checkpoint saved → {ckpt_path}")

    else:
        print(f"Loading VisA checkpoint → {ckpt_path}")
        ck   = torch.load(ckpt_path, map_location="cpu")
        mean = ck["mean"]
        std  = ck["std"]
        mb   = (
            ck["per_image_concat"],
            ck["per_layer_raw"],
            ck.get("per_image_last_patches"),
            ck.get("img_ids"),
            ck.get("pca_meta"),
        )
        results, _ = evaluate_mvtec(
            teacher, splits, cfg, mean, std, device,
            category_dir=category_dir, memory_bank=mb,
            dataset="visa",
        )

    print(f"\n  [VisA / {category.upper()}] Final Results:")
    for k, v in results.items():
        if v is not None:
            print(f"  {k:<30}: {v:.4f}")

    teacher.cpu()
    del teacher
    torch.cuda.empty_cache()
    gc.collect()

    return results


def _print_summary_table(all_results: Dict[str, dict], title: str = "Summary"):
    """Print a formatted per-category results table with column means."""
    if not all_results:
        return
    metric_keys = list(next(iter(all_results.values())).keys())
    col_w, cat_w = 22, 20
    print(f"\n\n═══ {title} ═══")
    header = f"{'Category':<{cat_w}}" + "".join(f"{m:<{col_w}}" for m in metric_keys)
    print(header)
    print("─" * len(header))
    for cat, res in all_results.items():
        row = f"{cat:<{cat_w}}"
        for m in metric_keys:
            v = res.get(m)
            row += f"{v:.4f}{' ' * (col_w - 6)}" if v is not None else f"{'N/A':<{col_w}}"
        print(row)
    print("─" * len(header))
    means_row = f"{'MEAN':<{cat_w}}"
    for m in metric_keys:
        vals = [all_results[c][m] for c in all_results if all_results[c].get(m) is not None]
        means_row += (
            f"{np.mean(vals):.4f}{' ' * (col_w - 6)}" if vals else f"{'N/A':<{col_w}}"
        )
    print(means_row)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    print_config(cfg)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    # ── Dataset dispatch ──────────────────────────────────────────────────
    if cfg.dataset == "visa":
        categories = VISA_CATEGORIES if cfg.run_all_categories else [cfg.category]
        print(f"[Dataset] VisA — {len(categories)} categories")
        all_results: Dict[str, dict] = {}
        for cat in categories:
            all_results[cat] = run_visa_category(cfg, cat, device)
        if len(categories) > 1:
            _print_summary_table(all_results, title="VisA Summary Across All Categories")

    elif cfg.dataset == "mvtec":
        categories = MVTEC_CATEGORIES if cfg.run_all_categories else [cfg.category]
        print(f"[Dataset] MVTec — {len(categories)} categories")
        all_results = {}
        for cat in categories:
            all_results[cat] = run_category(cfg, cat, device)
        if len(categories) > 1:
            _print_summary_table(all_results, title="MVTec Summary Across All Categories")

    else:
        raise ValueError(
            f"Unknown dataset: {cfg.dataset!r}. "
            f"Set cfg.dataset to 'mvtec' or 'visa'."
        )


if __name__ == "__main__":
    main()