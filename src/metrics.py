"""
metrics.py — canonical evaluation metrics shared across all MemoViT benchmarks.

Provides the four metrics reported in the paper, each matching the definition in
Section 6c: image/frame-level AUROC (Eq. 9), pixel-level AUROC (Eq. 10), AUPRO
(per-region overlap, Eq. 11), and equal error rate (Eq. 12).

This module is the canonical reference. Per-benchmark scripts may migrate to it
once equivalence with their in-script computation has been confirmed; until then
it is safe to import the metrics needed without changing existing numbers.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.ndimage import label as cc_label


# ---------------------------------------------------------------------------
# Image / frame-level AUROC (Eq. 9)
# ---------------------------------------------------------------------------
def image_auroc(labels, scores) -> float:
    """Area under the ROC curve over frames/images. Higher is better."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Pixel-level AUROC (Eq. 10)
# ---------------------------------------------------------------------------
def pixel_auroc(gt_masks, heatmaps, max_normal_px: int = 10**6, seed: int = 42) -> float:
    """Pixel-level AUROC of heatmap values against binary ground-truth masks.

    All anomalous pixels are kept; normal pixels are stratified-subsampled to at
    most ``max_normal_px`` for tractability (seeded for determinism).
    """
    gt = np.concatenate([np.asarray(m).ravel() for m in gt_masks]).astype(np.uint8)
    hm = np.concatenate([np.asarray(h).ravel() for h in heatmaps]).astype(np.float64)

    pos = np.where(gt == 1)[0]
    neg = np.where(gt == 0)[0]
    if neg.size > max_normal_px:
        neg = np.random.default_rng(seed).choice(neg, size=max_normal_px, replace=False)

    idx = np.concatenate([pos, neg])
    return float(roc_auc_score(gt[idx], hm[idx]))


# ---------------------------------------------------------------------------
# AUPRO — per-region overlap (Eq. 11)
# ---------------------------------------------------------------------------
def aupro(gt_masks, heatmaps, fpr_limit: float = 0.30, n_steps: int = 300) -> float:
    """Area under the per-region-overlap curve, normalised over FPR <= fpr_limit.

    For each threshold, PRO is the mean over connected ground-truth regions of the
    per-region overlap |P(tau) & R_k| / |R_k| (equal weight per region). The curve
    is integrated against the global false-positive rate and normalised by
    ``fpr_limit``.
    """
    gt_masks = [np.asarray(m).astype(np.uint8) for m in gt_masks]
    heatmaps = [np.asarray(h, dtype=np.float64) for h in heatmaps]

    all_scores = np.concatenate([h.ravel() for h in heatmaps])
    lo, hi = all_scores.min(), all_scores.max()
    thresholds = np.linspace(hi, lo, n_steps)

    # Precompute connected regions per image.
    regions = []                          # list of (image_index, region_pixel_mask)
    total_normal = 0
    for i, m in enumerate(gt_masks):
        total_normal += int((m == 0).sum())
        lbl, n = cc_label(m)
        for r in range(1, n + 1):
            regions.append((i, lbl == r))
    if not regions:
        return float("nan")

    pros, fprs = [], []
    for tau in thresholds:
        preds = [h >= tau for h in heatmaps]
        # Mean per-region overlap.
        overlaps = [float(preds[i][rm].sum()) / float(rm.sum()) for i, rm in regions]
        pros.append(np.mean(overlaps))
        # Global false-positive rate over normal pixels.
        fp = sum(int((preds[i] & (gt_masks[i] == 0)).sum()) for i in range(len(gt_masks)))
        fprs.append(fp / max(total_normal, 1))

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    keep = fprs <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    order = np.argsort(fprs[keep])
    x, y = fprs[keep][order], pros[keep][order]
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz(y, x) / fpr_limit)


# ---------------------------------------------------------------------------
# Equal error rate (Eq. 12) — lower is better
# ---------------------------------------------------------------------------
def eer(labels, scores) -> float:
    """Equal error rate: the operating point where FPR == 1 - TPR (FNR)."""
    labels = np.asarray(labels).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)
