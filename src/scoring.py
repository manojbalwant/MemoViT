"""
scoring.py — canonical MemoViT scoring functions.

This module consolidates the scoring logic that is duplicated across the
development scripts so that the *reported* metric has a single, unambiguous
implementation.

Reported metric (paper, Eq. 7):
    The frame/image-level anomaly score for "MemoViT (Ours)" is the robust,
    median-absolute-deviation (MAD) aggregation of the per-patch cosine
    discrepancy field. The multi-layer CLS descriptor is used ONLY as the
    k-NN retrieval key (Eq. 5); it does not produce the reported score.

Diagnostic scores (printed by the scripts but NOT the reported metric):
    - cls_proto  : per-layer CLS-vs-prototype aggregation
    - concat_knn : concatenated multi-layer CLS k-NN distance

Keep these names distinct from the reported scorer to avoid the inverted
naming that previously labelled the CLS score as `frame_auroc`. See
REPRODUCIBILITY.md, known issues #2 and #7.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-patch cosine discrepancy (Eq. 6)
# ---------------------------------------------------------------------------
def patch_discrepancy_field(
    test_patches: torch.Tensor,        # (P, C) last-block patch tokens of the query
    retrieved_patches: torch.Tensor,   # (K, P, C) patch tokens of the K retrieved neighbours
    metric: str = "cosine",
) -> torch.Tensor:
    """Return the per-patch discrepancy delta(p) in [0, 1], shape (P,).

    delta(p) = 1 - max over all retrieved-neighbour patches of cosine(q_p, r_{j,p'}).
    """
    if metric == "cosine":
        tn = F.normalize(test_patches, dim=1)              # (P, C)
        rn = F.normalize(retrieved_patches, dim=2)         # (K, P, C)
        sim = torch.matmul(tn, rn.reshape(-1, rn.shape[2]).t())  # (P, K*P)
        return 1.0 - sim.max(dim=1)[0]                     # (P,)
    else:  # euclidean fallback
        te = test_patches.unsqueeze(0).expand(retrieved_patches.shape[0], -1, -1)
        d2 = ((retrieved_patches - te) ** 2).sum(dim=2)
        return d2.min(dim=0)[0].sqrt()


# ---------------------------------------------------------------------------
# Reported frame/image score (Eq. 7) — robust MAD aggregation
# ---------------------------------------------------------------------------
def robust_patch_score(
    patch_discrepancy,                 # torch.Tensor or np.ndarray, shape (P,)
    n_sigma: float = 2.0,
    fallback_ratio: float = 0.05,
) -> float:
    """THE reported MemoViT frame/image score (Eq. 7).

    Flags patches whose discrepancy exceeds median + n_sigma * 1.4826 * MAD and
    returns the mean of the flagged set; if none are flagged, falls back to the
    mean of the top `fallback_ratio` fraction of patches.
    """
    if isinstance(patch_discrepancy, torch.Tensor):
        x = patch_discrepancy.detach().float().flatten().cpu().numpy()
    else:
        x = np.asarray(patch_discrepancy, dtype=np.float64).flatten()

    median = np.median(x)
    mad = np.median(np.abs(x - median))
    sigma = 1.4826 * mad + 1e-12
    threshold = median + n_sigma * sigma

    outliers = x[x > threshold]
    if outliers.size > 0:
        return float(outliers.mean())

    k = max(1, int(fallback_ratio * len(x)))               # top-5% fallback
    return float(np.partition(x, -k)[-k:].mean())


def frame_score(test_patches, retrieved_patches, metric: str = "cosine") -> float:
    """Convenience wrapper: discrepancy field (Eq. 6) -> reported score (Eq. 7)."""
    delta = patch_discrepancy_field(test_patches, retrieved_patches, metric=metric)
    return robust_patch_score(delta)


# ---------------------------------------------------------------------------
# Anomaly heatmap (Eq. 8) with optional per-benchmark Gaussian smoothing
# ---------------------------------------------------------------------------
def anomaly_heatmap(
    delta: torch.Tensor,               # (P,) per-patch discrepancy
    eval_size: int,                    # S_eval: 518 (aerial), 512 (agri), or GT mask size (MVTec/VisA)
    gaussian_sigma: float | None = None,  # set to 4.0 for MVTec/VisA and Agriculture; None for aerial
) -> np.ndarray:
    """Reshape -> bilinear upsample to eval_size -> optional Gaussian -> min-max [0,1]."""
    g = int(round(delta.shape[0] ** 0.5))                  # 37 for P = 1369
    hm = delta.view(1, 1, g, g).float()
    hm = F.interpolate(hm, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
    hm = hm.squeeze().cpu().numpy()

    if gaussian_sigma is not None:
        from scipy.ndimage import gaussian_filter
        hm = gaussian_filter(hm, sigma=gaussian_sigma).astype(np.float32)

    lo, hi = hm.min(), hm.max()
    return ((hm - lo) / (hi - lo + 1e-12)).astype(np.float32)
