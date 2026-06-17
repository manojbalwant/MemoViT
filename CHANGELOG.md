# Changelog

## v1.0.0 — Reproducible release

This release makes the default code paths reproduce the metrics reported in the
paper and removes the scorer ambiguity identified during the code-to-manuscript
audit. No experimental results change: the reported numbers were always the
robust-MAD patch score (Eq. 7); these edits make the code return and label that
score by default, and consolidate its (previously duplicated, sometimes
inverted) implementation.

### Added
- `scoring.py` — single canonical implementation of the reported scorer:
  `patch_discrepancy_field` (Eq. 6), `robust_patch_score` (Eq. 7),
  `frame_score`, and `anomaly_heatmap` (Eq. 8, with per-benchmark Gaussian σ).
  Smoke-tested: sparse-outlier and fallback branches, discrepancy field,
  frame score, and both heatmap variants execute correctly.
- `REPRODUCIBILITY.md`, `CHANGELOG.md`.

### Fixed
- **`baselines.py` (`ProposedModel.score()`):** returns the reported
  `robust_patch_score` (Eq. 7) instead of `cls_score`. CLS-proto, concat-kNN,
  and patch-mean values are preserved under `self.last_diagnostics`. (Issue #7,
  previously a release blocker.)
- **Scorer naming (issue #2):** the CLS-prototype diagnostic was named
  `frame_auroc` / `frame_auc`, inverting the paper's terminology. Renamed to
  `cls_proto_auroc` (`ablation.py`) and `cls_proto_auc` (`aerial_memovit.py`).
  The reported metric `patch_auroc` / `patch_auc` is now explicitly labelled
  "reported, Eq. 7" in console output and result tables.
- **Duplicate scorer definitions:** `robust_patch_score` was defined twice in
  `ablation.py` and once each in `aerial_memovit.py`, `mvtec_visa_memovit.py`,
  and `agriculture_memovit.py`. All local copies removed; all scripts now
  `from scoring import robust_patch_score`.
- **Ablation table headers (issue #11):** CLS column marked diagnostic, patch
  column marked reported; the CLS-PCA panel columns corrected from
  "Patch PCA / Bank dim" to "CLS-PCA type / CLS-key dim".

### Documented (code unchanged, paper corrected)
- **Normalization (issue #8):** ImageNet for MVTec/VisA; dataset-specific for
  aerial and Agriculture-Vision. See `REPRODUCIBILITY.md`.
- **Gaussian smoothing (issue #9):** σ = 4 for MVTec/VisA and Agriculture-Vision;
  none for aerial. Parameterised in `scoring.anomaly_heatmap`.

### Open (require new runs, not code fixes)
- **Issue #10:** the layer ablation co-varies the retrieval key and the
  patch-discrepancy layer; a clean retrieval-key ablation (patch layer fixed at
  block 11) is recommended.
- Missing `double_plant` AUPRO entry in the Agriculture-Vision table.

### Notes
- `scoring.py` is provided at the repository root (for the documented import
  path) and in `src/` (so the scripts import it directly when run from `src/`).
- Validation in this release was limited to syntax (AST) checks on all scripts
  and a functional smoke test of `scoring.py`; full table reproduction requires
  the datasets and the reference GPU environment.

## v1.1.0 — Presentation & readability

### Added
- `run.py` — a single CLI that dispatches to any benchmark pipeline
  (`python run.py {drone,uitadrone,baselines,mvtec,visa,agri,ablation} [args]`),
  presenting the five scripts through one entry point. Smoke-tested.
- `src/metrics.py` — canonical, unit-tested implementations of the four reported
  metrics: image/frame AUROC (Eq. 9), pixel AUROC (Eq. 10), AUPRO (Eq. 11), and
  EER (Eq. 12). Verified on synthetic data (separable → AUROC 1.0 / EER 0;
  informative heatmap → AUPRO 0.93; uninformative → 0.18). Provided as a shared
  reference; scripts may migrate to it after confirming equivalence.

### Changed
- `baselines.py`: condensed the file header and section banners; removed dead
  code — the unused `load_and_preprocess_images` helper and a superseded
  duplicate FPS function (`greedy_farthest_point_sampling_rows_cpu_`, no callers).
- `ablation.py`: condensed the header with an explicit note that `patch_auroc`
  (Eq. 7) is the reported column and the CLS/concat scores are diagnostics.

### Scope note
- Pipeline internals (feature extractors, dataset loaders, run logic) were
  preserved rather than rewritten line-by-line: the five scripts use subtly
  different DINOv2 teachers, so a blanket internal rewrite risks changing
  published numbers. The shared, verifiable logic (scoring, metrics) is
  consolidated into modules; deeper per-file rewrites should be done one
  benchmark at a time and validated against that benchmark's data.
