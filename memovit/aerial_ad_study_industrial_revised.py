"""
================================================================================
 aerial_ad_study_industrial_revised.py — MVTec AD / VisA with the Proposed method
 INDUSTRIAL_VERSION 3.3 · requires the core module aerial_ad_study_revised.py (3.2+)
================================================================================

 Role in the article (E5 · generality / transfer)
   Full-shot ONE-CLASS: the bank is built from normal TRAINING images of each
   category (NOT zero-shot). Three reportable settings:
     1. strict transfer   --config <development selected_config.json>  (frozen-v1)
     2. cross-dataset     --select on dataset A, then --config <selected_on_A.json>
                          evaluated on dataset B (never report A's numbers)
     3. baselines         --baselines patchcore anomalydino ... under the SAME
                          protocol (preprocessing, metric code, smoothing)

 Shared with the core module (one implementation for every dataset)
   ProposedMethod / ProposedScorer, DINOv2 backbone, feature cache, FPS bank,
   baselines (build_method), frame metrics, EXACT pixel metrics
   (pixel_metrics_exact), PixelEval frame policy, Ablation, selection grid,
   rules, tolerance and baseline grids (SELECT_PROPOSED / SELECT_RULES /
   SELECT_PATCH_LAYER / SELECT_TOL / SELECT_BASELINES), run files, env.json,
   table writer, overlay figures.

 Pixel-evaluation protocol
   evaluation: native mask resolution (--pixel_eval_div 1, default);
   --select: native / 4 (default; selection uses image AUROC only).
   Anomaly maps bilinearly resized to the mask aspect, then Gaussian-smoothed
   with sigma = 4 px at 256 px, rescaled to the evaluation size, identically
   for every method. --pixel_smooth_sigma 0 for the unsmoothed supplementary run.

 Modes
   evaluate (default)   Table 5 (+ baselines table, overlays, ablations)
   --select             cross-dataset selection with the core grid; writes
                        {save_path}/{dataset}/selected_on_{dataset}.json and exits
   --print_config       print the effective configuration and exit

 Usage
   export AD_CORE_MODULE=aerial_ad_study_revised
   PYI="python ANDT/aerial_ad_study_industrial_revised.py"
   CFG=experiments/development/selected_config.json

   $PYI --dataset visa --data_root dataset/VisAMVtech --config $CFG --select \
        --select_baselines --baselines patchcore anomalydino
   # evaluation of the other dataset with the selected configuration
   $PYI --dataset mvtec --data_root dataset/MVtech --seeds 42 \
        --config experiments/industrial/mvtec/selected_on_mvtec.json --baselines patchcore anomalydino
================================================================================
"""

import os, sys, csv, json, time, random, argparse, importlib, warnings
from collections import defaultdict
from dataclasses import dataclass, fields, replace, asdict
from glob import glob
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tud
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
core = importlib.import_module(os.environ.get('AD_CORE_MODULE', 'aerial_ad_study_revised'))
if not str(getattr(core, 'CODE_VERSION', '')).startswith('3.'):
    raise RuntimeError(f'core module {core.__name__} is {getattr(core, "CODE_VERSION", "?")}; need 3.x')
for _name in ('SELECT_PROPOSED', 'SELECT_RULES', 'SELECT_TOL', 'SELECT_BASELINES', 'SELECT_PATCH_LAYER',
              'Ablation', '_proposed_eval', 'build_method', 'pixel_metrics_exact', 'write_env',
              '_add_bool'):
    if not hasattr(core, _name):
        raise RuntimeError(f'core module {core.__name__} lacks {_name}; update it first')

INDUSTRIAL_VERSION = '3.3-industrial'


# ═══════════════════════════════════════════════════════════════════════════
#  §0  DATASET CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

MVTEC_CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
                    'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood',
                    'zipper']
VISA_CATEGORIES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                   'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


# ═══════════════════════════════════════════════════════════════════════════
#  §1  SELECTION GRID — taken from the core (single source of truth)
# ═══════════════════════════════════════════════════════════════════════════
# Direction 1: select on MVTec AD -> evaluate on VisA. Direction 2: the reverse.
# Grid, rules, tolerance, tie-break and baseline grids are the core's. ONE
# declared deviation: fps_ratio = None (all training images) — image datasets
# have no temporal redundancy for FPS.
# Tie-break: first configuration within SEL_TOL of the best, in the order
#   retrieval layers as listed -> K ascending -> rules as listed;
#   baselines: core.SELECT_BASELINES order.

SEL_LAYERS      = tuple(tuple(l) for l in core.SELECT_PROPOSED['selected_layers'])
SEL_PATCH_LAYER = int(core.SELECT_PATCH_LAYER)
SEL_K           = (3, 5, 10, 20, 50)#tuple(core.SELECT_PROPOSED['knn_k'])
SEL_FPS         = None                                   # declared deviation (full bank)
SEL_RULES       = dict(core.SELECT_RULES)                # scored in one pass
SEL_TOL         = float(core.SELECT_TOL)
SEL_CRITERION   = 'mean over categories of image AUROC (categories weighted equally)'
SEL_BASELINE_DEFAULT = ('patchcore', 'anomalydino')
# Fixed for selection AND final evaluation (part of the protocol, not searched)
SEL_FIXED = {'image_size': 518, 'resize': 'cv2_linear', 'seed': 42}

# Fixed industrial protocol (deliberately NOT command-line options)
PIXEL_SMOOTH_REF   = 256      # sigma is defined at 256 px and rescaled to the evaluation size
VISA_SPLIT_CSV     = None     # None -> {data_root}/split_csv/1cls.csv (official split)
VISA_TEST_FRACTION = 0.2      # only for the non-reportable seeded fallback
A4_REPEATS         = 5        # runs per stochastic strategy in ablation A4
DIV_EVAL, DIV_SELECT = 4, 4   # pixel_eval_div defaults: final evaluation / selection

_BC_FIELDS = {f.name for f in fields(core.BenchConfig)}


# ═══════════════════════════════════════════════════════════════════════════
#  §2  FEATURES, IMAGES, SPLITS, GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IndustrialFeatureSpec(core.FeatureSpec):
    """Core FeatureSpec + the resize policy (part of the cache key)."""
    resize: str = 'cv2_linear'                # cv2_linear (= aerial pipeline) | torch_antialias


def load_image(path: str, fspec) -> torch.Tensor:
    s = fspec.image_size
    if fspec.resize == 'cv2_linear':
        t = TF.to_tensor(core.load_rgb(path, s))
    else:
        t = TF.to_tensor(Image.open(path).convert('RGB'))
        try:
            t = F.interpolate(t[None], size=(s, s), mode='bilinear', align_corners=False,
                              antialias=True)[0]
        except TypeError:                                    # torch < 1.11
            t = F.interpolate(t[None], size=(s, s), mode='bilinear', align_corners=False)[0]
    return TF.normalize(t, list(fspec.mean), list(fspec.std))


class IndustrialDataset(tud.Dataset):
    def __init__(self, files: List[str], fspec):
        self.files, self.fspec = list(files), fspec

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return {'img': load_image(self.files[i], self.fspec), 'idx': i}


@dataclass
class Rec:
    path: str
    label: int            # 0 normal, 1 anomalous
    defect: str
    mask: Optional[str]   # GT mask path (None for normal images)


_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def _imgs(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(p for p in glob(os.path.join(d, '*')) if p.lower().endswith(_EXT))


def mvtec_records(root: str, cat: str):
    """train/good -> train; test/good -> 0; test/<defect> -> 1 with ground_truth/<defect>/<stem>_mask.png."""
    cdir = os.path.join(root, cat)
    train = [Rec(p, 0, 'good', None) for p in _imgs(os.path.join(cdir, 'train', 'good'))]
    test, tdir = [], os.path.join(cdir, 'test')
    for defect in sorted(os.listdir(tdir)):
        d = os.path.join(tdir, defect)
        if not os.path.isdir(d):
            continue
        for p in _imgs(d):
            if defect == 'good':
                test.append(Rec(p, 0, defect, None))
                continue
            stem = os.path.splitext(os.path.basename(p))[0]
            m = os.path.join(cdir, 'ground_truth', defect, f'{stem}_mask.png')
            test.append(Rec(p, 1, defect, m if os.path.exists(m) else None))
    return train, test, 'official MVTec AD split', True


def visa_records(root: str, cat: str, seed: int = 42, test_fraction: float = 0.2,
                 csv_path: Optional[str] = None):
    """Official split_csv/1cls.csv when present; otherwise a seeded split (NOT reportable)."""
    csv_path = csv_path or os.path.join(root, 'split_csv', '1cls.csv')
    train, test = [], []
    if os.path.exists(csv_path):
        with open(csv_path, newline='') as fh:
            for row in csv.DictReader(fh):
                img = row.get('image', '').strip().replace('\\', '/')
                if not img.startswith(cat + '/'):
                    continue
                label = 0 if row.get('label', '').strip().lower() == 'normal' else 1
                mrel = row.get('mask', '').strip()
                mask = os.path.join(root, mrel) if (label and mrel) else None
                rec = Rec(os.path.join(root, img), label, 'normal' if label == 0 else 'anomaly', mask)
                (train if row.get('split', '').strip().lower() == 'train' else test).append(rec)
        return train, test, f'official CSV {csv_path}', True
    warnings.warn(f'VisA official split not found ({csv_path}); seeded fallback is NOT reportable')
    data_dir = os.path.join(root, cat, 'Data')
    normal = [Rec(p, 0, 'normal', None) for p in _imgs(os.path.join(data_dir, 'Images', 'Normal'))]
    adir = os.path.join(data_dir, 'Images', 'Anomaly')
    pairs = [(p, '') for p in _imgs(adir)]
    if not pairs and os.path.isdir(adir):
        pairs = [(p, d) for d in sorted(os.listdir(adir)) if os.path.isdir(os.path.join(adir, d))
                 for p in _imgs(os.path.join(adir, d))]
    anom = []
    for p, d in pairs:
        stem = os.path.splitext(os.path.basename(p))[0]
        m = os.path.join(data_dir, 'Masks', 'Anomaly', *([d] if d else []), f'{stem}.png')
        anom.append(Rec(p, 1, d or 'anomaly', m if os.path.exists(m) else None))
    random.Random(seed).shuffle(normal)
    k = max(1, int(len(normal) * test_fraction))
    return normal[k:], normal[:k] + anom, f'seeded split (seed {seed}) — NOT official', False


class IndustrialGT:
    """core.PixelGT interface. MVTec: pixel > 127. VisA: any non-zero pixel."""

    def __init__(self, mask_paths: List[Optional[str]], nonzero: bool):
        self.paths, self.nonzero = mask_paths, nonzero

    def get(self, sc, f: int):
        p = self.paths[f]
        if p is None or not os.path.exists(p):
            return 'missing', None
        a = np.array(Image.open(p).convert('L'))
        return 'mask', ((a != 0) if self.nonzero else (a > 127))

    def native_shape(self, scenes=None):
        for p in self.paths:
            if p and os.path.exists(p):
                with Image.open(p) as im:
                    return im.size[1], im.size[0]
        return None


class IndustrialPixelEval(core.PixelEval):
    """
    core.PixelEval (same frame policy, exact metrics) with industrial ground truth.
    sigma is given at pixel_smooth_ref px and rescaled to the evaluation size, so the
    physical amount of smoothing does not depend on the evaluation resolution.
    """
    def __init__(self, ctx, eval_size: int, div: int):
        c = ctx.cfg
        self.gt = IndustrialGT(ctx.mask_paths, ctx.nonzero_masks)
        if eval_size:
            self.hw = (int(eval_size), int(eval_size))
        else:
            shp = self.gt.native_shape()
            self.hw = None if shp is None else (max(1, shp[0] // div), max(1, shp[1] // div))
        self.max_fpr = c.pixel_max_fpr
        self.skip = c.pixel_skip_inconsistent
        self.sigma = float(c.pixel_smooth_sigma)
        ref = int(getattr(c, 'pixel_smooth_ref', 0) or 0)
        if ref and self.hw and self.sigma > 0:
            self.sigma = self.sigma * (self.hw[0] * self.hw[1]) ** 0.5 / ref
        self.maps, self.masks = [], []
        self.n = dict.fromkeys(('used', 'no_map', 'unlabelled', 'normal_no_mask',
                                'anom_missing_mask', 'anom_empty_mask', 'anom_lost_resize',
                                'normal_with_region'), 0)
        self.agree = self.checked = 0


class IndustrialContext(core.Context):
    """core.Context for one category; the test set is one scene named 'test'."""

    def __init__(self, bcfg, fspec, teacher, device, patch_layer: int, dataset: str, root: str,
                 category: str, seed: int = 42, visa_test_fraction: float = 0.2,
                 visa_csv: Optional[str] = None):
        self.pix, self.mask_root = None, None
        self.cfg, self.fspec, self.teacher, self.device = bcfg, fspec, teacher, device
        self.patch_layers = (int(patch_layer),)
        self.dataset, self.category = dataset, category
        if dataset == 'mvtec':
            train, test, src, official = mvtec_records(root, category)
        else:
            train, test, src, official = visa_records(root, category, seed, visa_test_fraction, visa_csv)
        if not train or not test:
            raise FileNotFoundError(f'{dataset}/{category}: empty split under {root}')
        self.split_source, self.official_split = src, official
        self.train_files = [r.path for r in train]
        self.scenes = [core.Scene('test', [r.path for r in test],
                                  np.array([r.label for r in test], dtype=np.int64))]
        self.mask_paths = [r.mask for r in test]
        self.defects = [r.defect for r in test]
        self.nonzero_masks = dataset == 'visa'
        self.cache_root = os.path.join(bcfg.cache_dir, f'{dataset}_{core._safe(category)}', fspec.key())
        self._mem = {}
        n_anom = int(self.labels.sum())
        print(f'\n {dataset.upper()} {category} ({src}): train {len(train)}, test normal '
              f'{len(test) - n_anom}, test anomalous {n_anom} '
              f'[{", ".join(sorted({r.defect for r in test if r.label}))}]')

    def loader(self, files: List[str]) -> DataLoader:
        return DataLoader(IndustrialDataset(files, self.fspec), batch_size=self.fspec.batch_size,
                          shuffle=False, num_workers=self.cfg.num_workers,
                          pin_memory=True, drop_last=False)


def make_pixel_eval(ctx) -> Optional[IndustrialPixelEval]:
    c = ctx.cfg
    if not getattr(c, 'ind_pixel_eval', True):
        return None
    pe = IndustrialPixelEval(ctx, getattr(c, 'ind_eval_size', 0), getattr(c, 'ind_eval_div', DIV_EVAL))
    return pe if pe.hw else None


# The core's Ablation and _proposed_eval call core.PixelEval.make(ctx); route industrial
# contexts to the industrial evaluator (aerial contexts are unaffected).
_core_make = core.PixelEval.make


def _make_dispatch(ctx):
    return make_pixel_eval(ctx) if isinstance(ctx, IndustrialContext) else _core_make(ctx)


core.PixelEval.make = staticmethod(_make_dispatch)


def _context(args, bcfg, fspec, teacher, device, cat: str, patch_layer: int) -> IndustrialContext:
    return IndustrialContext(bcfg, fspec, teacher, device, patch_layer, args.dataset, args.data_root,
                             cat, SEL_FIXED['seed'], VISA_TEST_FRACTION, VISA_SPLIT_CSV)


# ═══════════════════════════════════════════════════════════════════════════
#  §3  EVALUATION OF ONE CATEGORY (Proposed and baselines)
# ═══════════════════════════════════════════════════════════════════════════

def cls_prototype_scores(train_cls, test_cls, bank_idx: List[int], layers) -> np.ndarray:
    """Diagnostic: 1 - max cosine of the test CLS against the bank CLS, mean over layers."""
    it = torch.as_tensor(bank_idx, dtype=torch.long)
    out = []
    for l in layers:
        B = F.normalize(train_cls[l][it].float(), dim=1)
        Q = F.normalize(test_cls[l].float(), dim=1)
        out.append(1.0 - (Q @ B.T).max(dim=1).values)
    return torch.stack(out).mean(0).numpy().astype(np.float64)


def per_defect_auroc(ctx: IndustrialContext, scores) -> Dict[str, float]:
    """Image AUROC of each defect type against all normal test images."""
    lab, d, out = ctx.labels, np.asarray(ctx.defects), {}
    for name in sorted(set(d[lab == 1])):
        sel = (lab == 0) | (d == name)
        if len(np.unique(lab[sel])) == 2:
            out[name] = float(roc_auc_score(lab[sel], np.asarray(scores)[sel]))
    return out


def _common_row(ctx: IndustrialContext) -> dict:
    return {'dataset': f'{ctx.dataset}/{ctx.category}', 'category': ctx.category,
            'n_train': len(ctx.train_files), 'n_test': len(ctx.scenes[0].files),
            'n_anomalous': int(ctx.labels.sum()), 'split': ctx.split_source,
            'official_split': ctx.official_split, 'frozen_config': ctx.cfg.frozen_config}


def evaluate_category(args, ctx: IndustrialContext, spec, out_dir: str) -> List[dict]:
    rows, pl = [], spec.patch_layer
    t0 = time.perf_counter()                       # feature extraction timed separately
    tr, te = ctx.features('train', [pl]), ctx.features('test', [pl])
    extract_s = time.perf_counter() - t0
    seeds = list(args.seeds or [spec.fps_seed])
    if (spec.fps_ratio is None or spec.fps_ratio >= 1.0) and len(seeds) > 1:
        print(f'  [note] full bank: the FPS seed has no effect; using seed {seeds[0]} only')
        seeds = seeds[:1]
    for sd in seeds:
        s = replace(spec, fps_seed=sd)
        m = core.ProposedMethod(s, ctx)
        t1 = time.perf_counter()
        m.fit(ctx)
        fit_s = time.perf_counter() - t1
        pix = make_pixel_eval(ctx)
        ctx.pix = pix
        t2 = time.perf_counter()
        try:
            r = m.score_frames(ctx)
        finally:
            ctx.pix = None
        score_s = time.perf_counter() - t2
        cls_sc = cls_prototype_scores(tr['cls'], te['cls'], m.scorer.idx, s.selected_layers)
        n_bank, k_eff = len(m.scorer.idx), m.scorer._k()
        row = {'method': m.name, 'seed': sd, **_common_row(ctx),
               **core.frame_metrics(ctx.labels, r['scores']),
               'concat_auroc': (core.frame_metrics(ctx.labels, r['concat']).get('auroc')
                                if np.isfinite(r['concat']).all() else None),
               'cls_auroc': core.frame_metrics(ctx.labels, cls_sc).get('auroc'),
               'bank_frames': n_bank, 'k_eff': k_eff, 'retrieval_active': bool(k_eff < n_bank),
               'extract_s': extract_s, 'fit_time_s': fit_s,
               'score_ms_per_image': 1e3 * score_s / max(len(r['scores']), 1),   # cached features
               'per_defect_auroc': per_defect_auroc(ctx, r['scores']), **m.bank_stats()}
        if pix is not None:
            row.update(pix.metrics())
            row['pixel_smooth_sigma_eff'] = pix.sigma
        core.save_run(out_dir, 'main', row, r['scores'], ctx)
        if sd == seeds[0]:
            core.dump_run(os.path.join(out_dir, 'proposed_scores.npy'), m.scorer, ctx.fspec.key(),
                          r['scores'], ctx.labels)
            if not row['retrieval_active']:
                print(f'  [note] bank = {n_bank} images <= K = {s.knn_k}: retrieval inactive')
        print(f'  Proposed seed {sd}: AUROC={core._f(row.get("auroc"))} AP={core._f(row.get("ap"))} '
              f'pAUROC={core._f(row.get("pixel_auroc"))} pAP={core._f(row.get("pixel_ap"))} '
              f'AUPRO={core._f(row.get("aupro"))}  bank={n_bank} K_eff={k_eff}')
        rows.append(row)
        del m
        core._free(ctx.device)
    return rows


def evaluate_baseline(name: str, ctx: IndustrialContext, spec, out_dir: str) -> dict:
    """A core baseline under the identical protocol (preprocessing, metrics, smoothing)."""
    m = core.build_method(name, ctx, ctx.cfg, spec)
    t0 = time.perf_counter()
    m.fit(ctx)
    fit_s = time.perf_counter() - t0
    pix = make_pixel_eval(ctx)
    ctx.pix = pix
    t1 = time.perf_counter()
    try:
        r = m.score_frames(ctx)
    finally:
        ctx.pix = None
    score_s = time.perf_counter() - t1
    row = {'method': m.name, 'seed': None, **_common_row(ctx),
           **core.frame_metrics(ctx.labels, r['scores']),
           'fit_time_s': fit_s, 'score_ms_per_image': 1e3 * score_s / max(len(r['scores']), 1),
           'training_free': m.training_free, 'trainable_params': m.trainable_params(),
           'per_defect_auroc': per_defect_auroc(ctx, r['scores']), **m.bank_stats()}
    if pix is not None:
        row.update(pix.metrics())
        row['pixel_smooth_sigma_eff'] = pix.sigma
    core.save_run(out_dir, 'main', row, r['scores'], ctx)
    print(f'  {m.name}: AUROC={core._f(row.get("auroc"))} AP={core._f(row.get("ap"))} '
          f'pAUROC={core._f(row.get("pixel_auroc"))} AUPRO={core._f(row.get("aupro"))}')
    del m
    core._free(ctx.device)
    return row


# ═══════════════════════════════════════════════════════════════════════════
#  §4  OVERLAYS (core overlay functions; images instead of video frames)
# ═══════════════════════════════════════════════════════════════════════════

def make_overlays(args, ctx: IndustrialContext, spec, out_dir: str):
    m = core.ProposedMethod(spec, ctx)
    m.fit(ctx)
    scores = m.score_frames(ctx)['scores']
    gt = IndustrialGT(ctx.mask_paths, ctx.nonzero_masks)
    out = os.path.join(out_dir, 'overlays')
    os.makedirs(out, exist_ok=True)
    groups = [g for g in core.OVERLAY_GROUPS if args.overlay_mode == 'both' or g.startswith(args.overlay_mode)]
    sel = {}
    for g in groups:
        lab, hi, _ = core.OVERLAY_GROUPS[g]
        sel[g] = core._select_frames(ctx, scores, lab, hi, args.overlay_n, 0, 10 ** 9)
        for e in sel[g]:
            e['map'], _ = core._probe(m, ctx, e['scene'], e['frame'], 1)
    shown = [e for g in groups for e in sel[g]]
    if not shown:
        return
    vmin, vmax = np.percentile(np.concatenate([e['map'].numpy().ravel() for e in shown]), args.vrange)
    grid, rows = (ctx.fspec.grid,) * 2, []
    for g in groups:
        tiles = []
        for rank, e in enumerate(sel[g], 1):
            img, hm, mk = core._tile(e, e['map'], grid, gt, args.sigma)
            stem = os.path.splitext(os.path.basename(e['scene'].files[e['frame']]))[0]
            tiles.append({'img': img, 'hm': hm, 'mask': mk,
                          'caption': f"{ctx.defects[e['frame']]}/{stem}  s={e['score']:.3f}"})
            rows.append({'group': g, 'rank': rank, 'defect': ctx.defects[e['frame']],
                         'label': e['label'], 'score': e['score'], 'file': e['scene'].files[e['frame']]})
        if tiles:
            title = core.OVERLAY_GROUPS[g][2].replace('frames', 'images')
            core._overlay_grid(tiles, f'{title} — {ctx.dataset}/{ctx.category}', out, f'fig_overlay_{g}',
                               vmin, vmax, args.cmap, args.ncol, not args.no_input,
                               'patch discrepancy (1 − max cosine)')
    with open(os.path.join(out, 'overlay_frames.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, 'overlay_meta.json'), 'w') as fh:
        json.dump({'dataset': f'{ctx.dataset}/{ctx.category}', 'n': args.overlay_n, 'groups': groups,
                   'vrange_percentiles': list(args.vrange), 'vmin': float(vmin), 'vmax': float(vmax),
                   'display_sigma': args.sigma, 'cmap': args.cmap,
                   'frozen_config': ctx.cfg.frozen_config}, fh, indent=2)
    del m
    core._free(ctx.device)


# ═══════════════════════════════════════════════════════════════════════════
#  §5  TABLES
# ═══════════════════════════════════════════════════════════════════════════

_KEYS = (('auroc', 'Image AUROC'), ('ap', 'Image AP'), ('pixel_auroc', 'Pixel AUROC'),
         ('pixel_ap', 'Pixel AP'), ('aupro', 'AUPRO@0.3'))
_NAMES = {'mvtec': 'MVTec AD', 'visa': 'VisA'}


def _protocol_text(args) -> str:
    res = ('native mask resolution' if args.pixel_eval_div == 1
           else f'native mask resolution / {args.pixel_eval_div}')
    if args.pixel_smooth_sigma <= 0:
        sm = 'no smoothing of the anomaly maps'
    else:
        sm = (f'anomaly maps smoothed with a Gaussian of σ = {args.pixel_smooth_sigma:g} px at '
              f'{PIXEL_SMOOTH_REF} px, rescaled to the evaluation size, identically for every method')
    return (f'Input {SEL_FIXED["image_size"]}×{SEL_FIXED["image_size"]} ({SEL_FIXED["resize"]}); '
            f'pixel metrics at {res}; {sm}')


def summarize(args, all_rows: Dict[str, List[dict]], out: str):
    ds, name = args.dataset, _NAMES[args.dataset]
    hdr = ['Category'] + [h for _, h in _KEYS] + ['Bank', 'K_eff']
    table, cat_means = [], {k: [] for k, _ in _KEYS}
    for cat, rs in all_rows.items():
        table.append([cat] + [core._ms([r.get(k) for r in rs]) for k, _ in _KEYS]
                     + [str(rs[0]['bank_frames']), str(rs[0]['k_eff'])])
        for k, _ in _KEYS:
            v = core._mean(rs, k)
            if v is not None:
                cat_means[k].append(v)
    table.append(['**Mean**'] + [f'**{np.mean(cat_means[k]):.3f}**' if cat_means[k] else 'n/a'
                                 for k, _ in _KEYS] + ['', ''])
    seeds = max(len(rs) for rs in all_rows.values())
    full = all(rs[0]['bank_frames'] == rs[0]['n_train'] for rs in all_rows.values())
    inactive = [c for c, rs in all_rows.items() if not rs[0]['retrieval_active']]
    unofficial = [c for c, rs in all_rows.items() if not rs[0]['official_split']]
    excl = sum(int(rs[0].get('pixel_excluded') or 0) for rs in all_rows.values())
    cap = (f'Proposed method on {name} ({args.config_source}; nothing tuned on {name}); full-shot '
           f'one-class, bank {"= all normal training images" if full else "subsampled by FPS"}. '
           + (f'Mean±s.d. over {seeds} FPS seeds. ' if seeds > 1 else '')
           + f'{_protocol_text(args)}; {excl} anomalous image(s) without a usable mask excluded.'
           + (f' Retrieval inactive (bank ≤ K) in: {", ".join(inactive)}.' if inactive else '')
           + (f' Non-official split: {", ".join(unofficial)}.' if unofficial else ''))
    core._write(out, f'table5_{ds}', hdr, table, cap, f'tab:{ds}')
    drows = [[cat, d, core._ms([(r.get('per_defect_auroc') or {}).get(d) for r in rs])]
             for cat, rs in all_rows.items()
             for d in sorted({k for r in rs for k in (r.get('per_defect_auroc') or {})})]
    if drows and ds == 'mvtec':                    # VisA 1cls split has no defect types
        core._write(out, f'tableS_{ds}_defects', ['Category', 'Defect type', 'Image AUROC'], drows,
                    f'{name}: image AUROC per defect type against all normal test images.',
                    f'tab:{ds}_defects', 'llc')
    flat = [r for rs in all_rows.values() for r in rs]
    cols = ['category', 'seed'] + [k for k, _ in _KEYS] + [
        'eer', 'tpr_at_fpr05', 'concat_auroc', 'cls_auroc', 'bank_frames', 'k_eff',
        'retrieval_active', 'pixel_eval_hw', 'pixel_smooth_sigma_eff', 'pixel_frames',
        'pixel_regions', 'pixel_excluded', 'pixel_label_mask_agreement', 'bank_mb', 'extract_s',
        'fit_time_s', 'score_ms_per_image', 'n_train', 'n_test', 'n_anomalous', 'split',
        'frozen_config']
    p = os.path.join(out, f'{ds}_summary.csv')
    with open(p, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        w.writerows(flat)
    with open(os.path.join(out, f'captions_{ds}.md'), 'w') as fh:
        fh.write(f'**Table 5 ({name}).** {cap}\n')
    print('\n' + '═' * 80 + f'\n{name} — Proposed\n' + '═' * 80)
    core.print_table([[c.replace('**', '') for c in r] for r in table], hdr)
    print(f'[saved] {p}')


def summarize_methods(args, prop_rows: Dict[str, List[dict]], base_rows: Dict[str, Dict[str, dict]],
                      out: str):
    """Proposed vs baselines: category means of every metric under the identical protocol."""
    ds, name = args.dataset, _NAMES[args.dataset]
    per = defaultdict(dict)                         # method -> category -> {metric: value}
    for cat, rs in prop_rows.items():
        per[rs[0]['method']][cat] = {k: core._mean(rs, k) for k in [k for k, _ in _KEYS] + ['bank_mb']}
    for cat, d in base_rows.items():
        for row in d.values():
            per[row['method']][cat] = {k: core._num(row.get(k)) for k in [k for k, _ in _KEYS] + ['bank_mb']}
    hdr = ['Method', 'Categories'] + [h for _, h in _KEYS] + ['Memory (MB)']
    rows = []
    for meth, cats in per.items():
        r = [meth, str(len(cats))]
        for k in [k for k, _ in _KEYS] + ['bank_mb']:
            v = [x[k] for x in cats.values() if x.get(k) is not None]
            r.append(core._fmt(np.mean(v), 3 if k != 'bank_mb' else 1) if v else 'n/a')
        rows.append(r)
    core._bold_best(rows, range(2, 2 + len(_KEYS)))
    cap = (f'{name}: Proposed and baselines under the identical protocol (same backbone, '
           f'preprocessing and metric code), mean over categories. {_protocol_text(args)}. '
           f'Settings: {args.config_source}.')
    core._write(out, f'table5b_{ds}_methods', hdr, rows, cap, f'tab:{ds}_methods')
    core.print_table([[c.replace('**', '') for c in r] for r in rows], hdr)


def summarize_ablation(out: str, cats):
    """One table per axis: condition means over categories (categories = units of analysis)."""
    for axis in ('A1', 'A2', 'A4', 'A5a', 'A5b', 'A5c', 'A6', 'A7'):
        acc = defaultdict(lambda: defaultdict(list))
        for c in cats:
            p = os.path.join(out, c, 'ablation', f'{axis}.json')
            if os.path.exists(p):
                with open(p) as fh:
                    for r in json.load(fh)['rows']:
                        for k in ('auroc', 'aupro'):
                            if core._num(r.get(k)) is not None:
                                acc[r['name']][k].append(r[k])
        if acc:
            rows = [[n, len(d['auroc']), core._fmt(np.mean(d['auroc'])),
                     core._fmt(np.mean(d['aupro'])) if d['aupro'] else 'n/a'] for n, d in acc.items()]
            core._write(out, f'tableS_ablation_{axis}', ['Condition', 'Categories', 'Mean AUROC',
                        'Mean AUPRO'], rows, f'{axis}, averaged over categories.', f'tab:abl_{axis}')


# ═══════════════════════════════════════════════════════════════════════════
#  §6  CROSS-DATASET SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def _require_official(ctx: IndustrialContext):
    if not ctx.official_split:
        raise RuntimeError(f'{ctx.dataset}/{ctx.category}: non-official split ({ctx.split_source}); '
                           f'selection requires the official split')


def _valid_overrides(ov: dict, warned: set) -> dict:
    ok = {k: v for k, v in ov.items() if k in _BC_FIELDS}
    for k in set(ov) - set(ok) - warned:
        print(f'[select][WARN] BenchConfig has no field {k!r} (update the core); ignored')
        warned.add(k)
    return ok


def select_baselines(args, bcfg, fspec, spec, teacher, device, cats):
    """Baseline settings by the same protocol (core grids, in the core's declared order)."""
    names = [n for n in (args.baselines or SEL_BASELINE_DEFAULT) if n in core.SELECT_BASELINES]
    chosen, records, warned = {}, {}, set()
    for name in names:
        grid = [_valid_overrides(ov, warned) for ov in core.SELECT_BASELINES[name]]
        acc = [[] for _ in grid]
        for cat in cats:
            ctx = _context(args, bcfg, fspec, teacher, device, cat, SEL_PATCH_LAYER)
            _require_official(ctx)
            for i, ov in enumerate(grid):
                m = core.build_method(name, ctx, replace(bcfg, **ov), spec)
                m.fit(ctx)
                acc[i].append(core.frame_metrics(ctx.labels, m.score_frames(ctx)['scores']).get('auroc'))
                del m
                core._free(device)
            del ctx
        means = [float(np.mean(a)) for a in acc]
        j = next(i for i, v in enumerate(means) if v >= max(means) - SEL_TOL)
        chosen.update(grid[j])
        records[name] = [{**ov, 'auroc': v} for ov, v in zip(grid, means)]
        print(f'[select] {name}: {grid[j]}  (mean AUROC {means[j]:.4f})')
    return chosen, records


def run_select(args, bcfg, fspec, base_fspec, spec, teacher, device, cats, out):
    """Cross-dataset selection on args.dataset with the core grid (bank = all training images)."""
    if args.categories:
        print('[select][WARN] --categories given; selection should use ALL categories')
    grid = [replace(spec, fps_ratio=SEL_FPS, selected_layers=tuple(lay),
                    patch_layer=SEL_PATCH_LAYER, knn_k=k)
            for lay in SEL_LAYERS for k in SEL_K]                    # declared order
    acc = {}
    for cat in cats:
        ctx = _context(args, bcfg, fspec, teacher, device, cat, SEL_PATCH_LAYER)
        _require_official(ctx)
        for i, s in enumerate(grid):
            r = core._proposed_eval(ctx, s, SEL_RULES)               # all rules in one pass
            for rn in SEL_RULES:
                a = acc.setdefault((i, rn), {'auroc': [], 'aupro': []})
                a['auroc'].append(r['rules'][rn].get('auroc'))
                a['aupro'].append(r['aupro'])
            print(f'  [select] {cat} layers={list(s.selected_layers)} K={s.knn_k}: '
                  + ', '.join(f'{rn}={core._f(r["rules"][rn].get("auroc"))}' for rn in SEL_RULES)
                  + f'  AUPRO={core._f(r["aupro"])}')
        del ctx
        core._free(device)
    recs = []
    for (i, rn), a in acc.items():                                   # insertion order = declared order
        g, au = grid[i], float(np.mean(a['auroc']))
        pv = [x for x in a['aupro'] if x is not None]
        recs.append({'selected_layers': list(g.selected_layers), 'knn_k': g.knn_k, 'rule': rn,
                     'auroc': au, 'aupro': float(np.mean(pv)) if pv else None, 'score': au})
    best = max(r['score'] for r in recs)
    j = next(k for k, r in enumerate(recs) if r['score'] >= best - SEL_TOL)
    s, a = recs[j], SEL_RULES[recs[j]['rule']]
    chosen = replace(spec, fps_ratio=SEL_FPS, selected_layers=tuple(s['selected_layers']),
                     patch_layer=SEL_PATCH_LAYER, knn_k=s['knn_k'],
                     agg=a.agg, topk_ratio=a.topk_ratio, mad_sigma=a.mad_sigma)
    base_cfg, base_recs = ({}, {})
    if args.select_baselines:
        base_cfg, base_recs = select_baselines(args, bcfg, fspec, spec, teacher, device, cats)
    C = {'proposed': asdict(chosen), 'baselines': base_cfg, 'feature_spec': asdict(base_fspec),
         'dev_dataset': f'{args.dataset} (all {len(cats)} categories)', 'git': core._git_commit(),
         'code_version': core.CODE_VERSION, 'industrial_version': INDUSTRIAL_VERSION,
         'selected_index': j, 'results': recs, 'baseline_results': base_recs,
         'protocol': {
             'criterion': SEL_CRITERION, 'tolerance': SEL_TOL,
             'tie_break': 'first within tolerance in declared order: layers, then K ascending, '
                          'then rules; baselines: core.SELECT_BASELINES order',
             'grid_source': f'{core.__name__}: SELECT_PROPOSED / SELECT_RULES / SELECT_PATCH_LAYER / '
                            f'SELECT_TOL / SELECT_BASELINES',
             'grid': {'selected_layers': [list(l) for l in SEL_LAYERS], 'patch_layer': SEL_PATCH_LAYER,
                      'knn_k': list(SEL_K), 'rules': list(SEL_RULES)},
             'declared_deviation': 'fps_ratio = None (all training images): no temporal redundancy',
             'fixed': {**SEL_FIXED, 'resize_used': fspec.resize, 'image_size_used': fspec.image_size},
             'pixel_eval_div': args.pixel_eval_div, 'pixel_smooth_sigma': args.pixel_smooth_sigma,
             'pixel_smooth_ref': PIXEL_SMOOTH_REF,
             'direction': f'select on {args.dataset}; evaluate on the other industrial dataset'}}
    path = os.path.join(out, f'selected_on_{args.dataset}.json')
    with open(path, 'w') as fh:
        json.dump(C, fh, indent=2, default=str)
    print(f'[select] chosen: {recs[j]}\n[saved] {path}  -> commit, then evaluate the OTHER dataset')


# ═══════════════════════════════════════════════════════════════════════════
#  §7  CONFIGURATION PRINTOUT
# ═══════════════════════════════════════════════════════════════════════════

_BCFG_KEYS = ('data_path', 'save_path', 'cache_dir', 'num_workers', 'gpu_bank_budget_mb',
              'bank_dtype', 'nn_compute_dtype', 'pixel_smooth_sigma', 'pixel_smooth_ref',
              'pixel_max_fpr', 'pixel_skip_inconsistent', 'ind_pixel_eval', 'ind_eval_size',
              'ind_eval_div', 'frozen_config', 'patchcore_ratio', 'patchcore_agg',
              'patchcore_proj_dim', 'ad_tail_q', 'ad_max_bank', 'spade_k_retrieve',
              'simplenet_noise_std', 'padim_pca_dim')


def print_run_config(args, fspec, spec, bcfg, out_dir: Optional[str] = None) -> dict:
    groups = {
        'command line': dict(sorted(vars(args).items())),
        f'FeatureSpec (key {fspec.key()})': asdict(fspec),
        f'MethodSpec (key {spec.key()})': asdict(spec),
        'BenchConfig (used fields)': {k: getattr(bcfg, k) for k in _BCFG_KEYS if hasattr(bcfg, k)},
        'selection grid (from the core)': {'layers': [list(l) for l in SEL_LAYERS],
                                           'patch_layer': SEL_PATCH_LAYER, 'knn_k': list(SEL_K),
                                           'fps_ratio': SEL_FPS, 'rules': list(SEL_RULES),
                                           'tolerance': SEL_TOL, 'fixed': SEL_FIXED},
        'versions': {'core_module': core.__name__, 'core_version': core.CODE_VERSION,
                     'industrial_version': INDUSTRIAL_VERSION, 'git': core._git_commit()},
    }
    for title, d in groups.items():
        print(f'\n─── {title} ' + '─' * max(0, 60 - len(title)))
        for k, v in d.items():
            print(f'  {k:<26} = {v}')
    print()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'run_config.json'), 'w') as fh:
            json.dump(groups, fh, indent=2, default=str)
    return groups


# ═══════════════════════════════════════════════════════════════════════════
#  §8  COMMAND LINE AND MAIN
# ═══════════════════════════════════════════════════════════════════════════

_BASELINES = ['patchcore', 'anomalydino', 'spade', 'padim', 'simplenet']


def parse_args():
    ap = argparse.ArgumentParser(description='MVTec AD / VisA with the Proposed method and baselines')
    # what to run on
    ap.add_argument('--dataset', choices=['mvtec', 'visa'], required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--save_path', default='experiments/industrial/')
    ap.add_argument('--config', default=None,
                    help='development selected_config.json or selected_on_<dataset>.json')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--categories', nargs='+', default=None, help='quick tests only; default: all')
    # configuration variants and repetitions
    ap.add_argument('--full_bank', action='store_true',
                    help='declared variant: fps_ratio=None (all training images) on top of --config')
    ap.add_argument('--seeds', type=int, nargs='+', default=None,
                    help='FPS-start seeds (only meaningful with a subsampled bank)')
    # pixel protocol
    ap.add_argument('--pixel_eval_div', type=int, default=None,
                    help=f'default {DIV_EVAL} (native mask resolution) for evaluation, '
                         f'{DIV_SELECT} with --select')
    ap.add_argument('--pixel_smooth_sigma', type=float, default=4.0,
                    help=f'Gaussian sigma at {PIXEL_SMOOTH_REF} px (rescaled); 0 = unsmoothed run')
    # baselines, selection, ablation
    ap.add_argument('--baselines', nargs='+', default=None, choices=_BASELINES,
                    help='also evaluate these core baselines under the identical protocol')
    ap.add_argument('--select', action='store_true',
                    help='cross-dataset selection on --dataset; writes selected_on_<dataset>.json and exits')
    ap.add_argument('--select_baselines', action='store_true',
                    help='with --select: also select the settings of --baselines')
    ap.add_argument('--ablate_axes', nargs='+', default=None,
                    choices=['A1', 'A2', 'A4', 'A5', 'A6', 'A7'], help='core ablation axes per category')
    ap.add_argument('--ablate_full_bank', action='store_true', help='ablation reference uses fps_ratio=None')
    # overlay figures (presentation only)
    ap.add_argument('--overlays', action='store_true', help='best / worst overlay figures per category')
    ap.add_argument('--overlay_n', type=int, default=10)
    ap.add_argument('--overlay_mode', choices=['best', 'worst', 'both'], default='both')
    ap.add_argument('--ncol', type=int, default=5)
    ap.add_argument('--cmap', default='jet')
    ap.add_argument('--sigma', type=float, default=4.0, help='display smoothing (px); state it in captions')
    ap.add_argument('--vrange', type=float, nargs=2, default=[1.0, 99.5])
    ap.add_argument('--no_input', action='store_true')
    ap.add_argument('--print_config', action='store_true', help='print the effective configuration and exit')
    args = ap.parse_args()
    if args.pixel_eval_div is None:
        args.pixel_eval_div = DIV_SELECT if args.select else DIV_EVAL
    return args


def main():
    args = parse_args()
    core.set_seed(SEL_FIXED['seed'])
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    fspec = IndustrialFeatureSpec(image_size=SEL_FIXED['image_size'], resize=SEL_FIXED['resize'])
    bcfg = core.BenchConfig(data_path=args.data_root, save_path=args.save_path)   # cache, workers, budget,
                                                                                 # max FPR, skip policy: defaults
    bcfg.pixel_smooth_sigma = args.pixel_smooth_sigma
    bcfg.pixel_smooth_ref = PIXEL_SMOOTH_REF
    bcfg.ind_pixel_eval, bcfg.ind_eval_size, bcfg.ind_eval_div = True, 0, args.pixel_eval_div

    base = core.FeatureSpec(**{f.name: getattr(fspec, f.name) for f in fields(core.FeatureSpec)})
    spec = core.MethodSpec()
    args.config_source = 'a-priori defaults (no --config)'
    if args.config:
        spec = core._apply_config(args.config, bcfg, base, spec)
        with open(args.config) as fh:
            dev = json.load(fh).get('dev_dataset', '?')
        args.config_source = f'configuration {os.path.basename(args.config)}, selected on {dev}'
    else:
        print('[WARN] no --config: a-priori MethodSpec defaults, NOT a selected configuration')
    if args.full_bank:
        spec = replace(spec, fps_ratio=None)
        args.config_source += '; full bank (declared variant)'

    out = os.path.join(args.save_path, args.dataset)
    cats = args.categories or (MVTEC_CATEGORIES if args.dataset == 'mvtec' else VISA_CATEGORIES)
    print_run_config(args, fspec, spec, bcfg, out)
    if args.print_config:
        return
    for cat in cats:
        if not os.path.isdir(os.path.join(args.data_root, cat)):
            raise FileNotFoundError(f'category directory not found: {os.path.join(args.data_root, cat)}')
    core.write_env(out, device, fspec, bcfg)
    teacher = core.DINOv2AllLayers(fspec.backbone).to(device)

    if args.select:
        if args.seeds or args.overlays or args.ablate_axes:
            print('[select] note: --seeds / --overlays / --ablate_axes are ignored in selection mode')
        run_select(args, bcfg, fspec, base, spec, teacher, device, cats, out)
        return

    prop_rows: Dict[str, List[dict]] = {}
    base_rows: Dict[str, Dict[str, dict]] = {}
    for cat in cats:
        ctx = _context(args, bcfg, fspec, teacher, device, cat, spec.patch_layer)
        cdir = os.path.join(out, cat)
        prop_rows[cat] = evaluate_category(args, ctx, spec, cdir)
        for name in args.baselines or []:
            base_rows.setdefault(cat, {})[name] = evaluate_baseline(name, ctx, spec, cdir)
        if args.overlays:
            make_overlays(args, ctx, spec, cdir)
        if args.ablate_axes:
            ab_spec = replace(spec, fps_ratio=None) if args.ablate_full_bank else spec
            ab = core.Ablation(ctx, bcfg, ab_spec, os.path.join(cdir, 'ablation'),
                               n_boot=1000, block_len=1, a4_repeats=A4_REPEATS)  # i.i.d. bootstrap
            run = {'A1': ab.a1, 'A2': ab.a2, 'A4': ab.a4, 'A5': ab.a5, 'A6': ab.a6, 'A7': ab.a7}
            for a in args.ablate_axes:
                run[a]()
        del ctx
        core._free(device)

    summarize(args, prop_rows, out)
    if base_rows:
        summarize_methods(args, prop_rows, base_rows, out)
    if args.ablate_axes:
        summarize_ablation(out, list(prop_rows))


if __name__ == '__main__':
    main()
