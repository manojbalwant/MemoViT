"""
Revised DinoV2 zero-shot anomaly detection (multilayer CLS-concatenated KNN + last-layer patch discrepancy)
- More robust data collection (returns per-image last-layer patches too)
- Optional concat compression: random projection OR greedy farthest-point sampling (CPU)
- Better FAISS / torch fallback handling
- Cleaner plotting and error handling
- Minor fixes and clearer variable names

NOTE: This file is intended to replace the original script the user provided. It is self-contained
(except for dataset/DataLoader utilities which the user already had: `data_utils_norm.DataLoader` and
`compute_mean_and_std`).

Author: ChatGPT (code revision)
"""

import os
import glob
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import timm
from torch.nn import init
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scoring import robust_patch_score   # canonical reported scorer (Eq. 7)
import cv2
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
from scipy.ndimage import gaussian_filter

# Optional - try import faiss, otherwise fallback
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

# -------------------- Configuration --------------------
class Config:
    def __init__(self):
        self.data_path = 'dataset/Drone-Anomaly/Solar Panel Inspection/' #'dataset/UIT-ADrone/'    #'dataset/shanghaitech/'#'dataset/avenue/'      
        self.save_path = 'experiments/Solar Panel Inspection/' #'experiments/UIT-ADrone/' #'experiments/shanghaitech/' #'experiments/avenue/'     
        # IMAGE/PATCH
        self.image_size = 518
        # MODEL / TRAINING
        self.batch_size = 16
        self.num_workers = 8
        self.num_frames = 1
        self.train = 1
        # scoring
        self.scoring =  'cosine'#'l2'  # 'cosine', , 'mahalanobis'
        self.heatmap_interp = cv2.INTER_LINEAR
        # layer handling
        self.selected_layers = [7,8,9,10,11] #[9,10,11] #list(range(12))#[11] #None 
        self.display_frames = True#False

        # ----- new knn / retrieval params -----
        self.knn_k = 5
        self.knn_metric =  'cosine' #'Euclidean'#
        self.use_faiss_index = True and _HAS_FAISS
        # whether to use fps-based compression of concatenated dims (cpu)
        self.concat_fps_target = 0.25#4096  # if using fps #None for no FPS
        # single-shot PCA dimensionality to apply on full concatenated CLS features BEFORE FPS (None -> skip)
        self.per_layer_pca_dim = None   # e.g. 64 / 128 / 256; None to disable
        self.concat_pca_dim = None#1024#512#1280 #768   # e.g., 512; None to disable
        self.use_last_layer_patches_for_knn = True#False


def print_config(config):
    print("Configuration Summary:")
    for key, value in config.__dict__.items():
        print(f"{key}: {value}")

def compute_eer(y_true, y_score):
    """
    Equal Error Rate (EER) — the operating point where FPR ≈ FNR.

    Uses sklearn's roc_curve so it is consistent with how frame-level AUROC
    is already computed in this codebase.

    Algorithm
    ---------
    roc_curve returns (fpr, tpr, thresholds).
    FNR = 1 - TPR.  We find the index where |FPR - FNR| is minimised,
    then interpolate linearly between the two bracketing points so the
    returned value is not artificially quantised to the discrete threshold grid.

    Args
    ----
    y_true  : array-like  (N,)  binary ground-truth labels {0, 1}
    y_score : array-like  (N,)  anomaly scores (higher = more anomalous)

    Returns
    -------
    eer       : float in [0, 1]   lower is better
    threshold : float             decision threshold at the EER point
    """
    from sklearn.metrics import roc_curve

    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    if len(np.unique(y_true)) < 2:
        print("[EER] Undefined — GT has only one class. Returning (0.5, nan).")
        return 0.5, float('nan')

    fpr, tpr, thresholds = roc_curve(y_true, y_score, drop_intermediate=False)
    fnr = 1.0 - tpr                          # FNR = Miss Rate = 1 - Recall

    # Index of the threshold where |FPR - FNR| is smallest
    idx = int(np.argmin(np.abs(fpr - fnr)))

    # --- linear interpolation between the two bracketing points -----------
    # Avoids the quantisation artifact when the exact crossing falls between
    # two adjacent thresholds on the discrete curve.
    if idx > 0 and idx < len(fpr) - 1:
        # Bracket: points idx-1 and idx
        fpr0, fnr0 = fpr[idx - 1], fnr[idx - 1]
        fpr1, fnr1 = fpr[idx],     fnr[idx]
        # Parametric form: P(t) = P0 + t*(P1-P0)
        # Solve fpr(t) = fnr(t):  fpr0 + t*(fpr1-fpr0) = fnr0 + t*(fnr1-fnr0)
        denom = (fpr1 - fpr0) - (fnr1 - fnr0)
        if abs(denom) > 1e-12:
            t     = (fnr0 - fpr0) / denom          # interpolation parameter
            t     = float(np.clip(t, 0.0, 1.0))
            eer   = float(fpr0 + t * (fpr1 - fpr0))
            thr   = float(thresholds[idx - 1]
                          + t * (thresholds[idx] - thresholds[idx - 1]))
        else:
            eer = float((fpr[idx] + fnr[idx]) / 2.0)
            thr = float(thresholds[idx])
    else:
        eer = float((fpr[idx] + fnr[idx]) / 2.0)
        thr = float(thresholds[idx])

    return eer, thr
    
def compute_mean_and_std(train_folder, resize_height, resize_width, device, batch_size=64, num_workers=2):
    # Define the image transformation: resize and convert to tensor.
    transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
    ])
    
    # Create the dataset and data loader.
    dataset = datasets.ImageFolder(train_folder, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Determine the number of channels (e.g., 3 for RGB)
    channels = dataset[0][0].shape[0]
    
    # Initialize variables to accumulate the sums and squared sums.
    mean = torch.zeros(channels, device=device)  # Use GPU for mean
    std = torch.zeros(channels, device=device)   # Use GPU for std
    total_pixels = 0  # Total number of pixels per channel across all images.
    
    # Ensure no gradients are tracked.
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing mean and std"):
            images = images.to(device)  # Move images to GPU
            
            batch_size, c, h, w = images.shape
            # Update the total number of pixels.
            total_pixels += batch_size * h * w
            
            # Sum over all images in the batch (across height and width) for each channel.
            mean += images.sum(dim=[0, 2, 3])  # In-place addition
            # Sum of squares for each channel.
            std += (images ** 2).sum(dim=[0, 2, 3])  # In-place addition
    
    # Compute the mean per channel.
    mean /= total_pixels
    # Compute variance and then std deviation.
    std = torch.sqrt(std / total_pixels - mean ** 2)
    
    return mean.cpu(), std.cpu()  # Move results back to CPU if needed

def np_load_frame(filename, resize_height, resize_width):
    """
    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    return image_resized

class CustomFrameDataset(data.Dataset):
    def __init__(self, video_folder, transform, resize_height, resize_width, time_step=4, num_pred=0, frame_step=1, return_image_path=False):
        self.dir = video_folder
        self.transform = transform
        self.video_frames = []
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.return_image_path = return_image_path
        self._frame_step = frame_step
        self.index_samples = []
        self.setup()

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        videos.sort()
        
        all_video_frames = []
        if os.path.isdir(videos[0]):
            for video in videos:
                vide_frames = glob.glob(os.path.join(video, '*.jpg'))
                vide_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
                all_video_frames.extend(vide_frames)
        else:
            videos.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))
            all_video_frames = videos
        
        self.video_frames = all_video_frames
        max_index = len(all_video_frames) - (self._time_step + self._num_pred - 1) * self._frame_step
        self.index_samples = list(range(max_index))

    def __getitem__(self, index):
        frame_index = self.index_samples[index]
        batch_frames = np.zeros((self._time_step + self._num_pred, 3, self._resize_height, self._resize_width))
        
        # Pre-load frames in memory
        for i in range(self._time_step + self._num_pred):
            frame_path = self.video_frames[frame_index + i * self._frame_step]
            image = np_load_frame(frame_path, self._resize_height, self._resize_width)
            
            if self.transform:
                image = Image.fromarray(image)
                batch_frames[i] = self.transform(image)
        
        if self.return_image_path:
            image_path = self.video_frames[frame_index]
            return {'256': batch_frames, 'standard': batch_frames, 'image_path': image_path}
        
        return {'256': batch_frames, 'standard': batch_frames}

    def __len__(self):
        return len(self.index_samples)

# -------------------- Utilities --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Model Helpers --------------------

def _get_embed_dim_from_vit(vit):
    return getattr(vit, 'embed_dim', getattr(vit, 'num_features', None)) or 768


# -------------------- Teacher: collect per-layer CLS + per-layer patches --------------------
class TeacherMultiLayer(nn.Module):
    """
    Runs timm ViT, collects per-layer raw cls tokens and patch tokens.
    Returns:
      - cls_raw_list: list of tensors (B, C) raw CLS per layer
      - cls_raw_list_dup: duplicate for signature compatibility
      - patch_tokens_list: list of tensors (B, P, C) per layer
    """
    def __init__(self, model_name='vit_base_patch14_dinov2', pretrained=True, selected_layers=None):
        super(TeacherMultiLayer, self).__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        # remove heads if present
        for attr in ['head', 'head_dist', 'fc']:  # try commonly used head names
            if hasattr(self.vit, attr):
                try:
                    setattr(self.vit, attr, nn.Identity())
                except Exception:
                    pass

        self.embed_dim = _get_embed_dim_from_vit(self.vit)
        self.n_blocks = len(getattr(self.vit, 'blocks', []))
        # decide which layers to capture
        if selected_layers is None:
            self.selected_layers = list(range(self.n_blocks))
        else:
            self.selected_layers = [l for l in selected_layers if 0 <= l < self.n_blocks]

    def forward(self, x):
        # x: (B, C, H, W)
        # NOTE: different timm variants may implement patch_embed as a layer returning (B, N, D)
        x = self.vit.patch_embed(x)
        B, N, D = x.shape
        has_cls = hasattr(self.vit, 'cls_token')
        has_dist = hasattr(self.vit, 'dist_token')
        if has_cls:
            cls_token = self.vit.cls_token.expand(B, -1, -1)
        if has_dist:
            dist_token = self.vit.dist_token.expand(B, -1, -1)
            tokens = torch.cat((cls_token, dist_token, x), dim=1)
        elif has_cls:
            tokens = torch.cat((cls_token, x), dim=1)
        else:
            tokens = x

        if hasattr(self.vit, 'pos_embed'):
            pos = self.vit.pos_embed.to(tokens.device)
            tokens = tokens + pos

        cls_raw_list = []
        patch_tokens_list = []

        # iterate through blocks and capture at selected indices
        for i, block in enumerate(getattr(self.vit, 'blocks', [])):
            tokens = block(tokens)
            if i in self.selected_layers:
                if has_dist:
                    cls_raw = tokens[:, 0:2].mean(dim=1)
                    patch_tokens = tokens[:, 2:, :].contiguous()
                elif has_cls:
                    cls_raw = tokens[:, 0, :].contiguous()
                    patch_tokens = tokens[:, 1:, :].contiguous()
                else:
                    cls_raw = tokens.mean(dim=1)
                    patch_tokens = tokens.contiguous()
                cls_raw_list.append(cls_raw)
                patch_tokens_list.append(patch_tokens)

        # final norm replacement for last captured layer (stability)
        if hasattr(self.vit, 'norm') and len(self.selected_layers) > 0:
            normed = self.vit.norm(tokens)
            last_idx = self.selected_layers[-1]
            # replace last captured layer outputs with normed outputs
            if has_dist:
                cls_final = normed[:, 0:2].mean(dim=1)
                patch_final = normed[:, 2:, :].contiguous()
            elif has_cls:
                cls_final = normed[:, 0, :]
                patch_final = normed[:, 1:, :].contiguous()
            else:
                cls_final = normed.mean(dim=1)
                patch_final = normed.contiguous()
            cls_raw_list[-1] = cls_final
            patch_tokens_list[-1] = patch_final

        return cls_raw_list, patch_tokens_list

# -------------------- Helpers: load & preprocess images given indices --------------------
def load_and_preprocess_images(image_paths, indices, image_size, mean, std, device):
    # indices -> list ensured earlier...
    imgs = []
    for i in indices:
        p = image_paths[int(i)]
        img_np = np_load_frame(p, image_size, image_size)   # uses cv2.resize like dataset
        img = Image.fromarray(img_np)
        img_t = transforms.ToTensor()(img)
        img_t = transforms.Normalize(mean=mean, std=std)(img_t)
        imgs.append(img_t)
    batch = torch.stack(imgs, dim=0).float().to(device)
    return batch

    
# -------------------- Helpers: load & preprocess images given indices --------------------
# Image loading helpers removed — retrieval uses memory bank only.
# If you need to visualize or recompute patches on-the-fly, reintroduce dedicated loader functions.

# -------------------- Compression helpers: greedy farthest point sampling (CPU) --------------------

def greedy_farthest_point_sampling_rows_cpu(feature_matrix_np, target_n):
    """
    Farthest point sampling over ROWS (images). Given feature_matrix_np shape (N_images, D),
    select `target_n` rows that are diverse in the D-dimensional space.
    Returns selected row indices as a numpy int32 array.
    """
    assert feature_matrix_np.ndim == 2
    N, D = feature_matrix_np.shape
    if target_n >= N:
        return np.arange(N, dtype=np.int32)

    # normalize rows to avoid scale bias
    rows = feature_matrix_np.copy()
    norms = np.linalg.norm(rows, axis=1, keepdims=True) + 1e-12
    rows = rows / norms

    # pick first row index deterministically (largest norm after normalization -> first row)
    selected = [0]
    dists = np.sum((rows - rows[selected[0]:selected[0]+1])**2, axis=1)
    for _ in range(1, target_n):
        idx = int(np.argmax(dists))
        selected.append(idx)
        newd = np.sum((rows - rows[idx:idx+1])**2, axis=1)
        dists = np.minimum(dists, newd)

    return np.array(selected, dtype=np.int32)



# -------------------- Memory bank collection: concatenated per-image projected CLS + per-image last layer patches --------------------
import gc

def collect_concatenated_memory_bank(teacher, loader, device, config, train_folder=None):
    """
    Two-pass collection:
      Pass 1 — CLS tokens only (cheap) → FPS → sel_rows_list
      Pass 2 — Patch tokens only for FPS-selected rows (controlled memory)
    """
    teacher.eval()

    # ─────────────────────────────────────────────────────────────
    # PASS 1 — Collect CLS tokens only (NO patch accumulation)
    # Memory: O(N × L × C) ≈ manageable even for 70k images
    # ─────────────────────────────────────────────────────────────
    concat_list   = []
    per_layer_raw = None

    with torch.no_grad():
        for batch in tqdm(loader, desc='[Pass 1] CLS tokens (all images)'):
            inputs = batch['standard'][:, 0].float().to(device)   # (B, C, H, W)
            cls_raw_list, _patch_tokens = teacher(inputs)          # patches discarded ←─ KEY

            cls_raw_list = [t.detach().cpu() for t in cls_raw_list]
            concat_list.append(torch.cat(cls_raw_list, dim=1))     # (B, L*C)

            if per_layer_raw is None:
                per_layer_raw = [[] for _ in range(len(cls_raw_list))]
            for i, cr in enumerate(cls_raw_list):
                per_layer_raw[i].append(cr)

    # Assemble CLS tensors
    per_image_concat     = torch.cat(concat_list, dim=0)           # (N, L*C)
    per_image_concat_cpu = per_image_concat.cpu()
    per_layer_raw        = [torch.cat(lst, dim=0) for lst in per_layer_raw]

    # Free Pass-1 intermediates immediately
    del concat_list, per_image_concat
    gc.collect()

    # Recover dataset file list
    ds        = getattr(loader, 'dataset', None)
    file_list = list(ds.video_frames)

    # ── (Optional) per-layer PCA — unchanged from original ────────
    pca_meta = None
    per_layer_pca_dim = getattr(config, 'per_layer_pca_dim', None)

    if per_layer_pca_dim is not None:
        L = len(per_layer_raw)
        pca_meta_list, projected_layers = [], []

        for li in range(L):
            arr = per_layer_raw[li].numpy().astype('float32')
            N_img, C = arr.shape
            k = int(min(per_layer_pca_dim, N_img - 1, C))
            if k <= 0 or k >= C:
                projected_layers.append(arr); pca_meta_list.append(None); continue
            mean_l = arr.mean(axis=0, keepdims=True).astype('float32')
            Xc     = arr - mean_l
            try:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            except np.linalg.LinAlgError as e:
                print(f'[WARN] SVD failed layer {li} ({e}); using raw.'); projected_layers.append(arr); pca_meta_list.append(None); continue
            components = Vt[:k].astype('float32')
            projected_layers.append(np.dot(Xc, components.T).astype('float32'))
            pca_meta_list.append({'mean': mean_l, 'components': components})

        per_image_concat_cpu = torch.from_numpy(
            np.concatenate(projected_layers, axis=1))
        pca_meta = pca_meta_list
        del projected_layers; gc.collect()

    elif getattr(config, 'concat_pca_dim', None) is not None:
        arr = per_image_concat_cpu.numpy().astype('float32')
        N_img, D_concat = arr.shape
        k = int(min(config.concat_pca_dim, N_img - 1, D_concat))
        if 0 < k < D_concat:
            mean_col = arr.mean(axis=0, keepdims=True).astype('float32')
            Xc = arr - mean_col
            try:
                _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                components = Vt[:k].astype('float32')
                per_image_concat_cpu = torch.from_numpy(
                    np.dot(Xc, components.T).astype('float32'))
                pca_meta = [{'mean': mean_col, 'components': components,
                             '_legacy_single_shot': True}]
            except Exception as e:
                print(f'[WARN] Legacy PCA failed ({e}); skipping.')

    # ── FPS on CLS (cheap) ────────────────────────────────────────
    sel_rows_list = None
    if config.concat_fps_target is not None:
        arr   = per_image_concat_cpu.numpy().astype('float32')
        N_img = arr.shape[0]
        if config.concat_fps_target < N_img:
            print(f'[FPS] Selecting {config.concat_fps_target} / {N_img} images...')
            sel_rows      = greedy_farthest_point_sampling_rows_cpu(arr, config.concat_fps_target)
            sel_rows_list = [int(i) for i in sel_rows]
            per_image_concat_cpu = per_image_concat_cpu[sel_rows_list]
            per_layer_raw        = [feat[sel_rows_list] for feat in per_layer_raw]
            file_list            = [file_list[i] for i in sel_rows_list]
            print(f'[FPS] Done. Kept {len(sel_rows_list)} images.')
        else:
            print(f'[FPS] target >= N — skipping.')

    # ─────────────────────────────────────────────────────────────
    # PASS 2 — Patch tokens ONLY for FPS-selected images
    # Memory: O(FPS_target × P × C/2)  ← fp16 halves footprint
    # ─────────────────────────────────────────────────────────────
    per_image_last_patches = None

    if config.use_last_layer_patches_for_knn:
        subset_indices = sel_rows_list if sel_rows_list is not None \
                         else list(range(len(ds)))
        M = len(subset_indices)
        print(f'[Pass 2] Collecting patch tokens for {M} FPS-selected images...')

        subset_ds     = torch.utils.data.Subset(ds, subset_indices)
        subset_loader = torch.utils.data.DataLoader(
            subset_ds,
            batch_size  = config.batch_size,
            shuffle     = False,
            num_workers = config.num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        # ── Determine patch tensor shape from one forward pass ────
        with torch.no_grad():
            probe_batch = next(iter(subset_loader))
            probe_input = probe_batch['standard'][:1, 0].float().to(device)
            _, probe_patches = teacher(probe_input)
            P_last  = probe_patches[-1].shape[1]   # number of spatial patches
            C_last  = probe_patches[-1].shape[2]   # feature dim
            del probe_patches
        print(f'[Pass 2] Patch shape per image: ({P_last}, {C_last})')

        # ── Pre-allocate fp16 tensor (FIX 2 + FIX 3) ─────────────
        # fp16: saves 50% RAM vs fp32; sufficient precision for cosine scoring
        patch_bank = torch.empty(
            (M, P_last, C_last),
            dtype  = torch.float16,   # ← fp16
            device = 'cpu',
        )
        write_ptr = 0

        with torch.no_grad():
            for batch in tqdm(subset_loader, desc='[Pass 2] Patch tokens (FPS subset)'):
                inputs = batch['standard'][:, 0].float().to(device)
                _, patch_tokens_list = teacher(inputs)
                patches_cpu = patch_tokens_list[-1].detach().cpu().half()  # (B, P, C) fp16
                B_cur = patches_cpu.shape[0]
                patch_bank[write_ptr : write_ptr + B_cur] = patches_cpu
                write_ptr += B_cur

        per_image_last_patches = patch_bank   # (M, P_last, C_last) fp16
        print(f'[Pass 2] Patch bank shape: {per_image_last_patches.shape}, '
              f'dtype: {per_image_last_patches.dtype}, '
              f'RAM ≈ {per_image_last_patches.element_size() * per_image_last_patches.numel() / 1e9:.2f} GB') 

    return per_image_concat_cpu, per_layer_raw, per_image_last_patches, file_list, pca_meta


# -------------------- Build KNN index (FAISS preferred, else torch fallback) --------------------
def build_knn_index(features_cpu, metric='cosine', use_faiss=True):
    feats = features_cpu.astype('float32') if isinstance(features_cpu, np.ndarray) else features_cpu.numpy().astype('float32')
    N, D = feats.shape
    if use_faiss and _HAS_FAISS:
        if metric == 'cosine':
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats_n = feats / norms
            index = faiss.IndexFlatIP(D)
            index.add(feats_n)
            return {'index': index, 'normalized': True, 'feats_shape': feats.shape}
        else:
            index = faiss.IndexFlatL2(D)
            index.add(feats)
            return {'index': index, 'normalized': False, 'feats_shape': feats.shape}
    else:
        # Torch fallback: store normalized features for cosine, raw for l2
        if metric == 'cosine':
            norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
            feats_n = feats / norms
            return {'features': feats_n, 'metric': metric, 'normalized': True, 'feats_shape': feats.shape}
        else:
            return {'features': feats, 'metric': metric, 'normalized': False, 'feats_shape': feats.shape}


def knn_search(index_obj, query_vec, k=5):
    if isinstance(query_vec, torch.Tensor):
        q = query_vec.detach().cpu().numpy().astype('float32')
    else:
        q = np.asarray(query_vec).astype('float32')

    if 'index' in index_obj:
        idx = index_obj['index']
        if index_obj.get('normalized', False):
            qn = q / (np.linalg.norm(q) + 1e-8)
            qn = qn.reshape(1, -1)
            dists, inds = idx.search(qn, k)
            # FAISS IndexFlatIP returns inner products (higher -> more similar)
            return dists.ravel(), inds.ravel()
        else:
            qn = q.reshape(1, -1)
            dists, inds = idx.search(qn, k)
            # FAISS IndexFlatL2 returns squared L2 distances
            # Convert squared distances to Euclidean distances for consistency with torch.cdist:
            return np.sqrt(dists).ravel(), inds.ravel()
    else:
        feats = index_obj['features']  # (N, D) -- note: for cosine this is normalized already per build_knn_index
        metric = index_obj['metric']
        if index_obj.get('normalized', False):
            qn = q / (np.linalg.norm(q) + 1e-8)
            sims = feats @ qn  # feats should already be normalized
            inds = np.argsort(-sims)[:k]
            return sims[inds], inds
        else:
            # Return Euclidean distances (not squared)
            d2 = np.sum((feats - q.reshape(1, -1)) ** 2, axis=1)
            d = np.sqrt(d2)
            inds = np.argsort(d)[:k]
            return d[inds], inds

# -------------------- Scoring helpers (unchanged) --------------------
def score_cls_vector_vs_prototypes(cls_vec, prototypes, metric='cosine', mu=None, cov_inv=None):
    if cls_vec.device != prototypes.device:
        cls_vec = cls_vec.to(prototypes.device)
    cls = cls_vec.unsqueeze(0)  # (1, D)
    if metric == 'cosine':
        cls_n = F.normalize(cls, dim=1)
        prot_n = F.normalize(prototypes, dim=1)
        sim = torch.matmul(cls_n, prot_n.t())  # (1, K)
        max_sim = sim.max(dim=1)[0].item()
        return 1.0 - max_sim
    elif metric == 'l2':
        dists = torch.cdist(cls, prototypes)  # (1, K)
        min_d = dists.min().item()
        return float(min_d)
    elif metric == 'mahalanobis':
        d = (cls.squeeze(0).cpu() - mu.cpu()).unsqueeze(0)
        m = (d @ cov_inv.cpu() @ d.t()).squeeze().item()
        return float(m)
    else:
        raise ValueError('Unknown metric')


def aggregate_layer_scores(layer_scores, reduce='mean'):
    L = len(layer_scores)
    layer_weights = [1.0] * L
    if reduce == 'max':
        return max(layer_scores)
    ws = np.array(layer_weights, dtype=np.float32)
    vals = np.array(layer_scores, dtype=np.float32)
    return float((ws * vals).sum() / (ws.sum() + 1e-12))

def iqr_patch_score(disc_np):
    disc_np = disc_np.detach().float().flatten().cpu().numpy()
    q1 = np.percentile(disc_np, 25)
    q3 = np.percentile(disc_np, 75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    outlier_mask = disc_np > threshold
    k = int(np.sum(outlier_mask))
    if k == 0:
        # No outliers: frame is normal — return a near-zero reference score
        return float(np.percentile(disc_np, 95))   # soft fallback, not 0
    return float(np.mean(np.partition(disc_np, -k)[-k:]))

# robust_patch_score (Eq. 7, the reported scorer) is imported from scoring.py.

# -------------------- Localization: KNN over concatenated multi-layer CLS + on-the-fly patch discrepancy --------------------
def cls_patch_similarity_heatmaps_concatenated(
    cls_latents_list,
    patch_tokens_list,
    per_image_concat_cpu,
    concat_index_obj,
    config,
    teacher,
    train_image_paths,
    per_image_last_patches_cpu=None,
    mean=None, std=None,
    knn_k=5,
    knn_metric='cosine',
    pca_meta = None,
    img=None, mean_for_display=None, std_for_display=None,
    figsize=(12,5), alpha=0.7,
    device=None,
    # plotting control: function will only plot when this frame is labeled anomalous (matching previous behavior)
    frame_idx=None,
    frame_label_array=None
):
    """
    Retrieval now uses concatenated RAW CLS across layers (cls_latents_list holds raw CLS).
    Retrieved images' last-layer patches are recomputed on-the-fly and compared with test last-layer patches.
    Returns a heatmap upsampled to image resolution.
    """
    import matplotlib.pyplot as plt
    if device is None:
        device = next(teacher.parameters()).device

    def ensure_batch(t):
        if t.dim() == 2: return t.unsqueeze(0)
        if t.dim() == 1: return t.unsqueeze(0).unsqueeze(0)
        return t

    test_last_patches = ensure_batch(patch_tokens_list[-1])   # (B, P_last, C_last)
    B, P_last, C_last = test_last_patches.shape
    heatmaps = []
    patch_scores = []
    concat_scores = []

    for b in range(B):
        # ── build query vector (per-layer PCA aware) ──────────────────────────
        concat_vec_parts = [
            cls_latents_list[i][b].detach().cpu().numpy().astype('float32')
            for i in range(len(cls_latents_list))
        ]                                                    # list[L] each (C,)

        if pca_meta is not None:
            # Check whether this is the new per-layer list or the legacy single-shot list
            legacy = (
                len(pca_meta) == 1
                and isinstance(pca_meta[0], dict)
                and pca_meta[0].get('_legacy_single_shot', False)
            )
            if legacy:
                # ── legacy path: project full concat vector ────────────────
                raw_concat = np.concatenate(concat_vec_parts, axis=0)        # (L*C,)
                qc = raw_concat - pca_meta[0]['mean'].ravel()
                query_concat = np.dot(qc, pca_meta[0]['components'].T).astype('float32')
            else:
                # ── new path: project each layer independently, then concat ─
                projected_parts = []
                for li, vec in enumerate(concat_vec_parts):
                    meta_l = pca_meta[li] if li < len(pca_meta) else None
                    if meta_l is None:
                        # this layer was skipped (degenerate); use raw
                        projected_parts.append(vec)
                    else:
                        vec_c = vec - meta_l['mean'].ravel()                 # (C,)
                        vec_p = np.dot(vec_c, meta_l['components'].T)        # (k,)
                        projected_parts.append(vec_p.astype('float32'))
                query_concat = np.concatenate(projected_parts, axis=0)       # (L*k,)
        else:
            # no PCA: plain concatenation of raw CLS vectors
            query_concat = np.concatenate(concat_vec_parts, axis=0).astype('float32')  # (L*C,)

        dists, inds = knn_search(concat_index_obj, query_concat, k=knn_k)

        # retrieve last-layer patches from memory bank (no image loading)
        inds_clamped = [int(i) for i in inds]

        if config.use_last_layer_patches_for_knn:
            # Fast path: directly retrieve precomputed patch tokens from CPU cache
            retrieved_patches = per_image_last_patches_cpu[inds_clamped].to(device).float()  # (K, P_last, C_last)
        else:
            # Slow path: load images and run teacher model to extract patches
            retrieved_batch = load_and_preprocess_images(
                train_image_paths, inds_clamped, config.image_size,
                mean.tolist() if isinstance(mean, torch.Tensor) else mean,
                std.tolist() if isinstance(std, torch.Tensor) else std,
                device
            )
            teacher.eval()
            with torch.no_grad():
                r_cls_raw_list, r_patch_tokens_list = teacher(retrieved_batch)
            retrieved_patches = r_patch_tokens_list[-1]  # (K, P_last, C_last)     

        # --- compute concat-based score using retrieved concatenated CLS vectors from memory bank ---
        if knn_metric == 'cosine':
            # dists are cosine similarities for top-k, use 1 - max(sim) to match earlier semantics
            concat_score = float(1.0 - np.mean(dists))#float(1.0 - np.max(dists))
        else:
            # dists are Euclidean distances (not squared)
            concat_score = float(np.mean(dists))#float(np.min(dists)) 
        concat_scores.append(concat_score)

        # no image thumbnails available by default; retrieval-only mode — skip neighbor visualization
        K = retrieved_patches.shape[0]

        test_patches = test_last_patches[b].unsqueeze(0).to(device)  # (1,P_last,C)

        # compute per-patch discrepancy
        if knn_metric == 'cosine':
            test_n = F.normalize(test_patches, dim=2)
            retrieved_n = F.normalize(retrieved_patches.to(device), dim=2)
            retrieved_n_flat = retrieved_n.view(-1, retrieved_n.shape[2])  # (K*P, C)
            test_n_flat = test_n.squeeze(0)  # (P, C)
            sim_kp_flat = torch.matmul(test_n_flat.to(retrieved_n_flat.device), retrieved_n_flat.t())  # (P, K*P)
            max_sim_per_patch, _ = sim_kp_flat.max(dim=1)  # (P,)
            patch_discrepancy = 1.0 - max_sim_per_patch
        else:
            test_expand = test_patches.expand(retrieved_patches.shape[0], -1, -1).to(retrieved_patches.device)
            d2_kp = ((retrieved_patches.to(device) - test_expand) ** 2).sum(dim=2)
            min_d2_per_patch, _ = d2_kp.min(dim=0)
            patch_discrepancy = min_d2_per_patch.sqrt()

        grid_last = int(P_last ** 0.5)
        patch_map = patch_discrepancy.view(1, grid_last, grid_last).unsqueeze(1)
        patch_map_up = F.interpolate(patch_map, size=(config.image_size, config.image_size), mode='bilinear', align_corners=False).squeeze(1)

        pm_flat = patch_map_up.view(-1)
        pmn = pm_flat.min()
        pmx = pm_flat.max()
        patch_norm = (patch_map_up - pmn) / (pmx - pmn + 1e-8)

        fused_map = patch_norm
        fused_map_np = fused_map.squeeze(0).cpu().numpy()
        # scalar patch-based anomaly score: mean normalized discrepancy over the image
        #patch_norm = patch_discrepancy # check and comment
        try:
            #patch_score = float(patch_discrepancy.mean().cpu().item())
            patch_score = robust_patch_score(patch_discrepancy)
        except Exception:
            #patch_score = float(patch_discrepancy.mean().item())
            patch_score = robust_patch_score(patch_discrepancy)
        patch_scores.append(patch_score)
        heatmaps.append(fused_map_np)

    final_hm = heatmaps[0] if len(heatmaps) == 1 else np.stack(heatmaps, axis=0)
    #final_hm = gaussian_filter(final_hm, sigma=4).astype(np.float32)

    # prepare patch score output
    if len(patch_scores) == 0:
        patch_scores_out = None
    elif len(patch_scores) == 1:
        patch_scores_out = patch_scores[0]
    else:
        patch_scores_out = np.stack(patch_scores, axis=0)

    # concat scores output
    if len(concat_scores) == 0:
        concat_scores_out = None
    elif len(concat_scores) == 1:
        concat_scores_out = concat_scores[0]
    else:
        concat_scores_out = np.stack(concat_scores, axis=0)

    # plotting: only plot when the original condition (frame is labeled anomalous) holds
    if img is not None and mean_for_display is not None and std_for_display is not None:
        try:
            should_plot = False
            if getattr(config, 'display_frames', False) and (frame_label_array is not None and frame_idx is not None):
                if frame_idx < len(frame_label_array) and int(frame_label_array[frame_idx]) == 1:
                    should_plot = True
            if should_plot:
                img_cpu = img[0].cpu()
                mean_t = torch.tensor(mean_for_display).view(3,1,1).cpu()
                std_t  = torch.tensor(std_for_display).view(3,1,1).cpu()
                img_disp = img_cpu * std_t + mean_t
                img_disp = img_disp.clamp(0,1).permute(1,2,0).numpy()
                import matplotlib.pyplot as plt
                plt.figure(figsize=figsize)
                plt.subplot(1,2,1); plt.imshow(img_disp); plt.axis('off'); plt.title('Input')
                plt.subplot(1,2,2); plt.imshow(img_disp); plt.imshow(final_hm, cmap='jet', alpha=alpha); plt.axis('off'); plt.title('Anomaly Map (fused)')
                plt.tight_layout(); plt.show()
        except Exception as e:
            print(f"[WARN] plotting failed: {e}")

    return final_hm, patch_scores_out, concat_scores_out


# -------------------- Zero-shot evaluation: build concatenated memory + KNN index --------------------
def zero_shot_multilayer_eval(teacher, config, mean, std, device, memory_bank=None):
    """
    Build per-image concatenated RAW CLS latents (N, D_concat), build KNN index once,
    then evaluate test frames. memory_bank (if provided) should be tuple:
      (per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, train_image_paths)
    """
    train_folder = os.path.join(config.data_path, 'train/frames')
    train_noaug_dataset = CustomFrameDataset(train_folder, transforms.Compose([
                                                                       transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]),
                                     resize_height=config.image_size, resize_width=config.image_size, time_step=config.num_frames)
    #train_noaug_dataset.index_samples = train_noaug_dataset.index_samples[:32]
    train_noaug_loader = torch.utils.data.DataLoader(train_noaug_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)

    config.concat_fps_target = int(config.concat_fps_target * len(train_noaug_dataset))
    print('Memory Size after FPS', config.concat_fps_target)

    if memory_bank is None:
        per_image_concat_cpu, per_layer_raw, per_image_last_patches_cpu, train_image_paths, pca_meta = collect_concatenated_memory_bank(teacher, train_noaug_loader, device, config, train_folder=train_folder)
    else:
        # support multiple memory_bank formats for backward compatibility
        if len(memory_bank) == 5:
            per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, train_image_paths, pca_meta = memory_bank
        elif len(memory_bank) == 4:
            per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, train_image_paths = memory_bank
            pca_meta = None
        elif len(memory_bank) == 3:
            per_layer_raw, per_image_concat_cpu, train_image_paths = memory_bank
            per_image_last_patches_cpu = None
            pca_meta = None
        else:
            raise ValueError('memory_bank must be length 3, 4, or 5')

    concat_index_obj = build_knn_index(per_image_concat_cpu, metric=config.knn_metric, use_faiss=config.use_faiss_index)

    # test loop unchanged except use raw cls lists for scoring
    test_path = os.path.join(config.data_path, 'test/')
    path_scenes = sorted(glob.glob(os.path.join(test_path, 'frames/*')))
    label_path = os.path.join(test_path, 'test_frame_mask/')

    all_frame_scores = []
    all_patch_scores = []
    all_concat_scores = []
    all_labels = []
    teacher.eval()
    with torch.no_grad():
        for idx_video, path_scene in enumerate(path_scenes):
            print(f'Zero-shot Multi-layer eval Video {idx_video+1}/{len(path_scenes)}: {os.path.basename(path_scene)}')
            test_dataset = CustomFrameDataset(path_scene, transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]),
                                      resize_height=config.image_size, resize_width=config.image_size, time_step=config.num_frames)
            #test_dataset.index_samples = test_dataset.index_samples[:32]
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers, pin_memory=True, drop_last=False)
            np_label = np.load(os.path.join(label_path, f'{os.path.basename(path_scene)}.npy'), allow_pickle=True)
            video_frame_scores = []
            video_heatmaps = []
            video_patch_scores = []
            video_concat_scores = []
            for frame_idx, batch in tqdm(enumerate(test_loader), desc=f'Zero-shot {os.path.basename(path_scene)}', total=len(test_loader)):
                img = batch['standard'].to(device)
                img = img[:, 0].float()
                # teacher forward: returns RAW CLS list in first return now
                cls_raw_list, patch_tokens_list = teacher(img)

                # frame-level score: compute distance between this frame's raw cls per layer and compressed_by_layer
                layer_scores = []
                for li, cls_raw in enumerate(cls_raw_list):
                    mem = per_layer_raw[li]  # (N_images, C)
                    try:
                        score = score_cls_vector_vs_prototypes(cls_raw.squeeze(0).cpu(), mem.cpu(), metric=config.scoring)
                    except Exception:
                        score = 0.0
                    layer_scores.append(score)
                final_score = aggregate_layer_scores(layer_scores) if len(layer_scores) > 0 else 0.0
                video_frame_scores.append(final_score)

                # compute heatmap / patch score (plotting controlled inside the function)
                hm, patch_score, concat_score = cls_patch_similarity_heatmaps_concatenated(
                    cls_latents_list=cls_raw_list,  # contains RAW CLS now
                    patch_tokens_list=patch_tokens_list,
                    per_image_concat_cpu=per_image_concat_cpu,
                    concat_index_obj=concat_index_obj,
                    config=config,
                    teacher=teacher,
                    train_image_paths=train_image_paths,
                    per_image_last_patches_cpu=per_image_last_patches_cpu,
                    mean=mean, std=std,
                    knn_k=config.knn_k,
                    knn_metric=config.knn_metric,
                    pca_meta = pca_meta,
                    img=img, mean_for_display=mean, std_for_display=std,
                    device=device,
                    frame_idx=frame_idx,
                    frame_label_array=np_label)
                #video_heatmaps.append(hm)
                #patch_score = concat_score = 0
                video_patch_scores.append(patch_score)
                video_concat_scores.append(concat_score)
            
            all_frame_scores.extend(video_frame_scores)
            all_patch_scores.extend(video_patch_scores)
            all_concat_scores.extend(video_concat_scores)
            all_labels.append(np_label[-len(video_frame_scores):])

    def _norm(arr):
        a = np.array(arr, dtype=np.float32)
        mn, mx = a.min(), a.max()
        return (a - mn) / (mx - mn + 1e-8) if mx > mn else a
    all_patch_scores = _norm(all_patch_scores)
    all_frame_scores = _norm(all_frame_scores)
    all_concat_scores = _norm(all_concat_scores)

    all_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([])
    if len(all_labels) > 0:
        cls_proto_auc = roc_auc_score(y_true=all_labels, y_score=all_frame_scores)
        frame_eer,  frame_thr  = compute_eer(all_labels, all_frame_scores)

        print(f'\n  {"Scorer":<28} {"AUROC":>7}   {"EER":>7}   {"EER-Threshold":>14}')
        print(f'  {"-"*60}')
        print(f'  {"CLS proto":<28} {cls_proto_auc:>7.4f}   {frame_eer:>7.4f}   {frame_thr:>14.6f}')
    else:
        cls_proto_auc = None
        print('No labels collected - skipping AUC calculation.')

    # additional reporting: print both AUCs if labels available
    try:
        if 'all_labels' in locals() and len(all_labels) > 0:
            concat_auc = roc_auc_score(y_true=all_labels, y_score=all_concat_scores) if len(all_concat_scores) > 0 else None
            concat_eer,  concat_thr  = compute_eer(all_labels, all_concat_scores)
            patch_auc = roc_auc_score(y_true=all_labels, y_score=all_patch_scores) if len(all_patch_scores) > 0 else None
            patch_eer,  patch_thr  = compute_eer(all_labels, all_patch_scores)
            if concat_auc is not None:
                print(f'Final report - CONCAT AUC: {concat_auc:.4f} {concat_eer:>7.4f}')
            else:
                print('Final report - CONCAT AUC: None')
            if patch_auc is not None:
                print(f'Final report - PATCH-KNN AUC [REPORTED, Eq. 7]: {patch_auc:.4f} {patch_eer:>7.4f}')
            else:
                print('Final report - PATCH AUC: None')
    except Exception as e:
        print(f'[WARN] final AUC reporting failed: {e}')
    return per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, train_image_paths, pca_meta


# -------------------- Main --------------------
def main():
    set_seed(42)
    config = Config()
    print_config(config)
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    teacher = TeacherMultiLayer(selected_layers=config.selected_layers)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher = teacher.to(device)

    if config.train:
        print('===== ZERO-SHOT MULTI-LAYER MODE (Concatenated KNN) =====')
        train_folder = os.path.join(config.data_path, 'train/frames')
        mean, std = compute_mean_and_std(train_folder, config.image_size, config.image_size, device=device, batch_size=config.batch_size)#[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]#
        
        print('mean, std:', mean, std)
        per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, train_image_paths, pca_meta = zero_shot_multilayer_eval(teacher, config, mean, std, device, memory_bank=None)

        os.makedirs(os.path.join(config.save_path, 'checkpoints'), exist_ok=True)
        #torch.save({'mean_std': {'mean': mean, 'std': std}, 'memory_bank_concat': per_image_concat_cpu, 'per_layer_raw': per_layer_raw,
            #'per_image_last_patches': per_image_last_patches_cpu, 'train_image_paths': train_image_paths, 'pca_meta': pca_meta 
                    #if pca_meta is not None else None}, os.path.join(config.save_path, 'checkpoints', 'best_multilayer_concat_knn.pth'))

    else:
        print('Testing ...')
        ck = torch.load(os.path.join(config.save_path, 'checkpoints', 'best_multilayer_concat_knn.pth'), map_location='cpu')
        per_image_concat_cpu = ck['memory_bank_concat']
        per_image_last_patches_cpu = ck.get('per_image_last_patches', None)
        per_layer_raw = ck.get('per_layer_raw', None)
        mean = ck['mean_std']['mean'].cpu()
        std = ck['mean_std']['std'].cpu()
        zero_shot_multilayer_eval(teacher, config, mean, std, device, memory_bank=(per_layer_raw, per_image_concat_cpu, per_image_last_patches_cpu, ck.get('train_image_paths', None), ck.get('pca_meta', None)))

    return 0


if __name__ == '__main__':
    main()
