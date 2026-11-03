import json
import os
import re
import shutil
import glob
import base64
from io import BytesIO

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# =====================================================================
# SHARED HELPERS
# =====================================================================
def resolve_frame_target(frame_paths, raw, prompts=None):
    """
    Map a free-text target typed by the annotator (or passed to
    generate_single_mask) to a 0-based index into `frame_paths`.

    Kept as one module-level function rather than duplicated inside each
    collector so the Colab and Jupyter widgets can never drift apart on
    what a given input means.

    Resolution order (first match wins):
        1. keywords: first | last | todo (first frame with no saved prompt)
        2. integer:  1-based position; negatives count back from the end
        3. exact case-insensitive basename match
        4. unique case-insensitive substring match on the full path

    Raises
    ------
    ValueError with a human-readable message on anything unresolvable or
    ambiguous. Callers surface that message rather than guessing: for a
    tool that produces ground truth, silently landing on the wrong frame
    is worse than refusing to move.
    """
    total = len(frame_paths)
    if total == 0:
        raise ValueError("No frames loaded.")
    if raw is None:
        raise ValueError("No target given.")
    s = str(raw).strip()
    if not s:
        raise ValueError("No target given.")

    key = s.lower()
    if key in ("first", "start", "begin"):
        return 0
    if key in ("last", "end"):
        return total - 1
    if key in ("todo", "next-todo", "unannotated", "resume"):
        if prompts is None:
            raise ValueError("'todo' is unavailable here (no prompt state).")
        for i, fp in enumerate(frame_paths):
            if fp not in prompts:
                return i
        raise ValueError("No unannotated frames left.")

    # Integer position. 1-based to match the "[i/N]" header the annotator
    # actually sees; -1 addresses the last frame, -2 the second-last, etc.
    if re.fullmatch(r"[+-]?\d+", s):
        n = int(s)
        if n == 0:
            raise ValueError(f"Frames are numbered 1-{total}; 0 is not a frame.")
        idx = (total + n) if n < 0 else (n - 1)
        if not 0 <= idx < total:
            raise ValueError(f"Frame {n} is out of range (1-{total}).")
        return idx

    # Exact basename match takes priority over substring matching, so a
    # filename that happens to be a prefix of longer ones still resolves.
    exact = [i for i, fp in enumerate(frame_paths)
             if os.path.basename(fp).lower() == key]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"'{s}' matches {len(exact)} frames with identical basenames; "
            "type more of the path."
        )

    matches = [i for i, fp in enumerate(frame_paths) if key in fp.lower()]
    if not matches:
        raise ValueError(f"No frame matches '{s}'.")
    if len(matches) > 1:
        preview = ", ".join(os.path.basename(frame_paths[i]) for i in matches[:3])
        more = ", ..." if len(matches) > 3 else ""
        raise ValueError(
            f"'{s}' matches {len(matches)} frames ({preview}{more}). "
            "Be more specific."
        )
    return matches[0]


# =====================================================================
# STAGE 1 — Extract anomalous frames
# =====================================================================
def extract_anomalous_frames(frame_dir, label_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    labels = np.load(label_file)

    frames = glob.glob(os.path.join(frame_dir, '**', '*.jpg'), recursive=True)
    frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[-1]))

    assert len(frames) == len(labels), "Mismatch between frames and labels"

    for f, l in zip(frames, labels):
        if l == 1:
            shutil.copy(f, os.path.join(output_dir, os.path.basename(f)))

    print(f"Done. Extracted {int(np.sum(labels == 1))} anomalous frames.")


# =====================================================================
# STAGE 2 — SAM2 model + mask generation
# =====================================================================
def load_sam2_predictor(
    config_file="sam2_hiera_l.yaml",
    checkpoint="/content/sam2/checkpoints/sam2_hiera_large.pt",
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not available in this Colab runtime — "
                  "falling back to CPU. Go to Runtime > Change runtime type "
                  "> GPU for much faster inference.")
    # NOTE: build_sam2's first argument is the Hydra CONFIG NAME, not the
    # checkpoint path — the checkpoint path is the second argument.
    model = build_sam2(config_file, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


# Loading SAM2 Hiera-Large takes tens of seconds and ~2.4 GB. Single-image
# work is inherently iterative (tweak a prompt, re-run, look), so cache the
# predictor per (config, checkpoint, device) instead of paying that cost on
# every call.
_PREDICTOR_CACHE = {}


def get_sam2_predictor(
    config_file="sam2_hiera_l.yaml",
    checkpoint="/content/sam2/checkpoints/sam2_hiera_large.pt",
    device=None,
    reload=False,
):
    """Cached wrapper around load_sam2_predictor()."""
    key = (config_file, checkpoint, device)
    if reload or key not in _PREDICTOR_CACHE:
        _PREDICTOR_CACHE[key] = load_sam2_predictor(config_file, checkpoint, device)
    return _PREDICTOR_CACHE[key]


def generate_sam2_mask(predictor, image_path, prompt):
    """
    Parameters
    ----------
    predictor  : SAM2ImagePredictor — already loaded
    image_path : str — path to a frame
    prompt     : dict — one entry from prompts.json, e.g.:
                 {"prompt_type": "point",
                  "points": [[320, 215]], "labels": [1]}
                 {"prompt_type": "box",
                  "boxes": [[180, 140, 460, 310], [500, 120, 620, 260]]}
                 {"prompt_type": "point+box",
                  "points": [[320, 215]], "labels": [1],
                  "boxes": [[180, 140, 460, 310]]}

    Returns
    -------
    best_mask : np.ndarray, shape (H, W), dtype bool
                When multiple boxes are supplied, the per-box masks are
                combined with a logical OR into a single anomaly mask.
    """
    image = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(image)

    prompt_type = prompt["prompt_type"]

    if prompt_type == "point":
        masks, scores, _ = predictor.predict(
            point_coords=np.array(prompt["points"]),
            point_labels=np.array(prompt["labels"]),
            box=None,
            multimask_output=True
        )
        return masks[scores.argmax()]

    # Accept both "boxes" (list of boxes) and "box" (a single box) keys.
    boxes = prompt.get("boxes", prompt.get("box"))
    if not boxes:
        raise ValueError(f"No boxes found in prompt for frame: {image_path}")
    # Normalize a single flat box [x0, y0, x1, y1] into a one-item list.
    if not isinstance(boxes[0], (list, tuple)):
        boxes = [boxes]
    box_array = np.array(boxes)

    if prompt_type == "box":
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_array,
            multimask_output=False
        )
        masks = np.squeeze(masks)
        if masks.ndim == 3:
            # One mask per box — union them into a single anomaly mask.
            return np.any(masks, axis=0)
        return masks.astype(bool)

    elif prompt_type == "point+box":
        # Each box is a separate object. SAM2 needs one predict() call per
        # object (a shared (M,2) point array can't be batched against an
        # (N,4) box array unless M == N and each point pairs 1:1 with a
        # box), so loop over boxes individually and union the results.
        points = np.array(prompt.get("points", []))
        labels = np.array(prompt.get("labels", []))
        n_boxes = box_array.shape[0]
        one_point_per_box = len(points) == n_boxes

        object_masks = []
        for i in range(n_boxes):
            if one_point_per_box:
                m, s, _ = predictor.predict(
                    point_coords=points[i:i + 1],
                    point_labels=labels[i:i + 1],
                    box=box_array[i],
                    multimask_output=False,
                )
            else:
                # Points don't line up 1:1 with boxes (e.g. free-form
                # click refinements) — fall back to box-only per object.
                m, s, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box_array[i],
                    multimask_output=False,
                )
            object_masks.append(np.squeeze(m))

        masks = np.stack(object_masks, axis=0)
        if masks.ndim == 3:
            return np.any(masks, axis=0)
        return masks.astype(bool)

    else:
        raise ValueError(
            f"Unknown prompt_type '{prompt_type}' for frame: {image_path}. "
            f"Expected 'point', 'box', or 'point+box'."
        )


def generate_single_mask(
    image_path=None,
    prompt=None,
    prompts_json=None,
    frame=None,
    frame_paths=None,
    frame_dir="",
    output_path=None,
    output_dir=None,
    predictor=None,
    show=False,
    verbose=True,
    figsize=(14, 4.5),
):
    """
    Generate and save ONE binary PNG mask, with the same file format and
    conventions as batch_generate_masks — dtype uint8, values {0, 255}, 0 =
    normal, 255 = anomaly, single channel.

    This exists so a prompt can be tested on one frame in seconds instead of
    committing to a full batch: it is the natural inner loop for checking
    whether a click landed on the right object, whether a box is tight
    enough, or why one frame came out empty.

    Choosing the frame and prompt — supply exactly one of:
      (a) image_path + prompt        : an explicit prompt dict, e.g.
                                       {"prompt_type": "point",
                                        "points": [[320, 215]], "labels": [1]}
                                       Nothing is read from disk except the
                                       image, so this works for prompts you
                                       are composing by hand.
      (b) prompts_json + image_path  : look the frame's saved prompt up by
                                       its key (or basename) in prompts.json.
      (c) prompts_json + frame       : look it up by frame number, filename,
                                       substring, or first/last/todo — the
                                       same targets the annotator's Jump box
                                       accepts.

    A note on frame NUMBERS, which are the one genuine trap here: the Jump
    box numbers frames over the full `frame_paths` list, while prompts.json
    holds only the frames that were actually annotated. "87" therefore means
    two different frames in the two places once anything has been skipped.
    Pass `frame_paths` (the same list handed to the collector) to number
    against the full sequence and keep the two consistent; without it,
    numeric targets index the prompts.json keys and a warning says so.
    Filename and substring targets are unaffected and are the safer habit.

    Other parameters
    ----------------
    frame_paths : list[str] or None — the full frame list, used only to give
                         numeric `frame` targets the same meaning they have
                         in the annotator.
    frame_dir   : str  — prefix joined to the prompts.json key to reach the
                         image, exactly as in batch_generate_masks. Use ""
                         when the keys are already usable paths.
    output_path : str  — explicit destination for the PNG. Wins over
                         output_dir.
    output_dir  : str  — destination root; the mask is written to
                         <output_dir>/<key without extension>_mask.png, i.e.
                         the same tree layout batch_generate_masks produces,
                         so single-frame fixes can be written straight back
                         into an existing mask directory.
                         If neither is given, the PNG lands next to the
                         source image as <stem>_mask.png.
    predictor   : SAM2ImagePredictor or None — reuses the cached predictor
                         when None.
    show        : bool — draw frame+prompt / binary mask / overlay side by
                         side for immediate visual QA.
    verbose     : bool — print a short sanity report (size agreement with
                         the source frame, value set, anomaly coverage).

    Returns
    -------
    dict with keys:
        "mask"        : np.ndarray (H, W) bool
        "mask_path"   : str — where the PNG was written
        "image_path"  : str
        "frame_key"   : str or None — prompts.json key, when one was used
        "coverage"    : float — fraction of pixels marked anomalous
        "prompt"      : dict — the prompt actually used
    """
    # ----- resolve the frame key, image path and prompt -----------------
    frame_key = None
    prompts = None

    if prompts_json is not None:
        with open(prompts_json) as f:
            prompts = json.load(f)
        keys = list(prompts.keys())
        if not keys:
            raise ValueError(f"{prompts_json} contains no prompts.")

        if frame is not None:
            # Reuse the annotator's target grammar.
            if frame_paths:
                # Number against the full sequence, exactly as the Jump box
                # does, then require that frame to actually have a prompt.
                frame_key = frame_paths[
                    resolve_frame_target(frame_paths, frame, prompts)
                ]
                if frame_key not in prompts:
                    raise ValueError(
                        f"Frame '{frame}' resolves to {frame_key}, which has "
                        f"no prompt in {prompts_json}. Annotate it first, or "
                        f"pass an explicit `prompt`."
                    )
            else:
                if re.fullmatch(r"[+-]?\d+", str(frame).strip()) and verbose:
                    print(
                        f"NOTE: '{frame}' is being numbered against the "
                        f"{len(keys)} annotated frames in {prompts_json}, not "
                        f"the full frame list — pass frame_paths=... to match "
                        f"the annotator's numbering."
                    )
                frame_key = keys[resolve_frame_target(keys, frame, prompts)]
        elif image_path is not None:
            if image_path in prompts:
                frame_key = image_path
            else:
                frame_key = keys[resolve_frame_target(keys, image_path, prompts)]
        else:
            raise ValueError(
                "With prompts_json, pass either `frame` (number, filename, "
                "substring, first/last/todo) or `image_path` to select which "
                "frame to process."
            )

        if prompt is None:
            prompt = prompts[frame_key]
        image_path = os.path.join(frame_dir, frame_key)

    if image_path is None:
        raise ValueError("Provide image_path, or prompts_json plus frame.")
    if prompt is None:
        raise ValueError(
            "No prompt supplied. Pass `prompt` directly, or `prompts_json` "
            "so the saved prompt can be looked up."
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Frame not found: {image_path}")

    # ----- resolve the output path --------------------------------------
    if output_path is None:
        if output_dir is not None:
            stem = os.path.splitext(frame_key if frame_key else
                                    os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, stem + "_mask.png")
        else:
            stem = os.path.splitext(image_path)[0]
            output_path = stem + "_mask.png"
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    # ----- inference ------------------------------------------------------
    if predictor is None:
        predictor = get_sam2_predictor()

    mask = generate_sam2_mask(predictor, image_path, prompt)
    mask = np.asarray(mask).astype(bool)

    # Save in exactly the format validate_masks() expects: single channel,
    # uint8, {0, 255}. Mode "L" is forced so a stray palette or RGB write
    # can't slip through and break downstream IoU code.
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(output_path)

    coverage = float(mask.mean())

    # ----- sanity report --------------------------------------------------
    if verbose:
        img_h, img_w = np.array(Image.open(image_path).convert("RGB")).shape[:2]
        print(f"Frame      : {image_path}")
        print(f"Prompt     : {prompt.get('prompt_type')} "
              f"({len(prompt.get('points', []) or [])} pt, "
              f"{len(prompt.get('boxes', []) or [])} box)")
        print(f"Mask       : {output_path}")
        print(f"Size       : {mask.shape[1]}x{mask.shape[0]} "
              f"(frame {img_w}x{img_h})"
              + ("" if (mask.shape[1], mask.shape[0]) == (img_w, img_h)
                 else "   <-- MISMATCH: mask and frame differ in size"))
        print(f"Values     : {sorted(set(np.unique((mask * 255).astype(np.uint8)).tolist()))}")
        print(f"Anomaly px : {int(mask.sum())} ({coverage * 100:.2f}% of frame)")
        if not mask.any():
            print("WARNING: empty mask — SAM2 returned nothing for this "
                  "prompt. Usually the point sits on background, or the box "
                  "encloses several objects at once.")
        elif coverage > 0.5:
            print("WARNING: over half the frame is marked anomalous — check "
                  "for an inverted or runaway segmentation before using this "
                  "as ground truth.")

    # ----- optional visual QA --------------------------------------------
    if show:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        image = np.array(Image.open(image_path).convert("RGB"))
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(image)
        axes[0].set_title(f"Prompt — {os.path.basename(image_path)}", fontsize=9)
        _draw_prompt_overlay(axes[0], prompt, patches)
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Binary mask (saved PNG)", fontsize=9)
        axes[1].axis("off")

        axes[2].imshow(image)
        axes[2].imshow(np.ma.masked_where(~mask, mask.astype(np.uint8) * 255),
                       cmap="autumn", alpha=0.5, vmin=0, vmax=255)
        axes[2].set_title(f"Overlay — {coverage * 100:.2f}% anomalous", fontsize=9)
        axes[2].axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "mask": mask,
        "mask_path": output_path,
        "image_path": image_path,
        "frame_key": frame_key,
        "coverage": coverage,
        "prompt": prompt,
    }


def batch_generate_masks(frame_dir, prompts_json, output_dir):
    """
    Output
    ------
    Binary PNG masks saved as:
        output_dir/scene/video/frame_NNNNN_mask.png
        dtype uint8, values: 0 (normal) or 255 (anomaly)
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(prompts_json) as f:
        prompts = json.load(f)

    predictor = get_sam2_predictor()
    if not torch.cuda.is_available():
        print(f"Processing {len(prompts)} frames on CPU — this can take a "
              f"long time (often minutes per frame for SAM2 Hiera-Large). "
              f"The progress bar below updates per frame; if it isn't "
              f"moving at all after a couple of minutes, something is "
              f"actually stuck. If it IS moving, just slowly, switch to a "
              f"GPU runtime (Runtime > Change runtime type > GPU) and "
              f"rerun for a large speedup.")

    counts = {"point": 0, "box": 0, "point+box": 0, "failed": 0}

    try:
        from tqdm.auto import tqdm
        items = tqdm(prompts.items(), total=len(prompts), desc="Generating masks")
    except ImportError:
        items = prompts.items()

    for relative_path, prompt in items:
        image_path = os.path.join(frame_dir, relative_path)

        if not os.path.exists(image_path):
            print(f"WARNING: Frame not found, skipping — {image_path}")
            counts["failed"] += 1
            continue

        base_name = os.path.splitext(relative_path)[0]
        mask_relpath = base_name + "_mask.png"
        mask_path = os.path.join(output_dir, mask_relpath)
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)

        try:
            best_mask = np.asarray(
                generate_sam2_mask(predictor, image_path, prompt)
            ).astype(bool)
            mask_img = Image.fromarray((best_mask * 255).astype(np.uint8), mode="L")
            mask_img.save(mask_path)
            counts[prompt["prompt_type"]] += 1
        except Exception as e:
            print(f"ERROR on {relative_path}: {e}")
            counts["failed"] += 1
            continue

    print("\nBatch mask generation complete.")
    print(f"  Point prompts processed  : {counts['point']}")
    print(f"  Box prompts processed    : {counts['box']}")
    print(f"  Point+Box processed      : {counts['point+box']}")
    print(f"  Failed / skipped         : {counts['failed']}")
    print(f"  Output root              : {output_dir}")


def validate_masks(mask_root, expected_size=(640, 640)):
    errors = []
    for root, _, files in os.walk(mask_root):
        for f in files:
            if not f.endswith("_mask.png"):
                continue
            path = os.path.join(root, f)
            mask = np.array(Image.open(path))

            if mask.shape != expected_size:
                errors.append(f"SIZE ERROR: {path} → {mask.shape}")

            unique_vals = np.unique(mask)
            if not set(unique_vals).issubset({0, 255}):
                errors.append(f"VALUE ERROR: {path} → {unique_vals}")

            if mask.max() == 0:
                errors.append(f"EMPTY MASK WARNING: {path}")

    if not errors:
        print("All masks valid.")
    else:
        for e in errors:
            print(e)


# =====================================================================
# STAGE 4 — Visual review: annotation vs. generated mask, side by side
# =====================================================================
def review_masks(frame_dir, prompts_json, mask_dir, frame_keys=None,
                  max_frames=10, figsize=(10, 5),
                  output_dir=None, show=True, dpi=150):
    """
    Display (and/or save) each frame's collected prompt (from
    PromptCollectorColab / PromptCollectorJupyter, as stored in
    prompts.json) next to its SAM2 generated mask (from
    batch_generate_masks), side by side, for quick visual QA.

    Left panel  : original frame with the annotation overlaid —
                  green boxes, green stars for positive points,
                  red stars for negative points.
    Right panel : original frame with the generated binary mask overlaid
                  in translucent red.

    Parameters
    ----------
    frame_dir    : str  — same value passed to batch_generate_masks
                          (use "" if prompts.json keys are already full
                          paths — see Stage 3).
    prompts_json : str  — path to prompts.json.
    mask_dir     : str  — output_dir used in batch_generate_masks
                          (e.g. "drone_anomaly_gt_masks/").
    frame_keys   : list[str] or None — specific prompts.json keys to
                          review. Defaults to all keys, in order.
    max_frames   : int or None — cap on how many frames to display
                          (reviewing all 300+ inline can be slow/huge).
                          Set to None to display every frame.
    figsize      : tuple — matplotlib figure size per frame pair.
    output_dir   : str or None — if given, each reviewed frame's
                          side-by-side figure is saved here as
                          "<basename>_review.png", mirroring any
                          subdirectory structure in the frame's relative
                          path (so keys from different clips don't
                          collide). Directory is created if missing.
    show         : bool — whether to also display the figure inline
                          (e.g. in Colab/Jupyter). Set False to save
                          only, which is faster for large batches and
                          avoids flooding notebook output.
    dpi          : int  — resolution used when saving.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    with open(prompts_json) as f:
        prompts = json.load(f)

    keys = frame_keys if frame_keys is not None else list(prompts.keys())
    if max_frames is not None:
        keys = keys[:max_frames]

    if not keys:
        print("No frames to review.")
        return

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    shown = 0
    saved = 0
    for relative_path in keys:
        prompt = prompts.get(relative_path)
        if prompt is None:
            print(f"WARNING: '{relative_path}' not found in {prompts_json}, skipping.")
            continue

        image_path = os.path.join(frame_dir, relative_path)
        base_name = os.path.splitext(relative_path)[0]
        mask_path = os.path.join(mask_dir, base_name + "_mask.png")

        if not os.path.exists(image_path):
            print(f"WARNING: Frame not found, skipping — {image_path}")
            continue
        if not os.path.exists(mask_path):
            print(f"WARNING: Mask not found (run batch_generate_masks first?), "
                  f"skipping — {mask_path}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # --- Left: annotation overlay ---
        axes[0].imshow(image)
        axes[0].set_title(f"Annotation — {relative_path}", fontsize=9)
        _draw_prompt_overlay(axes[0], prompt, patches)
        axes[0].axis("off")

        # --- Right: generated mask overlay ---
        axes[1].imshow(image)
        axes[1].imshow(np.ma.masked_where(mask == 0, mask),
                        cmap="autumn", alpha=0.5, vmin=0, vmax=255)
        axes[1].set_title("Generated mask", fontsize=9)
        axes[1].axis("off")

        plt.tight_layout()

        if output_dir is not None:
            # Mirror the relative path's subdirectories under output_dir so
            # frames with the same basename from different clips (e.g.
            # ".../s_s1_te_01/0.jpg" vs ".../s_s1_te_02/0.jpg") don't
            # overwrite each other.
            save_path = os.path.join(output_dir, base_name + "_review.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            saved += 1

        if show:
            plt.show()
        else:
            plt.close(fig)
        shown += 1

    msg = f"\nReviewed {shown}/{len(keys)} frame(s)."
    if output_dir is not None:
        msg += f" Saved {saved} figure(s) to '{output_dir}'."
    print(msg)


def _draw_prompt_overlay(ax, prompt, patches):
    """Draw a prompt's boxes/points onto a matplotlib axis."""
    boxes = prompt.get("boxes", prompt.get("box"))
    if boxes:
        if not isinstance(boxes[0], (list, tuple)):
            boxes = [boxes]
        for box in boxes:
            x0, y0, x1, y1 = box
            rect = patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="lime", facecolor="none"
            )
            ax.add_patch(rect)

    points = prompt.get("points")
    if points:
        points = np.array(points)
        labels = np.array(prompt.get("labels", [1] * len(points)))
        pos = points[labels == 1]
        neg = points[labels == 0]
        if len(pos):
            ax.scatter(pos[:, 0], pos[:, 1], c="lime", marker="*", s=150,
                       edgecolors="black", linewidths=0.8)
        if len(neg):
            ax.scatter(neg[:, 0], neg[:, 1], c="red", marker="*", s=150,
                       edgecolors="black", linewidths=0.8)


# =====================================================================
# STAGE 3 — Browser-based prompt collector for Colab
# =====================================================================
class PromptCollectorColab:
    """
    Colab-native prompt collector. Renders one frame at a time as an HTML5
    canvas. Mouse/keyboard interactions are captured in JavaScript and
    relayed to this Python object via `google.colab.output.register_callback`.
    Every "next" click writes an updated prompts.json immediately.

    Controls:
        Left click              -> add foreground point (label=1, green +)
        Right click              -> add background point (label=0, red x)
        Left-button drag          -> add a bounding box (yellow rectangle);
                                      boxes are additive, not replaced
        'p' key / Previous btn -> go back one frame WITHOUT saving the
                                    current screen; pre-loads whatever was
                                    previously saved for that earlier frame
                                    (if anything) so you can review/edit it
        'n' key / Next btn  -> save current prompt & advance
        's' key / Skip btn  -> skip frame, no prompt saved
        'u' key / Undo btn  -> undo the last thing drawn, point or box,
                                    whichever came last; press repeatedly to
                                    peel the frame back gesture by gesture
        'j' key / Jump box  -> focus the "Jump to" field; type a target and
                                    press Enter (or click Go) to jump straight
                                    to that frame. Like Previous, jumping is
                                    navigation only: it never saves whatever
                                    is currently drawn on screen. Accepted
                                    targets:
                                      12          -> 1-based frame number
                                      -1          -> counts back from the end
                                      frame_87.jpg-> exact basename
                                      _87         -> unique filename substring
                                      first/last  -> ends of the sequence
                                      todo        -> first frame with no
                                                     saved prompt yet
                                    Bad or ambiguous targets leave you on the
                                    current frame with an explanatory message
                                    in the status bar, so nothing you have
                                    drawn is lost.

    prompts.json schema:
        {"<relative_frame_path>": {"prompt_type": "point"|"box"|"point+box",
                                     "points": [[x,y], ...],   # if applicable
                                     "labels": [1, 0, ...],    # if applicable
                                     "boxes": [[x0,y0,x1,y1], ...],  # if applicable
                                     "draw_order": ["point","box",...]}}
                                     # "point+box" only; see below

    "draw_order" records which kind of gesture came first, second, and so on
    (the arrays themselves stay in order within each kind, so the two
    together reconstruct the exact sequence). It is written only for
    "point+box" prompts, where the interleaving is otherwise unrecoverable,
    and exists so that undo still works correctly after you leave a frame
    and jump back to it. It is purely editor metadata: Stage 2 reads only
    prompt_type / points / labels / boxes, so files written by older runs, or
    by PromptCollectorJupyter, load fine — undo then falls back to assuming
    boxes were drawn before points, so points come off first.
    """

    DRAG_THRESHOLD_PX = 10  # minimum drag distance (px) to register as a box

    def __init__(self, frame_paths, output_json, resume=True):
        self.frame_paths = frame_paths
        self.output_json = output_json
        self.prompts = self._load_existing(output_json) if resume else {}
        self.current_idx = self._find_resume_index()
        self._callback_registered = False

    # ---------------------------------------------------------------
    # State / persistence helpers
    # ---------------------------------------------------------------
    def _load_existing(self, path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    def _find_resume_index(self):
        for i, fp in enumerate(self.frame_paths):
            if fp not in self.prompts:
                return i
        return len(self.frame_paths)

    def _save(self):
        with open(self.output_json, 'w') as f:
            json.dump(self.prompts, f, indent=2)

    # ---------------------------------------------------------------
    # Jump-target resolution
    # ---------------------------------------------------------------
    def _resolve_jump_target(self, raw):
        """Delegate to the shared resolver (see resolve_frame_target)."""
        return resolve_frame_target(self.frame_paths, raw, self.prompts)

    # ---------------------------------------------------------------
    # Image -> data URI (frames must be embedded; the browser sandbox
    # cannot read the Colab VM's filesystem directly)
    # ---------------------------------------------------------------
    def _image_to_data_uri(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", w, h

    # ---------------------------------------------------------------
    # JS -> Python callback: receives one recorded gesture per frame
    # ---------------------------------------------------------------
    def _on_record(self, payload):
        """
        payload keys (sent from JS):
            action      : "next" | "skip" | "prev" | "jump"
            prompt_type : "point" | "box" | "point+box" | None
            points      : [[x,y], ...]  (ints, image-pixel coordinates)
            labels      : [1|0, ...]
            boxes       : [[x0,y0,x1,y1], ...] (ints, image-pixel coordinates)
            order       : ["point"|"box", ...] — one token per gesture, in
                          the order they were drawn; drives undo
            target      : str, only for action == "jump" — frame number,
                          filename (or substring), or a keyword
        """
        try:
            action = payload.get("action")

            if action == "next":
                # Only this branch writes, so only this branch needs the key.
                # Reading it eagerly threw IndexError whenever current_idx sat
                # one past the end, turning a perfectly valid jump into an
                # opaque "list index out of range".
                frame_key = self.frame_paths[self.current_idx]
                ptype = payload.get("prompt_type")
                points = payload.get("points") or []
                labels = payload.get("labels") or []
                boxes = payload.get("boxes") or []
                order = self._validate_draw_order(
                    payload.get("order"), len(points), len(boxes)
                )

                if ptype == "point" and points:
                    self.prompts[frame_key] = {
                        "prompt_type": "point",
                        "points": points,
                        "labels": labels,
                    }
                elif ptype == "box" and boxes:
                    self.prompts[frame_key] = {
                        "prompt_type": "box",
                        "boxes": boxes,
                    }
                elif ptype == "point+box" and points and boxes:
                    entry = {
                        "prompt_type": "point+box",
                        "points": points,
                        "labels": labels,
                        "boxes": boxes,
                    }
                    # Only point+box needs the interleaving recorded — for a
                    # single-kind prompt the arrays already are the order.
                    if order is not None:
                        entry["draw_order"] = order
                    self.prompts[frame_key] = entry
                # "next" pressed with nothing drawn is tolerated, treated
                # like a skip.
                self._save()
                self.current_idx += 1

            elif action == "skip":
                self.current_idx += 1

            elif action == "prev":
                # Navigation only — does NOT save whatever is currently
                # drawn on screen, so it can't silently overwrite a saved
                # prompt with an in-progress edit.
                if self.current_idx > 0:
                    self.current_idx -= 1

            elif action == "jump":
                # Navigation only, same contract as "prev".
                try:
                    self.current_idx = self._resolve_jump_target(
                        payload.get("target")
                    )
                except ValueError as e:
                    # Deliberately do NOT re-render: staying put preserves
                    # any in-progress points/boxes on the canvas and lets the
                    # status bar explain what went wrong.
                    return {"done": False, "error": str(e)}

            if self.current_idx >= len(self.frame_paths):
                return {"done": True}

            self._render_current_frame()
            return {"done": False}

        except Exception as e:
            return {"done": False, "error": str(e)}

    # ---------------------------------------------------------------
    # Draw-order bookkeeping (drives undo)
    # ---------------------------------------------------------------
    @staticmethod
    def _validate_draw_order(order, n_points, n_boxes):
        """
        Accept a draw_order token list only if it exactly accounts for the
        points and boxes it accompanies; otherwise return None so the caller
        falls back to a synthesized order.

        The list arrives from the browser, and a stale or truncated one would
        make undo delete the wrong element — a silent corruption of ground
        truth. Cheap to check, so check.
        """
        if not isinstance(order, list):
            return None
        if len(order) != n_points + n_boxes:
            return None
        if any(tok not in ("point", "box") for tok in order):
            return None
        if order.count("point") != n_points or order.count("box") != n_boxes:
            return None
        return list(order)

    @classmethod
    def _draw_order_for(cls, existing):
        """
        Reconstruct the gesture order for a previously-saved prompt.

        Uses the stored draw_order when it is present and consistent.

        Falls back to assuming boxes were drawn before points for prompts
        written before this key existed (or by PromptCollectorJupyter). Two
        reasons for that direction: it is the usual SAM2 workflow — drag a
        box around the object, then refine with clicks — and it makes undo
        peel points off first on legacy files, which is exactly what undo did
        before this change. Guessing is unavoidable here; guessing the way
        that matches both habit and history is the least surprising option.
        """
        existing = existing or {}
        points = existing.get("points", []) or []
        boxes = existing.get("boxes", []) or []
        stored = cls._validate_draw_order(
            existing.get("draw_order"), len(points), len(boxes)
        )
        if stored is not None:
            return stored
        return ["box"] * len(boxes) + ["point"] * len(points)

    # ---------------------------------------------------------------
    # Rendering
    # ---------------------------------------------------------------
    def _render_current_frame(self):
        from IPython.display import HTML, display, clear_output

        fp = self.frame_paths[self.current_idx]
        data_uri, w, h = self._image_to_data_uri(fp)
        existing = self.prompts.get(fp, {})
        html = self._build_html(data_uri, w, h, fp, existing)

        clear_output(wait=True)
        display(HTML(html))

    def _build_html(self, data_uri, img_w, img_h, frame_path, existing=None):
        existing = existing or {}
        idx = self.current_idx
        total = len(self.frame_paths)
        fname = os.path.basename(frame_path)
        # Cap the on-screen canvas width; JS scales click coords back
        # to true image-pixel space so saved prompts stay accurate
        # regardless of display size.
        display_w = min(img_w, 720)
        scale = display_w / img_w
        display_h = int(img_h * scale)
        # The control row is wider now that it carries the jump field; give the
        # container a floor so the buttons don't wrap on small frames.
        panel_w = max(display_w + 40, 660)

        # Pre-populate client-side state from a previously-saved prompt so
        # that navigating with Previous (or resuming) shows what was
        # already recorded, ready to review or edit.
        init_points = json.dumps(existing.get("points", []))
        init_labels = json.dumps(existing.get("labels", []))
        init_boxes = json.dumps(existing.get("boxes", []))
        init_order = json.dumps(self._draw_order_for(existing))

        # With free jumping the annotator can land anywhere in the sequence,
        # so the header states explicitly whether this frame already has a
        # saved prompt and how much of the set is done.
        n_done = sum(1 for fp in self.frame_paths if fp in self.prompts)
        if existing:
            saved_note = (
                f'<span style="color:#0a7d00;">&#10003; saved: '
                f'{existing.get("prompt_type", "?")} '
                f'({len(existing.get("points", []))} pt, '
                f'{len(existing.get("boxes", []))} box)</span>'
            )
        else:
            saved_note = '<span style="color:#b26b00;">not yet annotated</span>'

        return f"""
<div id="annot-root" tabindex="0"
     style="outline:none; font-family:sans-serif; max-width:{panel_w}px;">
  <div style="margin-bottom:6px; font-size:13px; color:#333;">
    <b>[{idx+1}/{total}]</b> {fname} &nbsp;&mdash;&nbsp; {saved_note}
    &nbsp;|&nbsp; {n_done}/{total} annotated<br>
    LClick = FG point (green) &nbsp;|&nbsp; RClick = BG point (red)
    &nbsp;|&nbsp; Drag = add box (yellow) &nbsp;|&nbsp;
    <b>P</b>=previous &nbsp; <b>N</b>=next &nbsp; <b>S</b>=skip &nbsp;
    <b>U</b>=undo last drawn &nbsp; <b>J</b>=jump
  </div>
  <canvas id="annot-canvas" width="{img_w}" height="{img_h}"
          style="width:{display_w}px; height:{display_h}px;
                 border:1px solid #999; cursor:crosshair;"></canvas>
  <div style="margin-top:8px;">
    <button id="annot-prev-btn" {"disabled" if idx == 0 else ""}>&laquo; Previous (p)</button>
    <button id="annot-next-btn">Next (n)</button>
    <button id="annot-skip-btn">Skip (s)</button>
    <button id="annot-undo-btn">Undo (u)</button>
    <span style="margin-left:14px; font-size:12px; color:#333;">
      Jump to:
      <input id="annot-jump-input" type="text" size="12"
             placeholder="# or name"
             title="Frame number 1-{total} (negatives count from the end), a filename or unique substring, or one of: first, last, todo"
             style="font-size:12px; padding:2px 4px; width:110px;">
      <button id="annot-jump-btn">Go (j)</button>
    </span>
    <span id="annot-status" style="margin-left:12px; font-size:12px; color:#666;"></span>
  </div>
</div>

<script>
(function() {{
  const canvas   = document.getElementById('annot-canvas');
  const ctx      = canvas.getContext('2d');
  const root     = document.getElementById('annot-root');
  const status   = document.getElementById('annot-status');
  const jumpBox  = document.getElementById('annot-jump-input');
  const img      = new Image();
  img.src        = "{data_uri}";

  // Client-side annotation state for THIS frame only, pre-populated from
  // any prompt already saved for this frame (e.g. after pressing Previous).
  let points = {init_points};   // [[x,y], ...] in true image-pixel coords
  let labels = {init_labels};   // [1|0, ...]
  let boxes  = {init_boxes};    // [[x0,y0,x1,y1], ...] in true image-pixel coords
  // One token per gesture, in the order drawn: "point" or "box". This is
  // what makes undo chronological — points and boxes live in separate
  // arrays, so without it the two can only be undone kind-by-kind.
  let order  = {init_order};

  let dragging   = false;
  let dragStartX = 0, dragStartY = 0;
  let dragCurX   = 0, dragCurY   = 0;

  function redraw() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Points
    ctx.lineWidth = 2;
    for (let i = 0; i < points.length; i++) {{
      const [x, y] = points[i];
      const fg = labels[i] === 1;
      ctx.strokeStyle = fg ? '#00c800' : '#ff2020';
      ctx.beginPath();
      if (fg) {{
        ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y);
        ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6);
      }} else {{
        ctx.moveTo(x - 6, y - 6); ctx.lineTo(x + 6, y + 6);
        ctx.moveTo(x - 6, y + 6); ctx.lineTo(x + 6, y - 6);
      }}
      ctx.stroke();
    }}

    // Committed boxes (all of them)
    ctx.strokeStyle = 'yellow';
    ctx.lineWidth = 2;
    for (let i = 0; i < boxes.length; i++) {{
      const b = boxes[i];
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    }}

    // In-progress drag rectangle
    if (dragging) {{
      const x0 = Math.min(dragStartX, dragCurX);
      const y0 = Math.min(dragStartY, dragCurY);
      const w  = Math.abs(dragCurX - dragStartX);
      const h  = Math.abs(dragCurY - dragStartY);
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'yellow';
      ctx.strokeRect(x0, y0, w, h);
      ctx.setLineDash([]);
    }}

    status.textContent = points.length + " point(s), " + boxes.length + " box(es)";
  }}

  img.onload = redraw;

  // Map a mouse event to true image-pixel coordinates
  function toImageCoords(evt) {{
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width  / rect.width;
    const sy = canvas.height / rect.height;
    const x = Math.round((evt.clientX - rect.left) * sx);
    const y = Math.round((evt.clientY - rect.top)  * sy);
    return [x, y];
  }}

  canvas.addEventListener('contextmenu', function(evt) {{
    evt.preventDefault();
    const [x, y] = toImageCoords(evt);
    points.push([x, y]);
    labels.push(0);
    order.push('point');
    redraw();
  }});

  canvas.addEventListener('mousedown', function(evt) {{
    if (evt.button !== 0) return; // only left button starts a gesture
    const [x, y] = toImageCoords(evt);
    dragging = true;
    dragStartX = x; dragStartY = y;
    dragCurX = x; dragCurY = y;
  }});

  canvas.addEventListener('mousemove', function(evt) {{
    if (!dragging) return;
    const [x, y] = toImageCoords(evt);
    dragCurX = x; dragCurY = y;
    redraw();
  }});

  canvas.addEventListener('mouseup', function(evt) {{
    if (!dragging || evt.button !== 0) return;
    dragging = false;
    const dx = Math.abs(dragCurX - dragStartX);
    const dy = Math.abs(dragCurY - dragStartY);

    if (dx > {self.DRAG_THRESHOLD_PX} && dy > {self.DRAG_THRESHOLD_PX}) {{
      // Real drag -> commit as a new box (additive, does not replace)
      boxes.push([
        Math.min(dragStartX, dragCurX), Math.min(dragStartY, dragCurY),
        Math.max(dragStartX, dragCurX), Math.max(dragStartY, dragCurY)
      ]);
      order.push('box');
    }} else {{
      // Negligible movement -> treat as a foreground point click
      points.push([dragStartX, dragStartY]);
      labels.push(1);
      order.push('point');
    }}
    redraw();
  }});

  function currentPromptType() {{
    const hasPts = points.length > 0;
    const hasBoxes = boxes.length > 0;
    if (hasPts && hasBoxes) return "point+box";
    if (hasBoxes) return "box";
    if (hasPts) return "point";
    return null;
  }}

  function sendAction(action, extra) {{
    const payload = {{
      action: action,
      prompt_type: currentPromptType(),
      points: points,
      labels: labels,
      boxes: boxes,
      order: order
    }};
    if (extra) {{ Object.assign(payload, extra); }}
    status.textContent = (action === 'next') ? "saving..." : "working...";
    google.colab.kernel.invokeFunction(
      'notebook.recordPrompt', [payload], {{}}
    ).then(function(result) {{
      const data = result.data && result.data['application/json'];
      if (data && data.error) {{
        // Python rejected the request (e.g. an unresolvable jump target)
        // and left us on this frame — surface why and keep the drawing.
        status.textContent = "ERROR: " + data.error;
        if (action === 'jump') {{ jumpBox.focus(); jumpBox.select(); }}
      }}
      // On success the Python side re-renders the whole cell output
      // (new frame or "all done" message), so nothing further to do here.
    }});
  }}

  function doJump() {{
    const target = jumpBox.value.trim();
    if (!target) {{
      status.textContent = "Type a frame number, filename, or first/last/todo.";
      jumpBox.focus();
      return;
    }}
    sendAction('jump', {{target: target}});
  }}

  function undo() {{
    // Undo the most recent gesture, point or box, whichever it was.
    if (order.length === 0) {{
      // Belt and braces: if the order stack is somehow empty but shapes
      // remain (hand-edited prompts.json, say), fall back to removing
      // something rather than appearing to ignore the keypress.
      if (points.length > 0) {{ points.pop(); labels.pop(); }}
      else if (boxes.length > 0) {{ boxes.pop(); }}
      else {{ status.textContent = "Nothing to undo."; return; }}
      redraw();
      return;
    }}

    let last = order.pop();
    // If the stack disagrees with the arrays, trust the arrays — undoing a
    // shape that isn't there would silently shift every later index.
    if (last === 'box' && boxes.length === 0) {{ last = 'point'; }}
    if (last === 'point' && points.length === 0) {{ last = 'box'; }}

    if (last === 'box') {{
      if (boxes.length === 0) {{ status.textContent = "Nothing to undo."; return; }}
      boxes.pop();
    }} else {{
      if (points.length === 0) {{ status.textContent = "Nothing to undo."; return; }}
      points.pop();
      labels.pop();
    }}

    redraw();
    status.textContent += "  —  undid last " + last;
  }}

  document.getElementById('annot-prev-btn').addEventListener('click', function() {{ sendAction('prev'); }});
  document.getElementById('annot-next-btn').addEventListener('click', function() {{ sendAction('next'); }});
  document.getElementById('annot-skip-btn').addEventListener('click', function() {{ sendAction('skip'); }});
  document.getElementById('annot-undo-btn').addEventListener('click', undo);
  document.getElementById('annot-jump-btn').addEventListener('click', doJump);

  root.addEventListener('keydown', function(evt) {{
    // While the jump field has focus, letters must type normally — otherwise
    // "frame_87.jpg" would fire next/skip/undo/prev on n, s, u and p.
    if (evt.target === jumpBox) {{
      if (evt.key === 'Enter') {{
        evt.preventDefault();
        doJump();
      }} else if (evt.key === 'Escape') {{
        evt.preventDefault();
        jumpBox.value = '';
        jumpBox.blur();
        root.focus();
      }}
      evt.stopPropagation();
      return;
    }}

    const k = evt.key.toLowerCase();
    if (k === 'n') {{ evt.preventDefault(); sendAction('next'); }}
    else if (k === 's') {{ evt.preventDefault(); sendAction('skip'); }}
    else if (k === 'u') {{ evt.preventDefault(); undo(); }}
    else if (k === 'p') {{ evt.preventDefault(); sendAction('prev'); }}
    else if (k === 'j') {{
      // preventDefault first so the 'j' itself is not typed into the field
      evt.preventDefault();
      jumpBox.focus();
      jumpBox.select();
    }}
  }});

  root.focus();
}})();
</script>
"""

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------
    def run(self):
        try:
            from google.colab import output as colab_output
        except ImportError:
            raise RuntimeError(
                "PromptCollectorColab requires the Colab runtime "
                "(google.colab.output was not importable). Run this in a "
                "Google Colab notebook."
            )

        # Guard against running this via `!python this_file.py` (a detached
        # subprocess with no kernel connection) instead of importing it and
        # calling .run() from a live notebook cell — that failure mode
        # otherwise degrades silently instead of raising.
        from IPython.core.getipython import get_ipython
        if get_ipython() is None:
            raise RuntimeError(
                "No active IPython/Colab kernel detected. Do NOT launch "
                "this with `!python Drone_Anomaly_Annotation_Colab.py` — "
                "that runs as a detached subprocess with no connection to "
                "the notebook frontend, so nothing will render.\n"
                "Instead, in a Colab cell:\n"
                "    from Drone_Anomaly_Annotation_Colab import PromptCollectorColab\n"
                "    collector = PromptCollectorColab(frame_paths=..., output_json='prompts.json')\n"
                "    collector.run()"
            )

        if not self._callback_registered:
            colab_output.register_callback('notebook.recordPrompt', self._on_record)
            self._callback_registered = True

        if self.current_idx >= len(self.frame_paths):
            print("All frames already prompted. Nothing to do — "
                  f"see {self.output_json}.")
            return

        self._render_current_frame()
        print("Annotator is live above. Click on the image, then press "
              "P / N / S / U or use the buttons. Press J (or click the "
              "'Jump to' box) to go straight to a frame by number, filename, "
              "or the keywords first / last / todo, then Enter. Note: after "
              "clicking the "
              "canvas, click the canvas (or its border) once more before "
              "using keyboard shortcuts so the browser focuses the widget.")



def make_prompt_collector(frame_paths, output_json, resume=True):
    """
    Creates a PromptCollectorColab instance.
    """
    return PromptCollectorColab(
        frame_paths=frame_paths,
        output_json=output_json,
        resume=resume
    )


# ── Entry point ──────────────────────────────────────────
if __name__ == "__main__":
    # Stage 1 (run once): 
    #extract_anomalous_frames(
    #     frame_dir="Drone-Anomaly/Solar Panel Inspection/test/frames/s_s1_te_01/",
    #     label_file="Drone-Anomaly/Solar Panel Inspection/test/test_frame_mask/s_s1_te_01.npy",
    #     output_dir="Drone-Anomaly-Anomalous/Solar Panel Inspection/test/frames/s_s1_te_01/",
    # )

    # Stage 2: interactive prompt collection (Colab or plain Jupyter —
    # make_prompt_collector() picks the right backend automatically).
    #frame_paths = sorted(
    #    glob.glob("Drone-Anomaly-Anomalous/Farmland Inspection/**/*.jpg", recursive=True)
    #)

    #collector = make_prompt_collector(
    #    frame_paths=frame_paths,
    #    output_json="prompts.json",
    #    resume=True,
    #)
    #collector.run()

    # Stage 3 (run in a later cell, once prompts.json is fully populated):
    # frame_dir="" because the keys in prompts.json are already full paths
    # (e.g. "Drone-Anomaly-Anomalous/Highway/test/frames/h_s1_te_01/0.jpg"),
    # relative to the working directory the prompt collector was run from.
    # Prefixing frame_dir on top of that duplicated the path and caused
    # every frame lookup to fail.
    batch_generate_masks(
        frame_dir="",
        prompts_json="prompts_b_chk.json",
        output_dir="drone_anomaly_gt_masks/",
    )
    validate_masks("drone_anomaly_gt_masks/")

    # Stage 3, single frame: test one prompt before (or instead of) a full
    # batch. `frame` accepts the same targets as the annotator's Jump box,
    # so a frame you were just looking at transfers over directly.
    # result = generate_single_mask(
    #     prompts_json="prompts.json",
    #     frame="89.jpg",                      # or "0.jpg", or "todo", or "last"
    #     frame_dir="",
    #     output_dir="drone_anomaly_gt_masks/",   # same tree as the batch
    #     show=True,
    # )
    # print(result["mask_path"], result["coverage"])
    #
    # Or with a hand-written prompt, no prompts.json involved:
    #generate_single_mask(
    #    image_path="Drone-Anomaly-Anomalous/Solar Panel Inspection/test/frames/s_s1_te_02/355.jpg", prompt={"prompt_type":"point+box"
    #    ,"points":[[42,118],[122,86],[90,150],[230,86],[98,13],[21,27],[87,70],[160,54],[204,54],[221,117],[166,103],[29,148],[222,147]
    #    ,[231,12],[231,12]],"labels":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"boxes":[[1,0,241,168]],"draw_order":["box","point","point","point"
    #    ,"point","point","point","point","point","point","point","point","point","point","point","point"]},
    #     output_path="scratch/0_mask.png", show=True,
    #     )

    # Stage 4 (run after Stage 3, for visual QA): display each frame's
    # collected annotation next to its generated mask, side by side.
    # review_masks(
    # frame_dir="",
    # prompts_json="prompts_s.json",
    # mask_dir="drone_anomaly_gt_masks/",
    # max_frames=None,
    # output_dir="review_output/",   # new
    # show=False,                    # optional: skip inline display, just save
    # )