# MemoViT

Code, frozen configurations and pixel annotations for the paper
"MemoViT for Training-Free Aerial Anomaly Detection and Localisation Using
k-NN Frame Retrieval from a Compact DINOv2 Feature Memory" (submitted).

MemoViT keeps a compact bank of whole normal frames, selected by
farthest-point sampling on frozen DINOv2 features, retrieves the bank frames
most similar to a test frame, and scores the frame by comparing its patches
with those of the retrieved frames. No training is required.

## Installation
```bash
pip install -r requirements.txt
```

## Data
Download ShanghaiTech, Drone-Anomaly, UIT-ADrone, VisA and MVTec AD from
their original sources and arrange them as:
```
dataset/<Dataset>/<Scene>/train/frames/<video>/<frame>.jpg
dataset/<Dataset>/<Scene>/test/frames/<video>/<frame>.jpg
dataset/<Dataset>/<Scene>/test/test_frame_mask/<video>.npy
dataset/<Dataset>/<Scene>/test/test_pixel_mask/<video>/<frame>_mask.png   # our annotations
```

## Reproducing the results
```bash
bash run_article.sh
```
All hyperparameters are loaded from the frozen files in `configs/`
(tags `frozen-v1` and `frozen-industrial-v1`). Per-frame scores of all
reported runs are in `results/scores/`.

## Pixel annotations
Masks for DA-BikeRbt, DA-Railway, DA-Solar, DA-Farmland and DA-Highway:
[Zenodo DOI]. See `annotations/README.md` for the format and procedure.
Their use is subject to the licence of the Drone-Anomaly dataset.

## Citation
[BibTeX entry once published]

## Licence
Code: [MIT / Apache-2.0]. Annotations: [CC BY 4.0].
