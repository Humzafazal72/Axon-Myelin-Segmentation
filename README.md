# Axon & Myelin Segmentation in TEM Images

A two-stage instance segmentation pipeline for detecting nerve fibers and separating **axons** from their surrounding **myelin sheaths** in transmission electron microscopy (TEM) images.

The pipeline combines **fine-tuned Cellpose** for complete fiber instance detection with a **fiber-conditioned, two-channel U-Net** for axon segmentation. Myelin is computed as the remaining portion of the segmented fiber.

> **Research implementation:** Training images and annotations are not included. The checkpoint download URLs are defined in [`code/config.py`](code/config.py). The repository also includes a [method description](method_description.pdf) and [detailed technical documentation](code/README.md).

## Method overview

```text
Original grayscale TEM image (3008 × 4096)
                    |
           Area downsampling
                    v
           1504 × 2048 image
                    |
       Fine-tuned Cellpose (Stage 1)
                    v
           Fiber instance masks
                    |
     Nearest-neighbor upsampling
                    v
          Native-size fiber masks
                    |
    Crop each fiber with 15% context
                    |
          +---------+---------+
          |                   |
     Original TEM crop    Binary fiber mask
          |                   |
          +---------+---------+
                    v
        Two-input U-Net (Stage 2)
                    |
          Axon probability map
                    |
        Threshold and clip to fiber
                    v
             Axon instance
                    |
           Myelin = Fiber − Axon
                    v
      Paired uint16 instance labels
```

| Stage | Model | Purpose |
| --- | --- | --- |
| **1: Fiber detection** | Fine-tuned **Cellpose** (CPDINO backbone by default) | Detect individual complete nerve fibers in downsampled images. |
| **2: Axon segmentation** | Small two-input U-Net with GroupNorm | Predict each axon using the original-resolution TEM crop and corresponding fiber mask. |
| **Postprocessing** | Deterministic compartment assignment | Constrain the axon to the detected fiber and assign remaining fiber pixels to myelin. |

The postprocessing guarantees that an axon stays inside its predicted fiber and that predicted axon and myelin regions do not overlap. It cannot correct fibers that Stage 1 misses or merges.

## Quick start

**Requirements:** Python 3.11+, Git, and preferably an NVIDIA GPU with a compatible CUDA-enabled PyTorch build. CPU execution may be slow.

```bash
git clone https://github.com/Humzafazal72/Axon-Myelin-Segmentation.git
cd Axon-Myelin-Segmentation

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

The requirements include **PyTorch 2.6.0**, **torchvision 0.21.0**, **Cellpose 4.2.1.1**, and a pinned revision of **DINOv3**. If needed, install PyTorch/torchvision using the appropriate CUDA wheel index before installing the remaining dependencies. Keep the pinned Cellpose version because the training implementation uses version-specific internal APIs.

## Inference with pretrained weights

1. Put original test micrographs (`.tif`, `.tiff`, or `.png`) in `test/images/`, or edit `PredictionSettings.images_dir` in [`code/config.py`](code/config.py).
2. In `CheckpointSettings`, verify the checkpoint filenames and Google Drive links. Defaults are `code/checkpoints/cellpose_best.pt` and `code/checkpoints/best_unet.pt`.
3. Run:

```bash
# Optional: fetch any missing model checkpoints
python code/download_weights.py

# Generate predictions (also attempts to fetch missing weights)
python code/predict.py
```

The script creates one lossless, **single-channel uint16** mask per input image in `predictions/` by default. Output filenames and dimensions match their inputs. An accompanying `predictions_manifest.json` records the inference settings, checkpoint paths, output names, and completion status.

**Notes:** Both weights are required for the complete pipeline. Downloading depends on having access to the configured Google Drive files. Existing predictions are not overwritten unless `PredictionSettings.overwrite=True`. If using CPSAM rather than CPDINO weights, set `CellposeSettings.bsize=256`.

## Dataset and annotation format

The current configuration expects original **grayscale TEM images of shape (3008, 4096)** in NumPy height × width order. Training images and labels must have identical filename stems and dimensions.

```text
dataset/
├── images/
│   ├── sample_0001.tif
│   └── ...
└── labels/
    ├── sample_0001.tif
    └── ...
test/
└── images/
    ├── test_0001.tif
    └── ...
```

Ground-truth labels follow a paired instance-ID convention:

| Pixel label | Meaning |
| --- | --- |
| `0` | Background |
| `1` | Ignored / artifact region |
| Even ID `a ≥ 2` | Axon of a fiber |
| Odd ID `a + 1` | Myelin of that same fiber |

For example, IDs **2/3** represent the axon and myelin of one fiber, while **4/5** represent the next. Stage 1 derives each full-fiber mask by combining the even and following odd labels; a separate `fiber_masks/` directory is **not required**. Original artifact-label pixels are excluded from the Stage 1 losses. Stage 2 skips incomplete ground-truth compartment pairs.

**Prediction format:** Each predicted fiber `k` is encoded as `2*k` for its axon and `2*k + 1` for its myelin. Background remains `0`; artifact label `1` is not used at inference. The uint16 representation supports at most **32,767** paired fiber instances per image.

For dataset and challenge details, see the [TEM Challenge website](https://alexandrojim.github.io/tem-challenge/). Follow its own data-access and usage terms.

## Training

Change settings in [`code/config.py`](code/config.py), including `DataSettings.images_dir`, `DataSettings.labels_dir`, `TRAIN_STAGE`, and `DEVICE`:

```python
TRAIN_STAGE = "both"  # "fiber", "axon", or "both"
DEVICE = "auto"       # "auto", "cpu", or "cuda:0"
```

Run:

```bash
python code/train.py
```

Training is configured through `code/config.py`; the supplied scripts do **not** require command-line flags.

### Data splits

A first training run creates `code/splits/micrograph_splits.csv` using seed **42** with **70% train / 15% validation / 15% test** proportions. Splitting happens **by micrograph**, not by extracted fiber crop, to avoid leakage between partitions. Both stages reuse this persistent split. You can provide a pre-existing CSV with `micrograph_id` and `split` columns; it must cover the dataset exactly. The held-out test partition is excluded from fitting and checkpoint selection.

### Stage 1: Cellpose fiber detection

- Derive complete-fiber instance labels by merging paired axon/myelin annotations.
- Downsample images to **1504 × 2048** (area interpolation), and labels/validity maps via nearest-neighbor interpolation.
- Apply aligned spatial augmentations to image, targets, and ignored-pixel validity mask.
- Optimize **artifact-masked flow MSE + foreground BCE**, using AdamW and a learning-rate schedule.
- Save best weights based on validation loss, plus latest weights and resumable state.

### Stage 2: Fiber-conditioned axon segmentation

- Extract native-resolution crops around each **ground-truth** fiber, adding **15% context** per side.
- Feed the original normalized TEM image and the binary mask of that fiber as **two channels** to a lightweight U-Net.
- Use **fiber-masked binary cross-entropy + soft Dice loss** to learn the axon compartment.
- Pad crops to convenient model dimensions rather than resizing away fine structure. Large crops run individually.
- Use AdamW, a validation-loss learning-rate scheduler, checkpoint selection, and resumable training.

**Important:** Stage 2 can be trained independently with ground-truth fibers; during actual inference it uses **predicted** fibers. Training artifacts are saved in `code/runs/fiber/` and `code/runs/axon/`. Selected inference weights are published in `code/checkpoints/` under names configured by `CheckpointSettings`. Epoch settings specify **total** epochs, including completed epochs when resuming.

## Repository structure

```text
.
├── README.md
├── method_description.pdf
├── requirements.txt
├── code/
│   ├── README.md                  # Detailed implementation documentation
│   ├── requirements.txt
│   ├── config.py                  # Dataset, model, and runtime configuration
│   ├── train.py                   # Both training entry points
│   ├── predict.py                 # End-to-end inference
│   ├── download_weights.py        # Fetch missing checkpoints
│   ├── checkpoints/               # Weights (not tracked)
│   ├── splits/                    # Persistent micrograph splits
│   └── tem_pipeline/
│       ├── data.py                # Image/label discovery and splitting
│       ├── preprocessing.py       # Geometry, labels, cropping and padding
│       ├── inference.py           # Full prediction pipeline
│       ├── checkpoints.py         # Checkpoint retrieval
│       ├── models/unet.py         # Two-channel SmallUNet
│       └── training/
│           ├── fiber.py           # Cellpose training
│           └── axon.py            # U-Net training
└── predictions/                   # Generated masks (not tracked)
```

## Reproducibility and limitations

- **Native resolution matters.** The code validates images against the configured shape `(3008, 4096)` and keeps Stage 2 crops at original pixel resolution.
- **Stage 1 errors propagate.** A missed, merged, or incorrectly split fiber can affect the corresponding axon/myelin prediction even if the U-Net is otherwise accurate.
- **Training and deployment differ in Stage 2 input quality.** The U-Net sees ground-truth fiber masks during training but predicted masks at inference.
- **Checkpoint compatibility matters.** Inference loads the Cellpose backbone and U-Net width from the supplied weights, and enforces Cellpose version **4.2.1.1**.
- **No verified benchmark score is claimed here.** A complete end-to-end evaluation should be conducted on a held-out micrograph-level dataset, without using test labels for model selection.
- **Research use only.** The segmentation output is not validated for clinical diagnosis. Biological measurements derived from the masks require independent quality control.

## Further documentation and acknowledgments

See the [detailed implementation guide](code/README.md) and [method description](method_description.pdf). This work builds on [Cellpose](https://github.com/MouseLand/cellpose), [PyTorch](https://pytorch.org/), and resources from the [TEM Challenge](https://alexandrojim.github.io/tem-challenge/). If using this project in research, cite the original model and dataset/challenge publications.
