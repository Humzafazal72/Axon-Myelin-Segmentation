# TEM fiber, axon, and myelin segmentation

This submission trains two independent models and combines them for prediction:

1. A fine-tuned Cellpose model detects complete fiber instances.
2. A two-input U-Net receives each native TEM crop and its predicted fiber mask.
   It predicts the inner axon; myelin is the remainder of that fiber.

The implementation runs independently of `Experiments/`. There is no separate
evaluation script, notebook dependency, argparse interface, or external logging
account requirement.

## Layout

```text
code/
├── README.md
├── requirements.txt
├── config.py
├── train.py
├── predict.py
├── download_weights.py
├── checkpoints/                 # fiber.pt and axon.pt, local or downloaded
├── splits/                     # persistent micrograph_splits.csv
├── tests/                      # synthetic CPU checks
└── tem_pipeline/
    ├── data.py
    ├── preprocessing.py
    ├── inference.py
    ├── checkpoints.py
    ├── utils.py
    ├── models/unet.py
    └── training/
        ├── fiber.py
        └── axon.py

predictions/                    # single-channel uint16 TIFF/PNG masks
```

## Environment

Use Python 3.11 or later. A CUDA-enabled PyTorch installation and a GPU are
recommended for training and full-resolution prediction. Git is needed to install
the pinned DINOv3 dependency used by the default CPDINO backbone.

From the project root:

```bash
python -m pip install -r code/requirements.txt
```

The requirements pin Cellpose 4.2.1.1, PyTorch 2.6.0, and torchvision 0.21.0.
When your platform needs a specific CUDA wheel index, install the matching
PyTorch/torchvision pair for that platform first. Do not upgrade Cellpose without
checking its internal augmentation and loss APIs used by the fiber trainer.

## Configure paths and settings

Edit `code/config.py`. Paths are resolved relative to that file, not the shell's
working directory. The dataclass defaults are the editable settings; the objects
at the bottom are passed to the entry points.

For training, configure `DataSettings.images_dir` and `labels_dir`:

```text
dataset/
├── images/
│   ├── sample_0001.tif
│   └── ...
└── labels/
    ├── sample_0001.tif
    └── ...
```

Images must be original grayscale arrays of shape **(3008, 4096)**, in NumPy
**(height, width)** order. Each label image must have the same shape. Pairing uses
filename stems; missing pairs and duplicate stems are errors. The GT encoding is:

- `0`: background.
- `1`: ignored artifact pixels.
- Even ID `a >= 2`: axon.
- Next odd ID `a + 1`: the same fiber's myelin.

The fiber trainer derives each fiber from that even/odd union during loading;
it does not require a separately prepared `fiber_masks/` folder.

For prediction, configure `PredictionSettings.images_dir` with original test
micrographs, and `output_dir` with the desired submission folder. Labels are
never read during prediction.

## Train

Set `TRAIN_STAGE` in `config.py` to `"fiber"`, `"axon"`, or `"both"`, then run:

```bash
python code/train.py
```

The first run creates `code/splits/micrograph_splits.csv` from sorted micrograph
IDs, with seed 42 and a 70/15/15 train/validation/test split. Later runs reuse
that file, and both models use the same assignments. The test split is excluded
from training and model selection.

You can supply an existing split CSV with `micrograph_id` and `split` columns
(`train`, `val`, `test`). The U-Net notebook's saved micrograph split is accepted;
its old absolute image paths are ignored. The CSV must cover the configured
dataset exactly. Do not silently change the split when reproducing a trained
checkpoint: the two experimental trainers previously used different splits.

**Fiber training** retains the custom artifact-masked flow MSE and foreground
BCE, AdamW, learning-rate schedule, aligned validity augmentation, and best/latest
checkpoint behavior. Original images are downsampled with area interpolation to
**(1504, 2048)**. Labels and validity use nearest-neighbor interpolation. Both
training and prediction use the same shared resize function; there is no automatic
height/width swap. Original label-1 pixels are excluded from both loss terms.

**Axon training** retains the notebook's SmallUNet, GroupNorm, augmentations,
AdamW, plateau scheduler, gradient clipping, and masked BCE + Dice loss. Each GT
fiber defines a tight native crop with 15% context on each side. Pixels are
divided by their integer dtype maximum; floating inputs must already be in
`[0, 1]`. Image padding replicates edge pixels, while mask padding is zero.
Spatial dimensions are padded to multiples of 32. Incomplete GT compartment
pairs are skipped and logged. Large crops use batch size one. Stage 2 training
uses **GT** fibers and does not require a trained Stage 1 model.

Training artifacts and resumable states live under `code/runs/fiber/` and
`code/runs/axon/`. Both stage settings specify **total** target epochs, including
already completed epochs. The fiber trainer can resume after interrupted
validation using optimizer/RNG sidecars and saved file size/time metadata. The axon trainer resumes
from completed epochs with optimizer, scheduler, scaler, and RNG state. With
multiple persistent data-loader workers, worker-local augmentation streams are
not guaranteed to match an uninterrupted process exactly; use `num_workers=0`
when checking exact CPU reproducibility.

After successful training, the selected best models are published to:

```text
code/checkpoints/fiber.pt        # ordinary Cellpose state_dict
code/checkpoints/axon.pt         # dict containing model_state and small metadata
```

The original experimental Cellpose best weights can be copied to `fiber.pt`.
The U-Net notebook's `best.pt` can be copied to `axon.pt`; prediction accepts its
`model_state` format or a plain SmallUNet state_dict. Copying existing weights
does not retrain them or change their historical data splits.

## Download weights

The download links are intentionally blank. Fill the two Google Drive **file**
links in `CheckpointSettings` in `config.py`:

```python
fiber_url: str = ""  # TODO: https://drive.google.com/file/d/FIBER_FILE_ID/view
axon_url: str = ""  # TODO: https://drive.google.com/file/d/AXON_FILE_ID/view
```

Make both files accessible to “Anyone with the link.” Raw file IDs and Drive
`uc?id=...` links also work. Folder links are not supported.

Download explicitly with:

```bash
python code/download_weights.py
```

The script uses `gdown`, reuses valid local files, checks for empty files or HTML
error pages, and atomically publishes successful downloads. It checks all missing
links before starting a network request. Missing links produce an error naming
the settings to fill; they are not fabricated or silently replaced with base
model weights.

## Predict

After configuring test images and checkpoints, run:

```bash
python code/predict.py
```

`predict.py` checks **both files** in the checkpoint folder before loading models.
If the folder or either checkpoint is missing, it invokes the same `gdown`
download helper automatically. Providing both files locally allows offline
prediction without needing download links.

For every test micrograph:

1. Resize original pixels to **(1504, 2048)** for Cellpose.
2. Restore the instance map to **(3008, 4096)** with nearest-neighbor interpolation.
3. Crop each predicted fiber from the original TEM image with 15% context.
4. Run the U-Net with the predicted fiber mask as its second input channel.
5. Threshold the axon probability at 0.5, clip it to its fiber, and set
   `myelin = fiber & ~axon`.
6. Save a single-channel **uint16** mask with the exact input filename.

Each predicted fiber `k` is encoded as `2*k` for axon and `2*k+1` for myelin;
background is zero. TIFF, TIFF with `.tiff` extension, and PNG inputs retain their
respective filename and extension. No ground-truth artifact mask is available or
used during prediction. More than 32,767 fibers is rejected before uint16
conversion. Set `PredictionSettings.overwrite=True` to replace existing masks.

The prediction manifest is written **beside** the output folder, for example
`predictions_manifest.json`; `predictions/` contains only masks. It records
checkpoint paths, settings, written filenames, and completion status.

The default Cellpose tile size is 512 for CPDINO. Set `CellposeSettings.bsize=256`
for a CPSAM checkpoint. The backbone and U-Net width are loaded from their
respective weights. Inspect the configured thresholds if reproducing a run with
non-default inference settings.

## Notebook use

The entry points are also callable without a CLI. For example, from the project
root:

```python
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path("code").resolve()))

from config import CHECKPOINTS, PREDICTION
from predict import main

main(
    prediction=replace(
        PREDICTION,
        images_dir=Path("/kaggle/input/test/images"),
        output_dir=Path("/kaggle/working/predictions"),
    ),
    checkpoints=replace(CHECKPOINTS, directory=Path("/kaggle/working/checkpoints")),
)
```

## Checks

The tests use synthetic data and small CPU models; they do not download trained
weights or run the actual TEM dataset:

```bash
python -m unittest discover -s code/tests -v
```
