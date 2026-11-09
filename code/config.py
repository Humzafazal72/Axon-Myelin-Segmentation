"""User-editable settings for training, downloading weights, and prediction.

All image shapes use NumPy's (height, width) order. Paths below are anchored to
this file, so running a script from another working directory does not move its
checkpoints or predictions. No command-line arguments are required.
"""

from dataclasses import dataclass
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent


@dataclass(frozen=True)
class DataSettings:
    # Supply ORIGINAL images and original paired compartment labels. Fiber
    # targets are derived during loading; no separate fiber_masks folder is needed.
    images_dir: Path = PROJECT_DIR / "dataset" / "images"
    labels_dir: Path = PROJECT_DIR / "dataset" / "labels"
    split_file: Path = CODE_DIR / "splits" / "micrograph_splits.csv"
    native_shape: tuple[int, int] = (3008, 4096)
    seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass(frozen=True)
class CellposeSettings:
    # Used to initialize NEW training runs. Inference detects the backbone from
    # the supplied checkpoint rather than substituting a built-in model.
    model_type: str = "cpdino"
    shape: tuple[int, int] = (1504, 2048)
    bsize: int = 512  # Inference/training tile side; set 256 for CPSAM.
    batch_size: int = 1  # Number of inference tiles processed together.
    tile_overlap: float = 0.25
    niter: int = 1200
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 2000  # Instance area in Stage 1 pixels, before upsampling.


@dataclass(frozen=True)
class AxonSettings:
    base_channels: int = 16
    context: float = 0.15  # Extra fraction of the tight box on EACH side.
    threshold: float = 0.5
    amp: bool = True  # Used only on CUDA; model weights remain float32.


@dataclass(frozen=True)
class FiberTrainingSettings:
    output_dir: Path = CODE_DIR / "runs" / "fiber"
    cache_dir: Path = CODE_DIR / "runs" / "cache" / "fiber"
    epochs: int = 100
    batch_size: int = 1  # Number of random image crops per optimizer step.
    lr: float = 1e-5
    weight_decay: float = 0.1
    scale_range: float = 0.5
    log_every: int = 10
    resume: str = "auto"  # "auto", "required", or "never".
    init_weights: Path | None = None  # Initialize a NEW run from custom weights.
    weights_only: bool = False  # False keeps optimizer/RNG sidecars for resume.


@dataclass(frozen=True)
class AxonTrainingSettings:
    output_dir: Path = CODE_DIR / "runs" / "axon"
    epochs: int = 50
    batch_size: int = 8
    large_crop_side: int = 700  # Larger crops are processed individually.
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    num_workers: int = 2  # Set 0 in environments that cannot spawn workers.
    resume: bool = True


@dataclass(frozen=True)
class CheckpointSettings:
    directory: Path = CODE_DIR / "checkpoints"
    fiber_filename: str = "cellpose_best.pt"
    axon_filename: str = "best_unet.pt"
    fiber_url: str = "https://drive.google.com/file/d/18fV4cuvhUACkzaBLVdC3ehnvMTTr1S5v/view?usp=sharing"
    axon_url: str = "https://drive.google.com/file/d/1CxJCYN2mkspmId95eLLG9xtcFRO6vFyM/view?usp=drive_link"


@dataclass(frozen=True)
class PredictionSettings:
    images_dir: Path = PROJECT_DIR / "test" / "images"
    output_dir: Path = PROJECT_DIR / "predictions"
    overwrite: bool = False


# Change these objects directly, or construct replacements in a notebook and
# pass them to train.main(), predict.main(), or download_weights.main().
DATA = DataSettings()
CELLPOSE = CellposeSettings()
AXON = AxonSettings()
FIBER_TRAINING = FiberTrainingSettings()
AXON_TRAINING = AxonTrainingSettings()
CHECKPOINTS = CheckpointSettings()
PREDICTION = PredictionSettings()

TRAIN_STAGE = "both"  # "fiber", "axon", or "both"; the two stages train independently.
DEVICE = "auto"  # "auto", "cpu", or a specific CUDA device such as "cuda:0".
