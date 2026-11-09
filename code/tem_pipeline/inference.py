"""Inference-only pipeline: native TEM -> fibers -> axons -> paired uint16 mask."""

import importlib.metadata
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .preprocessing import (
    expand_box,
    instance_boxes,
    normalize_crop,
    pad_inputs,
    resize_image,
    resize_instances,
    validate_image,
)


def encode_compartments(fibers, axons):
    """Renumber fibers and encode axon=2*k, myelin=2*k+1, background=0.

    The uint16 format permits at most 32,767 paired fibers. Check that limit
    BEFORE conversion so a large integer ID can never silently wrap around.
    """
    if fibers.shape != axons.shape or fibers.ndim != 2:
        raise ValueError("Fiber and axon instance maps must be aligned 2D arrays")

    if not np.issubdtype(fibers.dtype, np.integer) or np.any(fibers < 0):
        raise ValueError("Fiber instances must contain nonnegative integers")

    if np.any((axons != 0) & (axons != fibers)):
        raise ValueError("Every predicted axon pixel must belong to its corresponding fiber")

    ids = np.unique(fibers[fibers > 0])

    if len(ids) > 32767:
        raise ValueError("Too many fiber instances for paired uint16 output (maximum 32767)")

    foreground = fibers > 0
    result = np.zeros(fibers.shape, dtype=np.uint16)
    ranks = np.searchsorted(ids, fibers[foreground]) + 1
    result[foreground] = (2 * ranks + (axons[foreground] == 0)).astype(np.uint16)

    return result


def save_prediction(path, prediction):
    """Write a single-channel lossless mask with the input image's filename."""
    path = Path(path)

    if prediction.ndim != 2 or prediction.dtype != np.uint16:
        raise ValueError("Submission masks must be single-channel uint16 arrays")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            tifffile.imwrite(temporary, prediction, photometric="minisblack", compression="deflate")

        elif path.suffix.lower() == ".png":
            Image.fromarray(prediction).save(temporary, format="PNG")

        else:
            raise ValueError(f"Unsupported submission mask format: {path.suffix}")

        temporary.replace(path)

    finally:
        temporary.unlink(missing_ok=True)


class PipelinePredictor:
    def __init__(self, paths, cellpose_settings, axon_settings, native_shape, device):
        import torch
        from cellpose import models

        from .models.unet import SmallUNet

        if importlib.metadata.version("cellpose") != "4.2.1.1":
            raise RuntimeError("Install cellpose==4.2.1.1 from code/requirements.txt")

        self.cellpose_settings = cellpose_settings
        self.axon_settings = axon_settings
        self.native_shape = tuple(native_shape)
        self.device = device

        if not 0 <= axon_settings.threshold <= 1 or axon_settings.context < 0:
            raise ValueError("Invalid axon probability threshold or crop context")

        # Passing an existing absolute weight path prevents Cellpose's fallback
        # to a built-in model when a user-supplied checkpoint is missing.
        for path in paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)

        self.cellpose = models.CellposeModel(
            pretrained_model=str(Path(paths["fiber"]).resolve()),
            device=device,
            gpu=device.type == "cuda",
            use_bfloat16=False,
        )
        self.cellpose.net.eval()

        if self.cellpose.backbone == "sam_vitl" and cellpose_settings.bsize != 256:
            raise ValueError("CPSAM checkpoints require CELLPOSE.bsize=256")

        checkpoint = torch.load(paths["axon"], map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        base = int(state["encoders.0.0.weight"].shape[0])
        self.axon_base_channels = base

        # Infer the channel count from the state_dict so the original notebook's
        # best.pt and plain state_dict exports both remain loadable.
        self.unet = SmallUNet(base)
        self.unet.load_state_dict(state, strict=True)
        self.unet.to(device).eval()

    def _fibers(self, image):
        import torch

        settings = self.cellpose_settings
        small = resize_image(image, settings.shape)

        with torch.inference_mode():
            masks, _, _ = self.cellpose.eval(
                small,
                diameter=None,
                normalize=True,
                bsize=settings.bsize,
                batch_size=settings.batch_size,
                tile_overlap=settings.tile_overlap,
                niter=settings.niter,
                flow_threshold=settings.flow_threshold,
                cellprob_threshold=settings.cellprob_threshold,
                min_size=settings.min_size,
            )

        if masks.shape != tuple(settings.shape):
            raise RuntimeError(f"Cellpose output geometry differs: {masks.shape}/{settings.shape}")

        return resize_instances(masks, image.shape)

    def _axons(self, image, fibers):
        import torch

        axons = np.zeros(image.shape, dtype=np.int32)

        # One variable-size crop at a time keeps original pixels and avoids
        # padding a batch of small fibers to the size of an unusually large one.
        for instance_id, tight_box in instance_boxes(fibers).items():
            box = expand_box(tight_box, image.shape, self.axon_settings.context)
            fiber = fibers[box] == instance_id
            pixels = normalize_crop(image[box])
            inputs, _ = pad_inputs([pixels], [fiber])
            h, w = fiber.shape

            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=self.device.type,
                    enabled=self.axon_settings.amp and self.device.type == "cuda",
                ),
            ):
                probabilities = self.unet(inputs.to(self.device)).float().sigmoid()

            probabilities = probabilities[0, 0, :h, :w].cpu().numpy()
            selected = (probabilities >= self.axon_settings.threshold) & fiber
            axons[box][selected] = instance_id

        return axons

    def predict(self, image):
        validate_image(image, self.native_shape)
        fibers = self._fibers(image)
        axons = self._axons(image, fibers)

        # Myelin is exactly F \ A. No ground-truth labels or artifact masks are
        # accessed anywhere in prediction, including model loading and cropping.
        return encode_compartments(fibers, axons)
