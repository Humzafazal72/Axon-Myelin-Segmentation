"""Shared geometry and normalization; shapes are always (height, width)."""

import math

import numpy as np
from scipy import ndimage as ndi


def validate_image(image, native_shape):
    if image.ndim != 2 or image.shape != tuple(native_shape):
        raise ValueError(
            f"Expected native grayscale image {tuple(native_shape)}, got {image.shape}"
        )

    if not np.isfinite(image).all() or image.min() < 0:
        raise ValueError("Image intensities must be finite and nonnegative")

    if not np.issubdtype(image.dtype, np.integer) and image.max() > 1:
        raise ValueError("Floating TEM images must already be in [0, 1]")


def validate_pair(image, labels, native_shape):
    validate_image(image, native_shape)

    if labels.ndim != 2 or labels.shape != image.shape:
        raise ValueError(f"Native image/label geometry differs: {image.shape}/{labels.shape}")

    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError("Labels must contain nonnegative integer instance IDs")


def resize_image(image, shape):
    import cv2

    h, w = shape

    # OpenCV's size argument is (width, height). Do not transpose the image or
    # infer orientation from a previously resized file.
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def resize_instances(labels, shape):
    import cv2

    h, w = shape
    return cv2.resize(labels.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST)


def fiber_instances(labels):
    """Union each even axon ID with its next odd myelin ID; ignore label 1."""
    result = labels.astype(np.int32)
    result[result < 2] = 0
    result -= result % 2
    return result


def fiber_training_arrays(image, labels, shape):
    import fastremap

    # Compute validity from ORIGINAL labels before merging paired compartments.
    # An artifact is excluded from loss, rather than treated as background.
    valid = resize_instances((labels != 1).astype(np.uint8), shape).astype(np.uint8)
    fibers = resize_instances(fiber_instances(labels), shape)
    fibers = fastremap.renumber(fibers, in_place=False)[0].astype(np.int32)

    return resize_image(image, shape), fibers, valid


def normalize_crop(image):
    """Match the notebook: integer dtype maximum, not per-crop percentiles."""
    scale = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1
    return image.astype(np.float32) / scale


def instance_boxes(instances):
    return {
        index: box
        for index, box in enumerate(ndi.find_objects(instances), start=1)
        if box is not None
    }


def expand_box(box, shape, context):
    if context < 0:
        raise ValueError("Crop context cannot be negative")

    expanded = []

    for side, limit in zip(box, shape):
        margin = math.ceil((side.stop - side.start) * context)
        expanded.append(slice(max(0, side.start - margin), min(limit, side.stop + margin)))

    return tuple(expanded)


def pad_inputs(images, fibers):
    """Batch native crops without rescaling: replicate pixels, zero-pad masks.

    Both training and inference use this function, including when inference has
    a single crop. Padding is only on the bottom/right and is removed afterward.
    """
    import torch
    from torch.nn import functional as F

    shapes = [image.shape for image in images]
    h = ((max(shape[0] for shape in shapes) + 31) // 32) * 32
    w = ((max(shape[1] for shape in shapes) + 31) // 32) * 32
    inputs = []

    for image, fiber in zip(images, fibers):
        ih, iw = image.shape
        pad = (0, w - iw, 0, h - ih)
        pixels = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))[None, None]
        mask = torch.from_numpy(np.ascontiguousarray(fiber, dtype=np.float32))[None, None]
        inputs.append(torch.cat([F.pad(pixels, pad, mode="replicate"), F.pad(mask, pad)], dim=1)[0])

    return torch.stack(inputs), shapes
