"""Native image discovery and one persistent micrograph split for both stages."""

import csv
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

from .utils import write_csv

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png"}


def read_array(path):
    path = Path(path)

    if path.suffix.lower() in {".tif", ".tiff"}:
        return tifffile.imread(path)

    with Image.open(path) as image:
        return np.array(image)


def index_images(folder):
    """Pair by filename stem; reject ambiguity instead of choosing an extension."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")

    result = {}

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if path.stem in result:
            raise ValueError(f"Duplicate filename stem {path.stem!r} in {folder}")

        result[path.stem] = path.resolve()

    if not result:
        raise ValueError(f"No TIFF/PNG images found in {folder}")

    return result


def discover_pairs(settings):
    images = index_images(settings.images_dir)
    labels = index_images(settings.labels_dir)

    if images.keys() != labels.keys():
        raise ValueError(
            f"Unpaired files: images without labels={sorted(images.keys() - labels.keys())}; "
            f"labels without images={sorted(labels.keys() - images.keys())}"
        )

    return [
        dict(micrograph_id=stem, image_path=images[stem], label_path=labels[stem])
        for stem in sorted(images)
    ]


def load_training_split(settings):
    """Create once, then reuse the saved assignments even if paths move.

    The notebook's micrograph_splits.csv is also accepted: its image_path and
    label_path columns are intentionally ignored in favor of the configured
    dataset. Splits must cover the dataset exactly, with no duplicated IDs.
    """
    pairs = discover_pairs(settings)
    path = Path(settings.split_file)

    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))

        assignments = {}

        for row in rows:
            stem, split = row.get("micrograph_id", ""), row.get("split", "")

            if not stem or stem in assignments or split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid or duplicate row in {path}: {row}")

            assignments[stem] = split

        if set(assignments) != {pair["micrograph_id"] for pair in pairs}:
            raise ValueError(
                f"{path} must cover exactly the configured dataset; do not silently regenerate a split"
            )

    else:
        if not 0 < settings.val_fraction < 1 or not 0 <= settings.test_fraction < 1:
            raise ValueError("Invalid validation/test fractions")

        count = len(pairs)
        n_val = max(1, round(count * settings.val_fraction))
        n_test = max(1, round(count * settings.test_fraction)) if settings.test_fraction else 0

        if n_val + n_test >= count:
            raise ValueError("Not enough micrographs for the requested train/val/test split")

        order = np.random.default_rng(settings.seed).permutation(count)
        assignments = {pair["micrograph_id"]: "train" for pair in pairs}

        for index in order[:n_val]:
            assignments[pairs[index]["micrograph_id"]] = "val"

        for index in order[n_val : n_val + n_test]:
            assignments[pairs[index]["micrograph_id"]] = "test"

        rows = [
            dict(micrograph_id=pair["micrograph_id"], split=assignments[pair["micrograph_id"]])
            for pair in pairs
        ]
        write_csv(path, rows, ["micrograph_id", "split"])

    result = [dict(pair, split=assignments[pair["micrograph_id"]]) for pair in pairs]

    if not all(any(row["split"] == split for row in result) for split in ("train", "val")):
        raise ValueError("Both train and val must contain micrographs")

    return result
