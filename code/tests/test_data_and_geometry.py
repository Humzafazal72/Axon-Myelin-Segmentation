import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import _support  # noqa: F401
import numpy as np
import tifffile
import torch

from config import DataSettings
from tem_pipeline.data import load_training_split
from tem_pipeline.preprocessing import (
    fiber_training_arrays,
    normalize_crop,
    pad_inputs,
    resize_image,
    resize_instances,
    validate_pair,
)


class GeometryTests(unittest.TestCase):
    def test_native_to_cellpose_geometry_and_nearest_instance_ids(self):
        image = np.broadcast_to(np.arange(4096, dtype=np.uint16), (3008, 4096))
        small = resize_image(image, (1504, 2048))
        self.assertEqual(small.shape, (1504, 2048))
        self.assertGreater(int(small[0, -1]), int(small[-1, 0]))

        ids = np.array([[0, 1, 1024], [1025, 0, 60000]], dtype=np.uint16)
        resized = resize_instances(ids, (4, 6))
        self.assertEqual(resized.shape, (4, 6))
        self.assertEqual(set(np.unique(resized)), set(np.unique(ids)))
        np.testing.assert_array_equal(resize_instances(resized, ids.shape), ids)

    def test_paired_compartments_merge_but_artifact_validity_is_preserved(self):
        labels = np.array([[1, 0, 2, 3], [0, 4, 5, 0]], dtype=np.uint16)
        image = np.full(labels.shape, 100, dtype=np.uint8)
        pixels, fibers, valid = fiber_training_arrays(image, labels, labels.shape)
        np.testing.assert_array_equal(fibers, [[0, 0, 1, 1], [0, 2, 2, 0]])
        np.testing.assert_array_equal(valid, labels != 1)
        np.testing.assert_array_equal(pixels, image)

        with self.assertRaisesRegex(ValueError, "geometry differs"):
            validate_pair(image, labels.T, image.shape)

    def test_native_crop_normalization_and_padding(self):
        raw = np.array([[0, 255, 128], [64, 32, 16]], dtype=np.uint8)
        pixels = normalize_crop(raw)
        fiber = np.array([[False, True, False], [True, True, False]])
        inputs, shapes = pad_inputs([pixels], [fiber])
        self.assertEqual(shapes, [(2, 3)])
        self.assertEqual(tuple(inputs.shape), (1, 2, 32, 32))
        np.testing.assert_allclose(inputs[0, 0, :2, :3], raw / 255)
        self.assertTrue(torch.all(inputs[0, 0, 2:, 3:] == pixels[-1, -1]))
        self.assertEqual(float(inputs[0, 1].sum()), float(fiber.sum()))


class SplitTests(unittest.TestCase):
    def test_saved_split_is_reused_and_paths_are_relocated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, labels = root / "images", root / "labels"
            images.mkdir()
            labels.mkdir()

            for index in range(10):
                name = f"sample_{index:04d}.tif"
                tifffile.imwrite(images / name, np.zeros((4, 6), dtype=np.uint8))
                tifffile.imwrite(labels / name, np.zeros((4, 6), dtype=np.uint16))

            settings = DataSettings(
                images_dir=images, labels_dir=labels, split_file=root / "splits.csv"
            )
            original = load_training_split(settings)
            repeated = load_training_split(replace(settings, seed=999))
            self.assertEqual(original, repeated)
            self.assertEqual({row["split"] for row in original}, {"train", "val", "test"})
            self.assertEqual(len({row["micrograph_id"] for row in original}), 10)

            # Old paths in a notebook manifest are ignored, while assignments
            # remain stable and missing/extra dataset rows are rejected.
            text = settings.split_file.read_text().replace(
                "micrograph_id,split", "micrograph_id,split,image_path"
            )
            text = (
                "\n".join(
                    line + (",/old/path.tif" if index else "")
                    for index, line in enumerate(text.splitlines())
                )
                + "\n"
            )
            settings.split_file.write_text(text)
            self.assertEqual(load_training_split(settings), original)
            (labels / "sample_0000.tif").unlink()

            with self.assertRaisesRegex(ValueError, "Unpaired files"):
                load_training_split(settings)


if __name__ == "__main__":
    unittest.main()
