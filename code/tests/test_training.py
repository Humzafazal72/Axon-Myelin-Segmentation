import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import _support  # noqa: F401
import numpy as np
import tifffile
import torch
from torch import nn

from config import (
    AxonSettings,
    AxonTrainingSettings,
    CellposeSettings,
    CheckpointSettings,
    DataSettings,
    FiberTrainingSettings,
)
from tem_pipeline.data import load_training_split
from tem_pipeline.training import axon, fiber


class TinyNet(nn.Module):
    """Small network using real Cellpose augmentation, flow targets, and loss."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Conv2d(3, 3, 1)
        self.diam_labels = nn.Parameter(torch.tensor([30.0]), requires_grad=False)

    def forward(self, x):
        prediction = torch.nn.functional.conv2d(
            x, self.layer.weight[:, : x.shape[1]], self.layer.bias
        )
        return prediction, torch.zeros((len(x), 256), device=x.device)


class TinyCellpose:
    def __init__(self, **kwargs):
        self.net = TinyNet()
        self.device = torch.device("cpu")
        self.backbone = "dino_vitb"


def make_training_data(root):
    images, labels = root / "images", root / "labels"
    images.mkdir()
    labels.mkdir()

    for index in range(5):
        label = np.zeros((48, 64), dtype=np.uint16)
        label[2:6, 2:6] = 1
        label[10:36, 15:45] = 3
        label[16:30, 22:38] = 2
        image = np.random.default_rng(index).integers(0, 256, label.shape, dtype=np.uint8)
        tifffile.imwrite(images / f"sample_{index:04d}.tif", image)
        tifffile.imwrite(labels / f"sample_{index:04d}.tif", label)

    settings = DataSettings(
        images_dir=images,
        labels_dir=labels,
        native_shape=(48, 64),
        split_file=root / "splits.csv",
        val_fraction=0.2,
        test_fraction=0.2,
    )
    return settings, load_training_split(settings)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cellpose_loss_has_zero_gradient_on_artifacts(self):
        target = torch.zeros((1, 3, 8, 8))
        target[:, 0, 2:6, 2:6] = 1
        valid = torch.ones((1, 8, 8))
        valid[:, :2] = 0
        logits = torch.randn((1, 3, 8, 8), requires_grad=True)
        changed = logits.detach().clone()
        changed[:, :, :2] = 1000
        first, _ = fiber.masked_cellpose_loss(target, logits, valid)
        second, _ = fiber.masked_cellpose_loss(target, changed, valid)
        self.assertAlmostEqual(first.item(), second.item())
        first.backward()
        self.assertEqual(float(logits.grad[:, :, :2].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, :, 2:].abs().sum()), 0)

    def test_fiber_training_publishes_weights_and_resumes_with_optimizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, records = make_training_data(root)
            training = FiberTrainingSettings(
                output_dir=root / "fiber-run",
                cache_dir=root / "cache",
                epochs=2,
                batch_size=1,
                lr=1e-3,
                scale_range=0,
                log_every=1,
            )
            cellpose = CellposeSettings(shape=(32, 48), bsize=32, min_size=0)
            checkpoints = CheckpointSettings(directory=root / "checkpoints")

            with patch("cellpose.models.CellposeModel", side_effect=TinyCellpose):
                result = fiber.train(records, data, training, cellpose, checkpoints, "cpu")
                self.assertEqual(result, checkpoints.directory / "fiber.pt")
                self.assertIn("layer.weight", torch.load(result, weights_only=True))
                fiber.train(
                    records, data, replace(training, epochs=3), cellpose, checkpoints, "cpu"
                )

            metrics = json.loads((training.output_dir / "training_metrics.json").read_text())
            self.assertEqual([row["epoch"] for row in metrics["history"]], [1, 2, 3])
            self.assertEqual(len(list(training.output_dir.glob("models/*_optimizer.pt"))), 1)
            optimizer_path = next(training.output_dir.glob("models/*_optimizer.pt"))
            optimizer_state = torch.load(optimizer_path, weights_only=True)["optimizer_state"]
            train_count = sum(row["split"] == "train" for row in records)
            self.assertEqual(
                {int(state["step"]) for state in optimizer_state["state"].values()},
                {train_count * 3},
            )
            self.assertFalse(list(training.output_dir.rglob("*.tmp")))

    def test_axon_training_uses_only_train_val_then_resumes_and_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, records = make_training_data(root)

            # Test paths deliberately do not exist: training must never read them.
            for row in records:
                if row["split"] == "test":
                    row["image_path"] = root / "test-image-must-not-be-read.tif"
                    row["label_path"] = root / "test-label-must-not-be-read.tif"

            training = AxonTrainingSettings(
                output_dir=root / "axon-run", epochs=1, num_workers=0, batch_size=2
            )
            model = AxonSettings(base_channels=4, amp=False)
            checkpoints = CheckpointSettings(directory=root / "checkpoints")
            result = axon.train(records, data, training, model, checkpoints, "cpu")
            self.assertEqual(result, checkpoints.directory / "axon.pt")
            axon.train(records, data, replace(training, epochs=2), model, checkpoints, "cpu")
            last = torch.load(training.output_dir / "last.pt", weights_only=True)
            self.assertEqual(last["epoch"], 2)
            self.assertEqual(len(last["history"]), 2)
            published = torch.load(result, weights_only=True)
            self.assertIn("model_state", published)
            self.assertNotIn("optimizer_state", published)

            with self.assertRaisesRegex(ValueError, "differs"):
                axon.train(
                    records,
                    data,
                    replace(training, epochs=3),
                    replace(model, context=0.25),
                    checkpoints,
                    "cpu",
                )


if __name__ == "__main__":
    unittest.main()
