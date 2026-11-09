import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _support  # noqa: F401
import numpy as np
import torch
from torch import nn

import predict as predict_entry
from config import AxonSettings, CellposeSettings, CheckpointSettings, PredictionSettings
from tem_pipeline.data import read_array
from tem_pipeline.inference import PipelinePredictor, encode_compartments, save_prediction
from tem_pipeline.models.unet import SmallUNet
from tem_pipeline.preprocessing import resize_image, resize_instances


class FakeCellpose:
    """A deterministic Stage 1 boundary; U-Net inference remains real PyTorch."""

    def __init__(self, **kwargs):
        self.net = nn.Identity()
        self.backbone = "dino_vitb"
        self.seen = []

    def eval(self, pixels, **kwargs):
        self.seen.append(pixels.copy())
        mask = np.zeros(pixels.shape, dtype=np.int32)
        mask[3:12, 4:15] = 13
        mask[18:28, 25:42] = 257
        return mask, [], None


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_pair_encoding_compacts_ids_and_rejects_overflow(self):
        fibers = np.array([[0, 50000, 50000], [9, 9, 0]], dtype=np.int32)
        axons = np.array([[0, 50000, 0], [9, 0, 0]], dtype=np.int32)
        encoded = encode_compartments(fibers, axons)
        np.testing.assert_array_equal(encoded, [[0, 4, 5], [2, 3, 0]])
        self.assertEqual(encoded.dtype, np.uint16)

        too_many = np.arange(1, 32769, dtype=np.int32)[None]

        with self.assertRaisesRegex(ValueError, "Too many"):
            encode_compartments(too_many, np.zeros_like(too_many))

        with self.assertRaisesRegex(ValueError, "corresponding fiber"):
            encode_compartments(fibers, np.ones_like(fibers))

    def test_uint16_tiff_and_png_roundtrip(self):
        mask = np.array([[0, 2, 3], [1024, 1025, 65535]], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as temporary:
            for extension in (".tif", ".tiff", ".png"):
                path = Path(temporary) / ("test" + extension)
                save_prediction(path, mask)
                np.testing.assert_array_equal(read_array(path), mask)

    def test_predict_entry_uses_native_pixels_and_never_requires_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, output, weights = root / "images", root / "predictions", root / "checkpoints"
            images.mkdir()
            weights.mkdir()
            raw = np.broadcast_to(np.arange(96, dtype=np.uint8), (64, 96)).copy()
            import tifffile

            tifffile.imwrite(images / "sample_0001.tif", raw)
            (weights / "fiber.pt").write_bytes(b"stub-cellpose-weights")
            torch.save(dict(model_state=SmallUNet(4).state_dict()), weights / "axon.pt")
            checkpoint_settings = CheckpointSettings(directory=weights)
            cellpose_settings = CellposeSettings(shape=(32, 48), min_size=0)
            fake = FakeCellpose()

            with (
                patch("cellpose.models.CellposeModel", return_value=fake),
                patch("gdown.download") as download,
            ):
                result = predict_entry.main(
                    PredictionSettings(images_dir=images, output_dir=output),
                    checkpoint_settings,
                    cellpose_settings,
                    AxonSettings(amp=False),
                    native_shape=raw.shape,
                    device="cpu",
                )
                download.assert_not_called()

            self.assertEqual(result, output)
            self.assertEqual({path.name for path in output.iterdir()}, {"sample_0001.tif"})
            np.testing.assert_array_equal(fake.seen[0], resize_image(raw, (32, 48)))
            expected_fibers = resize_instances(fake.eval(fake.seen[0])[0], raw.shape) > 0
            mask = read_array(output / "sample_0001.tif")
            self.assertEqual(mask.shape, raw.shape)
            self.assertEqual(mask.dtype, np.uint16)
            np.testing.assert_array_equal(mask > 0, expected_fibers)
            self.assertTrue((root / "predictions_manifest.json").is_file())

            # Existing outputs must not be silently overwritten on a rerun.
            with self.assertRaises(FileExistsError):
                predict_entry.main(PredictionSettings(images_dir=images, output_dir=output))

    def test_plain_unet_weights_load_and_empty_fibers_produce_empty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {stage: root / f"{stage}.pt" for stage in ("fiber", "axon")}
            paths["fiber"].write_bytes(b"stub-cellpose-weights")
            torch.save(SmallUNet(4).state_dict(), paths["axon"])
            fake = FakeCellpose()
            fake.eval = lambda image, **_: (np.zeros(image.shape, dtype=np.int32), [], None)

            with patch("cellpose.models.CellposeModel", return_value=fake):
                predictor = PipelinePredictor(
                    paths,
                    CellposeSettings(shape=(32, 48)),
                    AxonSettings(amp=False),
                    (64, 96),
                    torch.device("cpu"),
                )
                mask = predictor.predict(np.zeros((64, 96), dtype=np.uint8))

            self.assertFalse(mask.any())
            self.assertEqual(mask.dtype, np.uint16)


if __name__ == "__main__":
    unittest.main()
