import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import _support  # noqa: F401

from config import CheckpointSettings
from tem_pipeline.checkpoints import ensure_checkpoints, google_drive_id


class DownloadTests(unittest.TestCase):
    def test_blank_links_fail_before_creating_folder_or_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            settings = CheckpointSettings(directory=Path(temporary) / "absent")

            with patch("gdown.download") as download:
                with self.assertRaisesRegex(ValueError, "fiber_url, axon_url"):
                    ensure_checkpoints(settings)

                download.assert_not_called()

            self.assertFalse(settings.directory.exists())

    def test_missing_directory_is_created_and_missing_files_download_once(self):
        payload = b"test-checkpoint-bytes"

        def fake_download(*, id, output, quiet):
            self.assertIn(id, {"fiber_ID", "axon_ID"})
            Path(output).write_bytes(payload)
            return output

        with tempfile.TemporaryDirectory() as temporary:
            settings = CheckpointSettings(
                directory=Path(temporary) / "checkpoints",
                fiber_url="https://drive.google.com/file/d/fiber_ID/view?usp=sharing",
                axon_url="https://drive.google.com/uc?id=axon_ID",
                fiber_sha256=hashlib.sha256(payload).hexdigest(),
            )

            with patch("gdown.download", side_effect=fake_download) as download:
                paths = ensure_checkpoints(settings)
                self.assertEqual(download.call_count, 2)
                self.assertEqual(paths["fiber"].read_bytes(), payload)

                ensure_checkpoints(settings)
                self.assertEqual(download.call_count, 2)

            self.assertEqual(
                {path.name for path in settings.directory.iterdir()}, {"fiber.pt", "axon.pt"}
            )

    def test_existing_folder_with_one_weight_still_downloads_the_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "fiber.pt").write_bytes(b"existing-fiber")
            settings = CheckpointSettings(directory=folder, axon_url="axon_ID")

            def fake_download(**kwargs):
                Path(kwargs["output"]).write_bytes(b"new-axon")
                return kwargs["output"]

            with patch("gdown.download", side_effect=fake_download) as download:
                ensure_checkpoints(settings)
                self.assertEqual(download.call_count, 1)
                self.assertEqual(download.call_args.kwargs["id"], "axon_ID")

            self.assertEqual((folder / "fiber.pt").read_bytes(), b"existing-fiber")

    def test_error_pages_and_bad_hashes_are_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / "fiber.pt").write_bytes(b"existing-fiber")
            settings = CheckpointSettings(directory=folder, axon_url="axon_ID")

            for payload, checksum in (
                (b"<!DOCTYPE html><html>permission denied</html>", ""),
                (b"wrong-hash", "a" * 64),
            ):

                def fake_download(**kwargs):
                    Path(kwargs["output"]).write_bytes(payload)
                    return kwargs["output"]

                with patch("gdown.download", side_effect=fake_download):
                    with self.assertRaises(RuntimeError):
                        ensure_checkpoints(replace(settings, axon_sha256=checksum))

                self.assertFalse((folder / "axon.pt").exists())
                self.assertEqual({path.name for path in folder.iterdir()}, {"fiber.pt"})

    def test_drive_link_formats_and_folder_rejection(self):
        self.assertEqual(google_drive_id("https://drive.google.com/open?id=abc-123_"), "abc-123_")
        self.assertEqual(google_drive_id("abc-123_"), "abc-123_")

        with self.assertRaises(ValueError):
            google_drive_id("https://drive.google.com/drive/folders/folder_ID")


if __name__ == "__main__":
    unittest.main()
