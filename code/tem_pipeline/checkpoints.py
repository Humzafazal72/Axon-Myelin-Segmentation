"""Check local weights and download missing files from Google Drive with gdown."""

import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .utils import sha256_file


def google_drive_id(link):
    """Accept a Drive file share URL, uc/open URL, or a bare file ID."""
    link = link.strip()

    if re.fullmatch(r"[A-Za-z0-9_-]+", link):
        return link

    parsed = urlparse(link)

    if parsed.scheme != "https" or parsed.hostname != "drive.google.com":
        raise ValueError("Checkpoint links must be HTTPS Google Drive FILE links or file IDs")

    match = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
    file_id = match.group(1) if match else parse_qs(parsed.query).get("id", [""])[0]

    if not re.fullmatch(r"[A-Za-z0-9_-]+", file_id):
        raise ValueError(
            "Could not extract a Google Drive file ID; use a file link, not a folder link"
        )

    return file_id


def checkpoint_problem(path, expected_sha256=""):
    """Catch empty files, Drive error pages, and optional checksum mismatches.

    Full model compatibility is checked when the networks load. This lighter
    check lets the standalone downloader run without importing PyTorch.
    """
    path = Path(path)

    if not path.is_file() or path.stat().st_size == 0:
        return "missing or empty file"

    with path.open("rb") as stream:
        header = stream.read(512).lstrip().lower()

    if header.startswith((b"<!doctype html", b"<html")):
        return "HTML page instead of model weights"

    if expected_sha256 and sha256_file(path) != expected_sha256.lower():
        return "SHA256 mismatch"

    return None


def ensure_checkpoints(settings):
    """Return both local paths, downloading only missing/invalid checkpoints.

    Check both links before making any network request. A blank placeholder
    produces an actionable error, rather than a partial two-model download.
    Completed downloads replace their destination atomically.
    """
    folder = Path(settings.directory)

    if settings.fiber_filename == settings.axon_filename:
        raise ValueError("Fiber and axon checkpoints need different filenames")

    records = {
        "fiber": (settings.fiber_filename, settings.fiber_url, settings.fiber_sha256),
        "axon": (settings.axon_filename, settings.axon_url, settings.axon_sha256),
    }
    paths, pending = {}, []

    for stage, (filename, link, checksum) in records.items():
        if Path(filename).name != filename or not filename:
            raise ValueError("Checkpoint filenames must be plain filenames inside checkpoints/")

        if checksum and not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
            raise ValueError(f"{stage}_sha256 must be blank or a 64-character SHA256 hash")

        path = folder / filename
        paths[stage] = path
        problem = checkpoint_problem(path, checksum)

        if problem is not None:
            pending.append((stage, path, link, checksum, problem))

    missing_links = [stage for stage, _, link, _, _ in pending if not link.strip()]

    if missing_links:
        raise ValueError(
            f"Missing/invalid checkpoints: {', '.join(stage for stage, *_ in pending)}. "
            f"Fill {', '.join(stage + '_url' for stage in missing_links)} in code/config.py "
            f"with Google Drive file links, or place the weight files in {folder}."
        )

    # Resolve every link before downloading the first file. A folder link or a
    # typo in the second URL must not leave a half-configured submission.
    jobs = [
        (stage, path, google_drive_id(link), checksum) for stage, path, link, checksum, _ in pending
    ]

    if jobs:
        import gdown

        folder.mkdir(parents=True, exist_ok=True)

        for stage, path, file_id, checksum in jobs:
            print(f"Downloading {stage} weights -> {path}", flush=True)

            with tempfile.TemporaryDirectory(prefix=f".{stage}-download-", dir=folder) as temporary:
                download_path = Path(temporary) / path.name
                result = gdown.download(id=file_id, output=str(download_path), quiet=False)
                problem = checkpoint_problem(download_path, checksum)

                if result is None or problem is not None:
                    raise RuntimeError(
                        f"Failed to download {stage} weights ({problem or 'download failed'}). "
                        "Check the Drive link and its 'Anyone with the link' access setting."
                    )

                download_path.replace(path)

    return paths
