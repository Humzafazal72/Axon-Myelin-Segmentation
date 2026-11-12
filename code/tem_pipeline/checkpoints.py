"""Download missing checkpoint files directly into the checkpoint folder."""

from pathlib import Path


def ensure_checkpoints(settings):
    """Return local checkpoint paths, downloading files only if they are missing."""
    folder = Path(settings.directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}

    for stage, filename, url in (
        ("fiber", settings.fiber_filename, settings.fiber_url),
        ("axon", settings.axon_filename, settings.axon_url),
    ):
        path = folder / filename
        paths[stage] = path

        if not path.exists():
            import gdown
            print(f"Downloading weights for {stage} model:\n")
            gdown.download(url=url, output=str(path), quiet=False)

    return paths
