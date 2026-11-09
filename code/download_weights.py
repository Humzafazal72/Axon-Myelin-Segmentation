"""Download missing checkpoints after filling the Drive links in config.py."""

from config import CHECKPOINTS
from tem_pipeline.checkpoints import ensure_checkpoints


def main(checkpoints=CHECKPOINTS):
    paths = ensure_checkpoints(checkpoints)

    for stage, path in paths.items():
        print(f"{stage}: {path.resolve()}")

    return paths


if __name__ == "__main__":
    main()
