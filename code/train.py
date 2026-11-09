"""Submission training entry point. Edit config.py, then run: python code/train.py."""

from config import (
    AXON,
    AXON_TRAINING,
    CELLPOSE,
    CHECKPOINTS,
    DATA,
    DEVICE,
    FIBER_TRAINING,
    TRAIN_STAGE,
)
from tem_pipeline.data import load_training_split


def main(
    stage=TRAIN_STAGE,
    data=DATA,
    fiber_training=FIBER_TRAINING,
    axon_training=AXON_TRAINING,
    cellpose=CELLPOSE,
    axon=AXON,
    checkpoints=CHECKPOINTS,
    device=DEVICE,
):
    if stage not in {"fiber", "axon", "both"}:
        raise ValueError("TRAIN_STAGE must be 'fiber', 'axon', or 'both'")

    # Resolve the split once. Stage 2 must not silently generate a different
    # split and train on micrographs held out from Stage 1.
    records = load_training_split(data)
    results = {}

    if stage in {"fiber", "both"}:
        from tem_pipeline.training.fiber import train as train_fiber

        results["fiber"] = train_fiber(records, data, fiber_training, cellpose, checkpoints, device)

    if stage in {"axon", "both"}:
        from tem_pipeline.training.axon import train as train_axon

        results["axon"] = train_axon(records, data, axon_training, axon, checkpoints, device)

    # Stage 2 uses GT masks during training, so it can also be run on its own.
    # No call here generates submission predictions or reads the test split.
    return results


if __name__ == "__main__":
    main()
