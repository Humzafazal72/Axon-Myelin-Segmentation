"""Generate submission masks. Edit config.py, then run: python code/predict.py."""

from dataclasses import asdict
from pathlib import Path

from config import AXON, CELLPOSE, CHECKPOINTS, DATA, DEVICE, PREDICTION
from tem_pipeline.checkpoints import ensure_checkpoints
from tem_pipeline.data import index_images, read_array
from tem_pipeline.inference import PipelinePredictor, save_prediction
from tem_pipeline.utils import resolve_device, write_json


def main(
    prediction=PREDICTION,
    checkpoints=CHECKPOINTS,
    cellpose=CELLPOSE,
    axon=AXON,
    native_shape=DATA.native_shape,
    device=DEVICE,
):
    images = index_images(prediction.images_dir)
    output_dir = Path(prediction.output_dir)

    if output_dir.resolve() == Path(prediction.images_dir).resolve():
        raise ValueError("The predictions folder must be separate from the input images")

    if not prediction.overwrite:
        existing = [
            output_dir / path.name for path in images.values() if (output_dir / path.name).exists()
        ]

        if existing:
            raise FileExistsError(
                f"Prediction already exists: {existing[0]}; set PREDICTION.overwrite=True to replace masks"
            )

    # Check FILES as well as the directory. An existing checkpoints/ folder may
    # still be missing one model. Local files are reused without any download.
    paths = ensure_checkpoints(checkpoints)
    selected_device = resolve_device(device)
    predictor = PipelinePredictor(paths, cellpose, axon, native_shape, selected_device)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "complete": False,
        "native_shape_hw": list(native_shape),
        "cellpose_shape_hw": list(cellpose.shape),
        "cellpose": asdict(cellpose),
        "axon": asdict(axon),
        "checkpoints": {stage: str(path) for stage, path in paths.items()},
        "outputs": [],
    }
    manifest["axon"]["base_channels"] = predictor.axon_base_channels
    # Keep bookkeeping next to the submission folder so predictions/ contains
    # only the requested mask files.
    manifest_path = output_dir.parent / f"{output_dir.name}_manifest.json"
    write_json(manifest_path, manifest)

    for index, path in enumerate(images.values(), start=1):
        print(
            f"[{index}/{len(images)}] {path.name}: {native_shape} -> {cellpose.shape} -> native crops",
            flush=True,
        )
        mask = predictor.predict(read_array(path))
        destination = output_dir / path.name
        save_prediction(destination, mask)
        manifest["outputs"].append(destination.name)
        manifest["complete"] = index == len(images)
        write_json(manifest_path, manifest)

    return output_dir


if __name__ == "__main__":
    main()
