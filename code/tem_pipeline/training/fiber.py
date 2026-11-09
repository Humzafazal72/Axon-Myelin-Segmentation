"""Cellpose training with artifact-masked foreground and flow losses.

The optimizer, learning-rate schedule, aligned validity augmentation, and
resumable optimizer/RNG sidecars are retained from the experimental trainer.
This module consumes original paired labels through shared preprocessing and
uses the submission's persistent split. It does not import Experiments, require
W&B credentials, or run the separate end-to-end evaluation experiment.
"""

import importlib.metadata
import json
import math
import random
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from ..data import read_array, split_assignments
from ..preprocessing import fiber_training_arrays, validate_pair
from ..utils import resolve_device, write_json

CELLPOSE_VERSION = "4.2.1.1"


def load_and_preprocess_pair(
    image_path, label_path, target_h=1504, target_w=2048, native_shape=None
):
    """Create fiber/validity targets without losing original artifact locations."""
    image, labels = read_array(image_path), read_array(label_path)
    validate_pair(image, labels, native_shape or image.shape)

    return fiber_training_arrays(image, labels, (target_h, target_w))


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_json(path, data):
    # Replace the previous JSON only after the new file has finished writing.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def weights_stamp(path):
    """Detect interrupted checkpoint writes from file size and modification time."""
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns]


def atomic_torch_save(state, path):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, str(temporary))
    temporary.replace(path)


def save_weights(net, path):
    # The inference file stays a normal Cellpose state_dict. Resume information
    # goes into separate files so datasets/classes are never needed for inference.
    atomic_torch_save(net.state_dict(), path)


def checkpoint_paths(args):
    folder = Path(args.output_dir) / "models"
    latest = folder / f"{args.model_name}_latest.pt"
    best = folder / f"{args.model_name}_best.pt"

    if not latest.is_file():
        # Also accept a checkpoint that the user has named simply latest.pt.
        for candidate in (folder / "latest.pt", Path(args.output_dir) / "latest.pt"):
            if candidate.is_file():
                latest, best = candidate, candidate.with_name("best.pt")
                break

    return {
        "latest": latest,
        "best": best,
        "progress": latest.with_name(latest.stem + "_progress.pt"),
        "optimizer": latest.with_name(latest.stem + "_optimizer.pt"),
    }


def find_resume_weights(args):
    mode = getattr(args, "resume", "auto")
    explicit = getattr(args, "resume_from", None)
    if mode == "never":
        if explicit:
            raise ValueError("resume never cannot be combined with resume_from")
        return None

    path = Path(explicit) if explicit else checkpoint_paths(args)["latest"]
    if path.is_file():
        return path.resolve()
    if explicit or mode == "required":
        raise FileNotFoundError(f"Resume weights not found: {path}")
    return None


@contextmanager
def training_lock(output_dir):
    # A restarted process must not train over another active writer's output.
    import fcntl

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".training.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another training process is using {folder}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def capture_rng_state(device):
    import numpy as np
    import torch

    # Save every RNG used by training so a restart can continue its random sequence.
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": (
            state[0],
            torch.from_numpy(state[1].astype(np.int64)),
            state[2],
            state[3],
            state[4],
        ),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng_state(state, device):
    import numpy as np
    import torch

    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])

    algorithm, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state(
        (algorithm, keys.numpy().astype(np.uint32), position, has_gauss, cached_gauss)
    )
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def resume_contract(args, train_entries, val_entries):
    # Paths may change when moving an instance; source image order and geometry
    # must stay the same to continue this run with its existing validation split.
    def describe(entries):
        return [
            [Path(entry["source"]["files"][0][0]).stem, entry["shape"], entry["n_instances"]]
            for entry in entries
        ]

    return {
        "model_type": getattr(args, "model_type", None),
        "bsize": args.bsize,
        "target_h": args.target_h,
        "target_w": args.target_w,
        "ignore_policy": "original_id_1_excluded_v1",
        "train": describe(train_entries),
        "validation": describe(val_entries),
    }


def check_resume_storage(net, args):
    # Atomic replacement temporarily keeps the old and new optimizer files.
    # Check this before preprocessing/training rather than failing after an epoch.
    paths = checkpoint_paths(args)
    folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    weights = sum(value.numel() * value.element_size() for value in net.state_dict().values())
    # AdamW keeps two moment tensors for each trainable parameter.
    moments = 2 * sum(
        parameter.numel() * parameter.element_size()
        for parameter in net.parameters()
        if parameter.requires_grad
    )
    weights_only = getattr(args, "weights_only", False)
    peak = 3 * weights if weights_only else max(3 * weights + moments, 2 * weights + 2 * moments)
    reusable = [paths["best"], paths["latest"]] + ([] if weights_only else [paths["optimizer"]])
    available = shutil.disk_usage(folder).free + sum(
        path.stat().st_size for path in reusable if path.is_file()
    )

    if available < peak + 16 * 1024**2:
        raise RuntimeError(
            f"Model/resume files need about {peak / 1024**3:.2f} GiB including temporary writes; "
            f"{available / 1024**3:.2f} GiB is available to this run. Free space, use another output "
            "volume, or use weights_only to omit AdamW state."
        )


def set_global_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # TF32 accelerates supported float32 GPU operations. These seed/backend
    # settings improve repeatability, but do not promise identical GPU results.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prepare_cached_pair(pair, args, device):
    """Precompute one image's targets, then persist them before reading the next."""
    import numpy as np
    import tifffile
    from cellpose import dynamics, models, train, utils

    image_path, mask_path = pair

    # Reuse one folder per image and target shape. The complete source metadata
    # below determines whether the files in that folder need to be regenerated.
    source = {
        "version": CELLPOSE_VERSION,
        "cache_format": 4,
        "label_encoding": "paired_compartments",
        "ignore_policy": "original_id_1_excluded_v1",
        "target_h": args.target_h,
        "target_w": args.target_w,
        "files": [],
    }
    for path in pair:
        stat = path.stat()
        source["files"].append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    folder = Path(args.cache_dir).resolve() / f"{image_path.stem}_{args.target_h}_{args.target_w}"
    paths = {
        "image": folder / "image.tif",
        "mask": folder / "mask.tif",
        "valid": folder / "valid.tif",
        "targets": folder / "targets.npz",
        "metadata": folder / "metadata.json",
    }

    # Reuse only complete cache entries whose metadata matches the current inputs.
    if all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        try:
            metadata = json.loads(paths["metadata"].read_text())
            if metadata["source"] == source:
                return {**metadata, **{key: str(value) for key, value in paths.items()}}
        except (ValueError, KeyError):
            pass

    image, mask, valid = load_and_preprocess_pair(
        image_path, mask_path, args.target_h, args.target_w, native_shape=args.native_shape
    )
    count = int(mask.max())
    if count:
        # Cellpose returns [instance IDs, foreground probability, Y flow, X flow].
        # The loss needs the final three channels, not the numeric instance IDs.
        flow = dynamics.labels_to_flows([mask], device=device)[0]
        if flow.shape != (4, *mask.shape):
            raise RuntimeError(f"Unexpected Cellpose target shape: {flow.shape}")
        targets = np.asarray(flow[1:], dtype=np.float32)
        diameter = float(utils.diameters(mask)[0])
    else:
        targets = np.zeros((3, *mask.shape), dtype=np.float32)
        diameter = 0.0

    normalized = train._reshape_norm(
        [image], normalize_params={**models.normalize_default, "normalize": True}
    )[0]

    # Cache normalization once instead of repeating it on every training epoch.
    # Retain the resized raw TIFF and ID mask for inspecting cached geometry.
    folder.mkdir(parents=True, exist_ok=True)
    photometric = "rgb" if image.ndim == 3 and image.shape[-1] in (3, 4) else "minisblack"
    tifffile.imwrite(paths["image"], image, photometric=photometric, compression="deflate")
    mask_dtype = np.uint16 if count <= 65535 else np.uint32
    tifffile.imwrite(
        paths["mask"], mask.astype(mask_dtype), photometric="minisblack", compression="deflate"
    )
    tifffile.imwrite(paths["valid"], valid, photometric="minisblack", compression="deflate")

    temporary = paths["targets"].with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, image=normalized, targets=targets, valid=valid)
    temporary.replace(paths["targets"])

    metadata = {
        "source": source,
        "shape": list(mask.shape),
        "n_instances": count,
        "ignored_fraction": float(1 - valid.mean()),
        "diameter": diameter,
    }
    # Metadata is written last: its presence marks a complete reusable entry.
    atomic_json(paths["metadata"], metadata)
    return {**metadata, **{key: str(value) for key, value in paths.items()}}


def load_batch(entries):
    """Read one batch, appending validity as a temporary image channel.

    Cellpose's random_rotate_and_resize() transforms all image channels using
    EXACTLY the same affine grid, including flips, rotations and crops. Its
    target augmentation, however, consumes ONLY the last three flow channels.
    Appending valid to targets would silently drop it. Appending to the image
    instead makes it possible to recover a pixel-aligned validity mask.
    The extra channel is removed before feeding x to the network.
    """
    import numpy as np

    images, targets = [], []

    for entry in entries:
        with np.load(entry["targets"], allow_pickle=False) as data:
            image = np.asarray(data["image"], dtype=np.float32)
            valid = np.asarray(data["valid"], dtype=np.float32)
            if image.ndim != 3 or image.shape[1:] != valid.shape:
                raise ValueError(f"Invalid cache image/valid shapes: {image.shape}, {valid.shape}")
            images.append(np.concatenate((image, valid[None]), axis=0))
            targets.append(np.asarray(data["targets"], dtype=np.float32))

    return images, targets


def augment_with_validity(images, targets, args, device):
    """Augment all channels together and return image, flows, valid-pixel map.

    The validity channel is bilinearly sampled. Pixels with any nonvalid
    contribution (including synthetic padding outside the image) are excluded.
    This conservative threshold avoids accidentally supervising ignore edges.
    """
    import numpy as np
    from cellpose import transforms

    augmented, flows = transforms.random_rotate_and_resize(
        images,
        lbls=targets,
        rescale=np.ones(len(images), dtype=np.float32),
        bsize=args.bsize,
        scale_range=args.scale_range,
        device=device,
    )[:2]

    # Remove the temporary validity channel before passing images to the network.
    x = augmented[:, :-1]
    valid = (augmented[:, -1] > 0.999).to(dtype=x.dtype)
    if x.shape[-2:] != flows.shape[-2:] or x.shape[-2:] != valid.shape[-2:]:
        raise RuntimeError("Misaligned transformed image, flow targets, and validity")
    return x, flows, valid


def masked_cellpose_loss(target, prediction, valid):
    """Same loss weights as Cellpose 4.2.1.1, ignoring original-ID-1 pixels.

    The official function averages MSE for two flow channels and divides by
    two, then adds foreground BCEWithLogits. Both terms are masked here.
    """
    import torch.nn.functional as F

    if target.shape[1] != 3 or prediction.shape[1] < 3:
        raise ValueError(f"Unexpected Cellpose shapes: {target.shape}, {prediction.shape}")
    if valid.shape != target.shape[:1] + target.shape[-2:]:
        raise ValueError("Validity shape differs from targets")

    valid_count = int(valid.sum().item())
    # Keep the zero loss connected to predictions, but let callers skip this crop.
    if valid_count == 0:
        return prediction.sum() * 0.0, 0

    valid_sum = valid.sum()
    # Cellpose: MSE(predicted Y/X flow, 5 * target Y/X flow) / 2.
    squared_flow_error = (prediction[:, -3:-1] - 5.0 * target[:, -2:]).square()
    flow_loss = (squared_flow_error * valid[:, None]).sum() / (2.0 * valid_sum)
    flow_loss = flow_loss / 2.0

    # Cellpose: foreground BCE, excluding ignored pixels.
    target_foreground = (target[:, -3] > 0.5).to(dtype=prediction.dtype)
    bce = F.binary_cross_entropy_with_logits(prediction[:, -1], target_foreground, reduction="none")
    foreground_loss = (bce * valid).sum() / valid_sum

    return flow_loss + foreground_loss, valid_count


def learning_rate(epoch, epochs, maximum):
    # Ramp up over at most ten epochs, starting above zero so even a one-epoch
    # smoke test updates weights. Long runs halve the rate near their end.
    warmup = min(10, epochs)
    value = maximum * min(1.0, (epoch + 1) / warmup)

    if epochs > 300 and epoch >= epochs - 100:
        value *= 0.5 ** (1 + (epoch - (epochs - 100)) // 10)
    elif epochs > 99 and epoch >= epochs - 50:
        value *= 0.5 ** (1 + (epoch - (epochs - 50)) // 5)
    return value


def check_training_batch(model, entries, args):
    """Check forward/backward before preprocessing the rest of a large dataset."""
    import numpy as np
    import torch

    np_state, was_training = np.random.get_state(), model.net.training
    buffers = [(buffer, buffer.detach().clone()) for buffer in model.net.buffers()]
    devices = [model.device.index or 0] if model.device.type == "cuda" else []

    try:
        # Restore RNG state afterward; this check should not change the sequence
        # of augmentations/dropout used during actual training.
        with torch.random.fork_rng(devices=devices):
            model.net.train()
            model.net.zero_grad(set_to_none=True)
            images, targets = load_batch(entries)
            x, y, valid = augment_with_validity(images, targets, args, model.device)
            loss, valid_pixels = masked_cellpose_loss(y, model.net(x)[0], valid)
            if valid_pixels == 0:
                raise RuntimeError("Initial smoke-test crop has no supervised pixels")
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss in the first training-batch check")
            loss.backward()
            # There is deliberately no optimizer.step(): only test the compute path.
            log(
                f"First batch forward/backward passed: batch={len(images)}, tile={args.bsize}; no optimizer step"
            )
    finally:
        # Clear test gradients and restore buffers/mode as well as RNG state.
        model.net.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        model.net.train(was_training)
        np.random.set_state(np_state)


def train_epoch(model, optimizer, entries, args, run, epoch, global_step):
    import numpy as np
    import torch
    from tqdm import tqdm

    model.net.train()
    order = np.random.permutation(len(entries))
    total_loss, count = 0.0, 0
    progress = tqdm(
        range(0, len(order), args.batch_size), desc=f"Train epoch {epoch + 1}/{args.epochs}"
    )

    for start in progress:
        batch_entries = [entries[index] for index in order[start : start + args.batch_size]]
        images, targets = load_batch(batch_entries)

        # The cache holds full resized images; training uses random bsize crops.
        # rescale=1 disables diameter-based normalization, while scale_range
        # still permits scale jitter (0.5 means approximately 0.75x to 1.25x).
        x, y, valid = augment_with_validity(images, targets, args, model.device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model.net(x)[0]
        loss, valid_pixels = masked_cellpose_loss(y, prediction, valid)
        if valid_pixels == 0:
            log(f"Skipping batch with no supervised pixels (epoch={epoch + 1})")
            continue
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss in epoch {epoch + 1}, step {global_step + 1}")

        loss.backward()
        optimizer.step()

        # Accumulate an image-weighted epoch mean and advance the logging step.
        value = float(loss.detach().item())
        count += len(images)
        total_loss += value * len(images)
        global_step += 1
        progress.set_postfix(loss=f"{value:.4f}")
        if global_step % args.log_every == 0 or start + args.batch_size >= len(order):
            # Keep training steps monotonic across resumed epochs in the local log.
            run.log({"global_step": global_step, "train/batch_loss": value, "epoch": epoch + 1})

        # Release full-resolution arrays before loading the next batch.
        del images, targets, x, y, valid, prediction, loss, batch_entries

    if count == 0:
        raise RuntimeError("No valid supervised pixels in any training crop")
    return total_loss / count, global_step


def validation_loss(model, entries, args):
    import numpy as np
    import torch
    from tqdm import tqdm

    state, was_training = np.random.get_state(), model.net.training

    # Use the same augmented validation crops each epoch, then restore the
    # training RNG so validation does not alter the next epoch's augmentation.
    np.random.seed(42)
    model.net.eval()
    total_loss, count = 0.0, 0

    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(entries), args.batch_size), desc="Validation"):
                images, targets = load_batch(entries[start : start + args.batch_size])
                x, y, valid = augment_with_validity(images, targets, args, model.device)
                prediction = model.net(x)[0]
                loss, valid_pixels = masked_cellpose_loss(y, prediction, valid)
                if valid_pixels == 0:
                    continue

                value = float(loss.item())
                if not math.isfinite(value):
                    raise RuntimeError(
                        "Nonfinite validation loss; latest weights have already been saved"
                    )
                total_loss += value * len(images)
                count += len(images)
                del images, targets, x, y, valid, prediction, loss

        if count == 0:
            raise RuntimeError("No valid supervised pixels in any validation crop")
        return total_loss / count
    finally:
        np.random.set_state(state)
        model.net.train(was_training)


def run_training(model, train_entries, val_entries, args, run):
    import numpy as np
    import torch

    out_dir = Path(args.output_dir)
    paths = checkpoint_paths(args)
    best_path, latest_path = paths["best"], paths["latest"]
    contract = resume_contract(args, train_entries, val_entries)
    resume_file = find_resume_weights(args)
    progress, latest_stamp = None, None
    optimizer_restored = False

    # Restore weights first. File size/time metadata detects a newer checkpoint
    # whose matching progress sidecar was not written before an interruption.
    if resume_file:
        state = torch.load(str(resume_file), map_location="cpu", weights_only=True)
        model.net.load_state_dict(state)
        del state
        latest_stamp = weights_stamp(resume_file)
        if paths["progress"].is_file():
            candidate = torch.load(str(paths["progress"]), map_location="cpu", weights_only=True)
            if candidate["contract"] != contract:
                raise ValueError(
                    "Resume dataset/split/model geometry differs from the saved run; use a new output directory"
                )

            if candidate.get("weights_stamp") == latest_stamp:
                progress = candidate
            else:
                log("Resume metadata does not match latest weights; using weights-only recovery")
        log(f"LOADED LATEST WEIGHTS: {resume_file}")
    else:
        log("No latest weights found; starting from the selected pretrained Cellpose model")

    check_resume_storage(model.net, args)

    # Verify checkpoint serialization before starting expensive training work.
    preflight = best_path.with_name("serialization_preflight.pt")
    save_weights(model.net, preflight)
    restored = torch.load(str(preflight), map_location="cpu", weights_only=True)
    if set(restored) != set(model.net.state_dict()):
        raise RuntimeError("Model weight save/load preflight failed")
    del restored
    preflight.unlink()

    # Initialize the dataset diameter only for a new run.
    if not resume_file:
        model.net.diam_labels.data = torch.tensor(
            [np.mean([max(5.0, entry["diameter"]) for entry in train_entries])],
            device=model.device,
            dtype=model.net.diam_labels.dtype,
        )

    optimizer = torch.optim.AdamW(
        model.net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    summary = {
        "best_selection": "lowest validation loss on deterministic augmented crops",
        "heldout_role": "validation: used for best-model selection",
        "history": [],
    }

    # Prefer full restart metadata, then fall back to weights and JSON epoch history.
    if progress:
        summary = progress["summary"]
        start_epoch = int(progress["completed_epoch"])
        global_step = int(progress["global_step"])
        pending = bool(progress["validation_pending"])
        supplied = getattr(args, "start_epoch", None)
        if supplied is not None and supplied != start_epoch:
            raise ValueError("start_epoch conflicts with the saved resume metadata")

        if paths["optimizer"].is_file() and not getattr(args, "weights_only", False):
            saved = torch.load(str(paths["optimizer"]), map_location="cpu", weights_only=True)
            if (
                saved.get("weights_stamp") == latest_stamp
                and saved.get("completed_epoch") == start_epoch
                and saved.get("global_step") == global_step
            ):
                optimizer.load_state_dict(saved["optimizer_state"])
                optimizer_restored = True
            del saved

        restore_rng_state(progress["rng_state"], model.device)
    elif resume_file:
        metrics_path = out_dir / "training_metrics.json"
        if metrics_path.is_file():
            summary = json.loads(metrics_path.read_text())
        history = summary.get("history", [])
        recorded = int(history[-1]["epoch"]) if history else 0
        supplied = getattr(args, "start_epoch", None)
        if history and supplied is not None and supplied != recorded:
            raise ValueError("start_epoch conflicts with the recorded epoch history")
        start_epoch = supplied if supplied is not None else recorded
        global_step = int(history[-1].get("global_step", 0)) if history else 0
        pending = bool(history and "validation_loss" not in history[-1])
    else:
        start_epoch, global_step, pending = 0, 0, False

    # Track the best validation result across both new and resumed epochs.
    history = summary.setdefault("history", [])
    best_loss = float(summary.get("best_validation_loss", math.inf))
    summary.update(
        best_model=str(best_path),
        latest_model=str(latest_path),
        n_train=len(train_entries),
        n_validation=len(val_entries),
    )
    if resume_file and not optimizer_restored:
        log("WEIGHTS-ONLY RESUME: previous AdamW state is unavailable; using a fresh optimizer")
    for group in optimizer.param_groups:
        group["weight_decay"] = args.weight_decay
    if resume_file and resume_file.resolve() != latest_path.resolve():
        # A user-supplied checkpoint may be outside this run's output directory.
        save_weights(model.net, latest_path)
        latest_stamp = weights_stamp(latest_path)

    run.summary.update(
        {
            "best_model": str(best_path),
            "latest_model": str(latest_path),
            "resumed_from": str(resume_file) if resume_file else "pretrained",
            "resume_completed_epoch": start_epoch,
            "optimizer_restored": optimizer_restored,
        }
    )
    if resume_file:
        log(
            f"RESUME: {start_epoch} training epochs completed; target={args.epochs}; optimizer restored={optimizer_restored}"
        )

    def save_progress(validation_pending, save_optimizer=False):
        # Record the weight file's size/time alongside epoch and step counts so
        # stale sidecars are rejected without reading the whole weight file.
        snapshot = {
            "format_version": 2,
            "weights_stamp": latest_stamp,
            "completed_epoch": start_epoch,
            "global_step": global_step,
            "validation_pending": validation_pending,
            "contract": contract,
            "rng_state": capture_rng_state(model.device),
            "summary": summary,
        }
        atomic_torch_save(snapshot, paths["progress"])
        if save_optimizer and not getattr(args, "weights_only", False):
            atomic_torch_save(
                {
                    "weights_stamp": latest_stamp,
                    "completed_epoch": start_epoch,
                    "global_step": global_step,
                    "optimizer_state": optimizer.state_dict(),
                },
                paths["optimizer"],
            )
        atomic_json(out_dir / "training_metrics.json", summary)

    def finish_validation(row):
        nonlocal best_loss
        run.summary["phase"] = "validation"
        val_loss = validation_loss(model, val_entries, args)
        row["validation_loss"] = val_loss
        if val_loss < best_loss or not best_path.is_file():
            save_weights(model.net, best_path)
            best_loss = val_loss
            summary.update(best_epoch=row["epoch"], best_validation_loss=best_loss)
            run.summary.update({"best_epoch": row["epoch"], "best_validation_loss": best_loss})
            log(f"BEST SAVED: epoch={row['epoch']}, validation loss={best_loss:.6f}")

        save_progress(False)
        run.log(
            {
                "epoch": row["epoch"],
                "global_step": global_step,
                "train/epoch_loss": row["train_loss"],
                "val/loss": val_loss,
                "train/learning_rate": row["learning_rate"],
            }
        )
        log(
            f"Epoch {row['epoch']}/{args.epochs}: train loss={row['train_loss']:.6f}, validation loss={val_loss:.6f}"
        )

    if pending:
        # Latest weights were saved after training but validation was interrupted.
        # Finish that validation before advancing; do not retrain the completed epoch.
        log(f"Finishing pending validation for epoch {start_epoch}")
        finish_validation(history[-1])
    elif resume_file and (not best_path.is_file() or not math.isfinite(best_loss)):
        # Older weights-only files may lack best-model metadata. Establish a
        # baseline so the resumed run cannot replace a better model blindly.
        if best_path.is_file():
            state = torch.load(str(best_path), map_location="cpu", weights_only=True)
            model.net.load_state_dict(state)
            del state
        best_loss = validation_loss(model, val_entries, args)
        if not best_path.is_file():
            save_weights(model.net, best_path)
        summary.update(best_validation_loss=best_loss)
        state = torch.load(str(latest_path), map_location="cpu", weights_only=True)
        model.net.load_state_dict(state)
        del state
        save_progress(False, save_optimizer=True)

    # Train to the total epoch target, saving each completed epoch before validation.
    for epoch in range(start_epoch, args.epochs):
        lr = learning_rate(epoch, args.epochs, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        run.summary["phase"] = "training"
        loss, global_step = train_epoch(
            model, optimizer, train_entries, args, run, epoch, global_step
        )
        start_epoch = epoch + 1
        row = {
            "epoch": start_epoch,
            "global_step": global_step,
            "train_loss": loss,
            "learning_rate": lr,
        }
        history.append(row)

        # Save before validation, with optimizer/RNG sidecars for the next restart.
        save_weights(model.net, latest_path)
        latest_stamp = weights_stamp(latest_path)
        save_progress(True, save_optimizer=True)
        log(f"LATEST AND RESUME STATE SAVED before validation: {latest_path}")
        finish_validation(row)

    # Export the model selected by validation loss, not the final epoch.
    # Submission inference does not need an evaluation module or dataset labels.
    if not best_path.is_file():
        raise RuntimeError("No best model is available; check resume and epoch settings")

    model.net.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    atomic_json(out_dir / "training_metrics.json", summary)
    return summary


class LocalRun:
    """Keep the trainer's progress hooks without an external logging account."""

    def __init__(self, output_dir):
        self.summary = {}
        self.path = Path(output_dir) / "progress.jsonl"

    def log(self, values):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(values, allow_nan=False) + "\n")


def train(records, data_settings, settings, cellpose_settings, checkpoints, device_name):
    """Train Stage 1 and publish its best plain Cellpose state_dict as fiber.pt."""
    import torch
    from cellpose import models

    if importlib.metadata.version("cellpose") != CELLPOSE_VERSION:
        raise RuntimeError(f"Cellpose training requires cellpose=={CELLPOSE_VERSION}")

    if cellpose_settings.model_type not in models.MODEL_NAMES:
        raise ValueError(f"Unknown Cellpose initialization model: {cellpose_settings.model_type}")

    if settings.epochs < 1 or settings.batch_size < 1 or settings.lr <= 0:
        raise ValueError("Training epochs, batch size, and learning rate must be positive")

    if settings.resume not in {"auto", "required", "never"}:
        raise ValueError("Fiber resume must be 'auto', 'required', or 'never'")

    device = resolve_device(device_name)
    set_global_seed(data_settings.seed)

    # The core trainer uses a flat settings object. Keep that adaptation here;
    # the public configuration remains grouped into data/model/training sections.
    args = SimpleNamespace(
        **asdict(settings),
        seed=data_settings.seed,
        native_shape=tuple(data_settings.native_shape),
        model_type=cellpose_settings.model_type,
        model_name="fiber",
        target_h=cellpose_settings.shape[0],
        target_w=cellpose_settings.shape[1],
        bsize=cellpose_settings.bsize,
        eval_batch_size=cellpose_settings.batch_size,
        tile_overlap=cellpose_settings.tile_overlap,
        niter=cellpose_settings.niter,
        flow_threshold=cellpose_settings.flow_threshold,
        cellprob_threshold=cellpose_settings.cellprob_threshold,
        min_size=cellpose_settings.min_size,
        resume_from=None,
        start_epoch=None,
    )

    selected = {
        split: [(row["image_path"], row["label_path"]) for row in records if row["split"] == split]
        for split in ("train", "val")
    }
    config = {
        "data": asdict(data_settings),
        "cellpose": asdict(cellpose_settings),
        "training": asdict(settings),
        "split_assignments": split_assignments(records),
        "cellpose_version": CELLPOSE_VERSION,
        "ignore_policy": "original_id_1_excluded_v1",
    }
    # JSON stores tuples as lists. Normalize the shape before comparing an
    # in-memory configuration with the previous run's serialized configuration.
    config["cellpose"]["shape"] = list(cellpose_settings.shape)

    with training_lock(args.output_dir):
        resume_file = find_resume_weights(args)
        paths = checkpoint_paths(args)
        config_path = Path(args.output_dir) / "config.json"

        if config_path.is_file():
            previous = json.loads(config_path.read_text())

            if (
                previous["cellpose"] != config["cellpose"]
                or previous.get("split_assignments", config["split_assignments"])
                != config["split_assignments"]
            ):
                raise ValueError(
                    "Saved Cellpose configuration/split differs; choose a new fiber output_dir"
                )

        if not resume_file and any(path.exists() for path in paths.values()):
            raise FileExistsError(
                "Fiber run already has checkpoints; enable resume or use a new output_dir"
            )

        if resume_file and args.init_weights is not None:
            raise ValueError("init_weights starts a new run; omit it when resuming")

        if args.init_weights is not None and not Path(args.init_weights).is_file():
            raise FileNotFoundError(args.init_weights)

        write_json(config_path, config)
        initialization = str(resume_file or args.init_weights or args.model_type)
        model = models.CellposeModel(
            pretrained_model=initialization,
            device=device,
            gpu=device.type == "cuda",
            use_bfloat16=False,
        )

        if model.backbone == "sam_vitl" and args.bsize != 256:
            raise ValueError("CPSAM requires CELLPOSE.bsize=256")

        # Prepare and release each original image in sequence. Cached targets
        # hold normalized pixels, fiber flows, and the aligned artifact-validity map.
        entries = {}

        for split, pairs in selected.items():
            entries[split] = []

            for index, pair in enumerate(pairs, start=1):
                entries[split].append(prepare_cached_pair(pair, args, device))
                log(f"Cached {split} {index}/{len(pairs)}: {pair[0].name}")

                if split == "train" and index == min(args.batch_size, len(pairs)):
                    check_training_batch(model, entries[split], args)

        if any(entry["n_instances"] == 0 for entry in entries["train"]):
            raise ValueError("A training image has no fiber instances after resizing")

        run = LocalRun(args.output_dir)
        summary = run_training(model, entries["train"], entries["val"], args, run)

        # Publish only after training/validation succeeds. The local latest/best
        # and optimizer sidecars remain in runs/fiber for subsequent resumes.
        destination = Path(checkpoints.directory) / checkpoints.fiber_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(summary["best_model"], temporary)
        temporary.replace(destination)
        log(f"Published best Cellpose weights: {destination}")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return destination
