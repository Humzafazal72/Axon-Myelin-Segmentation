"""Train the notebook's U-Net on native crops and ground-truth fiber masks."""

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import read_array
from ..models.unet import SmallUNet
from ..preprocessing import (
    expand_box,
    fiber_instances,
    instance_boxes,
    normalize_crop,
    pad_inputs,
    validate_pair,
)
from ..utils import resolve_device, save_torch, seed_everything, sha256_file, write_csv, write_json


def prepare_crops(records, data_settings, axon_settings, output_dir):
    """Write native crops one micrograph at a time, skipping incomplete GT pairs.

    This matches the notebook's target: 0=context, 1=axon, 2=myelin for ONE fiber.
    Other fibers inside the context are image pixels, never positive mask input.
    Test images are not loaded or used to select the best checkpoint.
    """
    crop_dir = Path(output_dir) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    frames = {"train": [], "val": []}
    issues = []

    for row in tqdm(
        [row for row in records if row["split"] in frames], desc="Prepare native axon crops"
    ):
        image, labels = read_array(row["image_path"]), read_array(row["label_path"])
        validate_pair(image, labels, data_settings.native_shape)
        observed = set(np.unique(labels).tolist())
        boxes = instance_boxes(fiber_instances(labels))

        for axon_id, tight_box in boxes.items():
            if axon_id not in observed or axon_id + 1 not in observed:
                issues.append(
                    dict(
                        micrograph_id=row["micrograph_id"],
                        axon_id=axon_id,
                        reason="missing axon or myelin partner",
                    )
                )
                continue

            box = expand_box(tight_box, image.shape, axon_settings.context)
            local = labels[box]
            target = (local == axon_id).astype(np.uint8)
            target[local == axon_id + 1] = 2
            path = crop_dir / f"{row['micrograph_id']}_{axon_id}.npz"
            np.savez(path, image=image[box], target=target)

            y, x = box
            frames[row["split"]].append(
                dict(
                    micrograph_id=row["micrograph_id"],
                    axon_id=axon_id,
                    crop_path=str(path),
                    crop_h=y.stop - y.start,
                    crop_w=x.stop - x.start,
                    max_side=max(y.stop - y.start, x.stop - x.start),
                )
            )

    write_csv(
        Path(output_dir) / "annotation_issues.csv", issues, ["micrograph_id", "axon_id", "reason"]
    )
    all_crops = [dict(row, split=split) for split, rows in frames.items() for row in rows]
    write_csv(
        Path(output_dir) / "crop_index.csv",
        all_crops,
        ["micrograph_id", "axon_id", "crop_path", "crop_h", "crop_w", "max_side", "split"],
    )

    if any(not rows for rows in frames.values()):
        raise ValueError("Both train and val must contain complete annotated fibers")

    return frames


class FiberCropDataset(Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with np.load(self.rows[index]["crop_path"], allow_pickle=False) as crop:
            image = normalize_crop(crop["image"])
            target = crop["target"]

        if self.augment:
            # Apply the same spatial transform to pixels and compartments. The
            # intensity augmentation intentionally affects only the TEM pixels.
            rotation = np.random.randint(4)
            image, target = np.rot90(image, rotation), np.rot90(target, rotation)

            for axis in (0, 1):
                if np.random.rand() < 0.5:
                    image, target = np.flip(image, axis), np.flip(target, axis)

            image = np.clip(
                (image - 0.5) * np.random.uniform(0.9, 1.1) + 0.5 + np.random.uniform(-0.05, 0.05),
                0,
                1,
            )
            image = image ** np.random.uniform(0.85, 1.15)

        return dict(image=image.copy(), fiber=(target > 0).copy(), axon=(target == 1).copy())


def collate_crops(samples):
    inputs, shapes = pad_inputs(
        [sample["image"] for sample in samples], [sample["fiber"] for sample in samples]
    )
    height, width = inputs.shape[-2:]
    targets = []

    for sample, (h, w) in zip(samples, shapes):
        target = torch.from_numpy(sample["axon"].astype(np.float32))[None]
        targets.append(F.pad(target, (0, width - w, 0, height - h)))

    return dict(x=inputs, y=torch.stack(targets), fiber=inputs[:, 1:2])


def make_batches(rows, settings):
    """Bucket by longest side as in the notebook; never resize a large crop."""
    buckets = {}

    for index, row in enumerate(rows):
        buckets.setdefault((row["max_side"] - 1) // 128, []).append(index)

    batches = []

    for bucket in sorted(buckets):
        indices = buckets[bucket]
        largest = max(rows[index]["max_side"] for index in indices)
        size = 1 if largest > settings.large_crop_side else settings.batch_size
        batches.extend(indices[start : start + size] for start in range(0, len(indices), size))

    return batches


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def masked_loss(logits, target, fiber):
    """Notebook loss: per-fiber BCE + soft Dice, supervised only inside GT fiber."""
    logits, target, mask = logits.float(), target.float(), fiber.float()
    dims = (1, 2, 3)
    count = mask.sum(dims).clamp_min(1)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum(
        dims
    ) / count
    probabilities, truth = logits.sigmoid() * mask, target * mask
    dice = (2 * (probabilities * truth).sum(dims) + 1e-6) / (
        probabilities.sum(dims) + truth.sum(dims) + 1e-6
    )

    return (bce + 1 - dice).mean()


def run_epoch(model, loader, optimizer, scaler, device, amp, training):
    model.train(training)
    total, count = 0.0, 0

    for batch in tqdm(loader, desc="Train" if training else "Validation", leave=False):
        x, target, fiber = [batch[key].to(device, non_blocking=True) for key in ("x", "y", "fiber")]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), torch.autocast(device_type=device.type, enabled=amp):
            loss = masked_loss(model(x), target, fiber)

        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite axon training/validation loss")

        if training:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        total += loss.item() * len(x)
        count += len(x)

    return total / count


def capture_rng(device, generators):
    numpy_state = np.random.get_state()

    # Convert NumPy's uint32 state array to a plain list so torch.load can use
    # weights_only=True, without requiring pickle allowlists for NumPy classes.
    return dict(
        python=random.getstate(),
        numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        loaders={name: generator.get_state() for name, generator in generators.items()},
    )


def restore_rng(state, device, generators):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])

    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)

    for name, generator in generators.items():
        generator.set_state(state["loaders"][name])


def train(records, data_settings, settings, axon_settings, checkpoints, device_name):
    """Train independently of Cellpose, then publish the best axon checkpoint."""
    if settings.epochs < 1 or settings.batch_size < 1 or settings.patience < 1 or settings.lr <= 0:
        raise ValueError("Epochs, batch size, patience, and learning rate must be positive")

    device = resolve_device(device_name)
    seed_everything(data_settings.seed)
    amp = axon_settings.amp and device.type == "cuda"
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = output_dir / "last.pt", output_dir / "best.pt"
    contract = dict(
        split_sha256=sha256_file(data_settings.split_file),
        native_shape=list(data_settings.native_shape),
        base_channels=axon_settings.base_channels,
        context=axon_settings.context,
    )

    if not settings.resume and (latest_path.exists() or best_path.exists()):
        raise FileExistsError(
            "Axon run already has checkpoints; enable resume or choose a new output_dir"
        )

    # Verify the saved contract before rewriting caches or the run configuration.
    saved = (
        torch.load(latest_path, map_location="cpu", weights_only=True)
        if latest_path.is_file()
        else None
    )

    if saved is not None and saved.get("contract") != contract:
        raise ValueError("Saved axon geometry/model/split differs; choose a new axon output_dir")

    if saved is not None and not best_path.is_file():
        raise FileNotFoundError(f"Cannot resume without the run's best checkpoint: {best_path}")

    write_json(
        output_dir / "config.json",
        dict(
            data=asdict(data_settings),
            training=asdict(settings),
            axon=asdict(axon_settings),
            contract=contract,
        ),
    )
    frames = prepare_crops(records, data_settings, axon_settings, output_dir)
    generators = {name: torch.Generator().manual_seed(data_settings.seed) for name in frames}
    loaders = {
        name: DataLoader(
            FiberCropDataset(rows, augment=name == "train"),
            batch_sampler=make_batches(rows, settings),
            collate_fn=collate_crops,
            num_workers=settings.num_workers,
            pin_memory=amp,
            persistent_workers=settings.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generators[name],
        )
        for name, rows in frames.items()
    }
    model = SmallUNet(axon_settings.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_loss, bad_epochs, history = 0, float("inf"), 0, []

    if saved is not None:
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        scheduler.load_state_dict(saved["scheduler_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_epoch, best_loss, bad_epochs = saved["epoch"], saved["best_loss"], saved["bad_epochs"]
        history = saved["history"]
        restore_rng(saved["rng"], device, generators)
        print(f"Resuming axon training after epoch {start_epoch}", flush=True)
        del saved

    for epoch in range(start_epoch + 1, settings.epochs + 1):
        if bad_epochs >= settings.patience:
            break

        # Start from canonical batches before shuffling so a resumed run does
        # not depend on the previous process's mutable list ordering.
        loaders["train"].batch_sampler[:] = make_batches(frames["train"], settings)
        random.shuffle(loaders["train"].batch_sampler)
        train_loss = run_epoch(model, loaders["train"], optimizer, scaler, device, amp, True)
        val_loss = run_epoch(model, loaders["val"], optimizer, scaler, device, amp, False)
        scheduler.step(val_loss)
        improved = val_loss < best_loss
        best_loss, bad_epochs = (val_loss, 0) if improved else (best_loss, bad_epochs + 1)
        history.append(
            dict(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        state = dict(
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict(),
            val_loss=val_loss,
            best_loss=best_loss,
            bad_epochs=bad_epochs,
            history=history,
            contract=contract,
            rng=capture_rng(device, generators),
        )

        if improved:
            save_torch(best_path, state)

        save_torch(latest_path, state)
        write_csv(
            output_dir / "training_history.csv",
            history,
            ["epoch", "train_loss", "val_loss", "learning_rate"],
        )
        print(
            f"Axon epoch {epoch}/{settings.epochs}: train={train_loss:.6f}, val={val_loss:.6f}",
            flush=True,
        )

    if not best_path.is_file():
        raise RuntimeError("No best axon checkpoint was produced")

    # The published file keeps the notebook's 'model_state' convention, with
    # small architecture/geometry metadata instead of training optimizer states.
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    destination = Path(checkpoints.directory) / checkpoints.axon_filename
    save_torch(
        destination,
        dict(
            model_state=best["model_state"],
            epoch=best["epoch"],
            val_loss=best["val_loss"],
            contract=contract,
        ),
    )
    print(f"Published best axon weights: {destination}", flush=True)

    return destination
