"""Select committed training checkpoints and resolve evaluation weights."""
import json
import shutil
import tempfile
from pathlib import Path
import warnings

from omegaconf import OmegaConf


def same_recipe(saved, requested):
    def recipe(cfg):
        values = OmegaConf.to_container(cfg, resolve=True)
        for key in ("auto_resume", "resume_from_checkpoint", "resume_step", "stop_steps", "ablation_checkpoint_step",
                    "plot_states", "plot_hidden_samples", "logging_steps"):
            values.pop(key, None)
        return values
    return recipe(saved) == recipe(requested)


def weight_files(directory):
    for name in ("model.safetensors", "pytorch_model.bin"):
        index = directory / (name + ".index.json")
        if (directory / name).is_file():
            return [directory / name]
        if index.is_file():
            shards = json.loads(index.read_text())["weight_map"].values()
            return [directory / shard for shard in set(shards)]
    return []


def has_weights(directory):
    try:
        files = weight_files(directory)
        return bool(files) and all(path.is_file() and path.stat().st_size for path in files)
    except (ValueError, KeyError, OSError):
        return False


def checkpoint_dirs(directory):
    return sorted((p for p in directory.glob("checkpoint-*") if p.is_dir() and p.name[11:].isdigit()),
                  key=lambda p: int(p.name[11:]), reverse=True)


def checkpoint_step(directory):
    return int(json.loads((directory / "trainer_state.json").read_text())["global_step"])


def complete_checkpoint(path, routed=False, world_size=None, fp16=False):
    try:
        if (path / ".saving").exists() or checkpoint_step(path) != int(path.name[11:]):
            return False
        marker = path / "resume.json"
        if marker.is_file():
            manifest = json.loads(marker.read_text())
            return (manifest["version"] == 1 and manifest["global_step"] == checkpoint_step(path)
                    and bool(manifest["files"]) and all(
                        (path / name).stat().st_size == size for name, size in manifest["files"].items()))
        # Pre-manifest checkpoints. The old dataset.pt contains a prefetched position.
        # Trainer reproduces their consumed data via deterministic batch replay.
        if not has_weights(path):
            return False
        rank_files = list(path.glob("rng_state_*.pth"))
        count = world_size or max(1, len(rank_files))
        if rank_files and len(rank_files) != count:
            return False
        rng = ["rng_state.pth"] if count == 1 else [f"rng_state_{i}.pth" for i in range(count)]
        if routed:
            rng.append("router_steps.json")
        if not all((path / name).is_file() and (path / name).stat().st_size for name in rng):
            return False
        # Older ZeRO saves can only be identified by their native model/optimizer partitions.
        ds = path / f"global_step{checkpoint_step(path)}"
        if ds.is_dir():
            return bool(list(ds.glob("*model_states.pt"))) and len(list(ds.glob("*optim_states.pt"))) == count
        return all((path / name).is_file() and (path / name).stat().st_size
                   for name in ["optimizer.pt", "scheduler.pt"] + (["scaler.pt"] if fp16 else []))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_checkpoint(directory, **kwargs):
    candidates = checkpoint_dirs(directory)
    for candidate in candidates:
        if complete_checkpoint(candidate, **kwargs):
            return candidate
        warnings.warn(f"Ignoring incomplete checkpoint {candidate}")
    if candidates:
        raise ValueError(f"No complete training checkpoint in {directory}")
    return None


def evaluation_weights(directory):
    """Prefer the most recent scored step, including checkpoints of interrupted runs."""
    for checkpoint in checkpoint_dirs(directory):
        if complete_checkpoint(checkpoint) and has_weights(checkpoint):
            try:
                if has_weights(directory) and checkpoint_step(directory) > checkpoint_step(checkpoint):
                    return directory
            except (FileNotFoundError, ValueError, KeyError):
                pass
            return checkpoint
    if has_weights(directory):
        return directory
    raise FileNotFoundError(f"No saved evaluation weights in {directory}")


def weights_at_step(directory, step):
    """Return only an evaluation snapshot, committed checkpoint or final export at this step."""
    snapshot = directory / f"ablation-step-{step}"
    checkpoint = directory / f"checkpoint-{step}"
    for path in (snapshot, checkpoint, directory):
        if path == checkpoint and not complete_checkpoint(path):
            continue
        try:
            if has_weights(path) and checkpoint_step(path) == step:
                return path
        except (OSError, ValueError, KeyError):
            continue
    raise FileNotFoundError(f"No saved weights at optimizer step {step} in {directory}")


def preserve_ablation_weights(directory, source, step):
    """Publish evaluation weights outside Trainer's rotating checkpoint directories."""
    snapshot = directory / f"ablation-step-{step}"
    if source == snapshot:
        return
    if not has_weights(source) or checkpoint_step(source) != step:
        raise ValueError(f"Expected evaluation weights from step {step} in {source}")
    files = [*weight_files(source), *source.glob("*.index.json"), source / "trainer_state.json"]
    with tempfile.TemporaryDirectory(dir=directory, prefix=".ablation-") as temp:
        staging = Path(temp) / "weights"
        staging.mkdir()
        for file in files:
            target = staging / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            # A final export or retried checkpoint can overwrite a file in place.
            # Keep the snapshot independent of mutable files in either source.
            shutil.copyfile(file, target)
        if snapshot.exists():
            if has_weights(snapshot) and checkpoint_step(snapshot) == step:
                return
            raise ValueError(f"Existing ablation snapshot is incomplete: {snapshot}")
        staging.rename(snapshot)
