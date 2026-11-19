"""Read v5 safetensors and older PyTorch checkpoints into a configured MoR model."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def parameter_aliases(model):
    first = {}
    aliases = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        canonical = first.setdefault(id(param), name)
        if canonical != name:
            aliases[name] = canonical
    return aliases


def expand_shared_weights(model, weights):
    aliases = parameter_aliases(model)
    if all(alias in weights and canonical in weights for alias, canonical in aliases.items()):
        return weights
    expanded = weights.copy()
    if hasattr(weights, "_metadata"):
        expanded._metadata = weights._metadata
    for alias, canonical in aliases.items():
        if alias in expanded:
            expanded.setdefault(canonical, expanded[alias])
    for alias, canonical in aliases.items():
        if canonical in expanded:
            expanded.setdefault(alias, expanded[canonical])
    return expanded


def load_model_weights(model, directory):
    directory = Path(directory)
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = directory / filename
        index = directory / (filename + ".index.json")
        if path.is_file():
            paths = [path]
        elif index.is_file():
            shards = json.loads(index.read_text())["weight_map"].values()
            paths = [directory / shard for shard in dict.fromkeys(shards)]
        else:
            continue
        weights = {}
        for shard in paths:
            if filename.endswith(".safetensors"):
                weights.update(load_file(str(shard), device="cpu"))
            else:
                weights.update(torch.load(shard, map_location="cpu", weights_only=True))
        model.load_state_dict(expand_shared_weights(model, weights))
        return
    raise FileNotFoundError(f"No model weights found in {directory}")
