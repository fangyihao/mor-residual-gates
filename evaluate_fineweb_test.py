from util.checkpoint import load_model_weights
from util.resume import evaluation_weights
import os
import hashlib
from itertools import islice
import yaml
import copy
import math
import warnings
from pathlib import Path
from paths import SAVE_DIR, DATA_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
from typing import Callable, List, Optional, Tuple, Union, Dict # Added Dict
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import argparse # Added for command-line arguments
import json # Added for JSON output

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import transformers
from datasets import load_dataset, interleave_datasets
from accelerate import PartialState
from transformers import AutoTokenizer, AutoConfig
from transformers.utils import ModelOutput
from transformers.processing_utils import Unpack

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from lm_dataset.data_preprocessing import AddLabels, RemoveIndex
from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from model.relaxation.util import relax_weight_sharing
from util.tokenizer import load_tokenizer_from_config
from util.config import preprocess_config, training_config_path
from util.misc import get_torch_dtype, get_latest_checkpoint_path
from util.device import select_npu

# LM_DATASETS definition (same as in the original code)
LM_DATASETS = {
    "slimpajama": {"path": f"{DATA_DIR}/slimpajama", "split": "train"},
    "slimpajama_chunk1": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_chunk1/*.jsonl", "split": "train"},
    "slimpajama_test": {"path": "json", "data_files": f"{DATA_DIR}/slimpajama_test/*.jsonl", "split": "train"},
    "pg19": {"path": f"emozilla/pg19-test", "split": "test", "trust_remote_code": "True"},
    "cosmopedia": {"path": f"{DATA_DIR}/cosmopedia-v2", "split": "train"},
    "fineweb_edu": {"path": f"{DATA_DIR}/fineweb-edu-dedup", "split": "train"},
    "fineweb_test": {"path": f"{DATA_DIR}/fineweb-test", "split": "train"},
    "python_edu": {"path": f"{DATA_DIR}/python-edu", "split": "train"},
    "open_web_math": {"path": f"{DATA_DIR}/open-web-math", "split": "train"},
    "math_code_pile": {"path": f"{DATA_DIR}/math-code-pile", "split": "train"},
    "starcoderdata": {"path": f"{DATA_DIR}/starcoderdata", "split": "train"},  # "data_dir": "python",
    "finemath": {"path": f"{DATA_DIR}/finemath", "split": "train"},  # "name": "finemath-4plus",
}


def load_dataset_from_config(cfg, tokenizer, continuous=True):
    dataset_name_str = cfg.dataset # cfg.dataset might be a string or already a list
    if isinstance(dataset_name_str, str):
        dataset_names = [ds.strip() for ds in dataset_name_str.split(',')]
    else: # Assuming it's already a list-like OmegaConf object
        dataset_names = [str(ds).strip() for ds in dataset_name_str]

    if len(dataset_names) > 1:
        assert all(ds in LM_DATASETS for ds in dataset_names), "Only LM datasets can be combined"
        assert "weights" in cfg, "When combining datasets, weights must be provided"
        weights_str = cfg.weights
        if isinstance(weights_str, str):
            weights = [float(w) for w in weights_str.split(',')]
        else: # Assuming it's already a list-like OmegaConf object of numbers
            weights = [float(w) for w in weights_str]
        assert len(dataset_names) == len(weights), "Number of weights must match number of datasets"

    if all(ds in LM_DATASETS for ds in dataset_names):
        dataset_type = "lm"
        loaded_datasets = []
        for ds_name in dataset_names:
            _dataset = load_dataset(**LM_DATASETS[ds_name], streaming=True)
            if ds_name == "starcoderdata":
                _dataset = _dataset.rename_column("content", "text") # Ensure renaming happens before appending
            loaded_datasets.append(_dataset)

        if len(loaded_datasets) == 1:
            final_dataset = loaded_datasets[0]
        else:
            # Ensure weights are correctly parsed as floats for interleave_datasets
            probabilities = [float(w) for w in cfg.weights.split(',')] if isinstance(cfg.weights, str) else [float(w) for w in cfg.weights]
            final_dataset = interleave_datasets(loaded_datasets, probabilities=probabilities, seed=42)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    transforms = [
        AddLabels(),
        RemoveIndex(),
    ]

    if dataset_type == "lm":
        return LanguageModelingDataset(final_dataset, tokenizer,
                                    max_length=cfg.max_length, continuous=continuous,
                                    transforms=transforms,
                                    global_shuffling=cfg.get("global_shuffling", False),
                                    local_shuffling=cfg.get("local_shuffling", False),
                                    add_bos_token=cfg.get("add_bos_token", False),)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def evaluation_state():
    device = select_npu()
    if device is not None:
        return PartialState(backend="hccl") if "LOCAL_RANK" in os.environ else PartialState()
    return PartialState(cpu=not torch.cuda.is_available())


def load_scoring_model(cfg, run_dir, checkpoint_dir):
    """Build the saved architecture before loading native or residual-gated weights."""
    model = load_model_from_config(cfg)

    lora_init_dict = None # Initialize lora_init_dict

    if cfg.recursive.get("enable"):
        # Assuming cfg.model is a DictConfig, access name with .name or adjust as needed
        model_strategy_key = cfg.model.name if hasattr(cfg.model, "name") else cfg.model
        model, lora_init_dict = SHARING_STRATEGY[model_strategy_key](cfg, model)

    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)

    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        model = relax_weight_sharing(cfg, model, lora_init_dict=lora_init_dict)

        if checkpoint_dir != run_dir:
            load_model_weights(model.get_base_model(), checkpoint_dir)
        elif cfg.get("resume_from_checkpoint"): # Check if resume_from_checkpoint is True
            latest_checkpoint_path_obj = get_latest_checkpoint_path(cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None)
            if latest_checkpoint_path_obj:
                load_model_weights(model.get_base_model(), latest_checkpoint_path_obj)

    if "mor" in cfg and cfg.mor.get("enable"):
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        else:
            raise ValueError(f"Unknown MoR type {cfg.mor.type}.")

    load_model_weights(model, checkpoint_dir)
    return model


def fingerprint_prefix(dataset, size, digest):
    for sample in islice(dataset, size):
        ids = sample["input_ids"].detach().cpu().numpy().astype("<i8", copy=False)
        digest.update(len(ids).to_bytes(8, "little"))
        digest.update(ids.tobytes())
        yield sample


def evaluate_model(exp_name: str, global_sample_number: int, state=None, attn_implementation=None, strict=False, latest_saved=False, weights_dir=None, data_manifest=None) -> Optional[float]:
    """Score the same packed sample prefix for any number of data-parallel workers."""
    if state is None:
        state = evaluation_state()
    print(f"Evaluating experiment: {exp_name}")
    cfg_path = training_config_path(os.path.join(SAVE_DIR, "pretrain", exp_name), "conf/pretrain")
    if not os.path.exists(cfg_path):
        if strict or state.num_processes > 1:
            raise FileNotFoundError(cfg_path)
        print(f"Configuration file not found for {exp_name} at {cfg_path}. Skipping.")
        return None

    cfg = OmegaConf.load(cfg_path)
    cfg = preprocess_config(cfg) # This should handle setting default max_length if not present

    cfg.resume_from_checkpoint = False
    if attn_implementation is not None:
        cfg.attn_implementation = attn_implementation
    elif state.device.type == "npu" and cfg.get("attn_implementation", "flash_attention_2") == "flash_attention_2":
        cfg.attn_implementation = "sdpa"

    run_dir = Path(SAVE_DIR, "pretrain", exp_name)
    checkpoint_dir = Path(weights_dir) if weights_dir else (evaluation_weights(run_dir) if latest_saved else run_dir)
    try:
        model = load_scoring_model(cfg, run_dir, checkpoint_dir)
        print(f"Loaded experiment weights from {checkpoint_dir}")
    except (FileNotFoundError, RuntimeError) as exc:
        if strict or state.num_processes > 1:
            raise
        warnings.warn(f"Skipping {exp_name}: {exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    # Load dataset
    cfg_for_dataset = deepcopy(cfg)
    cfg_for_dataset.dataset = "fineweb_test"

    tokenizer = load_tokenizer_from_config(cfg_for_dataset)
    dataset = load_dataset_from_config(cfg_for_dataset, tokenizer, continuous=data_manifest is None)
    if data_manifest is not None and state.is_main_process:
        digest = hashlib.sha256()
        dataset = fingerprint_prefix(dataset, global_sample_number, digest)

    device = state.device
    model.to(device, dtype=get_torch_dtype(cfg))
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    evaluated_samples = 0
    state.print(f"Scoring {global_sample_number} packed samples across {state.num_processes} workers.")
    samples = islice(dataset, state.process_index, global_sample_number, state.num_processes)
    with torch.no_grad():
        for sample in samples:
            try:
                output = model(
                    input_ids=sample["input_ids"].unsqueeze(0).to(device),
                    attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
                    labels=sample["labels"].unsqueeze(0).to(device),
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                total_loss += output.loss.float()
                evaluated_samples += 1
            except Exception as exc:
                if strict or state.num_processes > 1:
                    raise
                warnings.warn(f"Error scoring a sample for {exp_name}: {exc}")
    count = torch.tensor(evaluated_samples, device=device, dtype=torch.long)
    if state.num_processes > 1:
        torch.distributed.all_reduce(total_loss)
        torch.distributed.all_reduce(count)
    count = count.item()
    if strict and count != global_sample_number:
        raise ValueError(f"{exp_name} scored {count} of {global_sample_number} requested samples")
    if count == 0:
        return None
    if data_manifest is not None and state.is_main_process:
        data_manifest.update(packed_sequences=count, sequence_length=cfg.max_length,
                             input_ids_sha256=digest.hexdigest(), tokenizer=cfg.tokenizer)
    avg_loss = total_loss.item() / count
    if strict and not math.isfinite(avg_loss):
        raise ValueError(f"{exp_name} produced non-finite FineWeb loss {avg_loss}")
    if state.is_main_process:
        print(f"Experiment: {exp_name}, fineweb_test loss ({count} samples): {avg_loss:.4f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple language model experiments.")
    parser.add_argument(
        "--exp_names",
        type=str,
        nargs='+',
        required=True,
        help="List of experiment names to evaluate (e.g., exp_A exp_B)."
    )
    parser.add_argument(
        "--sample_number",
        type=int,
        default=500,
        help="Number of samples to evaluate for each experiment."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="evaluation_results.json",
        help="Path to save the JSON file with evaluation results."
    )

    parser.add_argument("--exp-labels", nargs="+", help="unique output keys in --exp_names order")
    parser.add_argument("--data-manifest", type=Path, help="score a finite prefix and record its packed-token digest")
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"))
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint-dirs", nargs="+", help="explicit weight directories in --exp_names order")
    weights.add_argument("--latest-saved", action="store_true", help="score the latest usable weights of each run, including interrupted runs")
    parser.add_argument("--strict", action="store_true", help="require a finite loss on every requested sample")
    args = parser.parse_args()
    if args.data_manifest:
        args.strict = True
    labels = args.exp_labels or args.exp_names
    if len(labels) != len(args.exp_names) or len(set(labels)) != len(labels):
        parser.error("Supply one unique result label per experiment")
    if args.checkpoint_dirs and len(args.checkpoint_dirs) != len(args.exp_names):
        parser.error("--checkpoint-dirs needs one entry per experiment")
    if args.sample_number < 0:
        parser.error("--sample_number must be nonnegative")
    if args.strict and args.sample_number == 0:
        parser.error("--strict requires at least one sample")
    state = evaluation_state()
    experiment_names = args.exp_names
    global_sample_number = args.sample_number
    output_file_path = args.output_file

    if not experiment_names:
        print("Please provide at least one experiment name using --exp_names.")
        return

    print(f"Running evaluation for experiments: {', '.join(experiment_names)}")
    print(f"Number of samples per experiment: {global_sample_number}")

    all_results: Dict[str, Optional[float]] = {} # Initialize dictionary to store all results

    manifests = {}
    for exp_name, label, weights_dir in zip(experiment_names, labels, args.checkpoint_dirs or [None] * len(labels), strict=True):
        data = {} if args.data_manifest else None
        loss = evaluate_model(exp_name, global_sample_number, state, args.attn_implementation,
                              args.strict, args.latest_saved, weights_dir, data)
        if data is not None:
            manifests[label] = data
        all_results[label] = loss # loss can be float or None
        print("-" * 70)

    if not state.is_main_process:
        return

    if args.data_manifest:
        if len({m["input_ids_sha256"] for m in manifests.values()}) != 1:
            raise ValueError("The compared models received different packed validation tokens")
        args.data_manifest.write_text(json.dumps(manifests, indent=2) + "\n")
    # Save the results to a JSON file
    try:
        with open(output_file_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Evaluation results saved to {output_file_path}")
    except IOError as e:
        if args.strict:
            raise
        print(f"Error saving results to JSON file {output_file_path}: {e}")

    print("\nSummary of results:")
    for exp, res_loss in all_results.items():
        if res_loss is not None:
            print(f"  {exp}: {res_loss}")
        else:
            print(f"  {exp}: Evaluation failed or no samples processed.")


if __name__ == "__main__":
    main()