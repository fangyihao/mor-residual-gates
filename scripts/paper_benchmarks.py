"""Score Table 3 downstream tasks on one device with the vendored MoR evaluation harness."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TASKS = {"lambada_openai": ("LD", "acc"), "hellaswag": ("HS", "acc_norm"), "piqa": ("PQ", "acc_norm"),
         "winogrande": ("WG", "acc"), "arc_easy": ("ARC-E", "acc_norm"), "arc_challenge": ("ARC-C", "acc_norm"),
         "mmlu_continuation": ("MMLU", "acc")}


def shot_overrides(value):
    changes = {}
    for entry in value.split(","):
        name, _, count = entry.partition("=")
        if name not in TASKS or name in changes or not count.isdecimal():
            raise argparse.ArgumentTypeError("Use comma-separated harness task names and nonnegative counts, e.g. mmlu_continuation=5")
        changes[name] = int(count)
    return changes


def task_specs(shots=None):
    specs = []
    for name, (_, metric) in TASKS.items():
        options = {}
        if metric == "acc_norm":
            options["normalize_choice_by_bytes"] = True
        if name in (shots or {}):
            options["num_fewshot"] = shots[name]
        specs.append({"task": name, **options} if options else name)
    return specs


def table_metrics(raw):
    scores = {}
    for task, (column, metric) in TASKS.items():
        table = raw["groups"] if task == "mmlu_continuation" else raw["results"]
        value = float(table[task][metric + ",none"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid {task} accuracy: {value}")
        scores[column] = 100 * value
    scores["ARC"] = mean([scores["ARC-E"], scores["ARC-C"]])
    scores["Avg"] = mean(scores[k] for k in ("LD", "HS", "PQ", "WG", "ARC", "MMLU"))
    return scores


def score(directory, weights, shots, native_mor=False):
    from evaluate_fineweb_test import evaluation_state, load_scoring_model
    from lm_eval import evaluator
    from lm_eval.models.recursive_lm import RecursiveTransformer
    from lm_eval.tasks import TaskManager
    from util.config import load_scoring_config
    from util.misc import get_torch_dtype
    from util.tokenizer import load_tokenizer_from_config

    state = evaluation_state()
    if state.num_processes != 1:
        raise ValueError("The paper downstream evaluator expects one process")
    cfg = load_scoring_config(directory / "training_config.yaml", native_mor)
    if state.device.type == "npu" and cfg.attn_implementation == "flash_attention_2":
        cfg.attn_implementation = "sdpa"
    model = load_scoring_model(cfg, directory, weights).eval().to(state.device, dtype=get_torch_dtype(cfg))
    lm = RecursiveTransformer(pretrained=model, tokenizer=load_tokenizer_from_config(cfg),
                              device=str(state.device), batch_size=32, max_length=cfg.max_length,
                              add_bos_token=cfg.add_bos_token)
    raw = evaluator.simple_evaluate(model=lm, tasks=task_specs(shots), task_manager=TaskManager(),
                                    num_fewshot=None, log_samples=False, bootstrap_iters=0)
    for task, expected in (shots or {}).items():
        observed = [n for name, n in raw["n-shot"].items() if name == task or name.startswith(task + "_")]
        if not observed or any(n != expected for n in observed):
            raise ValueError(f"The harness did not apply {expected} shots to every {task} task")
    return {"metrics": table_metrics(raw), "n_shot": raw["n-shot"], "n_samples": raw["n-samples"],
            "task_versions": raw["versions"], "batch_size": 32, "normalized_choice_unit": "UTF-8 bytes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--native-mor-label", help="load the native MoR architecture for this external checkpoint label")
    parser.add_argument("--shots", type=shot_overrides)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or args.runs
    if len(labels) != len(args.runs) or len(set(labels)) != len(labels) or len(args.runs) != len(args.weights):
        parser.error("Pass one checkpoint per run")
    if args.native_mor_label and args.native_mor_label not in labels:
        parser.error("--native-mor-label must match a result label")
    results = {}
    for name, label, weights in zip(args.runs, labels, args.weights, strict=True):
        results[label] = score(ROOT / "results/pretrain" / name, weights, args.shots, native_mor=label == args.native_mor_label)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)


if __name__ == "__main__":
    main()
