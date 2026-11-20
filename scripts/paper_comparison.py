"""Local paired comparisons and the separately sourced NeurIPS 2025 reference row."""
from collections import defaultdict
import hashlib
import json
from statistics import mean, stdev

from omegaconf import OmegaConf

from scripts.paper_table3 import FLOP_CONVENTION, METRICS, forward_work, write_table3
from util.resume import weight_files

RELEASED_CONFIG = "250720_pretrain_smollm-360m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001"
PUBLISHED = {
    "source": "Bae et al., NeurIPS 2025, Table 3, Expert / Cache / M-Cyc / 3, fixed FLOPs",
    "url": "https://papers.nips.cc/paper_files/paper/2025/file/8b08bbf8b420faa6eeb4020720582ec7-Paper-Conference.pdf",
    "non_embedding_parameters_rounded": 118_000_000, "training_tokens_rounded": 30_000_000_000,
    "reported_training_flops": 16.5e18,
    "metrics": {"fineweb_nll": 2.7925, "LD": 33.1, "HS": 37.9, "PQ": 66.9, "WG": 52.1,
                "ARC": 38.3, "MMLU": 27.4, "Avg": 42.6},
}

# These upstream fields define the scientific recipe. Device, data parallelism and
# checkpoint settings are recorded in the saved configs, independently of these fields.
RECIPE_FIELDS = ("dataset", "weights", "tokenizer", "model", "model_name_or_path", "model_config",
                 "max_length", "total_batch_size", "precision", "add_bos_token", "use_pretrained_weights",
                 "global_shuffling", "local_shuffling", "batch_size_rampup_steps", "recursive", "kv_sharing", "mor", "relaxation",
                 "lr_scheduler_type", "lr_scheduler_kwargs", "learning_rate", "adam_beta1", "adam_beta2",
                 "weight_decay", "max_grad_norm", "num_train_steps", "num_warmup_steps")


def recipe_changes(cfg, reference):
    values = OmegaConf.to_container(cfg, resolve=True)
    values["recursive"].pop("residual", None)
    return {k: {"released": reference[k], "local": values[k]} for k in RECIPE_FIELDS if values[k] != reference[k]}


def summary(values):
    return {"n": len(values), "mean": mean(values), "sample_sd": stdev(values) if len(values) > 1 else None}


def paired_results(rows):
    by_method = {(r["seed"], r["method"]): r for r in rows}
    comparisons = {}
    for method, control in (("rezero", "original_mor"), ("rezero_stage", "original_mor"), ("rezero", "rezero_stage")):
        pairs = {}
        for row in rows:
            if row["method"] != method:
                continue
            other = by_method.get((row["seed"], control))
            if (not other or row["step"] != other["step"] or row["input_tokens"] is None
                    or row["input_tokens"] != other["input_tokens"]):
                continue
            pairs[row["seed"]] = {k: v - other["metrics"][k] for k, v in row["metrics"].items() if k in other["metrics"]}
        if pairs:
            estimates = defaultdict(list)
            for deltas in pairs.values():
                for metric, value in deltas.items():
                    estimates[metric].append(value)
            comparisons[f"{method}_minus_{control}"] = {
                "per_seed": pairs, "summary": {k: summary(v) for k, v in estimates.items()}}
    return comparisons


def method_summaries(rows):
    methods = {}
    for row in rows:
        metrics = methods.setdefault(row["method"], defaultdict(list))
        for key, value in row["metrics"].items():
            metrics[key].append(value)
    return {method: {k: summary(v) for k, v in metrics.items()} for method, metrics in methods.items()}


def weight_digests(directory):
    digests = {}
    for path in sorted(weight_files(directory)):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        digests[str(path.relative_to(directory))] = digest.hexdigest()
    return digests


def write_table(path, result):
    columns = METRICS
    lines = ["# MoR comparison", "", "| Source | Method | Seeds | Training tokens | Forward FLOPs (10^18) | FineWeb NLL ↓ | LD | HS | PQ | WG | ARC | MMLU | Avg ↑ |",
             "|---|---|---:|---:|---:|" + "---:|" * len(columns)]

    def add_row(source, method, seeds, tokens, flops, metrics):
        values = []
        for col in columns:
            if col not in metrics:
                values.append("—")
                continue
            value = metrics[col]
            digits = 4 if col == "fineweb_nll" else 1
            if isinstance(value, dict):
                label = f"{value['mean']:.{digits}f}"
                if value["sample_sd"] is not None:
                    label += f" ± {value['sample_sd']:.{digits}f}"
            else:
                label = f"{value:.{digits}f}"
            values.append(label)
        lines.append("| " + " | ".join([source, method, str(seeds), tokens, flops, *values]) + " |")

    add_row("Bae et al., Table 3", "MoR", "—", "30B (reported)", "16.5 (reported)", PUBLISHED["metrics"])
    if "authors_checkpoint_rescored" in result:
        add_row("Supplied MoR checkpoint scored locally", "MoR", "—", "—", "—", result["authors_checkpoint_rescored"]["metrics"])
    for mode, metrics in result["local_summary"].items():
        counts = [row["input_tokens"] for row in result["local_runs"] if row["method"] == mode]
        tokens = "—" if None in counts else (f"{min(counts):,}" if min(counts) == max(counts)
                                              else f"{min(counts):,}–{max(counts):,}")
        estimates = [r["forward_work"] for r in result["local_runs"] if r["method"] == mode]
        flops = (f"{mean(e['total'] for e in estimates) / 1e18:.3g} (est.)" if all(estimates) else "—")
        add_row("Local training", mode, metrics["fineweb_nll"]["n"], tokens, flops, metrics)
    sequences = next(iter(result["evaluation_data"].values()))["packed_sequences"]
    lines += ["", f"Local NLL uses {sequences} packed sequences. Accuracy is reported in percent. "
              "Local entries show the mean and sample standard deviation across available training seeds.",
              "", "The accompanying JSON records training work, protocol differences and paired seed-level contrasts."]
    if any(row["recipe_changes"] or not row["released_token_budget_reached"] for row in result["local_runs"]):
        lines += ["", "**Modified recipe or training budget.** Treat the published row as context for these local runs."]
    lines += ["", FLOP_CONVENTION, "", "Table 3 rows: " + result["table3_export_note"]]
    if result["benchmark_protocol"]:
        shots = sorted({n for b in result["benchmark_protocol"].values() for n in b["n_shot"].values() if isinstance(n, int)})
        lines += ["", f"Local harness shot counts observed across tasks: {shots}. Per-task counts are in the JSON. "
                  "The published paper does not enumerate the counts for each task."]
    lines += ["", "[Published reference and evaluation details](" + PUBLISHED["url"] + ")"]
    path.write_text("\n".join(lines) + "\n")


def write_report(root, runs, output, downstream, author_checkpoint=None):
    reference = OmegaConf.to_container(OmegaConf.load(root / "conf/pretrain" / (RELEASED_CONFIG + ".yaml")))
    target_tokens = reference["num_train_steps"] * reference["total_batch_size"] * reference["max_length"]
    losses = json.loads(output.read_text())
    data = json.loads(output.with_suffix(".data.json").read_text())
    rows = []
    for run in runs:
        directory = root / "results/pretrain" / run.directory
        saved = OmegaConf.load(directory / "training_config.yaml")
        state = json.loads((run.weights / "trainer_state.json").read_text())
        logs = [r for r in state["log_history"] if "loss" in r]
        def total(key):
            return sum(r[key] for r in logs) if logs and all(key in r for r in logs) else None

        count_path = directory / "paper_parameters.json"
        downstream_result = downstream.get(run.directory, {})
        changes = recipe_changes(saved, reference)
        tokens = total("input_tokens")
        token_layers = total("executed_token_layers")
        model_config = run.weights / "config.json"
        work = (forward_work(json.loads(model_config.read_text()), saved.max_length, tokens, token_layers)
                if model_config.is_file() and not saved.relaxation.enable else None)
        rows.append({"run": run.directory, "method": run.mode, "seed": run.seed, "step": state["global_step"],
                     "weights": str(run.weights.relative_to(directory)), "input_tokens": tokens,
                     "executed_token_layers": token_layers, "forward_work": work,
                     "parameters": json.loads(count_path.read_text()) if count_path.is_file() else None,
                     "execution": {k: saved.get(k) for k in ("attn_implementation", "gradient_checkpointing",
                                                              "per_device_train_batch_size", "precision")},
                     "recipe_changes": changes, "released_token_budget_reached": tokens == target_tokens,
                     "metrics": {"fineweb_nll": losses[run.directory], **downstream_result.get("metrics", {})}})
    result = {"published_reference": PUBLISHED, "released_config": RELEASED_CONFIG,
              "released_recipe_input_tokens": target_tokens, "evaluation_data": data,
              "benchmark_protocol": {k: {f: v for f, v in result.items() if f != "metrics"}
                                     for k, result in downstream.items()},
              "forward_flops_convention": FLOP_CONVENTION, "accuracy_units": "percent", "local_runs": rows, "local_summary": method_summaries(rows),
              "paired_local_differences": paired_results(rows)}
    if author_checkpoint:
        label = "authors_checkpoint_rescored"
        result[label] = {"weights": str(author_checkpoint), "weight_sha256": weight_digests(author_checkpoint), "metrics": {
            "fineweb_nll": losses[label], **downstream.get(label, {}).get("metrics", {})}}
    result["table3_export_note"] = write_table3(output.with_suffix(".table3-rows.tex"), result)
    path = output.with_suffix(".paper.json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_table(path.with_suffix(".md"), result)
    print(f"Published Table 3 MoR reference: NLL 2.7925, six-task average 42.6%. Local comparisons: {path}")
    for label, pair in result["paired_local_differences"].items():
        nll = pair["summary"]["fineweb_nll"]
        print(f"{label}  paired mean NLL difference={nll['mean']:+.7f}  seeds={nll['n']}")
