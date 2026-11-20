"""Train every recursive-residual mode with paired seeds, then score the checkpoints."""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from statistics import mean, stdev
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.resume import checkpoint_step, evaluation_weights, same_recipe, weights_at_step

MODES = ("rezero", "rezero_stage", "full", "uniform", "power")
DEFAULT_SEEDS = (1234, 1235, 1236)


@dataclass
class Run:
    seed: int
    mode: str
    name: str
    directory: str
    overrides: list[str]
    config: DictConfig
    weights: Path | None = None


def seed_list(value):
    try:
        seeds = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer seeds") from exc
    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("each seed must be distinct")
    return seeds


def training_target(cfg):
    return cfg.num_train_steps if cfg.stop_steps is None else min(cfg.num_train_steps, cfg.stop_steps)


def plan(config_name, overrides, seeds, ablation_steps=None, full_budget_control=None):
    keys = {arg.partition("=")[0].lstrip("+~") for arg in overrides}
    if full_budget_control is not None and ablation_steps is None:
        raise ValueError("--full-budget-control requires --ablation-steps")
    long_modes = {"rezero", full_budget_control or "rezero_stage"}
    if ablation_steps is not None and "ablation_checkpoint_step" in keys:
        raise ValueError("--ablation-steps sets ablation_checkpoint_step automatically")
    if "recursive.residual.mode" in keys:
        raise ValueError("all selects every residual mode. Use the train command to select one mode.")
    if "seed" in keys and seeds is not None:
        raise ValueError("Choose seed=... for one seed or --seeds for a list.")
    defaults = {"recursive.num_recursion": "8", "gradient_checkpointing": "true", "track_token_layers": "true"}
    common = [f"{k}={v}" for k, v in defaults.items() if k not in keys] + overrides
    runs = []
    with initialize_config_dir(config_dir=str(ROOT / "conf/pretrain"), version_base=None):
        if seeds is None:
            seeds = [compose(config_name, overrides=common).seed] if "seed" in keys else DEFAULT_SEEDS
        for seed in seeds:
            for mode in MODES:
                args = [*common, f"recursive.residual.mode={mode}"]
                if "seed" not in keys:
                    args.append(f"seed={seed}")
                if ablation_steps is not None:
                    original = compose(config_name, overrides=args)
                    if not 0 < ablation_steps < training_target(original):
                        raise ValueError(f"--ablation-steps must be positive and below the full training "
                                         f"target ({training_target(original)} optimizer steps)")
                    args.append(f"++ablation_checkpoint_step={ablation_steps}")
                    if mode not in long_modes:
                        args = [a for a in args if a.partition("=")[0].lstrip("+~") != "stop_steps"]
                        args.append(f"stop_steps={ablation_steps}")
                cfg = compose(config_name, overrides=args)
                runs.append(Run(seed, mode, cfg.name, cfg.output_dir or cfg.name, args, cfg))

    # A literal name or output_dir supplied to the whole suite is a prefix.
    # Names that already distinguish every run retain the preset's resume paths.
    names = Counter(run.name for run in runs)
    paths = Counter(run.directory for run in runs)
    for run in runs:
        suffix = f"_{run.mode}_seed{run.seed}"
        if names[run.name] > 1:
            run.name += suffix
            run.config.name = run.name
            run.overrides.append(f"name={json.dumps(run.name)}")
        if paths[run.directory] > 1:
            run.directory += suffix
            run.config.output_dir = run.directory
            run.overrides.append(f"output_dir={json.dumps(run.directory)}")
    resolved = {(ROOT / "results/pretrain" / run.directory).resolve() for run in runs}
    if len(resolved) != len(runs):
        raise ValueError("The output_dir overrides resolve to overlapping checkpoint directories.")
    return runs


def completed(run):
    directory = ROOT / "results/pretrain" / run.directory
    state_path = directory / "trainer_state.json"
    if not state_path.is_file() and not any(directory.glob("checkpoint-*")):
        return False
    saved = directory / "training_config.yaml"
    if not saved.is_file() or not same_recipe(OmegaConf.load(saved), run.config):
        raise ValueError(f"{directory} contains a different or unknown training configuration. "
                         "Choose a fresh name= prefix for the new suite.")
    try:
        weights = evaluation_weights(directory)
        step = checkpoint_step(weights)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    milestone = run.config.get("ablation_checkpoint_step")
    if milestone and step >= milestone:
        try:
            weights_at_step(directory, milestone)
        except FileNotFoundError as exc:
            raise ValueError(f"{exc}. Use a retained checkpoint step or a fresh run prefix.") from exc
    return step >= training_target(run.config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("npu", "accelerate", "deepspeed"))
    parser.add_argument("devices")
    parser.add_argument("--seeds", type=seed_list, help="comma-separated seeds (default 1234,1235,1236)")
    parser.add_argument("--ablation-steps", type=int, metavar="S",
                        help="stop auxiliary runs at S optimizer updates and score all modes at S")
    parser.add_argument("--full-budget-control", choices=(*MODES[1:], "none"),
                        help="control to continue alongside rezero (default rezero_stage in a shorter suite)")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None,
                        help="resume interrupted runs (default on). Disabling it requires new output for unfinished runs")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--eval-only", action="store_true", help="score saved runs without launching training")
    phases.add_argument("--train-only", action="store_true", help="finish after all training runs")
    parser.add_argument("--sample-number", type=int, default=200, help="packed FineWeb samples per checkpoint")
    parser.add_argument("--output-file", type=Path, help="loss JSON (the matched-step results when using --ablation-steps)")
    parser.add_argument("overrides", nargs="*", help="shared Hydra training overrides")
    args = parser.parse_intermixed_args()
    if args.sample_number < 1 and not args.train_only:
        parser.error("--sample-number must be positive for evaluation")
    npu = args.backend == "npu"
    config = os.environ.get("CONFIG_NAME", "recursive_residual") if npu else "recursive_residual"
    try:
        overrides = list(args.overrides)
        if npu and not any(arg.partition("=")[0] == "attn_implementation" for arg in overrides):
            overrides.append("attn_implementation=sdpa")
        if args.auto_resume is not None:
            if any(arg.partition("=")[0] == "auto_resume" for arg in overrides):
                raise ValueError("Specify --auto-resume or the Hydra auto_resume override only once")
            overrides.append("auto_resume=" + str(args.auto_resume).lower())
        runs = plan(config, overrides, args.seeds, args.ablation_steps, args.full_budget_control)
        if args.eval_only and args.ablation_steps is None:
            available = []
            for run in runs:
                try:
                    run.weights = evaluation_weights(ROOT / "results/pretrain" / run.directory)
                except FileNotFoundError as exc:
                    print(f"Skipping {run.directory}: {exc}", flush=True)
                else:
                    available.append(run)
            runs = available
            if not runs:
                parser.error("No saved experiments match this suite. Check the name, output_dir and seeds.")
        finished = set() if args.eval_only else {run.directory for run in runs if completed(run)}
    except ValueError as exc:
        parser.error(str(exc))

    if not args.train_only:
        default_name = (f"residual_suite_fineweb.step-{args.ablation_steps}.json" if args.ablation_steps
                        else "residual_suite_fineweb.json")
        output = ROOT / (args.output_file or default_name)
        if output.is_dir():
            parser.error(f"--output-file points to a directory: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
    if npu:
        train = ["bash", "scripts/run_npu.sh", "train", args.devices]
    else:
        train = ["bash", "scripts/pretrain.sh", args.backend, os.environ.get("WANDB_MODE", "online"),
                 args.devices, config, "--"]
    for index, run in enumerate([] if args.eval_only else runs, 1):
        if run.directory in finished:
            print(f"[{index}/{len(runs)}] {run.directory} already complete", flush=True)
            continue
        print(f"[{index}/{len(runs)}] {run.directory} (mode={run.mode}, seed={run.seed})", flush=True)
        subprocess.run([*train, *run.overrides], cwd=ROOT, check=True)
    if args.train_only:
        return

    if args.ablation_steps is None:
        evaluate(runs, args, output)
        return

    long_runs = [run for run in runs if training_target(run.config) > args.ablation_steps]
    full_target = training_target(long_runs[0].config)
    base = ROOT / (args.output_file or "residual_suite_fineweb.json")
    full_output = base.with_name(f"{base.stem}.full-step-{full_target}{base.suffix}")
    evaluated = False
    for step, group, destination in ((args.ablation_steps, runs, output), (full_target, long_runs, full_output)):
        selected = []
        for run in group:
            directory = ROOT / "results/pretrain" / run.directory
            try:
                weights = weights_at_step(directory, step)
            except FileNotFoundError as exc:
                if not args.eval_only:
                    raise
                print(f"Skipping {run.directory} at step {step}: {exc}", flush=True)
                continue
            if args.eval_only:
                saved = directory / "training_config.yaml"
                if not saved.is_file() or not same_recipe(OmegaConf.load(saved), run.config):
                    parser.error(f"{directory} has a different training recipe. Repeat the original training overrides.")
            selected.append(replace(run, weights=weights))
        if selected:
            print(f"Evaluating {len(selected)} runs at optimizer step {step} -> {destination}", flush=True)
            evaluate(selected, args, destination)
            evaluated = True
    if not evaluated:
        parser.error("No saved weights match the requested ablation or full training step.")


def evaluate(runs, args, output):
    env = os.environ.copy()
    if args.backend == "npu":
        evaluate = ["bash", "scripts/run_npu.sh", "eval", args.devices]
    else:
        env.update(CUDA_VISIBLE_DEVICES=args.devices, ACCELERATE_USE_DEEPSPEED="false")
        evaluate = [sys.executable]
        workers = len(args.devices.split(","))
        if workers > 1:
            evaluate += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={workers}"]
        evaluate.append("evaluate_fineweb_test.py")
    subprocess.run([*evaluate, "--exp_names", *(run.directory for run in runs),
                    "--sample_number", str(args.sample_number), "--output_file", str(output),
                    *(["--checkpoint-dirs", *(str(run.weights) for run in runs)] if runs[0].weights is not None else ["--latest-saved"]),
                    "--strict"], cwd=ROOT, env=env, check=True)
    if args.backend == "npu" and int(env.get("NNODES", "1")) > 1 and int(env["NODE_RANK"]) != 0:
        return
    losses = json.loads(output.read_text())
    layer_specific = {}
    paired_deltas = defaultdict(list)
    for run in runs:
        directory = ROOT / "results/pretrain" / run.directory
        state_path = (run.weights or evaluation_weights(directory)) / "trainer_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        step = state.get("global_step")
        rows = [row for row in state.get("log_history", []) if "loss" in row]
        work = ""
        tokens = None
        if rows and all("executed_token_layers" in row and "input_tokens" in row for row in rows):
            tokens = sum(row["input_tokens"] for row in rows)
            token_layers = sum(row["executed_token_layers"] for row in rows)
            work = f"  tokens={tokens}  executed_token_layers={token_layers}"
        print(f"{run.directory}  step={step}  NLL={losses[run.directory]:.7f}{work}")
        if run.mode == "rezero":
            layer_specific[run.seed] = (step, tokens, losses[run.directory])
            continue
        if run.seed not in layer_specific or (args.ablation_steps is None and run.mode != "rezero_stage"):
            continue
        reference_step, reference_tokens, reference_loss = layer_specific[run.seed]
        if reference_step != step:
            continue
        if args.ablation_steps and reference_tokens is not None and tokens is not None and reference_tokens != tokens:
            print(f"Skipping pair for {run.directory}: training input-token counts differ ({reference_tokens}, {tokens})")
            continue
        delta = reference_loss - losses[run.directory]
        label = "stage-shared" if run.mode == "rezero_stage" else run.mode
        print(f"seed={run.seed}  layer-specific minus {label} NLL={delta:+.7f}")
        paired_deltas[label].append(delta)
    if args.ablation_steps:
        for label, values in paired_deltas.items():
            if len(values) > 1:
                print(f"layer-specific minus {label}: mean paired NLL difference={mean(values):+.7f}  "
                      f"sample SD={stdev(values):.7f}  paired seeds={len(values)}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
