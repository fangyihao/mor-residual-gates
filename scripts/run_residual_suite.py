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
from util.resume import checkpoint_step, evaluation_weights, has_weights, same_recipe, weights_at_step

MODES = ("rezero", "rezero_stage", "full", "uniform", "power")
PAPER_METHODS = ("original_mor", "rezero", "rezero_stage")
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
    paper = config_name == "paper_mor"
    if paper and full_budget_control is not None:
        raise ValueError("The paper comparison keeps original_mor and both ReZero variants at full budget.")
    if paper and ("paper_method" in keys or "recursive.residual" in keys):
        raise ValueError("The paper comparison sets the method and recursive residual per run.")
    if full_budget_control is not None and ablation_steps is None:
        raise ValueError("--full-budget-control requires --ablation-steps")
    long_modes = set(PAPER_METHODS) if paper else {"rezero", full_budget_control or "rezero_stage"}
    modes = PAPER_METHODS if paper else MODES
    if paper and ablation_steps:
        modes += ("uniform", "power")
    if ablation_steps is not None and "ablation_checkpoint_step" in keys:
        raise ValueError("--ablation-steps sets ablation_checkpoint_step automatically")
    if "recursive.residual.mode" in keys:
        raise ValueError("all selects every residual mode. Use the train command to select one mode.")
    if "seed" in keys and seeds is not None:
        raise ValueError("Choose seed=... for one seed or --seeds for a list.")
    defaults = {"recursive.num_recursion": "8", "gradient_checkpointing": "true", "track_token_layers": "true"}
    if paper:
        del defaults["recursive.num_recursion"]
    common = [f"{k}={v}" for k, v in defaults.items() if k not in keys] + overrides
    runs = []
    with initialize_config_dir(config_dir=str(ROOT / "conf/pretrain"), version_base=None):
        if seeds is None:
            seeds = [compose(config_name, overrides=common).seed] if "seed" in keys else DEFAULT_SEEDS
        for seed in seeds:
            for mode in modes:
                args = list(common)
                if paper:
                    args.append(f"paper_method={mode}")
                args.append("recursive.residual=null" if mode == "original_mor" else f"recursive.residual.mode={mode}")
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
    parser.add_argument("--paper-comparison", action="store_true",
                        help="compare native MoR and both ReZero variants using the released Table 3 training recipe")
    parser.add_argument("--author-checkpoint", type=Path, help="also score the authors’ released Table 3 K=3 MoR weight directory")
    parser.add_argument("--benchmark-shots", help="override released harness shot counts, e.g. mmlu_continuation=5")
    parser.add_argument("--nll-only", action="store_true", help="skip the downstream benchmarks in the paper comparison")
    parser.add_argument("--ablation-steps", type=int, metavar="S",
                        help="stop auxiliary runs at S optimizer updates and score all modes at S")
    parser.add_argument("--full-budget-control", choices=(*MODES[1:], "none"),
                        help="control to continue alongside rezero (default rezero_stage in a shorter suite)")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None,
                        help="resume interrupted runs (default on). Disabling it requires new output for unfinished runs")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--eval-only", action="store_true", help="score saved runs without launching training")
    phases.add_argument("--train-only", action="store_true", help="finish after all training runs")
    parser.add_argument("--sample-number", type=int, help="packed FineWeb samples (paper comparison default 500, otherwise 200)")
    parser.add_argument("--output-file", type=Path, help="loss JSON (the matched-step results when using --ablation-steps)")
    parser.add_argument("overrides", nargs="*", help="shared Hydra training overrides")
    args = parser.parse_intermixed_args()
    if (args.nll_only or args.author_checkpoint or args.benchmark_shots) and not args.paper_comparison:
        parser.error("Paper evaluation options require --paper-comparison")
    if args.benchmark_shots:
        from scripts.paper_benchmarks import shot_overrides
        try:
            shot_overrides(args.benchmark_shots)
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))
    if args.author_checkpoint:
        args.author_checkpoint = args.author_checkpoint.resolve()
        if not has_weights(args.author_checkpoint):
            parser.error(f"Missing or incomplete model weights in {args.author_checkpoint}")
    if args.sample_number is None:
        args.sample_number = 500 if args.paper_comparison else 200
    if args.sample_number < 1 and not args.train_only:
        parser.error("--sample-number must be positive for evaluation")
    npu = args.backend == "npu"
    config = os.environ.get("CONFIG_NAME", "recursive_residual") if npu else "recursive_residual"
    if args.paper_comparison:
        config = "paper_mor"
    try:
        overrides = list(args.overrides)
        if npu and not any(arg.partition("=")[0] == "attn_implementation" for arg in overrides):
            overrides.append("attn_implementation=sdpa")
        if args.auto_resume is not None:
            if any(arg.partition("=")[0] == "auto_resume" for arg in overrides):
                raise ValueError("Specify --auto-resume or the Hydra auto_resume override only once")
            overrides.append("auto_resume=" + str(args.auto_resume).lower())
        runs = plan(config, overrides, args.seeds, args.ablation_steps, args.full_budget_control)
        if args.author_checkpoint:
            from scripts.paper_comparison import RELEASED_CONFIG, recipe_changes
            reference = OmegaConf.to_container(OmegaConf.load(ROOT / "conf/pretrain" / (RELEASED_CONFIG + ".yaml")))
            changes = recipe_changes(next(r.config for r in runs if r.mode == "original_mor"), reference)
            structural = {"model", "model_config", "recursive", "mor", "kv_sharing", "relaxation", "tokenizer"} & changes.keys()
            if structural:
                raise ValueError(f"Author checkpoint scoring requires the released architecture and router: {sorted(structural)}")
        if args.eval_only and args.ablation_steps is None and not args.paper_comparison:
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
        if args.paper_comparison:
            default_name = default_name.replace("residual_suite", "paper_mor")
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
        subprocess.run([*train, *run.overrides], cwd=ROOT, env={**os.environ, "CONFIG_NAME": config}, check=True)
    if args.train_only:
        return

    if args.ablation_steps is None and not args.paper_comparison:
        evaluate(runs, args, output)
        return

    long_runs = [run for run in runs if training_target(run.config) > (args.ablation_steps or 0)]
    full_target = training_target(long_runs[0].config)
    base = ROOT / (args.output_file or ("paper_mor_fineweb.json" if args.paper_comparison else "residual_suite_fineweb.json"))
    full_output = base.with_name(f"{base.stem}.full-step-{full_target}{base.suffix}")
    evaluated = False
    panels = [(full_target, long_runs, output)]
    if args.ablation_steps:
        panels = [(args.ablation_steps, runs, output), (full_target, long_runs, full_output)]
    for step, group, destination in panels:
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
            evaluate(selected, args, destination, paper_final=args.paper_comparison and step == full_target)
            evaluated = True
    if not evaluated:
        parser.error("No saved weights match the requested ablation or full training step.")


def evaluate(runs, args, output, paper_final=False):
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
    sources = [r.directory for r in runs]
    labels = list(sources)
    weights = [str(r.weights) for r in runs if r.weights is not None]
    if paper_final and args.author_checkpoint:
        native = next((r for r in runs if r.mode == "original_mor"), None)
        if native is None:
            raise ValueError("Rescoring the author checkpoint needs a saved native MoR run configuration.")
        sources.append(native.directory)
        labels.append("authors_checkpoint_rescored")
        weights.append(str(args.author_checkpoint))
    scoring = ["--checkpoint-dirs", *weights] if weights else ["--latest-saved"]
    if args.paper_comparison:
        scoring.extend(["--exp-labels", *labels, "--data-manifest", str(output.with_suffix(".data.json"))])
    subprocess.run([*evaluate, "--exp_names", *sources, "--sample_number", str(args.sample_number),
                    "--output_file", str(output), *scoring, "--strict"], cwd=ROOT, env=env, check=True)
    if args.backend == "npu" and int(env.get("NNODES", "1")) > 1 and int(env["NODE_RANK"]) != 0:
        return
    if paper_final:
        from scripts.paper_comparison import write_report
        downstream = {}
        if not args.nll_only:
            first_device = args.devices.split(",")[0]
            if args.backend == "npu":
                env.update(MOR_DEVICE="npu", ASCEND_RT_VISIBLE_DEVICES=first_device, ACCELERATE_USE_CPU="false")
            else:
                env["CUDA_VISIBLE_DEVICES"] = first_device
            benchmark_file = output.with_suffix(".benchmarks.json")
            subprocess.run([sys.executable, "scripts/paper_benchmarks.py", "--runs", *sources, "--labels", *labels,
                            "--weights", *weights, "--output", str(benchmark_file),
                            *(["--shots", args.benchmark_shots] if args.benchmark_shots else [])],
                           cwd=ROOT, env=env, check=True)
            downstream = json.loads(benchmark_file.read_text())
        write_report(ROOT, runs, output, downstream, args.author_checkpoint)
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
