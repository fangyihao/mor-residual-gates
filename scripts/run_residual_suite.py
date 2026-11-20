"""Train every recursive-residual mode with paired seeds, then score the checkpoints."""
import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.resume import checkpoint_step, evaluation_weights, same_recipe

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


def plan(config_name, overrides, seeds):
    keys = {arg.partition("=")[0].lstrip("+~") for arg in overrides}
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
    target = run.config.num_train_steps
    if run.config.stop_steps is not None:
        target = min(target, run.config.stop_steps)
    return step >= target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("npu", "accelerate", "deepspeed"))
    parser.add_argument("devices")
    parser.add_argument("--seeds", type=seed_list, help="comma-separated seeds (default 1234,1235,1236)")
    parser.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=None,
                        help="resume interrupted runs (default on). Disabling it requires new output for unfinished runs")
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--eval-only", action="store_true", help="score saved runs without launching training")
    phases.add_argument("--train-only", action="store_true", help="finish after all training runs")
    parser.add_argument("--sample-number", type=int, default=200, help="packed FineWeb samples per checkpoint")
    parser.add_argument("--output-file", type=Path, default=Path("residual_suite_fineweb.json"))
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
        runs = plan(config, overrides, args.seeds)
        if args.eval_only:
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
        output = ROOT / args.output_file
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

    env = os.environ.copy()
    if npu:
        evaluate = ["bash", "scripts/run_npu.sh", "eval", args.devices]
    else:
        env.update(CUDA_VISIBLE_DEVICES=args.devices, ACCELERATE_USE_DEEPSPEED="false")
        evaluate = [sys.executable]
        workers = len(args.devices.split(","))
        if workers > 1:
            evaluate += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={workers}"]
        evaluate.append("evaluate_fineweb_test.py")
    subprocess.run([*evaluate, "--exp_names", *(run.directory for run in runs),
                    "--sample_number", str(args.sample_number), "--output_file", str(args.output_file),
                    *(["--checkpoint-dirs", *(str(run.weights) for run in runs)] if args.eval_only else ["--latest-saved"]),
                    "--strict"], cwd=ROOT, env=env, check=True)
    if npu and int(env.get("NNODES", "1")) > 1 and int(env["NODE_RANK"]) != 0:
        return
    losses = json.loads((ROOT / args.output_file).read_text())
    layer_specific = {}
    for run in runs:
        directory = ROOT / "results/pretrain" / run.directory
        state_path = (run.weights or evaluation_weights(directory)) / "trainer_state.json"
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        step = state.get("global_step")
        rows = [row for row in state.get("log_history", []) if "loss" in row]
        work = ""
        if rows and all("executed_token_layers" in row and "input_tokens" in row for row in rows):
            tokens = sum(row["input_tokens"] for row in rows)
            token_layers = sum(row["executed_token_layers"] for row in rows)
            work = f"  tokens={tokens}  executed_token_layers={token_layers}"
        print(f"{run.directory}  step={step}  NLL={losses[run.directory]:.7f}{work}")
        if run.mode == "rezero":
            layer_specific[run.seed] = (step, losses[run.directory])
        if run.mode == "rezero_stage" and run.seed in layer_specific and layer_specific[run.seed][0] == step:
            delta = layer_specific[run.seed][1] - losses[run.directory]
            print(f"seed={run.seed}  layer-specific minus stage-shared NLL={delta:+.7f}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
