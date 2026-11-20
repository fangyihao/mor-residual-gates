"""Check the all-experiments matrix, device launchers, evaluation and exit status."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir

from scripts.run_residual_suite import DEFAULT_SEEDS, MODES, ROOT, plan


def test_default_matrix_and_names():
    runs = plan("recursive_residual", [], None)
    assert [(r.seed, r.mode) for r in runs] == [(seed, mode) for seed in DEFAULT_SEEDS for mode in MODES]
    for r in runs:
        assert r.directory == f"mor_expert_sequence_k8_{r.mode}_seed{r.seed}"
        assert {"recursive.num_recursion=8", "gradient_checkpointing=true", "track_token_layers=true"} <= set(r.overrides)


def test_custom_directories_match_hydra():
    overrides = ["seed=77", 'name="my experiment"', "output_dir=group/ablations", "recursive.num_recursion=4"]
    runs = plan("recursive_residual", overrides, None)
    assert len(runs) == len(MODES)
    with initialize_config_dir(config_dir=str(ROOT / "conf/pretrain"), version_base=None):
        for run in runs:
            cfg = compose("recursive_residual", overrides=run.overrides)
            assert cfg.name == f"my experiment_{run.mode}_seed77" == run.name
            assert cfg.output_dir == f"group/ablations_{run.mode}_seed77" == run.directory
            assert (cfg.recursive.num_recursion, cfg.recursive.residual.mode, cfg.seed) == (4, run.mode, 77)


@pytest.fixture
def launch(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    (repo / "conf").symlink_to(ROOT / "conf", target_is_directory=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    stub = bin_dir / "python"
    stub.write_text(f"#!{sys.executable}\n" + '''
import json, os, runpy, sys
from pathlib import Path
args = sys.argv[1:]
if args[0].endswith("run_residual_suite.py"):
    # Keep GPU torchrun behind the same recorder as Accelerate, DeepSpeed and NPU torchrun.
    sys.executable = sys.argv[0]
    sys.argv = args
    runpy.run_path(args[0], run_name="__main__")
    sys.exit(0)
with Path(os.environ["CALLS"]).open("a") as stream:
    print(json.dumps([Path(sys.argv[0]).name, args, os.environ.get("CUDA_VISIBLE_DEVICES")]), file=stream)
if "recursive.residual.mode=" + os.environ.get("FAIL_MODE", "none") in args:
    sys.exit(37)
if "evaluate_fineweb_test.py" in args:
    names = args[args.index("--exp_names") + 1:args.index("--sample_number")]
    path = Path(args[args.index("--output_file") + 1])
    path.write_text(json.dumps(dict.fromkeys(names, 3.0)))
    for name in names:
        run = Path("results/pretrain") / name
        run.mkdir(parents=True)
        (run / "trainer_state.json").write_text('{"log_history":[{"loss":3,"input_tokens":20,"executed_token_layers":40}]}')
''')
    stub.chmod(0o755)
    for command in ("accelerate", "deepspeed"):
        (bin_dir / command).symlink_to(stub)

    def invoke(backend, options, **extra_env):
        if backend == "npu":
            args = ["scripts/run_npu.sh", "all", "0,1"]
        else:
            args = ["scripts/pretrain.sh", backend, "offline", "0,1", "all", "--"]
        env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", CALLS=str(calls), NNODES="1",
                   NODE_RANK="0", CONFIG_NAME="recursive_residual", **extra_env)
        result = subprocess.run(["bash", *args, *options], cwd=repo, env=env, text=True, capture_output=True, timeout=40)
        return result, [json.loads(line) for line in calls.read_text().splitlines()]
    return invoke


@pytest.mark.parametrize("backend", ["npu", "accelerate", "deepspeed"])
def test_suite_launch_and_eval(launch, backend):
    result, calls = launch(backend, ["--seeds", "19,23", "--sample-number", "4", "name=smoke", "max_length=128"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == 11
    directories = []
    for (_, argv, _), (seed, mode) in zip(calls[:-1], [(i, m) for i in (19, 23) for m in MODES], strict=True):
        directory = f"smoke_{mode}_seed{seed}"
        directories.append(directory)
        assert {f"seed={seed}", f"recursive.residual.mode={mode}", "max_length=128",
                f'output_dir="{directory}"'} <= set(argv)
    _, evaluation, visible = calls[-1]
    assert evaluation[evaluation.index("--exp_names") + 1:evaluation.index("--sample_number")] == directories
    assert "--strict" in evaluation and evaluation[evaluation.index("--sample_number") + 1] == "4"
    if backend != "npu":
        assert "--nproc-per-node=2" in evaluation and visible == "0,1"
    assert result.stdout.count("layer-specific minus stage-shared") == 2
    assert result.stdout.count("tokens=20  executed_token_layers=40") == 10


@pytest.mark.parametrize("backend", ["accelerate", "npu"])
def test_suite_propagates_training_failure(launch, backend):
    result, calls = launch(backend, ["seed=12"], FAIL_MODE="full")
    assert result.returncode == 37
    assert len(calls) == 3 and all("pretrain.py" in call[1] for call in calls)


def test_suite_train_only(launch):
    result, calls = launch("npu", ["--train-only", "seed=1234"])
    assert result.returncode == 0, result.stderr
    assert len(calls) == len(MODES) and all("pretrain.py" in call[1] for call in calls)


def test_completed_and_interrupted_runs(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from scripts import run_residual_suite as suite

    run = plan("recursive_residual", ["name=test", "num_train_steps=8", "stop_steps=3"], [1])[0]
    monkeypatch.setattr(suite, "ROOT", tmp_path)
    directory = tmp_path / "results/pretrain" / run.directory
    (directory / "checkpoint-2").mkdir(parents=True)
    OmegaConf.save(run.config, directory / "training_config.yaml", resolve=True)
    assert not suite.completed(run)
    state = directory / "trainer_state.json"
    state.write_text(json.dumps({"global_step": 3}))
    assert not suite.completed(run)
    (directory / "model.safetensors").touch()
    assert suite.completed(run)
    state.write_text(json.dumps({"global_step": 2}))
    assert not suite.completed(run)
    run.config.recursive.residual.lambda_r = 0.5
    with pytest.raises(ValueError, match="fresh name="):
        suite.completed(run)
