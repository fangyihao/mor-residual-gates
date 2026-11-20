"""The historical reference, native MoR control and locally paired measurements."""
import json
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest

from scripts.paper_benchmarks import TASKS, shot_overrides, table_metrics, task_specs
from scripts.paper_comparison import PUBLISHED, RELEASED_CONFIG, paired_results, recipe_changes, write_report
from scripts.run_residual_suite import PAPER_METHODS, ROOT, plan, training_target


def test_paper_recipes_and_control_budgets():
    reference = OmegaConf.to_container(OmegaConf.load(ROOT / "conf/pretrain" / (RELEASED_CONFIG + ".yaml")))
    runs = plan("paper_mor", [], [1234], 3540)
    assert [r.mode for r in runs] == [*PAPER_METHODS, "uniform", "power"]
    for run in runs:
        cfg = run.config
        with initialize_config_dir(config_dir=str(ROOT / "conf/pretrain"), version_base=None):
            assert compose("paper_mor", overrides=run.overrides) == cfg
        assert cfg.seed == 1234 and cfg.paper_method == run.mode
        assert cfg.name == f"paper_mor_k3_{run.mode}_seed1234"
        assert not recipe_changes(cfg, reference)
        assert cfg.use_pretrained_weights is False and cfg.recursive.sharing == "middle_cycle"
        assert cfg.recursive.initialization == "random"
        if run.mode == "original_mor":
            assert cfg.recursive.residual is None and cfg.mor.expert.gating == "weighted"
        else:
            assert cfg.recursive.residual.mode == run.mode
        assert training_target(cfg) == (14161 if run.mode in PAPER_METHODS else 3540)
    with pytest.raises(ValueError, match="full budget"):
        plan("paper_mor", [], [1234], 3540, "none")


def test_benchmark_metrics():
    raw = {"groups": {"mmlu_continuation": {"acc,none": 0.3}}, "results": {}}
    for task, (_, metric) in TASKS.items():
        if task != "mmlu_continuation":
            raw["results"][task] = {metric + ",none": 0.1 if task == "arc_challenge" else 0.5}
    metrics = table_metrics(raw)
    assert metrics["ARC"] == 30 and metrics["MMLU"] == 30
    assert metrics["Avg"] == pytest.approx((50 * 4 + 30 * 2) / 6)
    specs = task_specs(shot_overrides("mmlu_continuation=5,piqa=2"))
    assert {"task": "piqa", "num_fewshot": 2, "normalize_choice_by_bytes": True} in specs
    assert {"task": "mmlu_continuation", "num_fewshot": 5} in specs
    assert "lambada_openai" in specs
    del raw["results"]["hellaswag"]
    with pytest.raises(KeyError):
        table_metrics(raw)


def test_report_keeps_historical_and_paired_estimates_distinct(tmp_path):
    (tmp_path / "conf").symlink_to(ROOT / "conf", target_is_directory=True)
    runs = plan("paper_mor", ["stop_steps=2"], [1, 2])
    losses = {"authors_checkpoint_rescored": 2.9}
    manifests = {}
    for run in runs:
        directory = tmp_path / "results/pretrain" / run.directory
        run.weights = directory / "checkpoint-2"
        run.weights.mkdir(parents=True)
        OmegaConf.save(run.config, directory / "training_config.yaml", resolve=True)
        (run.weights / "trainer_state.json").write_text(json.dumps({"global_step": 2, "log_history": [
            {"loss": 3.0, "input_tokens": 10, "executed_token_layers": 100}]}))
        losses[run.directory] = 2 + (run.seed * 0.1 if run.mode == "rezero_stage" else 0)
        manifests[run.directory] = {"packed_sequences": 4, "input_ids_sha256": "test hash"}
    output = tmp_path / "scores.json"
    output.write_text(json.dumps(losses))
    output.with_suffix(".data.json").write_text(json.dumps(manifests))
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "model.safetensors").write_bytes(b"abc")
    write_report(tmp_path, runs, output, {}, weights)
    report = json.loads(output.with_suffix(".paper.json").read_text())
    assert report["released_recipe_input_tokens"] == 29_697_769_472
    assert report["published_reference"]["metrics"] == PUBLISHED["metrics"]
    assert report["authors_checkpoint_rescored"]["weight_sha256"]["model.safetensors"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert report["authors_checkpoint_rescored"]["metrics"] == {"fineweb_nll": 2.9}
    assert all(not row["released_token_budget_reached"] for row in report["local_runs"])
    delta = report["paired_local_differences"]["rezero_stage_minus_original_mor"]
    assert delta["summary"]["fineweb_nll"]["mean"] == pytest.approx(0.15)
    assert delta["summary"]["fineweb_nll"]["n"] == 2
    assert "Modified recipe or training budget" in output.with_suffix(".paper.md").read_text()
    rows = report["local_runs"]
    next(r for r in rows if r["seed"] == 1 and r["method"] == "original_mor")["input_tokens"] = 11
    assert paired_results(rows)["rezero_stage_minus_original_mor"]["summary"]["fineweb_nll"]["n"] == 1

def test_normalized_choice_uses_utf8_bytes_only_when_requested():
    from lm_eval.api.task import ConfigurableTask

    config = SimpleNamespace(process_results=None, normalize_choice_by_bytes=True)
    task = SimpleNamespace(config=config, OUTPUT_TYPE="multiple_choice", _metric_fn_list={"acc": None, "acc_norm": None},
                           doc_to_choice=lambda doc: ["a", "é"], count_bytes=ConfigurableTask.count_bytes,
                           multiple_input=False, multiple_target=False, doc_to_target=lambda doc: 1)
    results = [(-0.9, False), (-1.0, False)]
    assert ConfigurableTask.process_results(task, {}, results) == {"acc": 0, "acc_norm": 1}
    config.normalize_choice_by_bytes = False
    assert ConfigurableTask.process_results(task, {}, results)["acc_norm"] == 0


def test_fewshot_override_reaches_every_mmlu_continuation_subject(monkeypatch):
    from datasets import Dataset, DatasetDict
    from lm_eval.tasks import TaskManager, get_task_dict

    data = Dataset.from_list([{"question": "Which?", "choices": ["a", "b", "c", "d"], "answer": 0}])
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: DatasetDict({"test": data, "dev": data}))
    tree = get_task_dict([{"task": "mmlu_continuation", "num_fewshot": 1}], TaskManager())

    def subjects(group):
        for node in group.values():
            if isinstance(node, dict):
                yield from subjects(node)
            else:
                yield node.get_config("num_fewshot")

    assert list(subjects(tree)) == [1] * 57
