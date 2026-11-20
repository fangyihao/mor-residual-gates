"""Regression coverage for prefetched data, shuffled blocks and partial checkpoints."""
from itertools import islice
import json

from datasets import Dataset, interleave_datasets
import pytest
import torch

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from util.resume import complete_checkpoint, evaluation_weights, latest_checkpoint


class Tokenizer:
    eos_token, eos_token_id, padding_side = "|", 124, "right"

    def __call__(self, texts, **kwargs):
        return {"input_ids": [list(text.encode()) for text in texts]}


@pytest.mark.parametrize("global_shuffle,mixture", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("consumed", [3, 16, 97])
def test_resume_after_prefetch(tmp_path, global_shuffle, mixture, consumed):
    def dataset():
        docs = Dataset.from_dict({"text": [f"doc {i}" for i in range(19)]}).to_iterable_dataset()
        if mixture:
            second = Dataset.from_dict({"text": [f"example {i}" for i in range(23)]}).to_iterable_dataset()
            docs = interleave_datasets([docs, second], probabilities=[0.3, 0.7], seed=17)
        packed = LanguageModelingDataset(docs, Tokenizer(), 7, buffer_size=29,
                                         global_shuffling=global_shuffle, local_shuffling=True)
        packed.track_position = True
        return packed

    stream = dataset()
    it = iter(stream)
    expected = [v["input_ids"] for v in islice(it, consumed + 38)]
    stream.commit_position(consumed)
    path = tmp_path / "cursor.pt"
    torch.save(stream.state_dict(consumed), path)
    restored = dataset()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual = [v["input_ids"] for v in islice(restored, 38)]
    assert torch.equal(torch.stack(actual), torch.stack(expected[consumed:]))


def test_incomplete_and_truncated_saves(tmp_path):
    for step in (7, 8, 9):
        directory = tmp_path / f"checkpoint-{step}"
        directory.mkdir()
        state = directory / "trainer_state.json"
        state.write_text(json.dumps({"global_step": step}))
        weights = directory / "model.safetensors"
        weights.write_bytes(b"saved weights")
        sizes = {p.name: p.stat().st_size for p in directory.iterdir()}
        (directory / "resume.json").write_text(json.dumps(dict(version=1, global_step=step, files=sizes)))
    (tmp_path / "checkpoint-9" / ".saving").touch()
    (tmp_path / "checkpoint-8" / "model.safetensors").write_bytes(b"short")
    with pytest.warns(UserWarning, match="incomplete checkpoint"):
        assert latest_checkpoint(tmp_path).name == "checkpoint-7"
    assert evaluation_weights(tmp_path).name == "checkpoint-7"
    assert not complete_checkpoint(tmp_path / "checkpoint-8")


def test_ablation_snapshot_retains_shards_and_frozen_state(tmp_path):
    from util.resume import preserve_ablation_weights, weights_at_step
    from shutil import rmtree

    checkpoint = tmp_path / "checkpoint-3"
    checkpoint.mkdir()
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for f in shards:
        (checkpoint / f).write_bytes(f.encode())
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict(zip("ab", shards))}))
    (checkpoint / "trainer_state.json").write_text('{"global_step":3}')
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    preserve_ablation_weights(tmp_path, checkpoint, 3)
    (checkpoint / shards[0]).write_bytes(b"changed")
    rmtree(checkpoint)
    snapshot = weights_at_step(tmp_path, 3)
    assert (snapshot / shards[0]).read_bytes() == shards[0].encode()
    assert (snapshot / "model.safetensors.index.json").is_file()
    assert not (snapshot / "optimizer.pt").exists()
    with pytest.raises(FileNotFoundError):
        weights_at_step(tmp_path, 4)
