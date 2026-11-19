"""CPU regression tests using tiny real decoder blocks and deterministic routes."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM as HFLlamaForCausalLM, TrainingArguments

from model.mor_model.modeling_llama import MoRLlamaForCausalLM
from model.recursive_model.modeling_llama import LlamaForCausalLM
from model.recursive_residual import stage_scales
from model.sharing_strategy.llama import sharing_strategy
from util.checkpoint import load_model_weights
from util.trainer_pt import MoRTrainer

PRESET = Path(__file__).resolve().parents[1] / "conf/pretrain/recursive_residual.yaml"


def tiny_model(mode="rezero", router="expert", sharing="middle_cycle", k=3, share_kv=False):
    cfg = OmegaConf.load(PRESET)
    cfg.precision = "fp32"
    cfg.gradient_accumulation_steps = 1
    cfg.recursive.update(num_recursion=k, sharing=sharing, base_depth=2, initialization="stepwise")
    if mode is None:
        del cfg.recursive.residual
    else:
        cfg.recursive.residual.mode = mode
    cfg.mor.enable = router is not None
    cfg.mor.type = router
    cfg.mor.capacity = ",".join(str((k - i) / k) for i in range(k))
    cfg.kv_sharing.update(enable=share_kv, num_recursion=k, sharing=sharing)
    torch.manual_seed(2026)
    config = LlamaConfig(
        vocab_size=48, hidden_size=16, intermediate_size=24,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        num_hidden_layers=2 * k + 2 * sharing.startswith("middle"), use_cache=False,
    )
    config._attn_implementation = "eager"
    model_cls = MoRLlamaForCausalLM if router else LlamaForCausalLM
    model, _ = sharing_strategy(cfg, model_cls(config))
    if share_kv:
        model.set_kv_sharing_config(cfg)
    if router:
        getattr(model, "transform_layer_to_mor_" + router)(cfg)
    return model, cfg


def gates(model):
    return [param for name, param in model.named_parameters() if name.endswith(".recursive_gate")]


@pytest.mark.parametrize("k", [1, 2, 3, 8])
def test_fixed_schedules(k):
    assert stage_scales("full", k) == [1.0] * (k - 1)
    uniform = stage_scales("uniform", k, 2.5)
    assert uniform == stage_scales("power", k, 2.5, 0)
    for zeta in (0.5, 2.0, 1e5):
        power = stage_scales("power", k, 2.5, zeta)
        assert sum(power) == pytest.approx(2.5 if k > 1 else 0.0)
        if k > 1 and zeta < 10:
            weights = [j ** -zeta for j in range(2, k + 1)]
            assert power == pytest.approx([2.5 * w / sum(weights) for w in weights])
    assert stage_scales("power", k, 0.0) == [0.0] * (k - 1)


@pytest.mark.parametrize("args", [("bad", 2), ("power", 0), ("power", 3, -1), ("power", 3, 1, -1)])
def test_invalid_schedule(args):
    with pytest.raises(ValueError):
        stage_scales(*args)


@pytest.mark.parametrize("sharing", ["cycle", "sequence", "middle_cycle", "middle_sequence"])
def test_dense_layout_and_checkpoints(sharing, tmp_path):
    model, _ = tiny_model(router=None, sharing=sharing)
    offset = int(sharing.startswith("middle"))
    for physical in range(2):
        for stage in range(3):
            pos = stage * 2 + physical if sharing.endswith("cycle") else physical * 3 + stage
            layer = model.model.layers[offset + pos]
            assert hasattr(layer, "recursive_gate") == (stage > 0)
    assert len(gates(model)) == 4
    for i, gate in enumerate(gates(model)):
        gate.data.fill_(i - 1.5)
    ids = torch.arange(12).view(2, 6)
    model.eval()
    model.save_pretrained(tmp_path)
    loaded = LlamaForCausalLM.from_pretrained(tmp_path, attn_implementation="eager")
    torch.testing.assert_close(loaded(ids).logits, model(ids).logits, rtol=0, atol=0)


def test_k_one_has_no_gate():
    model, _ = tiny_model(router=None, sharing="cycle", k=1)
    assert not gates(model)


@pytest.mark.parametrize("mode", ["full", "uniform", "power", "rezero"])
def test_whole_block_equation(mode):
    model, _ = tiny_model(mode, router=None, sharing="sequence")
    torch.manual_seed(543)
    x = torch.randn(2, 6, 16)
    positions = model.model.rotary_emb(x, torch.arange(6)[None])
    scales = stage_scales(mode, 3)
    for i in range(3):
        layer = model.model.layers[i]
        raw = deepcopy(layer)
        if hasattr(raw, "recursive_gate"):
            del raw.recursive_gate
        expected_full = raw(x, position_embeddings=positions)[0]
        values = [-0.75, 0.0, 1.0, 1.5] if mode == "rezero" and i else [1.0 if i == 0 else scales[i - 1]]
        for value in values:
            if mode == "rezero" and i:
                layer.recursive_gate.data.fill_(value)
            actual = layer(x, position_embeddings=positions)[0]
            torch.testing.assert_close(actual, x + value * (expected_full - x))


class FixedDepthRouter(nn.Module):
    """Row zero exits after one pass. Other rows exercise different exit positions."""
    def forward(self, x):
        b, s, _ = x.shape
        depths = torch.arange(s, device=x.device).expand(b, s).clone() % 3
        depths[0] = 0
        return torch.nn.functional.one_hot(depths, num_classes=3).to(x.dtype)


def set_depth_router(model):
    for module in model.modules():
        if getattr(module, "mor_type", None) == "token":
            module.mor_router = FixedDepthRouter()


@pytest.mark.parametrize("router", ["expert", "token"])
@pytest.mark.parametrize("sharing", ["cycle", "middle_cycle"])
@pytest.mark.parametrize("share_kv", [False, True])
def test_zero_gates_match_single_pass(router, sharing, share_kv):
    model, _ = tiny_model(router=router, sharing=sharing, share_kv=share_kv)
    one_pass = deepcopy(model)
    layers = one_pass.model.layers
    middle = sharing.startswith("middle")
    wrapper = layers[int(middle)]
    first_pass = wrapper.block if router == "expert" else wrapper.block_list[0]
    one_pass.model.layers = nn.ModuleList(
        ([layers[0]] if middle else []) + list(first_pass) + ([layers[-1]] if middle else [])
    )
    model.eval()
    if router == "token":
        set_depth_router(model)
    ids = torch.arange(24).view(3, 8)
    expected = one_pass.eval()(ids).logits
    actual = model(ids, use_cache=share_kv).logits
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("router", ["expert", "token"])
def test_sparse_routes_match_fixed_ablations(router):
    model, _ = tiny_model(router=router)
    models = [tiny_model(mode, router=router)[0] for mode in ("full", "uniform", "power")]
    ids = torch.arange(24).view(3, 8)
    for candidate in [model] + models:
        candidate.eval()
        if router == "token":
            set_depth_router(candidate)
    for mode, reference in zip(("full", "uniform", "power"), models):
        for gate, scale in zip(gates(model), [s for s in stage_scales(mode, 3) for _ in range(2)]):
            gate.data.fill_(scale)
        torch.testing.assert_close(model(ids).logits, reference(ids).logits)
        assert not gates(reference)


@pytest.mark.parametrize("router", ["expert", "token"])
@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("sharing", ["middle_cycle", "sequence"])
def test_gate_gradients_groups_and_reload(router, checkpointing, sharing, tmp_path):
    model, cfg = tiny_model(router=router, sharing=sharing)
    if checkpointing:
        model.activation_checkpointing_enable({"use_reentrant": False})
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, weight_decay=0.1, report_to=[])
    trainer = MoRTrainer(model=model, cfg=cfg, args=args)
    optimizer = trainer.create_optimizer()
    gate_group, = [group for group in optimizer.param_groups if group.get("name") == "recursive_gates"]
    assert gate_group["weight_decay"] == 0
    assert {id(p) for p in gate_group["params"]} == {id(p) for p in gates(model)}
    assert all(p.ndim == 0 and p.item() == 0.0 for p in gates(model))
    ids = torch.arange(24).view(3, 8)
    result = model(ids, labels=ids, num_items_in_batch=torch.tensor(ids.numel() - len(ids)))
    (result.loss + result.sampling_loss + result.balancing_loss).backward()
    assert all(p.grad is not None and p.grad.abs() > 0 for p in gates(model))
    optimizer.step()
    assert all(p.detach().abs() > 0 for p in gates(model))
    assert all(m.training_step == 1 for m in model.modules() if getattr(m, "mor", False))
    model.save_pretrained(tmp_path)
    reloaded, _ = tiny_model(router=router, sharing=sharing)
    load_model_weights(reloaded, tmp_path)
    native_reload = MoRLlamaForCausalLM.from_pretrained(tmp_path, attn_implementation="eager")
    assert sum(p.numel() for p in native_reload.parameters()) == sum(p.numel() for p in model.parameters())
    torch.testing.assert_close(native_reload.eval()(ids).logits, model.eval()(ids).logits, atol=0, rtol=0)
    torch.testing.assert_close(reloaded.eval()(ids).logits, model.eval()(ids).logits, atol=0, rtol=0)


@pytest.mark.parametrize("router", ["expert", "token"])
@pytest.mark.parametrize("sharing", ["sequence", "middle_sequence", "cycle", "middle_cycle"])
@pytest.mark.parametrize("share_kv", [False, True])
def test_preserves_entire_pretrained_model(router, sharing, share_kv, tmp_path):
    cfg = OmegaConf.load(PRESET)
    cfg.recursive.sharing = sharing
    cfg.mor.type = router
    cfg.mor.capacity = "1,0.6,0.3"
    cfg.kv_sharing.update(enable=share_kv, sharing=sharing, num_recursion=3)
    cfg.precision = "fp32"
    torch.manual_seed(39)
    config = LlamaConfig(vocab_size=48, num_hidden_layers=4, hidden_size=12, intermediate_size=24,
                         num_attention_heads=3, num_key_value_heads=1, max_position_embeddings=32)
    config._attn_implementation = "eager"
    dense = HFLlamaForCausalLM(config).eval()
    dense.save_pretrained(tmp_path)
    model = MoRLlamaForCausalLM.from_pretrained(tmp_path, attn_implementation="eager")
    model, _ = sharing_strategy(cfg, model)
    physical_depth = 4 - 2 * sharing.startswith("middle")
    assert len(model.model.layers) == 4 + 2 * physical_depth
    assert len({id(layer.mlp.up_proj.weight) for layer in model.model.layers}) == 4
    if share_kv:
        model.set_kv_sharing_config(cfg)
    getattr(model, "transform_layer_to_mor_" + router)(cfg)
    assert len(gates(model)) == 2 * physical_depth
    if router == "token":
        set_depth_router(model)
    ids = torch.arange(24).view(3, 8)
    torch.testing.assert_close(model.eval()(ids, use_cache=share_kv).logits,
                               dense(ids, use_cache=False).logits, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
@pytest.mark.parametrize("router", ["token", "expert"])
@torch.no_grad()
def test_cached_sparse_decode_matches_individual_rows(backend, router):
    model, _ = tiny_model(router=router, sharing="sequence")
    model.eval().set_attn_implementation(backend)
    for gate in gates(model):
        gate.fill_(0.7)
    sequences = torch.tensor([[1, 4, 3, 2, 18, 22], [3, 2, 9, 8, 14, 6], [8, 11, 6, 1, 7, 20]])

    def decode(tokens):
        out = model(tokens[:, :4], use_cache=True)
        torch.testing.assert_close(out.logits, model(tokens[:, :4], use_cache=False).logits,
                                   atol=2e-6, rtol=1e-5)
        logits = []
        for pos in range(4, 6):
            out = model(tokens[:, pos:pos + 1], past_key_values=out.past_key_values, use_cache=True)
            logits.append(out.logits)
            if router == "token":
                # Token-choice decisions depend only on each token's causal hidden state.
                reference = model(tokens[:, :pos + 1], use_cache=False).logits[:, -1:]
                torch.testing.assert_close(out.logits, reference, atol=2e-6, rtol=1e-5)
        return torch.cat(logits, dim=1)

    batched = decode(sequences)
    individual = torch.cat([decode(row[None]) for row in sequences])
    torch.testing.assert_close(batched, individual, atol=2e-6, rtol=1e-5)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_formats_and_shards(tmp_path, legacy):
    source, _ = tiny_model(router="token")
    with torch.no_grad():
        for gate in gates(source):
            gate.fill_(-1.25)
    if legacy:
        torch.save(source.state_dict(), tmp_path / "pytorch_model.bin")
    else:
        source.save_pretrained(tmp_path, max_shard_size="4KB")
        assert (tmp_path / "model.safetensors.index.json").is_file()
    target, _ = tiny_model(router="token")
    load_model_weights(target, tmp_path)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, atol=0, rtol=0)
