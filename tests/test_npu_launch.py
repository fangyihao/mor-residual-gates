"""Check Ascend launch arguments and hardware-independent distributed configuration."""
import json
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import pytest

from util.config import preprocess_config


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/run_npu.sh"


def launch(tmp_path, args, **env):
    # Capture the real launcher's exec arguments without requiring CANN on CI.
    import sys
    stub = tmp_path / "python"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    'print(json.dumps([os.environ["ASCEND_RT_VISIBLE_DEVICES"], os.environ["MOR_DEVICE"], sys.argv[1:]]))')
    stub.chmod(0o755)
    environ = {k: v for k, v in os.environ.items() if k not in {"NNODES", "NODE_RANK", "MASTER_ADDR", "MASTER_PORT"}}
    return subprocess.run(["bash", str(LAUNCHER), *args], text=True, capture_output=True,
                          env=dict(environ, PATH=f"{tmp_path}:{environ['PATH']}", **env))


def test_single_node_npu_launch(tmp_path):
    result = launch(tmp_path, ["train", "1,3", "recursive.residual.mode=power"])
    assert result.returncode == 0, result.stderr
    devices, target, argv = json.loads(result.stdout)
    assert (devices, target) == ("1,3", "npu")
    assert "--standalone" in argv and "--nproc-per-node=2" in argv
    assert argv[argv.index("pretrain.py") + 1:][:3] == ["--config-name", "recursive_residual", "attn_implementation=sdpa"]
    assert argv[-1] == "recursive.residual.mode=power"


@pytest.mark.parametrize("action,expected", [("eval", "evaluate_fineweb_test.py"), ("check", "util.device")])
def test_multinode_npu_launch(tmp_path, action, expected):
    tail = ["--exp_names", "my_run"] if action == "eval" else []
    result = launch(tmp_path, [action, "0,1", *tail], NNODES="2", NODE_RANK="1", MASTER_ADDR="host0", MASTER_PORT="23456")
    assert result.returncode == 0, result.stderr
    argv = json.loads(result.stdout)[2]
    assert {"--nnodes=2", "--node-rank=1", "--nproc-per-node=2", "--master-addr=host0", "--master-port=23456"} <= set(argv)
    assert argv[argv.index(expected) + 1:] == tail


@pytest.mark.parametrize("ids", ["2,1", "0,0", "1,,2", "-1", "0, 1"])
def test_invalid_npu_list(tmp_path, ids):
    assert launch(tmp_path, ["train", ids]).returncode != 0


def test_global_batch_uses_all_nodes(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    cfg = OmegaConf.load(ROOT / "conf/pretrain/recursive_residual.yaml")
    cfg.stop_steps = None
    assert preprocess_config(cfg).gradient_accumulation_steps == 4


def test_npu_binding_uses_local_rank(monkeypatch):
    import sys
    from types import SimpleNamespace
    from util import device

    calls = []
    monkeypatch.setenv("MOR_DEVICE", "npu")
    monkeypatch.setenv("RANK", "9")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "4,6")
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    fake = SimpleNamespace(is_available=lambda: True, device_count=lambda: 2, set_device=calls.append)
    monkeypatch.setattr(device.torch, "npu", fake, raising=False)
    monkeypatch.setattr(device.torch, "device", lambda kind, index: f"{kind}:{index}")
    assert device.select_npu() == "npu:1"
    assert calls == ["npu:1"]
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "3")
    with pytest.raises(RuntimeError, match="2 visible"):
        device.select_npu()
