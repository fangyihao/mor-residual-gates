"""Device binding for the opt-in Ascend torchrun launcher."""
import os

import torch


def select_npu():
    if os.environ.get("MOR_DEVICE") != "npu":
        return None

    try:
        import torch_npu  # noqa: F401 - registers torch.npu and the HCCL backend
    except ImportError as exc:
        raise RuntimeError("Ascend training requires torch_npu matched to the installed PyTorch and CANN.") from exc

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    visible = torch.npu.device_count()
    if not torch.npu.is_available() or not 0 <= local_rank < local_size <= visible:
        raise RuntimeError(f"Local rank {local_rank} of {local_size} cannot use {visible} visible Ascend devices.")
    device = torch.device("npu", local_rank)
    torch.npu.set_device(device)
    return device

if __name__ == "__main__":
    # Exercise the actual device mapping and inter-rank HCCL collective without model files.
    from torch import distributed as dist

    device = select_npu()
    if device is None:
        raise SystemExit("Run this check with scripts/run_npu.sh check DEVICE_IDS")
    dist.init_process_group(backend="hccl")
    try:
        rank = dist.get_rank()
        workers = dist.get_world_size()
        value = torch.tensor([rank + 1], device=device, dtype=torch.int32)
        dist.all_reduce(value)
        if value.item() != workers * (workers + 1) // 2:
            raise RuntimeError("Unexpected result from the HCCL all-reduce")
        print(f"Rank {rank}/{workers} on {device}: HCCL all-reduce passed", flush=True)
    finally:
        dist.destroy_process_group()
