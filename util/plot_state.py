"""Plotting records for routed runs. All tensor observations are detached."""
from functools import partial
import json
import math
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

from model.base_model.modeling_llama import LlamaDecoderLayer
from model.recursive_residual import stage_scales


class PlotState(TrainerCallback):
    def __init__(self, trainer, cfg):
        self.trainer = trainer
        self.cfg = cfg
        self.path = Path(trainer.args.output_dir) / "training_metrics.jsonl"
        self.hidden_samples = int(cfg.get("plot_hidden_samples", 16))
        if self.hidden_samples < 0:
            raise ValueError("plot_hidden_samples must be nonnegative")
        self.capture = False
        self.sample_hidden = False
        self.hidden = {}
        self.routed = {}
        self.writebacks = {}
        self.gate_grads = {}
        self.lm = self.aux = 0
        self.gates = {name: p for name, p in trainer.model.named_parameters() if name.endswith(".recursive_gate")}
        residual = cfg.recursive.get("residual")
        self.fixed = {}
        if residual:
            scales = stage_scales(residual.mode, cfg.recursive.num_recursion,
                                  residual.get("lambda_r", 1.0), residual.get("zeta", 1.0))
            self.fixed = {f"epsilon/stage_{j}": scale for j, scale in enumerate(scales, 2) if scale is not None}
        self.identity = {"run": cfg.get("name"), "method": cfg.get("paper_method") or
                         (cfg.recursive.get("residual") or {}).get("mode", "original_mor"),
                         "k": cfg.recursive.num_recursion, "seed": trainer.args.seed}

    def observe_block(self, key, x, raw, y):
        if not (self.capture and self.sample_hidden):
            return
        b, s, _ = x.shape
        n = min(self.hidden_samples, b * s)
        index = torch.arange(n, device=x.device) * (b * s) // n
        bx, sx = index // s, index % s
        x, raw, y = (v.detach()[bx, sx].float() for v in (x, raw, y))
        self.hidden[key] = torch.stack([x.square().mean(), y.square().mean(),
                                        (raw - x).square().mean(), (y - x).square().mean()])

    def observe_router(self, key, module, args, out):
        if not self.capture:
            return
        x = args[0]
        selected = out.selected_tokens
        counts = self.routed.setdefault(key, [0, 0])
        counts[0] += selected.numel()
        counts[1] += x.shape[0] * x.shape[1]
        for field in ("sampling_acc", "sampling_loss", "sampling_topk_acc", "router_z_loss"):
            value = getattr(out, field, None)
            if value is not None:
                self.trainer.record(f"router/{key}/{field}", value)
        if self.hidden_samples and self.sample_hidden:
            # Use selected positions in the original sequence for both sides of writeback.
            n = min(self.hidden_samples, selected.numel())
            index = torch.arange(n, device=x.device) * selected.numel() // n
            bx = index // selected.shape[1]
            sx = selected.reshape(-1)[index]
            before = x.detach()[bx, sx].float()
            after = out.hidden_state.detach()[bx, sx].float()
            self.writebacks[key] = (after - before).square().mean()

    def install(self, stack):
        for name, module in self.trainer.model.named_modules():
            if getattr(module, "mor_type", None) == "expert":
                stack.enter_context(module.register_forward_hook(partial(self.observe_router, name)))
            elif self.hidden_samples and isinstance(module, LlamaDecoderLayer):
                module._observe_recursive_update = partial(self.observe_block, name)
                stack.callback(delattr, module, "_observe_recursive_update")

    def on_train_begin(self, args, state, control, **kwargs):
        state.logging_steps = math.ceil(args.logging_steps * state.max_steps) if args.logging_steps < 1 else args.logging_steps
        self.step = state.global_step
        self.tokens = sum(r.get("input_tokens", 0) for r in state.log_history)
        self.work = sum(r.get("executed_token_layers", 0) for r in state.log_history)
        self.clock = time.monotonic()
        if state.is_world_process_zero:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # A crash can leave newer uncommitted or partial lines after the checkpoint.
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w") as f:
                for row in state.log_history:
                    self.write(f, row)
            tmp.replace(self.path)

    def write(self, file, row):
        event = "train" if "loss" in row else "eval" if any(k.startswith("eval_") for k in row) else "trainer"
        values = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in row.items()}
        file.write(json.dumps({**self.identity, "event": event, **values}, allow_nan=False) + "\n")

    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            with self.path.open("a") as f:
                self.write(f, state.log_history[-1])

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # HF invokes this after clipping, before the optimizer clears gradients.
        self.gate_grads = {n: p.grad.detach().float().reshape(()).clone() for n, p in self.gates.items()
                           if p.grad is not None and p.grad.numel() == 1}

    def collect(self, logs, state):
        trainer = self.trainer
        steps = state.global_step - self.step
        if steps <= 0:
            return
        loss_parts = torch.stack([self.lm, self.aux]).float() / steps
        logs["lm_loss"], logs["weighted_aux_loss"] = trainer.accelerator.reduce(loss_parts, reduction="mean").tolist()
        self.lm = self.aux = 0
        if self.routed:
            keys = list(self.routed)
            counts = torch.tensor([self.routed[key] for key in keys], device=trainer.args.device, dtype=torch.long)
            reaches = []
            for key, (selected, total) in zip(keys, trainer.accelerator.reduce(counts, reduction="sum").tolist()):
                logs[f"router/{key}/selected_tokens"] = selected
                logs[f"router/{key}/reach_fraction"] = selected / total
                reaches.append(selected / total)
            if self.cfg.recursive.sharing in {"cycle", "middle_cycle"} and len(reaches) == self.identity["k"]:
                for j, (reach, deeper) in enumerate(zip(reaches, reaches[1:] + [0.0]), 1):
                    logs[f"depth/{j}/fraction"] = reach - deeper
            self.routed.clear()
        if self.hidden:
            keys = list(self.hidden)
            values = torch.stack([self.hidden[key] for key in keys])
            rms = trainer.accelerator.reduce(values, reduction="mean").sqrt().tolist()
            for key, moments in zip(keys, rms):
                for field, value in zip(("input_rms", "output_rms", "raw_update_rms", "update_rms"), moments):
                    logs[f"hidden/{key}/{field}"] = value
            self.hidden.clear()
        if self.writebacks:
            values = torch.stack(list(self.writebacks.values()))
            rms = trainer.accelerator.reduce(values, reduction="mean").sqrt().tolist()
            for key, value in zip(self.writebacks, rms):
                logs[f"router/{key}/writeback_rms"] = value
            self.writebacks.clear()
        # The paper uses replicated parameters under DDP or ZeRO-2.
        gate_metrics = {f"gate/{n}": p.detach().float().reshape(()) for n, p in self.gates.items() if p.numel() == 1}
        gate_metrics.update({f"gate_grad/{n}": g for n, g in self.gate_grads.items()})
        if gate_metrics:
            logs.update(zip(gate_metrics, torch.stack(list(gate_metrics.values())).tolist()))
        logs.update(self.fixed)
        for i, group in enumerate(trainer.optimizer.param_groups):
            label = group.get("name", str(i))
            logs[f"optimizer/{label}/lr"] = float(group["lr"])
            logs[f"optimizer/{label}/weight_decay"] = float(group.get("weight_decay", 0))
        scaler = trainer.accelerator.scaler
        if scaler is not None:
            logs["loss_scale"] = scaler.get_scale()
        now = time.monotonic()
        seconds = now - self.clock
        self.tokens += logs["input_tokens"]
        self.work += logs["executed_token_layers"]
        logs.update(interval_optimizer_steps=steps, cumulative_input_tokens=self.tokens,
                    cumulative_executed_token_layers=self.work,
                    interval_seconds=seconds, tokens_per_second=logs["input_tokens"] / max(seconds, 1e-9),
                    gradient_accumulation=trainer.args.gradient_accumulation_steps,
                    world_size=trainer.args.world_size, per_device_batch_size=trainer._train_batch_size)
        self.step, self.clock = state.global_step, now
