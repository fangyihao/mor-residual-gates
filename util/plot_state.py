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


class TrainingLog(TrainerCallback):
    """Persist the Trainer's committed history and subsequent events on global rank zero."""

    filename = "training_metrics.jsonl"

    def __init__(self, identity=None):
        self.identity = dict(identity or {})

    def write(self, file, row):
        event = "train" if "loss" in row else "eval" if any(k.startswith("eval_") for k in row) else "trainer"
        values = {"step": row.get("step"), "event": event, **self.identity, **row}
        values = {k: None if isinstance(v, float) and not math.isfinite(v) else v for k, v in values.items()}
        file.write(json.dumps(values, allow_nan=False) + "\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            directory = Path(args.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.identity.setdefault("run", directory.name)
            self.identity.setdefault("seed", args.seed)
            path = directory / self.filename
            tmp = path.with_suffix(".jsonl.tmp")
            with tmp.open("w", encoding="utf-8") as file:
                for row in state.log_history:
                    self.write(file, row)
            tmp.replace(path)

    def on_log(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            with (Path(args.output_dir) / self.filename).open("a", encoding="utf-8") as file:
                self.write(file, state.log_history[-1])


class PlotState(TrainingLog):
    def __init__(self, trainer, cfg):
        self.trainer = trainer
        self.cfg = cfg
        detailed = cfg.get("plot_states", False)
        self.hidden_samples = int(cfg.get("plot_hidden_samples", 16)) if detailed else 0
        if self.hidden_samples < 0:
            raise ValueError("plot_hidden_samples must be nonnegative")
        self.capture = False
        self.sample_hidden = False
        self.hidden = {}
        self.block_names = []
        self.routed = {}
        self.branch_counts = {}
        self.token_mask = None
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
        super().__init__({"run": cfg.get("name"), "method": cfg.get("paper_method") or
                          (residual or {}).get("mode", "original_mor"), "router_type": cfg.mor.type,
                          "k": cfg.recursive.num_recursion, "seed": trainer.args.seed})

    def observe_block(self, key, x, raw, y):
        if not (self.capture and self.sample_hidden):
            return
        b, s, _ = x.shape
        n = min(self.hidden_samples, b * s)
        index = torch.arange(n, device=x.device) * (b * s) // n
        bx, sx = index // s, index % s
        x, raw, y = (v.detach()[bx, sx].float() for v in (x, raw, y))
        moments = [x.square().mean(), y.square().mean(), (raw - x).square().mean(), (y - x).square().mean()]
        self.hidden[key] = torch.stack([m * n for m in moments] + [x.new_tensor(n)])

    def observe_token_choices(self, key, choices, counts):
        if not self.capture:
            return
        k = self.identity["k"]
        if self.token_mask is not None and self.token_mask.ndim == 2:
            counts = choices.new_zeros(k).scatter_add_(
                0, choices.reshape(-1), self.token_mask.bool().long().reshape(-1))
        elif counts is None:
            counts = torch.bincount(choices.reshape(-1), minlength=k)
        self.branch_counts[key] = self.branch_counts.get(key, 0) + counts

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
        self.block_names.clear()
        for name, module in self.trainer.model.named_modules():
            if getattr(module, "mor_type", None) == "expert":
                stack.enter_context(module.register_forward_hook(partial(self.observe_router, name)))
            elif getattr(module, "mor_type", None) == "token":
                module._observe_token_choices = partial(self.observe_token_choices, name)
                stack.callback(delattr, module, "_observe_token_choices")
            elif self.hidden_samples and isinstance(module, LlamaDecoderLayer):
                self.block_names.append(name)
                module._observe_recursive_update = partial(self.observe_block, name)
                stack.callback(delattr, module, "_observe_recursive_update")

    def on_train_begin(self, args, state, control, **kwargs):
        super().on_train_begin(args, state, control, **kwargs)
        state.logging_steps = math.ceil(args.logging_steps * state.max_steps) if args.logging_steps < 1 else args.logging_steps
        self.step = state.global_step
        self.tokens = sum(r.get("input_tokens", 0) for r in state.log_history)
        self.work = sum(r.get("executed_token_layers", 0) for r in state.log_history)
        self.clock = time.monotonic()

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
        if self.branch_counts:
            counts = torch.stack(list(self.branch_counts.values()))
            totals = trainer.accelerator.reduce(counts, reduction="sum").tolist()
            for key, exits in zip(self.branch_counts, totals):
                total = sum(exits)
                reaching = total
                for depth, count in enumerate(exits, 1):
                    branch = f"router/{key}/depth_{depth}"
                    logs[f"{branch}/exit_tokens"] = count
                    logs[f"{branch}/exit_fraction"] = count / total if total else 0.0
                    logs[f"{branch}/reach_tokens"] = reaching
                    logs[f"{branch}/reach_fraction"] = reaching / total if total else 0.0
                    reaching -= count
            self.branch_counts.clear()
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
        if self.block_names:
            # Different ranks may skip different blocks. Reduce in model order with zero weight for absent samples.
            zero = torch.zeros(5, device=trainer.args.device)
            values = torch.stack([self.hidden.get(key, zero) for key in self.block_names])
            for key, (*moments, n) in zip(self.block_names, trainer.accelerator.reduce(values, reduction="sum").tolist()):
                if n:
                    for field, moment in zip(("input_rms", "output_rms", "raw_update_rms", "update_rms"), moments):
                        logs[f"hidden/{key}/{field}"] = math.sqrt(moment / n)
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
        if trainer.track_token_layers:
            self.tokens += logs["input_tokens"]
            self.work += logs["executed_token_layers"]
            logs.update(cumulative_input_tokens=self.tokens, cumulative_executed_token_layers=self.work,
                        tokens_per_second=logs["input_tokens"] / max(seconds, 1e-9))
        logs.update(interval_optimizer_steps=steps, interval_seconds=seconds,
                    gradient_accumulation=trainer.args.gradient_accumulation_steps,
                    world_size=trainer.args.world_size, per_device_batch_size=trainer._train_batch_size)
        self.step, self.clock = state.global_step, now
