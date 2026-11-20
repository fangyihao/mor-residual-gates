"""Recursive optimizer groups and routing objectives for the Transformers Trainer."""
import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback

from model.base_model.modeling_llama import LlamaDecoderLayer
from util.plot_state import PlotState


class RecursiveTrainer(Trainer):
    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model if model is None else model
        decay_names = set(self.get_decay_parameter_names(model))
        decay, no_decay, gates, sampling = [], [], [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".recursive_gate"):
                gates.append(param)
            elif "mlp_router" in name:
                sampling.append(param)
            elif name in decay_names:
                decay.append(param)
            else:
                no_decay.append(param)
        groups = [
            {"params": decay, "weight_decay": self.args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if gates:
            groups.append({"params": gates, "weight_decay": 0.0, "name": "recursive_gates"})
        if sampling:
            groups.append({"params": sampling, "weight_decay": self.args.weight_decay,
                           "lr": self.args.learning_rate * 10, "name": "sampling_router"})
        cls, kwargs = self.optimizer_cls_and_kwargs or self.get_optimizer_cls_and_kwargs(self.args, model)
        self.optimizer = cls(groups, **kwargs)
        return self.optimizer


class RouterCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, model, **kwargs):
        trainer = self.trainer
        if control.should_save or control.should_training_stop:
            # Keep the final partial interval, including at a resumable checkpoint.
            control.should_log = True
        if trainer.ratio_count:
            ratios = trainer.accelerator.reduce(trainer.ratio_sum / trainer.ratio_count, reduction="mean")
            ratios = ratios.reshape(-1, trainer.cfg.recursive.num_recursion)
            if trainer.cfg.mor.token.balancing == "loss_free":
                updates = (ratios.mean(dim=-1, keepdim=True) - ratios).sign()
                routers = [m for m in model.modules() if hasattr(m, "router_bias")]
                with torch.no_grad():
                    for router, update in zip(routers, updates, strict=True):
                        router.router_bias.add_(trainer.cfg.mor.token.u * update)
            trainer.record("balancing_entropy", -(ratios * ratios.clamp_min(1e-10).log()).sum(-1).mean())
            trainer.ratio_sum = 0
            trainer.ratio_count = 0

    def on_save(self, args, state, control, model, **kwargs):
        if args.should_save:
            steps = {name: m.training_step for name, m in model.named_modules() if getattr(m, "mor", False)}
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}" / "router_steps.json"
            path.write_text(json.dumps(steps))


class MoRTrainer(RecursiveTrainer):
    def __init__(self, *args, cfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.plot = PlotState(self, cfg)
        self.add_callback(self.plot)
        self.track_token_layers = cfg.get("track_token_layers", False) or cfg.get("plot_states", False)
        self.metric_sums = defaultdict(float)
        self.metric_counts = defaultdict(int)
        self.ratio_sum = 0
        self.ratio_count = 0
        self.add_callback(RouterCallback(self))
        self.batch_token_layers = None
        self.interval_token_layers = 0
        self.interval_input_tokens = 0

    def train(self, *args, **kwargs):
        with ExitStack() as hooks:
            self.plot.install(hooks)
            if self.track_token_layers:
                for block in self.model.modules():
                    if isinstance(block, LlamaDecoderLayer):
                        hooks.enter_context(block.register_forward_pre_hook(self._count_decoder_inputs))
            return super().train(*args, **kwargs)

    def _count_decoder_inputs(self, _module, args):
        if self.batch_token_layers is not None:
            batch, length, _ = args[0].shape
            self.batch_token_layers += batch * length

    def _load_optimizer_and_scheduler(self, checkpoint):
        # v4 saved the auxiliary router's AdamW state in a separate file.
        # Join its disjoint parameter IDs with the main state during HF restore.
        sampling_path = Path(checkpoint, "sam_optimizer.pt") if checkpoint else None
        hook = None
        has_sampling = self.optimizer is not None and any(
            group.get("name") == "sampling_router" for group in self.optimizer.param_groups
        )
        if has_sampling and sampling_path is not None and sampling_path.is_file() and not self.is_deepspeed_enabled:
            sampling = torch.load(sampling_path, map_location="cpu", weights_only=True)
            def merge_sampling_state(optimizer, state):
                if len(state["param_groups"]) == len(optimizer.param_groups):
                    return state
                group, = sampling["param_groups"]
                start = sum(len(g["params"]) for g in state["param_groups"])
                ids = {old: start + i for i, old in enumerate(group["params"])}
                state["param_groups"].append(dict(group, name="sampling_router", params=list(ids.values())))
                state["state"].update({ids[k]: v for k, v in sampling["state"].items()})
                return state
            optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            hook = optimizer.register_load_state_dict_pre_hook(merge_sampling_state)
        try:
            super()._load_optimizer_and_scheduler(checkpoint)
        finally:
            if hook is not None:
                hook.remove()
        if hook is not None and (path := Path(checkpoint, "sam_lr_scheduler.pt")).is_file():
            scheduler = getattr(self.lr_scheduler, "scheduler", self.lr_scheduler)
            state = scheduler.state_dict()
            auxiliary = torch.load(path, map_location="cpu", weights_only=True)
            for key in ("base_lrs", "_last_lr", "lr_lambdas"):
                if key in state and key in auxiliary and len(state[key]) < len(self.optimizer.param_groups):
                    state[key] = state[key] + auxiliary[key]
            scheduler.load_state_dict(state)
        if checkpoint and (path := Path(checkpoint, "router_steps.json")).is_file():
            for name, step in json.loads(path.read_text()).items():
                self.model.get_submodule(name).training_step = step
        elif checkpoint:
            # Older checkpoints did not consistently save the router warmup clock.
            step = self.state.global_step * self.args.gradient_accumulation_steps
            state_path = Path(checkpoint, "trainer_state.json")
            if state_path.is_file():
                step = json.loads(state_path.read_text())["global_step"] * self.args.gradient_accumulation_steps
            for module in self.model.modules():
                if getattr(module, "mor", False):
                    module.training_step = step

    def record(self, name, value):
        self.metric_sums[name] = self.metric_sums[name] + value.detach()
        self.metric_counts[name] += 1

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        supplied_items = num_items_in_batch is not None
        if num_items_in_batch is None and model.training:
            num_items_in_batch = (inputs["labels"][..., 1:] != -100).sum().clamp_min(1)
        self.batch_token_layers = 0 if self.track_token_layers and model.training else None
        self.plot.capture = model.training
        self.plot.sample_hidden = not self.plot.hidden
        self.plot.token_mask = inputs.get("attention_mask")
        try:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        finally:
            self.plot.capture = False
            self.plot.token_mask = None
        lm_loss = loss.detach()
        if self.batch_token_layers is not None:
            self.interval_token_layers += self.batch_token_layers
            self.interval_input_tokens += inputs["input_ids"].numel()
            # Checkpoint recomputation happens after compute_loss returns.
            self.batch_token_layers = None
        if model.training:
            mor = self.cfg.mor
            world = self.accelerator.num_processes if self.args.average_tokens_across_devices else 1
            accumulation = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            if mor.type == "expert":
                coeff = mor.expert.get("coeff", 0.01) if mor.expert.sampling == "aux_loss" else 1.0
                # Expert BCE is already a local mean. DDP averages workers and the
                # Trainer accumulates the microbatches for this optimizer update.
                loss = loss + coeff * outputs.sampling_loss / accumulation
                stages = self.cfg.recursive.num_recursion - int(not mor.expert.get("include_first", True))
                self.record("sampling_loss", outputs.sampling_loss / max(stages, 1))
                self.record("sampling_acc", outputs.sampling_acc)
                self.record("sampling_topk_acc", outputs.sampling_topk_acc)
            elif mor.type == "token" and not mor.rand_router:
                if mor.token.balancing == "loss":
                    loss = loss + mor.token.get("coeff", 0.01) * outputs.balancing_loss * world
                    self.record("balancing_loss", outputs.balancing_loss * accumulation * world)
                self.ratio_sum = self.ratio_sum + outputs.balancing_ratio.detach()
                self.ratio_count += 1
            if mor.get("z_loss"):
                loss = loss + mor.get("z_coeff", 0.001) * outputs.router_z_loss * world
                self.record("router_z_loss", outputs.router_z_loss * accumulation * world)
            # Match the Trainer's final GA normalization in training_step.
            scaled = self.loss_is_scaled_for_ga
            if scaled is None:
                scaled = (self.model_accepts_loss_kwargs and supplied_items) or self.compute_loss_func is not None
            divisor = 1 if scaled else accumulation
            self.plot.lm = self.plot.lm + lm_loss / divisor
            self.plot.aux = self.plot.aux + (loss.detach() - lm_loss) / divisor
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            if self.metric_sums:
                means = torch.stack([value.float() / self.metric_counts[name] for name, value in self.metric_sums.items()])
                logs.update(zip(self.metric_sums, self.accelerator.reduce(means, reduction="mean").tolist()))
            if self.track_token_layers:
                counts = torch.tensor([self.interval_token_layers, self.interval_input_tokens],
                                      dtype=torch.long, device=self.accelerator.device)
                work, tokens = self.accelerator.reduce(counts, reduction="sum").tolist()
                logs.update(executed_token_layers=work, input_tokens=tokens)
                self.interval_token_layers = self.interval_input_tokens = 0
            self.plot.collect(logs, self.state)
            self.metric_sums.clear()
            self.metric_counts.clear()
        super().log(logs, start_time)
