"""Commit a consumed data cursor after the model, optimizer and all ranks' RNG states."""
import json
from pathlib import Path

import torch
from transformers import TrainerCallback
from transformers.trainer_utils import rotate_checkpoints

from lm_dataset.language_modeling_dataset import LanguageModelingDataset
from util.resume import checkpoint_step


class ResumeCallback(TrainerCallback):
    def __init__(self, trainer, checkpoint):
        self.trainer = trainer
        args = trainer.args
        self.dataset = trainer.train_dataset
        self.geometry = dict(world_size=args.world_size, batch_size=trainer._train_batch_size,
                             accumulation=args.gradient_accumulation_steps, workers=args.dataloader_num_workers,
                             device=args.device.type, distributed=trainer.accelerator.distributed_type.value)
        self.packed = isinstance(self.dataset, LanguageModelingDataset) and args.dataloader_num_workers == 0
        if self.packed:
            # Sized corpora use Trainer's ordinary epoch boundaries and sampler replay.
            try:
                len(self.dataset)
            except TypeError:
                self.dataset.track_position = trainer.is_world_process_zero()
                if checkpoint:
                    self.dataset.commit_position(self.consumed(checkpoint_step(Path(checkpoint))))
            else:
                self.packed = False
        self.restore_rng = None
        self.direct_resume = False
        if checkpoint and (Path(checkpoint) / "resume.json").is_file():
            path = Path(checkpoint)
            manifest = json.loads((path / "resume.json").read_text())
            if self.geometry != manifest["geometry"]:
                raise ValueError(f"Resume needs the saved loader and device geometry: {manifest['geometry']}")
            if manifest["data_mode"] == "packed":
                if not self.packed:
                    raise ValueError("The saved packed cursor needs the original streaming dataset")
                self.dataset.load_state_dict(torch.load(path / "dataset.pt", map_location="cpu", weights_only=True))
                args.ignore_data_skip = True
                self.restore_rng = str(path)
                self.direct_resume = True
        self.keep = args.save_total_limit

    def on_step_begin(self, args, state, control, **kwargs):
        if self.restore_rng:
            # Opening a fresh DataLoader iterator draws a PyTorch seed. Undo that draw
            # after the first batch is available and before the first resumed forward.
            self.trainer._load_rng_state(self.restore_rng)
            self.restore_rng = None

    def consumed(self, step):
        g = self.geometry
        return step * g["accumulation"] * g["batch_size"] * g["world_size"]

    def on_step_end(self, args, state, control, **kwargs):
        if self.direct_resume:
            state.epoch = state.global_step / state.max_steps
        if self.packed and state.is_world_process_zero:
            self.dataset.commit_position(self.consumed(state.global_step))
        if control.should_save:
            # Close partial log intervals and retain the last good checkpoint until commit.
            control.should_log = True
            args.save_total_limit = None
            if state.is_world_process_zero:
                directory = Path(args.output_dir, f"checkpoint-{state.global_step}")
                directory.mkdir(parents=True, exist_ok=True)
                (directory / ".saving").touch()

    def on_save(self, args, state, control, **kwargs):
        directory = Path(args.output_dir, f"checkpoint-{state.global_step}")
        self.trainer.accelerator.wait_for_everyone()
        if state.is_world_process_zero:
            if self.packed:
                cursor = self.dataset.state_dict(self.consumed(state.global_step))
                torch.save(cursor, directory / "dataset.pt")
            files = {str(p.relative_to(directory)): p.stat().st_size for p in directory.rglob("*")
                     if p.is_file() and p.name not in (".saving", "resume.json", "resume.tmp")}
            manifest = dict(version=1, global_step=state.global_step, geometry=self.geometry,
                            data_mode="packed" if self.packed else "replay", files=files)
            if self.packed:
                manifest["consumed_packed_samples"] = self.consumed(state.global_step)
            temporary = directory / "resume.tmp"
            temporary.write_text(json.dumps(manifest, indent=2))
            temporary.replace(directory / "resume.json")
            (directory / ".saving").unlink(missing_ok=True)
            if self.keep:
                rotate_checkpoints(output_dir=args.output_dir, save_total_limit=self.keep,
                                   best_model_checkpoint=state.best_model_checkpoint, use_mtime=True)
        args.save_total_limit = self.keep
        self.trainer.accelerator.wait_for_everyone()
