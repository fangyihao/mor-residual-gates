from util.checkpoint import load_model_weights
import os 
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from paths import SAVE_DIR, PROJECT_ROOT, HF_CACHE_DIR; os.environ["HF_HOME"] = HF_CACHE_DIR
import json
import string
import warnings
from pathlib import Path

import random
import hydra
import torch
import torch.nn as nn 
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf 
from copy import deepcopy
from transformers import TrainingArguments, Trainer, set_seed
from accelerate import Accelerator

from lm_dataset.load_dataset import LM_DATASETS, load_dataset_from_config 
from model.util import load_model_from_config
from model.sharing_strategy import SHARING_STRATEGY
from model.relaxation.util import relax_weight_sharing
from util.config import preprocess_config
from util.tokenizer import load_tokenizer_from_config 
from util.trainer_pt import MoRTrainer, RecursiveTrainer
from util.callback import FixedStoppingCallback, EvalCallback, PeftSaveCallback, DatasetSaveCallback, ScalingLawsSaveCallback
from util.misc import print_trainable_parameters, get_latest_checkpoint_path, print_rank_zero, get_launcher_type; print_rank_zero()


@hydra.main(config_path="conf/pretrain", config_name="yymmdd_pretrain")
def main(cfg: DictConfig):
    residual = cfg.recursive.get("residual")
    if residual is not None:
        # Capture CLI overrides before preprocessing derives batch sizes and paths.
        experiment_config = OmegaConf.to_yaml(cfg, resolve=True)
        set_seed(cfg.get("seed", 42))
    cfg = preprocess_config(cfg)
    if residual is not None and int(os.environ.get("RANK", 0)) == 0:
        run_dir = Path(SAVE_DIR, "pretrain", cfg.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "training_config.yaml").write_text(experiment_config)
    
    if cfg.wandb and cfg.get("wandb_run_id") is None:
        characters = string.ascii_letters + string.digits
        wandb_run_id = "".join(random.choices(characters, k=8))
        raise KeyError(f"wandb_run_id is not set. Please set wandb_run_id as {wandb_run_id} in the config file and run again.")
                
    # wandb settings
    if cfg.get("wandb"):
        os.environ["WANDB_ENTITY"] = cfg.wandb_entity # name your W&B team
        os.environ["WANDB_PROJECT"] = cfg.wandb_project # name your W&B project
        if cfg.get("wandb_watch") is not None:
            os.environ["WANDB_WATCH"] = cfg.get("wandb_watch")
        os.environ ["WANDB_RESUME"] = "allow"
        os.environ["WANDB_RUN_ID"] = cfg.wandb_run_id
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if "WANDB_MODE" not in os.environ:
            os.environ["WANDB_MODE"] = cfg.get("WANDB_MODE", "online")
        if os.environ["WANDB_MODE"] == "offline":
            os.environ["WANDB_DIR"] = PROJECT_ROOT
        os.environ["WANDB_SAVE_CODE"] = "false"
        os.environ["WANDB LOG MODEL"] = "false"
    
    launcher_type = get_launcher_type()
    
    print ("Loading tokenizers...")
    tokenizer = load_tokenizer_from_config(cfg)

    print ("Loading dataset...")
    train_dataset = load_dataset_from_config(cfg, tokenizer)
    # The v5 Trainer replays consumed batches on resume. A dataset snapshot here
    # includes Accelerate's prefetched next batch and would skip real examples.

    print ("Loading models...")
    model = load_model_from_config(cfg)
    
    if cfg.recursive.get("enable"):        
        # KV cache sharing strategy
        model, lora_init_dict = SHARING_STRATEGY[cfg.model](cfg, model)
    
    if "kv_sharing" in cfg and cfg.kv_sharing.get("enable"):
        model.set_kv_sharing_config(cfg)
        
    if cfg.get("relaxation") and cfg.relaxation.get("enable"):
        model = relax_weight_sharing(cfg, model, lora_init_dict=lora_init_dict)
        
        if cfg.resume_from_checkpoint:
            if cfg.relaxation.get("enable"):
                latest_checkpoint = get_latest_checkpoint_path(cfg, resume_step=cfg.resume_step if ("resume_step" in cfg and cfg.resume_step is not None) else None)                
                load_model_weights(model.get_base_model(), latest_checkpoint)
                
    if "mor" in cfg and cfg.mor.get("enable"):            
        if cfg.mor.type == "expert":
            model.transform_layer_to_mor_expert(cfg)
        elif cfg.mor.type == "token":
            model.transform_layer_to_mor_token(cfg)
        else:
            raise ValueError(f"Unknown MoR type {cfg.mor.type}.")
        
    print_trainable_parameters(model)
        
    report_to = []
    if cfg.wandb:
        report_to.append("wandb")
    if cfg.tensorboard:
        report_to.append("tensorboard")
    
    ds_config = cfg.deepspeed if launcher_type == "deepspeed" else None
    if ds_config and residual is not None:
        # ZeRO wraps the Trainer's grouped AdamW. A DeepSpeed-owned optimizer
        # would put the gates back into the backbone's weight-decayed group.
        with open(ds_config) as f:
            ds_config = json.load(f)
        ds_config.pop("optimizer", None)
    if cfg.get("tensorboard"):
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(cfg.tensorboard_dir)
    train_args = TrainingArguments(
        seed=cfg.get("seed", 42),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine_with_min_lr"),
        lr_scheduler_kwargs=dict(cfg.get("lr_scheduler_kwargs", {"min_lr_rate": 0.1,})),
        learning_rate=cfg.learning_rate,
        adam_beta1=cfg.adam_beta1,
        adam_beta2=cfg.adam_beta2,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        output_dir=os.path.join(SAVE_DIR, "pretrain", cfg.output_dir),
        max_steps=cfg.num_train_steps,
        warmup_steps=cfg.num_warmup_steps,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        activation_checkpointing=cfg.gradient_checkpointing,
        activation_checkpointing_kwargs={"use_reentrant": False} if residual is not None else None,
        ddp_find_unused_parameters=True if residual is not None and cfg.get("mor", {}).get("enable") else None,
        max_grad_norm=cfg.max_grad_norm,
        dataloader_num_workers=cfg.dataloader_num_workers,
        bf16=cfg.precision == "bf16",
        fp16=cfg.precision == "fp16",
        report_to=report_to,
        run_name=cfg.wandb_run_name,
        deepspeed=ds_config,
        log_on_each_node=False,
    )
    
    callbacks = []
    fixed_save_steps = cfg.fixed_save_steps if ("fixed_save_steps" in cfg and cfg.fixed_save_steps) else None
    if cfg.stop_steps is not None:
        callbacks.append(FixedStoppingCallback(cfg.stop_steps))
    if "evaluation" in cfg and cfg.evaluation.enable:
        callbacks.append(EvalCallback(cfg, tokenizer))
    if cfg.relaxation.get("enable") and cfg.relaxation.method in ["lora", "dora", "adaption_prompt"]:
        callbacks.append(PeftSaveCallback(cfg.save_steps, fixed_save_steps=fixed_save_steps))
    if all(ds in LM_DATASETS for ds in cfg.dataset.split(',')):
        callbacks.append(DatasetSaveCallback(cfg.save_steps, fixed_save_steps=fixed_save_steps))
    if fixed_save_steps is not None:
        callbacks.append(ScalingLawsSaveCallback(fixed_save_steps,))
        
    if "mor" in cfg and cfg.mor.get("enable"):
        trainer = MoRTrainer(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks, cfg=cfg,)
    else:
        trainer_cls = RecursiveTrainer if residual is not None and residual.mode == "rezero" else Trainer
        trainer = trainer_cls(model=model, args=train_args, train_dataset=train_dataset, callbacks=callbacks,)
    
    train_result = trainer.train(
        resume_from_checkpoint=cfg.resume_from_checkpoint
    )
    metrics = train_result.metrics
    trainer.log_metrics("pretrain", metrics)
    trainer.save_metrics("pretrain", metrics)
    trainer.save_state()
    trainer.save_model()
    
    if cfg.relaxation.get("enable"):
        trainer.model.base_model.model.save_pretrained(train_args.output_dir)
    
    
if __name__ == "__main__":
    main()