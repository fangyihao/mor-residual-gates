# Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation (NeurIPS 2025)

<a href="https://arxiv.org/abs/2507.10524"><img src="https://img.shields.io/badge/Paper-arXiv:2507.10524-Green"></a>
<a href="https://github.com/raymin0223/mixture_of_recursions?tab=readme-ov-file#-bibtex"><img src="https://img.shields.io/badge/Paper-BibTex-yellow"></a>

**[Sangmin Bae](https://scholar.google.com/citations?hl=en&user=T5rHY14AAAAJ)<sup>1\*</sup> &nbsp; [Yujin Kim](https://scholar.google.com/citations?hl=en&user=17yTpxsAAAAJ&hl)<sup>1\*</sup> &nbsp; [Reza Bayat](https://scholar.google.com/citations?hl=en&user=58et-JMAAAAJ)<sup>2\*</sup> &nbsp; [Sungnyun Kim](https://scholar.google.com/citations?hl=en&user=DsWny60AAAAJ&hl)<sup>1</sup> &nbsp; [Jiyoun Ha](https://scholar.google.com/citations?hl=en&user=tTBHK54AAAAJ)<sup>3</sup> &nbsp; [Tal Schuster](https://scholar.google.com/citations?hl=en&user=oo8QRmIAAAAJ)<sup>4</sup> &nbsp; [Adam Fisch](https://scholar.google.com/citations?hl=en&user=LYRkQhMAAAAJ&hl)<sup>4</sup>  &nbsp; [Hrayr Harutyunyan](https://scholar.google.com/citations?hl=en&user=GaCGz8wAAAAJ)<sup>5</sup>  &nbsp; [Ziwei Ji](https://scholar.google.com/citations?hl=en&user=3l_6H5sAAAAJ)<sup>4</sup> &nbsp; [Aaron Courville](https://scholar.google.com/citations?hl=en&user=km6CP8cAAAAJ)<sup>2,6&dagger;</sup> &nbsp; [Se-Young Yun](https://scholar.google.com/citations?hl=en&user=X_IAjb8AAAAJ)<sup>1&dagger;</sup>**.  
<sup>**1**</sup>KAIST AI &nbsp; <sup>**2**</sup>Mila &nbsp; <sup>**3**</sup>Google Cloud &nbsp;  <sup>**4**</sup>Google DeepMind &nbsp; <sup>**5**</sup>Google Research &nbsp; <sup>**6**</sup>Université de Montréal    
\*Equal contribution. &nbsp; &dagger;Corresponding authors.

<p align="left">
<img width="700" src="asset/mor_fig1.png">
</p>


## Learnable ReZero-style recursive residual gates

The proposed experiment continues pretraining from SmolLM-360M. Each physical decoder layer runs once on every token. The router can then select up to $K-1$ extra executions of that layer. With sequence sharing, these executions are consecutive, and every token starts the next physical layer even if it exited the previous layer early.

For each selected extra execution $j$, the whole attention-plus-MLP block returns $h + c_{l,j}(F_l(h)-h)$. The first execution always returns $F_l(h)$. The proposed method learns one independent, unconstrained scalar per physical layer and extra execution. The gates start at zero and have a dedicated AdamW parameter group with zero weight decay. A selected block still executes at zero so its gate receives a gradient.

| Experiment | `recursive.residual.mode` | Extra-execution coefficient |
| --- | --- | --- |
| **Proposed ReZero-style gates** | `rezero` | Learned $\alpha_{l,j}$, initialized to zero |
| Stage-shared ReZero ablation | `rezero_stage` | Learned $\alpha_j$, shared across physical layers and initialized to zero |
| Full-strength ablation | `full` | $1$ |
| Uniform ablation | `uniform` | $\Lambda_R/(K-1)$ |
| Normalized power-law ablation | `power` | $\Lambda_R j^{-\zeta}/\sum_{r=2}^{K} r^{-\zeta}$ |

For example, with $K=3$, $\Lambda_R=1$ and $\zeta=1$, the extra-stage coefficients are $(0.5,0.5)$ for uniform and $(0.6,0.4)$ for power. `lambda_r` sets the sum of update coefficients across the available extra stages. The actual perturbation also depends on the block outputs and which stages a token reaches. Router capacity controls how many tokens execute a stage.

The experiment preset uses `recursive.initialization=pretrained` to retain every source layer and tie its weights across executions. With $L$ source layers and sequence sharing, `rezero` learns $L(K-1)$ independent scalars. The `rezero_stage` ablation ties together all scalars at the same extra stage, giving $K-1$ trainable scalars in total. Gradients from every selected layer accumulate into the same stage parameter. Both modes use the same gate initialization, learning rate and zero weight decay. Zero gates reproduce the source model's one-pass representation at initialization. All five modes write the selected block's returned state back to the token. The original MoR presets use their published router-weighted scatter-add convention. In these residual modes hard selection trains the router through its auxiliary objective. The preset uses expert choice with `aux_loss`. Token choice is available with `mor.type=token` and its balancing loss.

### Running the recursive residual experiments

Prepare FineWeb-Edu and the environment as described below. The fork reads the model and tokenizers from its local Hugging Face cache. To populate that cache on a fresh checkout, run from the repository root.

```bash
HF_HOME="$PWD/hf_cache" python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

snapshot_download("HuggingFaceTB/SmolLM-360M", allow_patterns=["config.json", "*.safetensors", "*.index.json", "pytorch_model.bin"])
for model in ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM2-135M"):
    AutoTokenizer.from_pretrained(model)
PY
```

To train the proposed layer-specific gates on a single GPU with the preset defaults, run:

```bash
bash scripts/pretrain.sh accelerate offline 0 recursive_residual
```

Set `recursive.num_recursion`, `recursive.residual.lambda_r` and `recursive.residual.zeta` on the command line to change the schedules. Supply a distinct `name=...` when sweeping coefficients at the same mode and seed, since the training entry point resumes existing run directories. Each run saves its resolved input configuration next to its checkpoints. For the default proposal, evaluate held-out FineWeb loss with this command.

```bash
python evaluate_fineweb_test.py --exp_names mor_expert_sequence_k3_rezero_seed1234 --sample_number 200
```

To reload a routed checkpoint saved with Transformers 5.23.0 directly from Python:

```python
from model.mor_model.modeling_llama import MoRLlamaForCausalLM

model = MoRLlamaForCausalLM.from_pretrained(
    "results/pretrain/mor_expert_sequence_k3_rezero_seed1234", attn_implementation="sdpa"
).eval()
```

Run the CPU regression suite with `python -m pytest -q tests/test_recursive_residual.py`. It uses tiny local Llama checkpoints to test pretrained preservation, sparse routing, signed gates, fixed coefficients, checkpoint restoration, and optimizer grouping.


### Compare ReZero with the published MoR result

The [LaTeX manuscript](paper/recursive_residual_gates.tex) documents the methods and evaluation protocol, with placeholders for the study results. Build it from `paper/` with `latexmk -pdf recursive_residual_gates.tex`.

Use `--paper-comparison` for the main experiment. It inherits the authors’ released SmolLM-360M configuration for the **Table 3 fixed-FLOPs, Expert / Cache / Middle-Cycle / K=3** row. The published MoR model has 118M non-embedding parameters and scores **2.7925 FineWeb NLL** and **42.6%** mean downstream accuracy after approximately 30B tokens at $16.5 \times 10^{18}$ training FLOPs. The released recipe trains from scratch on FineWeb-Edu for 14,161 optimizer steps, with 1,024 sequences of 2,048 tokens per global batch (29,697,769,472 input tokens). The pretrained sequence-sharing experiment above addresses the separate question of preserving a pretrained first pass. [Paper, Table 3 and Appendix D](https://papers.nips.cc/paper_files/paper/2025/file/8b08bbf8b420faa6eeb4020720582ec7-Paper-Conference.pdf) · [Released training configuration](https://github.com/raymin0223/mixture_of_recursions/blob/82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6/conf/pretrain/250720_pretrain_smollm-360m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001.yaml)

| Local method | What the comparison tests |
|---|---|
| `rezero` | Learns one zero-initialized scalar per physical layer and extra stage. This is the proposed method. |
| `rezero_stage` | Shares one scalar across physical layers at each extra stage. Comparing it with `rezero` measures the value of layer specificity. |

The default trains the two gated methods on seeds 1234, 1235 and 1236 (six full runs) with gradient checkpointing and automatic resume. Prepare the model config, tokenizer and both FineWeb datasets as described in this README. Downstream evaluation also requires access to the benchmark datasets or their local cache. The released harness includes legacy dataset loaders. Use `python -m pip install 'datasets==3.6.0'` to reproduce the loader version tested with this suite.

```bash
# Train and evaluate the main comparison on eight Ascend NPUs
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --paper-comparison

# Equivalent suite on two GPUs
bash scripts/pretrain.sh accelerate offline 0,1 all -- --paper-comparison

# Both full-budget runs and the 15 matched early evaluations per seed
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --paper-comparison --ablation-steps 512

# Run only the short panel (512 updates by default)
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --paper-comparison --ablations-only

# Score full-budget checkpoints, including a local copy of the authors’ matching native MoR weights
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --paper-comparison --eval-only \
  --author-checkpoint /path/to/original-mor-360m-k3-expert-cache
```

The **main comparison uses $K=3$**. The **short ablations use $K=3,4,6$**. Here $K$ counts the first execution and all possible extra executions. The $K=3$ and $K=4$ settings use their released Table 3 recipes, including 34 virtual layers at $K=4$. The local $K=6$ extension inherits the $K=3$ training recipe. With two unshared end layers, the 32-layer SmolLM-360M backbone has a body of 30 virtual layers, which gives five physical blocks reused six times. Within each depth, every short run uses the same initialization recipe, seed, input-token budget, router capacities and learning-rate schedule. Across depths the physical model size and executed work also change. [SmolLM-360M model configuration](https://huggingface.co/HuggingFaceTB/SmolLM-360M/blob/main/config.json)

| Method | Short FineWeb evaluation at each of $K=3,4,6$ | Full FineWeb and downstream at $K=3$ |
|---|---:|---:|
| Layer-specific ReZero | Step 512 | Step 14,161 |
| Stage-shared ReZero | Step 512 | Step 14,161 |
| Uniform scaling | Step 512 | — |
| Power-law scaling | Step 512 | — |
| Native MoR (released router-weighted writeback) | Step 512 | Published reference |

At the default decay exponent of one, uniform and power-law scaling differ at all three depths. At $K=6$ the ReZero variants have five extra stages after the ungated first execution. The layer-specific model learns 25 gates (five physical blocks times five extra stages), and the stage-shared model learns five. Compare the $K=6$ methods against the locally trained native MoR run at the same depth and token budget.

The combined command uses 15 trajectories per seed. The first 512 updates of the two main $K=3$ runs also supply their short-panel checkpoints. The other 13 trajectories stop at 512. At the released batch size this is 1,073,741,824 training tokens per short measurement, including 86 warmup updates. Report these results as early-training NLL. The shorter panel supports within-depth comparisons of optimization and representation drift. Use the full runs for end-of-training claims. Native MoR retains the authors’ router-weighted additive writeback. Its comparison with ReZero includes the change in writeback as well as residual scaling.

To reduce the budget further, specify e.g. `--ablations-only --ablation-steps 256` (536,870,912 tokens per trajectory), or select just the deeper settings with `--ablation-ks 4,6`. Earlier $K=2$ runs remain selectable with `--ablation-ks 2`. Use the combined command from the start to retain the exact early checkpoint, or run the short panel first and repeat the command without `--ablations-only`, explicitly supplying the same `--ablation-steps`, to continue both main models. An existing long run can supply the short measurement if it retained weights at the requested step.

Repeat the original command to resume training. Use `--train-only` to defer evaluation, `--eval-only` to score saved checkpoints at the requested step, or `--nll-only` to defer downstream scoring. Repeat the same training overrides in evaluation-only mode. For a separate pilot, use a distinct `name=...` prefix. The optional authors’ checkpoint directory accepts Hugging Face safetensors or PyTorch weights, including shards. Select the fixed-FLOPs K=3 checkpoint from the authors’ release and retain its provenance. The runner records the supplied weight hashes. [Authors’ checkpoint release](https://drive.google.com/drive/folders/1pYKJOu2aBGC-jgoWbfP6T_vqEYtUVxa4?usp=drive_link)

The default output `paper_mor_fineweb.paper.md` contains the reported MoR row and the local means, sample standard deviations, seed counts and measured training tokens. The adjacent `.paper.json` includes per-seed results, paired differences, trainable gate counts, executed token-layers and differences from the released training recipe. The short panel also writes `paper_mor_fineweb.step-512.ablations.csv` with one row per depth, seed and method. Its signed `nll_minus_native` pairs each method with a native run at the same depth, seed, step and measured training-token count. Negative values favor the method. A missing match leaves the field blank. The adjacent `.data.json` records the held-out token fingerprint. Use the full-budget paired results for end-of-training claims. The native baseline in the report is the published measurement. Optionally rescore the authors’ weights with `--author-checkpoint` to check for evaluation differences without training a baseline. Comparisons against published MoR describe the overall method difference. Comparing the two ReZero variants isolates gate sharing. To measure the contribution of adding a gate under identical writeback, include a fixed full-strength control in a separate ablation.

#### Plot training and recursion dynamics

Paper runs write one rank-zero `results/pretrain/<run>/training_metrics.jsonl` file each. Training records include optimizer step, seed, method, $K$, total loss, LM loss, weighted auxiliary loss, global gradient norm, optimizer-group learning rates and weight decay, and the mixed-precision loss scale when used. Token counts, executed token-layers, throughput, global worker count, per-device batch size and accumulation are recorded for each interval. Use `cumulative_input_tokens` to align runs by data exposure.

Expert-router records include per-stage selected-token counts, reach fractions, sampling losses and accuracies where defined. For Middle-Cycle, `depth/<j>/fraction` is the fraction of tokens that stopped at exactly $j$ executions. Each independent learnable scalar has a signed `gate/<parameter>` and a `gate_grad/<parameter>` when the optimizer exposes its clipped gradient. Fixed schedules record `epsilon/stage_<j>`. Per-block `hidden/<module>/input_rms`, `output_rms`, `raw_update_rms` and `update_rms` measure the block input, output, raw repeated-block change and applied change. These RMS measurements sample up to 16 evenly spaced active positions on the first microbatch of each interval per worker. For native MoR the block-output sample precedes router weighting. Use `router/<module>/writeback_rms` for the actual stage-level change at sampled selected token positions in every method.

The paper presets log every ten optimizer steps. The separate pretrained `recursive_residual` preset also accepts `plot_states=true`. Pass `logging_steps=1` for every update, `plot_hidden_samples=0` to skip activation sampling, or `plot_states=false` to disable the additional plotting records. Checkpoint and stop steps also flush interval counters. Resuming rebuilds the JSONL through the restored checkpoint and appends subsequent observations. The full model, optimizer, scheduler, available scaler, per-rank RNG and dataset continuation state are saved by the checkpoint machinery. Earlier checkpoints keep their original metric history.

```python
import pandas as pd

df = pd.read_json(
    "results/pretrain/paper_mor_k3_rezero_seed1234/training_metrics.jsonl", lines=True
)
training = df.loc[df["event"].eq("train")].set_index("cumulative_input_tokens")
training[["lm_loss", "weighted_aux_loss"]].plot(subplots=True)
training.filter(regex=r"^gate/").plot(legend=False)
```

For the main manuscript, insert `\input{paper_mor_fineweb.table3-rows.tex}` immediately before the final rule in Table 3. With shorter ablations the fragment is named after the full-step output. It appends **Layer-specific ReZero** and **Stage-shared ReZero** in the original 16-column order. The cells contain seed means, measured non-embedding parameter counts rounded to millions, and the actual training tokens in billions. Include the seed count in the caption and retain per-seed results in the supplement. A pilot, an NLL-only evaluation, missing work records or unpaired gated runs produce a comment explaining what is missing in place of publishable rows.

Mark the two new rows as local measurements and attribute the historical rows to Bae et al. The asterisk in the FLOPs column identifies the local forward-only estimate in units of $10^{18}$. Suggested caption text: “Local FLOPs sum decoder, router and vocabulary projections and causal attention at the executed stage lengths. Elementwise operations, backward passes and checkpoint recomputation are excluded.” The JSON gives the components for each gated model. Interpret comparisons with the 30B row as token-matched. Inspect the locally counted FLOPs alongside the published 16.5 before claiming matched compute. [Original forward-FLOP convention, Appendix F](https://papers.nips.cc/paper_files/paper/2025/file/8b08bbf8b420faa6eeb4020720582ec7-Paper-Conference.pdf)

The scorer fingerprints the ordered FineWeb token IDs for the same finite prefix (500 packed sequences by default). Downstream evaluation uses LAMBADA, HellaSwag, PIQA, WinoGrande, ARC and MMLU continuation. ARC averages Easy and Challenge equally, and the overall mean gives each of the six task families equal weight. HellaSwag, PIQA and both ARC tasks use target byte-length normalization. The released task definitions resolve to zero shots by default. Appendix D does not give per-task shot counts or identify the precise held-out prefix. The report records the local shot counts, sample counts and task versions. If an evaluation setting is independently confirmed, `--benchmark-shots TASK=N` applies it to every arm, including the supplied authors’ checkpoint. Interpret paper-to-local gaps in light of this calibration. The released expert-choice router selects tokens across the sequence, so these scores describe offline held-out scoring. Claims about strictly causal decoding require a separate evaluation.

### Run all five pretrained experiments

The `all` option trains `rezero`, `rezero_stage`, `full`, `uniform` and `power` sequentially for each of three paired seeds (1234, 1235 and 1236). The default suite uses $K=8$, gradient checkpointing and token-layer tracking. After all 15 training runs finish, it evaluates every checkpoint on the same 200 packed FineWeb samples. A failed run stops the suite and returns a nonzero exit status.

```bash
# Eight Ascend NPUs
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7

# Two GPUs with Accelerate
bash scripts/pretrain.sh accelerate offline 0,1 all

# One seed, five runs, with a custom checkpoint prefix
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --seeds 1234 name=pilot

# Train all five modes at the original single-GPU preset depth and seed
bash scripts/pretrain.sh accelerate offline 0 all -- seed=1234 recursive.num_recursion=3 --train-only

# Two paired seeds on GPUs, with a larger evaluation set
bash scripts/pretrain.sh accelerate offline 0,1 all -- --seeds 1234,1235 --sample-number 500
```

Use `--train-only` to stop after training. Shared Hydra overrides such as `recursive.num_recursion=4` or `recursive.residual.lambda_r=0.5` apply to every run. On the GPU launcher, pass suite options and Hydra overrides after `--`. On multiple Ascend hosts, run the same command on each host with its usual rendezvous settings.

#### Shorter ablations at a matched training step

Use `--ablation-steps S` to compare all five methods after **S optimizer updates**. The proposed `rezero` and the stage-shared control `rezero_stage` then continue to the original full budget. The three fixed schedules stop at S. All five follow the same learning-rate schedule through S, including the original warmup and decay horizon. Gradient accumulation, seeds, training data and held-out evaluation samples follow the shared suite settings.

```bash
# Default preset, three paired seeds, two full-length runs per seed
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --ablation-steps 3540 name=short_ablation

# Re-score the saved checkpoints at both budgets
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --ablation-steps 3540 name=short_ablation --eval-only

# Use the full-strength residual update as the full-budget control on two GPUs
bash scripts/pretrain.sh accelerate offline 0,1 all -- --ablation-steps 3540 --full-budget-control full name=mor_control
```

With the preset’s 14,161-step full budget, this example uses **38,942 optimizer updates per seed**, compared with 70,805 for five full runs, a reduction of approximately 45%. Actual accelerator time depends on routing and device throughput. Choose S before inspecting held-out results. `--full-budget-control` accepts `rezero_stage`, `full`, `uniform`, `power`, or `none`. Choosing `none` stops all four controls at S. A shared `stop_steps` override sets the full-run endpoint up to `num_train_steps`. S must be smaller than that endpoint.

The default outputs in the example are `residual_suite_fineweb.step-3540.json` (five methods) and `residual_suite_fineweb.full-step-14161.json` (two methods). With `--output-file pilot.json`, the files are `pilot.json` and `pilot.full-step-14161.json`. Console output includes checkpoint steps, per-seed NLL differences and, when tracking is enabled, input tokens and executed token-layers. With at least two paired seeds it reports the mean and sample standard deviation of the paired differences. A weights-only `ablation-step-S/` snapshot survives ordinary checkpoint rotation, including `save_total_limit=1`. Resume and evaluation-only commands use the same suite arguments and training overrides. Changed budgets can reuse an exact saved checkpoint. A run that has passed S without retaining it requires a different saved comparison step or a fresh run.

For a paper, report the five-way comparison at S and the paired full-budget comparison in separate rows or tables. Report variability across seeds and the observed token-layer work. The short run tests how much a design helps by S. For a claim about the benefit of layer-specific gates at the end of training, use the full-budget layer-specific versus stage-shared pair. Use the `--paper-comparison` preset above for a full-budget comparison with the published MoR reference. [NeurIPS guidance on empirical claims and reporting](https://neurips.cc/public/guides/PaperChecklist)

The preset gives each mode and seed its own checkpoint directory. A literal `name=pilot` or `output_dir=pilot` becomes a prefix, for example `pilot_rezero_stage_seed1234`. Use the same prefix and seeds on restart.

```bash
# Resume the suite on eight Ascend NPUs (also the default on a repeat invocation).
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --auto-resume name=pilot

# Score available pilot runs only, including runs stopped after a checkpoint.
bash scripts/run_npu.sh all 0,1,2,3,4,5,6,7 --eval-only name=pilot --output-file pilot_eval.json

# GPU suite or an individual experiment
bash scripts/pretrain.sh accelerate offline 0,1 all -- --auto-resume name=gpu_pilot
bash scripts/pretrain.sh accelerate offline 0,1 all -- --eval-only name=gpu_pilot
bash scripts/run_npu.sh train 0,1 auto_resume=true name=one_run
bash scripts/run_npu.sh eval 0,1 --exp_names one_run --latest-saved --strict --sample_number 200
```

**Resuming.** Each completed save records the model and gates, optimizer, learning-rate scheduler, active AMP scaler, trainer step and logs, router counters, and each rank’s RNG state. With the default zero-worker FineWeb loader, the checkpoint also records the next **consumed** packed-sample position, the token remainder, the current packed block, the source position, and the local shuffle generator. This accounts for loader prefetch and gradient accumulation. With global shuffling, the loader reconstructs the current source epoch’s shuffle buffer from its seed and document offset. Multi-worker loaders, map-style datasets and compatible older checkpoints use deterministic Trainer replay to reach the saved batch. Keep the input data, tokenizer, training recipe, effective batch and device topology the same across restarts.

Restarting continues from the latest complete optimizer-step checkpoint and repeats work done after it. Set `save_interval=null save_steps=100`, for example, to bound this work to 100 updates. A planned `stop_steps` also saves at its stopping step. You can raise or remove `stop_steps` while keeping the original `num_train_steps` and scheduler. The suite skips finished runs. Use `--no-auto-resume` (single run: `auto_resume=false`) to require a fresh directory for unfinished training.

**Evaluation only.** `--eval-only` scores the newest available saved weights per matching run on the same FineWeb sample prefix. It lists absent runs, reports the evaluated step, and prints paired ReZero differences only for matching steps and seeds. The JSON maps run names to losses. For one run, use the individual evaluation command above.

Held-out losses are saved in `residual_suite_fineweb.json` in the repository root. Set `--output-file FILE` to choose a different path. The report prints NLL for each run, total input tokens and executed token-layers when tracking is enabled, and a paired loss difference for each seed. Negative layer-specific minus stage-shared NLL favors `rezero`. Token-layer counts include compacted-batch padding and exclude backward recomputation. Compare matched totals and use separate validation data when choosing fixed schedule coefficients.

### Paired stage-sharing comparison at $K=8$

The example below runs three paired seeds with the same expert router, token budget and optimizer on eight Ascend NPUs. It compares seven shared stage scalars, seven independent scalars per physical layer, and a fixed schedule. The example uses uniform scaling with $\Lambda_R=1$. Choose the fixed schedule and its coefficients on validation data before scoring the held-out FineWeb set.

```bash
set -e
devices=0,1,2,3,4,5,6,7
fixed_mode=uniform
fixed_args=(recursive.residual.lambda_r=1.0)
runs=()
for seed in 1234 1235 1236; do
    for mode in "$fixed_mode" rezero_stage rezero; do
        run="gate_sharing_k8_${mode}_seed${seed}"
        args=()
        if [[ "$mode" == "$fixed_mode" ]]; then args=("${fixed_args[@]}"); fi
        bash scripts/run_npu.sh train "$devices" name="$run" seed="$seed" \
            recursive.num_recursion=8 recursive.residual.mode="$mode" \
            gradient_checkpointing=true track_token_layers=true "${args[@]}"
        runs+=("$run")
    done
done
bash scripts/run_npu.sh eval "$devices" --exp_names "${runs[@]}" \
    --sample_number 200 --output_file stage_gate_fineweb.json

python - "${runs[@]}" <<'PY'
import json
from pathlib import Path
import sys

losses = json.loads(Path("stage_gate_fineweb.json").read_text())
names = sys.argv[1:]
for fixed, shared, proposed in zip(names[::3], names[1::3], names[2::3], strict=True):
    for name in (fixed, shared, proposed):
        state = json.loads((Path("results/pretrain") / name / "trainer_state.json").read_text())
        rows = [row for row in state["log_history"] if "loss" in row]
        tokens = sum(row["input_tokens"] for row in rows)
        work = sum(row["executed_token_layers"] for row in rows)
        print(f"{name}  NLL={losses[name]:.7f}  tokens={tokens}  token-layers={work}")
    print(f"{proposed}  layer-specific minus stage-shared NLL={losses[proposed] - losses[shared]:+.7f}")
PY
```

Compare runs with the same measured input-token and token-layer totals. The training logs sum both counts across all workers. One executed token-layer is one position presented to a decoder block during the training forward pass. This includes compacted-batch padding. The count excludes backward recomputation. Use the same held-out samples for every run and report the per-seed loss difference $\mathrm{NLL}_{\text{rezero}}-\mathrm{NLL}_{\text{rezero\_stage}}$. Negative values favor layer-specific gates.

### Gradient checkpointing

Append `gradient_checkpointing=true` to a training command to save activation memory by recomputing decoder blocks during backward. The default is `false`. The same option works for both learned gate modes and the three fixed ablations. Each router selects tokens and updates its warmup counter once per forward pass. Checkpointed training uses full input sequences with the KV cache disabled.

```bash
# Two GPUs
bash scripts/pretrain.sh accelerate offline 0,1 recursive_residual -- gradient_checkpointing=true name=gpu_rezero_gc

# Eight Ascend NPUs
bash scripts/run_npu.sh train 0,1,2,3,4,5,6,7 gradient_checkpointing=true name=ascend_rezero_gc
```

### Ascend NPU training and FineWeb evaluation

Use an Ascend environment with the driver and CANN runtime active. Install a matching PyTorch (2.6 or newer) and `torch_npu` pair for your hardware and Python version, then install the shared project dependencies. See the [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).

```bash
python -m pip install -r requirements-base.txt
bash scripts/run_npu.sh check 0,1,2,3,4,5,6,7
bash scripts/run_npu.sh train 0,1,2,3,4,5,6,7 name=ascend_rezero
bash scripts/run_npu.sh eval 0,1,2,3,4,5,6,7 --exp_names ascend_rezero --sample_number 200
```

The device list selects visible Ascend device IDs on the current host. List them in ascending order. The launcher binds one worker to each selected device. The `check` command verifies an HCCL all-reduce across every rank before you load a model. Training uses replicated data parallelism, ordinary PyTorch AdamW, and SDPA by default. Pass `attn_implementation=eager` to training or `--attn_implementation eager` to evaluation to select eager attention. All five residual modes use this launcher. For example, append `recursive.residual.mode=power name=ascend_power` to the training command.

The default global batch of 1,024 sequences on eight NPUs uses 16 sequences per NPU and eight accumulation steps. The launcher uses the total number of workers across hosts when deriving accumulation. Expert-router BCE is averaged across workers and accumulated microbatches, so its coefficient has the same scale when splitting a batch. FineWeb evaluation distributes the requested packed samples across workers, aggregates their losses, and writes one results file from rank zero.

For two eight-NPU hosts, run the following on **each host**. Set `NODE_RANK=0` on the first and `NODE_RANK=1` on the second. Use the same reachable address and port for rank zero, model and data paths on each host, and a shared `results/` directory for checkpointing and resume.

```bash
export NNODES=2 MASTER_ADDR=host0.example.org MASTER_PORT=29500
export NODE_RANK=0  # use 1 on the second host
bash scripts/run_npu.sh train 0,1,2,3,4,5,6,7 name=ascend_rezero_16npu
```

## 🔥 Research Motivation

**Early-exiting [[CALM](https://arxiv.org/abs/2207.07061)]**, an Adaptive Computation technique, helps LLMs run more efficiently by allowing them to skip unnecessary computations when they're confident in their predictions. This creates dynamic pathways through the model. However, putting early-exiting into practice faces **two main bottlenecks**:
1.  **Missing Key-Value (KV) cache problem:** 
    When tokens exit early, they skip computing the KV pairs for remained deeper layers. But these missing values are essential for decoding future tokens, and trying to approximate them often hurts performance.   

2.  **Inefficient batched inference:** 
    Tokens that exit early end up waiting for others in the same batch to finish their full computation. This "idling" prevents efficient batching and wastes processing time.
    
Previous work tackled these challenges individually:

*  **[FREE Framework](https://arxiv.org/abs/2310.05424)** 
    addressed the missing KV cache problem using **parallel decoding** (an early form of self-speculative decoding), which efficiently computes the exact KV pairs for early-exited tokens. However, compatibility with batch inference remained poor.

*  **[Recursive Transformers](https://arxiv.org/abs/2410.20672)** 
    aimed to mitigate inefficient batched inference through **parameter sharing**, enabling tokens at different depths to be processed together. Yet, two separate training processes for integrating parameter sharing and early-exiting degraded performance, and this model still required handling the missing KV cache.


Our new research **[[MoR](https://arxiv.org/abs/2507.10524)]** introduces a **unified framework** that directly tackles both the missing KV cache and batched inference issues. We achieve this with a ✨ **routing mechanism** trained end-to-end effectively, which dynamically assigns the optimal recursion depth to each token. We further enhance this by introducing a ✨ **recursion-wise KV caching strategy** that selectively stores KV pairs, resolving the missing cache problem while optimizing memory usage.
We achieve up to **2× greater inference throughput** compared to standard transformers at similar accuracy, while also reducing total training FLOPs and memory requirements.




## 🏃 Environment Setup

To get started, follow these steps to set up your development environment. We recommend using `conda` for dependency management.

1.  **Create and Activate Conda Environment:**
    ```bash
    conda create -n mor python=3.12
    conda activate mor
    ```

2.  **Install Required Packages:**
    Install the PyTorch build for your hardware, then install this branch's requirements. The recursive residual implementation targets **Transformers 5.23.0**. Training writes safetensors checkpoints and the evaluation scripts also read previous PyTorch weight files. The recursive residual preset enables automatic checkpoint resume. See the suite instructions above for packed-stream and multi-worker behavior.

    ```bash
    python -m pip install -r requirements.txt
    # For the default flash_attention_2 configuration on a supported GPU:
    python -m pip install flash-attn --no-build-isolation
    ```

    To use PyTorch attention without FlashAttention, pass `attn_implementation=sdpa` to the experiment command.


## 📚 Dataset Download

Our models are pretrained on a deduplicated subset of the **FineWeb-Edu** dataset, available as part of the [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus).

Follow these steps to download and prepare the dataset:

1.  **Create Data Directories:**
    Create the necessary directories (`hf_cache`, `hf_datasets`, `hf_models`, and `results`) under your designated data path. Replace `{your_data_path}` with your actual path.

    ```bash
    mkdir -p {your_data_path}/mixture_of_recursions/hf_cache
    mkdir -p {your_data_path}/mixture_of_recursions/hf_datasets
    mkdir -p {your_data_path}/mixture_of_recursions/hf_models
    mkdir -p {your_data_path}/mixture_of_recursions/results
    ```

2.  **Create Symbolic Links:**
    Establish symbolic links from your data path to your project's current path. Replace `{your_data_path}` and `{your_project_path}` accordingly.

    ```bash
    ln -s {your_data_path}/mixture_of_recursions/* {your_project_path}/mixture_of_recursions/
    ```

3.  **Download Pretraining Corpus:**
    Execute the provided script to download the `fineweb-edu-dedup` dataset.

    ```bash
    bash lm_dataset/download_scripts/download_fineweb-edu-dedup.sh
    ```

4.  **Move Cached Dataset:**
    Move the cached dataset to the `DATA_DIR` as specified in [lm_dataset/load_dataset.py](https://github.com/raymin0223/mixture_of_recursions/blob/884fa5a0b4fb4c886f55bd7f62d8a2750cdb4773/lm_dataset/load_dataset.py#L16-L27). This ensures the dataset is located in the correct directory for loading. You can do steps 3 and 4 together by downloading the dataset using the script at [download_langauge_modeling_datasets.sh](https://github.com/raymin0223/mixture_of_recursions/blob/main/lm_dataset/download_scripts/download_langauge_modeling_datasets.sh).


## 🔎 Implementation Overview

### 👉 Dataset and Data Handling

For details on the dataset used, please refer to the [lm_dataset](https://github.com/raymin0223/mixture_of_recursions/tree/main/lm_dataset) directory. We used our custom language modeling dataset class for this project. There are opportunities for optimization:
1. Improving the loading speed of the data `state_dict` when restarting interrupted training. 
2. Additionally, our current input packing allows for attention across different documents, which could be mitigated by integrating future advancements like [FlexAttention](https://pytorch.org/blog/flexattention/).


### 👉 MoR Architecture

Our models are built upon the Llama architecture, specifically by modifying the [LlamaForCausalLM](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/llama/modeling_llama.py) class.
The Expert-choice and Token-choice versions of our MoR architecture can be found in [expert_choice_router.py](https://github.com/raymin0223/mixture_of_recursions/tree/main/model/mor_model/expert_choice_router.py) and [token_choice_router.py](https://github.com/raymin0223/mixture_of_recursions/tree/main/model/mor_model/token_choice_router.py), respectively. 

The high-level routing process follows these steps:

1.  **Obtain Indices:**
    Get indices from the top-k selections of the routers.

2.  **Index Tokens:**
    Index the current tokens using these obtained indices. For expert-choice, we can avoid variable-length complexities since all samples have the same length. However, for token-choice, we handle variable lengths by adding padding using `rnn_utils.pad_sequence`.

3.  **Compute Shared Blocks:** 
    Perform computations on the shared blocks once.

4.  **Scatter and Combine:**
    `scatter_add` the computed tokens back to their original shape using the previously obtained indices.

5.  **Repeat:**
    Repeat this process for the desired number of recursions.

**Note:**  We believe further optimizations are possible through the use of `FlexAttention` and `grouped_mm`. 
Moreover in case of KV sharing, we computed all sequences against a shared KV cache for simplicity. The outputs for tokens participating in a specific recursion were then extracted by masking.

## 🚆 Training and Evaluation

### 👉 Configuration Generation

To streamline the creation of training and evaluation scripts, we made an automated Python utility: [generate_pretrain_eval_fewshot_configs.py](https://github.com/raymin0223/mixture_of_recursions/tree/main/util/generate_pretrain_eval_fewshot_configs.py). By specifying your custom config name and providing arguments, the script will automatically generate a YAML file for your configuration. This script will generate configurations based on `example.yaml` files located under [conf/pretrain](https://github.com/raymin0223/mixture_of_recursions/tree/main/conf/pretrain) or [conf/eval_fewshot](https://github.com/raymin0223/mixture_of_recursions/tree/main/conf/eval_fewshot). 

To generate a configuration, simply run:

```bash
python util/generate_pretrain_eval_fewshot_configs --name {config_name}
# Example: python util/generate_pretrain_eval_fewshot_configs --name 250720_pretrain_smollm-360m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001
```


### 👉 Training Setup

We typically conducted training using 4 H100 or A100 GPUs. For distributed training, we leveraged either `Accelerate` or `DeepSpeed` ZeRO Stage 2. Exploring FSDP, Tensor Parallelism, and Pipeline Parallelism for MoR models is left for future work.


Here are the training commands for both versions:


```bash
# DeepSpeed
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 25720 pretrain.py --config-name example

# Accelerate
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file acc_configs/default_config.yaml --main_process_port 25720 pretrain.py --config-name example
```


### 👉 Evaluation

Few-shot accuracy was measured using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Here are the evaluation commands:

```bash
# DeepSpeed
HYDRA_FULL_ERROR=1 deepspeed --include localhost:0,1,2,3 --no_local_rank --master_port 23393 eval_fewshot.py --config-name example

# Accelerate
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file acc_configs/default_config.yaml --main_process_port 23393 eval_fewshot.py --config-name example
```

You can also evaluate the validation loss using [evaluate_fineweb_test.py](https://github.com/raymin0223/mixture_of_recursions/tree/main/evaluate_fineweb_test.py). We specifically measure validation loss for scaling laws analysis.


### 👉 Combined Training and Evaluation

To run training and few-shot evaluation concurrently with a single command, execute the following shell script:
```bash
bash scripts/pretrain_eval_fewshot.sh {launcher} {wandb_mode} {gpu_indices} {exp1_config} {exp2_config} ...
# Example: bash scripts/pretrain_eval_fewshot.sh deepspeed online 0,1,2,3 250720_pretrain_smollm-360m_rec3_middle_cycle_random_lr3e-3_mor_expert_linear_alpha_0.1_sigmoid_aux_loss_0.001
```


## ✅ Pretrained Checkpoints

We share pretrained checkpoints for our 360M parameter Vanilla, Recursive, and MoR models in [Google Drive](https://drive.google.com/drive/folders/1pYKJOu2aBGC-jgoWbfP6T_vqEYtUVxa4?usp=drive_link). Move checkpoints under `./checkpoints` folder.

Alternatively, you can use the following commands to download them, but please be aware of a potential bug:

```bash
pip install gdown

mkdir -p checkpoints
gdown --folder 'https://drive.google.com/drive/folders/1pYKJOu2aBGC-jgoWbfP6T_vqEYtUVxa4?usp=sharing' -O checkpoints
```

Additionally, you can find a script to explore the routing behavior of the expert-choice MoR model in the `notebooks/250727_get_mor_routing_decision.ipynb` notebook.


## 🙏 BibTeX

If you find our work useful, please cite it as:

```
@misc{bae2025mixtureofrecursionslearningdynamicrecursive,
    title={Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation}, 
    author={Sangmin Bae and Yujin Kim and Reza Bayat and Sungnyun Kim and Jiyoun Ha and Tal Schuster and Adam Fisch and Hrayr Harutyunyan and Ziwei Ji and Aaron Courville and Se-Young Yun},
    year={2025},
    eprint={2507.10524},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2507.10524}, 
}
```
