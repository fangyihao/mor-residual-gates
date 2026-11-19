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
| Full-strength ablation | `full` | $1$ |
| Uniform ablation | `uniform` | $\Lambda_R/(K-1)$ |
| Normalized power-law ablation | `power` | $\Lambda_R j^{-\zeta}/\sum_{r=2}^{K} r^{-\zeta}$ |

For example, with $K=3$, $\Lambda_R=1$ and $\zeta=1$, the extra-stage coefficients are $(0.5,0.5)$ for uniform and $(0.6,0.4)$ for power. `lambda_r` sets the sum of update coefficients across the available extra stages. The actual perturbation also depends on the block outputs and which stages a token reaches. Router capacity controls how many tokens execute a stage.

The experiment preset uses `recursive.initialization=pretrained` to retain every source layer and tie its weights across executions. With $L$ source layers and sequence sharing, this gives $L(K-1)$ learned scalars. Zero gates reproduce the source model's one-pass representation at initialization. All four modes write the selected block's returned state back to the token. The original MoR presets use their published router-weighted scatter-add convention. In the four residual modes hard selection trains the router through its auxiliary objective. The preset uses expert choice with `aux_loss`. Token choice is available with `mor.type=token` and its balancing loss.

### Running the four experiments

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

The single-GPU example below uses the same pretrained weights, seed, router, $K=3$, data, and learning rate for all four experiments. To run only the proposal, use `rezero` as the sole mode.

```bash
for mode in rezero full uniform power; do
    accelerate launch --config_file acc_configs/single_gpu_config.yaml \
        pretrain.py --config-name recursive_residual \
        recursive.residual.mode="$mode" || break
done
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
    Install the PyTorch build for your hardware, then install this branch's requirements. The recursive residual implementation targets **Transformers 5.23.0**. Training writes safetensors checkpoints and the evaluation scripts also read previous PyTorch weight files. On resume, Trainer replays the consumed batches in the current epoch to restore the packed sequence order. This replay takes longer for checkpoints deep into a large epoch.

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
