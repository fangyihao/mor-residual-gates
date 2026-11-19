import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import rotate_half


def llama_compute_query_states(model: nn.Module, rotary_emb: nn.Module, **kwargs) -> torch.Tensor:
    hidden = kwargs["hidden_states"]
    query = model.q_proj(hidden).view(*hidden.shape[:-1], model.num_heads, model.head_dim).transpose(1, 2)
    cos, sin = kwargs["position_embeddings"]
    return query * cos.unsqueeze(1) + rotate_half(query) * sin.unsqueeze(1)


def is_adaption_prompt_trainable(params: str) -> bool:
    """Return True if module is trainable under adaption prompt fine-tuning."""
    return params.split(".")[-1].startswith("adaption_")
