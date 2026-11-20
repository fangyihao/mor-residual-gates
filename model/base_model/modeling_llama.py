"""Llama blocks with optional whole-block residual scaling and physical KV sharing."""
from functools import partial
from typing import Optional, Tuple

import torch
from torch import nn
from transformers import LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import (
    LlamaAttention as HFLlamaAttention, LlamaMLP, LlamaRMSNorm,
    apply_rotary_pos_emb, eager_attention_forward,
)
from transformers.utils import TransformersKwargs
from typing_extensions import Unpack

from model.kv_caches.cache_utils import RecursiveDynamicCache, RoutedDynamicCache


def run_decoder_block(block, hidden_states, checkpoint_block, **kwargs):
    if checkpoint_block is None:
        return block(hidden_states, **kwargs)
    # Recompute just the decoder block. Routing and capacity warmup run once.
    return checkpoint_block(partial(block.__call__, **kwargs), hidden_states)


class LlamaAttention(HFLlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        selected = kwargs.pop("selected_tokens", None)
        routing = kwargs.pop("cache_routing", None)
        backend = self.config._attn_implementation
        if routing is not None and isinstance(past_key_value, RoutedDynamicCache):
            # Sparse cached queries need an explicit row-specific key mask.
            backend = "eager" if backend == "eager" else "sdpa"
            key_states, value_states, attention_mask = past_key_value.update_selected(
                key_states, value_states, self.layer_idx, routing, backend
            )
        elif isinstance(past_key_value, RecursiveDynamicCache):
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, selected_tokens=selected
            )
        elif past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            backend, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        block_input = hidden_states
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        observer = getattr(self, "_observe_recursive_update", None)
        raw_output = hidden_states if observer is not None else None
        stage_gate = getattr(self, "stage_gate", None)
        gate = stage_gate["recursive_gate"] if stage_gate is not None else getattr(self, "recursive_gate", None)
        if gate is not None:
            if isinstance(gate, torch.Tensor):
                gate = gate.to(block_input.dtype)
            hidden_states = torch.lerp(block_input, hidden_states, gate)

        if observer is not None:
            observer(block_input, raw_output, hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs
