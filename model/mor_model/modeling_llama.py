from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import LlamaConfig
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from transformers.utils import ModelOutput, TransformersKwargs, logging
from typing_extensions import Unpack

from model.kv_caches.cache_utils import prepare_cache
from model.recursive_model.modeling_llama import LlamaModel, LlamaForCausalLM
from model.recursive_residual import configure_residual_layers

logger = logging.get_logger(__name__)


@dataclass
class MoRBaseModelOutputWithPast(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None


@dataclass
class MoRCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    sampling_loss: Optional[torch.FloatTensor] = None
    sampling_acc: Optional[torch.FloatTensor] = None
    sampling_topk_acc: Optional[torch.FloatTensor] = None
    uniformity: Optional[torch.FloatTensor] = None
    dead_token_seq: Optional[torch.FloatTensor] = None
    balancing_loss: Optional[torch.FloatTensor] = None
    balancing_ratio: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    
    
class MoRLlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, MoRBaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            past_key_values = prepare_cache(self.config, past_key_values, routed=True)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        prev_selected_tokens = None
        sampling_loss = torch.tensor(0.0, device=hidden_states.device)
        sampling_acc_list = []
        sampling_topk_acc_list = []
        uniformity = None # torch.tensor(0.0, device=hidden_states.device)
        dead_token_seq = None # torch.tensor([0.0] * self.config.max_position_embeddings, device=hidden_states.device)
        balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        balancing_ratios = []
        router_z_loss = torch.tensor(0.0, device=hidden_states.device)
        
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:            
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            checkpointing = self.gradient_checkpointing and self.training
            checkpoint_blocks = checkpointing and getattr(decoder_layer, "residual_recursion", False)
            if checkpointing and not checkpoint_blocks:
                # Legacy routing configurations retain their checkpoint path.
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                if hasattr(decoder_layer, "mor") and decoder_layer.mor:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        prev_selected_tokens=None if getattr(decoder_layer, "first_stage", False) else prev_selected_tokens,
                        checkpoint_block=self._gradient_checkpointing_func if checkpoint_blocks else None,
                        cache_prefix=past_seen_tokens,
                        **flash_attn_kwargs,
                    )
                    if decoder_layer.mor_type == "expert":
                        prev_selected_tokens = layer_outputs.selected_tokens
                        if layer_outputs.sampling_loss is not None:
                            sampling_loss += layer_outputs.sampling_loss
                        if layer_outputs.sampling_acc is not None:
                            sampling_acc_list.append(layer_outputs.sampling_acc)
                        if layer_outputs.sampling_topk_acc is not None:
                            sampling_topk_acc_list.append(layer_outputs.sampling_topk_acc)
                        # if layer_outputs.uniformity is not None:
                        #     uniformity += layer_outputs.uniformity
                        # if layer_outputs.dead_token_seq is not None:
                        #     dead_token_seq = layer_outputs.dead_token_seq
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss += layer_outputs.router_z_loss
                            
                    elif decoder_layer.mor_type == "token":
                        if layer_outputs.balancing_loss is not None:
                            balancing_loss += layer_outputs.balancing_loss
                        if layer_outputs.balancing_ratio is not None:
                            balancing_ratios.append(layer_outputs.balancing_ratio)
                        if layer_outputs.router_z_loss is not None:
                            router_z_loss += layer_outputs.router_z_loss
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        **flash_attn_kwargs,
                    )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        groups = getattr(self, "num_routing_groups", 1)
        balancing_ratio = torch.tensor(0.0, device=hidden_states.device)
        if balancing_ratios:
            balancing_ratio = balancing_ratios[0] if groups == 1 else torch.stack(balancing_ratios)

        output = MoRBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            sampling_loss=sampling_loss / groups,
            sampling_acc=sum(sampling_acc_list)/len(sampling_acc_list) if len(sampling_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            sampling_topk_acc=sum(sampling_topk_acc_list)/len(sampling_topk_acc_list) if len(sampling_topk_acc_list) > 0 else torch.tensor(0.0, device=hidden_states.device),
            uniformity=uniformity,
            dead_token_seq=dead_token_seq,
            balancing_loss=balancing_loss / groups,
            balancing_ratio=balancing_ratio,
            router_z_loss=router_z_loss / groups,
        )
        return output if return_dict else output.to_tuple()


class MoRLlamaForCausalLM(LlamaForCausalLM):
    
    backbone_class = MoRLlamaModel

    def __init__(self, config):
        super().__init__(config)
        routing = getattr(config, "mor_routing", None)
        if routing is not None:
            cfg = OmegaConf.create(routing)
            if cfg.mor.type == "expert":
                self.transform_layer_to_mor_expert(cfg)
            else:
                self.transform_layer_to_mor_token(cfg)
            self.post_init()

    def _wrap_recursions(self, cfg, wrap_group):
        configure_residual_layers(self, cfg.recursive)
        layers = self.model.layers
        k = cfg.recursive.num_recursion
        sharing = cfg.recursive.sharing
        middle = sharing.startswith("middle_")
        body = layers[1:-1] if middle else layers
        depth = len(body) // k
        if sharing.endswith("cycle"):
            groups = [[body[i:i + depth] for i in range(0, len(body), depth)]]
        elif sharing.endswith("sequence"):
            # Each physical layer makes an independent depth decision.
            groups = [[body[i + j:i + j + 1] for j in range(k)] for i in range(0, len(body), k)]
        else:
            raise ValueError(f"Unknown sharing layout {sharing!r}")
        wrapped = [module for group in groups for module in wrap_group(group)]
        if middle:
            wrapped = [layers[0], *wrapped, layers[-1]]
        self.model.layers = nn.ModuleList(wrapped)
        self.model.num_routing_groups = len(groups)
        # Save only the architecture needed by from_pretrained, omitting run paths and data settings.
        keys = ["mor", "recursive", "kv_sharing", "precision", "num_warmup_steps", "gradient_accumulation_steps"]
        self.config.mor_routing = OmegaConf.to_container(OmegaConf.masked_copy(cfg, keys), resolve=True)

    def transform_layer_to_mor_expert(self, cfg):
        from model.mor_model.expert_choice_router import MoRLlamaDecoderLayer

        capacity = [float(cap) for cap in cfg.mor.capacity.split(',')]
        if cfg.recursive.get("residual") is not None and capacity[0] != 1.0:
            raise ValueError("Recursive residual modes require a first-stage capacity of 1")
        warmup = cfg.mor.expert.get("cap_warmup_step")
        if warmup is None:
            warmup = cfg.num_warmup_steps * cfg.gradient_accumulation_steps

        def wrap_group(stages):
            for stage, blocks in enumerate(stages):
                wrapper = MoRLlamaDecoderLayer(self.config, blocks, cfg, capacity[stage], warmup)
                wrapper.first_stage = stage == 0
                yield wrapper

        self._wrap_recursions(cfg, wrap_group)

    def transform_layer_to_mor_token(self, cfg):
        from model.mor_model.token_choice_router import MoRLlamaDecoderLayer

        warmup = cfg.mor.token.get("bal_warmup_step", 0) or 0
        if warmup:
            warmup *= cfg.gradient_accumulation_steps

        def wrap_group(stages):
            return [MoRLlamaDecoderLayer(self.config, nn.ModuleList(stages), cfg, warmup)]

        self._wrap_recursions(cfg, wrap_group)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, MoRCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return MoRCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            sampling_loss=outputs.sampling_loss,
            sampling_acc=outputs.sampling_acc,
            sampling_topk_acc=outputs.sampling_topk_acc,
            uniformity=outputs.uniformity,
            dead_token_seq=outputs.dead_token_seq,
            balancing_loss=outputs.balancing_loss,
            balancing_ratio=outputs.balancing_ratio,
            router_z_loss=outputs.router_z_loss,
        )
