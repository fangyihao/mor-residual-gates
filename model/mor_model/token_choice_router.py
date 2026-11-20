from typing import Callable, List, Optional, Tuple, Union

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from typing_extensions import Unpack
from transformers.utils import TransformersKwargs

from model.kv_caches.cache_utils import Cache, RoutedDynamicCache
from model.base_model.modeling_llama import run_decoder_block
from model.mor_model.util import ROUTER_TYPES, MoRLayerOutputWithPast
from util.misc import get_torch_dtype


class MoRLlamaDecoderLayer(nn.Module):
    """The Mixtures of Depth Block that dynamically which tokens to process in a block.
    Wraps around decoder block to allow for token dropping.
    """

    def __init__(self, config, block_list, cfg, bal_warmup_step=0):
        super().__init__()
        self.mor = True
        self.mor_type = "token"
        
        self.config = config
        self.block_list = block_list
        self.cfg = cfg
        self.residual_recursion = cfg.recursive.get("residual") is not None
        self.bal_warmup_step = bal_warmup_step
        
        self.training_step = 0
        self.num_recursion = cfg.recursive.num_recursion
        assert len(block_list) == self.num_recursion, "Number of recursion should be equal to number of blocks"
        
        torch_dtype = get_torch_dtype(cfg)
        
        if not cfg.mor.rand_router:
            self.mor_router = ROUTER_TYPES[cfg.mor.router_type](config, out_dim=self.num_recursion).to(torch_dtype)
        
        if self.cfg.mor.token.balancing == "loss_free":
            self.register_parameter("router_bias", torch.nn.Parameter(torch.zeros(self.num_recursion), requires_grad=False))
                
    def reset_parameters(self):
        for block in self.block_list:
            if isinstance(block, nn.ModuleList):
                for blk in block:
                    blk.reset_parameters()
            else:
                block.reset_parameters()

    def set_activation_checkpointing(self, strategy):
        for block in self.block_list:
            if isinstance(block, nn.ModuleList):
                for blk in block:
                    blk.set_activation_checkpointing(strategy)
        else:
            block.set_activation_checkpointing(strategy)
        
    def select_tokens_and_batch_with_padding(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        top_expert_indices: Optional[torch.Tensor] = None, 
        index: Optional[int] = None, 
        padding_value: float = 0.0,
    ):
        batched_x = []
        selected_batch_indices = []
        selected_seq_indices = []
        
        for i in range(x.shape[0]):  
            indices = torch.where(top_expert_indices[i] >= index)[0]
            
            if indices.numel() == 0:
                continue
            else:
                selected_batch_indices.append(i)
                selected_seq_indices.append(indices)
                
            batched_x.append(x[i, indices])
        
        if len(batched_x) == 0:
            return None, None, None, None, None, None, None
        
        batched_x = rnn_utils.pad_sequence(
            batched_x, 
            batch_first=True, 
            padding_value=padding_value,
        ).to(x.device)
        
        new_bs, new_seq_len, _ = batched_x.shape
        bs, seq_len, _ = x.shape
                
        new_attention_mask = torch.zeros(
            (new_bs, new_seq_len),
            dtype=x.dtype,
            device=x.device,
        )
        for b in range(new_bs):
            indices = selected_seq_indices[b]
            s = indices.numel()
            
            new_attention_mask[b, :s] = 1
            
        if attention_mask is not None: 
            if attention_mask.dim() == 4:
                new_attention_mask = torch.full(
                    (new_bs, 1, new_seq_len, new_seq_len),
                    False if attention_mask.dtype == torch.bool else torch.finfo(attention_mask.dtype).min,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                
                for b in range(new_bs):
                    indices = selected_seq_indices[b]
                    s = indices.numel()
                    
                    batch_idx = selected_batch_indices[b] if attention_mask.size(0) > 1 else 0
                    mask = attention_mask[batch_idx].index_select(1, indices).index_select(2, indices)
                    new_attention_mask[b, :, :s, :s] = mask
            elif attention_mask.dim() == 2:
                for row, (batch_idx, positions) in enumerate(zip(selected_batch_indices, selected_seq_indices)):
                    new_attention_mask[row, :positions.numel()] = attention_mask[batch_idx, positions]
            else: 
                raise NotImplementedError("Attention mask has unexpected dimensions")
        
        elif self.config._attn_implementation == "sdpa":
            # Right padding follows all real keys, so causal SDPA can use its native mask.
            new_attention_mask = None

        new_position_ids = None
        if position_ids is not None:
            new_position_ids = torch.arange(new_seq_len, dtype=torch.long, device=x.device).unsqueeze(0).to(position_ids.device)
        
        new_position_embeddings = None                
        if position_embeddings is not None:
            head_dim = position_embeddings[0].shape[-1]            
            new_position_embeddings = ()
            
            for i, emb in enumerate(position_embeddings):
                new_position_embeddings += (torch.zeros(
                    (new_bs, new_seq_len, head_dim),
                    dtype=emb.dtype,
                    device=emb.device,
                ),)
                for b in range(new_bs):
                    indices = selected_seq_indices[b]
                    s = indices.numel()
                    batch_idx = selected_batch_indices[b] if emb.size(0) > 1 else 0
                    new_position_embeddings[i][b, :s] = emb[batch_idx].index_select(0, indices)
        
        new_cache_position = None                    
        if cache_position is not None:
            new_cache_position = torch.zeros(
                (new_bs, new_seq_len),
                dtype=cache_position.dtype,
                device=cache_position.device,
            )
            for b in range(new_bs):
                indices = selected_seq_indices[b]
                s = indices.numel()
                
                new_cache_position[b, :s] = torch.gather(cache_position, dim=0, index=indices)
        
        return batched_x, new_attention_mask, new_position_ids, new_cache_position, new_position_embeddings, selected_batch_indices, selected_seq_indices

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        prev_selected_tokens: Optional[torch.LongTensor] = None,
        checkpoint_block=None,
        cache_prefix=0,
        **kwargs: Unpack[TransformersKwargs]
    ):
        bs, seq_len, hidden_dim =  x.shape
                
        if self.training:
            self.training_step += 1
                    
        if not self.residual_recursion:
            final_x, updates = x.clone(), x.clone()
        
        if not self.cfg.mor.rand_router:
            # Top-1 token-choice routing
            router_weights = self.mor_router(x / self.cfg.mor.temp) 
            if "router_func" in self.cfg.mor.token and self.cfg.mor.token.router_func == "sigmoid":
                router_probs = _router_probs = F.sigmoid(router_weights) * self.cfg.mor.token.get("alpha", 0.1) 
            else:
                router_probs = _router_probs = F.softmax(router_weights, dim=-1) * self.cfg.mor.token.get("alpha", 1.0)
            
            if self.cfg.mor.token.balancing == "loss_free":
                router_probs = _router_probs + self.router_bias
            
        else:
            router_weights = torch.rand(bs, seq_len, self.num_recursion, device=x.device, dtype=x.dtype)
            router_probs = _router_probs = router_weights * self.cfg.mor.token.get("alpha", 0.1)
        
        if self.training and self.training_step < self.bal_warmup_step:
            top_expert_indices = torch.ones(bs, seq_len, 1, device=x.device, dtype=torch.long) * (self.num_recursion - 1)
            if self.cfg.mor.token.balancing == "loss_free":
                self.router_bias *= 0.0
        else:
            _, top_expert_indices = torch.topk(router_probs, 1, dim=-1, sorted=False)
        weights = torch.gather(_router_probs, dim=-1, index=top_expert_indices) # [bs, seq_len, 1]
        top_expert_indices = top_expert_indices.squeeze(-1)  # [bs, seq_len]
        
        share_kv = self.cfg.get("kv_sharing", {}).get("enable", False)
        for index, block in enumerate(self.block_list): 
                        
            full_stage = share_kv or (index == 0 and isinstance(past_key_value, RoutedDynamicCache))
            if full_stage:
                batched_x = x.clone()
                active = (top_expert_indices >= index).unsqueeze(-1) if self.residual_recursion else None
                new_attention_mask = attention_mask
                new_position_ids = position_ids
                new_cache_position = cache_position
                new_position_embeddings = position_embeddings
                        
            else: 
                batched_x, new_attention_mask, new_position_ids, new_cache_position, new_position_embeddings, \
                    selected_batch_indices, selected_seq_indices = \
                        self.select_tokens_and_batch_with_padding(
                            x,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            top_expert_indices=top_expert_indices, 
                            index=index, 
                        )
                
                if batched_x is None:
                    continue
                if isinstance(past_key_value, RoutedDynamicCache):
                    rows = torch.tensor(selected_batch_indices, device=x.device)
                    positions = rnn_utils.pad_sequence(selected_seq_indices, batch_first=True, padding_value=-1)
                    kwargs["cache_routing"] = (rows, positions, bs, seq_len, cache_prefix, attention_mask)

            for blk in block: 
                outputs = run_decoder_block(
                    blk, batched_x, checkpoint_block,
                    attention_mask=new_attention_mask,
                    position_ids=new_position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=new_cache_position,
                    position_embeddings=new_position_embeddings,
                    **kwargs
                )
                if self.residual_recursion and full_stage:
                    batched_x = torch.where(active, outputs[0], batched_x)
                else:
                    batched_x = outputs[0]
                
            if full_stage:
                batched_x_processed, _, _, _, _, selected_batch_indices, selected_seq_indices = \
                    self.select_tokens_and_batch_with_padding(
                            batched_x,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            position_embeddings=position_embeddings,
                            top_expert_indices=top_expert_indices, 
                            index=index, 
                        )
                if selected_batch_indices is None:
                    continue
            else:
                batched_x_processed = outputs[0]
        
            if self.residual_recursion:
                # Carry the state of exited tokens across all subsequent stages.
                updates = x.clone()
                for row, (batch_idx, positions) in enumerate(zip(selected_batch_indices, selected_seq_indices)):
                    updates[batch_idx, positions] = batched_x_processed[row, :positions.numel()]
                x = updates
                continue

            for i, batch_idx in enumerate(selected_batch_indices):
                
                processed_indices = selected_seq_indices[i] 
                processed_unpad_x = batched_x_processed[i, :processed_indices.numel()] 
                processed_expert_indices = torch.gather(top_expert_indices[batch_idx], dim=0, index=processed_indices)
                
                finished_indices = torch.where(processed_expert_indices == index)[0]
                finished_indices_in_total = processed_indices[finished_indices]
                
                finished_x = torch.gather(processed_unpad_x, dim=0, index=finished_indices.view(-1, 1).expand(-1, hidden_dim)) 
                finished_w = torch.gather(weights[batch_idx], dim=0, index=finished_indices_in_total.view(-1, 1))
                finished_src = finished_x * finished_w if self.cfg.mor.token.get("gating", "weighted") == "weighted" else finished_x
            
                final_x[batch_idx] = torch.scatter_add(
                    final_x[batch_idx],
                    dim=0,
                    index=finished_indices_in_total.view(-1,1).expand(-1,hidden_dim),
                    src=finished_src.to(x.dtype)
                )
                
                if index < self.num_recursion - 1:
                    unfinished_indices = torch.where(processed_expert_indices > index)[0]
                    unfinished_indices_in_total = processed_indices[unfinished_indices]
                    
                    unfinished_src = torch.gather(processed_unpad_x, dim=0, index=unfinished_indices.view(-1, 1).expand(-1, hidden_dim)) 
                    
                    updates[batch_idx] = torch.scatter(
                        x[batch_idx],
                        dim=0,
                        index=unfinished_indices_in_total.view(-1,1).expand(-1,hidden_dim),
                        src=unfinished_src.to(x.dtype)  
                    )
            x = updates
                
        balancing_loss = None
        balancing_ratio = None
        branch_counts = None
        router_z_loss = None
        
        if self.training and not self.cfg.mor.rand_router:
            if self.cfg.mor.token.balancing in {"loss", "loss_free"}:
                branch_counts = torch.bincount(top_expert_indices.reshape(-1), minlength=self.num_recursion)
                balancing_ratio = branch_counts / (bs * seq_len)
            if self.cfg.mor.token.balancing == "loss":
                P_i = torch.sum(router_probs, dim=(0,1)) / (bs * seq_len)
                f_i = self.num_recursion * balancing_ratio
                balancing_loss = sum(P_i * f_i) / (kwargs["num_items_in_batch"] / bs / seq_len)
                
            if "z_loss" in self.cfg.mor and self.cfg.mor.z_loss:   
                router_z_loss = torch.logsumexp(router_weights, dim=-1)
                router_z_loss = torch.square(router_z_loss)
                router_z_loss = router_z_loss.mean() / (kwargs["num_items_in_batch"] / bs / seq_len)           
            
        observer = getattr(self, "_observe_token_choices", None)
        if self.training and observer is not None:
            observer(top_expert_indices, branch_counts)

        return MoRLayerOutputWithPast(
            hidden_state=x if self.residual_recursion else final_x,
            attention_weights=outputs[1:],
            selected_tokens=None,
            sampling_loss=None,
            sampling_acc=None,
            sampling_topk_acc=None,
            uniformity=None,
            dead_token_seq=None,
            balancing_loss=balancing_loss if self.training else None,
            balancing_ratio=balancing_ratio if self.training else None,
            router_z_loss=router_z_loss if self.training else None,
        )
