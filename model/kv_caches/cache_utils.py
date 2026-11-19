"""Dynamic KV caching with optional sharing across virtual recursions."""
import torch
from transformers.cache_utils import Cache, DynamicCache


class RecursiveDynamicCache(DynamicCache):
    def __init__(self, base_depth, num_recursion, sharing, update_cache=False):
        super().__init__()
        self.base_depth = base_depth
        self.num_recursion = num_recursion
        self.sharing = sharing
        self.update_cache = update_cache

    def _physical_stage(self, layer_idx):
        middle = self.sharing.startswith("middle_")
        if middle and layer_idx == 0:
            return 0, 0
        pos = layer_idx - int(middle)
        if middle and pos == self.base_depth * self.num_recursion:
            return self.base_depth + 1, 0
        if self.sharing.endswith("sequence"):
            physical, stage = divmod(pos, self.num_recursion)
        else:
            stage, physical = divmod(pos, self.base_depth)
        return physical + int(middle), stage

    def get_seq_length(self, layer_idx=0):
        physical, _ = self._physical_stage(layer_idx)
        return super().get_seq_length(physical)

    def get_mask_sizes(self, query_length, layer_idx):
        physical, _ = self._physical_stage(layer_idx)
        return super().get_mask_sizes(query_length, physical)

    def update(self, key_states, value_states, layer_idx, *, selected_tokens=None):
        physical, stage = self._physical_stage(layer_idx)
        if stage == 0:
            return super().update(key_states, value_states, physical)
        keys, values = self.layers[physical].keys, self.layers[physical].values
        if self.update_cache and selected_tokens is not None:
            indices = selected_tokens.unsqueeze(1).expand(-1, key_states.size(1), -1, key_states.size(-1))
            offset = keys.size(2) - key_states.size(2)
            keys = keys.scatter(2, indices + offset, key_states.gather(2, indices))
            values = values.scatter(2, indices + offset, value_states.gather(2, indices))
        return keys, values


class RoutedDynamicCache(DynamicCache):
    """Cache selected KVs at their original batch rows and sequence positions."""
    def __init__(self, num_layers):
        super().__init__()
        self.num_decoder_layers = num_layers

    def update_selected(self, keys, values, layer_idx, routing, backend):
        rows, positions, batch_size, query_length, prefix, full_mask = routing
        # Rows or whole stages can be absent on previous generation steps.
        gap = prefix - self.get_seq_length(layer_idx)
        batch, token = (positions >= 0).nonzero(as_tuple=True)
        dest_rows = rows[batch]
        dest_cols = gap + positions[batch, token]
        shape = (batch_size, gap + query_length, keys.size(1), keys.size(3))
        full_keys = keys.new_zeros(shape)
        full_values = values.new_zeros(shape)
        full_keys[dest_rows, dest_cols] = keys.transpose(1, 2)[batch, token]
        full_values[dest_rows, dest_cols] = values.transpose(1, 2)[batch, token]
        keys, values = self.update(full_keys.transpose(1, 2), full_values.transpose(1, 2), layer_idx)

        # A companion cache layer keeps validity aligned through beam reordering and cropping.
        valid = torch.zeros((batch_size, 1, gap + query_length, 1), dtype=torch.bool, device=keys.device)
        valid[dest_rows, 0, dest_cols, 0] = True
        valid, _ = self.update(valid, valid, self.num_decoder_layers + layer_idx)
        allowed = valid[rows, 0, :, 0][:, None, None, :]
        allowed = allowed & (torch.arange(keys.size(2), device=keys.device) <=
                             (prefix + positions.clamp_min(0))[:, None, :, None])
        if full_mask is not None and full_mask.ndim == 4:
            mask = full_mask.expand(batch_size, -1, -1, -1)[rows]
            indices = positions.clamp_min(0)[:, None, :, None].expand(-1, 1, -1, keys.size(2))
            mask = mask.gather(2, indices)
            allowed = allowed & (mask if mask.dtype == torch.bool else mask == 0)
        elif full_mask is not None:
            allowed = allowed & full_mask[rows, None, None, :].bool()
        if backend == "eager":
            allowed = torch.zeros_like(allowed, dtype=keys.dtype).masked_fill_(~allowed, torch.finfo(keys.dtype).min)
        return keys[rows], values[rows], allowed


def prepare_cache(config, cache, routed=False):
    sharing = getattr(config, "kv_sharing", None)
    if sharing and (cache is None or isinstance(cache, DynamicCache)
                    and not isinstance(cache, RecursiveDynamicCache) and cache.get_seq_length() == 0):
        return RecursiveDynamicCache(sharing["base_depth"], sharing["num_recursion"],
                                     sharing["sharing"], sharing.get("update_cache", False))
    if routed and not sharing and (cache is None or type(cache) is DynamicCache and cache.get_seq_length() == 0):
        return RoutedDynamicCache(config.num_hidden_layers)
    return DynamicCache(config=config) if cache is None else cache
