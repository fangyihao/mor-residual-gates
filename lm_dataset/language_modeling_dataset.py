from typing import Iterator, Dict, Any

from collections import deque
from copy import deepcopy
from itertools import islice
import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer

from lm_dataset.util import load_python_edu_text


class LanguageModelingDataset(IterableDataset):
    def __init__(self, dataset, tokenizer: PreTrainedTokenizer, max_length, continuous=True, 
                 buffer_size=2 ** 22, seed=42, transforms: list = None, global_shuffling=True, 
                 local_shuffling=True, add_bos_token=False,):
        # Note that types of dataset and dataset_text_field are list to handle combined pretraining corpus.
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.continuous = continuous
        self.buffer_size = buffer_size
        self.seed = seed
        self.transforms = transforms
        self.global_shuffling = global_shuffling
        self.local_shuffling = local_shuffling
        self.track_position = False
        self.consumed = 0
        self._chunks = deque()

        if self.tokenizer.padding_side == "left":
            raise ValueError("The tokenizer padding side must be right.")

        if self.tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must have an eos token")
        
        self.add_bos_token = add_bos_token
        if self.add_bos_token:
            self.max_length -= 1
            
        self._loaded_state = None

    def __len__(self):
        # Not the actual number of packed samples
        return len(self.dataset)

    def commit_position(self, consumed):
        """Discard chunks strictly before the next example needed by the trainer."""
        self.consumed = consumed
        while len(self._chunks) > 1 and self._chunks[1]["start"] <= consumed:
            self._chunks.popleft()

    def state_dict(self, consumed=None) -> Dict[str, Any]:
        """Snapshot the next unconsumed packed sample, independent of loader prefetch."""
        if consumed is None:
            consumed = self.total_yield_index
        self.commit_position(consumed)
        chunk = self._chunks[0]
        offset = consumed - chunk["start"]
        if not 0 <= offset <= len(chunk["samples"]):
            raise ValueError(f"Packed position {consumed} is outside the retained chunk")
        return {**chunk, "offset": offset}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._loaded_state = state_dict

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        saved, self._loaded_state = self._loaded_state, None
        self._chunks.clear()
        epoch = saved["epoch"] if saved else 0
        document = saved["document"] if saved else 0
        carry = saved["carry"] if saved else []
        local_rng = np.random.RandomState(self.seed)
        if saved and self.local_shuffling:
            algorithm, keys, pos, has_gauss, cached_gaussian = saved["shuffle_state"]
            local_rng.set_state((algorithm, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached_gaussian))

        def source_for_epoch():
            return self.dataset.shuffle(seed=self.seed + epoch) if self.global_shuffling else self.dataset

        source = source_for_epoch()
        if saved and saved["source_state"] is not None:
            # HF remembers a loaded state on subsequent iterations. Only the current epoch resumes.
            source = deepcopy(source)
            source.load_state_dict(saved["source_state"])
            iterator = iter(source)
        else:
            iterator = iter(source)
            if document:
                # HF's shuffle buffer is not checkpointed. Recreate its exact order from the epoch seed.
                deque(islice(iterator, document), maxlen=0)
        start = saved["start"] if saved else 0
        finished = False
        while not finished:
            if saved:
                chunk = saved
                offset = chunk["offset"]
                saved = None
            else:
                texts = []
                size = 0
                while size < self.buffer_size:
                    try:
                        sample = next(iterator)
                    except StopIteration:
                        if not self.continuous or document == 0:
                            finished = True
                            break
                        epoch += 1
                        document = 0
                        source = source_for_epoch()
                        iterator = iter(source)
                        continue
                    document += 1
                    if "blob_id" in sample:
                        sample = load_python_edu_text(sample)
                        if not sample["download_success"]:
                            continue
                    text = sample["text"] + (self.tokenizer.bos_token if self.add_bos_token else self.tokenizer.eos_token)
                    texts.append(text)
                    size += len(text)
                if not texts:
                    return
                tokens = carry.copy()
                for ids in self.tokenizer(texts, add_special_tokens=False)["input_ids"]:
                    tokens.extend(ids)
                length = self.max_length - int(self.add_bos_token)
                end = len(tokens) // length * length
                carry = tokens[end:]
                if not end:
                    continue
                samples = torch.tensor(tokens[:end], dtype=torch.long).reshape(-1, length)
                if self.add_bos_token:
                    samples = torch.cat((samples.new_full((len(samples), 1), self.tokenizer.bos_token_id), samples), dim=1)
                shuffle_state = None
                if self.local_shuffling:
                    samples = samples[local_rng.permutation(len(samples))]
                    algorithm, keys, pos, has_gauss, cached_gaussian = local_rng.get_state()
                    shuffle_state = (algorithm, keys.tolist(), pos, has_gauss, cached_gaussian)
                # Shuffled HF streams omit their shuffle buffer from state_dict().
                source_state = (source.state_dict() if self.track_position and not self.global_shuffling
                                and hasattr(source, "state_dict") else None)
                chunk = dict(start=start, samples=samples, carry=carry, epoch=epoch, document=document,
                             source_state=source_state, shuffle_state=shuffle_state)
                offset = 0

            if not self.track_position:
                self._chunks.clear()
            self._chunks.append(chunk)
            if self.track_position:
                self.commit_position(self.consumed)
            for i in range(offset, len(chunk["samples"])):
                input_ids = chunk["samples"][i]
                self.total_yield_index = start + i + 1
                sample = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
                for transform in self.transforms or ():
                    sample = transform(sample)
                yield sample
            start += len(chunk["samples"])
