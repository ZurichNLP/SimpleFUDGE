#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import LogitsWarper

class TypicalLogitsWarper(LogitsWarper):
    """
    from https://github.com/cimeister/typical-sampling/blob/665cae2c11544c63679d31177429aca8dba4a72b/src/transformers/generation_logits_process.py#L242
    """
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 2):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).sum(-1, keepdim=True)

        #shift and sort
        shifted_scores = torch.abs((-ent) - normalized)
        _, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.mass
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        return scores


if __name__ == '__main__':
    pass