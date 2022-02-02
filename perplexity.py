#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapted from https://github.com/dapascual/K2T/blob/64e25a08adce7d2772b5e60a387aff345a2755ca/perplexity.py

import math
import torch
import os
import numpy as np
os.environ['TRANSFORMERS_CACHE']='.'

from transformers import GPT2TokenizerFast, GPT2LMHeadModel
# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('distilgpt2')
model.eval()
# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2TokenizerFast.from_pretrained('distilgpt2')

def distilGPT2_perplexity_score(sentence):
    sentence = sentence.strip()
    if not sentence:
        return np.nan # for empty string
    else:
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor(
            [tokenizer.convert_tokens_to_ids(tokenize_input)])
        loss, logits = model(tensor_input, labels=tensor_input)[:2]
        return math.exp(loss)