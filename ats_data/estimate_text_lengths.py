#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper script to compute the average number of tokens per line in
tsv files containing simplified text data.

Example Call:

    python estimate_text_lengths.py \
        /srv/scratch1/kew/ats/data/en/aligned/newsela_v0_V1.tsv \
        /srv/scratch1/kew/ats/data/en/aligned/newsela_v0_V2.tsv \
        /srv/scratch1/kew/ats/data/en/aligned/newsela_v0_V3.tsv \
        /srv/scratch1/kew/ats/data/en/aligned/newsela_v0_V4.tsv \
        /srv/scratch1/kew/ats/data/en/aligned/turk_test.tsv \
        /srv/scratch1/kew/ats/data/en/aligned/turk_validation.tsv
"""


import sys
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer


np.set_printoptions(threshold=np.inf)

tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

infiles = sys.argv[1:]

for infile in infiles:
    total_lines = 0
    token_counts = None
    max_length = 0
    with open(infile, 'r', encoding='utf8') as inf:
        for line in tqdm(inf):
            entries = line.strip().split('\t')
            if total_lines == 0:
                token_counts = np.zeros(len(entries))
            for i, entry in enumerate(entries):              
                token_len = len(tokenizer.encode(entry))
                if token_len > max_length:
                    max_length = token_len
                token_counts[i] += token_len
            
            total_lines += 1

        print()
        print(infile, token_counts/total_lines, 'avg tokens per line')
        print('max length found:', max_length)
                

