#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert jsonl <field> to txt file

Usage:
    python jsonl_to_txt.py <infile> <outfile> <field>

"""

import sys
import json

infile = sys.argv[1]
outfile = sys.argv[2]
tgt_field = sys.argv[3]

with open(infile, 'r', encoding='utf8') as inf:
    with open(outfile, 'w', encoding='utf8') as outf:
        for line in inf:
            line = json.loads(line)
            tgt = line[tgt_field]
            if isinstance(tgt, list):
                tgt = ' '.join(tgt)
            outf.write(f'{tgt}\n')

print('done.')