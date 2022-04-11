#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import random

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=Path)
    ap.add_argument('--out_dir', type=Path, default=None)
    ap.add_argument('--src_suffix', type=str, default='complex')
    ap.add_argument('--tgt_suffix', type=str, default='simple')
    ap.add_argument('--splits', type=str, nargs='*', default=['train', 'test', 'valid'])
    ap.add_argument('--levels', type=str, nargs='*', default=[1, 2, 3, 4])
    ap.add_argument('--label_src', action="store_true")
    ap.add_argument('--dataset', type=str, choices=['muss', 'newsela'])
    return ap.parse_args()

def write_to_json(src_lines, tgt_lines, outfile):
    with open(outfile, 'w', encoding='utf8') as outf:
        for s, t in zip(src_lines, tgt_lines):
            outf.write(json.dumps({'complex': s, 'simple': t}, ensure_ascii=False) + '\n')

def read_lines(file):
    with open(file, 'r', encoding='utf8') as inf:
        lines = []
        for line in tqdm(inf):
            lines.append(line.strip())
    return lines

def convert_paraphrase_data_to_jsonl(args):
    for split in args.splits:
        src_lines = read_lines(args.data_dir / f'{split}.{args.src_suffix}')
        tgt_lines = read_lines(args.data_dir / f'{split}.{args.tgt_suffix}')
        if args.out_dir is not None:
            write_to_json(src_lines, tgt_lines, args.out_dir / f'{split}.json')
        else:
            write_to_json(src_lines, tgt_lines, args.data_dir / f'{split}.json')
        print(f'Wrote {len(src_lines)} lines to output file')
    return

def convert_newsela_data_to_jsonl(args):

    for split in args.splits:
        src_lines = []
        tgt_lines = []
        for level in args.levels:
            infile = args.data_dir / f'newsela_manual_v0_v{level}_{split}.tsv'
            with open(infile, 'r', encoding='utf8') as inf:
                for line in inf:
                    src, tgt = line.strip().split('\t')
                    if args.label_src:
                        src_lines.append(f'<l{level}> {src}')
                    else:
                        src_lines.append(src)
                    tgt_lines.append(tgt)

        assert len(src_lines) == len(tgt_lines)

        if split == 'train': # shuffle
            c = list(zip(src_lines, tgt_lines))
            random.Random(4).shuffle(c)
            src_lines, tgt_lines = zip(*c)

        if args.out_dir is not None:
            write_to_json(src_lines, tgt_lines, args.out_dir / f'{split}.json')

    return

if __name__ == "__main__":

    args = set_args()

    if args.dataset == 'newsela':
        convert_newsela_data_to_jsonl(args)
    elif args.dataset == 'muss':
        convert_paraphrase_data_to_jsonl(args)



