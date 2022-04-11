#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    python extract_newsela_data_for_fudge.py \
        --indir $DATADIR \
        --outdir $DATADIR/article_paragraphs \
        --meta_data $DATA_DIR/articles_metadata_en_splits.csv \
        --format paragraph 

    python extract_newsela_data_for_fudge.py \
        --indir $DATADIR \
        --outdir $DATADIR/article_sentences \
        --meta_data $DATA_DIR/articles_metadata_en_splits.csv \
        --format sentence 

    python extract_newsela_data_for_fudge.py \
        --indir $DATADIR \
        --outdir $DATADIR/article_para_sents \
        --meta_data $DATA_DIR/articles_metadata_en_splits.csv \
        --format mixed

"""

import argparse
from pathlib import Path
import pandas as pd
import nltk
from tqdm import tqdm
import random

SEED = 42
r = random.Random(SEED)

def read_article(filepath):
    """ keeps paragraph structure """
    para_sents = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                para_sents += [nltk.sent_tokenize(line)]           
    return para_sents

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', required=True, type=Path, default=None, help='path to newsela corpus')
    ap.add_argument('--outdir', required=True, type=Path, default=None, help='path to output directory')
    ap.add_argument('--meta_data', required=True, type=str, default=None, help='csv file containing newsela article names and test/train/validation split information.')
    ap.add_argument('--format', required=True, type=str, choices=['sentence', 'paragraph', 'mixed'], default='sentence', help='whether or not to write one sentence per line or retain some sequences of sentences (experimental)')
    args = ap.parse_args()
        
    df = pd.read_csv(args.meta_data, header=0)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for split in df['split'].unique():
        print(f'processing split: {split}...')
        
        for simp_level in tqdm(df['version'].unique()):        
            # subset df according to split and for each
            # split process all grades
            df_ = df[(df['version'] == simp_level) & (df['split'] == split)]

            outfile = args.outdir / f'{split}_{str(int(simp_level))}.txt'
            
            with open(outfile, 'w', encoding='utf8') as outf:
                for filename in df_['filename']:
                    filepath = args.indir / 'articles' / filename
                    if not filepath.exists():
                        print(f'[!] {filepath} does not exist!')
                        continue

                    article = read_article(filepath)
                    for para in article:
                        if args.format == 'paragraph':
                            outf.write(f'{" ".join(para)}\n')
                        elif args.format == 'sentence':
                            for sent in para:
                                outf.write(f'{sent}\n')
                        else: # mixed - split all long parapraphs to single sents and coin flip to decide for others.
                            if len(para) < 5:
                                if bool(r.randint(0, 1)): # if 1 (TRUE) split sentences
                                    outf.write(f'{" ".join(para)}\n')
                                else:
                                    for sent in para:
                                        outf.write(f'{sent}\n')
                            else: # split all long paragraphs
                                for sent in para:
                                    outf.write(f'{sent}\n')
                        
