#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    python extract_newsela_sents.py \
        --indir /srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/ \
        --outdir /srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/article_sents \
        --lang en
    
"""

import argparse
from pathlib import Path
import pandas as pd
import nltk
from tqdm import tqdm

def read_article_sentences(filepath):
    sents = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                for sent in nltk.sent_tokenize(line):
                    sents.append(sent)
    return sents

if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--indir', required=True, type=Path, default='/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29', help='path to newsela corpus')
    ap.add_argument('--outdir', required=True, type=Path, default='/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/article_sents', help='path to newsela corpus')
    ap.add_argument('--lang', type=str, default='en')
    args = ap.parse_args()
    
    meta_data = args.indir / f'articles_metadata_{args.lang}_splits.csv'
    
    df = pd.read_csv(meta_data, header=0)

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

                    article_sentences = read_article_sentences(filepath)
                    for sent in article_sentences:
                        outf.write(sent + '\n')