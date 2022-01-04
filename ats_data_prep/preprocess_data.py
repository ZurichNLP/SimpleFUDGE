#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    python preprocess_data.py --data /srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/
    

"""

import argparse
from pathlib import Path
import pandas as pd
import nltk
from tqdm import tqdm

from newsela import assign_newsela_splits

def read_article(filepath):
    sents = []
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line:
                for sent in nltk.sent_tokenize(line):
                    sents.append(sent)
    return sents

if __name__ == '__main__':
    ###################
    # command line args
    ###################
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path, default='/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29', help='path to newsela corpus')
    ap.add_argument('--lang', type=str, default='en')
    ap.add_argument('--force', type=bool, default=False, help='whether or not to attempt to recreate metadata csv with assigned splits.')
    ap.add_argument('--verbose', type=bool, default=False)
    args = ap.parse_args()
    
    meta_data = args.data / f'articles_metadata_{args.lang}_splits.csv'
    
    if not meta_data.exists():
        print('[!] could not find split assigned meta data')
        if args.force:
            print('creating new splits...')
            meta_data = args.data / f'articles_metadata.csv'
            assign_newsela_splits(meta_data)
            meta_data = args.data / f'articles_metadata_{args.lang}_splits.csv'
        else:
            raise RuntimeError(f'Failed to find existing metadata with assigned splits \
                if you want to recreate these, add `--force` as cmdline arg.')

    df = pd.read_csv(meta_data, header=0)

    # breakpoint()
    outpath = args.data / 'article_sents' / args.lang
    outpath.mkdir(parents=True, exist_ok=True)

    for split in df['split'].unique():
        print(f'processing split: {split}...')
        
        for simp_level in tqdm(df['version'].unique()):
            # print(f'processing grade level: {grade}...')
        
            # subset df according to split and for each
            # split process all grades
            df_ = df[(df['version'] == simp_level) & (df['split'] == split)]

            outfile = outpath / f'{split}_{str(int(simp_level))}.txt'
            
            with open(outfile, 'w', encoding='utf8') as outf:
                for filename in df_['filename']:
                    filepath = args.data / 'articles' / filename
                    if not filepath.exists():
                        print(f'[!] {filepath} does not exist!')
                        continue

                    article_sentences = read_article(filepath)
                    for sent in article_sentences:
                        outf.write(sent + '\n')