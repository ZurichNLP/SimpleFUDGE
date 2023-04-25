#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Extracts all text units from Newsela corpus and writes them to a single file per split and grade/version level.
Units can be sentences, paragraphs or a mix of both (experimental).

Example call:

    python extract_newsela_texts.py \
        --corpus_dir $DATADIR \
        --unit paragraph 

    python extract_newsela_texts.py \
        --corpus_dir $DATADIR \
        --unit sentence 

    python extract_newsela_texts.py \
        --corpus_dir $DATADIR \
        --unit para_sent

# NOTE:
- meta data file with split information is assumed to be in the same directory as the corpus if not provided
- if no output directory is provided, the output dir is created in the same directory as the corpus
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
    ap.add_argument('--corpus_dir', required=True, type=str, default="resources/data/en/newsela_article_corpus_2016-01-29", help='')
    ap.add_argument('--outdir', required=False, type=str, default=None, help='path to output directory')
    ap.add_argument('--meta_data', required=False, type=str, default=None, help='csv file containing newsela article names and test/train/validation split information.')
    # ap.add_argument('--unit', required=True, type=str, choices=['sentence', 'paragraph', 'mixed'], default='sentence', help='whether or not to write one sentence per line or retain some sequences of sentences (experimental)')
    ap.add_argument('--unit', type=str, default='sentence', choices=['sentence', 'paragraph', 'para_sent', 'document'], required=True, help='')
    ap.add_argument('--grade_level', action='store_true', required=False, help='')

    args = ap.parse_args()
        
    if not args.meta_data: # if no meta data file is provided, use the default one with added split information
        args.meta_data = Path(args.corpus_dir) / 'articles_metadata_en_splits.csv'
    
    if not args.outdir:
        dir_name = f'article_{args.unit}s'
        if args.grade_level:
            dir_name += '_level'
        else:
            dir_name += '_version'
        args.outdir = Path(args.corpus_dir) / dir_name 
        print(f'using default output directory: {args.outdir}')
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.meta_data, header=0)
    
    for split in df['split'].unique():
        print(f'processing split: {split}...')
        
        if args.grade_level:
            levels = df['grade_level'].unique()
        else:
            levels = df['version'].unique()

        for simp_level in tqdm(levels):        
        
            # subset df according to split and for each split process all grades
            if args.grade_level:
                df_ = df[(df['grade_level'] == simp_level) & (df['split'] == split)]
                print(f'processing grade level: {simp_level} ({len(df_)} articles) ...')
            else:
                df_ = df[(df['version'] == simp_level) & (df['split'] == split)]
                print(f'processing article versions: {simp_level} ({len(df_)} articles) ...')

            if len(df_) == 0:
                print(f'[!] no articles for split {split} and level {simp_level}!')
                continue

            outfile = Path(args.outdir) / f'{str(int(simp_level))}_{split}.txt'

            c = 0
            with open(outfile, 'w', encoding='utf8') as outf:
                for filename in df_['filename']:
                    filepath = Path(args.corpus_dir) / 'articles' / filename
                    if not filepath.exists():
                        print(f'[!] {filepath} does not exist!')
                        continue

                    article = read_article(filepath)
                
                    if args.unit == 'document':
                        doc = ' '.join(['</p><p>'.join([' '.join(para) for para in article])])
                        doc = '<p>' + doc + '</p>'
                        outf.write(f'{doc}\n')
                        c += 1
                    
                    elif args.unit == 'paragraph':
                        for para in article:
                            outf.write(f'{" ".join(para)}\n')
                            c += 1

                    elif args.unit == 'sentence':
                        for para in article:
                            for sent in para:
                                outf.write(f'{sent}\n')
                                c += 1

                    elif args.unit == 'para_sent': # mixed - split all long parapraphs to single sents and coin flip to decide for others.
                        for para in article:
                            if len(para) < 5:
                                if bool(r.randint(0, 1)): # if 1 (TRUE) split sentences
                                    outf.write(f'{" ".join(para)}\n')
                                    c += 1
                                else:
                                    for sent in para:
                                        outf.write(f'{sent}\n')
                                        c += 1
                            else: # split all long paragraphs
                                for sent in para:
                                    outf.write(f'{sent}\n')
                                    c += 1
            
            print(f'wrote {c} {args.unit}s to {outfile}')