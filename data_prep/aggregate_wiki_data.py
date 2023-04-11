#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew

"""
Reads in all available sentences from collected and
segmented from wikipedia dumps and compiles them into a
single csv. 

Sentences are scred with FKGL score for filtering.

Example Call:

    python aggregate_wiki_data.py \
        /srv/scratch1/kew/ats/data/en/wiki_dumps/enwiki/enwiki_sents.txt \
        /srv/scratch1/kew/ats/data/en/wiki_dumps/simplewiki/simplewiki_sents.txt \
        /srv/scratch1/kew/ats/data/en/wiki_dumps/enwiki_simplewiki.csv

"""

import time
from multiprocessing import  Pool
import random
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

from easse import fkgl

seed = 23
enwiki_sents = sys.argv[1]
siwiki_sents = sys.argv[2] 
outfile = sys.argv[3] 

def fetch_dataframe(file):
    df = pd.read_csv(file, sep='\t', header=None, names=['text'])
    return df

def parallelize_dataframe(df, func, n_cores=10, verbose=False):
    print('Running jobs on {} CPU(s)'.format(n_cores))

    START_TIME = time.time()

    df_splits = np.array_split(df, n_cores)
    with Pool(n_cores) as p:
        result = list(tqdm(p.imap(func, df_splits), total=len(df_splits)))
        p.close()
        p.join()

    df = pd.concat(result)

    END_TIME = time.time()
    t = END_TIME - START_TIME
    if verbose:
        print('Time taken: {:.2f} seconds'.format(t))

    return df

def assign_readability_score(df):
    df['fkgl'] = df['text'].apply(lambda x: fkgl.corpus_fkgl([x]))
    return df

if __name__ == '__main__':
    
    sim = fetch_dataframe(siwiki_sents)
    sim = parallelize_dataframe(sim, assign_readability_score, n_cores=16)
    eng = fetch_dataframe(enwiki_sents).sample(n=len(sim), random_state=seed)
    eng = parallelize_dataframe(eng, assign_readability_score, n_cores=16)
    # randomly subset eng to match sim dataset

    sim = sim[sim['fkgl'] < 9.74] # remove top 25% according to FKGL
    eng = eng[eng['fkgl'] > 6.50] # remove bottom 25% according to FKGL
    sim['source'] = 'simplewiki'
    eng['source'] = 'enwiki'
    df = pd.concat([sim, eng])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.to_csv(outfile, sep='\t', header=True, index=False)