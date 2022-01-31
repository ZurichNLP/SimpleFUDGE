#!/usr/bin/env python
# coding: utf-8

# Author: Tannon Kew

"""
Example Call:

    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/test.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_test.tsv \
        --complex_level 0 \
        --simple_level 4
    
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/dev.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_dev.tsv \
        --complex_level 0 \
        --simple_level 4
    
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/train.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_train.tsv \
        --complex_level 0 \
        --simple_level 4

    # wiki-manual
    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/wiki-auto/wiki-manual/test.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/wiki_manual_test.tsv \
        --complex_level 0 \
        --simple_level 1 --wiki

    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/wiki-auto/wiki-manual/train.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/wiki_manual_train.tsv \
        --complex_level 0 \
        --simple_level 1 --wiki

    python extract_aligned_sents_wiki_newsela_manual.py \
        --infile /srv/scratch6/kew/ats/data/en/wiki-auto/wiki-manual/dev.tsv \
        --outfile /srv/scratch6/kew/ats/data/en/aligned/wiki_manual_dev.tsv \
        --complex_level 0 \
        --simple_level 1 --wiki
"""

import argparse
from bdb import set_trace
from typing import List
import csv

import pandas as pd

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, help='')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='')
    parser.add_argument('--complex_level', type=int, required=False, default=0, help='')
    parser.add_argument('--simple_level', type=int, required=False, default=4, help='')
    parser.add_argument('--verbose', action='store_true', required=False, help='')
    parser.add_argument('--debug', action='store_true', required=False, help='')
    parser.add_argument('--wiki', action='store_true', required=False, help='')
    return parser.parse_args()

def get_title_from_full_id(id: List):
    """
    extracts title from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    title, id = id[0].split('.')
    return title
    
def get_level_from_full_id(id: List):
    """
    extracts simplification level from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """
    if not isinstance(id, list):
        raise RuntimeError(f'Expected as list of id(s), but got {type(id)}')
    title, id = id[0].split('.')
    return int(id.split('-')[1])
             
def extract_pairs(df, cid: List, tgt_level: int = 4):
    """
    Recursive function to extract aligned sentences from the
    next adjacent simplification level
    """
    if not cid: 
        return None
    
    c_level = get_level_from_full_id(cid)
    if c_level == tgt_level: # found aligned simple units
        sub_df = df[df['sid'].isin(cid)]
        sents = ' '.join(sub_df.ssent.tolist())
        return sents
        
    else: # recursion
        sub_df = df[df['cid'].isin(cid)]
        next_cid = sub_df.sid.tolist()
        return extract_pairs(df, next_cid, tgt_level)

def test():
    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    print(extract_pairs(df, ['chinook-recognition.en-0-0-0']))
    print(extract_pairs(df, ['chinook-recognition.en-0-0-1']))
    print(extract_pairs(df, ['chinook-recognition.en-0-0-2']))

def parse_newsela_data(args):
    # infile = '/srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/test.tsv'
    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent'], quoting=csv.QUOTE_NONE)
    if args.verbose:
        print(f'DF contains {len(df)} items')
        
    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    if args.verbose:
        print(f'Removed `notAligned`. DF contains {len(df)} items')

    root_nodes = [[id] for id in df['cid'].unique() if get_level_from_full_id([id]) == args.complex_level]
    
    if args.verbose:
        print(len(root_nodes))
        print(root_nodes[:10])

    # collect alignments
    alignments = []
    for root_node in root_nodes:
        sub_df = df[(df['cid'].isin(root_node))]
        csents = sub_df.csent.tolist()
        try:
            src = ' '.join(csents)
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue
            
        tgt = extract_pairs(df, root_node, args.simple_level)
        if tgt:
            alignments.append((src, tgt, get_title_from_full_id(root_node)))
         
    # write collected alignments to outfile
    if args.outfile:
        with open(args.outfile, 'w', encoding='utf8') as outf:
            for src, tgt, _ in alignments:
                outf.write(f'{src}\t{tgt}\n')

    print(f'Finished writing {len(alignments)} alignments to {args.outfile}')

def parse_wiki_data(args, max_sim=0.6):
    # infile = '/srv/scratch6/kew/ats/data/en/newsela-auto/newsela-manual/all/test.tsv'
    df = pd.read_csv(args.infile, sep='\t', header=None, names=['label', 'sid', 'cid', 'ssent', 'csent', 'gleu_score'], quoting=csv.QUOTE_NONE)
    if args.verbose:
        print(f'DF contains {len(df)} items')

    df = df[df['label'] != 'notAligned'] # filter all not aligned sentences
    if args.verbose:
        print(f'Removed `notAligned`. DF contains {len(df)} items')

    df = df[df['gleu_score'] < max_sim] # filter all aligned sentence that are too similar
    if args.verbose:
        print(f'Removed items with sim above `{max_sim}`. DF contains {len(df)} items')


    root_nodes = [[id] for id in df['cid'].unique()]

    if args.verbose:
        print(len(root_nodes))
        print(root_nodes[:10])

    # collect alignments
    alignments = []
    for root_node in root_nodes:
        sub_df = df[(df['cid'].isin(root_node))]
        csents = sub_df.csent.tolist()
        ssents = sub_df.ssent.tolist()
        try:
            src = ' '.join(csents).strip()
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue
            
        try:
            tgt = ' '.join(ssents).strip()
        except TypeError:
            print(f'Could not parse sentence {root_node} as string. Check for unclosed quotations!', csents)
            continue

        if src and tgt:
            alignments.append((src, tgt, root_node[0]))
                
    # write collected alignments to outfile
    if args.outfile:
        with open(args.outfile, 'w', encoding='utf8') as outf:
            for src, tgt, _ in alignments:
                outf.write(f'{src}\t{tgt}\n')

    print(f'Finished writing {len(alignments)} alignments to {args.outfile}')


if __name__ == '__main__':

    args = set_args()
    
    if args.debug:
        test()
    else:
        if args.wiki:
            parse_wiki_data(args)
        else:
            parse_newsela_data(args)