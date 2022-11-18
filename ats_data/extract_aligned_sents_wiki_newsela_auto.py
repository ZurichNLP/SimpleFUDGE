#!/usr/bin/env python
# coding: utf-8

"""

Author: Tannon Kew

Example Call:

    # newsela-auto
    python extract_aligned_sents_wiki_newsela_auto.py --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_auto_v0_v4_train.tsv --complex_level 0 --simple_level 4
    python extract_aligned_sents_wiki_newsela_auto.py --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_auto_v0_v3_train.tsv --complex_level 0 --simple_level 3
    python extract_aligned_sents_wiki_newsela_auto.py --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_auto_v0_v2_train.tsv --complex_level 0 --simple_level 2
    python extract_aligned_sents_wiki_newsela_auto.py --infile /srv/scratch6/kew/ats/data/en/newsela-auto/newsela-auto/all_data/aligned-sentence-pairs-all.tsv --outfile /srv/scratch6/kew/ats/data/en/aligned/newsela_auto_v0_v1_train.tsv --complex_level 0 --simple_level 1


"""

import argparse
from typing import List

from annotate_newsela_splits import (
    newsela_manual_train_article_titles, 
    newsela_manual_dev_article_titles, 
    newsela_manual_test_article_titles,
) 

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', type=str, required=True, help='')
    parser.add_argument('-o', '--outfile', type=str, required=True, help='')
    parser.add_argument('--complex_level', type=int, required=False, default=0, help='')
    parser.add_argument('--simple_level', type=int, required=False, default=4, help='')
    parser.add_argument('--verbose', action='store_true', required=False, help='')
    parser.add_argument('--wiki', action='store_true', required=False, help='')

    return parser.parse_args()

def get_level_from_full_id(id: str):
    """
    extracts simplification level from a sentence identifier used in
    newsela-manual/wiki-manual, e.g. `['chinook-recognition.en-1-0-0']`

    Note: we handle only lists of ids to simplify cases of
    m:n and 1:1 alignments
    """

    article_name, split_id = id.split('.')
    return article_name, int(split_id.split('-')[1])
             
def parse_newsela_auto_data(args):
    """
    Processes alignment file from Newsela-Auto (e.g. `newsela-auto/all_data/aligned-sentence-pairs-all.tsv`)
    """

    c = 0
    with open(args.infile, 'r', encoding='utf8') as f:
        with open(args.outfile, 'w', encoding='utf8') as outf:
            for line in f:
                
                s1_id, s1, s2_id, s2 = line.strip().split('\t')
                s1_article_name, s1_level = get_level_from_full_id(s1_id)
                s2_article_name, s2_level = get_level_from_full_id(s2_id)

                if s1_article_name not in newsela_manual_test_article_titles+newsela_manual_dev_article_titles:
                    if s1_level == args.complex_level and s2_level == args.simple_level:
                        outf.write(f'{s1}\t{s2}\n')
                        c += 1
                    elif s2_level == args.complex_level and s1_level == args.simple_level:
                        outf.write(f'{s2}\t{s1}\n')
                        c += 1

    print(f'Finished writing {c} alignments to {args.outfile}')

if __name__ == '__main__':

    args = set_args()
    

    # if args.wiki:
    #     parse_wiki_data(args)
    # else:
    parse_newsela_auto_data(args)