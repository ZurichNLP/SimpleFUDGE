#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gathers simplification test data for experiments.

Example call:
    python aggregate_test_data.py --outpath /srv/scratch6/kew/ats/data/en/aligned/
    
"""

import random
from pathlib import Path
import argparse
from datasets import load_dataset
import pandas as pd
from sacremoses import MosesDetokenizer

SEED = 23

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--outpath', type=str, required=False, default='/srv/scratch6/kew/ats/data/en/aligned/', help='path to directory to save validation and test sets')
    ap.add_argument('--datasets', type=str, nargs='+', required=False, default=['turk', 'newsela', 'asset', 'wiki_auto'])
    ap.add_argument('--newsela_path', default='/srv/scratch6/kew/ats/data/en/newsela_article_corpus_2016-01-29/newsela_data_share-20150302/', help='path to newsela_data_share directory containing newsela_articles_*.aligned.sents.txt.')
    ap.add_argument('--wiki_auto_path', default='/srv/scratch6/kew/ats/data/en/wiki-auto', help='path to directory containing contents of https://github.com/chaojiang06/wiki-auto')
    return ap.parse_args()


def save_asset_to_disk(outpath):
    """
    Similar to save_turk_to_disk() but for ASSET.
    
    Downloads the ASSET corpus (crowdsourced simplifications for
    sentences taken from Simple Wikipedia) from huggingface
    datasets library and saves the validation (2000)
    and test (359) items to tsv files.

    Each item in ASSET has 10 manually written,
    crowdsourced simplifications.
    All simplifications are separated by tab char
    """
    asset = load_dataset("asset")

    for split in asset.keys():
        # initialise outfile
        outfile = Path(outpath) / f'asset_{split}.tsv'
        with open(outfile, 'w', encoding='utf8') as f:
            for item in asset[split]:
                src = item['original']
                tgts = '\t'.join(item['simplifications'])
                f.write(f'{src}\t{tgts}\n')
    
        print(f'ASSET {split} saved to disk ({outfile})')

    return

def save_turk_to_disk(outpath):
    """
    Similar to save_asset_to_disk() but for TURK.

    Downloads the TURK corpus (crowdsourced simplifications for
    sentences taken from Simple Wikipedia) from huggingface
    datasets library and saves the validation (2000)
    and test (359) items to tsv files.

    Each item in TURK has 8 manually written,
    crowdsourced simplifications.
    All simplifications are separated by tab char

    """
    turk = load_dataset("turk")
    
    for split in turk.keys():
        # initialise outfile
        outfile = Path(outpath) / f'turk_{split}.tsv'
        with open(outfile, 'w', encoding='utf8') as f:
            for item in turk[split]:
                src = item['original']
                tgts = '\t'.join(item['simplifications'])
                f.write(f'{src}\t{tgts}\n')
    
        print(f'TURK {split} saved to disk ({outfile})')

    return
    
def save_newsela_to_disk(newsela_path, outpath, detok=True, tgt_levels=['V1', 'V2', 'V3', 'V4']):
    """
    aligned sentences from Newsela are expected to be 
    """
    
    md = MosesDetokenizer(lang='en') if detok else None
    
    aligned_sents = Path(newsela_path) / 'newsela_articles_20150302.aligned.sents.txt'

    for tgt_level in tgt_levels:
        c = 0
        outfile = Path(outpath) / f'newsela_v0_{tgt_level}.tsv'
        with open(aligned_sents, 'r', encoding='utf8') as inf:
            with open(outfile, 'w', encoding='utf8') as outf:
                for line in inf:
                    doc_id, src_v, tgt_v, src_text, tgt_text = line.strip().split('\t')
                    if src_v == 'V0' and tgt_v == tgt_level:
                        if md is not None:
                            src_text = md.detokenize(src_text.split())
                            tgt_text = md.detokenize(tgt_text.split())
                        outf.write(f'{src_text.strip()}\t{tgt_text.strip()}\n')
                        c += 1
        print(f'NEWSELA V0-{tgt_level} ({c} items) saved to disk ({outfile})')

def extract_aligned_wiki(wiki_auto_path, outpath, sim_threshold=0.8):
    """
    extract only aligned sentences from wiki manual 
    """

    for split in ['train', 'test', 'dev']:
        wiki_man = Path(wiki_auto_path) / 'wiki-manual' / f'{split}.tsv'
        outfile = Path(outpath) / f'wiki_manual_{split}.tsv'
        c = 0
        with open(wiki_man, 'r', encoding='utf8') as inf:
            with open(outfile, 'w', encoding='utf8') as outf:
                for line in inf:
                    label, id1, id2, simple_sent, complex_sent, gleu_score = line.strip().split('\t')
                    if label == 'aligned' and float(gleu_score) < sim_threshold:
                        outf.write(f'{complex_sent}\t{simple_sent}\n')
                        c += 1

        print(f'WIKI-MANUAL {split} ({c} items) saved to disk ({outfile})')
    


if __name__ == '__main__':
    args = set_args()

    Path(args.outpath).mkdir(parents=True, exist_ok=True)

    if 'turk' in args.datasets:
        save_turk_to_disk(args.outpath)

    if 'asset' in args.datasets:
        save_asset_to_disk(args.outpath)

    if 'newsela' in args.datasets:
        save_newsela_to_disk(args.newsela_path, args.outpath)

    if 'wiki_auto' in args.datasets:
        extract_aligned_wiki(args.wiki_auto_path, args.outpath)