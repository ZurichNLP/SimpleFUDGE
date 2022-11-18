#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Author: Tannon Kew

Example call:

    compare output sentences by metrics and write to file for inspection

    python manual_inspection.py \
    --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v4_test.tsv \
    --muss_outputs /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v4_test_lr0.47_ls0.79_wr0.43_td0.42.pred \
    --super_outputs /srv/scratch6/kew/ats/supervised/newsela_manual/results/newsela_manual_v0_v4_test/lambda0.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.0_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --fudge_outputs /srv/scratch6/kew/ats/fudge/results/bart_large_muss_mined_en/newsela-lp_l4_article_paragraphs/newsela_manual_v0_v4_test/lambda5.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.2_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --score all \
    --outpath /srv/scratch6/kew/ats/data/comparative_scores_all_l4.jsonl

    python manual_inspection.py \
    --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v3_test.tsv \
    --muss_outputs /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v3_test_lr0.52_ls0.85_wr0.45_td0.62.pred \
    --super_outputs /srv/scratch6/kew/ats/supervised/newsela_manual/results/newsela_manual_v0_v3_test/lambda0.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.0_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --fudge_outputs /srv/scratch6/kew/ats/fudge/results/bart_large_muss_mined_en/newsela-lp_l3_article_paragraphs/newsela_manual_v0_v3_test/lambda4.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.2_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --score all \
    --outpath /srv/scratch6/kew/ats/data/comparative_scores_all_l3.jsonl

    python manual_inspection.py \
    --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v2_test.tsv \
    --muss_outputs /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v2_test_lr0.75_ls0.82_wr0.94_td0.22.pred \
    --super_outputs /srv/scratch6/kew/ats/supervised/newsela_manual/results/newsela_manual_v0_v2_test/lambda0.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.0_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --fudge_outputs /srv/scratch6/kew/ats/fudge/results/bart_large_muss_mined_en/newsela-lp_l2_article_paragraphs/newsela_manual_v0_v2_test/lambda4.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.2_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --score all \
    --outpath /srv/scratch6/kew/ats/data/comparative_scores_all_l2.jsonl

    python manual_inspection.py \
    --src_file /srv/scratch6/kew/ats/data/en/aligned/newsela_manual_v0_v1_test.tsv \
    --muss_outputs /srv/scratch6/kew/ats/muss/outputs/newsela_manual_v0_v1_test_lr0.3_ls0.99_wr0.54_td1.45.pred \
    --super_outputs /srv/scratch6/kew/ats/supervised/newsela_manual/results/newsela_manual_v0_v1_test/lambda0.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.0_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --fudge_outputs /srv/scratch6/kew/ats/fudge/results/bart_large_muss_mined_en/newsela-lp_l1_article_paragraphs/newsela_manual_v0_v1_test/lambda1.0_pretopk200_beams5_estopFalse_maxl128_minl10_sampleFalse_lp1.0_norep1_bgrps1_nbest5_repp1.2_softFalse_temp1.0_topk0_topp1.0_bs1.txt \
    --score all \
    --outpath /srv/scratch6/kew/ats/data/comparative_scores_all_l1.jsonl

"""

import argparse
import random
import sys
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd

from easse import sari, bleu, fkgl, bertscore

rand = random.Random(42)

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_file', type=str, required=True, default=None)
    ap.add_argument('--muss_outputs', type=str, required=False, default=None)
    ap.add_argument('--fudge_outputs', type=str, required=False, default=None)
    ap.add_argument('--super_outputs', type=str, required=False, default=None)
    ap.add_argument('-n', type=int, required=False, default=5)
    ap.add_argument('--seed', type=int, required=False, default=42)
    ap.add_argument('--score', type=str, choices=['sari', 'bleu', 'fkgl', 'all'], required=False, default=None)
    ap.add_argument('--outpath', type=str, required=False, default=None)
    ap.add_argument('--max_items', type=int, default=-1, required=False)
    return ap.parse_args()

def read_split_lines(filename: str, split_sep: str = '\t') -> Tuple[List[str]]:
    """from easse/utils/helpers.py"""
    texts, more_texts = [], []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(split_sep)
            texts.append(line[0])
            if len(line) == 2:
                more_texts.append(line[1])
    return texts, more_texts

def read_parallel_files(args) -> Dict:

    src_texts, tgt_texts = read_split_lines(args.src_file) if args.src_file is not None else None
    muss_texts, _ = read_split_lines(args.muss_outputs) if args.muss_outputs is not None else None
    fudge_texts, _ = read_split_lines(args.fudge_outputs) if args.fudge_outputs is not None else None
    super_texts, _ = read_split_lines(args.super_outputs) if args.super_outputs is not None else None

    if src_texts is not None and muss_texts is not None:
        assert len(src_texts) == len(muss_texts)

    elif src_texts is not None and fudge_texts is not None:
        assert len(src_texts) == len(fudge_texts)

    if src_texts is not None and super_texts is not None:
        assert len(src_texts) == len(super_texts)

    return {
        'src_texts': src_texts,
        'tgt_texts': tgt_texts,
        'muss_texts': muss_texts,
        'fudge_texts': fudge_texts,
        'super_texts': super_texts,
    }


def view_samples(data: Dict) -> None:
    """ inspect a sample of outputs on command line """
    samples = rand.sample(list(range(len(data['src_texts']))), args.n)
    for idx in samples:
        print('-'*5)
        if data['src_texts'] is not None:
            print(f"SRC {idx}\t: {data['src_texts'][idx]}")
        if data['tgt_texts'] is not None:
            print(f"TGT \t: {data['tgt_texts'][idx]}")
        if data['muss_texts'] is not None:
            print(f"MUSS\t: {data['muss_texts'][idx]}")
        if data['fudge_texts']  is not None:
            print(f"FUDGE\t: {data['fudge_texts'][idx]}")
        if data['super_texts']  is not None:
            print(f"SUPER\t: {data['super_texts'][idx]}")
    return


def _score_sents_sari(src_texts, ref_texts, hyp_texts):
    """
    orig_sents: list of original sentences (len = n_samples)
    sys_sents: list of system sentences (len = n_samples)
    refs_sents: list of list of reference sentences (shape = (n_references, n_samples))
    """

    scores = np.zeros(len(src_texts))
    for i in tqdm(range(len(src_texts))):
        scores[i] = sari.corpus_sari(
            [src_texts[i]], 
            [hyp_texts[i]],
            [[ref_texts[i]]], 
            )
    return scores

def _score_sents_bleu(ref_texts, hyp_texts):
    """
    hyp_texts / sys_sents: list of system sentences (len = n_samples)
    ref_texts / refs_sents: list of list of reference sentences (shape = (n_references, n_samples))
    """

    scores = np.zeros(len(hyp_texts))
    for i in tqdm(range(len(hyp_texts))):
        scores[i] = bleu.corpus_bleu(
            [hyp_texts[i]],
            [[ref_texts[i]]], 
            )
    return scores

def _score_sents_fkgl(hyp_texts):

    scores = np.zeros(len(hyp_texts))
    for i in tqdm(range(len(hyp_texts))):
        scores[i] = fkgl.corpus_fkgl([hyp_texts[i]])
    return scores

def _score_sents_bertscore(ref_texts, hyp_texts):
    scores = np.zeros(len(hyp_texts))
    for i in tqdm(range(len(hyp_texts))):
        precision_ref, recall_ref, f1_ref = bertscore.corpus_bertscore(
            [hyp_texts[i]],
            [[ref_texts[i]]],
        )
        scores[i] = f1_ref
    return scores
    
def compute_divergence(data, score):
    

    if data['muss_texts'] is not None:
        if score == 'sari':
            muss_scores = _score_sents_sari(data['src_texts'], data['tgt_texts'], data['muss_texts'])
        elif score == 'bleu':
            muss_scores = _score_sents_bleu(data['tgt_texts'], data['muss_texts'])
        elif score == 'fkgl':
            muss_scores = _score_sents_fkgl(data['muss_texts'])
        elif score == 'bertscore':
            muss_scores = _score_sents_bertscore(data['tgt_texts'], data['muss_texts'])

    

    if data['fudge_texts'] is not None:
        if score == 'sari':
            fudge_scores = _score_sents_sari(data['src_texts'], data['tgt_texts'], data['fudge_texts'])
        elif score == 'bleu':
            fudge_scores = _score_sents_bleu(data['tgt_texts'], data['fudge_texts'])
        elif score == 'fkgl':
            fudge_scores = _score_sents_fkgl(data['fudge_texts'])
        elif score == 'bertscore':
            fudge_scores = _score_sents_bertscore(data['tgt_texts'], data['fudge_texts'])

    if data['super_texts'] is not None:
        if score == 'sari':
            super_scores = _score_sents_sari(data['src_texts'], data['tgt_texts'], data['super_texts'])
        elif score == 'bleu':
            super_scores = _score_sents_bleu(data['tgt_texts'], data['super_texts'])
        elif score == 'fkgl':
            super_scores = _score_sents_fkgl(data['super_texts'])
        elif score == 'bertscore':
            super_scores = _score_sents_bertscore(data['tgt_texts'], data['super_texts'])

    data[f'muss_{score}'] = muss_scores
    data[f'fudge_{score}'] = fudge_scores
    data[f'super_{score}'] = super_scores
    data[f'diff_{score}_muss_fudge'] = muss_scores - fudge_scores
    data[f'diff_{score}_super_fudge'] = super_scores - fudge_scores

    return data

def build_dataframe(data: Dict, outpath: str = None):

    df = pd.DataFrame.from_dict(data)
    if outpath is not None:
        if outpath[-4:] == '.tsv':
            df.to_csv(outpath, sep='\t', header=True, index=False)
        elif outpath[-5:] == 'jsonl':
            df.to_json(outpath, orient='records', force_ascii=False, lines=True)
    return df
    
def pretty_print(df, score, max_items=-1):
    
    if score == 'sari':
        for i, (row_idx, row) in enumerate(df.sort_values('diff_sari_muss_fudge').iterrows()):
            if max_items > 0 and i > max_items:
                break
            print(f"Score diff \t: {row.diff_sari_muss_fudge:.2f}")
            print(f"SRC {row_idx}\t\t: {row.src_texts}")
            print(f"TGT\t\t\t: {row.tgt_texts}")
            print(f"MUSS {row.muss_sari:.2f}\t: {row.muss_texts}")
            print(f"FUDGE {row.fudge_sari:.2f}\t: {row.fudge_texts}")
            print(f"SUPER {row.super_sari:.2f}\t: {row.super_texts}")
            print('---')


    if score == 'bleu':
        for i, (row_idx, row) in enumerate(df.sort_values('diff_bleu_muss_fudge').iterrows()):
            if max_items > 0 and  i > max_items:
                break
            print(f"Score diff \t: {row.diff_bleu_muss_fudge:.2f}")
            print(f"SRC {row_idx}\t\t: {row.src_texts}")
            print(f"TGT\t\t\t: {row.tgt_texts}")
            print(f"MUSS {row.muss_bleu:.2f}\t: {row.muss_texts}")
            print(f"FUDGE {row.fudge_bleu:.2f}\t: {row.fudge_texts}")
            print(f"SUPER {row.super_bleu:.2f}\t: {row.super_texts}")
            print('---')


    if score == 'fkgl':
        for i, (row_idx, row) in enumerate(df.sort_values('diff_fkgl_muss_fudge').iterrows()):
            if max_items > 0 and i > max_items:
                break
            print(f"Score diff \t: {row.diff_fkgl_muss_fudge:.2f}")
            print(f"SRC {row_idx}\t\t: {row.src_texts}")
            print(f"TGT\t\t\t: {row.tgt_texts}")
            print(f"MUSS {row.muss_fkgl:.2f}\t: {row.muss_texts}")
            print(f"FUDGE {row.fudge_fkgl:.2f}\t: {row.fudge_texts}")
            print(f"SUPER {row.super_fkgl:.2f}\t: {row.super_texts}")
            print('---')

    if score == 'bertscore':
        for i, (row_idx, row) in enumerate(df.sort_values('diff_bertscore_muss_fudge').iterrows()):
            if max_items > 0 and i > max_items:
                break
            print(f"Score diff \t: {row.diff_bertscore_muss_fudge:.2f}")
            print(f"SRC {row_idx}\t\t: {row.src_texts}")
            print(f"TGT\t\t\t: {row.tgt_texts}")
            print(f"MUSS {row.muss_bertscore:.2f}\t: {row.muss_texts}")
            print(f"FUDGE {row.fudge_bertscore:.2f}\t: {row.fudge_texts}")
            print(f"SUPER {row.super_bertscore:.2f}\t: {row.super_texts}")
            print('---')

    return

if __name__ == "__main__":

    args = set_args()

    data = read_parallel_files(args)

    if not args.score:
        loop = True
        while loop:
            view_samples(data)
            loop = bool(not input('\nCountinue?\t'))
        sys.exit()
    elif args.score == 'all':
        for score in ['sari', 'bleu', 'fkgl', 'bertscore']:
            data = compute_divergence(data, score)    
    else:
        data = compute_divergence(data, args.score)
        
    df = build_dataframe(data, args.outpath)
    pretty_print(df, ('sari' if args.score in ['sari', 'all'] else args.score), args.max_items)
