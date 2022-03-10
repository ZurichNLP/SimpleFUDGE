#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    python simple_eval.py \
        --src_file \
        --ref_file \
        --hyp_file 

    python simplification_evaluation.py \
        --src_file /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv \
        --hyp_file /srv/scratch6/kew/ats/fudge/results/bart_large_paraNMT_filt_fr/turk_test/lambda0.0_pretopk200_beams4_estopFalse_maxl128_minl1_sampleFalse_lp1.0_norep1_bgrps1_nbest1_repp1.0_softFalse_temp1.0_topk0_topp1.0.txt

"""


import sys
import argparse
import numpy as np
from tqdm import tqdm
import random
from typing import List, Optional

import pandas as pd

import torch # for bertscore
from easse import sari, bleu, fkgl, bertscore, quality_estimation # samsa fails dep: tupa

from perplexity import distilGPT2_perplexity_score

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')

def rerank_nbest(hyp_sents, ref_sent):

    embs = model.encode(hyp_sents + [ref_sent], convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(embs, embs)    
    index_reranking = torch.argsort(cosine_scores[-1][:-1], descending=True).cpu().tolist()
    hyp_sents[:] = [hyp_sents[i] for i in index_reranking] # rerank according to ranked indices

    return hyp_sents


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True, help='')
    parser.add_argument('--ref_file', type=str, required=False, help='')
    parser.add_argument('--hyp_file', type=str, required=True, help='')
    parser.add_argument('--ppl', action='store_true', help='warning: takes quite some time')
    parser.add_argument('--mode', type=str, required=False, default='model', help='hypothesis selection method to use if input file contains n-best list.')
    return parser.parse_args()

def read_lines(filename):
    """from easse/utils/helpers.py"""
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    return lines

def read_split_lines(filename, split_sep='<::::>'):
    """from easse/utils/helpers.py"""
    with open(filename, encoding="utf-8") as f:
        split_instances = []
        for line in f:
            split_instances.append([split.strip() for split in line.split(split_sep)])

    return split_instances

def ppl_score(sents):
    return np.array([distilGPT2_perplexity_score('. '+s) for s in tqdm(sents)])

def select_hyp(nbest_hyps: List[List[str]], src_sents: Optional[List[str]] = None, mode='model') -> List[str]:
    # breakpoint()
    if mode == 'sim' and src_sents is not None and len(nbest_hyps[0]) > 1:
        hyp_sents = []
        for nbest, src in zip(nbest_hyps, src_sents):
            nbest = rerank_nbest(nbest, src)
            hyp_sents.append(nbest[0])
    elif mode == 'random':
        hyp_sents = [random.choice(nbest) for nbest in nbest_hyps]
    elif mode == 'model':
        hyp_sents = [i[0] for i in nbest_hyps]
    else:
        raise RuntimeError(f'Could not select hypothesis with mode {mode}')
    return hyp_sents

def ppl_diff(hyp_sents, refs_sents, max_refs_sets=2):
    refs_ppls = []
    for ref_sents in refs_sents[:max_refs_sets]:
        refs_ppls.append(ppl_score(ref_sents))

    refs_ppls = np.stack(refs_ppls)

    if refs_ppls.ndim == 1:
        refs_ppls = np.expand_dims(refs_ppls, axis=0)
    # average over all reference simplifications
    refs_ppls = refs_ppls.mean(axis=0)

    hyp_ppls = ppl_score(hyp_sents)

    diff_ppls = refs_ppls - hyp_ppls
    return diff_ppls.mean()

if __name__ == '__main__':

    args = set_args()

    if not args.ref_file: # assumes human refs are in src file after first column
        sents = read_split_lines(args.src_file, split_sep='\t')
        src_sents = [i[0] for i in sents]
        refs_sents = [i[1:] for i in sents]
        refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose to number samples x len(test set)
    else:
        src_sents = read_lines(args.src_file)
        refs_sents = read_lines(args.ref_file)
    
    # for n-best list generations
    
    # hyp_sents = read_lines(args.hyp_file)
    nbest_hyps = read_split_lines(args.hyp_file, split_sep='\t')
    hyp_sents = select_hyp(nbest_hyps, src_sents, mode=args.mode)
    
    assert len(hyp_sents) == len(src_sents)

    results = {'file': args.hyp_file}
    
    results['ppl_diff'] = None
    if args.ppl:
        results['ppl_diff'] = ppl_diff(hyp_sents, refs_sents)

    if torch.cuda.is_available():
        precision, recall, f1 = bertscore.corpus_bertscore(hyp_sents, refs_sents)
    else:
        precision, recall, f1 = None, None, None

    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
    results['bertscore_p'] = precision * 100 if precision else None
    results['bertscore_r'] = recall * 100 if recall else None
    results['bertscore_f1'] = f1 * 100 if f1 else None
    qe = quality_estimation.corpus_quality_estimation(src_sents, hyp_sents)
    results.update(qe)
    
    # print(results)
    # breakpoint()
    df = pd.DataFrame(data=results, index=[0])
    print(df.to_csv(sep=';', index=False))