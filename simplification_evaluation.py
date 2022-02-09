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


import argparse

import torch # for bertscore
from easse import sari, bleu, fkgl, bertscore, quality_estimation # samsa fails dep: tupa
 
def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_file', type=str, required=True, help='')
    parser.add_argument('--ref_file', type=str, required=False, help='')
    parser.add_argument('--hyp_file', type=str, required=True, help='')
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
    hyp_sents = read_lines(args.hyp_file)

    assert len(hyp_sents) == len(src_sents)

    results = {'file': args.hyp_file}
    

    # breakpoint()
    # if torch.cuda.is_available():
    #     precision, recall, f1 = bertscore.corpus_bertscore(hyp_sents, refs_sents)
    # else:
    precision, recall, f1 = None, None, None

    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
    results['bertscore_p'] = precision * 100 if precision else None
    results['bertscore_r'] = recall * 100 if recall else None
    results['bertscore_f1'] = f1 * 100 if f1 else None
    qe = quality_estimation.corpus_quality_estimation(src_sents, hyp_sents)
    results.update(qe)
    
    print(results)