#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example call:

    python simple_eval.py \
        --src_file \
        --ref_file \
        --hyp_file 

    python simple_eval.py \
        --src_file /srv/scratch6/kew/ats/data/en/aligned/turk_test.tsv \
        --hyp_file /srv/scratch6/kew/ats/fudge/results/bart_large_paraNMT_filt_fr/turk_test/lambda0.0_pretopk200_beams4_estopFalse_maxl128_minl1_sampleFalse_lp1.0_norep1_bgrps1_nbest1_repp1.0_softFalse_temp1.0_topk0_topp1.0.txt

"""

import argparse
# import easse
from easse import sari, bleu, fkgl # samsa fails dep: tupa

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

# def get_lines(file, contains_refs=False):
#     """
#     if contains_refs=True, expects a tsv format, with src
#     texts in column 1 and all available references in
#     subsequent columns, e.g. TURK has 8 simple references per
#     complex sentence.
#     """
#     all_texts, more_texts = [], []
#     with open(file, 'r', encoding='utf8') as f:
#         for line in f:
#             if contains_refs:
#                 texts = line.strip().split('\t')
#                 src = texts[0]
#                 ref = texts[1:] # all colums after col 1
#                 all_texts.append(src)   
#                 more_texts.append(ref)
#             else:
#                 all_texts.append(line.strip())

#     return all_texts, more_texts

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
    results['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
    results['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
    results['fkgl'] = fkgl.corpus_fkgl(hyp_sents)

    print(results)