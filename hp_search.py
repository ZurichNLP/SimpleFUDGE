#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Tannon Kew

"""
Runs a small hyperparamter sweep of validation sets.

Considers options for the following params:

- precondition_topk 200
- condition_lambda 10
- soft vs. hard

Example Call:
    
    python hp_search.py \
        --condition_model /srv/scratch6/kew/ats/fudge/discriminators/wiki100M_bart_glove/model_best.pth.tar \
        --generation_model /srv/scratch6/kew/ats/fudge/generators/bart_large_paraNMT_filt_fr \
        --do_sample --top_k=5 \
        --logging_file hp_search_results/topk5_sweep.csv \
        --outpath hp_search_results/topk5_sweep.log \
        --max_lines 50
"""

import sys
from pathlib import Path
import logging
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from types import SimpleNamespace

from transformers import BartTokenizer, BartForConditionalGeneration

from model import Model
from predict_simplify import predict_simplicity, generation_arg_parser
from simplification_evaluation import *
from perplexity import distilGPT2_perplexity_score


logger = logging.getLogger()

###########
# variables
###########
max_lines = 50
condition_lambda_sweep = [0, 1, 2, 3, 5, 8, 10] # [0, 1, 2, 3, 5, 6]
precondition_topk_sweep = [50, 100, 150, 200]
soft_hard_sweep = [True, False]


def chunker(iterable, batch_size=4):
    return (iterable[pos:pos + batch_size] for pos in range(0, len(iterable), batch_size))

if __name__ == '__main__':

    parser = generation_arg_parser(description="SimpleFUDGE")
    
    parser.add_argument('--logging_file', type=str, default=None, required=False, help='file for logging, if not provided, logs are printed to stdout')
    parser.add_argument('--outpath', type=str, default=None, required=True, help='output file for results csv')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='number of lines to process as a batch for prediction')
    parser.add_argument('--max_lines', type=int, default=10, required=False, help='number of lines from validation file to process for generation')

    parser.add_argument('--data_dir', type=str, default='/srv/scratch6/kew/ats/data/en/aligned', required=False, help='directory containing aligned test/validation files')

    parser.add_argument('--datasets', type=str, 
        default=['asset_validation','turk_validation','newsela_manual_v0_v4_dev', 'wiki_manual_dev'], 
        required=False, nargs='*', help='names of test/validation files to run inference on')


    args = parser.parse_args()

    if args.logging_file is not None:
        logging.basicConfig(filename=args.logging_file , format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)
    else:
        logging.basicConfig(format='%(asctime)s | %(levelname)s: %(message)s', level=logging.NOTSET)

    logger.info(f'Loading generation model')
    tokenizer = BartTokenizer.from_pretrained(args.generation_model)
    generator_model = BartForConditionalGeneration.from_pretrained(args.generation_model, return_dict=True).to(args.device)
    generator_model.eval()

    logger.info(f'Loading conditioning model')
    checkpoint = torch.load(args.condition_model, map_location=args.device)
    model_args = checkpoint['args']
    breakpoint()
    conditioning_model = Model(model_args, tokenizer.pad_token_id, tokenizer.vocab_size)
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()

    results = []
    


    vargs = vars(args)
    for dataset in args.datasets:
        infile = Path(args.data_dir) / f'{dataset}.tsv'
        if infile.exists():
            logger.info(f'Running hp sweep on {dataset}...')
            for condition_lambda in condition_lambda_sweep:
                vargs['condition_lambda'] = condition_lambda
                logger.info(f'condition_lambda = {condition_lambda}...')
                for precondition_topk in precondition_topk_sweep:
                    vargs['precondition_topk'] = precondition_topk
                    logger.info(f'precondition_topk = {precondition_topk}...')
                    for val in soft_hard_sweep:
                        vargs['soft'] = val
                        logger.info(f'soft = {val}...')

                        hyp_sents = []
                        sents = read_split_lines(infile, '\t')[:args.max_lines] # threshold lines used for validation
                        src_sents = [i[0] for i in sents]
                        refs_sents = [i[1:] for i in sents]
                        refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose to number samples x len(test set)
                        for batch in tqdm(chunker(src_sents, args.batch_size), total=len(src_sents)//args.batch_size):
                            outputs = predict_simplicity(generator_model, tokenizer, conditioning_model, batch, SimpleNamespace(**vargs))
                            hyp_sents.extend(outputs)

                        lresults = {}
                        lresults.update(vargs)
                        lresults['data'] = dataset
                        lresults['bleu'] = bleu.corpus_bleu(hyp_sents, refs_sents)
                        lresults['sari'] = sari.corpus_sari(src_sents, hyp_sents, refs_sents, legacy=False)
                        lresults['fkgl'] = fkgl.corpus_fkgl(hyp_sents)
                        # NOTE: when computing ppl with gpt-2, we prefix each sentence with a
                        # full-stop so that the entire generated sentence is scored correctly!
                        lresults['ppl'] = np.nanmean(np.array([distilGPT2_perplexity_score('. ' + sent) for sent in hyp_sents]))
                        lresults['wlen'] = sum([len(sent.strip().split()) for sent in hyp_sents]) / len(hyp_sents)
                        lresults['empty'] = sum([1 for sent in hyp_sents if len(sent.strip()) == 0])
                        results.append(lresults)
                        logger.info(f'*********')
                        logger.info(f'{lresults}')
                        logger.info(f'*********')

                        generated_output_file = Path(args.outpath) / f'{dataset}_{condition_lambda}_{precondition_topk}_{val}.txt'
                        with open(generated_output_file, 'r', encoding='utf8') as outf:
                            for sent in hyp_sents:
                                outf.write(f'{sent}\n')


    logger.info(f'*********')
    logger.info(f'{results}')
    logger.info(f'*********')

    results_output_file = Path(args.outpath) / f'results.csv'
    logger.info(f'Writing Dataframe to {results_output_file}...')
    df = pd.DataFrame(results)
    df.to_csv(results_output_file)






