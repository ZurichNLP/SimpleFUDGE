#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Call:

    python inference.py \
        --condition_model resources/fudge/ckpt/simplify/simplify_l4_v3/model_best.pth.tar \
        --generation_model resources/fudge/generators/bart_large_paraNMT_filt_fr/ \
        --infile resources/data/en/aligned/turk_test.tsv \
        --batch_size 10 --condition_lambda 0

"""

from pathlib import Path
import random
import time

from tqdm import tqdm
import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from model import Model
from util import num_params
from constants import *
from predict_simplify import predict_simplicity, generation_arg_parser

def quick_lc(infile):
    lc = 0
    with open(infile, 'rb') as inf:
        for line in inf:
            lc += 1
    return lc

def preprocess_lines(line):
    """
    Return only the source sentence from local dataset
    formats. These are expected to be tsv files with the
    source in the first column and the target(s) in the
    susbsequent columns. As input, the generator takes only
    the source.
    """
    # could add further preprocessing here...
    line = line.strip().split('\t')
    return line[0]

def infer_outfile_name_from_args(args):
    """Helper function for inferring outfile name for
    experiment tracking"""
    filename = ''
    filename += f'lambda{args.condition_lambda}'
    filename += f'_pretopk{args.precondition_topk}'
    filename += f'_beams{args.num_beams}'
    filename += f'_estop{args.do_early_stopping}'
    filename += f'_maxl{args.max_length}'
    filename += f'_minl{args.min_length}'
    filename += f'_sample{args.do_sample}'
    filename += f'_lp{args.length_penalty}'
    filename += f'_norep{args.no_repeat_ngram_size}'
    filename += f'_bgrps{args.num_beam_groups}'
    filename += f'_nbest{args.num_return_sequences}'
    filename += f'_repp{args.repetition_penalty}'
    filename += f'_soft{args.soft}'
    filename += f'_temp{args.temperature}'
    filename += f'_topk{args.top_k}'
    filename += f'_topp{args.top_p}'
    filename += f'_bs{args.batch_size}'
    filename += '.txt'

    # expected format: outpath/generationmodel/testset/monsterhparamstring
    if args.generation_model and args.condition_model:
        outfile = Path(args.outpath) / Path(args.generation_model).parts[-1] / Path(args.condition_model).parts[-1] / Path(args.infile).stem / filename
    else:
        outfile = Path(args.outpath) / Path(args.infile).stem / filename

    # create output dir if not exists already 
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    if outfile.is_file():
        print(f'[!] {outfile} exists and will be overwritten...')

    return outfile

def chunker(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(args):
    
    # load generator
    tokenizer = BartTokenizer.from_pretrained(args.generation_model)
    generator_model = BartForConditionalGeneration.from_pretrained(args.generation_model, return_dict=True).to(args.device)
    generator_model.eval()

    # load fudge conditioning model
    if args.condition_model:
        condition_model_ckpt = Path(args.condition_model) / 'model_best.pth.tar'
        checkpoint = torch.load(condition_model_ckpt, map_location=args.device)
        model_args = checkpoint['args']
        conditioning_model = Model(model_args, tokenizer.pad_token_id, tokenizer.vocab_size)
        conditioning_model.load_state_dict(checkpoint['state_dict'])
        conditioning_model = conditioning_model.to(args.device)
        conditioning_model.eval()
    else:
        conditioning_model = None

    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(condition_model_ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    outfile = infer_outfile_name_from_args(args)
    
    
    generated_texts = 0
    start_time = time.time()
    with tqdm(total=quick_lc(args.infile)) as pbar:
        with open(outfile, 'w', encoding='utf8') as outf:
            with open(args.infile, 'r', encoding='utf8') as inf:
                lines = inf.readlines()
                for batch_lines in chunker(lines, args.batch_size):
                
                    batch_lines = list(map(preprocess_lines, batch_lines))
                    batch_results = predict_simplicity(generator_model, tokenizer, conditioning_model, batch_lines, args)

                    generated_texts += len(batch_results)
                    if args.batch_size > 1:
                        raise RuntimeError('[!] batched implementation is bugged! Use batch_size=1')
                        
                    else:
                        texts = '\t'.join(batch_results)
                        outf.write(f'{texts}\n')
                        
                    
                    pbar.update(args.batch_size)

    elapsed_time = time.time() - start_time
    print(f'generated {generated_texts} texts in {elapsed_time} seconds')
    print(f'outfile: {outfile}')


if __name__=='__main__':

    parser = generation_arg_parser(description="SimpleFUDGE")
    
    # add evaluation specific arguments
    parser.add_argument('--infile', type=str, default=None, required=True, help='file containing text to run pred on')
    parser.add_argument('--outpath', type=str, default='resources/fudge/results', required=False, help='file to write generated outputs to')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='number of lines to process as a batch for prediction')
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
