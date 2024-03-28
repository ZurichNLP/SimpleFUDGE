#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example Call:

python paraphrase_inference.py \
    --infile resources/data/en/aligned/newsela_manual_sents_version/0-1_dev.tsv \
    --outfile resources/uni_eval/0-1_dev.para \
    --generation_model resources/fudge/generators/bart_large_muss_mined_en \
    --batch_size 1 \
    --do_sample True \
    --top_p 0.9 \
    --temperature 1.0 \
    --num_beams 4 \
    --num_return_sequences 1 \
    --repetition_penalty 1.2

"""

from pathlib import Path
import random
import time
from typing import List
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

def chunker(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def reshape_model_outputs(outputs: List[str], input_batch_size: int) -> List[List[str]]:
        """
        Reshapes a 1D list of output sequences with size [num_return_sequences]
        to a 2D list of output sequences with size [batch_size, num_return_sequences]
        """
        
        num_return_sequences = len(outputs)
        return_seqs_per_input = num_return_sequences//input_batch_size

        # pack outputs into a list of lists, i.e. batch_size x num_return_seqs
        outputs = [outputs[i:i+return_seqs_per_input] for i in range(0, num_return_sequences, return_seqs_per_input)]
        
        if len(outputs) != input_batch_size:
            raise ValueError(f"Got {len(outputs)} outputs from model but expected {input_batch_size}!")
        
        if len(outputs[0]) != return_seqs_per_input:
            raise ValueError(f"Got {len(outputs[0])} return sequences but expected {return_seqs_per_input}!")

        return outputs

def main(args):
    
    # load generator
    tokenizer = BartTokenizer.from_pretrained(args.generation_model)
    generator_model = BartForConditionalGeneration.from_pretrained(args.generation_model, return_dict=True).to(args.device)
    generator_model.eval()
        
    generated_texts = 0
    start_time = time.time()
    # with tqdm(total=quick_lc(args.infile), hide=True, desc='generating') as pbar:
    with open(args.outfile, 'w', encoding='utf8') as outf:
        with open(args.infile, 'r', encoding='utf8') as inf:
            lines = inf.readlines()[:4]
            for batch_lines in chunker(lines, args.batch_size):
            
                batch_lines = list(map(preprocess_lines, batch_lines))

                batch_inputs = tokenizer(batch_lines, return_tensors='pt', padding=True)
                # breakpoint()
                batch_outputs = generator_model.generate(
                    input_ids=batch_inputs['input_ids'].to(args.device),
                    attention_mask=batch_inputs['attention_mask'].to(args.device),
                    num_beams=args.num_beams,
                    early_stopping=args.do_early_stopping,
                    max_length=args.max_length,
                    min_length=args.min_length,
                    do_sample=args.do_sample,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    num_return_sequences=args.num_return_sequences,
                )

                batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
                
                generated_texts += len(batch_outputs)
                # if args.batch_size > 1:
                #     raise RuntimeError('[!] batched implementation is bugged! Use batch_size=1')
                    
                # breakpoint()
                batch_outputs = reshape_model_outputs(batch_outputs, len(batch_lines))
                # breakpoint()
                for line in batch_outputs:
                    if isinstance(line, list):
                        outf.write(f'\t'.join(line) + '\n')
                        print('\t'.join(line))
                    else:
                        outf.write(f'{line}' + '\n')
                        print(line)
                        
                    # pbar.update(args.batch_size)

    elapsed_time = time.time() - start_time
    print(f'generated {generated_texts} texts in {elapsed_time} seconds')
    print(f'outfile: {args.outfile}')


if __name__=='__main__':

    parser = generation_arg_parser(description="SimpleFUDGE")
    
    # add evaluation specific arguments
    parser.add_argument('--infile', type=str, required=True, help='file containing text to run pred on')
    parser.add_argument('--outfile', type=str, required=True, help='file to write generated outputs to')
    parser.add_argument('--batch_size', type=int, default=10, required=False, help='number of lines to process as a batch for prediction')
    
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
