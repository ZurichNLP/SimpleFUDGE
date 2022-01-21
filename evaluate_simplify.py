#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import time
import pickle
import math
import argparse
from collections import namedtuple
from itertools import islice

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from predict_simplify_lp import predict_simplicity, generation_arg_parser

def quick_lc(infile):
    lc = 0
    with open(infile, 'rb') as inf:
        for line in inf:
            lc += 1
    return lc

def preprocess_lines(line):
    # could add further preprocessing here...
    return line.strip()

def main(args):

    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    
    # load generator
    tokenizer = BartTokenizer.from_pretrained(SIMPLIFY_MODEL_STRING)
    generator_model = BartForConditionalGeneration.from_pretrained(SIMPLIFY_MODEL_STRING, return_dict=True).to(args.device)
    generator_model.eval()

    # load fudge conditioning model
    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, tokenizer.pad_token_id, tokenizer.vocab_size)
    conditioning_model.load_state_dict(checkpoint['state_dict'])
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()

    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    # create output dir if not exists already 
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    
    with tqdm(total=quick_lc(args.infile)) as pbar:
        with open(args.outfile, 'w', encoding='utf8') as outf:
            with open(args.infile, 'r', encoding='utf8') as inf:
                while True:
                    batch_lines = list(islice(inf, args.batch_size))
                    if not batch_lines:
                        break
                    batch_lines = list(map(preprocess_lines, batch_lines))
                    batch_results = predict_simplicity(generator_model, tokenizer, conditioning_model, batch_lines, dataset_info, args)

                    assert args.num_return_sequences == 1

                    for text in batch_results:
                        outf.write(f'{text}\n')
                    
                    pbar.update(args.batch_size)
                   
if __name__=='__main__':

    parser = generation_arg_parser(description="SimpleFUDGE")
    
    # add evaluation specific arguments
    parser.add_argument('--infile', type=str, default=None, required=True, help='file containing text to run pred on')
    parser.add_argument('--outfile', type=str, default=None, required=True, help='file to write generated outputs to')
    parser.add_argument('--batch_size', type=int, default=4, required=False, help='number of lines to process as a batch for prediction')


    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
