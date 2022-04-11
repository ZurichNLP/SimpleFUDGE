import os
import random
import time
import pickle
import math
from pathlib import Path
import pprint
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSeq2SeqLM, pipeline, set_seed, GPT2Tokenizer, GPT2Model, MarianTokenizer, MarianMTModel
from transformers import BartForConditionalGeneration, BartTokenizer

from data import Dataset
from model import Model
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *

from transformers import (
    BeamSearchScorer,
    LogitsProcessorList,
    StoppingCriteriaList,
    MaxLengthCriteria,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    MinLengthLogitsProcessor,
    RepetitionPenaltyLogitsProcessor
)

from fudge import FUDGELogits
from typical import TypicalLogitsWarper

def generation_arg_parser(description=None):
    parser = ArgumentParser(description)

    # DATA
    parser.add_argument('--condition_model', type=str, required=False, default=None)
    # parser.add_argument('--dataset_info', type=str, required=False, help='saved dataset info')
    parser.add_argument('--generation_model', type=str, required=True, help='path to finetuned model or huggingface identifier')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False, help='helper argument for testing and debugging runs')
    parser.add_argument('--verbose', action='store_true', default=False, help='print verbose output')
    # generation args
    parser.add_argument('--input_text', type=str, default=None, required=False, help='text to run pred on')
    
    #####################
    # beam search params: 
    # inherited from https://github.com/huggingface/transformers/blob/master/src/transformers/generation_beam_search.py#L118
    parser.add_argument('--num_beams', type=int, default=4, help='Number of beams for beam search.')
    parser.add_argument('--num_return_sequences', type=int, default=1, help='The number of beam hypotheses that shall be returned when finalising beams.')
    parser.add_argument('--do_early_stopping', type=bool, default=False, help='Whether to stop the beam search when at least `num_beams` sentences are finished per batch or not.')
    parser.add_argument('--length_penalty', type=float, default=1.0, help='Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.')
    parser.add_argument('--num_beam_groups', type=int, default=1, help='Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. See [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details. NOTE: not working with FUDGE')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature used to modify logits for generation.')
    parser.add_argument('--min_length', type=int, default=10, help='minimum length of target sequence, used to instantiate a MinLengthLogitProcessor')
    
    ############################
    # stochastic decoing params: 
    parser.add_argument('--do_sample', type=bool, default=False, help='sample instead of greedy')
    parser.add_argument('--top_k', type=int, default=0, help='')
    parser.add_argument('--top_p', type=float, default=1.0, help='')
    parser.add_argument('--typical_p', type=float, default=None, help='')
    parser.add_argument('--max_length', type=int, default=128, help='max generated sequence length')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='')
    parser.add_argument('--no_repeat_ngram_size', type=int, default=1, help='')
    parser.add_argument('--bad_words', nargs='*', default=None, help='')

    # FUDGE-specific args
    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model. Note: if set to 0, FUDGE is not applied!')
    parser.add_argument('--vectorized', type=bool, default=True, help='whether or not to use the vectorized implementation of FUDGE logits_processor')
    parser.add_argument('--soft', type=bool, default=False, help="type of fudge: if True, all logits not in FUDGE's topk preselection are set to -inf and cannot be generated. Default: False, i.e. these logits are left untouched and could potential still be sampled.")
    parser.add_argument('--analysis_file', type=str, default=None, help="File path, if given logits and pre-/post-fudge logits are written to file for analysis")
    
    return parser

def predict_simplicity(model, tokenizer, conditioning_model, input_text, args):

    with torch.no_grad():

        batch_size = len(input_text) # infer batch size
        # print(batch_size)
        encoder_inputs = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding=True).to(args.device)

        model_kwargs = {
            "encoder_outputs": model.get_encoder()(encoder_inputs['input_ids'].repeat_interleave(args.num_beams, dim=0), return_dict=True)
        }
        
        # prepare decoder input ids
        decoder_input_ids = torch.ones((args.num_beams*batch_size, 1), device=model.device, dtype=torch.long)
        # for BART: https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig
        decoder_input_ids = decoder_input_ids * model.config.decoder_start_token_id

        logits_processor = LogitsProcessorList()
            
        if args.condition_lambda > 0.0:
            # instantiate FUDGE logits processor
            fudge_proc = FUDGELogits(
                tokenizer=tokenizer, 
                conditioning_model=conditioning_model, 
                condition_lambda=args.condition_lambda, 
                precondition_topk=args.precondition_topk, 
                batch_size=batch_size,
                soft=args.soft,
                vectorized=args.vectorized,
                analysis_file=args.analysis_file,
                )
            logits_processor.insert(0, fudge_proc)
        
        if args.repetition_penalty != 1.0:
            logits_processor.insert(0, RepetitionPenaltyLogitsProcessor(args.repetition_penalty))

        if args.min_length and args.num_beams > 1:
            # min length logits processor needs to be before FUDGE
            logits_processor.insert(0, MinLengthLogitsProcessor(args.min_length, eos_token_id=model.config.eos_token_id))

        if args.verbose:
            print('Logits Processor List:', logits_processor)

        stopping_criterion = StoppingCriteriaList([MaxLengthCriteria(max_length=args.max_length)])
        
        if args.do_sample: 
            # instantiate logits warpers for multinomial sampling techniques
            # default to temperature==1.0, i.e. no effect        
            logits_warper = LogitsProcessorList([TemperatureLogitsWarper(args.temperature)])

            if args.top_k is not None and args.top_k > 0: # stochastic decoding with beam
                logits_warper.insert(0, TopKLogitsWarper(args.top_k, min_tokens_to_keep=args.num_beams))
            if args.top_p is not None and args.top_p < 1.0:
                logits_warper.insert(0, TopPLogitsWarper(args.top_p, min_tokens_to_keep=args.num_beams))
            if args.typical_p is not None:
                logits_warper.insert(0, TypicalLogitsWarper(args.typical_p, min_tokens_to_keep=args.num_beams))
                # print(logits_warper)

            if args.verbose:
                print('Logits Warper List:', logits_warper)

        if args.num_beams > 1: # beam decoding
            
            # breakpoint()
            # instantiate a BeamSearchScorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=args.num_beams,
                num_beam_hyps_to_keep=args.num_return_sequences,
                length_penalty=args.length_penalty,
                # do_early_stopping=args.do_early_stopping,
                num_beam_groups=args.num_beam_groups,
                device=args.device,
                )
            
            if args.do_sample: # stochastic decoding with beam - uses beam_sample()
                outputs = model.beam_sample(
                    decoder_input_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    stopping_criteria=stopping_criterion,
                    **model_kwargs
                    )

            else: # regular (greedy) beam search with FUDGE - uses beam_search()
                outputs = model.beam_search(
                    decoder_input_ids,
                    beam_scorer,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criterion,
                    **model_kwargs
                    )
        
        else: 
            
            if args.do_sample: # simple sampling - no beam!
                outputs = model.sample(
                    decoder_input_ids, 
                    logits_processor=logits_processor,
                    logits_warper=logits_warper,
                    **model_kwargs
                    )

            else: # regular geedy decoding with FUDGE 
                # NOTE: should be the same as original implementation
                # NOTE: greedy decoding fails with min_length_logits_processor!
                outputs = model.greedy_search(
                    decoder_input_ids, 
                    logits_processor=logits_processor,
                    **model_kwargs
                    )
            
        return tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def main(args):
    
    tokenizer = BartTokenizer.from_pretrained(args.generation_model)
    model = BartForConditionalGeneration.from_pretrained(args.generation_model, return_dict=True).to(args.device)
    model.eval()

    if args.condition_model:
        condition_model_ckpt = Path(args.condition_model) / 'model_best.pth.tar'
        checkpoint = torch.load(condition_model_ckpt, map_location=args.device)
        model_args = checkpoint['args']
        conditioning_model = Model(model_args, tokenizer.pad_token_id, tokenizer.vocab_size) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
        conditioning_model.load_state_dict(checkpoint['state_dict']) # NOTE when loading state_dict for Model, size mismatch for marian_embed.weight: copying a param with shape torch.Size([65002, 300]) from checkpoint, the shape in current model is torch.Size([50266, 300])
        conditioning_model = conditioning_model.to(args.device)
        conditioning_model.eval()
    else:
        conditioning_model = None

    if args.verbose:
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(condition_model_ckpt, checkpoint['epoch']))
        print('num params', num_params(conditioning_model))

    if args.analysis_file is not None:
        Path(args.analysis_file).parent.mkdir(parents=True, exist_ok=True) # ensure directory exists, no harm if it does
        if Path(args.analysis_file).exists() and Path(args.analysis_file).is_file():
            Path(args.analysis_file).unlink() # remove an existing analysis file - necessary as it gets opened in append mode 
        

    if args.debug: # dummy input for testing
        input_text = [
            "Jack be quick.",
            "Jack jumped over the candlestick."
            ]
    elif not args.input_text: # example of batched input for simplification
        input_text = [
            "This is extremely hard to comprehend.",
            "The mouse was eaten by the cat.",
            "Saint Petersburg, formerly known as Petrograd and later Leningrad, is the second-largest city in Russia.",
            "Effective altruism advocates using evidence to determine the most effective ways to benefit others.",
            "Memorial West's class is one of several programs offered through hospitals to help children stay healthy through exercise and proper eating",
            "The idea is to encourage healthy eating and exercise as early as possible to prevent health problems later on."
            ]
    else:
        input_text = [args.input_text]

    results = predict_simplicity(model, tokenizer, conditioning_model, input_text, args)
    
    print('***')
    for in_text, out_text in zip(input_text, results):
        print('Complex:', in_text)
        print('Simple:', out_text)
        print('***')
    # print('INPUT:')
    # pprint.pprint(input_text)
    # print('>>>')
    # pprint.pprint(results)
    # print('***')

if __name__=='__main__':
    
    args = generation_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)