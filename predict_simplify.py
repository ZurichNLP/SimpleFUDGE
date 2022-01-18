import os
import random
import time
import pickle
import math
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

# factored out common generation utility methods
from generation_utils import top_k_top_p_filtering, _postprocess_next_token_scores

def main(args):
    with open(args.dataset_info, 'rb') as rf:
        dataset_info = pickle.load(rf)
    
    # modification: BART model as G
    # NOTE special tokens and ids: 
    # {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'sep_token': '</s>', 'pad_token': '<pad>', 'cls_token': '<s>', 'mask_token': '<mask>'}
    #   {0: '<s>', 1: '<pad>', 2: '</s>', 3: '<unk>'}
    # breakpoint()
    tokenizer = BartTokenizer.from_pretrained(SIMPLIFY_MODEL_STRING)
    model = BartForConditionalGeneration.from_pretrained(SIMPLIFY_MODEL_STRING, return_dict=True).to(args.device)
    # breakpoint()
    # TODO sort out special tokens 
    tokenizer.add_special_tokens({'pad_token': PAD_TOKEN})
    pad_id = tokenizer.encode(PAD_TOKEN, add_special_tokens=False)[0]
        
    model.eval()

    checkpoint = torch.load(args.ckpt, map_location=args.device)
    model_args = checkpoint['args']
    conditioning_model = Model(model_args, pad_id, len(dataset_info.index2word)) # no need to get the glove embeddings when reloading since they're saved in model ckpt anyway
    conditioning_model.load_state_dict(checkpoint['state_dict']) # NOTE when loading state_dict for Model, size mismatch for marian_embed.weight: copying a param with shape torch.Size([65002, 300]) from checkpoint, the shape in current model is torch.Size([50266, 300])
    # TODO first need to train discriminator with bart args
    conditioning_model = conditioning_model.to(args.device)
    conditioning_model.eval()
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.ckpt, checkpoint['epoch']))
    print('num params', num_params(conditioning_model))

    # while True:
    results = predict_simplicity(model, 
                    tokenizer, 
                    conditioning_model, 
                    [args.input_text], 
                    dataset_info, 
                    args,
                    # precondition_topk=args.precondition_topk,
                    # do_sample=args.do_sample,
                    # length_cutoff=args.length_cutoff,
                    # condition_lambda=args.condition_lambda,
                    # device=args.device
                    )
    print(results)


def predict_simplicity(model, tokenizer, conditioning_model, input_text, dataset_info, args):
    # breakpoint()
    # precondition_topk=200, do_sample=False, length_cutoff=512, condition_lambda=1.0, device='cuda'
    with torch.no_grad():
        batch_size = len(input_text)

        # assumes initially all same length.
        encoded_input = [tokenizer.encode(it, return_tensors='pt').to(args.device) for it in input_text] # batch x seq
        encoded_input = torch.cat(encoded_input, dim=0)

        
        # breakpoint()
        # https://huggingface.co/docs/transformers/model_doc/bart#transformers.BartConfig
        input_ids = torch.LongTensor([[model.config.decoder_start_token_id]]).to(args.device)
        cur_len = 1
        # generation settings
        max_length = args.length_cutoff
        min_length = 0
        temperature = args.temperature
        do_sample = args.do_sample
        top_k = args.top_k
        top_p = args.top_p
        repetition_penalty = 1.0
        no_repeat_ngram_size = 0
        bad_words_ids = None #[[tokenizer.pad_token_id]] TODO: causes index error in BART's model.postprocess_next_token_scores()
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        effective_batch_size = batch_size
        attention_mask = encoded_input.new_ones(encoded_input.shape)
        use_cache = True
        model_specific_kwargs = {'encoder_outputs': model.get_encoder()(encoded_input, attention_mask=attention_mask)}

        # fudge specific controls
        condition_lambda = args.condition_lambda
        precondition_topk = args.precondition_topk

        output = _generate_no_beam_search(model,
                                        conditioning_model,
                                        condition_lambda,
                                        precondition_topk,
                                        input_ids,
                                        cur_len,
                                        max_length,
                                        min_length,
                                        do_sample,
                                        temperature,
                                        top_k,
                                        top_p,
                                        repetition_penalty,
                                        no_repeat_ngram_size,
                                        bad_words_ids,
                                        pad_token_id,
                                        eos_token_id,
                                        batch_size,
                                        attention_mask,
                                        use_cache,
                                        model_specific_kwargs)

        return [tokenizer.decode(s[1:]) for s in output] # 1: to delete the pad token


# hack of code from transformers/generation_utils.py
# to get our conditioning
def _generate_no_beam_search(
        model,
        conditioning_model,
        condition_lambda,
        precondition_topk,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
        model_kwargs,
    ):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        # breakpoint()
        # length of generated sentences / unfinished sentences
        unfinished_sents = input_ids.new(batch_size).fill_(1)
        sent_lengths = input_ids.new(batch_size).fill_(max_length)

        past = None
        while cur_len < max_length:
            model_inputs = model.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
            )

            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adapted for transfomers v4.16 (dev)
            scores = _postprocess_next_token_scores(
                scores=next_token_logits,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
            )

            # if model has past, then set the past variable to speed up decoding
            if "past_key_values" in outputs:
                past = outputs.past_key_values
            elif "mems" in outputs:
                past = outputs.mems

            # breakpoint()
            top_logits, top_indices = scores.topk(precondition_topk, dim=1) # batch x topk
            tplus1_candidates = torch.cat([input_ids.unsqueeze(1).expand(-1, precondition_topk, -1), top_indices.unsqueeze(2)], dim=2)[:, :, 1:] # batch x topk x seq+1, with pad dropped
            expanded_lengths = torch.LongTensor([[cur_len for _ in range(precondition_topk)] for _ in range(batch_size)]).to(scores.device)
            if condition_lambda == 0:
                condition_logits = torch.zeros_like(top_logits).float()
            else:
                condition_logits = conditioning_model(tplus1_candidates.flatten(0, 1), # batch*topk x seq+1
                                                    expanded_lengths.flatten(0, 1), # batch*topk
                                                    None,
                                                    None,
                                                    None)
                condition_logits = condition_logits.view(batch_size, precondition_topk, -1)[:, :, -1] # batch x topk of last formality pred
                # breakpoint()
                # TODO: check logic applied to logits
                condition_logits = condition_logits - torch.log(1 + torch.exp(condition_logits)) # get correct log probs
                # condition_logits = - torch.log(1 + torch.exp(condition_logits)) # for informal
            full_logits = top_logits + condition_lambda * condition_logits
            
            if temperature != 1.0:
                full_logits = full_logits / temperature
            
            if do_sample:
                # raise NotImplementedError
                next_token_logscores = top_k_top_p_filtering(full_logits, top_k=top_k, top_p=top_p)
                # Sample
                # breakpoint()
                probs = F.softmax(next_token_logscores, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            else:
                # Greedy decoding
                next_token = top_indices[torch.arange(batch_size).to(top_indices.device), torch.argmax(full_logits, dim=-1)]

            # update generations and finished sentences
            if eos_token_id is not None:
                # pad finished sentences if eos_token_id exist
                tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            
            # add token and increase length by one
            input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
                is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
                sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
                # unfinished_sents is set to zero if eos in sentence
                unfinished_sents.mul_((~eos_in_sents).long())

            # stop when there is a </s> in each sentence, or if we exceed the maximul length
            if unfinished_sents.max() == 0:
                break

            # extend attention_mask for new generated input if only decoder
            if model.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return input_ids

if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--dataset_info', type=str, required=True, help='saved dataset info')
    parser.add_argument('--model_string', type=str, default='Helsinki-NLP/opus-mt-es-en')

    parser.add_argument('--input_text', type=str, default=None, required=True, help='text to run pred on')

    parser.add_argument('--precondition_topk', type=int, default=200, help='consider top k outputs from gpt at each step before conditioning and re-pruning')
    parser.add_argument('--do_sample', action='store_true', default=False, help='sample instead of greedy')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature used to modify logits for generation.')
    parser.add_argument('--top_k', type=int, default=0, help='')
    parser.add_argument('--top_p', type=float, default=1.0, help='')
    parser.add_argument('--condition_lambda', type=float, default=1.0, help='lambda weight on conditioning model')
    parser.add_argument('--length_cutoff', type=int, default=512, help='max length')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--debug', action='store_true', default=False)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)