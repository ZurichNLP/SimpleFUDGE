#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch

from train import LitModel
from transformers import BartForConditionalGeneration

"""

Example Call:
    python convert_ckpt_to_hf.py --model_path /srv/scratch6/kew/paraphrase/models/test_finetune --checkpoint_name ckpepoch=02_vloss=1.68270.ckpt --test

Adapted from forum post: https://github.com/PyTorchLightning/pytorch-lightning/issues/3096#issuecomment-686877242

"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path')
    ap.add_argument('--checkpoint_name')
    ap.add_argument('--test', action='store_true')
    return ap.parse_args()

if __name__ == "__main__":

    args = set_args()

    checkpoint_path = os.path.join(args.model_path, args.checkpoint_name)
    print(f'loading checkpoint from {checkpoint_path} ...')
    model = LitModel.load_from_checkpoint(checkpoint_path)
    out_dir = os.path.join(args.model_path, 'best_model')
    print(f'saving checkpoint as model to {out_dir} ...')
    model.model.save_pretrained(out_dir)
    print(f'saving tokenizer to {out_dir} ...')
    model.tokenizer.save_pretrained(out_dir)

    print()
    print(f'To use model:')
    print(f'from transformers import BartTokenizer, BartForConditionalGeneration')
    print(f'model = BartForConditionalGeneration.from_pretrained("{out_dir}")')
    print(f'tokenizer = BartTokenizer.from_pretrained("{out_dir}")')
    print()

    if args.test:
        print('sanity check...')
        from transformers import BartTokenizer, BartForConditionalGeneration

        model = BartForConditionalGeneration.from_pretrained(out_dir)
        tokenizer = BartTokenizer.from_pretrained(out_dir)
        model.eval()

        input_texts = [
            "1, 2, 3, 4, and get the hell out of there!",
            "A person's character is reflected in his wathan.",
            "I shall therefore conduct my analysis on the basis of that premiss.",
        ]

        inputs = tokenizer(input_texts, return_tensors='pt', max_length=128, truncation=True, padding=True)
        
        # Generate
        output = model.generate(inputs['input_ids'], num_beams=4, max_length=128, early_stopping=True)
        output_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in output]

        for i, o in zip(input_texts, output_texts):
            print(f'INPUT: {i}\n\t>>> OUTPUT: {o}')